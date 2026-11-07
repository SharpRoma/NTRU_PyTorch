"""HailoRT 4.23 adapter. No vendor import before opening a hardware session."""

import json
from contextlib import ExitStack
from fractions import Fraction
from importlib.metadata import version
from pathlib import Path

import numpy as np

from .artifacts import ArtifactError


class HailoSession:
    def __init__(self, directory, manifest, scheme):
        self.directory = Path(directory)
        self.manifest, self.scheme = manifest, scheme
        self.stack = None

    def __enter__(self):
        try:
            import hailo_platform as hp
        except ImportError as exc:
            raise ArtifactError(
                "Install the official Windows PyHailoRT 4.23.0 wheel separately"
            ) from exc
        if version("hailort") != "4.23.0":
            raise ArtifactError("This profile requires PyHailoRT 4.23.0")
        stack = ExitStack()
        try:
            hef = hp.HEF(str(self.directory / "model.hef"))
            c = self.manifest["compiled"]
            for direction, infos in (
                ("input", hef.get_input_vstream_infos()),
                ("output", hef.get_output_vstream_infos()),
            ):
                port = c[direction]
                if len(infos) != 1:
                    raise ArtifactError("Only a single input and output are supported")
                info = infos[0]
                if info.name != port["name"] or list(info.shape) != port["shape"]:
                    raise ArtifactError("HEF stream differs from manifest")
                if info.format.type != hp.FormatType.UINT8:
                    raise ArtifactError("HEF stream is not UINT8")
                if Fraction(float(info.quant_info.qp_scale)) != Fraction(
                    str(port["scale"])
                ) or Fraction(float(info.quant_info.qp_zp)) != Fraction(str(port["zero_point"])):
                    raise ArtifactError("HEF quantization is not identity")
            if not hp.Device.scan():
                raise ArtifactError("No Hailo device detected")
            device = stack.enter_context(hp.VDevice())
            config = hp.ConfigureParams.create_from_hef(hef, interface=hp.HailoStreamInterface.PCIe)
            networks = device.configure(hef, config)
            if len(networks) != 1:
                raise ArtifactError("Only one network group is supported")
            network = networks[0]
            inp = hp.InputVStreamParams.make(network, format_type=hp.FormatType.UINT8)
            out = hp.OutputVStreamParams.make(network, format_type=hp.FormatType.UINT8)
            self.pipeline = stack.enter_context(hp.InferVStreams(network, inp, out))
            stack.enter_context(network.activate())
            self.stack = stack
            return self
        except BaseException:
            stack.close()
            raise

    def infer(self, frames):
        if self.stack is None:
            raise RuntimeError("Hardware session is not open")
        self.scheme.validate_frames(frames)
        c = self.manifest["compiled"]
        affine = c.get("arithmetic_profile") == "binary-circulant-affine-v1"
        if affine:
            from .proof import decode_affine, encode_affine

            spec = json.loads(
                (self.directory / "compiler-report.json").read_text(encoding="utf-8")
            )["affine"]
            frames = encode_affine(frames, spec)
        result = self.pipeline.infer({c["input"]["name"]: np.ascontiguousarray(frames)})
        output = result[c["output"]["name"]]
        return decode_affine(output, spec) if affine else output

    def __exit__(self, *exc):
        stack, self.stack = self.stack, None
        if stack is not None:
            return stack.__exit__(*exc)
