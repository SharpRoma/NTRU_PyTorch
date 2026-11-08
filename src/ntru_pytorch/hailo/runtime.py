"""HailoRT 4.23 adapter. No vendor import before opening a hardware session."""

import json
from contextlib import ExitStack
from fractions import Fraction
from importlib.metadata import version
from pathlib import Path
from time import perf_counter_ns

import numpy as np

from .artifacts import ArtifactError


def validate_hef_ports(hef, compiled, hp):
    for direction, infos in (
        ("input", hef.get_input_vstream_infos()),
        ("output", hef.get_output_vstream_infos()),
    ):
        port = compiled[direction]
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
            raise ArtifactError("HEF quantization differs from admitted stream constants")


class HailoSession:
    def __init__(self, directory, manifest, scheme):
        self.directory = Path(directory)
        self.manifest, self.scheme = manifest, scheme
        self.stack = None
        self.cache_affine_lut = True
        self.host_processing = "reference"
        self.last_timings = {}

    def __enter__(self):
        try:
            import hailo_platform as hp
        except ImportError as exc:
            raise ArtifactError(
                "Install the official PyHailoRT 4.23.0 wheel for this OS and Python separately"
            ) from exc
        if version("hailort") != "4.23.0":
            raise ArtifactError("This profile requires PyHailoRT 4.23.0")
        stack = ExitStack()
        try:
            hef = hp.HEF(str(self.directory / "model.hef"))
            c = self.manifest["compiled"]
            validate_hef_ports(hef, c, hp)
            self.affine_spec = None
            self.affine_lut = None
            self.compact_lut = None
            if c.get("arithmetic_profile") == "binary-circulant-affine-v1":
                from .proof import affine_lookup, compact_affine_lookup

                self.affine_spec = json.loads(
                    (self.directory / "compiler-report.json").read_text(encoding="utf-8")
                )["affine"]
                if self.cache_affine_lut:
                    self.affine_lut = affine_lookup(self.affine_spec)
                    if self.host_processing == "optimized":
                        self.compact_lut = compact_affine_lookup(self.affine_lut)
            if not hp.Device.scan():
                raise ArtifactError("No Hailo device detected")
            device = stack.enter_context(hp.VDevice())
            physical = device.get_physical_devices()
            if len(physical) != 1:
                raise ArtifactError("Exact validation requires one physical Hailo device")
            stack.callback(physical[0].release)
            board = physical[0].control.identify()
            fw = board.firmware_version
            firmware = (fw.major, fw.minor, fw.revision)
            if str(board.device_architecture) != "HAILO8L" or firmware != (4, 23, 0):
                raise ArtifactError(
                    "Expected HAILO8L / firmware 4.23.0; no firmware update was attempted"
                )
            self.identity = {
                "architecture": str(board.device_architecture),
                "firmware_version": ".".join(map(str, firmware)),
                "board_name": board.board_name.rstrip("\0"),
                "device_id": str(physical[0].device_id),
                "runtime_version": version("hailort"),
            }
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
        validation_started = perf_counter_ns()
        self.scheme.validate_frames(frames)
        c = self.manifest["compiled"]
        start = perf_counter_ns()
        affine = self.affine_spec is not None
        if affine:
            from .proof import (
                affine_lookup,
                compact_affine_lookup,
                decode_affine,
                decode_affine_compact,
                decode_affine_lookup,
                encode_affine,
            )

            frames = encode_affine(frames, self.affine_spec)
        frames = np.ascontiguousarray(frames)
        encoded = perf_counter_ns()
        result = self.pipeline.infer({c["input"]["name"]: frames})
        called = perf_counter_ns()
        output = result[c["output"]["name"]]
        if affine:
            if self.host_processing == "optimized":
                table = self.compact_lut if self.cache_affine_lut else compact_affine_lookup(
                    affine_lookup(self.affine_spec))
                output = decode_affine_compact(output, table)
            else:
                output = (
                    decode_affine_lookup(output, self.affine_lut) if self.cache_affine_lut
                    else decode_affine(output, self.affine_spec)
                )
        ended = perf_counter_ns()
        self.last_timings = {
            "frame_validation_ns": start - validation_started,
            "affine_encode_ns": encoded - start,
            "sdk_call_with_io_ns": called - encoded,
            "affine_decode_ns": ended - called,
        }
        return output

    def __exit__(self, *exc):
        stack, self.stack = self.stack, None
        if stack is not None:
            return stack.__exit__(*exc)
