"""Portable export and fail-closed artifact admission."""

import hashlib
import json
import platform
from dataclasses import asdict
from importlib.metadata import version
from pathlib import Path
from time import perf_counter_ns

import numpy as np

from ..formats import load, save
from ..models import PublicKey
from .scheme import PROFILE, BinaryScheme, key_hash


class ArtifactError(ValueError):
    pass


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def export_onnx(scheme, path):
    try:
        import onnx
        from onnx import TensorProto, helper, numpy_helper
    except ImportError as exc:
        raise ArtifactError("Install ntru-pytorch[hailo-export] to prepare ONNX") from exc
    weights = scheme.weights.astype(np.float32)[:, :, None, None]
    graph = helper.make_graph(
        [helper.make_node("Conv", ["r", "weights"], ["partial"], kernel_shape=[1, 1])],
        "ntru_binary_partial_sums",
        [helper.make_tensor_value_info("r", TensorProto.FLOAT, [1, scheme.cin, 1, 1])],
        [helper.make_tensor_value_info("partial", TensorProto.FLOAT, [1, scheme.cout, 1, 1])],
        [numpy_helper.from_array(weights, "weights")],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)], ir_version=8)
    onnx.checker.check_model(model)
    onnx.save(model, path)


def prepare_hailo(public_key, output_dir):
    started = perf_counter_ns()
    if not isinstance(public_key, PublicKey):
        raise TypeError("Expected PublicKey")
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    if any(root.iterdir()):
        raise ArtifactError("Output directory must be empty; do not overwrite a compiled artifact")
    scheme = BinaryScheme(public_key)
    export_onnx(scheme, root / "model.onnx")
    save(public_key, root / "public.json")
    np.save(root / "weights.npy", scheme.weights, allow_pickle=False)
    probes = [
        (0,) * scheme.n,
        (1,) + (0,) * (scheme.n - 1),
        (0,) * (scheme.n - 1) + (-1,),
        tuple([1] * min(16, scheme.n) + [0] * max(0, scheme.n - 16)),
    ]
    frames = scheme.pack(probes)
    outputs = scheme.software(frames)
    np.savez(
        root / "vectors.npz",
        r=np.asarray(probes),
        inputs=frames,
        outputs=outputs,
        products=scheme.reconstruct(outputs, len(probes)),
    )
    manifest = {
        "version": 1,
        "profile": PROFILE,
        "status": "prepared",
        "key_sha256": key_hash(public_key),
        "params": asdict(public_key.params),
        "cin": scheme.cin,
        "cout": scheme.cout,
        "bits": scheme.bits,
        "chunk": 16,
        "onnx_layout": "NCHW",
        "runtime_layout": "NHWC",
        "python": platform.python_version(),
        "onnx": version("onnx"),
        "numpy": version("numpy"),
        "preparation_ns": perf_counter_ns() - started,
        "files": {
            name: sha256(root / name)
            for name in ("model.onnx", "weights.npy", "public.json", "vectors.npz")
        },
        "compiled": None,
    }
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def load_artifact(directory, *, compiled=False):
    root = Path(directory)
    try:
        m = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
        if m["version"] != 1 or m["profile"] != PROFILE:
            raise ArtifactError("Unsupported artifact profile")
        required = {"model.onnx", "weights.npy", "public.json", "vectors.npz"}
        if set(m["files"]) != required:
            raise ArtifactError("Missing artifact checksums")
        for name in required:
            if sha256(root / name) != m["files"][name]:
                raise ArtifactError(f"Checksum mismatch: {name}")
        key = load(root / "public.json")
        if not isinstance(key, PublicKey) or key_hash(key) != m["key_sha256"]:
            raise ArtifactError("Public key mismatch")
        scheme = BinaryScheme(key)
        if m["params"] != asdict(key.params) or any(
            m[k] != getattr(scheme, k) for k in ("cin", "cout", "bits")
        ):
            raise ArtifactError("Inconsistent dimensions or parameters")
        if m["chunk"] != 16 or m["onnx_layout"] != "NCHW" or m["runtime_layout"] != "NHWC":
            raise ArtifactError("Inconsistent layout")
        weights = np.load(root / "weights.npy", allow_pickle=False)
        if weights.dtype != np.uint8 or not np.array_equal(weights, scheme.weights):
            raise ArtifactError("Weights do not match the public key")
        if compiled:
            validate_compiled(root, m, scheme)
        return m, scheme
    except (OSError, KeyError, TypeError, json.JSONDecodeError) as exc:
        raise ArtifactError(f"Malformed or missing artifact: {exc}") from exc


def validate_compiled(root, manifest, scheme):
    c = manifest.get("compiled")
    if not c:
        raise ArtifactError(
            "No compiled HEF. Preparation is not an NPU implementation; compile and validate with DFC first"
        )
    if c.get("runtime_version") != "4.23.0" or c.get("architecture") != "HAILO8L":
        raise ArtifactError("Expected HAILO8L / HailoRT 4.23.0")
    for name in ("model.hef", "numeric-evidence.npz", "compiler-report.json"):
        if sha256(root / name) != c.get("files", {}).get(name):
            raise ArtifactError(f"Compiled evidence checksum mismatch: {name}")
    report = json.loads((root / "compiler-report.json").read_text(encoding="utf-8"))
    if c.get("arithmetic_profile", "unit-v1") == "binary-circulant-affine-v1":
        from .proof import prove_affine

        if (
            report.get("hef_sha256") != c["files"]["model.hef"]
            or report.get("onnx_sha256") != manifest["files"]["model.onnx"]
        ):
            raise ArtifactError("Affine evidence is not bound to the source and HEF")
        if not report.get("compiler_version"):
            raise ArtifactError("Missing compiler version")
        spec = report["affine"]
        for direction, size in (("input", scheme.cin), ("output", scheme.cout)):
            port = c[direction]
            if (
                not port.get("name")
                or port.get("shape") != [1, 1, size]
                or port.get("dtype") != "UINT8"
            ):
                raise ArtifactError("Unsupported affine stream shape or type")
            if any(port[k] != spec[direction][k] for k in ("scale", "zero_point")):
                raise ArtifactError("Affine stream constants differ from evidence")
        with np.load(root / "numeric-evidence.npz", allow_pickle=False) as evidence:
            prove_affine(scheme, evidence["effective_weights"], evidence["effective_bias"], spec)
        return
    if c.get("arithmetic_profile", "unit-v1") != "unit-v1":
        raise ArtifactError("Unknown arithmetic profile")
    # This report must be produced by a trusted DFC extraction tool, not hand certification.
    expected = {
        "hef_sha256": c["files"]["model.hef"],
        "onnx_sha256": manifest["files"]["model.onnx"],
        "operation": "single_linear_conv1x1",
        "activation": "identity",
        "accumulator_bits": 32,
        "requantization": "identity",
        "saturation": False,
        "input_scale": 1,
        "input_zero_point": 0,
        "output_scale": 1,
        "output_zero_point": 0,
    }
    if any(report.get(k) != v for k, v in expected.items()) or not report.get("compiler_version"):
        raise ArtifactError("Compiler report does not establish the exact linear profile")
    with np.load(root / "numeric-evidence.npz", allow_pickle=False) as evidence:
        w = evidence["effective_weights"]
        if w.shape != scheme.weights.shape or not np.array_equal(w, scheme.weights):
            raise ArtifactError("Compiled effective weights differ from binary weights")
        if evidence["effective_bias"].shape != (scheme.cout,) or np.any(evidence["effective_bias"]):
            raise ArtifactError("Compiled graph has a nonzero or malformed bias")
    for direction, size in (("input", scheme.cin), ("output", scheme.cout)):
        port = c.get(direction, {})
        if (
            not port.get("name")
            or port.get("shape") != [1, 1, size]
            or port.get("dtype") != "UINT8"
        ):
            raise ArtifactError("Unsupported compiled stream")
        if port.get("scale") != 1 or port.get("zero_point") != 0:
            raise ArtifactError("Stream quantization is not exact")
