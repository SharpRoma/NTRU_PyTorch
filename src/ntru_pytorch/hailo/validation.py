"""Reproducible checks; reports distinguish host, ONNX and hardware execution."""

import platform
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import numpy as np

from ..randomness import RandomSource
from .artifacts import ArtifactError, load_artifact
from .engine import SoftwareHailoEncryptor
from .scheme import key_hash


def environment():
    versions = {}
    for name in ("numpy", "torch", "onnx", "onnxruntime", "hailort", "netifaces"):
        try:
            versions[name] = version(name)
        except PackageNotFoundError:
            versions[name] = None
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": versions,
    }


def validate_software(key, *, cases=1000, seed=42):
    if type(cases) is not int or cases < 1:
        raise ValueError("cases must be positive")
    engine = SoftwareHailoEncryptor(key)
    rs = RandomSource(seed)
    h = np.asarray([v % key.params.q for v in key.h], dtype=np.int64)
    n = key.params.N
    for _ in range(cases):
        r = rs.ternary(n, key.params.d, key.params.d)
        frames = engine.scheme.pack([r])
        actual = engine.scheme.reconstruct(engine.scheme.software(frames), 1)[0]
        linear = np.convolve(np.asarray(r, dtype=np.int64), h)
        expected = linear[:n].copy()
        expected[: n - 1] += linear[n:]
        if not np.array_equal(actual, expected):
            raise ArtifactError("Binary decomposition differs from independent integer convolution")
    return {
        "stage": "software",
        "cases": cases,
        "seed": seed,
        "exact": True,
        "key_sha256": key_hash(key),
    }


def validate_onnx(directory):
    try:
        import onnxruntime as ort
    except ImportError as exc:
        raise ArtifactError("Install ntru-pytorch[hailo-export] for ONNX validation") from exc
    _, scheme = load_artifact(directory)
    root = Path(directory)
    session = ort.InferenceSession(str(root / "model.onnx"), providers=["CPUExecutionProvider"])
    checked = 0
    with np.load(root / "vectors.npz", allow_pickle=False) as vectors:
        for frame, expected in zip(vectors["inputs"], vectors["outputs"], strict=True):
            out = session.run(
                ["partial"], {"r": frame.transpose(2, 0, 1)[None].astype(np.float32)}
            )[0]
            actual = out.transpose(0, 2, 3, 1)[0]
            if not np.array_equal(actual, expected):
                raise ArtifactError("ONNX output differs from exact partial sums")
            checked += 1
    return {"stage": "onnx_cpu", "frames": checked, "exact": True, "hardware_tested": False}


def validate_hardware(engine, *, cases=1000, seed=42):
    rs = RandomSource(seed)
    n = engine.scheme.n
    probes = [(0,) * n]
    # Every column and both signs, plus max tile sum and the tail tile.
    for i in range(n):
        probes.append(tuple(1 if j == i else 0 for j in range(n)))
        probes.append(tuple(-1 if j == i else 0 for j in range(n)))
    probes.extend([tuple(1 if i < 16 else 0 for i in range(n)), (1,) * n, (-1,) * n])
    probes.extend(
        rs.ternary(n, engine.scheme.key.params.d, engine.scheme.key.params.d) for _ in range(cases)
    )
    for start in range(0, len(probes), 8):
        frames = engine.scheme.pack(probes[start : start + 8])
        actual = engine._infer(frames)
        expected = engine.scheme.software(frames)
        if not np.array_equal(actual, expected):
            raise ArtifactError("Hardware partial sums differ from exact reference")
    return {"stage": "hardware", "cases": len(probes), "seed": seed, "exact": True}
