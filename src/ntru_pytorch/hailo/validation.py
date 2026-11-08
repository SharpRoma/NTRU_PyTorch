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
    for name in ("numpy", "torch", "sympy", "onnx", "onnxruntime", "hailort", "netifaces"):
        try:
            versions[name] = version(name)
        except PackageNotFoundError:
            versions[name] = None
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": versions,
    }


def validate_software(key, *, cases=1000, seed=42, packing="position16-v1"):
    if type(cases) is not int or cases < 1:
        raise ValueError("cases must be positive")
    engine = SoftwareHailoEncryptor(key, packing=packing)
    rs = RandomSource(seed)
    h = np.asarray([v % key.params.q for v in key.h], dtype=np.int64)
    n = key.params.N
    for _ in range(cases):
        r = rs.ternary(n, key.params.d, key.params.d)
        batch = engine.scheme.pack_batch([r], packing)
        actual = engine.scheme.reconstruct_batch(engine.scheme.software(batch.frames), batch)[0]
        linear = np.convolve(np.asarray(r, dtype=np.int64), h)
        expected = linear[:n].copy()
        expected[: n - 1] += linear[n:]
        if not np.array_equal(actual, expected):
            raise ArtifactError("Binary decomposition differs from independent integer convolution")
    return {
        "stage": "software",
        "packing": packing,
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
    if type(cases) is not int or cases < 1000:
        raise ValueError("Hardware qualification requires at least 1000 random polynomials")
    rs = RandomSource(seed)
    n = engine.scheme.n
    probes = [(0,) * n]
    # Every column and both signs, plus max tile sum and the tail tile.
    for i in range(n):
        probes.append(tuple(1 if j == i else 0 for j in range(n)))
        probes.append(tuple(-1 if j == i else 0 for j in range(n)))
    for lo in range(0, n, 16):
        for sign in (1, -1):
            probes.append(tuple(sign if lo <= i < lo + 16 else 0 for i in range(n)))
    probes.extend([(1,) * n, (-1,) * n, tuple(1 if i % 2 else -1 for i in range(n))])
    for count in (15, 16, 17):
        for sign in (1, -1):
            # Spread active positions across positional tiles where possible.
            active = set(np.linspace(0, n - 1, min(count, n), dtype=int).tolist())
            probes.append(tuple(sign if i in active else 0 for i in range(n)))
    probes.extend(
        rs.ternary(n, engine.scheme.key.params.d, engine.scheme.key.params.d) for _ in range(cases)
    )
    for start in range(0, len(probes), 8):
        subset = probes[start : start + 8]
        mode = getattr(engine, "host_processing", "reference")
        batch = engine.scheme.pack_batch(subset, engine.packing, host_processing=mode)
        actual = engine._infer(batch.frames)
        expected = engine.scheme.software(batch.frames)
        if not np.array_equal(actual, expected):
            _hardware_mismatch(engine, "partial_sums", seed, start, subset, expected, actual)
        products = engine.scheme.reconstruct_batch(actual, batch, host_processing=mode)
        h = np.asarray([v % engine.scheme.key.params.q for v in engine.scheme.key.h])
        for offset, (polynomial, product) in enumerate(zip(subset, products, strict=True)):
            linear = np.convolve(np.asarray(polynomial, dtype=np.int64), h)
            reference = linear[:n].copy()
            reference[:n - 1] += linear[n:]
            if not np.array_equal(product, reference):
                _hardware_mismatch(engine, "reconstruction", seed, start+offset,
                                   [polynomial], reference, product)
    return {
        "stage": "hardware", "cases": len(probes), "random_cases": cases,
        "seed": seed, "exact": True,
        "packing": engine.packing,
        "host_processing": getattr(engine, "host_processing", "reference"),
    }


def _hardware_mismatch(engine, stage, seed, index, inputs, expected, actual):
    directory = getattr(engine, "diagnostic_directory", None)
    if directory is not None:
        target = Path(directory)
        target.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(target / f"mismatch-{stage}-{index}.npz", seed=seed, index=index,
                            inputs=inputs, expected=expected, actual=actual)
    if np.asarray(expected).shape != np.asarray(actual).shape:
        raise ArtifactError(f"Hardware {stage} mismatch: seed={seed}, input={index}, "
                            f"expected shape={np.asarray(expected).shape}, "
                            f"actual shape={np.asarray(actual).shape}")
    unequal = np.argwhere(np.asarray(expected) != np.asarray(actual))
    first = tuple(unequal[0])
    raise ArtifactError(f"Hardware {stage} mismatch: seed={seed}, input={index}, "
                        f"coordinate={first}, expected={np.asarray(expected)[first]}, "
                        f"actual={np.asarray(actual)[first]}")


def qualify_hardware(directory, private_key, *, cases=1000, seed=42,
                     packing="position16-v1", cache_affine_lut=True,
                     host_processing="reference", diagnostic_directory=None):
    """Qualify actual admitted HEF sessions, including recovery after a host exception.

    This deliberately does not inject device faults or claim recovery from a
    physical PCIe failure. Every reopened session must execute real inference.
    """
    from .. import SympyNTRU, TorchNTRU
    from ..keygen import validate_keys
    from .engine import HailoEncryptor

    if type(cases) is not int or cases < 1000:
        raise ValueError("Hardware qualification requires at least 1000 random polynomials")
    engine = HailoEncryptor(directory, packing=packing, cache_affine_lut=cache_affine_lut,
                            host_processing=host_processing)
    engine.diagnostic_directory = diagnostic_directory
    key = engine.scheme.key
    validate_keys(key, private_key)
    messages = {
        "empty": b"", "binary_nuls": b"\0a\0b\0",
        "unicode": "NTRU: Привет 😀\0".encode(),
        "all_bytes_multiblock": bytes(range(256)) * 2,
    }
    report = {
        "hardware_tested": False, "key_sha256": key_hash(key),
        "hef_sha256": engine.manifest["compiled"]["files"]["model.hef"],
        "environment": environment(), "seed": seed,
        "packing": packing, "cache_affine_lut": cache_affine_lut,
        "host_processing": host_processing,
    }
    with engine:
        report["device"] = engine.session.identity
        report["partial_sums"] = validate_hardware(engine, cases=cases, seed=seed)
        for message_index, data in enumerate(messages.values()):
            actual = engine.encrypt_bytes(key, data, random_source=RandomSource(seed))
            for reference in (SympyNTRU(), TorchNTRU()):
                expected = reference.encrypt_bytes(key, data, random_source=RandomSource(seed))
                if expected != actual:
                    _hardware_mismatch(engine, "ciphertext", seed, message_index,
                                       np.frombuffer(data, dtype=np.uint8),
                                       expected.blocks, actual.blocks)
                try:
                    recovered = reference.decrypt_bytes(private_key, actual)
                except Exception as exc:
                    if diagnostic_directory is not None:
                        target = Path(diagnostic_directory)
                        target.mkdir(parents=True, exist_ok=True)
                        np.savez_compressed(target / f"decryption-error-{message_index}.npz",
                                            seed=seed, index=message_index,
                                            expected=np.frombuffer(data, dtype=np.uint8),
                                            ciphertext=actual.blocks, error=str(exc))
                    raise
                if recovered != data:
                    _hardware_mismatch(engine, "plaintext", seed, message_index,
                                       np.asarray(actual.blocks),
                                       np.frombuffer(data, dtype=np.uint8),
                                       np.frombuffer(recovered, dtype=np.uint8))
    report["messages"] = list(messages)

    class QualificationHostError(Exception):
        pass

    try:
        with engine:
            actual = engine.encrypt_poly(key, [0], r=[0])
            if actual != (0,) * key.params.N:
                raise ArtifactError("Reopened hardware session returned a wrong zero result")
            raise QualificationHostError("Controlled host failure after successful inference")
    except QualificationHostError:
        pass
    with engine:
        actual = engine.encrypt_poly(key, [1], r=[0])
        if actual != (1,) + (0,) * (key.params.N - 1):
            raise ArtifactError("Hardware session did not recover after host exception")
        if engine.session.identity != report["device"]:
            raise ArtifactError("Physical device identity changed between sessions")
    report.update(
        hardware_tested=True, exact=True, session_opens=3,
        cleanup_after_host_exception="passed", cross_decryption="passed",
    )
    return report
