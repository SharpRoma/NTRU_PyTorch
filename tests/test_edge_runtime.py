"""Low-memory loading and desktop API compatibility; no hardware substitutes."""

import hashlib
import subprocess
import sys
from io import BytesIO
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

import numpy as np
import pytest

from ntru_pytorch.hailo.artifacts import ArtifactError, sha256
from ntru_pytorch.hailo.evidence import evidence_rows
from ntru_pytorch.hailo.proof import prove_affine


def test_lazy_public_api_and_hailo_import():
    code = """
import sys
import ntru_pytorch as p
from ntru_pytorch.hailo import HailoEncryptor
from ntru_pytorch.cli import main
assert "torch" not in sys.modules and "sympy" not in sys.modules
assert "hailo_platform" not in sys.modules and "onnx" not in sys.modules
assert "TorchNTRU" in dir(p)
from ntru_pytorch import TorchNTRU, SympyNTRU, generate_keys, validate_keys
assert p.TorchNTRU is TorchNTRU
assert p.SympyNTRU is SympyNTRU
assert callable(generate_keys) and callable(validate_keys)
try:
    p.missing_attribute
except AttributeError:
    pass
else:
    raise AssertionError("unknown attribute accepted")
"""
    subprocess.run([sys.executable, "-c", code], check=True)


def test_hash_does_not_materialize_file(tmp_path, monkeypatch):
    data = bytes(range(256)) * 16384
    p = tmp_path / "blob"
    p.write_bytes(data)

    def no_read_bytes(*args):
        raise AssertionError("whole file read")

    monkeypatch.setattr(Path, "read_bytes", no_read_bytes)
    assert sha256(p) == hashlib.sha256(data).hexdigest()


@pytest.mark.parametrize("compressed", [False, True])
@pytest.mark.parametrize("kind", ["uint8", "strings"])
def test_streamed_evidence_same_proof(tmp_path, compressed, kind):
    from test_hailo_build import scheme, spec_for

    s = scheme()
    weights = s.weights if kind == "uint8" else s.weights.astype(str)
    bias = np.zeros(s.cout, dtype=np.int64)
    path = tmp_path / "evidence.npz"
    (np.savez_compressed if compressed else np.savez)(
        path, effective_weights=weights, effective_bias=bias
    )
    expected = prove_affine(s, weights, bias, spec_for(s))
    with evidence_rows(path, s.weights.shape) as (rows, b):
        assert prove_affine(s, rows, b, spec_for(s)) == expected
        assert all(np.array_equal(a, c) for a, c in zip(rows, weights, strict=True))
    bad = weights.copy()
    bad[0, 0] = "1/2" if kind == "strings" else 2
    np.savez(path, effective_weights=bad, effective_bias=bias)
    with evidence_rows(path, s.weights.shape) as (rows, b):
        with pytest.raises(ArtifactError):
            prove_affine(s, rows, b, spec_for(s))


@pytest.mark.parametrize("kind", ["shape", "object", "fortran", "truncated", "bias", "duplicate"])
def test_malformed_stream_fails_closed(tmp_path, kind):
    shape = (3, 4)
    a = np.ones(shape, dtype=np.uint8)
    b = np.zeros(3, dtype=np.int64)
    if kind == "shape":
        a = np.ones((2, 4), dtype=np.uint8)
    if kind == "object":
        a = a.astype(object)
    if kind == "fortran":
        a = np.asfortranarray(a)
    if kind == "bias":
        b = np.zeros(2)
    path = tmp_path / "bad.npz"

    def payload(array):
        out = BytesIO()
        np.save(out, array)
        return out.getvalue()

    raw = payload(a)
    if kind == "truncated":
        raw = raw[:-1]
    with ZipFile(path, "w", compression=ZIP_DEFLATED) as z:
        z.writestr("effective_weights.npy", raw)
        z.writestr("effective_bias.npy", payload(b))
        if kind == "duplicate":
            with pytest.warns(UserWarning, match="Duplicate"):
                z.writestr("effective_weights.npy", raw)
    with pytest.raises(ArtifactError):
        with evidence_rows(path, shape) as (rows, bias):
            list(rows)


@pytest.mark.parametrize("member", ["effective_weights.npy", "effective_bias.npy"])
def test_oversized_dtype_header_rejected_before_allocation(tmp_path, member):
    path = tmp_path / "oversized.npz"
    with ZipFile(path, "w") as z:
        for name, shape in [("effective_weights.npy", (3, 4)), ("effective_bias.npy", (3,))]:
            data = BytesIO()
            if name == member:
                np.lib.format.write_array_header_1_0(
                    data, dict(descr=np.dtype("U1048577").str, fortran_order=False, shape=shape)
                )
            else:
                np.save(data, np.zeros(shape, dtype=np.uint8))
            z.writestr(name, data.getvalue())
    with pytest.raises(ArtifactError, match="limit|size"):
        with evidence_rows(path, (3, 4)) as (rows, bias):
            list(rows)
