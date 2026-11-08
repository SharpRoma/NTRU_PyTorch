"""CPU correctness tests, independent of NPU execution."""

import json
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from ntru_pytorch import (
    PRESETS,
    Parameters,
    PrivateKey,
    PublicKey,
    RandomSource,
    SparseNTRU,
    SympyNTRU,
    TorchNTRU,
    generate_keys,
)
from ntru_pytorch.sparse_backend import KERNELS


@pytest.fixture(params=[17, 107, 701])
def key(request):
    p = (
        Parameters(17, 3, 256, 3, 2, 1)
        if request.param == 17
        else (PRESETS["moderate"] if request.param == 107 else PRESETS["dead"])
    )
    return PublicKey(p, tuple((19 * i + 7) % p.q - p.q // 2 for i in range(p.N)))


@pytest.mark.parametrize("kernel", KERNELS)
def test_sparse_products_and_traces(key, kernel):
    n = key.params.N
    cpu = SparseNTRU(kernel=kernel)
    probes = [(0,) * n, (1,) * n, (-1,) * n, tuple((-1) ** i for i in range(n))]
    for count in (1, 15, 16, 17):
        for sign in (1, -1):
            indices = set(np.linspace(0, n - 1, min(count, n), dtype=int))
            probes.append(tuple(sign if i in indices else 0 for i in range(n)))
    rng = RandomSource(42)
    probes += [rng.ternary(n, key.params.d, key.params.d) for _ in range(1000)]
    for start in range(0, len(probes), 16):
        rs = probes[start : start + 16]
        actual = cpu.encrypt_polys(key, [(0,) * n] * len(rs), r_polys=rs)
        for r, out in zip(rs, actual, strict=True):
            raw = np.convolve(np.asarray(r, dtype=np.int64), np.asarray(key.h, dtype=np.int64))
            expected = raw[:n].copy()
            expected[:-1] += raw[n:]
            expected %= key.params.q
            expected = np.where(
                expected >= (key.params.q + 1) // 2, expected - key.params.q, expected
            )
            assert out == tuple(expected)
    for r in probes[:12]:
        traces = []
        for engine in (cpu, TorchNTRU(), SympyNTRU()):
            events = []
            result = engine.encrypt_poly(
                key, [1, -1], r=r, trace=lambda *event, sink=events: sink.append(event)
            )
            traces.append((result, events))
        assert traces[0] == traces[1] == traces[2]
    assert cpu.preparation["prepared_bytes"] <= cpu.max_prepared_bytes


@pytest.mark.parametrize("kernel", KERNELS)
def test_messages_and_cross_decryption(kernel):
    public, private = generate_keys(Parameters(17, 3, 256, 3, 2, 1), random_source=RandomSource(42))
    engines = (SparseNTRU(kernel=kernel), TorchNTRU(), SympyNTRU())
    for data in (b"", b"a\0b\0", "Привет 😀".encode(), bytes(range(256)), b"x" * 40):
        ciphers = [e.encrypt_bytes(public, data, random_source=RandomSource(4)) for e in engines]
        assert ciphers[0] == ciphers[1] == ciphers[2]
        for cipher in ciphers:
            assert all(e.decrypt_bytes(private, cipher) == data for e in engines)
    events = []
    for e in engines:
        trace = []
        cipher = e.encrypt_poly(
            public, [1, 0, 1], r=[1, -1], trace=lambda *a, sink=trace: sink.append(a)
        )
        assert (
            e.decrypt_poly(private, cipher, trace=lambda *a, sink=trace: sink.append(a))
            == (1, 0, 1) + (0,) * 14
        )
        events.append(trace)
    assert events[0] == events[1] == events[2]


@pytest.mark.parametrize("kernel", KERNELS)
def test_cache_and_invalid_inputs(key, kernel):
    e = SparseNTRU(kernel=kernel)
    first = e.prepare(key)
    assert e.prepare(key) == first and e.prepare(replace(key)) == first
    changed = replace(key, h=(key.h[0] + 1,) + key.h[1:])
    assert e.encrypt_poly(changed, [1], r=[1]) != e.encrypt_poly(key, [1], r=[1])
    assert e.prepare(key)["kernel"] == kernel
    for bad in ([True], [1.0], [2], [2**80], [1] * (key.params.N + 1), [np.bool_(True)]):
        with pytest.raises(ValueError):
            e.encrypt_poly(key, bad, r=[0])
        with pytest.raises(ValueError):
            e.encrypt_poly(key, [0], r=bad)
    with pytest.raises(ValueError):
        e.encrypt_bytes(key, b"abc", r_polys=[])
    with pytest.raises(ValueError):
        e.encrypt_polys(key, [], r_polys=[])
    with pytest.raises(TypeError):
        e.encrypt_bytes(key, "text")
    prepared_before_error = e.prepare(key)
    with pytest.raises(OverflowError):
        e.prepare(replace(key, h=(2**63,) + (0,) * (key.params.N - 1)))
    assert e.prepare(key) == prepared_before_error
    with pytest.raises(ValueError, match="memory limit"):
        SparseNTRU(kernel=kernel, max_prepared_bytes=1).prepare(key)
    with pytest.raises(ValueError):
        e.decrypt_poly(PrivateKey(key.params, *([(1,) + (0,) * (key.params.N - 1)] * 4)), [1])


def test_sparse_import_does_not_load_other_backends():
    code = """import sys
from ntru_pytorch import SparseNTRU
assert "torch" not in sys.modules and "sympy" not in sys.modules
assert "ntru_pytorch.hailo" not in sys.modules
"""
    subprocess.run([sys.executable, "-c", code], check=True)
    with pytest.raises(ValueError):
        SparseNTRU(kernel="unknown")


def test_cpu_only_benchmark_without_other_backends(tmp_path):
    from ntru_pytorch.formats import save

    p = PRESETS["moderate"]
    f = (1,) + (0,) * (p.N - 1)
    public = PublicKey(p, (3, -3) + (0,) * (p.N - 2))
    private = PrivateKey(p, f, f, f, (1, -1) + (0,) * (p.N - 2))
    save(public, tmp_path / "public.json")
    save(private, tmp_path / "private.json")
    script = Path(__file__).resolve().parents[1] / "tools/benchmark_sparse_cpu.py"
    args = [
        sys.executable,
        str(script),
        "--public-key",
        str(tmp_path / "public.json"),
        "--private-key",
        str(tmp_path / "private.json"),
        "--output",
        str(tmp_path / "cpu"),
        "--runs",
        "2",
        "--warmup",
        "0",
    ]
    subprocess.run(args, check=True, capture_output=True)
    report = json.loads((tmp_path / "cpu.json").read_text())
    assert len(report["samples"]) == 16 and len(report["summary"]) == 8
    assert report["metadata"]["hardware_tested"] is False
    assert not any(report["metadata"]["imported_backends"].values())
    assert len(report["metadata"]["cpu_key_preparation"]) == 2
    assert all(r["execution"] == "cpu" for r in report["samples"])
    repeated = subprocess.run(args, capture_output=True, text=True)
    assert repeated.returncode != 0 and "Preserve previous experiments" in repeated.stderr


@pytest.mark.parametrize("p,q", [(2, 257), (3, 25), (5, 256)])
def test_other_moduli(p, q):
    params = Parameters(17, p, q, 3, 2, 1)
    f = (1,) + (0,) * 16
    public = PublicKey(params, (p, -p) + (0,) * 15)
    private = PrivateKey(params, f, f, f, (1, -1) + (0,) * 15)
    for kernel in KERNELS:
        sparse = SparseNTRU(kernel=kernel)
        data = b"\0test"
        actual = sparse.encrypt_bytes(public, data, random_source=RandomSource(42))
        for backend in (SympyNTRU(), TorchNTRU()):
            assert actual == backend.encrypt_bytes(public, data, random_source=RandomSource(42))
            assert backend.decrypt_bytes(private, actual) == data
        assert sparse.decrypt_bytes(private, actual) == data


def test_legacy_wildcard_does_not_require_optional_numpy():
    code = """
import sys, types, importlib.abc
class NoNumpy(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "numpy" or fullname.startswith("numpy."):
            raise ModuleNotFoundError("optional numpy deliberately absent", name=fullname)
sys.meta_path.insert(0, NoNumpy())
# Model required desktop backend availability, independently of their own optional NumPy imports.
for name, attrs in {"torch_backend": ["TorchNTRU"], "sympy_backend": ["SympyNTRU"],
                    "keygen": ["generate_keys", "validate_keys"]}.items():
    module = types.ModuleType("ntru_pytorch."+name)
    for attr in attrs:
        setattr(module, attr, object())
    sys.modules[module.__name__] = module
import ntru_pytorch
namespace = {}
exec("from ntru_pytorch import *", namespace)
assert "TorchNTRU" in namespace and "SympyNTRU" in namespace
assert "SparseNTRU" not in namespace and "SparseNTRU" in dir(ntru_pytorch)
assert "numpy" not in sys.modules
"""
    subprocess.run([sys.executable, "-c", code], check=True)
