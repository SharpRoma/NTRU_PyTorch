"""No downloads/builds: optional native tests consume verified local artifacts."""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
BUILD = ROOT / "results/native/oqs/0.16.0/x86_64/portable"


def test_oqs_import_is_lazy():
    code = """
import sys
import ntru_pytorch.oqs
from ntru_pytorch.oqs import OqsKem
assert not any(x in sys.modules for x in ('torch','sympy','hailo_platform','numpy'))
import ctypes
# No CDLL is constructed while importing the public class.
"""
    subprocess.run([sys.executable, "-c", code], check=True)


def test_buffer_validation():
    from ntru_pytorch.oqs.kem import buffer

    with pytest.raises(TypeError):
        buffer(bytearray(2), 2)
    with pytest.raises(ValueError):
        buffer(b"x", 2)
    assert buffer(b"xy", 2).raw == b"xy"


def test_missing_manifest_never_builds(tmp_path):
    from ntru_pytorch.oqs.loader import verify

    with pytest.raises(FileNotFoundError):
        verify(tmp_path)
    assert not list(tmp_path.iterdir())


def test_manifest_validation(tmp_path):
    from ntru_pytorch.oqs.loader import verify

    (tmp_path / "manifest.json").write_text(
        json.dumps(dict(configuration=dict(version="wrong", commit="", bridge_abi=1)))
    )
    with pytest.raises(ValueError):
        verify(tmp_path)


@pytest.fixture
def build():
    selected = Path(os.environ.get("NTRU_OQS_TEST_BUILD", BUILD))
    if not (selected / "manifest.json").exists():
        pytest.skip("Build liboqs explicitly to run native adapter tests")
    return selected


def test_kem_errors_and_lifecycle(build):
    from ntru_pytorch.oqs import OqsKem

    with pytest.raises(ValueError):
        OqsKem(build, "ML-KEM-768")
    with OqsKem(build) as kem:
        pk, sk = kem.keypair()
        ct, ss = kem.encapsulate(pk)
        assert kem.decapsulate(sk, ct) == ss
        with pytest.raises(ValueError):
            kem.encapsulate(pk[:-1])
        with pytest.raises(TypeError):
            kem.encapsulate(list(pk))
        with pytest.raises(ValueError):
            kem.decapsulate(sk[:-1], ct)
        with pytest.raises(ValueError):
            kem.decapsulate(sk, ct + b"x")
        altered = bytes([ct[0] ^ 1]) + ct[1:]
        assert kem.decapsulate(sk, altered) != ss
    kem.close()
    with pytest.raises(RuntimeError):
        kem.keypair()
    with pytest.raises(RuntimeError):
        with OqsKem(build):
            raise RuntimeError("injected")
    with OqsKem(build) as reopened:
        assert reopened.decapsulate(sk, ct) == ss


def test_rq_exact_modular_and_api(build):
    import numpy as np

    from ntru_pytorch.models import PRESETS, PublicKey
    from ntru_pytorch.oqs import OqsArithmeticEncryptor
    from ntru_pytorch.sparse_backend import SparseNTRU

    rng = np.random.default_rng(42)
    key = PublicKey(PRESETS["dead"], tuple(map(int, rng.integers(-10000, 10000, 701))))
    rs = rng.integers(-1, 2, (17, 701), dtype=np.int64)
    messages = rng.integers(-1, 2, (17, 701), dtype=np.int64)
    with OqsArithmeticEncryptor(build) as engine:
        assert engine.encrypt_polys(key, messages, r_polys=rs) == SparseNTRU().encrypt_polys(
            key, messages, r_polys=rs
        )
        assert engine.prepare(key) == engine.prepare(key)
        for data in (b"", b"a" * 79, b"b" * 80, "Привет 🌍".encode(), bytes(range(256))):
            n = (8 * (len(data) + 8) + 700) // 701
            r = rs[:n]
            assert engine.encrypt_bytes(key, data, r_polys=r) == SparseNTRU().encrypt_bytes(
                key, data, r_polys=r
            )
        with pytest.raises(ValueError):
            engine.encrypt_poly(key, [0], r=[True])
        with pytest.raises(ValueError):
            engine.encrypt_poly(key, [2], r=[1])
        with pytest.raises(ValueError):
            engine.encrypt_poly(key, [0], r=[2])
        with pytest.raises(ValueError):
            engine.encrypt_polys(key, [], r_polys=[])
        with pytest.raises(ValueError):
            engine.encrypt_bytes(key, b"x", r_polys=[])
        with pytest.raises(TypeError):
            engine.encrypt_bytes(key, "text")
        with pytest.raises(ValueError, match="trace"):
            engine.encrypt_poly(key, [0], r=[1], trace=lambda *a: None)
        with pytest.raises(ValueError, match="N=701"):
            engine.prepare(PublicKey(PRESETS["moderate"], (0,) * 107))
    engine.close()
    with pytest.raises(RuntimeError):
        engine.prepare(key)
    with pytest.raises(RuntimeError):
        with OqsArithmeticEncryptor(build):
            raise RuntimeError("injected")
    with OqsArithmeticEncryptor(build) as reopened:
        reopened.prepare(key)


def test_modified_binary_rejected(build, tmp_path):
    import shutil

    from ntru_pytorch.oqs.loader import verify

    data = json.loads((build / "manifest.json").read_text())
    # Only the first hashed entry is needed: verification must fail before loading.
    rel = next(iter(data["files"]))
    (tmp_path / "manifest.json").write_text(json.dumps(data))
    p = tmp_path / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(build / rel, p)
    with p.open("ab") as f:
        f.write(b"modified")
    with pytest.raises(ValueError, match="checksum"):
        verify(tmp_path)


def test_process_profile_isolation(build):
    sibling = build.parent / ("portable" if build.name == "native" else "native")
    if not (sibling / "manifest.json").exists():
        pytest.skip("Native build unavailable")
    code = """
from ntru_pytorch.oqs import OqsKem
import sys
with OqsKem(sys.argv[1]): pass
try: OqsKem(sys.argv[2])
except RuntimeError: pass
else: raise AssertionError('second liboqs profile accepted')
"""
    subprocess.run([sys.executable, "-c", code, str(build), str(sibling)], check=True)


def test_arithmetic_class_does_not_import_ml_libraries():
    subprocess.run(
        [
            sys.executable,
            "-c",
            "from ntru_pytorch.oqs import OqsArithmeticEncryptor; import sys; assert not any(n in sys.modules for n in ('torch','sympy','hailo_platform'))",
        ],
        check=True,
    )


def test_sizes_rejected_before_ffi(build, monkeypatch):
    from ntru_pytorch.oqs import OqsKem

    def forbidden(*args):
        raise AssertionError("FFI must not run for malformed byte lengths")

    with OqsKem(build) as kem:
        monkeypatch.setattr(kem.lib, "nr_encaps", forbidden)
        with pytest.raises(ValueError):
            kem.encapsulate(b"")
        monkeypatch.setattr(kem.lib, "nr_decaps", forbidden)
        with pytest.raises(ValueError):
            kem.decapsulate(b"0" * 1450, b"")
        with pytest.raises(ValueError):
            kem.decapsulate(b"", b"0" * 1138)


def test_native_boundary_guards(build):
    import numpy as np

    from ntru_pytorch.models import PRESETS, PublicKey
    from ntru_pytorch.oqs import OqsArithmeticEncryptor

    with OqsArithmeticEncryptor(build) as engine:
        h = np.zeros(701, dtype=np.uint16)
        assert engine.lib.nr_rq_prepare(None, 701) is None
        assert engine.lib.nr_rq_prepare(h.ctypes.data, 700) is None
        h[0] = 8192
        assert engine.lib.nr_rq_prepare(h.ctypes.data, 701) is None
        key = PublicKey(PRESETS["dead"], (1,) + (0,) * 700)
        engine.prepare(key)
        out = np.zeros_like(h)
        assert engine.lib.nr_rq_batch(engine._handle, h.ctypes.data, 700, out.ctypes.data, 1) != 0
        assert engine.lib.nr_rq_batch(engine._handle, h.ctypes.data, 701, out.ctypes.data, 1) != 0
        assert engine.lib.nr_rq_batch(engine._handle, None, 701, out.ctypes.data, 1) != 0
        assert engine.lib.nr_rq_batch(engine._handle, h.ctypes.data, 701, None, 1) != 0
        assert engine.lib.nr_rq_batch(engine._handle, h.ctypes.data, 0, out.ctypes.data, 0) != 0


def test_invalid_elf_rejected(tmp_path):
    import platform

    from ntru_pytorch.oqs.loader import COMMIT, digest, verify

    path = tmp_path / "library.so"
    path.write_bytes(b"not a shared library")
    data = dict(
        configuration=dict(
            version="0.16.0",
            commit=COMMIT,
            bridge_abi=1,
            architecture=platform.machine(),
            sanitizer=False,
            implementation="clean",
        ),
        files={"library.so": digest(path)},
        libraries=dict(kem="library.so", arithmetic="library.so", oqs="library.so"),
    )
    (tmp_path / "manifest.json").write_text(json.dumps(data))
    with pytest.raises(ValueError, match="ELF"):
        verify(tmp_path)


def test_cli_oqs_defaults_and_dispatch(monkeypatch):
    from ntru_pytorch import cli
    from ntru_pytorch.oqs import cli as oqs_cli

    captured = []
    monkeypatch.setattr(oqs_cli, "run", lambda args: captured.append(args))
    monkeypatch.setattr(sys, "argv", ["ntru-research", "oqs-info"])
    cli.main()
    assert captured[0].output == "results/oqs-info"
    assert captured[0].qualification_root == "results/oqs-qualification"
    assert captured[0].oqs_profile == "portable"


def test_worker_failure_saved(tmp_path, monkeypatch):
    from types import SimpleNamespace

    from ntru_pytorch.oqs import benchmark

    def fail(*args, **kwargs):
        raise subprocess.CalledProcessError(1, args[0])

    monkeypatch.setattr(benchmark.subprocess, "run", fail)
    args = SimpleNamespace(
        output=str(tmp_path / "result"),
        oqs_root="build",
        qualification_root="check",
        runs=20,
        warmup=3,
        seed=42,
    )
    with pytest.raises(subprocess.CalledProcessError):
        benchmark.orchestrate(args, "kem")
    report = json.loads((tmp_path / "result/report.json").read_text())
    assert report["passed"] is False
    assert report["failed_worker"]["profile"] == "portable"
    assert report["completed_workers"] == []
    assert json.loads((tmp_path / "result/progress.json").read_text())["stage"] == "failed"


def test_statistics():
    from ntru_pytorch.oqs.benchmark import stats

    result = stats([10, 20, 30])
    assert result == dict(median_ns=20.0, p95_ns=29.0)
