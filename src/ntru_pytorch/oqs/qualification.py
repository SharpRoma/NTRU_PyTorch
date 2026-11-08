"""Native qualification. KAT deterministic RNG is confined to upstream executable."""

import hashlib
import json
import subprocess
from pathlib import Path
from time import perf_counter_ns

from .loader import digest, verify

KAT_SINGLE = "501e000c3eb374ffbfb81b0f16673a6282116465936608d7d164b05635e769e8"
KAT_ALL = "684d7e77641399b136fbb4b9261bdca6e83887453a0565e2aef3540821138ada"


def kat(build_dir, output):
    root, _ = verify(build_dir)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    hashes = {}
    for name, extra, expected in [("single", [], KAT_SINGLE), ("all", ["--all"], KAT_ALL)]:
        raw = subprocess.check_output(
            [str(root / "build/tests/kat_kem"), "NTRU-HRSS-701", *extra]
        ).replace(b"\r\n", b"\n")
        (output / f"kat-{name}.rsp").write_bytes(raw)
        value = hashlib.sha256(raw).hexdigest()
        if value != expected:
            raise AssertionError("Upstream KAT hash mismatch: " + name)
        hashes[name] = value
    # The expected full stdout hash authenticates every parsed KAT vector.
    vectors = []
    current = {}
    for line in raw.decode().splitlines():
        if " = " not in line:
            continue
        key, value = line.split(" = ", 1)
        if key == "count":
            if current:
                vectors.append(current)
            current = {"count": int(value)}
        elif key in ("pk", "sk", "ct", "ss"):
            current[key] = value
    if current:
        vectors.append(current)
    if len(vectors) != 100 or any(set(v) != {"count", "pk", "sk", "ct", "ss"} for v in vectors):
        raise AssertionError("Unexpected upstream KAT format")
    (output / "kat-corpus.json").write_text(json.dumps(vectors, indent=2) + "\n")
    return hashes


def check(build_dir, output, *, cases=1000, seed=42, public_key=None, private_key=None):
    if cases < 1000:
        raise ValueError("Qualification requires at least 1000 random cases")
    import numpy as np

    from ..models import PRESETS, PublicKey
    from .arithmetic import OqsArithmeticEncryptor
    from .kem import OqsKem

    root, manifest = verify(build_dir)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    report = dict(
        hardware_tested=False,
        build_manifest_sha256=digest(root / "manifest.json"),
        configuration=manifest["configuration"],
        seed=seed,
        random_cases=cases,
        stage="KAT",
    )
    started = perf_counter_ns()
    try:
        report["kat"] = kat(root, output)
        vectors = json.loads((output / "kat-corpus.json").read_text())
        report["stage"] = "KEM"
        with OqsKem(root) as kem:
            for vector in vectors:
                assert (
                    kem.decapsulate(bytes.fromhex(vector["sk"]), bytes.fromhex(vector["ct"]))
                    .hex()
                    .upper()
                    == vector["ss"].upper()
                )
            for index in range(cases):
                pk, sk = kem.keypair()
                ct, ss = kem.encapsulate(pk)
                recovered = kem.decapsulate(sk, ct)
                if recovered != ss:
                    (output / "kem-mismatch.json").write_text(
                        json.dumps(
                            dict(
                                stage="KEM roundtrip",
                                index=index,
                                pk=pk.hex(),
                                sk=sk.hex(),
                                ct=ct.hex(),
                                expected=ss.hex(),
                                actual=recovered.hex(),
                            ),
                            indent=2,
                        )
                        + "\n"
                    )
                    raise AssertionError(("KEM roundtrip", index))
            for method, args in [
                (kem.encapsulate, (b"",)),
                (kem.decapsulate, (sk, b"")),
                (kem.decapsulate, (b"", ct)),
            ]:
                try:
                    method(*args)
                except ValueError:
                    pass
                else:
                    raise AssertionError("Invalid size accepted")
            corrupted = bytes([ct[0] ^ 1]) + ct[1:]
            rejected = kem.decapsulate(sk, corrupted)
            assert len(rejected) == 32 and rejected != ss
            # Normal system RNG outputs for cross-profile interoperability workers.
            (output / "interop.json").write_text(
                json.dumps(dict(pk=pk.hex(), sk=sk.hex(), ct=ct.hex(), ss=ss.hex())) + "\n"
            )
        kem.close()
        with OqsKem(root) as reopened:
            assert reopened.decapsulate(sk, ct) == ss
        report.update(
            kem_roundtrips=cases, kat_vectors=100, implicit_rejection=True, kem_reopen=True
        )
        report["stage"] = "arithmetic"
        rng = np.random.default_rng(seed)
        h = rng.integers(0, 8192, 701, dtype=np.int64)
        key = PublicKey(PRESETS["dead"], tuple(map(int, h)))
        with OqsArithmeticEncryptor(root) as engine:
            engine.prepare(key)
            boundaries = [
                np.zeros(701, dtype=np.int64),
                np.ones(701, dtype=np.int64),
                -np.ones(701, dtype=np.int64),
                np.arange(701) % 2 * 2 - 1,
            ]
            for pos in range(701):
                row = np.zeros(701, dtype=np.int64)
                row[pos] = 1
                boundaries.extend((row, -row))
            for count in (15, 16, 17):
                for sign in (-1, 1):
                    row = np.zeros(701, dtype=np.int64)
                    row[-count:] = sign
                    boundaries.append(row)
            for kind, rows in [("boundary", boundaries), ("random", None)]:
                total = len(rows) if rows is not None else cases
                for begin in range(0, total, 16):
                    batch = (
                        np.asarray(rows[begin : begin + 16])
                        if rows is not None
                        else rng.integers(-1, 2, (min(16, total - begin), 701), dtype=np.int64)
                    )
                    actual = engine._products(batch)
                    for offset, row in enumerate(batch):
                        raw = np.convolve(h, row)
                        expected = raw[:701].copy()
                        expected[:-1] += raw[701:]
                        expected %= 8192
                        if not np.array_equal(actual[offset], expected):
                            np.savez(
                                output / "arithmetic-mismatch.npz",
                                h=h,
                                r=row,
                                expected=expected,
                                actual=actual[offset],
                            )
                            raise AssertionError((kind, begin + offset))
            # Canonical full-range residues test: also bounds the uint16/mod-q conversion.
            dense = rng.integers(0, 8192, (16, 701), dtype=np.int64)
            got = engine._products(dense)
            for r, a in zip(dense, got, strict=True):
                raw = np.convolve(h, r)
                e = raw[:701].copy()
                e[:-1] += raw[701:]
                assert np.array_equal(a, e % 8192)
        report.update(
            arithmetic_boundary_cases=len(boundaries),
            arithmetic_random_cases=cases,
            full_range_residue_cases=16,
        )
        if public_key and private_key:
            report["stage"] = "project_encryption"
            report["project_encryption"] = check_encryption(root, public_key, private_key, seed)
        report.update(stage="complete", passed=True, elapsed_ns=perf_counter_ns() - started)
    except Exception as exc:
        report.update(passed=False, error=repr(exc), elapsed_ns=perf_counter_ns() - started)
        raise
    finally:
        (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def check_encryption(build_dir, public_path, private_path, seed):
    from .. import RandomSource, SympyNTRU, TorchNTRU
    from ..formats import load
    from ..sparse_backend import SparseNTRU
    from .arithmetic import OqsArithmeticEncryptor

    public, private = load(public_path), load(private_path)
    cpu = [
        SympyNTRU(),
        TorchNTRU(),
        SparseNTRU(kernel="shift-sum"),
        SparseNTRU(kernel="cached-shifts"),
    ]
    source = RandomSource(seed)
    count = 0
    with OqsArithmeticEncryptor(build_dir) as oqs:
        for data in (
            b"",
            "Привет 🌍".encode(),
            bytes(range(256)),
            b"x" * 79,
            b"x" * 80,
            b"x" * 168,
        ):
            n = (8 * (len(data) + 8) + 700) // 701
            rs = tuple(source.ternary(701, public.params.d, public.params.d) for _ in range(n))
            actual = oqs.encrypt_bytes(public, data, r_polys=rs)
            for engine in cpu:
                assert engine.encrypt_bytes(public, data, r_polys=rs) == actual
                assert engine.decrypt_bytes(private, actual) == data
            count += 1
    return dict(
        messages=count,
        backends=["SymPy", "PyTorch", "SparseNTRU shift-sum", "SparseNTRU cached-shifts"],
        hardware_tested=False,
    )
