"""Small real-device N701 smoke; NumPy only. Not the 1000-case qualification."""

import argparse
import json
import os
import platform
import resource
import sys
from pathlib import Path
from time import perf_counter_ns


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", default=str(Path(__file__).resolve().parent))
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    root, output = Path(args.bundle).resolve(), Path(args.output).resolve()
    if output.exists():
        raise ValueError("Preserve previous reports; choose a new output file")
    output.parent.mkdir(parents=True, exist_ok=True)
    report = dict(
        passed=False,
        hardware_tested=False,
        hardware_qualification=False,
        platform=platform.platform(),
        python=platform.python_version(),
        page_size=os.sysconf("SC_PAGE_SIZE"),
        seed=42,
        stage="imports",
    )
    started = perf_counter_ns()
    engine = None
    try:
        import numpy as np

        from ntru_pytorch.formats import load
        from ntru_pytorch.hailo.artifacts import sha256
        from ntru_pytorch.hailo.engine import HailoEncryptor
        from ntru_pytorch.hailo.scheme import key_hash
        from ntru_pytorch.randomness import RandomSource
        from ntru_pytorch.sparse_backend import SparseNTRU

        artifact = root / "artifact"
        report["stage"] = "artifact_validation"
        print("Проверка хешей и численных свидетельств HEF. Ожидайте...", flush=True)
        engine = HailoEncryptor(artifact, packing="active16-v1", host_processing="optimized")
        engine.diagnostic_directory = str(output.parent / (output.stem + "-diagnostics"))
        key = engine.scheme.key
        private = load(root / "private.json")
        report.update(
            hef_sha256=sha256(artifact / "model.hef"),
            key_sha256=key_hash(key),
            artifact_validation_ns=perf_counter_ns() - started,
        )
        source = RandomSource(42)
        probes = [(0,) * 701]
        for pos in (0, 700):
            for sign in (-1, 1):
                probes.append(tuple(sign if i == pos else 0 for i in range(701)))
        for count in (15, 16, 17):
            for sign in (-1, 1):
                probes.append(tuple(sign if i >= 701 - count else 0 for i in range(701)))
        probes += [source.ternary(701, key.params.d, key.params.d) for _ in range(8)]
        report["stage"] = "NPU_partial_sums"
        print("Открытие Hailo и проверка частичных сумм...", flush=True)
        with engine:
            report["device"] = engine.session.identity
            batch = engine.scheme.pack_batch(probes, "active16-v1", host_processing="optimized")
            actual = engine._infer(batch.frames)
            expected = engine.scheme.software(batch.frames)
            if not np.array_equal(actual, expected):
                np.savez(
                    output.with_suffix(".mismatch.npz"),
                    inputs=batch.frames,
                    actual=actual,
                    expected=expected,
                )
                raise AssertionError("NPU partial sums mismatch")
            products = engine.scheme.reconstruct_batch(actual, batch, host_processing="optimized")
            h = np.asarray([v % 8192 for v in key.h], dtype=np.int64)
            for index, (r, product) in enumerate(zip(probes, products, strict=True)):
                raw = np.convolve(np.asarray(r, dtype=np.int64), h)
                reference = raw[:701].copy()
                reference[:-1] += raw[701:]
                if not np.array_equal(product, reference):
                    np.savez(
                        output.with_suffix(".mismatch.npz"), r=r, actual=product, expected=reference
                    )
                    raise AssertionError(("Cyclic product mismatch", index))
            report.update(polynomial_probes=len(probes), probe_frames=len(batch.frames))
            report["stage"] = "ciphertexts"
            print("Проверка шифротекстов и расшифрования...", flush=True)
            cpu = SparseNTRU()
            messages = [b"", "NTRU: Привет 🌍\0".encode(), bytes(range(256))]
            for data in messages:
                n = (8 * (len(data) + 8) + 700) // 701
                rs = tuple(source.ternary(701, key.params.d, key.params.d) for _ in range(n))
                ciphertext = engine.encrypt_bytes(key, data, r_polys=rs)
                reference = cpu.encrypt_bytes(key, data, r_polys=rs)
                if ciphertext != reference or cpu.decrypt_bytes(private, ciphertext) != data:
                    raise AssertionError("Ciphertext/cross-decryption mismatch")
            report["messages"] = len(messages)
        report["stage"] = "resource_recovery"

        class InjectedHostError(Exception):
            pass

        try:
            with engine:
                if engine.encrypt_poly(key, [0], r=[0]) != (0,) * 701:
                    raise AssertionError("Reopen zero mismatch")
                raise InjectedHostError()
        except InjectedHostError:
            pass
        with engine:
            if engine.encrypt_poly(key, [1], r=[0]) != (1,) + (0,) * 700:
                raise AssertionError("Recovery after exception failed")
        report.update(
            passed=True,
            hardware_tested=True,
            stage="complete",
            session_opens=3,
            cleanup_after_exception=True,
        )
        print("NTRU N701: аппаратный smoke пройден. Это ещё не полная квалификация.", flush=True)
    except Exception as exc:
        report["error"] = repr(exc)
        raise
    finally:
        report.update(
            elapsed_ns=perf_counter_ns() - started,
            peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
            imported_backends={n: n in sys.modules for n in ("torch", "sympy", "hailo_platform")},
        )
        output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
        print("Отчёт:", output, flush=True)


if __name__ == "__main__":
    main()
