"""Real N701 hardware qualification with NumPy CPU references; no timing benchmark."""

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
    parser.add_argument("--bundle", default=str(Path.home() / "ntru-pi-bundle"))
    parser.add_argument("--output", required=True)
    parser.add_argument("--cases", type=int, default=1000)
    args = parser.parse_args()
    if args.cases < 1000:
        parser.error("At least 1000 random polynomials required")
    root, output = Path(args.bundle).resolve(), Path(args.output).resolve()
    if output.exists():
        raise ValueError("Preserve previous reports: choose a new output file")
    output.parent.mkdir(parents=True, exist_ok=True)
    started = perf_counter_ns()
    report = dict(
        passed=False,
        hardware_tested=False,
        hardware_qualification=False,
        qualification_scope="NPU partial sums, independent INT64 convolution, SparseNTRU ciphertexts/decryption, session recovery",
        torch_sympy_checked_on_pi=False,
        performance_benchmark=False,
        platform=platform.platform(),
        python=platform.python_version(),
        seed=42,
        page_size=os.sysconf("SC_PAGE_SIZE"),
        stage="imports",
    )

    def checkpoint(stage):
        report["stage"] = stage
        report["elapsed_ns"] = perf_counter_ns() - started
        output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
        print(stage, flush=True)

    try:
        import numpy as np

        from ntru_pytorch.formats import load
        from ntru_pytorch.hailo.artifacts import sha256
        from ntru_pytorch.hailo.engine import HailoEncryptor
        from ntru_pytorch.hailo.scheme import key_hash
        from ntru_pytorch.hailo.validation import validate_hardware
        from ntru_pytorch.randomness import RandomSource
        from ntru_pytorch.sparse_backend import SparseNTRU

        param = Path("/sys/module/hailo_pci/parameters/force_desc_page_size")
        report["force_desc_page_size"] = int(param.read_text())
        if report["page_size"] > 4096 and report["force_desc_page_size"] != 4096:
            raise ValueError("This Pi setup requires force_desc_page_size=4096")
        checkpoint("Проверка артефакта: может занять около двух минут...")
        engine = HailoEncryptor(
            root / "artifact", packing="active16-v1", host_processing="optimized"
        )
        engine.diagnostic_directory = str(output.parent / (output.stem + "-diagnostics"))
        key = engine.scheme.key
        private = load(root / "private.json")
        report.update(
            hef_sha256=sha256(root / "artifact/model.hef"),
            key_sha256=key_hash(key),
            private_key_sha256=sha256(root / "private.json"),
            script_sha256=sha256(Path(__file__)),
            numpy=np.__version__,
            packing=engine.packing,
            host_processing=engine.host_processing,
        )
        with engine:
            report["device"] = dict(engine.session.identity)
            checkpoint("Проверка частичных сумм и произведений на NPU...")
            original_infer = engine._infer
            calls = 0

            def infer(frames):
                nonlocal calls
                result = original_infer(frames)
                calls += 1
                if calls % 32 == 0:
                    print(f"Выполнено аппаратных вызовов: {calls}", flush=True)
                return result

            engine._infer = infer
            try:
                report["arithmetic"] = validate_hardware(engine, cases=args.cases, seed=42)
            finally:
                engine._infer = original_infer
            checkpoint("Проверка шифротекстов и CPU-расшифрования...")
            source = RandomSource(42)
            cpus = [SparseNTRU(kernel=k) for k in ("shift-sum", "cached-shifts")]
            messages = {
                "empty": b"",
                "unicode": "NTRU: Привет 🌍\0".encode(),
                "binary_nuls": b"\x00\xff\x01\x00",
                "multiblock": bytes(range(256)) * 2,
            }
            for name, data in messages.items():
                n = (8 * (len(data) + 8) + 700) // 701
                rs = tuple(source.ternary(701, key.params.d, key.params.d) for _ in range(n))
                actual = engine.encrypt_bytes(key, data, r_polys=rs)
                for cpu in cpus:
                    reference = cpu.encrypt_bytes(key, data, r_polys=rs)
                    if actual != reference or cpu.decrypt_bytes(private, actual) != data:
                        np.savez(
                            output.with_suffix(".mismatch.npz"),
                            data=np.frombuffer(data, dtype=np.uint8),
                            r=rs,
                            expected=reference.blocks,
                            actual=actual.blocks,
                        )
                        raise AssertionError((name, cpu.kernel, "ciphertext/decryption mismatch"))
            report.update(messages=list(messages), cpu_kernels=[c.kernel for c in cpus])
        checkpoint("Повторное открытие и очистка после исключения...")

        class HostFailure(Exception):
            pass

        try:
            with engine:
                if engine.encrypt_poly(key, [0], r=[0]) != (0,) * 701:
                    raise AssertionError("Reopened zero mismatch")
                raise HostFailure()
        except HostFailure:
            pass
        with engine:
            if engine.encrypt_poly(key, [1], r=[0]) != (1,) + (0,) * 700:
                raise AssertionError("Device did not recover after exception")
            if engine.session.identity != report["device"]:
                raise AssertionError("Physical device changed between sessions")
        if "torch" in sys.modules or "sympy" in sys.modules:
            raise AssertionError("Unexpected heavyweight CPU import")
        report.update(
            passed=True,
            hardware_tested=True,
            hardware_qualification=True,
            session_opens=3,
            cleanup_after_exception=True,
        )
        checkpoint("complete")
        print("Аппаратная проверка N701 с NumPy CPU-эталонами пройдена.", flush=True)
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
