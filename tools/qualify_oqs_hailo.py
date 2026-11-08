"""Real N701 device qualification against both pinned OQS arithmetic profiles."""

import argparse
import json
from pathlib import Path

from ntru_pytorch import RandomSource, SympyNTRU, TorchNTRU
from ntru_pytorch.formats import load
from ntru_pytorch.hailo.engine import HailoEncryptor
from ntru_pytorch.hailo.validation import environment, validate_hardware
from ntru_pytorch.oqs import OqsArithmeticEncryptor
from ntru_pytorch.oqs.loader import digest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", required=True)
    parser.add_argument("--private-key", required=True)
    parser.add_argument("--oqs-root", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    target = Path(args.output)
    target.mkdir(parents=True, exist_ok=False)
    report = dict(hardware_tested=False, passed=False, stage="open", seed=42)
    adapters = []
    try:
        engine = HailoEncryptor(args.artifact, packing="active16-v1", host_processing="optimized")
        engine.diagnostic_directory = str(target)
        key = engine.scheme.key
        private = load(args.private_key)
        adapters = [
            OqsArithmeticEncryptor(Path(args.oqs_root) / profile)
            for profile in ("portable", "native")
        ]
        with engine:
            report.update(
                stage="partial_sums",
                device=engine.session.identity,
                environment=environment(),
                hef_sha256=digest(Path(args.artifact) / "model.hef"),
                oqs_manifest_sha256={
                    str(a.build_dir): digest(a.build_dir / "manifest.json") for a in adapters
                },
            )
            report["partial_sums"] = validate_hardware(engine, cases=1000, seed=42)
            report["stage"] = "ciphertexts_and_cross_decryption"
            messages = [b"", "Привет 🌍\0".encode(), bytes(range(256)) * 2]
            for index, data in enumerate(messages):
                rs = RandomSource(42 + index)
                n = (8 * (len(data) + 8) + 700) // 701
                polys = tuple(rs.ternary(701, key.params.d, key.params.d) for _ in range(n))
                actual = engine.encrypt_bytes(key, data, r_polys=polys)
                for adapter in adapters:
                    assert adapter.encrypt_bytes(key, data, r_polys=polys) == actual
                for cpu in (SympyNTRU(), TorchNTRU()):
                    assert cpu.encrypt_bytes(key, data, r_polys=polys) == actual
                    assert cpu.decrypt_bytes(private, actual) == data
        report["stage"] = "reopen_after_exception"

        class HostFailure(Exception):
            pass

        try:
            with engine:
                assert engine.encrypt_poly(key, [0], r=[0]) == (0,) * 701
                raise HostFailure()
        except HostFailure:
            pass
        with engine:
            assert engine.encrypt_poly(key, [1], r=[0]) == (1,) + (0,) * 700
        report.update(
            passed=True,
            hardware_tested=True,
            stage="complete",
            messages=len(messages),
            cross_decryption=True,
            session_opens=3,
            cleanup_after_exception=True,
        )
    except Exception as exc:
        report["error"] = repr(exc)
        raise
    finally:
        for adapter in adapters:
            adapter.close()
        (target / "report.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
