"""Lean runtime smoke under an externally imposed cgroup memory/swap limit."""

import argparse
import json
import platform
import resource
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--oqs-root", required=True)
    parser.add_argument("--artifact", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    target = Path(args.output)
    if target.exists():
        raise ValueError("Preserve previous memory reports")
    target.parent.mkdir(parents=True, exist_ok=True)
    report = dict(
        passed=False,
        hardware_tested=False,
        hardware_qualification=False,
        architecture=platform.machine(),
        phases=[],
    )

    def checkpoint(stage):
        group = next(
            line.split("::", 1)[1]
            for line in Path("/proc/self/cgroup").read_text().splitlines()
            if line.startswith("0::")
        )
        base = Path("/sys/fs/cgroup") / group.lstrip("/")
        report["phases"].append(
            dict(
                stage=stage,
                peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
                cgroup={
                    name: (base / name).read_text().strip()
                    for name in ["memory.max", "memory.swap.max", "memory.peak", "memory.events"]
                },
            )
        )
        report["imported_backends"] = {
            name: name in sys.modules for name in ("torch", "sympy", "hailo_platform")
        }
        target.write_text(json.dumps(report, indent=2) + "\n")

    checkpoint("python")
    try:
        from ntru_pytorch.hailo.engine import HailoEncryptor
        from ntru_pytorch.oqs import OqsArithmeticEncryptor, OqsKem
        from ntru_pytorch.randomness import RandomSource

        engine = HailoEncryptor(args.artifact, packing="active16-v1", host_processing="optimized")
        checkpoint("artifact_verified")
        key = engine.scheme.key
        with OqsKem(Path(args.oqs_root) / "portable") as kem:
            pk, sk = kem.keypair()
            ct, ss = kem.encapsulate(pk)
            assert kem.decapsulate(sk, ct) == ss
        source = RandomSource(42)
        rs = tuple(source.ternary(701, key.params.d, key.params.d) for _ in range(128))
        data = bytes(128 * 701 // 8 - 8)
        with OqsArithmeticEncryptor(Path(args.oqs_root) / "portable") as arith:
            reference = arith.encrypt_bytes(key, data, r_polys=rs)
        checkpoint("oqs_128_blocks")
        with engine:
            actual = engine.encrypt_bytes(key, data, r_polys=rs)
            assert actual == reference
            report.update(
                hardware_tested=True,
                device=engine.session.identity,
                frames=engine.last_timings["frames"],
                blocks=128,
            )
        assert "torch" not in sys.modules and "sympy" not in sys.modules
        checkpoint("hardware_128_blocks")
        final = report["phases"][-1]["cgroup"]
        assert final["memory.max"] == "805306368" and final["memory.swap.max"] == "0"
        assert "oom_kill 0" in final["memory.events"]
        report["passed"] = True
    except Exception as exc:
        report["error"] = repr(exc)
        raise
    finally:
        target.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
