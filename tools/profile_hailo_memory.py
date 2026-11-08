"""Linux per-process RSS/startup diagnostic; optional real-device zero-input smoke test."""

import argparse
import json
import platform
import resource
import sys
from pathlib import Path
from time import perf_counter_ns


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--open-device", action="store_true")
    parser.add_argument(
        "--blocks",
        type=int,
        default=0,
        help="Also encrypt this many normal-r blocks; requires --open-device",
    )
    args = parser.parse_args()
    if args.blocks < 0 or (args.blocks and not args.open_device):
        raise ValueError("--blocks must be nonnegative and requires --open-device")
    if platform.system() != "Linux":
        raise ValueError("RSS units and /proc collection require Linux")
    target = Path(args.output)
    if target.exists():
        raise ValueError("Preserve previous diagnostics")
    target.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "hardware_tested": False,
        "hardware_qualification": False,
        "phases": [],
    }
    previous = perf_counter_ns()

    def checkpoint(stage):
        nonlocal previous
        now = perf_counter_ns()
        status = Path("/proc/self/status").read_text().splitlines()
        rss = next(int(line.split()[1]) * 1024 for line in status if line.startswith("VmRSS:"))
        report["phases"].append(
            dict(
                stage=stage,
                elapsed_ns=now - previous,
                rss_bytes=rss,
                peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
            )
        )
        report["imported_backends"] = {
            name: name in sys.modules for name in ("torch", "sympy", "onnx", "hailo_platform")
        }
        cgroup = next(
            (
                line.split("::", 1)[1]
                for line in Path("/proc/self/cgroup").read_text().splitlines()
                if line.startswith("0::")
            ),
            None,
        )
        if cgroup is not None:
            base = Path("/sys/fs/cgroup") / cgroup.lstrip("/")
            report["cgroup"] = {"path": cgroup}
            for name in ("memory.max", "memory.swap.max", "memory.peak", "memory.events"):
                file = base / name
                if file.exists():
                    report["cgroup"][name] = file.read_text().strip()
        target.write_text(json.dumps(report, indent=2))
        previous = perf_counter_ns()

    checkpoint("python_start")
    try:
        from ntru_pytorch.hailo import HailoEncryptor, artifacts

        report["artifact_loader_source"] = artifacts.__file__
        checkpoint("imports")
        engine = HailoEncryptor(args.artifact, packing="active16-v1", host_processing="optimized")
        report.update(
            key_sha256=engine.manifest["key_sha256"],
            hef_sha256=engine.manifest["compiled"]["files"]["model.hef"],
        )
        checkpoint("artifact_validation")
        if args.open_device:
            with engine:
                report["device"] = engine.session.identity
                checkpoint("session_open")
                actual = engine.encrypt_poly(engine.scheme.key, [0], r=[0])
                if actual != (0,) * engine.scheme.n:
                    raise ValueError("Hardware zero-input smoke test mismatch")
                report["hardware_tested"] = True
                report["hardware_test_scope"] = "one zero polynomial; not full qualification"
                checkpoint("zero_inference")
                if args.blocks:
                    import numpy as np

                    from ntru_pytorch import RandomSource
                    from ntru_pytorch.codec import encode

                    key = engine.scheme.key
                    size = args.blocks * key.params.N // 8 - 8
                    if size < 0:
                        raise ValueError("Requested batch cannot contain the message header")
                    message = bytes(i % 256 for i in range(size))
                    messages = encode(message, key.params.N)
                    if len(messages) != args.blocks:
                        raise ValueError("Message does not fill requested block count")
                    source = RandomSource(42)
                    rs = tuple(
                        source.ternary(key.params.N, key.params.d, key.params.d) for _ in messages
                    )
                    checkpoint("batch_prepared")
                    ciphertext = engine.encrypt_bytes(key, message, r_polys=rs)
                    report["batch"] = dict(
                        blocks=args.blocks,
                        payload_bytes=len(message),
                        seed=42,
                        **engine.last_timings,
                    )
                    checkpoint("batch_encrypt")
                    h = np.asarray(key.h, dtype=np.int64) % key.params.q
                    expected = []
                    for polynomial, block in zip(rs, messages, strict=True):
                        linear = np.convolve(np.asarray(polynomial, dtype=np.int64), h)
                        product = linear[: key.params.N].copy()
                        product[:-1] += linear[key.params.N :]
                        e = (product + np.asarray(block, dtype=np.int64)) % key.params.q
                        e = np.where(e >= (key.params.q + 1) // 2, e - key.params.q, e)
                        expected.append(tuple(map(int, e)))
                    if ciphertext.blocks != tuple(expected):
                        raise ValueError(
                            "Hardware batch differs from independent INT64 convolution"
                        )
                    report["batch"]["exact_independent_check"] = True
                    report["hardware_test_scope"] = (
                        "zero polynomial and one normal-r batch; not full qualification"
                    )
                    checkpoint("batch_integer_check")
            checkpoint("session_closed")
        report["status"] = "passed"
    except Exception as exc:
        report.update(status="failed", error=repr(exc))
        raise
    finally:
        target.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
