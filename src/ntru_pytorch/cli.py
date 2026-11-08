import argparse
import json
import logging
import platform

from . import PRESETS, RandomSource


def main():
    parser = argparse.ArgumentParser(description="NTRU research tools")
    parser.add_argument(
        "command",
        choices=[
            "oqs-info",
            "oqs-check",
            "oqs-kem-benchmark",
            "oqs-arith-benchmark",
            "demo",
            "trace",
            "benchmark",
            "hailo-prepare",
            "hailo-check",
            "hailo-demo",
            "hailo-benchmark",
            "hailo-build",
            "hailo-admit",
        ],
    )
    parser.add_argument("--preset", choices=PRESETS, default="moderate")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--runs", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--output")
    parser.add_argument("--artifact", default="results/hailo")
    parser.add_argument("--public-key")
    parser.add_argument("--private-key")
    parser.add_argument("--cases", type=int, default=1000)
    parser.add_argument("--hardware", action="store_true")
    parser.add_argument("--packing", choices=["position16-v1", "active16-v1"],
                        default="position16-v1")
    parser.add_argument("--compare-variants", action="store_true")
    parser.add_argument("--skip-sympy-timing", action="store_true")
    parser.add_argument("--torch-threads", type=int)
    parser.add_argument("--host-processing", choices=["reference", "optimized"], default="reference")
    parser.add_argument("--include-sparse-cpu", action="store_true")
    parser.add_argument("--report")
    parser.add_argument("--build-dir", default="results/hailo-build")
    parser.add_argument("--admit-output", default="results/hailo-admitted")
    parser.add_argument("--distro", default="Ubuntu-22.04")
    parser.add_argument("--linux-python", default="/opt/ntru-dfc/bin/python")
    parser.add_argument(
        "--quant-profile", choices=["unit-v1", "binary-circulant-affine-v1"], default="unit-v1"
    )
    parser.add_argument("--calibration-samples", type=int, default=1024)
    parser.add_argument("--oqs-root", default="results/native/oqs/0.16.0/" + platform.machine())
    parser.add_argument("--oqs-build")
    parser.add_argument("--oqs-profile", choices=["portable", "native"], default="portable")
    parser.add_argument("--qualification-root", default="results/oqs-qualification")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    if args.output is None:
        args.output = (
            "results/" + args.command if args.command.startswith("oqs-") else
            "results/hailo-benchmark" if args.command == "hailo-benchmark" else "results/benchmark"
        )
    if args.command.startswith("oqs-"):
        from .oqs.cli import run

        run(args)
        return
    if args.command.startswith("hailo-"):
        from .hailo.cli import run

        run(args)
        return
    if args.command == "benchmark":
        from .research import benchmark

        benchmark(
            PRESETS[args.preset], args.output, seed=args.seed, runs=args.runs, warmup=args.warmup
        )
        print("Saved JSON and CSV measurements")
        return
    from . import SympyNTRU, TorchNTRU, generate_keys

    public, private = generate_keys(PRESETS[args.preset], random_source=RandomSource(args.seed))
    engines = (SympyNTRU(), TorchNTRU())
    if args.command == "demo":
        data = "NTRU: \u041f\u0440\u0438\u0432\u0435\u0442 \U0001f600\0".encode("utf-8")
        ciphertexts = [
            e.encrypt_bytes(public, data, random_source=RandomSource(args.seed)) for e in engines
        ]
        assert ciphertexts[0] == ciphertexts[1]
        for e, c in zip(engines, ciphertexts, strict=True):
            for d in engines:
                assert d.decrypt_bytes(private, c) == data
                print(type(e).__name__ + " -> " + type(d).__name__ + ": OK")
    else:
        r = RandomSource(args.seed).ternary(public.params.N, public.params.d, public.params.d)
        traces = []
        for engine in engines:
            rows = []
            cipher = engine.encrypt_poly(
                public, [1, 0, 1], r=r, trace=lambda name, a, rows=rows: rows.append((name, a))
            )
            engine.decrypt_poly(
                private, cipher, trace=lambda name, a, rows=rows: rows.append((name, a))
            )
            traces.append(rows)
        assert traces[0] == traces[1]
        print(json.dumps(traces[0]))
