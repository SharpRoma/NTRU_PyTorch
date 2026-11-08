"""CPU-only fixed-key benchmark. No device is opened; this is not NPU evidence."""

import argparse
import csv
import hashlib
import json
import platform
import sys
from dataclasses import asdict
from pathlib import Path
from time import perf_counter_ns

import numpy as np

from ntru_pytorch import PrivateKey, PublicKey, RandomSource, SparseNTRU
from ntru_pytorch.array_api import encode_array
from ntru_pytorch.formats import load
from ntru_pytorch.models import Ciphertext
from ntru_pytorch.sparse_backend import KERNELS, _cyclic


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--public-key", required=True)
    parser.add_argument("--private-key", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--runs", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--include-torch", action="store_true")
    parser.add_argument("--torch-threads", type=int, default=6)
    args = parser.parse_args()
    if args.runs < 2 or args.warmup < 0 or args.torch_threads < 1:
        raise ValueError("Need runs >= 2, warmup >= 0 and positive torch threads")
    target = Path(args.output)
    if any(target.with_suffix(s).exists() for s in (".json", ".csv", ".progress.json")):
        raise ValueError("Preserve previous experiments")
    target.parent.mkdir(parents=True, exist_ok=True)

    def progress(stage, **details):
        target.with_suffix(".progress.json").write_text(
            json.dumps(dict(stage=stage, **details), indent=2)
        )

    progress("key_loading")
    start = perf_counter_ns()
    public, private = load(args.public_key), load(args.private_key)
    if (
        not isinstance(public, PublicKey)
        or not isinstance(private, PrivateKey)
        or public.params != private.params
    ):
        raise ValueError("Need matching public/private key parameters")
    p = public.params
    identity = np.zeros(p.N, dtype=np.int64)
    identity[0] = 1
    for inverse, modulus in ((private.fp, p.p), (private.fq, p.q)):
        if not np.array_equal(_cyclic(private.f, inverse) % modulus, identity):
            raise ValueError("Private key inverse mismatch")
    if not np.array_equal(
        (_cyclic(private.fq, private.g) % p.q) * p.p % p.q,
        np.asarray(public.h, dtype=np.int64) % p.q,
    ):
        raise ValueError("Public/private key mismatch")
    loaded = perf_counter_ns() - start
    engines = {f"sparse-{kernel}": SparseNTRU(kernel=kernel) for kernel in KERNELS}
    preparation = {name: e.prepare(public) for name, e in engines.items()}
    original_threads = None
    if args.include_torch:
        import torch

        from ntru_pytorch import TorchNTRU

        original_threads = torch.get_num_threads()
        torch.set_num_threads(args.torch_threads)
        engines["torch"] = TorchNTRU()
    metadata = dict(
        hardware_tested=False,
        params=asdict(p),
        seed=args.seed,
        runs=args.runs,
        warmup=args.warmup,
        platform=platform.platform(),
        python=platform.python_version(),
        numpy=np.__version__,
        key_load_and_validation_ns=loaded,
        cpu_key_preparation=preparation,
        torch_threads=args.torch_threads if args.include_torch else None,
        timing_boundary="encrypt_bytes with pre-generated explicit r; key prep, IO and correctness outside timer",
    )
    for field in ("public_key", "private_key"):
        with Path(getattr(args, field)).open("rb") as f:
            metadata[field + "_sha256"] = hashlib.file_digest(f, "sha256").hexdigest()
    rows = []
    try:
        for count in (1, 8, 32, 128):
            size = count * p.N // 8 - 8
            if size < 0:
                raise ValueError("This byte benchmark requires N >= 64")
            data = bytes(i % 256 for i in range(size))
            messages = encode_array(data, p.N)
            if len(messages) != count:
                raise ValueError("Incorrect polynomial block count")
            source = RandomSource(args.seed)
            rs = tuple(source.ternary(p.N, p.d, p.d) for _ in messages)
            expected = []
            # Independent dense convolution, never the sparse kernel under test.
            for r, m in zip(rs, messages, strict=True):
                product = _cyclic(r, public.h)
                residue = (product % p.q + m) % p.q
                expected.append(
                    tuple(map(int, np.where(residue >= (p.q + 1) // 2, residue - p.q, residue)))
                )
            reference = Ciphertext(p, tuple(expected))
            progress("correctness", blocks=count)
            for name, e in engines.items():
                actual = e.encrypt_bytes(public, data, r_polys=rs)
                if actual != reference:
                    np.savez_compressed(
                        target.with_suffix(".mismatch.npz"),
                        inputs=rs,
                        expected=reference.blocks,
                        actual=actual.blocks,
                    )
                    raise ValueError(
                        f"CPU ciphertext mismatch: {name}, blocks={count}, seed={args.seed}"
                    )
                if e.decrypt_bytes(private, actual) != data:
                    raise ValueError(f"CPU cross-decryption mismatch: {name}, blocks={count}")
            progress("warmup", blocks=count)
            for e in engines.values():
                for _ in range(args.warmup):
                    e.encrypt_bytes(public, data, r_polys=rs)
            names = list(engines)
            progress("measurement", blocks=count)
            for iteration in range(args.runs):
                order = names[iteration % len(names) :] + names[: iteration % len(names)]
                for name in order:
                    e = engines[name]
                    start = perf_counter_ns()
                    e.encrypt_bytes(public, data, r_polys=rs)
                    row = dict(
                        backend=name,
                        blocks=count,
                        iteration=iteration,
                        total_ns=perf_counter_ns() - start,
                        payload_bytes=len(data),
                        execution="cpu",
                    )
                    if isinstance(e, SparseNTRU):
                        row.update(e.last_timings)
                    rows.append(row)
    except Exception as exc:
        progress("failed", error=repr(exc), seed=args.seed)
        raise
    finally:
        if original_threads is not None:
            torch.set_num_threads(original_threads)
    summary = []
    for name in engines:
        for count in (1, 8, 32, 128):
            matching = [r for r in rows if r["backend"] == name and r["blocks"] == count]
            mid = float(np.median([r["total_ns"] for r in matching]))
            summary.append(
                dict(
                    backend=name,
                    blocks=count,
                    median_ns=mid,
                    p95_ns=float(np.percentile([r["total_ns"] for r in matching], 95)),
                    payload_bytes=matching[0]["payload_bytes"],
                    blocks_per_second=count * 1e9 / mid,
                    bytes_per_second=matching[0]["payload_bytes"] * 1e9 / mid,
                )
            )
    metadata["imported_backends"] = {
        name: name in sys.modules for name in ("torch", "sympy", "hailo_platform")
    }
    target.with_suffix(".json").write_text(
        json.dumps(dict(metadata=metadata, samples=rows, summary=summary), indent=2)
    )
    with target.with_suffix(".csv").open("w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=sorted(set().union(*(r.keys() for r in rows))) + ["metadata_json"]
        )
        writer.writeheader()
        for row in rows:
            writer.writerow({**row, "metadata_json": json.dumps(metadata)})
    progress("complete", hardware_tested=False, samples=len(rows))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
