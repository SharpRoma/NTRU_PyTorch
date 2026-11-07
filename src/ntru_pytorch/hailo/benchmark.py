"""Hardware-only measurements. Compilation is external and never hidden in timings."""

import csv
import json
from pathlib import Path
from time import perf_counter_ns

import numpy as np
import torch

from .. import RandomSource, SympyNTRU, TorchNTRU
from ..codec import encode
from .engine import HailoEncryptor
from .validation import environment, validate_hardware


def benchmark_hailo(directory, private_key, output, *, runs=20, warmup=3, seed=42):
    if runs < 2 or warmup < 0:
        raise ValueError("Need runs >= 2 and warmup >= 0")
    rows = []
    metadata = {
        "environment": environment(),
        "runs": runs,
        "warmup": warmup,
        "seed": seed,
        "torch_threads": torch.get_num_threads(),
        "hardware_internal_ns": None,
        "compilation_ns": None,
        "boundary": "host wall clock; inference includes PCIe transfers; no energy claim",
    }
    with HailoEncryptor(directory) as hailo:
        key = hailo.scheme.key
        metadata["artifact_preparation_ns"] = hailo.manifest.get("preparation_ns")
        metadata.update(
            key_sha256=hailo.manifest["key_sha256"],
            load_hef_ns=hailo.load_ns,
            hef_sha256=hailo.manifest["compiled"]["files"]["model.hef"],
            compiler_version=json.loads((Path(directory) / "compiler-report.json").read_text())[
                "compiler_version"
            ],
        )
        metadata["validation"] = validate_hardware(hailo, cases=1000, seed=seed)
        for count in (1, 8, 32, 128):
            size = (count * key.params.N) // 8 - 8
            if size < 0:
                raise ValueError("Benchmark requires N >= 64")
            message = bytes((i % 256 for i in range(size)))
            blocks = encode(message, key.params.N)
            if len(blocks) != count:
                raise ValueError("Payload does not fill the requested block count")
            source = RandomSource(seed)
            rs = tuple(source.ternary(key.params.N, key.params.d, key.params.d) for _ in blocks)
            engines = {"sympy": SympyNTRU(), "torch": TorchNTRU(), "hailo": hailo}
            results = {
                name: e.encrypt_bytes(key, message, r_polys=rs) for name, e in engines.items()
            }
            if not all(c == results["torch"] for c in results.values()):
                raise ValueError("Ciphertext mismatch before benchmark")
            for d in (SympyNTRU(), TorchNTRU()):
                if d.decrypt_bytes(private_key, results["hailo"]) != message:
                    raise ValueError("Cross-decryption failed before benchmark")
            for name, e in engines.items():
                for _ in range(warmup):
                    e.encrypt_bytes(key, message, r_polys=rs)
                for iteration in range(runs):
                    start = perf_counter_ns()
                    e.encrypt_bytes(key, message, r_polys=rs)
                    elapsed = perf_counter_ns() - start
                    row = {
                        "backend": name,
                        "blocks": count,
                        "iteration": iteration,
                        "total_ns": elapsed,
                    }
                    if name == "hailo":
                        row.update(e.last_timings)
                    rows.append(row)
    summaries = []
    for name in ("sympy", "torch", "hailo"):
        for count in (1, 8, 32, 128):
            values = [r["total_ns"] for r in rows if r["backend"] == name and r["blocks"] == count]
            median = float(np.median(values))
            summaries.append(
                {
                    "backend": name,
                    "blocks": count,
                    "median_ns": median,
                    "p95_ns": float(np.percentile(values, 95)),
                    "blocks_per_second": count * 1e9 / median,
                }
            )
    target = Path(output)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.with_suffix(".json").write_text(
        json.dumps({"metadata": metadata, "samples": rows, "summary": summaries}, indent=2),
        encoding="utf-8",
    )
    fields = sorted(set().union(*(row.keys() for row in rows)))
    with target.with_suffix(".csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields + ["metadata_json"])
        writer.writeheader()
        for row in rows:
            writer.writerow({**row, "metadata_json": json.dumps(metadata)})
    return summaries
