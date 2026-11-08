"""Hardware measurements with exact prechecks and optional packing/cache ablation."""

import csv
import json
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter_ns

import numpy as np
import torch

from .. import RandomSource, SympyNTRU, TorchNTRU
from ..codec import encode
from .engine import HailoEncryptor
from .scheme import PACKINGS
from .validation import environment, validate_hardware


def benchmark_hailo(directory, private_key, output, *, runs=20, warmup=3, seed=42,
                    packing="position16-v1", compare_variants=False,
                    include_sympy_timing=True, torch_threads=None, host_processing="reference",
                    include_sparse_cpu=False):
    if type(include_sympy_timing) is not bool:
        raise ValueError("include_sympy_timing must be boolean")
    if type(include_sparse_cpu) is not bool:
        raise ValueError("include_sparse_cpu must be boolean")
    if torch_threads is not None and (type(torch_threads) is not int or torch_threads < 1):
        raise ValueError("torch_threads must be a positive integer")
    previous_threads = torch.get_num_threads()
    try:
        if torch_threads is not None:
            torch.set_num_threads(torch_threads)
        return _benchmark_hailo(
            directory, private_key, output, runs=runs, warmup=warmup, seed=seed,
            packing=packing, compare_variants=compare_variants,
            include_sympy_timing=include_sympy_timing,
            host_processing=host_processing,
            include_sparse_cpu=include_sparse_cpu,
        )
    finally:
        if torch_threads is not None:
            torch.set_num_threads(previous_threads)


def _benchmark_hailo(directory, private_key, output, *, runs, warmup, seed,
                     packing, compare_variants, include_sympy_timing, host_processing,
                     include_sparse_cpu):
    from .host import HOST_PROCESSING

    if host_processing not in HOST_PROCESSING:
        raise ValueError("Unknown host processing")
    if runs < 2 or warmup < 0 or packing not in PACKINGS:
        raise ValueError("Need runs >= 2, warmup >= 0 and a known packing")
    target = Path(output)
    if any(target.with_suffix(s).exists() for s in (".json", ".csv", ".progress.json")):
        raise ValueError("Benchmark output already exists; preserve previous experiments")
    target.parent.mkdir(parents=True, exist_ok=True)

    def progress(stage, **details):
        target.with_suffix(".progress.json").write_text(json.dumps({
            "stage": stage, "updated_utc": datetime.now(timezone.utc).isoformat(), **details,
        }, indent=2), encoding="utf-8")
    variants = [(p, cache) for p in PACKINGS for cache in (False, True)] if compare_variants else [
        (packing, True)
    ]
    rows, validations, loads, startups = [], {}, {}, {}
    sparse_engines = {}
    if include_sparse_cpu:
        from .. import SparseNTRU
        from ..sparse_backend import KERNELS

        sparse_engines = {f"sparse-{kernel}": SparseNTRU(kernel=kernel) for kernel in KERNELS}
    metadata = {
        "environment": environment(), "runs": runs, "warmup": warmup, "seed": seed,
        "torch_threads": torch.get_num_threads(), "hardware_internal_ns": None,
        "compilation_ns": None, "compare_variants": compare_variants,
        "include_sympy_timing": include_sympy_timing,
        "host_processing": host_processing,
        "include_sparse_cpu": include_sparse_cpu,
        "cpu_key_preparation": {},
        "cpu_execution_note": "sparse kernels use NumPy INT64 reductions, no BLAS matmul; torch_threads applies to PyTorch",
        "timing_hierarchy": {
            "encrypt_total_ns": ["message_encode_ns", "key_normalization_ns", "prepare_ns",
                                 "inference_with_io_ns", "reconstruct_ns",
                                 "ciphertext_construction_ns", "unaccounted_ns"],
            "inference_with_io_ns": ["frame_validation_ns", "affine_encode_ns",
                                     "sdk_call_with_io_ns", "affine_decode_ns"],
            "reconstruct_ns": ["output_validation_ns", "frame_sum_ns",
                               "bit_reconstruction_ns", "reduction_ns"],
        },
        "payload_definition": "user message bytes; excludes encrypted length header and padding",
        "batch_checks": {},
        "boundary": "host wall clock; SDK call includes PCIe; no energy claim",
        "timing_note": "inference_with_io_ns includes affine encode/decode; sdk_call_with_io_ns isolates the SDK call",
        "variant_order": [f"{p}-{'cached' if c else 'uncached'}" for p, c in variants],
    }
    for variant_index, (mode, cache) in enumerate(variants):
        label = f"hailo-{mode}-{'cached' if cache else 'uncached'}" if compare_variants else "hailo"
        progress("artifact_validation", variant=label)
        cold_started = perf_counter_ns()
        engine = HailoEncryptor(directory, packing=mode, cache_affine_lut=cache,
                                host_processing=host_processing)
        engine.diagnostic_directory = target.parent
        validated_at = perf_counter_ns()
        progress("session_open", variant=label)
        open_started = perf_counter_ns()
        with engine as hailo:
            opened_at = perf_counter_ns()
            startups[label] = {
                "artifact_validation_ns": validated_at - cold_started,
                "session_open_ns": opened_at - open_started,
                # Exclude progress-file IO between the two measured stages.
                "cold_start_ns": validated_at - cold_started + opened_at - open_started,
            }
            key = hailo.scheme.key
            if variant_index == 0:
                for name, cpu in sparse_engines.items():
                    metadata["cpu_key_preparation"][name] = cpu.prepare(key)
            loads[label] = hailo.load_ns
            metadata.update(
                key_sha256=hailo.manifest["key_sha256"],
                hef_sha256=hailo.manifest["compiled"]["files"]["model.hef"],
                compiler_version=json.loads((Path(directory) / "compiler-report.json").read_text())[
                    "compiler_version"
                ],
                compilation_ns=hailo.manifest["compiled"].get("compilation_ns"),
                artifact_preparation_ns=hailo.manifest.get("preparation_ns"),
                params=asdict(key.params),
            )
            session = getattr(hailo, "session", None)
            if session is not None:
                metadata["device"] = session.identity
            progress("hardware_validation", variant=label)
            validations[label] = validate_hardware(hailo, cases=1000, seed=seed)
            for count in (1, 8, 32, 128):
                size = (count * key.params.N) // 8 - 8
                if size < 0:
                    raise ValueError("Benchmark requires N >= 64")
                message = bytes(i % 256 for i in range(size))
                blocks = encode(message, key.params.N)
                if len(blocks) != count:
                    raise ValueError("Payload does not fill the requested block count")
                source = RandomSource(seed)
                rs = tuple(source.ternary(key.params.N, key.params.d, key.params.d) for _ in blocks)
                references = {"sympy": SympyNTRU(), "torch": TorchNTRU(), **sparse_engines}
                progress("cross_validation", variant=label, blocks=count)
                actual = hailo.encrypt_bytes(key, message, r_polys=rs)
                expected_frames = (
                    sum(max(1, sum((r.count(sign) + 15) // 16 for sign in (1, -1))) for r in rs)
                    if mode == "active16-v1" else count * 2 * hailo.scheme.tiles
                )
                expected_io = {"frames": expected_frames,
                               "input_bytes": expected_frames * hailo.scheme.cin,
                               "output_bytes": expected_frames * hailo.scheme.cout}

                def check_io(expected_io=expected_io):
                    if any(hailo.last_timings.get(k) != v for k, v in expected_io.items()):
                        raise ValueError(f"Frame/UINT8 payload mismatch: expected={expected_io}, "
                                         f"actual={hailo.last_timings}")

                check_io()
                for reference_name, reference in references.items():
                    expected = reference.encrypt_bytes(key, message, r_polys=rs)
                    if expected != actual:
                        np.savez_compressed(target.parent / f"ciphertext-mismatch-{count}.npz",
                                            inputs=rs, message=np.frombuffer(message, dtype=np.uint8),
                                            expected=expected.blocks, actual=actual.blocks)
                        raise ValueError(f"Ciphertext mismatch: seed={seed}, blocks={count}, "
                                         f"reference={reference_name}")
                    recovered = reference.decrypt_bytes(private_key, actual)
                    if recovered != message:
                        np.savez_compressed(target.parent / f"decryption-mismatch-{count}.npz",
                                            inputs=rs, ciphertext=actual.blocks,
                                            expected=np.frombuffer(message, dtype=np.uint8),
                                            actual=np.frombuffer(recovered, dtype=np.uint8))
                        raise ValueError(f"Cross-decryption failed: seed={seed}, blocks={count}, "
                                         f"reference={reference_name}")
                metadata["batch_checks"][f"{label}:{count}"] = {
                    **expected_io, "payload_bytes": len(message), "exact": True,
                }
                timed_references = references if include_sympy_timing else {
                    name: cpu for name, cpu in references.items() if name != "sympy"
                }
                engines = {**timed_references, label: hailo} if variant_index == 0 else {label: hailo}
                progress("warmup", variant=label, blocks=count)
                for engine in engines.values():
                    for _ in range(warmup):
                        engine.encrypt_bytes(key, message, r_polys=rs)
                # Rotate CPU/device measurement order to reduce within-variant order bias.
                names = list(engines)
                progress("measurement", variant=label, blocks=count)
                for iteration in range(runs):
                    order = names[iteration % len(names):] + names[:iteration % len(names)]
                    for name in order:
                        engine = engines[name]
                        start = perf_counter_ns()
                        engine.encrypt_bytes(key, message, r_polys=rs)
                        row = {"backend": name, "blocks": count, "iteration": iteration,
                               "total_ns": perf_counter_ns() - start,
                               "payload_bytes": len(message),
                               "execution": "npu" if name == label else "cpu"}
                        if name == label:
                            check_io()
                            row.update(engine.last_timings, packing=mode, cache_affine_lut=cache,
                                       host_processing=host_processing)
                        elif name in sparse_engines:
                            row.update(engine.last_timings, kernel=engine.kernel)
                        rows.append(row)
    metadata.update(validation=validations.get("hailo", validations), load_hef_ns=loads.get("hailo"),
                    variant_load_hef_ns=loads, variant_startup_ns=startups)
    metadata.update(startups.get("hailo", {}))
    summaries = []
    for name in dict.fromkeys(row["backend"] for row in rows):
        for count in (1, 8, 32, 128):
            values = [r["total_ns"] for r in rows if r["backend"] == name and r["blocks"] == count]
            median = float(np.median(values))
            summaries.append({"backend": name, "blocks": count, "median_ns": median,
                              "p95_ns": float(np.percentile(values, 95)),
                              "blocks_per_second": count * 1e9 / median,
                              "payload_bytes": next(r["payload_bytes"] for r in rows
                                                    if r["backend"] == name and r["blocks"] == count),
                              "bytes_per_second": next(r["payload_bytes"] for r in rows
                                                       if r["backend"] == name and r["blocks"] == count)
                                                  * 1e9 / median})
    target.parent.mkdir(parents=True, exist_ok=True)
    target.with_suffix(".json").write_text(
        json.dumps({"metadata": metadata, "samples": rows, "summary": summaries}, indent=2),
        encoding="utf-8",
    )
    fields = sorted(set().union(*(row.keys() for row in rows)))
    with target.with_suffix(".csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields + ["metadata_json"])
        writer.writeheader()
        for row in rows:
            writer.writerow({**row, "metadata_json": json.dumps(metadata)})
    progress("complete", hardware_tested=True, samples=len(rows))
    return summaries
