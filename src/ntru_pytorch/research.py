import csv
import json
import platform
import statistics
import time
from dataclasses import asdict
from pathlib import Path
from tempfile import TemporaryDirectory

import sympy
import torch
from sympy import Poly

from . import RandomSource, SympyNTRU, TorchNTRU, generate_keys, tensor
from .formats import load, save
from .sympy_backend import multiply, poly, vector, x


def benchmark(params, output, *, seed=42, runs=20, warmup=3):
    if runs < 2 or warmup < 0:
        raise ValueError("Need runs >= 2 and warmup >= 0")
    pub, priv = generate_keys(params, random_source=RandomSource(seed))
    rs = RandomSource(seed)
    a = rs.ternary(params.N, params.d, params.d)
    b = pub.h
    ta, tb = tensor.tensor(a), tensor.tensor(b)
    expected = multiply(a, b, params.N)
    if tuple(tensor.reduce(tensor.linear(ta, tb), params.N).tolist()) != expected:
        raise ValueError("Polynomial implementations disagree")
    s, t = SympyNTRU(), TorchNTRU()
    message = b"research benchmark"
    cs = s.encrypt_bytes(pub, message, random_source=RandomSource(seed))
    ct = t.encrypt_bytes(pub, message, random_source=RandomSource(seed))
    if cs != ct or not all(
        engine.decrypt_bytes(priv, c) == message for engine in (s, t) for c in (cs, ct)
    ):
        raise ValueError("Cross-backend validation failed")
    sa, sb, ideal = poly(a), poly(b), Poly(x**params.N - 1, x)
    cases = {
        "sympy_polynomial": lambda: vector((sa * sb) % ideal, params.N),
        "torch_polynomial": lambda: tuple(tensor.reduce(tensor.linear(ta, tb), params.N).tolist()),
    }
    for name, engine in [("sympy", s), ("torch", t)]:
        cases[name + "_encrypt_memory"] = lambda engine=engine: engine.encrypt_bytes(
            pub, message, random_source=RandomSource(seed)
        )
        cases[name + "_decrypt_memory"] = lambda engine=engine: engine.decrypt_bytes(priv, cs)
    results = {}
    with TemporaryDirectory() as directory:
        path = Path(directory) / "cipher.json"
        save(cs, path)
        cases["json_read"] = lambda: load(path)
        cases["json_write"] = lambda: save(cs, path)
        for name, fn in cases.items():
            for _ in range(warmup):
                fn()
            samples = []
            for _ in range(runs):
                start = time.perf_counter_ns()
                fn()
                samples.append(time.perf_counter_ns() - start)
            quartiles = statistics.quantiles(samples, n=4, method="inclusive")
            results[name] = {
                "samples_ns": samples,
                "median_ns": statistics.median(samples),
                "q1_ns": quartiles[0],
                "q3_ns": quartiles[2],
            }
    metadata = {
        "params": asdict(params),
        "seed": seed,
        "runs": runs,
        "warmup": warmup,
        "threads": torch.get_num_threads(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "sympy": sympy.__version__,
        "platform": platform.platform(),
        "processor": platform.processor(),
        "comparison": "SymPy symbolic implementation versus PyTorch CPU INT64; polynomial inputs are prepared outside timing for both backends; both return Python integer tuples; memory timing includes codec and sampling, excludes key generation and file IO",
    }
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.with_suffix(".json").write_text(
        json.dumps({"metadata": metadata, "results": results}, indent=2), encoding="utf-8"
    )
    with output.with_suffix(".csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["case", "sample", "time_ns", "median_ns", "q1_ns", "q3_ns", "metadata_json"]
        )
        for name, data in results.items():
            for i, value in enumerate(data["samples_ns"]):
                writer.writerow(
                    [
                        name,
                        i,
                        value,
                        data["median_ns"],
                        data["q1_ns"],
                        data["q3_ns"],
                        json.dumps(metadata),
                    ]
                )
    return results
