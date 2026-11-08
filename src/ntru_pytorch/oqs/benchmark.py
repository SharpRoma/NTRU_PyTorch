"""Process-isolated experiments; KEM and project encryption have separate reports."""

import csv
import ctypes as C
import json
import platform
import resource
import subprocess
import sys
from pathlib import Path
from time import perf_counter_ns

from .loader import digest


def save(path, data):
    Path(path).write_text(json.dumps(data, indent=2) + "\n")


def stats(values):
    import numpy as np

    return dict(median_ns=float(np.median(values)), p95_ns=float(np.percentile(values, 95)))


def metadata():
    return dict(
        platform=platform.platform(),
        python=platform.python_version(),
        peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
    )


def source_hashes():
    base = Path(__file__).resolve().parents[1]
    return {str(p.relative_to(base)): digest(p) for p in sorted(base.rglob("*.py"))}


def worker(config):
    config = Path(config)
    args = json.loads(config.read_text())
    target = Path(args["result"])
    try:
        result = kem_worker(args) if args["kind"] == "kem" else arith_worker(args)
        result.update(metadata(), source_sha256=source_hashes())
        save(target, result)
    except Exception as exc:
        save(target, dict(passed=False, error=repr(exc), configuration=args, **metadata()))
        raise


def kem_worker(args):
    from .kem import OqsKem, buffer
    from .qualification import KAT_ALL

    build = Path(args["build"])
    corpus_path = Path(args["corpus"])
    # Independently verify the expected KAT output before using the extracted corpus.
    if digest(corpus_path.parent / "kat-all.rsp") != KAT_ALL:
        raise ValueError("KAT corpus provenance mismatch")
    corpus = json.loads(corpus_path.read_text())
    rows = []
    with OqsKem(build) as kem:
        for vector in corpus:
            if (
                kem.decapsulate(bytes.fromhex(vector["sk"]), bytes.fromhex(vector["ct"]))
                .hex()
                .upper()
                != vector["ss"].upper()
            ):
                raise AssertionError("Invalid KAT corpus")
        for api in ("C", "Python"):
            for op, name in enumerate(("keypair", "encapsulate", "decapsulate")):
                iterations = 1
                for index in range(-args["warmup"], args["runs"]):
                    vector = corpus[(index + args["warmup"]) % len(corpus)]
                    pk0, sk0, ct0, ss0 = (
                        bytes.fromhex(vector[k]) for k in ("pk", "sk", "ct", "ss")
                    )
                    while True:
                        pk, sk, ct, ss = (
                            buffer(value, len(value)) for value in (pk0, sk0, ct0, ss0)
                        )
                        elapsed, checksum = C.c_uint64(), C.c_uint64()
                        if api == "C":
                            rc = kem.lib.nr_kem_window(
                                kem._handle,
                                op,
                                iterations,
                                pk,
                                sk,
                                ct,
                                ss,
                                C.byref(elapsed),
                                C.byref(checksum),
                            )
                            if rc:
                                raise RuntimeError("Native KEM window failed")
                            ns = elapsed.value
                            result = (
                                (pk.raw, sk.raw)
                                if op == 0
                                else ((ct.raw, ss.raw) if op == 1 else ss.raw)
                            )
                        else:
                            start = perf_counter_ns()
                            for _ in range(iterations):
                                if op == 0:
                                    result = kem.keypair()
                                elif op == 1:
                                    result = kem.encapsulate(pk0)
                                else:
                                    result = kem.decapsulate(sk0, ct0)
                            ns = perf_counter_ns() - start
                        # Check outputs outside the window, including calibration windows.
                        if op == 0:
                            checkct, checkss = kem.encapsulate(result[0])
                            assert kem.decapsulate(result[1], checkct) == checkss
                        elif op == 1:
                            assert kem.decapsulate(sk0, result[0]) == result[1]
                        else:
                            assert result == ss0
                        kem.lib.nr_cleanse(sk, len(sk))
                        kem.lib.nr_cleanse(ss, len(ss))
                        if ns >= args["window_ns"]:
                            break
                        iterations = max(
                            iterations + 1, int(iterations * args["window_ns"] / max(ns, 1) * 1.1)
                        )
                    if index >= 0:
                        rows.append(
                            dict(
                                api=api,
                                operation=name,
                                index=index,
                                iterations=iterations,
                                window_ns=ns,
                                ns_per_operation=ns / iterations,
                                operations_per_second=iterations * 1e9 / ns,
                                checksum=checksum.value if api == "C" else None,
                            )
                        )
    summary = {}
    for api in ("C", "Python"):
        for op in ("keypair", "encapsulate", "decapsulate"):
            summary[api + "/" + op] = stats(
                [r["ns_per_operation"] for r in rows if r["api"] == api and r["operation"] == op]
            )
    return dict(
        passed=True,
        kind="HRSS-701 KEM",
        hardware_tested=False,
        randomness="default liboqs system RNG; deterministic RNG only in upstream KAT executable",
        build=str(build),
        build_manifest_sha256=digest(build / "manifest.json"),
        kat_corpus_sha256=digest(corpus_path),
        configuration=kem.manifest["configuration"],
        buffer_bytes=dict(public_key=1138, secret_key=1450, ciphertext=1138, shared_secret=32),
        measurements=rows,
        summary=summary,
    )


def prepare_corpus(target, public_path, private_path, seed, runs, warmup):
    import numpy as np

    from ..array_api import encode_array
    from ..formats import load
    from ..randomness import RandomSource
    from ..sparse_backend import _cyclic

    public, private = load(public_path), load(private_path)
    n = public.params.N
    if (n, public.params.p, public.params.q) != (701, 3, 8192):
        raise ValueError("N701 profile required")
    identity = np.zeros(n, dtype=np.int64)
    identity[0] = 1
    for inv, mod in ((private.fp, 3), (private.fq, 8192)):
        assert np.array_equal(_cyclic(private.f, inv) % mod, identity)
    assert np.array_equal(_cyclic(private.fq, private.g) * 3 % 8192, np.asarray(public.h) % 8192)
    rng = np.random.default_rng(seed)
    source = RandomSource(seed)
    arrays = {}
    for batch in (1, 8, 32, 128):
        length = batch * n // 8 - 8
        data = rng.integers(0, 256, (runs + warmup, length), dtype=np.uint8)
        rs = np.asarray(
            [
                [source.ternary(n, public.params.d, public.params.d) for _ in range(batch)]
                for _ in range(runs + warmup)
            ],
            dtype=np.int8,
        )
        expected = []
        for payload, polys in zip(data, rs, strict=True):
            messages = encode_array(payload.tobytes(), n)
            assert len(messages) == batch
            product = np.asarray([_cyclic(public.h, tuple(map(int, r))) for r in polys])
            out = (product + messages) % 8192
            out = np.where(out >= 4096, out - 8192, out)
            expected.append(out)
        arrays.update(
            {
                f"data{batch}": data,
                f"r{batch}": rs,
                f"expected{batch}": np.asarray(expected, dtype=np.int64),
            }
        )
    path = target / "corpus.npz"
    np.savez(path, **arrays)
    save(
        target / "corpus.json",
        dict(
            seed=seed,
            runs=runs,
            warmup=warmup,
            batches=[1, 8, 32, 128],
            sha256=digest(path),
            public_key_sha256=digest(public_path),
            private_key_sha256=digest(private_path),
            oracle="independent INT64 np.convolve cyclic reduction",
            public_key=str(Path(public_path).resolve()),
        ),
    )
    # A sample of every measured batch size is also checked with the independent CPU backends.
    from .. import SympyNTRU, TorchNTRU
    from ..models import Ciphertext

    cross = []
    for batch in (1, 8, 32, 128):
        payload = arrays[f"data{batch}"][0].tobytes()
        rs = tuple(tuple(map(int, r)) for r in arrays[f"r{batch}"][0])
        reference = Ciphertext(public.params, tuple(map(tuple, arrays[f"expected{batch}"][0])))
        for cpu in (SympyNTRU(), TorchNTRU()):
            assert cpu.encrypt_bytes(public, payload, r_polys=rs) == reference
            assert cpu.decrypt_bytes(private, reference) == payload
        cross.append(batch)
        save(
            target / "corpus-validation.json",
            dict(
                passed=True,
                checked_batches=cross,
                hardware_tested=False,
                backends=["SymPy", "PyTorch"],
            ),
        )
    return path


def arith_worker(args):
    import numpy as np

    from ..formats import load

    public = load(args["public_key"])
    backend = args["backend"]
    if digest(args["corpus"]) != args["corpus_sha256"]:
        raise ValueError("Corpus checksum mismatch")
    corpus = np.load(args["corpus"], allow_pickle=False)
    begin = perf_counter_ns()
    if backend == "torch":
        import torch

        from .. import TorchNTRU

        torch.set_num_threads(6)
        engine = TorchNTRU()
    elif backend.startswith("sparse-"):
        from ..sparse_backend import SparseNTRU

        engine = SparseNTRU(kernel=backend.removeprefix("sparse-"))
    elif backend.startswith("oqs-"):
        from .arithmetic import OqsArithmeticEncryptor

        engine = OqsArithmeticEncryptor(args["build"])
    elif backend == "hailo":
        from ..hailo.engine import HailoEncryptor

        engine = HailoEncryptor(
            args["artifact"], packing="active16-v1", host_processing="optimized"
        )
        engine.__enter__()
    else:
        raise ValueError("Unknown backend")
    opened = perf_counter_ns()
    preparation = engine.prepare(public) if hasattr(engine, "prepare") else {}
    ready = perf_counter_ns()
    rows = []
    native_rows = []
    report = dict(
        backend=backend,
        hardware_tested=backend == "hailo",
        preparation=preparation,
        cold_open_ns=opened - begin,
        prepare_call_ns=ready - opened,
        torch_threads=6 if backend == "torch" else None,
        configuration=args,
    )
    if backend.startswith("oqs-"):
        report["build_manifest_sha256"] = digest(Path(args["build"]) / "manifest.json")
        report["native_configuration"] = engine.manifest["configuration"]
    try:
        if backend == "hailo":
            from ..hailo.scheme import key_hash
            from ..hailo.validation import environment, validate_hardware

            report["hardware_qualification"] = validate_hardware(
                engine, cases=1000, seed=args["seed"]
            )
            report.update(
                environment=environment(),
                key_sha256=key_hash(public),
                device=engine.session.identity,
            )
            artifact = Path(args["artifact"])
            report["artifact_hashes"] = {
                p.name: digest(p) for p in artifact.iterdir() if p.suffix in (".hef", ".json")
            }
        for batch in (1, 8, 32, 128):
            for index in range(args["warmup"] + args["runs"]):
                data = corpus[f"data{batch}"][index].tobytes()
                rs = tuple(tuple(map(int, r)) for r in corpus[f"r{batch}"][index])
                start = perf_counter_ns()
                result = engine.encrypt_bytes(public, data, r_polys=rs)
                elapsed = perf_counter_ns() - start
                phases = dict(getattr(engine, "last_timings", {}))
                expected = corpus[f"expected{batch}"][index]
                if not np.array_equal(np.asarray(result.blocks), expected):
                    np.savez(
                        Path(args["result"]).with_suffix(".mismatch.npz"),
                        actual=np.asarray(result.blocks),
                        expected=expected,
                        r=rs,
                        data=np.frombuffer(data, dtype=np.uint8),
                    )
                    raise AssertionError((backend, batch, index))
                frames = batch * 8 if backend == "hailo" else None
                if backend == "hailo":
                    assert phases["frames"] == frames, phases
                    assert phases["input_bytes"] == frames * 704
                    assert phases["output_bytes"] == frames * 9120
                if index >= args["warmup"]:
                    rows.append(
                        dict(
                            batch=batch,
                            index=index - args["warmup"],
                            encrypt_total_ns=elapsed,
                            payload_bytes=len(data),
                            frames=frames,
                            input_bytes=frames * 704 if frames else None,
                            output_bytes=frames * 9120 if frames else None,
                            phases=phases,
                        )
                    )
            if backend.startswith("oqs-"):
                # Native-only boundary: prepacked Rq inputs; includes output masking/checksum.
                inputs = np.ascontiguousarray(
                    corpus[f"r{batch}"][0].astype(np.int64) % 8192, dtype=np.uint16
                )
                outputs = np.empty_like(inputs)
                iterations = 1
                for index in range(-args["warmup"], args["runs"]):
                    while True:
                        ns, checksum = C.c_uint64(), C.c_uint64()
                        rc = engine.lib.nr_rq_window(
                            engine._handle,
                            inputs.ctypes.data,
                            inputs.size,
                            outputs.ctypes.data,
                            batch,
                            iterations,
                            C.byref(ns),
                            C.byref(checksum),
                        )
                        if rc:
                            raise RuntimeError("Native arithmetic window failed")
                        from ..sparse_backend import _cyclic

                        assert np.array_equal(
                            outputs,
                            np.asarray(
                                [
                                    _cyclic(public.h, tuple(map(int, r))) % 8192
                                    for r in corpus[f"r{batch}"][0]
                                ]
                            ),
                        )
                        if ns.value >= 100_000_000:
                            break
                        iterations = max(
                            iterations + 1, int(iterations * 110_000_000 / max(ns.value, 1))
                        )
                    if index >= 0:
                        native_rows.append(
                            dict(
                                batch=batch,
                                index=index,
                                iterations=iterations,
                                window_ns=ns.value,
                                ns_per_batch=ns.value / iterations,
                                checksum=checksum.value,
                            )
                        )
    finally:
        if backend == "hailo":
            engine.__exit__(None, None, None)
        elif hasattr(engine, "close"):
            engine.close()
        corpus.close()
    report.update(
        passed=True,
        measurements=rows,
        native_microbenchmark=native_rows,
        summary={
            str(b): dict(
                **stats([r["encrypt_total_ns"] for r in rows if r["batch"] == b]),
                median_blocks_per_second=b
                * 1e9
                / stats([r["encrypt_total_ns"] for r in rows if r["batch"] == b])["median_ns"],
                median_payload_bytes_per_second=(b * 701 // 8 - 8)
                * 1e9
                / stats([r["encrypt_total_ns"] for r in rows if r["batch"] == b])["median_ns"],
            )
            for b in (1, 8, 32, 128)
        },
    )
    return report


def export_auxiliary_tables(target, configurations, reports, summaries):
    """Export native windows and phase/summary tables without rerunning measurements."""

    def write(name, rows):
        if not rows:
            return
        keys = sorted({k for row in rows for k in row})
        with (Path(target) / name).open("w", newline="") as f:
            writer = csv.DictWriter(f, keys)
            writer.writeheader()
            writer.writerows(rows)

    native, phases, startup = [], [], []
    for index, (conf, report) in enumerate(zip(configurations, reports, strict=True)):
        identity = dict(
            worker=index, round=conf["round"], backend=conf.get("backend", conf.get("profile"))
        )
        native.extend(dict(**identity, **row) for row in report.get("native_microbenchmark", []))
        for row in report["measurements"]:
            phases.extend(
                dict(**identity, batch=row.get("batch"), index=row["index"], metric=k, value=v)
                for k, v in row.get("phases", {}).items()
            )
        if "cold_open_ns" in report:
            startup.append(
                dict(
                    **identity,
                    cold_open_ns=report["cold_open_ns"],
                    prepare_call_ns=report["prepare_call_ns"],
                )
            )
    flattened = []
    for entry in summaries:
        row = {k: v for k, v in entry.items() if k not in ("pooled", "by_run")}
        row.update(entry["pooled"])
        for run, values in entry["by_run"].items():
            row.update({"run_" + run + "_" + k: v for k, v in values.items()})
        flattened.append(row)
    write("native-microbenchmark.csv", native)
    write("phases.csv", phases)
    write("cold-start.csv", startup)
    write("summary.csv", flattened)


def orchestrate(args, kind):
    target = Path(args.output).resolve()
    target.mkdir(parents=True, exist_ok=False)
    root = Path(args.oqs_root).resolve()
    profiles = ["portable", "native"]
    config = dict(kind=kind, runs=args.runs, warmup=args.warmup, seed=args.seed)
    if args.runs < 2 or args.warmup < 0:
        raise ValueError("Invalid run/warmup count")
    workers = []
    if kind == "kem":
        for round_ in range(3):
            for profile in profiles[:: 1 if round_ % 2 == 0 else -1]:
                build = root / profile
                checkdir = Path(args.qualification_root) / ("check-" + profile)
                workers.append(
                    dict(
                        **config,
                        round=round_,
                        profile=profile,
                        build=str(build),
                        corpus=str((checkdir / "kat-corpus.json").resolve()),
                        window_ns=100_000_000,
                    )
                )
    else:
        corpus = prepare_corpus(
            target, args.public_key, args.private_key, args.seed, args.runs, args.warmup
        )
        names = [
            "torch",
            "sparse-shift-sum",
            "sparse-cached-shifts",
            "oqs-portable",
            "oqs-native",
        ] + (["hailo"] if args.hardware else [])
        for round_ in range(3):
            order = names[round_:] + names[:round_]
            if round_ % 2:
                order = order[::-1]
            for name in order:
                workers.append(
                    dict(
                        **config,
                        round=round_,
                        backend=name,
                        build=str(root / name.removeprefix("oqs-")),
                        corpus=str(corpus),
                        corpus_sha256=digest(corpus),
                        public_key=str(Path(args.public_key).resolve()),
                        artifact=str(Path(args.artifact).resolve()),
                    )
                )
    save(
        target / "experiment.json",
        dict(
            kind=kind,
            workers=workers,
            source_sha256=source_hashes(),
            timing_boundary="full encrypt_bytes or independent KEM operations; generation of arithmetic corpus and validation outside timers",
        ),
    )
    allrows = []
    reports = []
    for index, conf in enumerate(workers):
        name = conf.get("backend", conf.get("profile"))
        prefix = f"{index:02d}-{name}-r{conf['round']}"
        conf["result"] = str(target / (prefix + ".json"))
        cfg = target / (prefix + ".config.json")
        save(cfg, conf)
        save(
            target / "progress.json",
            dict(stage="worker", index=index, total=len(workers), name=name),
        )
        try:
            with (target / (prefix + ".log")).open("w") as log:
                subprocess.run(
                    [sys.executable, "-m", "ntru_pytorch.oqs.benchmark", str(cfg)],
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    check=True,
                )
        except Exception as exc:
            save(
                target / "progress.json",
                dict(stage="failed", index=index, name=name, error=repr(exc)),
            )
            save(
                target / "report.json",
                dict(
                    passed=False,
                    kind=kind,
                    completed_workers=reports,
                    failed_worker=conf,
                    error=repr(exc),
                ),
            )
            raise
        report = json.loads(Path(conf["result"]).read_text())
        reports.append(report)
        for row in report["measurements"]:
            allrows.append(dict(worker=index, round=conf["round"], backend=name, **row))
    save(
        target / "report.json",
        dict(
            passed=True,
            kind=kind,
            hardware_tested=kind == "arith" and args.hardware,
            workers=reports,
        ),
    )
    grouped = {}
    for row in allrows:
        key = (
            row["backend"],
            str(row.get("batch", "")),
            row.get("api", ""),
            row.get("operation", ""),
        )
        grouped.setdefault(key, []).append(row)
    summaries = []
    for (backend, batch, api, operation), entries in grouped.items():
        metric = "encrypt_total_ns" if kind == "arith" else "ns_per_operation"
        by_run = {str(r): stats([e[metric] for e in entries if e["round"] == r]) for r in range(3)}
        entry = dict(
            backend=backend,
            batch=int(batch) if batch else None,
            api=api,
            operation=operation,
            pooled=stats([e[metric] for e in entries]),
            by_run=by_run,
        )
        if batch:
            entry.update(
                payload_bytes=entries[0]["payload_bytes"],
                blocks_per_second=int(batch) * 1e9 / entry["pooled"]["median_ns"],
                payload_bytes_per_second=entries[0]["payload_bytes"]
                * 1e9
                / entry["pooled"]["median_ns"],
            )
        summaries.append(entry)
    save(target / "summary.json", summaries)
    keys = sorted({k for row in allrows for k in row})
    with (target / "measurements.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, keys)
        writer.writeheader()
        for row in allrows:
            writer.writerow(
                {
                    k: json.dumps(v, sort_keys=True) if isinstance(v, dict) else v
                    for k, v in row.items()
                }
            )
    export_auxiliary_tables(target, workers, reports, summaries)
    save(target / "progress.json", dict(stage="complete", workers=len(workers)))
    print(target / "report.json")


if __name__ == "__main__":
    worker(sys.argv[1])
