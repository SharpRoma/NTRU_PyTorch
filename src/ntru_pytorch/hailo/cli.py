"""Experimental commands; reports never conflate CPU simulation and NPU execution."""

import json
from pathlib import Path

from ..formats import load
from .artifacts import ArtifactError, load_artifact, prepare_hailo
from .engine import HailoEncryptor
from .validation import (
    environment,
    qualify_hardware,
    validate_hardware,
    validate_onnx,
    validate_software,
)


def run(args):
    root = Path(args.artifact)
    report = {"command": args.command, "environment": environment(), "hardware_tested": False}
    try:
        if args.command == "hailo-prepare":
            if not args.public_key:
                raise ValueError("--public-key is required")
            key = load(args.public_key)
            report["software"] = validate_software(
                key, cases=args.cases, seed=args.seed, packing=args.packing
            )
            report["manifest"] = prepare_hailo(key, root)
            report["onnx"] = validate_onnx(root)
        elif args.command == "hailo-build":
            from .build import build_hailo

            report["build"] = build_hailo(
                root,
                args.build_dir,
                distro=args.distro,
                linux_python=args.linux_python,
                profile=args.quant_profile,
                seed=args.seed,
                samples=args.calibration_samples,
            )
            if report["build"].get("status") != "ready-for-admission":
                raise ArtifactError(
                    report["build"].get("error", "Build is not ready for admission")
                )
        elif args.command == "hailo-admit":
            from .build import admit_build

            report["manifest"] = admit_build(root, args.build_dir, args.admit_output)
        elif args.command == "hailo-check":
            manifest, scheme = load_artifact(root)
            report["artifact_status"] = manifest["status"]
            report["software"] = validate_software(
                scheme.key, cases=args.cases, seed=args.seed, packing=args.packing
            )
            report["onnx"] = validate_onnx(root)
            report["hef_available"] = bool(manifest.get("compiled"))
            if args.hardware:
                with HailoEncryptor(root, packing=args.packing,
                                    host_processing=args.host_processing) as engine:
                    report["hardware"] = validate_hardware(engine, cases=args.cases, seed=args.seed)
                report["hardware_tested"] = True
        elif args.command == "hailo-demo":
            if not args.private_key:
                raise ValueError("--private-key is required for cross-decryption")
            private = load(args.private_key)
            report["qualification"] = qualify_hardware(
                root, private, cases=args.cases, seed=args.seed, packing=args.packing,
                host_processing=args.host_processing,
            )
            report.update(hardware_tested=True, cross_decryption="passed")
        else:
            if not args.private_key:
                raise ValueError("--private-key is required for benchmark validation")
            from .benchmark import benchmark_hailo

            report["summary"] = benchmark_hailo(
                root,
                load(args.private_key),
                args.output,
                runs=args.runs,
                warmup=args.warmup,
                seed=args.seed,
                packing=args.packing,
                compare_variants=args.compare_variants,
                include_sympy_timing=not args.skip_sympy_timing,
                torch_threads=args.torch_threads,
                host_processing=args.host_processing,
                include_sparse_cpu=args.include_sparse_cpu,
            )
            report["hardware_tested"] = True
        report["status"] = "passed"
    except Exception as exc:  # Preserve vendor SDK failures in the diagnostic report.
        report.update(status="failed", error=f"{type(exc).__name__}: {exc}")
    target = Path(args.report) if args.report else root / "last-check.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    if report["status"] != "passed":
        raise SystemExit(2)
