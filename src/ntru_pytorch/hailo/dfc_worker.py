"""Standalone Linux DFC 3.33 worker; does not import the host Torch package.

Raw SDK evidence is retained. Admission stays closed until an extraction adapter
accounts for the actual observed numeric graph; native weights are not evidence
of compiled integer semantics.
"""

import argparse
import hashlib
import json
import os
import time
import traceback
from importlib.metadata import version
from pathlib import Path


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    root = Path(args.output)
    cfg = json.loads((root / "build-config.json").read_text())
    report = {
        "status": "starting",
        "source": cfg["source"],
        "profile": cfg["profile"],
        "hardware_tested": False,
        "extraction_supported": False,
        "timings_ns": {},
        "execution_device": "cpu",
    }
    start = time.perf_counter_ns()
    try:
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
        import numpy as np
        from hailo_sdk_client import ClientRunner, InferenceContext

        if version("hailo-dataflow-compiler") != "3.33.0":
            raise RuntimeError("Expected Dataflow Compiler 3.33.0")
        report["compiler_version"] = "3.33.0"
        if digest(root / "model.onnx") != cfg["source"]["source_files"]["model.onnx"]:
            raise ValueError("ONNX checksum mismatch")
        runner = ClientRunner(hw_arch="hailo8l")
        stage = time.perf_counter_ns()
        runner.translate_onnx_model(
            str(root / "model.onnx"),
            net_name="ntru_binary",
            net_input_shapes={"r": [1, cfg["cin"], 1, 1]},
        )
        runner.save_har(str(root / "parsed.har"))
        report["timings_ns"]["parse"] = time.perf_counter_ns() - stage
        hn = runner.get_hn_dict()
        (root / "parsed-hn.json").write_text(json.dumps(hn, indent=2, default=lambda v: v.tolist()))
        script = "model_optimization_flavor(optimization_level=0, compression_level=0)\n"
        script += (
            "model_optimization_config(calibration, batch_size=1, "
            f"calibset_size={cfg['calibration_frames']})\n"
        )
        if cfg["profile"] in ("unit-v1", "binary-circulant-affine-v1"):
            layers = hn.get("layers", {})
            names = [
                name
                for name, layer in layers.items()
                if layer.get("type") in ("input_layer", "conv")
            ]
            if len(names) != 2:
                raise RuntimeError(
                    "Unsupported parsed topology: expected one input and one convolution"
                )
            for name in names:
                # Give the affine output headroom; binary host input codes stay 0/1.
                high = 32 if (
                    cfg["profile"] == "binary-circulant-affine-v1"
                    and layers[name]["type"] == "conv"
                ) else 255
                script += f"quantization_param([{name}], force_range_out=[0,{high}])\n"
        (root / "model.alls").write_text(script)
        runner.load_model_script(script)
        if (
            digest(root / "calibration.npy") != cfg["calibration_sha256"]
            or digest(root / "expected.npy") != cfg["expected_sha256"]
        ):
            raise ValueError("Calibration/reference checksums differ")
        data = np.load(root / "calibration.npy", allow_pickle=False)
        expected = np.load(root / "expected.npy", allow_pickle=False)
        with runner.infer_context(InferenceContext.SDK_NATIVE) as context:
            native = np.asarray(runner.infer(context, data, batch_size=1)).reshape(expected.shape)
        if not np.array_equal(native, expected):
            raise RuntimeError("DFC native graph differs from exact reference")
        np.save(root / "native-outputs.npy", native, allow_pickle=False)
        stage = time.perf_counter_ns()
        (root / "optimization").mkdir()
        runner.optimize(data, work_dir=str(root / "optimization"))
        runner.save_har(str(root / "quantized.har"))
        report["timings_ns"]["optimize"] = time.perf_counter_ns() - stage
        (root / "quantized-hn.json").write_text(
            json.dumps(runner.get_hn_dict(), indent=2, default=lambda v: v.tolist())
        )
        params = runner.get_params_translated()
        np.savez(
            root / "raw-translated-params.npz", **{k: np.asarray(v) for k, v in params.items()}
        )
        report["translated_parameter_keys"] = list(params.keys())
        with runner.infer_context(InferenceContext.SDK_BIT_EXACT) as context:
            numeric = np.asarray(runner.infer(context, data, batch_size=1)).reshape(expected.shape)
        np.save(root / "numeric-outputs.npy", numeric, allow_pickle=False)
        report["sample_max_error"] = float(np.max(np.abs(numeric - expected)))
        stage = time.perf_counter_ns()
        (root / "model.hef").write_bytes(runner.compile())
        runner.save_har(str(root / "compiled.har"))
        report["timings_ns"]["compile"] = time.perf_counter_ns() - stage
        report.update(
            status="compiled-unverified",
            hef_sha256=digest(root / "model.hef"),
            error="Raw translated SDK parameters and HN saved. Audited numeric extraction is required before admission; no fabricated effective weights or rounding bounds are emitted.",
        )
        if cfg["profile"] == "binary-circulant-affine-v1":
            from dfc_extract import extract_affine

            try:
                report["compiled"] = extract_affine(runner, root, cfg)
                report["compiled"]["compilation_ns"] = report["timings_ns"]["compile"]
                report.update(status="ready-for-admission", extraction_supported=True)
                report.pop("error", None)
            except Exception as exc:
                report.update(status="unsupported", error=f"{type(exc).__name__}: {exc}")
    except Exception as exc:
        report.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        (root / "failure-traceback.txt").write_text(traceback.format_exc())
    report["timings_ns"]["total"] = time.perf_counter_ns() - start
    (root / "build-report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0 if report["status"] == "ready-for-admission" else 2


if __name__ == "__main__":
    raise SystemExit(main())
