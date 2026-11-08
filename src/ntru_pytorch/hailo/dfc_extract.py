"""Narrow DFC 3.33 extractor for the observed binary affine Conv1x1 graph.

Runs inside the DFC interpreter without importing the host package. Only the
observed a8_w8_a8, zero-bias, identity-input topology is supported. Host admission
independently checks the rational error certificate and actual HEF streams.
"""

import hashlib
import inspect
import json
from fractions import Fraction
from pathlib import Path

import numpy as np


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError("Unsupported DFC numeric graph: " + message)


def uniform(value):
    a = np.asarray(value)
    require(a.size > 0 and np.isfinite(a).all() and np.all(a == a.flat[0]), "nonuniform constant")
    return Fraction(float(a.flat[0]))


def extract_affine(runner, root, cfg):
    from hailo_sdk_client import InferenceContext

    root = Path(root)
    hn = runner.get_hn_dict()
    layers = hn["layers"]
    require(sorted(item["type"] for item in layers.values()) == ["conv", "input_layer", "output_layer"],
            "expected exactly input -> conv -> output")
    inp = next(n for n, item in layers.items() if item["type"] == "input_layer")
    conv = next(n for n, item in layers.items() if item["type"] == "conv")
    out = next(n for n, item in layers.items() if item["type"] == "output_layer")
    require(layers[conv]["input"] == [inp] and layers[conv]["output"] == [out]
            and layers[inp]["output"] == [conv] and layers[out]["input"] == [conv], "connections")
    p = layers[conv]["params"]
    for k, v in {"kernel_shape": [1, 1, cfg["cin"], cfg["cout"]],
                 "strides": [1, 1, 1, 1], "dilations": [1, 1, 1, 1], "padding": "VALID",
                 "groups": 1, "batch_norm": False, "elementwise_add": False,
                 "activation": "linear"}.items():
        require(p.get(k) == v, k)
    require(all(item["quantization_params"]["precision_mode"] == "a8_w8_a8"
                and not item["quantization_params"]["signed_output"] for item in layers.values()),
            "precision or signed output")
    params = runner.get_params_translated()
    np.savez(root / "compiled-translated-params.npz",
             **{k: np.asarray(v) for k, v in params.items()})

    def raw(layer, suffix):
        return np.asarray(params[f"{layer}/{suffix}:0"])

    # These checks exclude offsets and any unaccounted bias/padding transforms.
    for k in ("zp_kernel", "bias", "bias_q", "padding_const_value", "precision_split_zp",
              "output_stage/piecewise/offsets", "layer_params/negative_slopes_correction_factor"):
        require(not np.any(raw(conv, k)), k)
    for n in (inp, conv, out):
        for k in ("zero_point_in", "zero_point_out", "layer_params/input_zero_points",
                  "layer_params/output_zero_points"):
            require(not np.any(raw(n, k)), n + "/" + k)
    require(np.array_equal(raw(inp, "qp_out"), [0, 1]), "input must use codes zero=0, one=1")
    require(np.array_equal(raw(conv, "qp_in"), [0, 1]), "convolution input quantization")
    require(np.array_equal(raw(out, "qp_out"), raw(conv, "qp_out")), "output rescaling")
    outs = Fraction(float(raw(conv, "qp_out")[1]))
    require(outs > 0 and raw(conv, "qp_out")[0] == 0, "output scale")
    for layer_name, input_scale, output_scale in ((inp, Fraction(1), Fraction(1)),
                                                (conv, Fraction(1), outs), (out, outs, outs)):
        require(uniform(raw(layer_name, "layer_params/input_scales")) == input_scale
                and uniform(raw(layer_name, "layer_params/output_scales")) == output_scale,
                "per-channel stream rescaling")
    kernel = raw(conv, "kernel")
    require(kernel.dtype == np.int8 and kernel.shape == (1, 1, cfg["cin"], cfg["cout"]), "kernel")
    require(np.all(kernel >= 0), "negative kernel")
    shift = int(uniform(raw(conv, "output_stage/mult_shift")))
    require(0 <= shift <= 15, "MAC shift")
    mantissa = uniform(raw(conv, "output_stage/piecewise/slopes_m"))
    encoded_exp = uniform(raw(conv, "output_stage/piecewise/slopes_e"))
    require(mantissa > 0 and mantissa.denominator == encoded_exp.denominator == 1,
            "nonintegral or nonpositive APU constants")

    with runner.infer_context(InferenceContext.SDK_BIT_EXACT) as context:
        model = runner.get_keras_model(context).model
        layer = model.layers[conv]
        require(type(layer).__name__ == "HailoConv", "SDK layer class")
        mac, bias, act, output = (getattr(layer, n) for n in
                                 ("conv_op", "bias_add_op", "act_op", "output_op"))
        require(type(mac).__name__ == "ConvStrippedOp" and type(act).__name__ == "ActivationOp"
                and type(output).__name__ == "PassthruOp", "atomic operations")
        require(uniform(mac.pre_acc_shift) == shift and mac.weight_placement_shift == 0
                and uniform(mac.feed_repeat) == 1, "MAC transforms")
        require(not np.any(np.asarray(bias.bias)), "SDK bias")
        acc = mac.output_lossy_elements[0]
        require(acc.bits in (16, 32) and acc.signed, "accumulator")
        require(act.input_lossy_elements[0].bits == acc.bits, "APU input precision")
        require(act.output_lossy_elements[0].bits == 8
                and not act.output_lossy_elements[0].signed, "APU output precision")
        final_shift = int(act.apu_final_shift)
        require(0 <= final_shift <= 15 and act.shift_data == 0, "APU final shift")
        exponent = int(encoded_exp) + int(act.apu_exp_bias)
        require(uniform(act.exponents) == -exponent
                and uniform(act.mantissas_candidate) == mantissa, "APU export mismatch")
        require(exponent >= 0, "APU exponent")
        clip1, clip2 = act.weight_lossy_elements.clip1, act.weight_lossy_elements.clip2
        require(clip1.min_value <= 0 and clip2.min_value <= 0, "APU clipping zero")
        # Binary input codes make each product rounding exactly enumerable (not an error bound).
        mac_weights = np.array([[round(Fraction(int(v), 2**shift)) for v in row]
                                for row in kernel[0, 0].T], dtype=np.int64)
        acc_bounds = [sum(sorted(map(int, row), reverse=True)[:16]) for row in mac_weights]
        require(max(acc_bounds) <= acc.max_value, "MAC accumulation overflow")
        thresholds = raw(conv, "output_stage/piecewise/x_points")
        require(thresholds.ndim == 2 and thresholds.shape[0] == 1
                and np.all(np.diff(thresholds, axis=1) > 0)
                and thresholds.min() <= 0 and thresholds.max() >= max(acc_bounds),
                "APU segment domain")
        # All segment slopes/offsets are identical, so thresholds cannot change the operation.
        apu_factor = mantissa / 2**exponent
        for bound in acc_bounds:
            first_upper = bound * apu_factor + 1
            require(first_upper <= clip1.max_value and first_upper <= clip2.max_value,
                    "internal APU clipping")
            require(first_upper / 2**final_shift + 1 <= 255, "output saturation")
        effective_factor = apu_factor * outs / 2**final_shift
        effective = np.asarray([[str(int(v) * effective_factor) for v in row]
                                for row in mac_weights])
        internal = outs / 2**final_shift  # a full first-round LSB, before final rounding
        spec = {
            "operation": "single_linear_conv1x1", "activation": "identity",
            "accumulator_bits": int(acc.bits),
            "accumulator_abs_bounds": list(map(str, acc_bounds)),
            "internal_error_bounds": [str(internal)] * cfg["cout"],
            "output_rounding": "nearest",
            "pre_output_rounding_bounds": [["0", str((v * apu_factor + 1) * outs / 2**final_shift)]
                                           for v in acc_bounds],
            "input": {"scale": "1", "zero_point": "0", "zero_code": 0, "one_code": 1},
            "output": {"scale": str(outs), "zero_point": "0"},
        }
        sources = {str(Path(inspect.getfile(type(o))).name): sha(inspect.getfile(type(o)))
                   for o in (mac, bias, act, output)}
        observation = {
            "mac_shift": shift, "apu_mantissa": str(mantissa),
            "apu_encoded_exponent": str(encoded_exp), "apu_exponent_bias": int(act.apu_exp_bias),
            "apu_final_shift": final_shift, "accumulator_bits": int(acc.bits),
            "clip1": [int(clip1.min_value), int(clip1.max_value)],
            "clip2": [int(clip2.min_value), int(clip2.max_value)],
            "sdk_source_sha256": sources, "extractor_sha256": sha(__file__),
            "proof_basis": "binary per-product nearest-even shift; linear APU; two bounded rounds",
        }
    np.savez(root / "numeric-evidence.npz", effective_weights=effective,
             effective_bias=np.asarray(["0"] * cfg["cout"]))
    report = {"compiler_version": "3.33.0", "hef_sha256": sha(root / "model.hef"),
              "onnx_sha256": sha(root / "model.onnx"), "affine": spec,
              "extraction": observation,
              "raw_parameters_sha256": sha(root / "raw-translated-params.npz"),
              "compiled_parameters_sha256": sha(root / "compiled-translated-params.npz"),
              "compiled_har_sha256": sha(root / "compiled.har")}
    (root / "compiler-report.json").write_text(json.dumps(report, indent=2))
    return {
        "runtime_version": "4.23.0", "architecture": "HAILO8L",
        "arithmetic_profile": "binary-circulant-affine-v1",
        "files": {n: sha(root / n) for n in
                  ("model.hef", "numeric-evidence.npz", "compiler-report.json")},
        "input": {"name": inp, "shape": [1, 1, cfg["cin"]], "dtype": "UINT8",
                  "scale": "1", "zero_point": "0"},
        "output": {"name": conv, "shape": [1, 1, cfg["cout"]], "dtype": "UINT8",
                   "scale": str(outs), "zero_point": "0"},
    }


def main():
    import argparse
    import os
    from importlib.metadata import version

    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    root = Path(args.output).resolve()
    cfg = json.loads((root / "build-config.json").read_text())
    report = json.loads((root / "build-report.json").read_text())
    require(cfg["profile"] == "binary-circulant-affine-v1", "requested profile")
    require(report["status"] in ("compiled-unverified", "ready-for-admission", "unsupported"),
            "build must have compiled successfully")
    require(report["source"] == cfg["source"], "build/source identity")
    require(report["hef_sha256"] == sha(root / "model.hef"), "HEF hash")
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    os.chdir(root)
    from hailo_sdk_client import ClientRunner

    require(version("hailo-dataflow-compiler") == "3.33.0", "DFC version")
    try:
        runner = ClientRunner(har=str(root / "compiled.har"))
        compiled = extract_affine(runner, root, cfg)
        compiled["compilation_ns"] = report["timings_ns"]["compile"]
        report.update(status="ready-for-admission", extraction_supported=True, compiled=compiled)
        report.pop("error", None)
    except Exception as exc:
        report.update(status="unsupported", extraction_supported=False,
                      error=f"{type(exc).__name__}: {exc}")
    (root / "build-report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps({k: report.get(k) for k in
                     ("status", "extraction_supported", "error", "hef_sha256")}, indent=2))
    return 0 if report["status"] == "ready-for-admission" else 2


if __name__ == "__main__":
    raise SystemExit(main())
