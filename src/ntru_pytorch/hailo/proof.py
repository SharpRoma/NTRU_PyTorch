"""Conservative rational bounds for a single affine-quantized linear layer."""

from fractions import Fraction

import numpy as np

from .artifacts import ArtifactError

AFFINE_PROFILE = "binary-circulant-affine-v1"


def rational(value):
    if isinstance(value, bool):
        raise ArtifactError("Boolean is not a quantization constant")
    try:
        return Fraction(str(value)) if isinstance(value, str) else Fraction(float(value))
    except (ValueError, OverflowError, TypeError, ZeroDivisionError) as exc:
        raise ArtifactError("Expected a finite exact rational constant") from exc


def prove_affine(scheme, weights, bias, spec):
    """Prove error < 1/2, no saturation, and no signed accumulator overflow.

    Effective coefficients describe the actual compiled decoded linear operation,
    not the original floating-point graph. Missing internal error evidence fails.
    """
    if weights.shape != scheme.weights.shape or bias.shape != (scheme.cout,):
        raise ArtifactError("Malformed effective affine coefficients")
    if spec.get("operation") != "single_linear_conv1x1" or spec.get("activation") != "identity":
        raise ArtifactError("Only a fully accounted single linear operation is supported")
    if spec.get("output_rounding") not in ("nearest", "truncate"):
        raise ArtifactError("Unaccounted output rounding")
    bits = spec.get("accumulator_bits")
    if type(bits) is not int or bits not in (16, 32):
        raise ArtifactError("Unknown accumulator semantics")
    accumulator_bounds = spec.get("accumulator_abs_bounds")
    if not isinstance(accumulator_bounds, list) or len(accumulator_bounds) != scheme.cout:
        raise ArtifactError("Missing integer accumulator bounds")
    if any(rational(v) < 0 or rational(v) >= 2 ** (bits - 1) for v in accumulator_bounds):
        raise ArtifactError("Accumulator overflow is not excluded")
    inp, out = spec["input"], spec["output"]
    ins, izp = rational(inp["scale"]), rational(inp["zero_point"])
    outs, ozp = rational(out["scale"]), rational(out["zero_point"])
    zero, one = inp["zero_code"], inp["one_code"]
    if (
        ins <= 0
        or outs <= 0
        or any(type(v) is not int or not 0 <= v <= 255 for v in (zero, one))
        or zero == one
    ):
        raise ArtifactError("Invalid binary input codes or scales")
    x0, x1 = ins * (zero - izp), ins * (one - izp)
    errors = spec.get("internal_error_bounds")
    if not isinstance(errors, list) or len(errors) != scheme.cout:
        raise ArtifactError("Missing per-output internal rounding bounds")
    low, high = outs * (-ozp), outs * (255 - ozp)
    # Optional independently extracted bounds before output rounding/clipping.
    # These must come from the integer operations, not from clipped samples.
    pre_output = spec.get("pre_output_rounding_bounds")
    if pre_output is not None and (
        not isinstance(pre_output, list) or len(pre_output) != scheme.cout
    ):
        raise ArtifactError("Missing per-output pre-rounding bounds")
    bounds = []
    for k, row in enumerate(weights):
        all_weights = [rational(v) for v in row]
        w = all_weights[: scheme.n]
        base = rational(bias[k]) + sum(v * x0 for v in all_weights)
        delta = [v * (x1 - x0) for v in w]
        deviations = sorted(
            (abs(v - int(scheme.weights[k, i])) for i, v in enumerate(delta)), reverse=True
        )
        internal = rational(errors[k])
        if internal < 0:
            raise ArtifactError("Negative internal error bound")
        # A full output LSB covers nearest and truncation without assuming tie behavior.
        bound = abs(base) + sum(deviations[:16]) + internal + outs
        minimum = base + sum(sorted(v for v in delta if v < 0)[:16]) - internal
        maximum = base + sum(sorted((v for v in delta if v > 0), reverse=True)[:16]) + internal
        if pre_output is not None:
            pair = pre_output[k]
            if not isinstance(pair, list) or len(pair) != 2:
                raise ArtifactError("Malformed pre-rounding bounds")
            certified_low, certified_high = map(rational, pair)
            if certified_low > certified_high:
                raise ArtifactError("Reversed pre-rounding bounds")
            minimum = max(minimum, certified_low)
            maximum = min(maximum, certified_high)
            if minimum > maximum:
                raise ArtifactError("Inconsistent pre-rounding bounds")
        if minimum < low or maximum > high:
            raise ArtifactError("Output saturation cannot be excluded")
        if bound >= Fraction(1, 2):
            raise ArtifactError("Strict reconstruction error bound < 1/2 is not established")
        bounds.append(str(bound))
    return {
        "profile": AFFINE_PROFILE,
        "error_bounds": bounds,
        "max_error_bound": str(max(map(Fraction, bounds))),
        "rounding": "nearest_integer_after_dequantization",
        "saturation_excluded": True,
    }


def encode_affine(frames, spec):
    inp = spec["input"]
    return np.where(frames == 0, inp["zero_code"], inp["one_code"]).astype(np.uint8)


def decode_affine(codes, spec):
    return decode_affine_lookup(codes, affine_lookup(spec))


def affine_lookup(spec):
    s, z = rational(spec["output"]["scale"]), rational(spec["output"]["zero_point"])
    # Exact lookup avoids floating point introducing new errors during recovery.
    lookup = []
    for code in range(256):
        value = s * (code - z)
        rounded = (value + Fraction(1, 2)).numerator // (value + Fraction(1, 2)).denominator
        lookup.append(rounded)
    lookup = np.asarray(lookup, dtype=np.int64)
    lookup.setflags(write=False)
    return lookup


def decode_affine_lookup(codes, lookup):
    if codes.dtype != np.uint8:
        raise ArtifactError("Expected raw UINT8 output codes")
    values = lookup[codes]
    if np.any(values < 0) or np.any(values > 16):
        raise ArtifactError("Recovered partial sum outside [0,16]")
    return values.astype(np.uint8)


def compact_affine_lookup(lookup):
    """Compress an exact rational-derived table, preserving invalid-code rejection."""
    compact = np.full(256, 255, dtype=np.uint8)
    valid = (lookup >= 0) & (lookup <= 16)
    compact[valid] = lookup[valid].astype(np.uint8)
    compact.setflags(write=False)
    return compact


def decode_affine_compact(codes, lookup):
    if codes.dtype != np.uint8:
        raise ArtifactError("Expected raw UINT8 output codes")
    values = lookup[codes]
    if np.any(values == 255):
        raise ArtifactError("Recovered partial sum outside [0,16]")
    return values
