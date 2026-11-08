import sys
from fractions import Fraction

import numpy as np
import pytest

from ntru_pytorch import Parameters, PublicKey
from ntru_pytorch.hailo import ArtifactError, BinaryScheme, prepare_hailo
from ntru_pytorch.hailo.build import admit_build, build_hailo, calibration, wsl_path
from ntru_pytorch.hailo.proof import decode_affine, encode_affine, prove_affine, rational


def scheme():
    return BinaryScheme(PublicKey(Parameters(17, 3, 256, 3, 2, 1), tuple(i * 7 for i in range(17))))


def spec_for(s):
    return {
        "operation": "single_linear_conv1x1",
        "activation": "identity",
        "accumulator_bits": 32,
        "accumulator_abs_bounds": ["4096"] * s.cout,
        "internal_error_bounds": ["0"] * s.cout,
        "output_rounding": "truncate",
        "input": {"scale": "1", "zero_point": "0", "zero_code": 0, "one_code": 1},
        "output": {"scale": "1/8", "zero_point": "0"},
    }


def test_rational_affine_recovery():
    s = scheme()
    spec = spec_for(s)
    proof = prove_affine(s, s.weights, np.zeros(s.cout), spec)
    assert Fraction(proof["max_error_bound"]) == Fraction(1, 8)
    frames = s.pack([[1, -1, 0, 1]])
    assert np.array_equal(encode_affine(frames, spec), frames)
    sums = s.software(frames)
    assert np.array_equal(decode_affine(sums * 8, spec), sums)
    assert rational(0.1) == Fraction.from_float(0.1)
    with pytest.raises(ArtifactError):
        rational(float("nan"))
    with pytest.raises(ArtifactError):
        rational(True)


def test_proof_fails_closed():
    s = scheme()
    spec = spec_for(s)
    spec["output"]["scale"] = "1/2"
    with pytest.raises(ArtifactError, match="1/2"):
        prove_affine(s, s.weights, np.zeros(s.cout), spec)
    spec = spec_for(s)
    spec["output"]["scale"] = "1/256"
    with pytest.raises(ArtifactError, match="saturation"):
        prove_affine(s, s.weights, np.zeros(s.cout), spec)
    spec = spec_for(s)
    spec["internal_error_bounds"] = ["1/2"] * s.cout
    with pytest.raises(ArtifactError):
        prove_affine(s, s.weights, np.zeros(s.cout), spec)
    spec = spec_for(s)
    spec["accumulator_abs_bounds"] = ["2147483648"] * s.cout
    with pytest.raises(ArtifactError, match="overflow"):
        prove_affine(s, s.weights, np.zeros(s.cout), spec)
    spec = spec_for(s)
    del spec["internal_error_bounds"]
    with pytest.raises(ArtifactError, match="Missing"):
        prove_affine(s, s.weights, np.zeros(s.cout), spec)


def test_observed_16bit_accumulator_and_pre_rounding_bounds():
    s = scheme()
    spec = spec_for(s)
    spec["accumulator_bits"] = 16
    spec["internal_error_bounds"] = ["1/64"] * s.cout
    # The generic symmetric error interval crosses zero; structural integer
    # bounds before clipping establish that this operation stays nonnegative.
    with pytest.raises(ArtifactError, match="saturation"):
        prove_affine(s, s.weights, np.zeros(s.cout), spec)
    spec["pre_output_rounding_bounds"] = [["0", "17"]] * s.cout
    proof = prove_affine(s, s.weights, np.zeros(s.cout), spec)
    assert Fraction(proof["max_error_bound"]) == Fraction(9, 64)
    spec["accumulator_abs_bounds"] = ["32768"] * s.cout
    with pytest.raises(ArtifactError, match="overflow"):
        prove_affine(s, s.weights, np.zeros(s.cout), spec)
    spec["accumulator_abs_bounds"] = ["1024"] * s.cout
    spec["pre_output_rounding_bounds"][0] = ["1", "0"]
    with pytest.raises(ArtifactError, match="Reversed"):
        prove_affine(s, s.weights, np.zeros(s.cout), spec)


def test_input_zero_offset_and_padding_accounted():
    s = scheme()
    spec = spec_for(s)
    spec["input"].update(scale="1", zero_point="1", zero_code=1, one_code=2)
    assert prove_affine(s, s.weights, np.zeros(s.cout), spec)
    # Padded channels still carry the decoded zero input: their weights cannot be ignored.
    spec["input"]["zero_point"] = "0"
    bad = s.weights.astype(float)
    bad[:, -1] = 100
    with pytest.raises(ArtifactError):
        prove_affine(s, bad, np.zeros(s.cout), spec)


def test_calibration_and_paths():
    s = scheme()
    a = calibration(s, samples=20)
    assert np.array_equal(a, calibration(s, samples=20))
    s.validate_frames(a)
    assert not a[0].any()
    assert a.reshape(len(a), -1).sum(axis=1).max() == 16
    assert wsl_path(r"C:\Research Files\model.onnx") == "/mnt/c/Research Files/model.onnx"
    with pytest.raises(ValueError):
        wsl_path("relative")


def test_missing_dfc_worker_reports_failure(tmp_path, monkeypatch):
    pytest.importorskip("onnx")
    source = tmp_path / "source"
    build = tmp_path / "build"
    prepare_hailo(scheme().key, source)
    monkeypatch.setattr("ntru_pytorch.hailo.build.sys.platform", "linux")
    result = build_hailo(source, build, linux_python=sys.executable, samples=2)
    assert result["status"] == "failed"
    assert result["extraction_supported"] is False
    assert (build / "build.log").exists()
    assert not (build / "numeric-evidence.npz").exists()
    with pytest.raises(ArtifactError):
        admit_build(source, build, tmp_path / "admitted")
    assert not (tmp_path / "admitted").exists()
    with pytest.raises(ArtifactError):
        build_hailo(source, build, linux_python=sys.executable)
