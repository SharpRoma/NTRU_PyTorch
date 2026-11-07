import json
import subprocess
import sys
from dataclasses import replace

import numpy as np
import pytest

from ntru_pytorch import (
    PRESETS,
    Parameters,
    PublicKey,
    RandomSource,
    SympyNTRU,
    TorchNTRU,
    generate_keys,
)
from ntru_pytorch.hailo import (
    ArtifactError,
    BinaryScheme,
    HailoEncryptor,
    SoftwareHailoEncryptor,
    prepare_hailo,
)
from ntru_pytorch.hailo.artifacts import load_artifact
from ntru_pytorch.hailo.scheme import key_hash
from ntru_pytorch.hailo.validation import validate_onnx, validate_software


@pytest.fixture(scope="module")
def pair():
    return generate_keys(Parameters(17, 3, 256, 3, 2, 1), random_source=RandomSource(14))


@pytest.mark.parametrize(
    "data",
    [
        b"",
        b"abc",
        b"a\0b\0",
        bytes(range(40)),
        "\u041f\u0440\u0438\u0432\u0435\u0442 \U0001f600".encode(),
    ],
)
def test_software_cross_compatibility(pair, data):
    public, private = pair
    engine = SoftwareHailoEncryptor(public)
    cipher = engine.encrypt_bytes(public, data, random_source=RandomSource(7))
    for reference in (SympyNTRU(), TorchNTRU()):
        assert reference.encrypt_bytes(public, data, random_source=RandomSource(7)) == cipher
        assert reference.decrypt_bytes(private, cipher) == data
    assert engine.last_timings["polynomial_blocks"] == len(cipher.blocks)
    assert engine.last_timings["frames"] == len(cipher.blocks) * 2 * engine.scheme.tiles


def test_scheme_bounds_and_orientation():
    key = PublicKey(Parameters(17, 3, 256, 3, 2, 1), (255,) * 17)
    scheme = BinaryScheme(key)
    r = (1,) * 16 + (0,)
    outputs = scheme.software(scheme.pack([r]))
    assert outputs.max() == 16
    assert np.all(scheme.reconstruct(outputs, 1) == 16 * 255)
    for sign in (1, -1):
        r = (0,) * 16 + (sign,)
        assert np.all(scheme.reconstruct(scheme.software(scheme.pack([r])), 1) == sign * 255)
    assert np.all(scheme.weights[17 * 8 :] == 0)
    assert np.all(scheme.weights[:, 17:] == 0)
    assert key_hash(key) == key_hash(PublicKey(key.params, tuple(v - 256 for v in key.h)))
    with pytest.raises(ValueError):
        scheme.pack([[2]])
    with pytest.raises(ValueError):
        scheme.pack([[0] * 18])
    bad = scheme.pack([[0] * 17])
    bad[:] = 1
    with pytest.raises(ValueError):
        scheme.software(bad)
    bad = outputs.astype(float)
    with pytest.raises(ValueError):
        scheme.reconstruct(bad, 1)
    bad = outputs.copy()
    bad[0, 0, 0, -1] = 1
    with pytest.raises(ValueError):
        scheme.reconstruct(bad, 1)


def test_mismatched_key_and_messages(pair):
    key, _ = pair
    engine = SoftwareHailoEncryptor(key)
    with pytest.raises(ArtifactError):
        engine.encrypt_poly(replace(key, h=(0,) * 17), [1])
    with pytest.raises(ValueError):
        engine.encrypt_poly(key, [2])
    with pytest.raises(ValueError):
        engine.encrypt_bytes(key, b"x", r_polys=[])
    rows = []
    engine.encrypt_poly(key, [1], r=[0] * 17, trace=lambda name, a: rows.append(name))
    assert rows == ["cyclic_mod_q", "ciphertext"]


def test_export_and_corruption(pair, tmp_path):
    pytest.importorskip("onnx")
    pytest.importorskip("onnxruntime")
    key, _ = pair
    manifest = prepare_hailo(key, tmp_path)
    assert manifest["status"] == "prepared"
    assert validate_onnx(tmp_path)["exact"]
    with pytest.raises(ArtifactError, match="No compiled HEF"):
        HailoEncryptor(tmp_path)
    with pytest.raises(ArtifactError):
        prepare_hailo(key, tmp_path)
    (tmp_path / "weights.npy").write_bytes(b"corrupt")
    with pytest.raises(ArtifactError, match="Checksum"):
        load_artifact(tmp_path)


def test_compiled_admission(pair, tmp_path):
    pytest.importorskip("onnx")
    key, _ = pair
    m = prepare_hailo(key, tmp_path)
    m["compiled"] = {"runtime_version": "4.24.0", "architecture": "HAILO8L"}
    (tmp_path / "manifest.json").write_text(json.dumps(m))
    with pytest.raises(ArtifactError, match="4.23.0"):
        HailoEncryptor(tmp_path)
    m["compiled"]["runtime_version"] = "4.23.0"
    (tmp_path / "manifest.json").write_text(json.dumps(m))
    with pytest.raises(ArtifactError):
        HailoEncryptor(tmp_path)


def test_cli_diagnostic(pair, tmp_path):
    pytest.importorskip("onnx")
    pytest.importorskip("onnxruntime")
    key, _ = pair
    prepare_hailo(key, tmp_path)
    command = [
        sys.executable,
        "-B",
        "-m",
        "ntru_pytorch",
        "hailo-check",
        "--artifact",
        str(tmp_path),
        "--cases",
        "2",
    ]
    result = subprocess.run(command, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["hardware_tested"] is False
    result = subprocess.run(command + ["--hardware"], capture_output=True, text=True)
    assert result.returncode == 2
    report = json.loads((tmp_path / "last-check.json").read_text())
    assert "No compiled HEF" in report["error"]
    assert report["hardware_tested"] is False


def test_no_vendor_import():
    subprocess.run(
        [
            sys.executable,
            "-B",
            "-c",
            "import sys; import ntru_pytorch.hailo; assert 'hailo_platform' not in sys.modules",
        ],
        check=True,
    )


@pytest.mark.slow
@pytest.mark.parametrize("name", list(PRESETS))
def test_thousand_polynomials_per_preset(name):
    params = PRESETS[name]
    source = RandomSource(123)
    key = PublicKey(params, tuple(source.rng.randrange(params.q) for _ in range(params.N)))
    result = validate_software(key, cases=1000, seed=42)
    assert result["cases"] == 1000 and result["exact"]
    engine = SoftwareHailoEncryptor(key)
    for _ in range(3):
        r = source.ternary(params.N, params.d, params.d)
        message = [0, 1, 0, 1]
        actual = engine.encrypt_poly(key, message, r=r)
        assert actual == TorchNTRU().encrypt_poly(key, message, r=r)
        assert actual == SympyNTRU().encrypt_poly(key, message, r=r)


@pytest.fixture
def compiled_fixture(pair, tmp_path):
    # Synthetic admission contract for adapter tests only; never execute this as real HEF.
    from ntru_pytorch.hailo.artifacts import sha256

    pytest.importorskip("onnx")
    key, _ = pair
    m = prepare_hailo(key, tmp_path)
    scheme = BinaryScheme(key)
    (tmp_path / "model.hef").write_bytes(b"SYNTHETIC TEST ONLY, NOT A HEF")
    np.savez(
        tmp_path / "numeric-evidence.npz",
        effective_weights=scheme.weights,
        effective_bias=np.zeros(scheme.cout),
    )
    report = {
        "hef_sha256": sha256(tmp_path / "model.hef"),
        "onnx_sha256": m["files"]["model.onnx"],
        "operation": "single_linear_conv1x1",
        "activation": "identity",
        "accumulator_bits": 32,
        "requantization": "identity",
        "saturation": False,
        "input_scale": 1,
        "input_zero_point": 0,
        "output_scale": 1,
        "output_zero_point": 0,
        "compiler_version": "synthetic-test",
    }
    (tmp_path / "compiler-report.json").write_text(json.dumps(report))
    m["compiled"] = {
        "runtime_version": "4.23.0",
        "architecture": "HAILO8L",
        "files": {
            name: sha256(tmp_path / name)
            for name in ("model.hef", "numeric-evidence.npz", "compiler-report.json")
        },
        "input": {
            "name": "r",
            "shape": [1, 1, scheme.cin],
            "dtype": "UINT8",
            "scale": 1,
            "zero_point": 0,
        },
        "output": {
            "name": "partial",
            "shape": [1, 1, scheme.cout],
            "dtype": "UINT8",
            "scale": 1,
            "zero_point": 0,
        },
    }
    (tmp_path / "manifest.json").write_text(json.dumps(m))
    return tmp_path, m, scheme


def test_numeric_evidence_rejection(compiled_fixture):
    from ntru_pytorch.hailo.artifacts import sha256

    root, m, scheme = compiled_fixture
    load_artifact(root, compiled=True)
    wrong = scheme.weights.copy()
    wrong[0, 0] ^= 1
    np.savez(
        root / "numeric-evidence.npz", effective_weights=wrong, effective_bias=np.zeros(scheme.cout)
    )
    m["compiled"]["files"]["numeric-evidence.npz"] = sha256(root / "numeric-evidence.npz")
    (root / "manifest.json").write_text(json.dumps(m))
    with pytest.raises(ArtifactError, match="weights"):
        load_artifact(root, compiled=True)


def test_runtime_context_and_stream_validation(compiled_fixture, monkeypatch):
    from contextlib import contextmanager
    from types import SimpleNamespace

    from ntru_pytorch.hailo import runtime

    root, m, scheme = compiled_fixture
    events = []

    @contextmanager
    def context(name, value):
        events.append("open-" + name)
        try:
            yield value
        finally:
            events.append("close-" + name)

    ports = []
    for direction in ("input", "output"):
        port = m["compiled"][direction]
        ports.append(
            SimpleNamespace(
                name=port["name"],
                shape=port["shape"],
                format=SimpleNamespace(type="uint8"),
                quant_info=SimpleNamespace(qp_scale=1, qp_zp=0),
            )
        )
    hef = SimpleNamespace(
        get_input_vstream_infos=lambda: [ports[0]], get_output_vstream_infos=lambda: [ports[1]]
    )
    network = SimpleNamespace(activate=lambda: context("activation", None))
    device = SimpleNamespace(configure=lambda *args: [network])
    pipeline = SimpleNamespace(infer=lambda data: {"partial": scheme.software(data["r"])})
    hp = SimpleNamespace(
        HEF=lambda path: hef,
        Device=SimpleNamespace(scan=lambda: ["test"]),
        VDevice=lambda: context("device", device),
        ConfigureParams=SimpleNamespace(create_from_hef=lambda *args, **kw: {}),
        HailoStreamInterface=SimpleNamespace(PCIe="pcie"),
        FormatType=SimpleNamespace(UINT8="uint8"),
        InputVStreamParams=SimpleNamespace(make=lambda *args, **kw: {}),
        OutputVStreamParams=SimpleNamespace(make=lambda *args, **kw: {}),
        InferVStreams=lambda *args: context("pipeline", pipeline),
    )
    monkeypatch.setitem(sys.modules, "hailo_platform", hp)
    monkeypatch.setattr(runtime, "version", lambda name: "4.23.0")
    session = runtime.HailoSession(root, m, scheme)
    frames = scheme.pack([[1]])
    with session as active:
        assert np.array_equal(active.infer(frames), scheme.software(frames))
    assert events == [
        "open-device",
        "open-pipeline",
        "open-activation",
        "close-activation",
        "close-pipeline",
        "close-device",
    ]
    with pytest.raises(RuntimeError):
        session.infer(frames)
    ports[0].quant_info.qp_scale = 0.5
    with pytest.raises(ArtifactError, match="quantization"):
        with runtime.HailoSession(root, m, scheme):
            pass


def test_hardware_engine_contract(compiled_fixture, monkeypatch, pair):
    from ntru_pytorch.hailo import engine

    root, m, scheme = compiled_fixture
    events = []

    class FakeSession:
        def __init__(self, *args):
            pass

        def __enter__(self):
            events.append("open")
            return self

        def __exit__(self, *exc):
            events.append("close")

        def infer(self, frames):
            return scheme.software(frames)

    monkeypatch.setattr(engine, "HailoSession", FakeSession)
    key, private = pair
    hailo = HailoEncryptor(root)
    with pytest.raises(RuntimeError):
        hailo.encrypt_poly(key, [1])
    with hailo:
        for _ in range(2):
            result = hailo.encrypt_bytes(key, b"abc", random_source=RandomSource(5))
            assert TorchNTRU().decrypt_bytes(private, result) == b"abc"
    assert events == ["open", "close"]


@pytest.mark.slow
@pytest.mark.parametrize("name", list(PRESETS))
def test_onnx_presets(name, tmp_path):
    pytest.importorskip("onnx")
    pytest.importorskip("onnxruntime")
    params = PRESETS[name]
    key = PublicKey(params, tuple(i % params.q for i in range(params.N)))
    prepare_hailo(key, tmp_path)
    assert validate_onnx(tmp_path)["exact"]


def test_benchmark_reporting_with_explicit_fake_device(tmp_path, monkeypatch):
    from ntru_pytorch import PrivateKey
    from ntru_pytorch.hailo import benchmark as bench

    params = PRESETS["moderate"]
    n = params.N
    f = (1,) + (0,) * (n - 1)
    g = (1, -1) + (0,) * (n - 2)
    key = PublicKey(params, (3, -3) + (0,) * (n - 2))
    private = PrivateKey(params, f, f, f, g)
    (tmp_path / "compiler-report.json").write_text(json.dumps({"compiler_version": "TEST-ONLY"}))

    class FakeDevice(SoftwareHailoEncryptor):
        def __init__(self, directory):
            super().__init__(key)
            self.load_ns = 1
            self.manifest = {
                "key_sha256": key_hash(key),
                "compiled": {"files": {"model.hef": "TEST-ONLY"}},
            }

    monkeypatch.setattr(bench, "HailoEncryptor", FakeDevice)
    monkeypatch.setattr(bench, "validate_hardware", lambda *args, **kw: {"stage": "TEST-ONLY"})
    summary = bench.benchmark_hailo(tmp_path, private, tmp_path / "benchmark", runs=2, warmup=0)
    report = json.loads((tmp_path / "benchmark.json").read_text())
    assert len(summary) == 12
    assert len(report["samples"]) == 24
    assert {s["blocks"] for s in summary} == {1, 8, 32, 128}
    hailo = [s for s in report["samples"] if s["backend"] == "hailo"]
    assert all(s["input_bytes"] > 0 and s["output_bytes"] > 0 for s in hailo)
    assert report["metadata"]["hardware_internal_ns"] is None
    assert (tmp_path / "benchmark.csv").read_text().count("TEST-ONLY") > 0


@pytest.mark.hailo_hardware
def test_actual_ntru_hef():
    import os

    from ntru_pytorch.formats import load
    from ntru_pytorch.hailo.validation import validate_hardware

    artifact = os.environ.get("NTRU_HAILO_ARTIFACT")
    private_path = os.environ.get("NTRU_HAILO_PRIVATE_KEY")
    if not artifact or not private_path:
        pytest.skip(
            "Requires a numerically validated NTRU HEF and matching private key; no CPU substitution"
        )
    private = load(private_path)
    with HailoEncryptor(artifact) as engine:
        assert validate_hardware(engine, cases=1000)["exact"]
        key = engine.scheme.key
        for data in (b"", b"a\0b", bytes(range(256))):
            actual = engine.encrypt_bytes(key, data, random_source=RandomSource(42))
            for reference in (SympyNTRU(), TorchNTRU()):
                assert reference.encrypt_bytes(key, data, random_source=RandomSource(42)) == actual
                assert reference.decrypt_bytes(private, actual) == data


def test_affine_admission_is_separate_profile(compiled_fixture):
    from ntru_pytorch.hailo.artifacts import sha256

    root, m, scheme = compiled_fixture
    report = json.loads((root / "compiler-report.json").read_text())
    spec = {
        "operation": "single_linear_conv1x1",
        "activation": "identity",
        "accumulator_bits": 32,
        "accumulator_abs_bounds": ["4096"] * scheme.cout,
        "internal_error_bounds": ["0"] * scheme.cout,
        "output_rounding": "truncate",
        "input": {"scale": "1", "zero_point": "1", "zero_code": 1, "one_code": 2},
        "output": {"scale": "1/8", "zero_point": "0"},
    }
    report["affine"] = spec
    (root / "compiler-report.json").write_text(json.dumps(report))
    m["compiled"]["arithmetic_profile"] = "binary-circulant-affine-v1"
    for direction in ("input", "output"):
        for name in ("scale", "zero_point"):
            m["compiled"][direction][name] = spec[direction][name]
    m["compiled"]["files"]["compiler-report.json"] = sha256(root / "compiler-report.json")
    (root / "manifest.json").write_text(json.dumps(m))
    load_artifact(root, compiled=True)
    spec["output"]["scale"] = "1/2"
    report["affine"] = spec
    m["compiled"]["output"]["scale"] = "1/2"
    (root / "compiler-report.json").write_text(json.dumps(report))
    m["compiled"]["files"]["compiler-report.json"] = sha256(root / "compiler-report.json")
    (root / "manifest.json").write_text(json.dumps(m))
    with pytest.raises(ArtifactError, match="1/2"):
        load_artifact(root, compiled=True)
