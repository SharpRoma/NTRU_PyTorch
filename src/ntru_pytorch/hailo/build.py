"""Native Linux / Windows-WSL builds and explicit admission of compiler evidence."""

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path, PureWindowsPath

import numpy as np

from ..randomness import RandomSource
from .artifacts import ArtifactError, load_artifact, sha256


def wsl_path(path):
    path = PureWindowsPath(str(path))
    if not path.is_absolute() or not path.drive or path.drive.startswith("\\\\"):
        raise ValueError("Expected a local absolute Windows path")
    return "/mnt/" + path.drive[0].lower() + "/" + "/".join(path.parts[1:])


def calibration(scheme, seed=42, samples=1024):
    if samples < 1:
        raise ValueError("Calibration sample count must be positive")
    rs = RandomSource(seed)
    frames = [np.zeros(scheme.cin, dtype=np.uint8)]
    for i in range(scheme.n):
        row = np.zeros(scheme.cin, dtype=np.uint8)
        row[i] = 1
        frames.append(row)
    for lo in range(0, scheme.n, 16):
        row = np.zeros(scheme.cin, dtype=np.uint8)
        row[lo : min(lo + 16, scheme.n)] = 1
        frames.append(row)
    for _ in range(samples):
        row = np.zeros(scheme.cin, dtype=np.uint8)
        lo = rs.rng.randrange(scheme.tiles) * 16
        for i in range(lo, min(lo + 16, scheme.n)):
            row[i] = rs.rng.randrange(2)
        frames.append(row)
    return np.stack(frames).reshape(-1, 1, 1, scheme.cin)


def build_hailo(
    artifact_dir,
    build_dir,
    *,
    distro="Ubuntu-22.04",
    linux_python="/opt/ntru-dfc/bin/python",
    profile="unit-v1",
    seed=42,
    samples=1024,
):
    if profile not in ("unit-v1", "binary-circulant-affine-v1"):
        raise ValueError("Unknown requested arithmetic profile")
    manifest, scheme = load_artifact(artifact_dir)
    root = Path(build_dir).resolve()
    root.mkdir(parents=True, exist_ok=True)
    if any(root.iterdir()):
        raise ArtifactError("Build directory must be empty; previous experiments are preserved")
    report = {
        "status": "starting",
        "profile": profile,
        "source_manifest_sha256": sha256(Path(artifact_dir) / "manifest.json"),
        "source_files": manifest["files"],
        "key_sha256": manifest["key_sha256"],
        "seed": seed,
        "hardware_tested": False,
    }
    try:
        frames = calibration(scheme, seed, samples)
        np.save(root / "calibration.npy", frames.astype(np.float32), allow_pickle=False)
        np.save(root / "expected.npy", scheme.software(frames), allow_pickle=False)
        shutil.copy2(Path(artifact_dir) / "model.onnx", root / "model.onnx")
        config = {
            "profile": profile,
            "cin": scheme.cin,
            "cout": scheme.cout,
            "seed": seed,
            "source": report,
            "calibration_frames": len(frames),
        }
        config["calibration_sha256"] = sha256(root / "calibration.npy")
        config["expected_sha256"] = sha256(root / "expected.npy")
        (root / "build-config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
        worker = Path(__file__).with_name("dfc_worker.py").resolve()
        if sys.platform == "win32":
            command = [
                "wsl.exe",
                "--distribution",
                distro,
                "--exec",
                linux_python,
                wsl_path(worker),
                "--output",
                wsl_path(root),
            ]
        else:
            command = [linux_python, str(worker), "--output", str(root)]
        (root / "command.json").write_text(json.dumps(command, indent=2), encoding="utf-8")
        with (root / "build.log").open("wb") as log:
            result = subprocess.run(
                command, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                check=False, cwd=root, env={**os.environ, "CUDA_VISIBLE_DEVICES": "-1"},
            )
        if (root / "build-report.json").exists():
            return json.loads((root / "build-report.json").read_text(encoding="utf-8"))
        raise ArtifactError(
            f"DFC worker did not produce a report (exit {result.returncode}); see build.log"
        )
    except Exception as exc:
        report.update(status="blocked", error=f"{type(exc).__name__}: {exc}")
        (root / "build-report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        raise ArtifactError(report["error"]) from exc


def admit_build(artifact_dir, build_dir, output_dir):
    source = Path(artifact_dir)
    build = Path(build_dir)
    target = Path(output_dir)
    manifest, _ = load_artifact(source)
    report = json.loads((build / "build-report.json").read_text(encoding="utf-8"))
    if (
        report.get("status") != "ready-for-admission"
        or report.get("extraction_supported") is not True
    ):
        raise ArtifactError(
            "Build has no audited numerical extraction; compiled does not mean admitted"
        )
    if report.get("source", {}).get("source_manifest_sha256") != sha256(source / "manifest.json"):
        raise ArtifactError("Build belongs to another source artifact")
    for name in ("model.onnx", "weights.npy", "public.json", "vectors.npz"):
        if manifest["files"][name] != report["source"]["source_files"].get(name):
            raise ArtifactError("Build/source hashes differ")
    if target.exists():
        raise ArtifactError("Admission output must not exist")
    # Validate in a temporary sibling before making a runtime artifact visible.
    from tempfile import TemporaryDirectory

    target.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=target.parent, prefix="hailo-admission-") as temporary:
        staging = Path(temporary)
        for name in manifest["files"]:
            shutil.copy2(source / name, staging / name)
        for name in ("model.hef", "numeric-evidence.npz", "compiler-report.json"):
            shutil.copy2(build / name, staging / name)
        manifest["compiled"] = report["compiled"]
        manifest["status"] = "numerically-admitted-not-hardware-tested"
        (staging / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        load_artifact(staging, compiled=True)
        # Inspect the actual HEF metadata without opening a physical device.
        import hailo_platform as hp

        from .runtime import validate_hef_ports

        validate_hef_ports(hp.HEF(str(staging / "model.hef")), manifest["compiled"], hp)
        shutil.copytree(staging, target)
    return manifest
