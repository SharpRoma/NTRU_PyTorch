#!/usr/bin/env python3
"""Pinned, local-only liboqs build. Run portable before native; no Python installs."""

import argparse
import hashlib
import json
import os
import platform
import shutil
import subprocess
from pathlib import Path

VERSION = "0.16.0"
COMMIT = "5a1a854b0dc9f2141bdc771c555ee60c37950183"
ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1048576), b""):
            h.update(chunk)
    return h.hexdigest()


def run(args, log=None):
    if log:
        with log.open("a") as f:
            subprocess.run(list(map(str, args)), stdout=f, stderr=subprocess.STDOUT, check=True)
    else:
        subprocess.run(list(map(str, args)), check=True)


def build(profile, jobs, sanitize=False):
    arch = platform.machine()
    if arch not in ("x86_64", "aarch64"):
        raise RuntimeError("Only Linux x86_64 and AArch64 are supported")
    if platform.system() != "Linux":
        raise RuntimeError("Linux required")
    base = ROOT / "results/native/oqs" / VERSION
    source = base / "source"
    if not source.exists():
        run(
            [
                "git",
                "clone",
                "--branch",
                VERSION,
                "--single-branch",
                "https://github.com/open-quantum-safe/liboqs.git",
                source,
            ]
        )
    if (
        subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"], text=True).strip()
        != COMMIT
    ):
        raise RuntimeError("Source commit mismatch")
    if subprocess.check_output(
        ["git", "-C", str(source), "status", "--porcelain"], text=True
    ).strip():
        raise RuntimeError("Modified upstream source checkout")
    impl = "clean"
    if profile == "native" and arch == "x86_64":
        flags = Path("/proc/cpuinfo").read_text().split("flags", 1)[1].split("\n", 1)[0].split()
        if not {"avx2", "bmi2"} <= set(flags):
            raise RuntimeError("Native AVX2/BMI2 unavailable; no silent fallback")
        impl = "avx2"
    name = profile + ("-sanitizer" if sanitize else "")
    dest = base / arch / name
    config = dict(
        version=VERSION,
        commit=COMMIT,
        architecture=arch,
        profile=profile,
        implementation=impl,
        sanitizer=sanitize,
        bridge_abi=1,
    )
    adapter_hashes = {
        p.name: sha(p) for p in sorted((ROOT / "native/oqs").glob("*")) if p.is_file()
    }
    config["adapter_sources"] = adapter_hashes
    manifest = dest / "manifest.json"
    if manifest.exists():
        old = json.loads(manifest.read_text())
        if old["configuration"] != config:
            raise RuntimeError("Incompatible existing build; retain it and use a fresh directory")
        for rel, digest in old["files"].items():
            if sha(dest / rel) != digest:
                raise RuntimeError("Existing build hash mismatch: " + rel)
        print(manifest)
        return
    dest.mkdir(parents=True, exist_ok=True)
    lock = dest / "configuration.json"
    if lock.exists() and json.loads(lock.read_text()) != config:
        raise RuntimeError("Incompatible incomplete build")
    lock.write_text(json.dumps(config, indent=2) + "\n")
    log = dest / "build.log"
    b = dest / "build"
    opts = [
        "-DCMAKE_BUILD_TYPE=" + ("Debug" if sanitize else "Release"),
        "-DBUILD_SHARED_LIBS=ON",
        "-DOQS_USE_OPENSSL=OFF",
        "-DOQS_DIST_BUILD=OFF",
        "-DOQS_MINIMAL_BUILD=KEM_ntru_hrss701",
        "-DOQS_BUILD_ONLY_LIB=OFF",
        "-DOQS_USE_CUPQC=OFF",
        "-DOQS_USE_ICICLE=OFF",
        "-DCMAKE_EXPORT_COMPILE_COMMANDS=ON",
        "-DOQS_ENABLE_KEM_ntru_hrss701_avx2=" + ("ON" if impl == "avx2" else "OFF"),
    ]
    if profile == "portable":
        opts += ["-DOQS_OPT_TARGET=generic"]
        if arch == "x86_64":
            opts += [
                "-DOQS_USE_" + flag + "_INSTRUCTIONS=OFF"
                for flag in ["AVX", "AVX2", "AVX512", "BMI1", "BMI2", "AES", "PCLMULQDQ", "POPCNT"]
            ]
    elif arch == "aarch64":
        opts += ["-DOQS_OPT_TARGET=cortex-a76"]
    san = "-fsanitize=address,undefined -fno-omit-frame-pointer"
    if sanitize:
        opts += [
            "-DCMAKE_C_FLAGS=" + san,
            "-DCMAKE_SHARED_LINKER_FLAGS=" + san,
            "-DCMAKE_EXE_LINKER_FLAGS=" + san,
        ]
    run(["cmake", "-S", source, "-B", b, "-G", "Unix Makefiles", *opts], log)
    run(["cmake", "--build", b, "--target", "oqs", "kat_kem", "test_kem", "--parallel", jobs], log)
    bridge = dest / "bridge-build"
    bopts = [
        "-DCMAKE_BUILD_TYPE=" + ("Debug" if sanitize else "Release"),
        "-DCMAKE_EXPORT_COMPILE_COMMANDS=ON",
        "-DOQS_SOURCE=" + str(source),
        "-DOQS_BUILD=" + str(b),
        "-DNR_IMPL=" + impl,
        "-DCMAKE_INSTALL_PREFIX=" + str(dest),
    ]
    if profile == "native" and arch == "aarch64":
        bopts += ["-DCMAKE_C_FLAGS=-mcpu=cortex-a76"]
    if sanitize:
        bopts += [
            "-DCMAKE_C_FLAGS=" + san,
            "-DCMAKE_SHARED_LINKER_FLAGS=" + san,
            "-DCMAKE_EXE_LINKER_FLAGS=" + san,
        ]
    run(["cmake", "-S", ROOT / "native/oqs", "-B", bridge, *bopts], log)
    run(["cmake", "--build", bridge, "--parallel", jobs], log)
    run(["cmake", "--install", bridge], log)
    lib = dest / "lib"
    lib.mkdir(exist_ok=True)
    for p in (b / "lib").glob("liboqs.so*"):
        target = lib / p.name
        if target.exists() or target.is_symlink():
            target.unlink()
        if p.is_symlink():
            target.symlink_to(os.readlink(p))
        else:
            shutil.copy2(p, target)
    licenses = dest / "licenses"
    licenses.mkdir(exist_ok=True)
    for p in source.glob("LICENSE*"):
        shutil.copy2(p, licenses / p.name)
    for variant in ("clean", "avx2"):
        p = source / f"src/kem/ntru/pqclean_ntruhrss701_{variant}/LICENSE"
        if p.exists():
            shutil.copy2(p, licenses / (variant + "-LICENSE"))
    tracked = (
        subprocess.check_output(["git", "-C", str(source), "ls-files", "-z"]).decode().split("\0")
    )
    source_hashes = {p: sha(source / p) for p in tracked if p and (source / p).is_file()}
    (dest / "source-sha256.json").write_text(
        json.dumps(source_hashes, sort_keys=True, indent=2) + "\n"
    )
    files = {}
    for parent in (lib, licenses):
        for p in sorted(parent.iterdir()):
            if p.is_file():
                files[str(p.relative_to(dest))] = sha(p)
    for p in [
        dest / "source-sha256.json",
        b / "compile_commands.json",
        b / "CMakeCache.txt",
        bridge / "compile_commands.json",
    ]:
        files[str(p.relative_to(dest))] = sha(p)
    data = dict(
        configuration=config,
        files=files,
        cmake_options=opts,
        bridge_options=bopts,
        compiler=subprocess.check_output(["cc", "--version"], text=True),
        elf={
            p.name: subprocess.check_output(["readelf", "-h", p], text=True)
            for p in lib.glob("*.so")
        },
        source_checkout=str(source),
        libraries=dict(
            kem="lib/libntru_kem_bridge.so",
            arithmetic="lib/libntru_rq_bridge.so",
            oqs="lib/liboqs.so",
        ),
    )
    manifest.write_text(json.dumps(data, indent=2) + "\n")
    print(manifest)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=["portable", "native"], default="portable")
    parser.add_argument("--jobs", type=int, default=1 if platform.machine() == "aarch64" else 2)
    parser.add_argument("--sanitize", action="store_true")
    args = parser.parse_args()
    if args.jobs < 1:
        parser.error("--jobs must be positive")
    build(args.profile, args.jobs, args.sanitize)
