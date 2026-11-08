"""Verified explicit-path loading, with no dependency on NumPy or ML SDKs."""

import ctypes as C
import hashlib
import json
import platform
import threading
from pathlib import Path

COMMIT = "5a1a854b0dc9f2141bdc771c555ee60c37950183"
_kem_binding = None
_lock = threading.RLock()


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1048576), b""):
            h.update(chunk)
    return h.hexdigest()


def verify(build_dir):
    root = Path(build_dir).resolve(strict=True)
    data = json.loads((root / "manifest.json").read_text())
    conf = data["configuration"]
    if (conf["version"], conf["commit"], conf["bridge_abi"]) != ("0.16.0", COMMIT, 1):
        raise ValueError("Unsupported liboqs build version/commit/ABI")
    if conf["architecture"] != platform.machine() or platform.system() != "Linux":
        raise ValueError("Build architecture does not match this Linux host")
    if conf["sanitizer"]:
        raise ValueError("Use the standalone smoke executable for sanitizer builds")
    if conf["implementation"] not in ("clean", "avx2"):
        raise ValueError("Unknown implementation")
    if conf["implementation"] == "avx2":
        flags = Path("/proc/cpuinfo").read_text().split("flags", 1)[1].split("\n", 1)[0].split()
        if not {"avx2", "bmi2"} <= set(flags):
            raise ValueError("AVX2/BMI2 unavailable")
    for rel, expected in data["files"].items():
        path = root / rel
        if not path.resolve().is_relative_to(root) or digest(path) != expected:
            raise ValueError("Build checksum/path mismatch: " + rel)
    for rel in data["libraries"].values():
        if rel not in data["files"]:
            raise ValueError("Unhashed library path")
        header = (root / rel).read_bytes()[:20]
        machine = 62 if platform.machine() == "x86_64" else 183
        if header[:6] != b"\x7fELF\x02\x01" or int.from_bytes(header[18:20], "little") != machine:
            raise ValueError("Invalid ELF architecture")
    return root, data


def signature(lib, name, result, *args):
    fn = getattr(lib, name)
    fn.restype, fn.argtypes = result, list(args)
    return fn


def load(build_dir, kind):
    global _kem_binding
    root, data = verify(build_dir)
    with _lock:
        if kind == "kem" and _kem_binding is not None and _kem_binding != root:
            raise RuntimeError("One liboqs profile per process: use a separate worker")
        lib = C.CDLL(str(root / data["libraries"][kind]), mode=C.RTLD_LOCAL)
        prefix = "nr" if kind == "kem" else "nr_rq"
        if signature(lib, prefix + "_abi", C.c_int)() != 1:
            raise ValueError("Bridge ABI mismatch")
        impl = signature(lib, prefix + "_implementation", C.c_char_p)().decode()
        if impl != data["configuration"]["implementation"]:
            raise ValueError("Bridge implementation mismatch")
        if kind == "kem":
            # A successfully dlopened DSO stays loaded even when a context is closed.
            _kem_binding = root
            path = signature(lib, "nr_oqs_path", C.c_char_p)().decode()
            if Path(path).resolve() != (root / data["libraries"]["oqs"]).resolve():
                raise ValueError("Unexpected liboqs dependency (SONAME collision)")
            version = signature(lib, "nr_oqs_version", C.c_char_p)().decode()
            if version != "0.16.0":
                raise ValueError("Runtime liboqs version mismatch")
    return lib, root, data
