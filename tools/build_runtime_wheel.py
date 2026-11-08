"""Build a lean wheel from the same sources; default desktop packaging stays unchanged."""

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import tempfile
import tomllib
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    source = Path(__file__).resolve().parents[1]
    target = Path(args.output).resolve()
    target.mkdir(parents=True, exist_ok=False)
    original = tomllib.loads((source / "pyproject.toml").read_text())
    project = original["project"]
    if "+" in project["version"]:
        raise ValueError("Expected a release version without a local suffix")
    version = project["version"] + "+edge"
    with tempfile.TemporaryDirectory(prefix="ntru-edge-build-") as staging:
        root = Path(staging)
        shutil.copyfile(source / "LICENSE", root / "LICENSE")
        shutil.copytree(
            source / "src/ntru_pytorch",
            root / "src/ntru_pytorch",
            ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
        )
        # JSON string/array syntax here is also valid TOML for these metadata fields.
        lines = [
            "[build-system]",
            'requires = ["setuptools>=68"]',
            'build-backend = "setuptools.build_meta"',
            "",
            "[project]",
        ]
        for key, value in dict(
            name=project["name"],
            version=version,
            description=project["description"],
            **{
                # This Hailo profile keeps NumPy 1.x, which supports Python through 3.12.
                "requires-python": project["requires-python"] + ",<3.13",
                "dependencies": project["optional-dependencies"]["hailo-runtime"],
            },
        ).items():
            lines.append(f"{key} = {json.dumps(value)}")
        lines += [
            "",
            "[project.optional-dependencies]",
            "desktop = " + json.dumps(project["dependencies"]),
            "export = " + json.dumps(project["optional-dependencies"]["hailo-export"]),
            "",
            "[project.scripts]",
        ]
        lines += [f"{name} = {json.dumps(value)}" for name, value in project["scripts"].items()]
        lines += ["", "[tool.setuptools.packages.find]", 'where = ["src"]']
        (root / "pyproject.toml").write_text("\n".join(lines) + "\n")
        subprocess.run(
            [
                sys.executable,
                "-m",
                "pip",
                "wheel",
                "--no-deps",
                "--no-build-isolation",
                "--no-cache-dir",
                "--wheel-dir",
                str(target),
                str(root),
            ],
            check=True,
        )
    manifest = {
        "profile": "edge-runtime",
        "version": version,
        "base_pyproject_sha256": hashlib.sha256(
            (source / "pyproject.toml").read_bytes()
        ).hexdigest(),
        "source_files": {
            str(p.relative_to(source)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted((source / "src/ntru_pytorch").rglob("*.py"))
        },
        "wheels": {
            p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in target.glob("*.whl")
        },
    }
    (target / "build-manifest.json").write_text(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
