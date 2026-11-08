"""Cross-profile KEM interoperability; each profile runs in its own process."""

import argparse
import json
import subprocess
import sys
from pathlib import Path

CODE = """
import json,sys
from pathlib import Path
from ntru_pytorch.oqs import OqsKem
peer=json.loads(Path(sys.argv[2]).read_text())
with OqsKem(sys.argv[1]) as kem:
    assert kem.decapsulate(bytes.fromhex(peer['sk']),bytes.fromhex(peer['ct'])).hex()==peer['ss']
    ct,ss=kem.encapsulate(bytes.fromhex(peer['pk']))
    result=dict(ct=ct.hex(),ss=ss.hex())
    Path(sys.argv[3]).write_text(json.dumps(result))
"""
VERIFY = """
import json,sys
from pathlib import Path
from ntru_pytorch.oqs import OqsKem
own=json.loads(Path(sys.argv[2]).read_text())
peer=json.loads(Path(sys.argv[3]).read_text())
with OqsKem(sys.argv[1]) as kem:
    assert kem.decapsulate(bytes.fromhex(own['sk']),bytes.fromhex(peer['ct'])).hex()==peer['ss']
"""


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--oqs-root", required=True)
    parser.add_argument("--qualification-root", required=True)
    args = parser.parse_args()
    root = Path(args.oqs_root).resolve()
    target = Path(args.qualification_root).resolve()
    output = target / "interop"
    output.mkdir(exist_ok=False)
    for profile, peer in [("portable", "native"), ("native", "portable")]:
        subprocess.run(
            [
                sys.executable,
                "-c",
                CODE,
                str(root / profile),
                str(target / ("check-" + peer) / "interop.json"),
                str(output / (profile + ".json")),
            ],
            check=True,
        )
    for profile, peer in [("portable", "native"), ("native", "portable")]:
        subprocess.run(
            [
                sys.executable,
                "-c",
                VERIFY,
                str(root / profile),
                str(target / ("check-" + profile) / "interop.json"),
                str(output / (peer + ".json")),
            ],
            check=True,
        )
    (output / "report.json").write_text(
        json.dumps(
            dict(
                passed=True, portable_to_native=True, native_to_portable=True, hardware_tested=False
            ),
            indent=2,
        )
        + "\n"
    )
    print("Portable/native KEM interoperability passed")


if __name__ == "__main__":
    main()
