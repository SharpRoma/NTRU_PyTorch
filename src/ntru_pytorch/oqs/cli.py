"""Explicit commands for externally built liboqs artifacts."""

import json
from pathlib import Path

from .loader import verify


def run(args):
    root = Path(args.oqs_root).resolve()
    build = Path(args.oqs_build).resolve() if args.oqs_build else root / args.oqs_profile
    if args.command == "oqs-info":
        path, data = verify(build)
        from .kem import OqsKem

        with OqsKem(path) as kem:
            print(
                json.dumps(
                    dict(
                        build=str(path),
                        configuration=data["configuration"],
                        sizes=dict(
                            public_key=kem.public_key_bytes,
                            secret_key=kem.secret_key_bytes,
                            ciphertext=kem.ciphertext_bytes,
                            shared_secret=kem.shared_secret_bytes,
                        ),
                    ),
                    indent=2,
                )
            )
    elif args.command == "oqs-check":
        from .qualification import check

        report = check(
            build,
            args.output,
            cases=args.cases,
            seed=args.seed,
            public_key=args.public_key,
            private_key=args.private_key,
        )
        print(json.dumps(report, indent=2))
    else:
        from .benchmark import orchestrate

        if args.command == "oqs-arith-benchmark" and (not args.public_key or not args.private_key):
            raise ValueError("Arithmetic benchmark requires --public-key and --private-key")
        orchestrate(args, "kem" if args.command == "oqs-kem-benchmark" else "arith")
