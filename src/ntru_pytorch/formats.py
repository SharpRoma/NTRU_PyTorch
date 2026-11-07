"""Versioned JSON and explicit read-only legacy adapters."""

import json
from dataclasses import asdict
from pathlib import Path

from .models import Ciphertext, Parameters, PrivateKey, PublicKey


def dumps(obj):
    if type(obj) not in (PublicKey, PrivateKey, Ciphertext):
        raise TypeError("Unsupported model")
    return json.dumps({"version": 1, "kind": type(obj).__name__, **asdict(obj)}, ensure_ascii=False)


def loads(text):
    data = json.loads(text)
    if type(data) is not dict or type(data.get("version")) is not int or data.pop("version") != 1:
        raise ValueError("Unsupported document version")
    classes = {"PublicKey": PublicKey, "PrivateKey": PrivateKey, "Ciphertext": Ciphertext}
    kind = data.pop("kind", None)
    if kind not in classes:
        raise ValueError("Unknown document kind")
    try:
        data["params"] = Parameters(**data["params"])
        return classes[kind](**data)
    except (TypeError, KeyError) as e:
        raise ValueError("Malformed document") from e


def save(obj, path):
    Path(path).write_text(dumps(obj), encoding="utf-8")


def load(path):
    return loads(Path(path).read_text(encoding="utf-8"))


def _legacy(path):
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    fields = {}
    rows = []
    for line in lines:
        line = line.removeprefix("#").strip()
        if ":::" in line:
            k, v = line.split(":::", 1)
            fields[k.strip()] = v.strip()
        elif line.strip():
            rows.append(tuple(int(v) for v in line.split()))
    return fields, rows


def read_legacy_public(path):
    f, _ = _legacy(path)
    p = Parameters(N=int(f["N"]), p=int(f["p"]), q=int(f["q"]), df=1, dg=0, d=int(f["d"]))
    return PublicKey(p, tuple(int(v) for v in f["h"].split()))


def read_legacy_private(path):
    f, rows = _legacy(path)
    if len(rows) != 4:
        raise ValueError("Expected four private polynomials")
    p = Parameters(
        N=int(f["N"]), p=int(f["p"]), q=int(f["q"]), df=int(f["df"]), dg=int(f["dg"]), d=int(f["d"])
    )
    return PrivateKey(p, *rows)


def read_legacy_ciphertext(text, params):
    values = tuple(int(v) for v in text.split())
    if len(values) % params.N:
        raise ValueError("Incomplete legacy block")
    return Ciphertext(
        params, tuple(values[i : i + params.N] for i in range(0, len(values), params.N))
    )


def decrypt_legacy_bytes(backend, key, cipher):
    if (key.params.N, key.params.p, key.params.q) != (
        cipher.params.N,
        cipher.params.p,
        cipher.params.q,
    ):
        raise ValueError("Ring mismatch")
    bits = tuple(v for block in cipher.blocks for v in backend.decrypt_poly(key, block))
    if any(v not in (0, 1) for v in bits):
        raise ValueError("Nonbinary legacy plaintext")
    raw = bytes(sum(bits[i + j] << (7 - j) for j in range(8)) for i in range(0, len(bits) - 7, 8))
    return raw.split(b"\0", 1)[0]
