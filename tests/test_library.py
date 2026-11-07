import json
import random

import pytest

from ntru_pytorch import (
    PRESETS,
    Parameters,
    PrivateKey,
    PublicKey,
    RandomSource,
    SympyNTRU,
    TorchNTRU,
    generate_keys,
    tensor,
    validate_keys,
)
from ntru_pytorch.codec import decode, encode
from ntru_pytorch.formats import (
    decrypt_legacy_bytes,
    dumps,
    loads,
    read_legacy_ciphertext,
    read_legacy_private,
    read_legacy_public,
)
from ntru_pytorch.models import coefficients
from ntru_pytorch.sympy_backend import center, multiply


@pytest.fixture(scope="module")
def keys():
    return generate_keys(Parameters(17, 3, 256, 3, 2, 1), random_source=RandomSource(42))


@pytest.mark.parametrize(
    "data",
    [
        b"",
        b"a",
        b"test",
        b"a\0b\0",
        bytes(range(256)),
        "\u041f\u0440\u0438\u0432\u0435\u0442 \U0001f600".encode(),
    ]
    + [b"x" * n for n in (8, 9, 10, 16, 17, 18, 26)],
)
def test_four_directions(keys, data):
    pub, priv = keys
    cs = [
        e.encrypt_bytes(pub, data, random_source=RandomSource(3))
        for e in (SympyNTRU(), TorchNTRU())
    ]
    assert cs[0] == cs[1]
    for c in cs:
        for d in (SympyNTRU(), TorchNTRU()):
            assert d.decrypt_bytes(priv, loads(dumps(c))) == data


def test_trace(keys):
    pub, priv = keys
    results = []
    for engine in (SympyNTRU(), TorchNTRU()):
        rows = []
        c = engine.encrypt_poly(
            pub, [1, 0, 1], r=(1, -1), trace=lambda name, a, rows=rows: rows.append((name, a))
        )
        assert (
            engine.decrypt_poly(priv, c, trace=lambda name, a, rows=rows: rows.append((name, a)))
            == (1, 0, 1) + (0,) * 14
        )
        results.append(rows)
    assert results[0] == results[1]


def test_kernels():
    rng = random.Random(12)
    for n in range(1, 15):
        for _ in range(5):
            a = [rng.randint(-100, 100) for _ in range(n)]
            b = [rng.randint(-100, 100) for _ in range(n)]
            reference = [sum(a[j] * b[(i - j) % n] for j in range(n)) for i in range(n)]
            ta, tb = tensor.tensor(a), tensor.tensor(b)
            assert tensor.circulant(ta, tb).tolist() == reference
            assert tensor.reduce(tensor.linear(ta, tb), n).tolist() == reference
            assert list(multiply(a, b, n)) == reference
    assert tensor.reduce(tensor.tensor([1, 2]), 5).tolist() == [1, 2, 0, 0, 0]
    assert tensor.linear(tensor.tensor([0]), tensor.tensor([0])).tolist() == [0]
    for q in (3, 4, 64):
        a = list(range(-q, q + 1))
        assert tensor.center(tensor.tensor(a), q).tolist() == list(center(a, q))


def test_validation(keys):
    pub, priv = keys
    validate_keys(pub, priv)
    for obj in (*keys, SympyNTRU().encrypt_bytes(pub, b"")):
        assert loads(dumps(obj)) == obj
    with pytest.raises(ValueError):
        Parameters(N=4, df=3)
    with pytest.raises(ValueError):
        generate_keys(Parameters(17, 9, 256, 3, 2, 1))
    with pytest.raises(ValueError):
        PublicKey(pub.params, (1,))
    with pytest.raises(ValueError):
        decode(((2,) * 64,))
    with pytest.raises(ValueError):
        decode(((0,) * 63,))
    with pytest.raises(ValueError):
        loads('{"version":2}')
    with pytest.raises(ValueError):
        read_legacy_ciphertext("1 x", pub.params)
    with pytest.raises(ValueError):
        read_legacy_ciphertext("1", pub.params)
    with pytest.raises(ValueError):
        validate_keys(PublicKey(pub.params, (0,) * 17), priv)
    with pytest.raises(OverflowError):
        tensor.linear(tensor.tensor([2**62] * 2), tensor.tensor([2, 2]))
    for engine in (SympyNTRU(), TorchNTRU()):
        with pytest.raises(ValueError):
            engine.encrypt_poly(pub, [0] * 18)
        with pytest.raises(ValueError):
            engine.decrypt_poly(priv, [0])
        with pytest.raises(ValueError):
            engine.encrypt_poly(pub, [1.5])
        # Polynomial decryption must not silently turn -1 into a bit.
        c = engine.encrypt_poly(pub, [-1], r=[0] * 17)
        assert engine.decrypt_poly(priv, c)[0] == -1


def test_legacy_synthetic(tmp_path):
    # Deliberately trivial artificial key; never used as a security parameter set.
    n = 107
    p = Parameters()
    f = (1,) + (0,) * (n - 1)
    g = (1, -1) + (0,) * (n - 2)
    priv = PrivateKey(p, f, f, f, g)
    pub = PublicKey(p, (3, -3) + (0,) * (n - 2))
    private = tmp_path / "synthetic.priv"
    public = tmp_path / "synthetic.pub"
    public.write_text(
        "# p ::: 3\n# q ::: 64\n# N ::: 107\n# d ::: 5\n# h ::: " + " ".join(map(str, pub.h)) + " "
    )
    private.write_text(
        "# p ::: 3\n# q ::: 64\n# N ::: 107\n# df ::: 15\n# dg ::: 12\n# d ::: 5\n# f/fp/fq/g :::\n"
        + "\n".join(" ".join(map(str, v)) for v in (f, f, f, g))
    )
    assert read_legacy_private(private) == priv
    legacy_pub = read_legacy_public(public)
    assert legacy_pub.h == pub.h
    validate_keys(legacy_pub, priv)
    bits = tuple((v >> i) & 1 for v in b"test" for i in range(7, -1, -1))
    c = SympyNTRU().encrypt_poly(pub, bits, r=(1, -1))
    legacy = read_legacy_ciphertext(" ".join(map(str, c)), p)
    for engine in (SympyNTRU(), TorchNTRU()):
        assert decrypt_legacy_bytes(engine, priv, legacy) == b"test"
        assert engine.encrypt_poly(legacy_pub, bits, r=(1, -1)) == c


@pytest.mark.slow
@pytest.mark.parametrize("name", list(PRESETS))
def test_presets(name):
    pub, priv = generate_keys(PRESETS[name], random_source=RandomSource(42))
    validate_keys(pub, priv)
    cs = [
        e.encrypt_bytes(pub, b"A", random_source=RandomSource(4))
        for e in (SympyNTRU(), TorchNTRU())
    ]
    assert cs[0] == cs[1]
    for e in (SympyNTRU(), TorchNTRU()):
        for c in cs:
            assert e.decrypt_bytes(priv, c) == b"A"


def test_captured_original_vector():
    from pathlib import Path

    v = json.loads((Path(__file__).parent / "fixtures/legacy_vector.json").read_text())
    pub = PublicKey(Parameters(), tuple(v["h"]))
    for engine in (SympyNTRU(), TorchNTRU()):
        assert engine.encrypt_poly(pub, v["message"], r=v["r"]) == tuple(v["cipher"])


def test_corrupt_formats_and_padding(keys):
    pub, _ = keys
    for text in ("{}", "[]", '{"version":true}', '{"version":1,"kind":"Unknown"}'):
        with pytest.raises(ValueError):
            loads(text)
    blocks = encode(b"a", 17)
    broken = list(blocks[-1])
    broken[-1] = 1
    with pytest.raises(ValueError):
        decode(blocks[:-1] + (tuple(broken),))
    with pytest.raises(ValueError):
        generate_keys(pub.params, max_attempts=0)
    with pytest.raises(ValueError):
        coefficients([True], 17)


def test_import_no_files(tmp_path):
    import subprocess
    import sys

    subprocess.run(
        [sys.executable, "-B", "-c", "import ntru_pytorch; import ntru_pytorch.cli"],
        cwd=tmp_path,
        check=True,
    )
    assert not list(tmp_path.iterdir())


def test_benchmark_export(tmp_path):
    from ntru_pytorch.research import benchmark

    output = tmp_path / "measurement"
    result = benchmark(Parameters(17, 3, 256, 3, 2, 1), output, runs=2, warmup=0)
    report = json.loads(output.with_suffix(".json").read_text())
    assert report["metadata"]["threads"] > 0
    assert len(result) == 8
    assert all(len(v["samples_ns"]) == 2 for v in result.values())
    assert output.with_suffix(".csv").read_text().startswith("case,sample,time_ns")
