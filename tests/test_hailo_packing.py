"""Packing arithmetic and cache checks; these are not hardware evidence."""

from dataclasses import replace

import numpy as np
import pytest

from ntru_pytorch import PRESETS, Parameters, PublicKey, RandomSource, TorchNTRU
from ntru_pytorch.hailo import ArtifactError, BinaryScheme, SoftwareHailoEncryptor
from ntru_pytorch.hailo.proof import affine_lookup, decode_affine, decode_affine_lookup


def scheme(n=107):
    params = PRESETS["moderate"] if n == 107 else Parameters(17, 3, 256, 3, 2, 1)
    return BinaryScheme(PublicKey(params, tuple((19 * i + 7) % params.q for i in range(n))))


def independent(s, polynomial):
    linear = np.convolve(np.asarray(polynomial, dtype=np.int64),
                         np.asarray(s.key.h, dtype=np.int64) % s.key.params.q)
    product = linear[:s.n].copy()
    product[:s.n - 1] += linear[s.n:]
    return product


@pytest.mark.parametrize("n", [17, 107])
def test_active_boundaries_and_mixed_batches(n):
    s = scheme(n)
    probes = [(0,) * n, (1,) * n, (-1,) * n, tuple((-1)**i for i in range(n))]
    for count in (1, 15, 16, 17):
        for sign in (1, -1):
            indices = set(np.linspace(0, n - 1, count, dtype=int))
            probes.append(tuple(sign if i in indices else 0 for i in range(n)))
    batch = s.pack_batch(probes, "active16-v1")
    s.validate_frames(batch.frames)
    assert len(batch.frames) == sum(max(1, (r.count(1) + 15)//16 + (r.count(-1) + 15)//16)
                                   for r in probes)
    assert not batch.frames[0].any()
    assert not batch.frames.flags.writeable
    assert not batch.owners.flags.writeable
    products = s.reconstruct_batch(s.software(batch.frames), batch)
    old = s.pack(probes)
    assert np.array_equal(products, s.reconstruct(s.software(old), len(probes)))
    for r, product in zip(probes, products, strict=True):
        assert np.array_equal(product, independent(s, r))


def test_thousand_active_polynomials_and_normal_frame_count():
    s = scheme()
    source = RandomSource(39)
    for start in range(0, 1000, 10):
        probes = [source.ternary(s.n, 5, 5) for _ in range(10)]
        # Also exercise dense and uneven signs beyond the normal r distribution.
        if start % 20 == 0:
            probes[0] = tuple(source.rng.choice((-1, 0, 1)) for _ in range(s.n))
        batch = s.pack_batch(probes, "active16-v1")
        s.validate_frames(batch.frames)
        products = s.reconstruct_batch(s.software(batch.frames), batch)
        for r, product in zip(probes, products, strict=True):
            assert np.array_equal(product, independent(s, r))
    r = source.ternary(s.n, 5, 5)
    assert len(s.pack_batch([r], "position16-v1").frames) == 14
    assert len(s.pack_batch([r], "active16-v1").frames) == 2
    for mode in ("position16-v1", "active16-v1"):
        engine = SoftwareHailoEncryptor(s.key, packing=mode)
        assert engine.encrypt_poly(s.key, [1, -1], r=r) == TorchNTRU().encrypt_poly(
            s.key, [1, -1], r=r
        )


def test_bad_maps_and_outputs_are_rejected():
    s = scheme()
    batch = s.pack_batch([[1, -1]], "active16-v1")
    output = s.software(batch.frames)
    for bad in (replace(batch, owners=np.array([-1, 0])),
                replace(batch, owners=np.array([0, 1])),
                replace(batch, signs=np.array([0, 1])),
                replace(batch, signs=np.array([1.0, -1.0])),
                replace(batch, count=0)):
        with pytest.raises(ValueError, match="ownership"):
            s.reconstruct_batch(output, bad)
    for bad in (output[:-1], output.astype(float), np.full_like(output, 17)):
        with pytest.raises(ValueError):
            s.reconstruct_batch(bad, batch)
    with pytest.raises(ValueError):
        s.pack_batch([[2]], "active16-v1")
    with pytest.raises(ValueError):
        s.pack_batch([[1]], "unknown")
    with pytest.raises(ValueError):
        s.pack_batch([], "active16-v1")


def test_cached_affine_table_matches_exact_uncached_decoder():
    spec = {"output": {"scale": "1/8", "zero_point": "0"}}
    lookup = affine_lookup(spec)
    assert lookup.shape == (256,) and not lookup.flags.writeable
    codes = np.arange(129, dtype=np.uint8).reshape(1, 1, 1, -1)
    assert np.array_equal(decode_affine(codes, spec), decode_affine_lookup(codes, lookup))
    for decode in (lambda a: decode_affine(a, spec), lambda a: decode_affine_lookup(a, lookup)):
        with pytest.raises(ArtifactError, match="outside"):
            decode(np.array([255], dtype=np.uint8))
        with pytest.raises(ArtifactError, match="UINT8"):
            decode(codes.astype(float))
