"""Exact sparse-ternary NumPy CPU baseline; not a constant-time cryptographic library."""

from time import perf_counter_ns

import numpy as np

from .array_api import encode_array, normalize
from .codec import MessageAPI
from .models import Ciphertext, PublicKey, coefficients
from .randomness import RandomSource

KERNELS = ("shift-sum", "cached-shifts")
LIMIT = 2**63 - 1


def _center(array, modulus):
    residue = array % modulus
    return np.where(residue >= (modulus + 1) // 2, residue - modulus, residue)


def _cyclic(a, b):
    """Independent dense INT64 operation used for decryption, outside encryption timings."""
    n = len(a)
    if n * max(map(abs, a), default=0) * max(map(abs, b), default=0) > LIMIT:
        raise OverflowError("INT64 cyclic accumulation may overflow")
    raw = np.convolve(np.asarray(a, dtype=np.int64), np.asarray(b, dtype=np.int64))
    out = raw[:n].copy()
    out[:-1] += raw[n:]
    return out


class SparseNTRU(MessageAPI):
    """One-key prepared cache; warm calls retain all public input/output validation.

    Changing the key rebuilds the cache. Call prepare(key) explicitly to separate
    setup from steady-state timing. Both kernels accept any ternary r, not only
    the preset distribution. No float, FFT, GPU or NPU arithmetic is used.
    """

    def __init__(self, *, kernel="cached-shifts", max_prepared_bytes=64 * 1024 * 1024):
        if kernel not in KERNELS:
            raise ValueError("Unknown sparse CPU kernel")
        if type(max_prepared_bytes) is not int or max_prepared_bytes < 1:
            raise ValueError("max_prepared_bytes must be a positive integer")
        self._kernel = kernel
        self.max_prepared_bytes = max_prepared_bytes
        self._key = None
        self.preparation = {}
        self.last_timings = {}

    @property
    def kernel(self):
        return self._kernel

    def prepare(self, key):
        if not isinstance(key, PublicKey):
            raise TypeError("Expected PublicKey")
        if key is self._key or key == self._key:
            return dict(self.preparation)
        start = perf_counter_ns()
        n = key.params.N
        bound = n * max(map(abs, key.h), default=0)
        if bound > LIMIT:
            raise OverflowError("Sparse INT64 accumulation may overflow")
        required = 8 * (n + (n * n if self.kernel == "cached-shifts" else 2 * n))
        if required > self.max_prepared_bytes:
            raise ValueError(
                "Prepared key exceeds CPU memory limit; use shift-sum or a larger limit"
            )
        h = np.asarray(key.h, dtype=np.int64)
        if self.kernel == "cached-shifts":
            positions = np.arange(n)
            prepared = h[(positions[None, :] - positions[:, None]) % n]
        else:
            prepared = np.concatenate((h, h))
        h.setflags(write=False)
        prepared.setflags(write=False)
        self._h, self._prepared, self._key = h, prepared, key
        self.preparation = dict(
            kernel=self.kernel,
            preparation_ns=perf_counter_ns() - start,
            prepared_bytes=h.nbytes + prepared.nbytes,
            max_product_abs=bound,
            accumulation="INT64",
        )
        return dict(self.preparation)

    def _products(self, rs):
        result = np.zeros(rs.shape, dtype=np.int64)
        n = rs.shape[1]
        for owner, row in enumerate(rs):
            positive, negative = np.flatnonzero(row == 1), np.flatnonzero(row == -1)
            if self.kernel == "cached-shifts":
                result[owner] = self._prepared[positive].sum(axis=0, dtype=np.int64)
                result[owner] -= self._prepared[negative].sum(axis=0, dtype=np.int64)
            else:
                for position in positive:
                    result[owner] += self._prepared[n - position : 2 * n - position]
                for position in negative:
                    result[owner] -= self._prepared[n - position : 2 * n - position]
        return result

    def _encrypt_arrays(self, key, messages, rs, trace, started):
        if len(messages) != len(rs):
            raise ValueError("One r per nonempty message batch required")
        validated = perf_counter_ns()
        products = self._products(rs)
        multiplied = perf_counter_ns()
        encrypted = _center(products % key.params.q + messages, key.params.q)
        blocks = tuple(tuple(map(int, row)) for row in encrypted)
        if trace is not None:
            for r, product, block in zip(rs, products, blocks, strict=True):
                raw = np.convolve(r, self._h)
                trace("product", tuple(map(int, raw)))
                trace("cyclic", tuple(map(int, product)))
                trace("ciphertext", block)
        ended = perf_counter_ns()
        self.last_timings = dict(
            key_and_validation_ns=validated - started,
            sparse_product_ns=multiplied - validated,
            reduction_ns=ended - multiplied,
            polynomial_total_ns=ended - started,
            polynomial_blocks=len(messages),
        )
        return blocks

    def encrypt_polys(self, key, messages, *, r_polys, trace=None):
        start = perf_counter_ns()
        self.prepare(key)
        messages = normalize(messages, key.params.N, key.params.p // 2)
        rs = normalize(r_polys, key.params.N, 1)
        return self._encrypt_arrays(key, messages, rs, trace, start)

    def encrypt_poly(self, key, message, *, r=None, trace=None):
        if r is None:
            r = RandomSource().ternary(key.params.N, key.params.d, key.params.d)
        return self.encrypt_polys(key, [message], r_polys=[r], trace=trace)[0]

    def encrypt_bytes(self, key, data, *, random_source=None, r_polys=None, trace=None):
        start = perf_counter_ns()
        self.prepare(key)
        prepared = perf_counter_ns()
        messages = encode_array(data, key.params.N)
        encoded = perf_counter_ns()
        source = RandomSource() if random_source is None else random_source
        rs = (
            tuple(source.ternary(key.params.N, key.params.d, key.params.d) for _ in messages)
            if r_polys is None
            else r_polys
        )
        rs = normalize(rs, key.params.N, 1)
        blocks = self._encrypt_arrays(key, messages, rs, trace, encoded)
        constructing = perf_counter_ns()
        result = Ciphertext(key.params, blocks)
        end = perf_counter_ns()
        self.last_timings.update(
            key_prepare_call_ns=prepared - start,
            message_encode_ns=encoded - prepared,
            ciphertext_construction_ns=end - constructing,
            encrypt_total_ns=end - start,
        )
        return result

    def decrypt_poly(self, key, cipher, *, trace=None):
        p = key.params
        if len(cipher) != p.N:
            raise ValueError("Ciphertext length must equal N")
        e = coefficients(cipher, p.N)
        a = _cyclic(key.f, e)
        if trace is not None:
            trace("product", tuple(map(int, a)))
        a = _center(a, p.q)
        if trace is not None:
            trace("centered", tuple(map(int, a)))
        b = _center(a, p.p) if p.p == 3 else a % p.p
        fp = tuple(v % p.p - p.p if 2 * (v % p.p) >= p.p else v % p.p for v in key.fp)
        result = _cyclic(fp, tuple(map(int, b)))
        result = _center(result, p.p) if p.p == 3 else result % p.p
        if trace is not None:
            trace("message", tuple(map(int, result)))
        return tuple(map(int, result))
