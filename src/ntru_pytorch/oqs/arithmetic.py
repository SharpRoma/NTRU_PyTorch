"""Upstream Rq cyclic multiplication in the project's encryption protocol."""

import ctypes as C
from time import perf_counter_ns

import numpy as np

from ..array_api import encode_array, normalize
from ..codec import MessageAPI
from ..models import Ciphertext, PublicKey
from ..randomness import RandomSource
from .loader import load, signature


class OqsArithmeticEncryptor(MessageAPI):
    def __init__(self, build_dir):
        self._handle = None
        self._key = None
        self.lib, self.build_dir, self.manifest = load(build_dir, "arithmetic")
        for name, ret, args in [
            ("nr_rq_n", C.c_int, ()),
            ("nr_rq_q", C.c_int, ()),
            ("nr_rq_prepared_bytes", C.c_size_t, ()),
            ("nr_rq_prepare", C.c_void_p, (C.c_void_p, C.c_size_t)),
            ("nr_rq_free", None, (C.c_void_p,)),
            ("nr_rq_batch", C.c_int, (C.c_void_p, C.c_void_p, C.c_size_t, C.c_void_p, C.c_size_t)),
            (
                "nr_rq_window",
                C.c_int,
                (
                    C.c_void_p,
                    C.c_void_p,
                    C.c_size_t,
                    C.c_void_p,
                    C.c_size_t,
                    C.c_size_t,
                    C.POINTER(C.c_uint64),
                    C.POINTER(C.c_uint64),
                ),
            ),
        ]:
            signature(self.lib, name, ret, *args)
        if (self.lib.nr_rq_n(), self.lib.nr_rq_q()) != (701, 8192):
            raise ValueError("Wrong arithmetic parameters")
        self.preparation = {}
        self.last_timings = {}
        self._closed = False

    def prepare(self, key):
        if self._closed:
            raise RuntimeError("Arithmetic context is closed")
        if not isinstance(key, PublicKey):
            raise TypeError("Expected PublicKey")
        if (key.params.N, key.params.p, key.params.q) != (701, 3, 8192):
            raise ValueError("OQS arithmetic supports only N=701,p=3,q=8192")
        if key is self._key or key == self._key:
            return dict(self.preparation)
        started = perf_counter_ns()
        # Python modulo precedes narrowing, including arbitrarily large public coefficients.
        h = np.asarray([v % 8192 for v in key.h], dtype=np.uint16)
        handle = self.lib.nr_rq_prepare(h.ctypes.data, len(h))
        if not handle:
            raise RuntimeError("Native key preparation failed")
        if self._handle:
            self.lib.nr_rq_free(self._handle)
        self._handle, self._key = handle, key
        self.preparation = dict(
            preparation_ns=perf_counter_ns() - started,
            prepared_bytes=self.lib.nr_rq_prepared_bytes(),
            implementation=self.manifest["configuration"]["implementation"],
        )
        return dict(self.preparation)

    def _products(self, rs):
        started = perf_counter_ns()
        inputs = np.ascontiguousarray(rs % 8192, dtype=np.uint16)
        out = np.empty(inputs.shape, dtype=np.uint16)
        converted = perf_counter_ns()
        if self.lib.nr_rq_batch(
            self._handle, inputs.ctypes.data, inputs.size, out.ctypes.data, len(inputs)
        ):
            raise RuntimeError("Native Rq multiplication failed")
        called = perf_counter_ns()
        self.last_timings.update(
            buffer_conversion_ns=converted - started, ffi_native_ns=called - converted
        )
        return out

    @staticmethod
    def _trace(trace):
        if trace is not None:
            raise ValueError("Unreduced product trace is unavailable for modular OQS arithmetic")

    def _encrypt(self, key, messages, rs):
        if len(messages) != len(rs):
            raise ValueError("One r per message block required")
        products = self._products(rs)
        start = perf_counter_ns()
        out = (products.astype(np.int64) + messages) % 8192
        out = np.where(out >= 4096, out - 8192, out)
        blocks = tuple(tuple(map(int, row)) for row in out)
        self.last_timings["restoration_ns"] = perf_counter_ns() - start
        return blocks

    def encrypt_polys(self, key, messages, *, r_polys, trace=None):
        self._trace(trace)
        start = perf_counter_ns()
        self.prepare(key)
        messages = normalize(messages, 701, 1)
        rs = normalize(r_polys, 701, 1)
        ready = perf_counter_ns()
        self.last_timings = {"key_and_validation_ns": ready - start}
        blocks = self._encrypt(key, messages, rs)
        self.last_timings["polynomial_total_ns"] = perf_counter_ns() - start
        return blocks

    def encrypt_poly(self, key, message, *, r=None, trace=None):
        if r is None:
            r = RandomSource().ternary(key.params.N, key.params.d, key.params.d)
        return self.encrypt_polys(key, [message], r_polys=[r], trace=trace)[0]

    def encrypt_bytes(self, key, data, *, random_source=None, r_polys=None, trace=None):
        self._trace(trace)
        start = perf_counter_ns()
        self.prepare(key)
        prepared = perf_counter_ns()
        messages = encode_array(data, 701)
        encoded = perf_counter_ns()
        source = RandomSource() if random_source is None else random_source
        rs = normalize(
            tuple(source.ternary(701, key.params.d, key.params.d) for _ in messages)
            if r_polys is None
            else r_polys,
            701,
            1,
        )
        ready = perf_counter_ns()
        self.last_timings = dict(
            key_prepare_call_ns=prepared - start,
            message_encode_ns=encoded - prepared,
            key_and_validation_ns=ready - encoded,
        )
        blocks = self._encrypt(key, messages, rs)
        constructing = perf_counter_ns()
        result = Ciphertext(key.params, blocks)
        end = perf_counter_ns()
        self.last_timings.update(
            ciphertext_construction_ns=end - constructing, encrypt_total_ns=end - start
        )
        return result

    def close(self):
        if self._handle:
            self.lib.nr_rq_free(self._handle)
            self._handle = None
        self._key = None
        self._closed = True

    def __enter__(self):
        if self._closed:
            raise RuntimeError("Arithmetic context is closed")
        return self

    def __exit__(self, *exc):
        self.close()

    def __del__(self):
        self.close()
