"""Public OQS_KEM API, with C-owned ABI layout and Python-owned byte values."""

import ctypes as C

from .loader import load, signature

P = C.c_void_p
Z = C.c_size_t


def buffer(value, length):
    if not isinstance(value, bytes):
        raise TypeError("Expected bytes")
    if len(value) != length:
        raise ValueError(f"Expected exactly {length} bytes")
    return C.create_string_buffer(value, length)


class OqsKem:
    def __init__(self, build_dir, algorithm="NTRU-HRSS-701"):
        self._handle = None
        if algorithm != "NTRU-HRSS-701":
            raise ValueError("Only NTRU-HRSS-701 is supported")
        self._handle = None
        self.lib, self.build_dir, self.manifest = load(build_dir, "kem")
        for name, ret, args in [
            ("nr_kem_new", P, ()),
            ("nr_kem_free", None, (P,)),
            ("nr_kem_sizes", C.c_int, (P, C.POINTER(Z))),
            ("nr_cleanse", None, (P, Z)),
            ("nr_keypair", C.c_int, (P, P, Z, P, Z)),
            ("nr_encaps", C.c_int, (P, P, Z, P, Z, P, Z)),
            ("nr_decaps", C.c_int, (P, P, Z, P, Z, P, Z)),
            (
                "nr_kem_window",
                C.c_int,
                (P, C.c_int, Z, P, P, P, P, C.POINTER(C.c_uint64), C.POINTER(C.c_uint64)),
            ),
        ]:
            signature(self.lib, name, ret, *args)
        self._handle = self.lib.nr_kem_new()
        if not self._handle:
            raise RuntimeError("NTRU-HRSS-701 unavailable")
        sizes = (Z * 4)()
        if self.lib.nr_kem_sizes(self._handle, sizes) or tuple(sizes) != (1138, 1450, 1138, 32):
            self.close()
            raise ValueError("Unexpected KEM buffer sizes")
        (
            self.public_key_bytes,
            self.secret_key_bytes,
            self.ciphertext_bytes,
            self.shared_secret_bytes,
        ) = map(int, sizes)

    def _open(self):
        if self._handle is None:
            raise RuntimeError("KEM context is closed")

    def keypair(self):
        self._open()
        pk = C.create_string_buffer(self.public_key_bytes)
        sk = C.create_string_buffer(self.secret_key_bytes)
        try:
            if self.lib.nr_keypair(self._handle, pk, len(pk), sk, len(sk)):
                raise RuntimeError("OQS keypair failed")
            return pk.raw, sk.raw
        finally:
            self.lib.nr_cleanse(sk, len(sk))

    def encapsulate(self, public_key):
        self._open()
        pk = buffer(public_key, self.public_key_bytes)
        ct = C.create_string_buffer(self.ciphertext_bytes)
        ss = C.create_string_buffer(self.shared_secret_bytes)
        try:
            if self.lib.nr_encaps(self._handle, ct, len(ct), ss, len(ss), pk, len(pk)):
                raise RuntimeError("OQS encapsulation failed")
            return ct.raw, ss.raw
        finally:
            self.lib.nr_cleanse(ss, len(ss))

    def decapsulate(self, secret_key, ciphertext):
        self._open()
        sk = buffer(secret_key, self.secret_key_bytes)
        try:
            ct = buffer(ciphertext, self.ciphertext_bytes)
            ss = C.create_string_buffer(self.shared_secret_bytes)
            try:
                if self.lib.nr_decaps(self._handle, ss, len(ss), ct, len(ct), sk, len(sk)):
                    raise RuntimeError("OQS decapsulation failed")
                return ss.raw
            finally:
                self.lib.nr_cleanse(ss, len(ss))
        finally:
            self.lib.nr_cleanse(sk, len(sk))

    def close(self):
        if self._handle is not None:
            self.lib.nr_kem_free(self._handle)
            self._handle = None

    def __enter__(self):
        self._open()
        return self

    def __exit__(self, *exc):
        self.close()

    def __del__(self):
        self.close()
