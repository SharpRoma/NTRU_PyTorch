from . import tensor as t
from .codec import MessageAPI
from .models import coefficients
from .randomness import RandomSource


def emit(trace, name, a):
    if trace is not None:
        trace(name, tuple(a.tolist()))


class TorchNTRU(MessageAPI):
    def encrypt_poly(self, key, message, *, r=None, trace=None):
        p = key.params
        m = coefficients(message, p.N)
        if any(abs(v) * 2 > p.p for v in m):
            raise ValueError("Message outside centered p range")
        r = RandomSource().ternary(p.N, p.d, p.d) if r is None else coefficients(r, p.N)
        if any(v not in (-1, 0, 1) for v in r):
            raise ValueError("r must be ternary")
        raw = t.linear(t.tensor(r), t.tensor(key.h))
        emit(trace, "product", raw)
        reduced = t.reduce(raw, p.N)
        emit(trace, "cyclic", reduced)
        # Reduce before adding message to avoid overflowing large raw sums.
        e = t.center(t.center(reduced, p.q) + t.tensor(m), p.q)
        emit(trace, "ciphertext", e)
        return tuple(e.tolist())

    def decrypt_poly(self, key, cipher, *, trace=None):
        p = key.params
        if len(cipher) != p.N:
            raise ValueError("Ciphertext length must equal N")
        e = t.tensor(coefficients(cipher, p.N))
        a = t.circulant(t.tensor(key.f), e)
        emit(trace, "product", a)
        a = t.center(a, p.q)
        emit(trace, "centered", a)
        b = t.center(a, p.p) if p.p == 3 else a.remainder(p.p)
        fp = t.center(t.tensor(key.fp), p.p)
        result = t.circulant(fp, b)
        result = t.center(result, p.p) if p.p == 3 else result.remainder(p.p)
        emit(trace, "message", result)
        return tuple(result.tolist())
