from sympy import Poly, symbols
from sympy.polys.domains import ZZ

from .codec import MessageAPI
from .models import coefficients
from .randomness import RandomSource

x = symbols("x")


def poly(a):
    return Poly(dict(enumerate(a)), x, domain=ZZ)


def vector(a, n):
    return tuple(int(a.nth(i)) for i in range(n))


def center(a, q):
    return tuple(v % q - q if 2 * (v % q) >= q else v % q for v in a)


def emit(trace, name, a):
    if trace is not None:
        trace(name, tuple(a))


def multiply(a, b, n):
    return vector((poly(a) * poly(b)) % Poly(x**n - 1, x), n)


class SympyNTRU(MessageAPI):
    def encrypt_poly(self, key, message, *, r=None, trace=None):
        p = key.params
        m = coefficients(message, p.N)
        if any(abs(v) * 2 > p.p for v in m):
            raise ValueError("Message outside centered p range")
        r = RandomSource().ternary(p.N, p.d, p.d) if r is None else coefficients(r, p.N)
        if any(v not in (-1, 0, 1) for v in r):
            raise ValueError("r must be ternary")
        raw = vector(poly(r) * poly(key.h), 2 * p.N - 1)
        emit(trace, "product", raw)
        reduced = vector(poly(raw) % Poly(x**p.N - 1, x), p.N)
        emit(trace, "cyclic", reduced)
        result = center(tuple(a + b for a, b in zip(reduced, m, strict=True)), p.q)
        emit(trace, "ciphertext", result)
        return result

    def decrypt_poly(self, key, cipher, *, trace=None):
        p = key.params
        if len(cipher) != p.N:
            raise ValueError("Ciphertext length must equal N")
        e = coefficients(cipher, p.N)
        a = multiply(key.f, e, p.N)
        emit(trace, "product", a)
        a = center(a, p.q)
        emit(trace, "centered", a)
        b = center(a, p.p) if p.p == 3 else tuple(v % p.p for v in a)
        result = multiply(key.fp, b, p.N)
        result = center(result, p.p) if p.p == 3 else tuple(v % p.p for v in result)
        emit(trace, "message", result)
        return result
