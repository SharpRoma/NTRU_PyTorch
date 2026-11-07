from sympy import GF, Poly, invert, isprime
from sympy.polys.polyerrors import NotInvertible

from .models import PrivateKey, PublicKey
from .randomness import RandomSource
from .sympy_backend import center, multiply, poly, vector, x


def inverse(a, n, q):
    ideal = Poly(x**n - 1, x)
    if isprime(q):
        return vector(invert(poly(a), ideal, domain=GF(q)), n)
    if q & (q - 1):
        raise ValueError("Only prime or power-of-two moduli supported")
    inv = poly(vector(invert(poly(a), ideal, domain=GF(2)), n))
    modulus = 2
    while modulus < q:
        modulus = min(modulus * modulus, q)
        inv = ((2 * inv - poly(a) * inv * inv) % ideal).trunc(modulus)
    return vector(inv, n)


def validate_keys(public, private):
    p = private.params
    if (public.params.N, public.params.p, public.params.q) != (p.N, p.p, p.q):
        raise ValueError("Parameter mismatch")
    identity = (1,) + (0,) * (p.N - 1)
    for inv, q in ((private.fp, p.p), (private.fq, p.q)):
        if tuple(v % q for v in multiply(private.f, inv, p.N)) != identity:
            raise ValueError("Invalid inverse")
    h = center(tuple(p.p * v for v in multiply(private.fq, private.g, p.N)), p.q)
    if tuple(v % p.q for v in public.h) != tuple(v % p.q for v in h):
        raise ValueError("Public key mismatch")


def generate_keys(params, *, random_source=None, max_attempts=100):
    if not isprime(params.p) or (not isprime(params.q) and params.q & (params.q - 1)):
        raise ValueError("Unsupported coefficient modulus")
    if max_attempts < 1:
        raise ValueError("max_attempts must be positive")
    rs = RandomSource() if random_source is None else random_source
    for _ in range(max_attempts):
        f = rs.ternary(params.N, params.df, params.df - 1)
        try:
            fp = inverse(f, params.N, params.p)
            fq = inverse(f, params.N, params.q)
        except NotInvertible:
            continue
        g = rs.ternary(params.N, params.dg, params.dg)
        private = PrivateKey(params, f, fp, fq, g)
        public = PublicKey(
            params, center(tuple(params.p * v for v in multiply(fq, g, params.N)), params.q)
        )
        validate_keys(public, private)
        return public, private
    raise RuntimeError("Key generation exhausted max_attempts")
