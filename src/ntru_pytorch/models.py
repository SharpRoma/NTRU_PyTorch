from dataclasses import dataclass
from math import gcd
from numbers import Integral


def coefficients(values, n):
    values = tuple(values)
    if len(values) > n or any(isinstance(v, bool) or not isinstance(v, Integral) for v in values):
        raise ValueError("Expected at most N integer coefficients")
    return tuple(int(v) for v in values) + (0,) * (n - len(values))


@dataclass(frozen=True)
class Parameters:
    N: int = 107
    p: int = 3
    q: int = 64
    df: int = 15
    dg: int = 12
    d: int = 5

    def __post_init__(self):
        vals = (self.N, self.p, self.q, self.df, self.dg, self.d)
        if any(type(v) is not int for v in vals):
            raise ValueError("Parameters must be integers")
        if self.q > 2**31 - 1:
            raise ValueError("Modulus exceeds supported INT64 safety envelope")
        if self.N < 2 or self.p < 2 or self.q < 8 * self.p or gcd(self.p, self.q) != 1:
            raise ValueError("Invalid ring parameters")
        if (
            self.df < 1
            or self.dg < 0
            or self.d < 0
            or max(2 * self.df - 1, 2 * self.dg, 2 * self.d) > self.N
        ):
            raise ValueError("Invalid polynomial weights")


@dataclass(frozen=True)
class PublicKey:
    params: Parameters
    h: tuple

    def __post_init__(self):
        if len(self.h) != self.params.N:
            raise ValueError("Key length must equal N")
        object.__setattr__(self, "h", coefficients(self.h, self.params.N))


@dataclass(frozen=True)
class PrivateKey:
    params: Parameters
    f: tuple
    fp: tuple
    fq: tuple
    g: tuple

    def __post_init__(self):
        for name in ("f", "fp", "fq", "g"):
            v = getattr(self, name)
            if len(v) != self.params.N:
                raise ValueError("Key length must equal N")
            object.__setattr__(self, name, coefficients(v, self.params.N))


@dataclass(frozen=True)
class Ciphertext:
    params: Parameters
    blocks: tuple

    def __post_init__(self):
        blocks = tuple(tuple(b) for b in self.blocks)
        if any(len(b) != self.params.N for b in blocks):
            raise ValueError("Invalid block length")
        object.__setattr__(self, "blocks", tuple(coefficients(b, self.params.N) for b in blocks))


PRESETS = {
    name: Parameters(*v)
    for name, v in {
        "moderate": (107, 3, 64, 15, 12, 5),
        "high": (167, 3, 128, 61, 20, 18),
        "highest": (503, 3, 256, 216, 72, 55),
        "dead": (701, 3, 8192, 216, 72, 55),
        "dead2": (821, 3, 4096, 216, 72, 55),
    }.items()
}
