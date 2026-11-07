"""Binary fixed-key profile. All host reconstruction uses exact INT64."""

import hashlib
import json

import numpy as np

from ..models import coefficients

CHUNK = 16
PROFILE = "binary-circulant-chunk16-v1"


def key_hash(key):
    payload = {
        "N": key.params.N,
        "p": key.params.p,
        "q": key.params.q,
        "h": [v % key.params.q for v in key.h],
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


class BinaryScheme:
    def __init__(self, key):
        self.key = key
        self.n = key.params.N
        self.bits = (key.params.q - 1).bit_length()
        self.cin = ((self.n + 15) // 16) * 16
        self.cout = ((self.n * self.bits + 15) // 16) * 16
        self.tiles = (self.n + CHUNK - 1) // CHUNK
        if self.n * (key.params.q - 1) > np.iinfo(np.int64).max:
            raise OverflowError("Reconstructed product exceeds INT64")
        h = np.array([v % key.params.q for v in key.h], dtype=np.int64)
        i = np.arange(self.n)
        circulant = h[(i[:, None] - i[None, :]) % self.n]
        self.weights = np.zeros((self.cout, self.cin), dtype=np.uint8)
        for bit in range(self.bits):
            self.weights[bit * self.n : (bit + 1) * self.n, : self.n] = (circulant >> bit) & 1

    def pack(self, polynomials):
        """NHWC frames: polynomial, sign (+ then -), tile, channel."""
        polynomials = tuple(coefficients(r, self.n) for r in polynomials)
        if any(v not in (-1, 0, 1) for r in polynomials for v in r):
            raise ValueError("r must be ternary")
        frames = np.zeros((len(polynomials), 2, self.tiles, self.cin), dtype=np.uint8)
        for b, r in enumerate(polynomials):
            for sign, value in enumerate((1, -1)):
                for tile in range(self.tiles):
                    lo, hi = tile * CHUNK, min((tile + 1) * CHUNK, self.n)
                    frames[b, sign, tile, lo:hi] = np.asarray(r[lo:hi]) == value
        return frames.reshape(-1, 1, 1, self.cin)

    def validate_frames(self, frames):
        if frames.dtype != np.uint8 or frames.ndim != 4 or frames.shape[1:] != (1, 1, self.cin):
            raise ValueError("Expected NHWC UINT8 frames")
        flat = frames.reshape(len(frames), self.cin)
        if np.any(flat > 1) or np.any(flat[:, self.n :]) or np.any(flat.sum(axis=1) > CHUNK):
            raise ValueError("Inputs must contain at most 16 binary coefficients")
        return flat

    def software(self, frames):
        flat = self.validate_frames(frames)
        out = np.zeros((len(flat), self.cout), dtype=np.uint8)
        for i, row in enumerate(flat):
            active = np.flatnonzero(row)
            if active.size:
                out[i] = self.weights[:, active].sum(axis=1, dtype=np.uint16)
        return out.reshape(-1, 1, 1, self.cout)

    def reconstruct(self, outputs, count):
        if outputs.shape != (count * 2 * self.tiles, 1, 1, self.cout):
            raise ValueError("Unexpected output shape")
        if (
            not np.issubdtype(outputs.dtype, np.integer)
            or np.any(outputs < 0)
            or np.any(outputs > CHUNK)
        ):
            raise ValueError("Partial sums must be exact integers in [0,16]")
        if np.any(outputs[..., self.n * self.bits :]):
            raise ValueError("Nonzero padded output channels")
        out = outputs.astype(np.int64).reshape(count, 2, self.tiles, self.cout)
        signed = out[:, 0].sum(axis=1) - out[:, 1].sum(axis=1)
        planes = signed[:, : self.bits * self.n].reshape(count, self.bits, self.n)
        return (planes * (1 << np.arange(self.bits))[None, :, None]).sum(axis=1)
