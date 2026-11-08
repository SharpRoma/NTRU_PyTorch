"""Binary fixed-key profile. All host reconstruction uses exact INT64."""

import hashlib
import json
from dataclasses import dataclass
from time import perf_counter_ns

import numpy as np

from ..models import coefficients

CHUNK = 16
PROFILE = "binary-circulant-chunk16-v1"
PACKINGS = ("position16-v1", "active16-v1")


@dataclass(frozen=True)
class PackedBatch:
    frames: np.ndarray
    owners: np.ndarray
    signs: np.ndarray
    count: int
    packing: str


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
        self.bit_weights = 1 << np.arange(self.bits, dtype=np.int64)
        self.bit_weights.setflags(write=False)
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
        return self._pack_position(polynomials)

    def _pack_position(self, polynomials):
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

    def pack_batch(self, polynomials, packing="position16-v1", *, host_processing="reference"):
        from .host import HOST_PROCESSING, normalize, pack_normalized

        if host_processing not in HOST_PROCESSING:
            raise ValueError("Unknown host processing")
        if packing not in PACKINGS:
            raise ValueError("Unknown packing")
        if host_processing == "optimized":
            return pack_normalized(self, normalize(polynomials, self.n, 1), packing)
        polynomials = tuple(coefficients(r, self.n) for r in polynomials)
        if not polynomials or any(v not in (-1, 0, 1) for r in polynomials for v in r):
            raise ValueError("Expected a nonempty batch of ternary polynomials")
        count = len(polynomials)
        if packing == "position16-v1":
            frames = self._pack_position(polynomials)
            owners = np.repeat(np.arange(count), 2 * self.tiles)
            signs = np.tile(np.repeat([1, -1], self.tiles), count)
        else:
            groups, owners, signs = [], [], []
            for owner, polynomial in enumerate(polynomials):
                vector = np.asarray(polynomial)
                before = len(groups)
                for sign in (1, -1):
                    active = np.flatnonzero(vector == sign)
                    for lo in range(0, len(active), CHUNK):
                        groups.append(active[lo : lo + CHUNK])
                        owners.append(owner)
                        signs.append(sign)
                if len(groups) == before:
                    groups.append(np.empty(0, dtype=np.intp))
                    owners.append(owner)
                    signs.append(1)
            frames = np.zeros((len(groups), 1, 1, self.cin), dtype=np.uint8)
            for i, indices in enumerate(groups):
                frames[i, 0, 0, indices] = 1
            owners = np.asarray(owners, dtype=np.intp)
            signs = np.asarray(signs, dtype=np.int8)
        for array in (frames, owners, signs):
            array.setflags(write=False)
        return PackedBatch(frames, owners, signs, count, packing)

    def reconstruct_batch(self, outputs, batch, *, host_processing="reference"):
        from .host import HOST_PROCESSING, reconstruct_grouped

        if host_processing not in HOST_PROCESSING:
            raise ValueError("Unknown host processing")
        if host_processing == "optimized":
            return reconstruct_grouped(self, outputs, batch)
        started = perf_counter_ns()
        size = len(batch.frames)
        if (
            type(batch.count) is not int or batch.count < 1
            or batch.packing not in PACKINGS
            or batch.owners.shape != (size,) or batch.signs.shape != (size,)
            or not np.issubdtype(batch.owners.dtype, np.integer)
            or not np.issubdtype(batch.signs.dtype, np.integer)
            or np.any(batch.owners < 0) or np.any(batch.owners >= batch.count)
            or np.any((batch.signs != 1) & (batch.signs != -1))
        ):
            raise ValueError("Malformed frame ownership map")
        if batch.packing == "position16-v1":
            mapped = perf_counter_ns()
            result = self.reconstruct(outputs, batch.count)
            self.reconstruction_timings["output_validation_ns"] += mapped - started
            return result
        self._validate_outputs(outputs, size)
        validated = perf_counter_ns()
        signed = np.zeros((batch.count, self.cout), dtype=np.int64)
        np.add.at(signed, batch.owners,
                  outputs.reshape(size, self.cout).astype(np.int64) * batch.signs[:, None])
        summed = perf_counter_ns()
        planes = signed[:, : self.bits * self.n].reshape(batch.count, self.bits, self.n)
        result = (planes * (1 << np.arange(self.bits))[None, :, None]).sum(axis=1)
        self.reconstruction_timings = {
            "output_validation_ns": validated-started,
            "frame_sum_ns": summed-validated,
            "bit_reconstruction_ns": perf_counter_ns()-summed,
        }
        return result

    def software(self, frames):
        flat = self.validate_frames(frames)
        out = np.zeros((len(flat), self.cout), dtype=np.uint8)
        for i, row in enumerate(flat):
            active = np.flatnonzero(row)
            if active.size:
                out[i] = self.weights[:, active].sum(axis=1, dtype=np.uint16)
        return out.reshape(-1, 1, 1, self.cout)

    def reconstruct(self, outputs, count):
        started = perf_counter_ns()
        self._validate_outputs(outputs, count * 2 * self.tiles)
        validated = perf_counter_ns()
        out = outputs.astype(np.int64).reshape(count, 2, self.tiles, self.cout)
        signed = out[:, 0].sum(axis=1) - out[:, 1].sum(axis=1)
        summed = perf_counter_ns()
        planes = signed[:, : self.bits * self.n].reshape(count, self.bits, self.n)
        result = (planes * (1 << np.arange(self.bits))[None, :, None]).sum(axis=1)
        self.reconstruction_timings = {
            "output_validation_ns": validated-started,
            "frame_sum_ns": summed-validated,
            "bit_reconstruction_ns": perf_counter_ns()-summed,
        }
        return result

    def _validate_outputs(self, outputs, frames):
        if outputs.shape != (frames, 1, 1, self.cout):
            raise ValueError("Unexpected output shape")
        if (
            not np.issubdtype(outputs.dtype, np.integer)
            or np.any(outputs < 0)
            or np.any(outputs > CHUNK)
        ):
            raise ValueError("Partial sums must be exact integers in [0,16]")
        if np.any(outputs[..., self.n * self.bits :]):
            raise ValueError("Nonzero padded output channels")
