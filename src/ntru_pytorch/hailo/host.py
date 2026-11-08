"""Exact optional host kernels; no device execution or admission shortcuts."""

from time import perf_counter_ns

import numpy as np

from ..array_api import encode_array as encode_array
from ..array_api import normalize as normalize

HOST_PROCESSING = ("reference", "optimized")


def pack_normalized(scheme, polynomials, packing):
    """Internal kernel: coefficients have already passed public validation."""
    from .scheme import PackedBatch

    count = len(polynomials)
    if packing == "position16-v1":
        frames = scheme._pack_position(polynomials)
        owners = np.repeat(np.arange(count), 2 * scheme.tiles)
        signs = np.tile(np.repeat([1, -1], scheme.tiles), count)
    else:
        positive = [np.flatnonzero(row == 1) for row in polynomials]
        negative = [np.flatnonzero(row == -1) for row in polynomials]
        counts = np.array([max(1, (len(p)+15)//16 + (len(m)+15)//16)
                           for p, m in zip(positive, negative, strict=True)])
        size = int(counts.sum())
        frames = np.zeros((size, 1, 1, scheme.cin), dtype=np.uint8)
        owners = np.repeat(np.arange(count), counts)
        signs = np.ones(size, dtype=np.int8)
        offset = 0
        for p, m, frame_count in zip(positive, negative, counts, strict=True):
            split = offset + (len(p)+15)//16
            frames[offset + np.arange(len(p))//16, 0, 0, p] = 1
            frames[split + np.arange(len(m))//16, 0, 0, m] = 1
            signs[split:offset+frame_count] = -1 if len(m) else 1
            offset += frame_count
    for array in (frames, owners, signs):
        array.setflags(write=False)
    return PackedBatch(frames, owners, signs, count, packing)


def reconstruct_grouped(scheme, outputs, batch):
    start = perf_counter_ns()
    size = len(batch.frames)
    if (
        type(batch.count) is not int or batch.count < 1
        or batch.packing not in ("position16-v1", "active16-v1")
        or batch.owners.shape != (size,) or batch.signs.shape != (size,)
        or not np.issubdtype(batch.owners.dtype, np.integer)
        or not np.issubdtype(batch.signs.dtype, np.integer)
        or np.any(batch.owners < 0) or np.any(batch.owners >= batch.count)
        or np.any((batch.signs != 1) & (batch.signs != -1))
    ):
        raise ValueError("Malformed frame ownership map")
    if batch.packing == "position16-v1":
        # Positional reconstruction has always used the fixed layout, not the map.
        scheme._validate_outputs(outputs, batch.count * 2 * scheme.tiles)
    else:
        scheme._validate_outputs(outputs, size)
    validated = perf_counter_ns()
    flat = outputs.reshape(-1, scheme.cout)
    if batch.packing == "position16-v1":
        view = flat.reshape(batch.count, 2, scheme.tiles, scheme.cout)
        signed = view[:, 0].sum(axis=1, dtype=np.int64)
        signed -= view[:, 1].sum(axis=1, dtype=np.int64)
    else:
        signed = np.zeros((batch.count, scheme.cout), dtype=np.int64)
        if size and np.all(batch.owners[1:] >= batch.owners[:-1]):
            starts = np.r_[0, np.flatnonzero(
                (batch.owners[1:] != batch.owners[:-1])
                | (batch.signs[1:] != batch.signs[:-1])) + 1]
            ends = np.r_[starts[1:], size]
            for lo, hi in zip(starts, ends, strict=True):
                partial = flat[lo:hi].sum(axis=0, dtype=np.int64)
                if batch.signs[lo] == 1:
                    signed[batch.owners[lo]] += partial
                else:
                    signed[batch.owners[lo]] -= partial
        else:
            np.add.at(signed, batch.owners,
                      flat.astype(np.int64) * batch.signs[:, None])
    summed = perf_counter_ns()
    planes = signed[:, :scheme.bits * scheme.n].reshape(batch.count, scheme.bits, scheme.n)
    result = (planes * scheme.bit_weights[None, :, None]).sum(axis=1, dtype=np.int64)
    end = perf_counter_ns()
    scheme.reconstruction_timings = {
        "output_validation_ns": validated - start,
        "frame_sum_ns": summed - validated,
        "bit_reconstruction_ns": end - summed,
    }
    return result
