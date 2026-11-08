"""Shared exact array normalization and byte encoding for CPU and Hailo hosts."""

import numpy as np

from .models import coefficients


def normalize(polynomials, n, bound):
    """Validate public coefficients before narrowing to an INT64 array."""
    if (
        isinstance(polynomials, np.ndarray)
        and polynomials.ndim == 2
        and (polynomials.shape[1] == n and polynomials.dtype.kind in "iu")
    ):
        if len(polynomials) == 0 or np.any(polynomials > bound) or np.any(polynomials < -bound):
            raise ValueError("Coefficients outside supported range")
        return polynomials.astype(np.int64, copy=False)
    rows = []
    for polynomial in polynomials:
        row = tuple(polynomial)
        if len(row) > n:
            raise ValueError("Expected at most N integer coefficients")
        if not all(type(v) is int for v in row):
            row = coefficients(row, n)
        if any(v < -bound or v > bound for v in row):
            raise ValueError("Coefficients outside supported range")
        rows.append(row + (0,) * (n - len(row)))
    if not rows:
        raise ValueError("Expected a nonempty batch")
    return np.asarray(rows, dtype=np.int64)


def encode_array(data, n):
    if not isinstance(data, bytes):
        raise TypeError("Expected bytes")
    payload = len(data).to_bytes(8, "big") + data
    bits = np.unpackbits(np.frombuffer(payload, dtype=np.uint8), bitorder="big")
    padded = np.zeros(((len(bits) + n - 1) // n) * n, dtype=np.uint8)
    padded[: len(bits)] = bits
    return padded.reshape(-1, n)
