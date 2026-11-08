"""Bounded-memory reader for existing exact DFC NPZ evidence, without conversion."""

from contextlib import contextmanager
from zipfile import BadZipFile, ZipFile

import numpy as np

from .artifacts import ArtifactError


def _header(stream):
    version = np.lib.format.read_magic(stream)
    readers = {
        (1, 0): np.lib.format.read_array_header_1_0,
        (2, 0): np.lib.format.read_array_header_2_0,
    }
    if version not in readers:
        raise ArtifactError("Unsupported numeric evidence NPY version")
    shape, fortran, dtype = readers[version](stream)
    if fortran or dtype.hasobject or dtype.kind not in "biufUS":
        raise ArtifactError("Evidence requires primitive C-order numeric or rational-string rows")
    return shape, dtype


class _Rows:
    def __init__(self, stream, member_size, expected_shape):
        self.stream = stream
        self.shape, self.dtype = _header(stream)
        if self.shape != expected_shape or len(self.shape) != 2:
            raise ArtifactError("Malformed effective affine coefficients")
        self.offset = stream.tell()
        self.row_bytes = self.shape[1] * self.dtype.itemsize
        if not 0 < self.row_bytes <= 4 * 1024 * 1024:
            raise ArtifactError("Evidence row exceeds bounded reader limit")
        if member_size != self.offset + self.row_bytes * self.shape[0]:
            raise ArtifactError("Truncated or trailing numeric evidence")

    def __iter__(self):
        self.stream.seek(self.offset)
        for _ in range(self.shape[0]):
            data = self.stream.read(self.row_bytes)
            if len(data) != self.row_bytes:
                raise ArtifactError("Truncated numeric evidence row")
            yield np.frombuffer(data, dtype=self.dtype)
        if self.stream.read(1):
            raise ArtifactError("Trailing numeric evidence")


@contextmanager
def evidence_rows(path, expected_shape):
    """Keep the original rational strings and proof; allocate at most one weight row."""
    try:
        with ZipFile(path) as archive:
            names = archive.namelist()
            required = ("effective_weights.npy", "effective_bias.npy")
            if any(names.count(name) != 1 for name in required):
                raise ArtifactError("Missing or duplicate numeric evidence members")
            # Bias is small; bound its header and payload just as for matrix rows.
            with archive.open(required[1]) as member:
                info = archive.getinfo(required[1])
                if info.file_size > 4 * 1024 * 1024:
                    raise ArtifactError("Evidence bias exceeds bounded reader limit")
                shape, dtype = _header(member)
                if shape != (expected_shape[0],):
                    raise ArtifactError("Malformed effective affine coefficients")
                size = shape[0] * dtype.itemsize
                if size > 4 * 1024 * 1024 or member.tell() + size != info.file_size:
                    raise ArtifactError("Malformed evidence bias size")
                payload = member.read(size)
                if len(payload) != size:
                    raise ArtifactError("Truncated evidence bias")
                bias = np.frombuffer(payload, dtype=dtype)
            with archive.open(required[0]) as member:
                rows = _Rows(member, archive.getinfo(required[0]).file_size, expected_shape)
                yield rows, bias
    except (BadZipFile, EOFError, OSError, ValueError) as exc:
        if isinstance(exc, ArtifactError):
            raise
        raise ArtifactError(f"Malformed numeric evidence: {exc}") from exc
