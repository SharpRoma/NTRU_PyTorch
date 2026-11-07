"""Exact CPU INT64 kernels; no SymPy dependency."""

import torch

LIMIT = 2**63 - 1


def tensor(a):
    if any(abs(int(v)) > LIMIT for v in a):
        raise OverflowError("Coefficient exceeds INT64")
    return torch.tensor(a, dtype=torch.int64, device="cpu")


def check(a, b):
    for t in (a, b):
        if t.ndim != 1 or not t.numel() or t.dtype != torch.int64 or t.device.type != "cpu":
            raise ValueError("Expected nonempty CPU INT64 vectors")
    ma = max(abs(int(v)) for v in a.tolist())
    mb = max(abs(int(v)) for v in b.tolist())
    if min(a.numel(), b.numel()) * ma * mb > LIMIT:
        raise OverflowError("INT64 accumulation may overflow")


def linear(a, b):
    check(a, b)
    return torch.nn.functional.conv1d(
        a.reshape(1, 1, -1), b.flip(0).reshape(1, 1, -1), padding=b.numel() - 1
    ).reshape(-1)


def reduce(a, n):
    if n < 1:
        raise ValueError("N must be positive")
    if sum(abs(int(v)) for v in a.tolist()) > LIMIT:
        raise OverflowError("Reduction may overflow")
    out = torch.zeros(n, dtype=torch.int64)
    out.scatter_add_(0, torch.arange(a.numel()) % n, a)
    return out


def circulant(a, b):
    check(a, b)
    if a.numel() != b.numel():
        raise ValueError("Lengths differ")
    i = torch.arange(a.numel())
    return a[(i[:, None] - i[None, :]) % a.numel()] @ b


def center(a, q):
    r = torch.remainder(a, q)
    return torch.where(r >= (q + 1) // 2, r - q, r)
