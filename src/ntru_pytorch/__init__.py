"""Public API; optional computational backends are loaded only when requested."""

from importlib import import_module

from .models import PRESETS, Ciphertext, Parameters, PrivateKey, PublicKey
from .randomness import RandomSource

_LAZY = {
    "generate_keys": (".keygen", "generate_keys"),
    "validate_keys": (".keygen", "validate_keys"),
    "SympyNTRU": (".sympy_backend", "SympyNTRU"),
    "TorchNTRU": (".torch_backend", "TorchNTRU"),
    "SparseNTRU": (".sparse_backend", "SparseNTRU"),
}


def __getattr__(name):
    if name not in _LAZY:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module, attribute = _LAZY[name]
    value = getattr(import_module(module, __name__), attribute)
    globals()[name] = value
    return value


def __dir__():
    return sorted(set(globals()) | set(__all__) | set(_LAZY))


__all__ = [
    "Parameters",
    "PublicKey",
    "PrivateKey",
    "Ciphertext",
    "PRESETS",
    "RandomSource",
    "TorchNTRU",
    "SympyNTRU",
    "generate_keys",
    "validate_keys",
]
