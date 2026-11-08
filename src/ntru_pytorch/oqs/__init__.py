"""Optional liboqs adapters. Importing this module never loads native code."""

__all__ = ["OqsKem", "OqsArithmeticEncryptor"]


def __getattr__(name):
    if name == "OqsKem":
        from .kem import OqsKem

        return OqsKem
    if name == "OqsArithmeticEncryptor":
        from .arithmetic import OqsArithmeticEncryptor

        return OqsArithmeticEncryptor
    raise AttributeError(name)
