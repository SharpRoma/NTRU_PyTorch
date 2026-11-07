"""Encryption orchestration; software mode is always explicitly selected."""

from time import perf_counter_ns

import numpy as np

from ..codec import encode
from ..models import Ciphertext, coefficients
from ..randomness import RandomSource
from .artifacts import ArtifactError, load_artifact
from .runtime import HailoSession
from .scheme import BinaryScheme, key_hash


class _Encryptor:
    def __enter__(self):
        self.last_timings = {}
        return self

    def __exit__(self, *exc):
        return False

    def _check_key(self, key):
        if key_hash(key) != key_hash(self.scheme.key):
            raise ArtifactError("Artifact was prepared for a different public key")

    def encrypt_polys(self, key, messages, *, r_polys, trace=None):
        self._check_key(key)
        messages = tuple(coefficients(m, key.params.N) for m in messages)
        rs = tuple(r_polys)
        if not messages or len(messages) != len(rs):
            raise ValueError("One r per nonempty message batch required")
        if any(abs(v) * 2 > key.params.p for m in messages for v in m):
            raise ValueError("Message outside centered p range")
        start = perf_counter_ns()
        frames = self.scheme.pack(rs)
        prepared = perf_counter_ns()
        outputs = self._infer(frames)
        inferred = perf_counter_ns()
        products = self.scheme.reconstruct(outputs, len(messages))
        blocks = []
        for product, message in zip(products, messages, strict=True):
            residue = product % key.params.q
            e = (residue + np.asarray(message, dtype=np.int64)) % key.params.q
            e = np.where(e >= (key.params.q + 1) // 2, e - key.params.q, e)
            blocks.append(tuple(int(v) for v in e))
            if trace is not None:
                trace("cyclic_mod_q", tuple(int(v) for v in residue))
                trace("ciphertext", blocks[-1])
        ended = perf_counter_ns()
        self.last_timings = {
            "prepare_ns": prepared - start,
            "inference_with_io_ns": inferred - prepared,
            "reconstruct_ns": ended - inferred,
            "pipeline_ns": ended - start,
            "input_bytes": frames.nbytes,
            "output_bytes": outputs.nbytes,
            "polynomial_blocks": len(messages),
            "frames": len(frames),
        }
        return tuple(blocks)

    def encrypt_poly(self, key, message, *, r=None, trace=None):
        if r is None:
            r = RandomSource().ternary(key.params.N, key.params.d, key.params.d)
        return self.encrypt_polys(key, [message], r_polys=[r], trace=trace)[0]

    def encrypt_bytes(self, key, data, *, random_source=None, r_polys=None, trace=None):
        start = perf_counter_ns()
        blocks = encode(data, key.params.N)
        source = RandomSource() if random_source is None else random_source
        rs = (
            tuple(source.ternary(key.params.N, key.params.d, key.params.d) for _ in blocks)
            if r_polys is None
            else tuple(r_polys)
        )
        result = Ciphertext(key.params, self.encrypt_polys(key, blocks, r_polys=rs, trace=trace))
        self.last_timings["encrypt_total_ns"] = perf_counter_ns() - start
        return result

    def encrypt_text(self, key, text, **kwargs):
        return self.encrypt_bytes(key, text.encode("utf-8"), **kwargs)


class SoftwareHailoEncryptor(_Encryptor):
    """Exact host reference, never presented as hardware execution."""

    def __init__(self, key):
        self.scheme = BinaryScheme(key)
        self.last_timings = {}

    def _infer(self, frames):
        return self.scheme.software(frames)


class HailoEncryptor(_Encryptor):
    """Use as `with HailoEncryptor(directory) as engine:`."""

    def __init__(self, artifact_dir):
        self.manifest, self.scheme = load_artifact(artifact_dir, compiled=True)
        self.session = HailoSession(artifact_dir, self.manifest, self.scheme)
        self.last_timings = {}
        self.open = False

    def __enter__(self):
        if self.open:
            raise RuntimeError("Session already open")
        start = perf_counter_ns()
        self.session.__enter__()
        self.load_ns = perf_counter_ns() - start
        self.open = True
        return self

    def _infer(self, frames):
        if not self.open:
            raise RuntimeError("Use HailoEncryptor as a context manager")
        return self.session.infer(frames)

    def __exit__(self, *exc):
        self.open = False
        return self.session.__exit__(*exc)
