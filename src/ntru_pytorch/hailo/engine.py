"""Encryption orchestration; software mode is always explicitly selected."""

from time import perf_counter_ns

import numpy as np

from ..codec import encode
from ..models import Ciphertext, coefficients
from ..randomness import RandomSource
from .artifacts import ArtifactError, load_artifact
from .host import HOST_PROCESSING, encode_array, normalize, pack_normalized
from .runtime import HailoSession
from .scheme import PACKINGS, BinaryScheme, key_hash


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
        checked_from = perf_counter_ns()
        self._check_key(key)
        if self.host_processing == "optimized":
            messages = normalize(messages, key.params.N, key.params.p // 2)
            rs = normalize(r_polys, key.params.N, 1)
        else:
            messages = tuple(coefficients(m, key.params.N) for m in messages)
            rs = tuple(r_polys)
        return self._encrypt_normalized(key, messages, rs, trace, checked_from)

    def _encrypt_normalized(self, key, messages, rs, trace, checked_from):
        if len(messages) == 0 or len(messages) != len(rs):
            raise ValueError("One r per nonempty message batch required")
        if self.host_processing == "reference" and any(
            abs(v) * 2 > key.params.p for m in messages for v in m
        ):
            raise ValueError("Message outside centered p range")
        start = perf_counter_ns()
        if self.host_processing == "optimized":
            batch = pack_normalized(self.scheme, rs, self.packing)
        else:
            batch = self.scheme.pack_batch(rs, self.packing)
        frames = batch.frames
        prepared = perf_counter_ns()
        outputs = self._infer(frames)
        inferred = perf_counter_ns()
        products = self.scheme.reconstruct_batch(outputs, batch, host_processing=self.host_processing)
        reconstructed = perf_counter_ns()
        blocks = []
        if self.host_processing == "optimized" and trace is None:
            residue = products % key.params.q
            e = (residue + messages) % key.params.q
            e = np.where(e >= (key.params.q + 1) // 2, e - key.params.q, e)
            blocks = tuple(tuple(map(int, row)) for row in e)
        else:
            blocks = self._reduce_sequential(key, products, messages, trace)
        ended = perf_counter_ns()
        self.last_timings = {
            "key_normalization_ns": start - checked_from,
            "prepare_ns": prepared - start,
            "inference_with_io_ns": inferred - prepared,
            "reconstruct_ns": ended - inferred,
            "reduction_ns": ended - reconstructed,
            "pipeline_ns": ended - start,
            "input_bytes": frames.nbytes,
            "output_bytes": outputs.nbytes,
            "polynomial_blocks": len(messages),
            "frames": len(frames),
            **getattr(self.scheme, "reconstruction_timings", {}),
            **getattr(self, "inference_timings", {}),
        }
        return tuple(blocks)

    def _reduce_sequential(self, key, products, messages, trace):
        blocks = []
        for product, message in zip(products, messages, strict=True):
            residue = product % key.params.q
            e = (residue + np.asarray(message, dtype=np.int64)) % key.params.q
            e = np.where(e >= (key.params.q + 1) // 2, e - key.params.q, e)
            blocks.append(tuple(int(v) for v in e))
            if trace is not None:
                trace("cyclic_mod_q", tuple(int(v) for v in residue))
                trace("ciphertext", blocks[-1])
        return tuple(blocks)

    def encrypt_poly(self, key, message, *, r=None, trace=None):
        if r is None:
            r = RandomSource().ternary(key.params.N, key.params.d, key.params.d)
        return self.encrypt_polys(key, [message], r_polys=[r], trace=trace)[0]

    def encrypt_bytes(self, key, data, *, random_source=None, r_polys=None, trace=None):
        start = perf_counter_ns()
        blocks = encode_array(data, key.params.N) if self.host_processing == "optimized" else encode(
            data, key.params.N)
        encoded = perf_counter_ns()
        source = RandomSource() if random_source is None else random_source
        rs = (
            tuple(source.ternary(key.params.N, key.params.d, key.params.d) for _ in blocks)
            if r_polys is None else tuple(r_polys)
        )
        if self.host_processing == "optimized":
            checked_from = perf_counter_ns()
            self._check_key(key)
            rs = normalize(rs, key.params.N, 1)
            encrypted = self._encrypt_normalized(key, blocks, rs, trace, checked_from)
        else:
            encrypted = self.encrypt_polys(key, blocks, r_polys=rs, trace=trace)
        constructed_from = perf_counter_ns()
        result = Ciphertext(key.params, encrypted)
        ended = perf_counter_ns()
        self.last_timings.update(
            message_encode_ns=encoded-start,
            ciphertext_construction_ns=ended-constructed_from,
            encrypt_total_ns=ended-start,
        )
        phases = ("message_encode_ns", "key_normalization_ns", "prepare_ns",
                  "inference_with_io_ns", "reconstruct_ns", "ciphertext_construction_ns")
        self.last_timings["unaccounted_ns"] = ended-start-sum(self.last_timings[k] for k in phases)
        return result

    def encrypt_text(self, key, text, **kwargs):
        return self.encrypt_bytes(key, text.encode("utf-8"), **kwargs)


class SoftwareHailoEncryptor(_Encryptor):
    """Exact host reference, never presented as hardware execution."""

    def __init__(self, key, *, packing="position16-v1", host_processing="reference"):
        if packing not in PACKINGS:
            raise ValueError("Unknown packing")
        if host_processing not in HOST_PROCESSING:
            raise ValueError("Unknown host processing")
        self.host_processing = host_processing
        self.packing = packing
        self.scheme = BinaryScheme(key)
        self.last_timings = {}

    def _infer(self, frames):
        return self.scheme.software(frames)


class HailoEncryptor(_Encryptor):
    """Use as `with HailoEncryptor(directory) as engine:`."""

    def __init__(self, artifact_dir, *, packing="position16-v1", cache_affine_lut=True,
                 host_processing="reference"):
        if packing not in PACKINGS:
            raise ValueError("Unknown packing")
        if type(cache_affine_lut) is not bool:
            raise ValueError("cache_affine_lut must be boolean")
        if host_processing not in HOST_PROCESSING:
            raise ValueError("Unknown host processing")
        self.host_processing = host_processing
        self.packing = packing
        self.cache_affine_lut = cache_affine_lut
        self.manifest, self.scheme = load_artifact(artifact_dir, compiled=True)
        self.session = HailoSession(artifact_dir, self.manifest, self.scheme)
        self.session.cache_affine_lut = cache_affine_lut
        self.session.host_processing = host_processing
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
        output = self.session.infer(frames)
        self.inference_timings = dict(getattr(self.session, "last_timings", {}))
        return output

    def __exit__(self, *exc):
        self.open = False
        return self.session.__exit__(*exc)
