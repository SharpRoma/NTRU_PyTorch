# NTRU PyTorch research library

Актуальное состояние и передача другому чату Codex: **[docs/CODEX_HANDOFF_2026-11-09.md](docs/CODEX_HANDOFF_2026-11-09.md)**. N701 прошёл реальную аппаратную проверку на Raspberry Pi 5 1 ГБ; скорость на Pi и liboqs для ARM ещё не проверены.

Two independent implementations of NTRU polynomial encryption: SymPy and PyTorch CPU INT64. A ciphertext produced by either backend can be decrypted by either backend. Identical keys, message coefficients and blinding polynomials produce identical ciphertexts.

This is an experimental NTRUEncrypt-style research implementation, not an implementation of the standardized NTRU-HPS/HRSS KEM. Preset names are historical experiment labels and do not assert security levels. The message format is not authenticated; successful parsing does not prove ciphertext authenticity. CPU INT64 support does not establish constant-time execution or NPU feasibility.

## Installation and commands

Python 3.11 or later:

```powershell
python -m pip install -e ".[test]"
python -m ntru_pytorch demo
python -m ntru_pytorch trace
python -m ntru_pytorch benchmark --preset moderate --runs 20 --warmup 3 --output results/benchmark
python -m pytest -m "not slow"
python -m pytest -m slow
```

The installed `ntru-research` command has the same arguments. `trace` explicitly prints intermediate cryptographic data; other commands do not. Importing the package does not generate keys or write files. Install `.[plots]` only if you need matplotlib for your own figures.

## API

```python
from ntru_pytorch import PRESETS, generate_keys, SympyNTRU, TorchNTRU
from ntru_pytorch.formats import dumps, loads, save, load

public, private = generate_keys(PRESETS["moderate"])
sympy, torch = SympyNTRU(), TorchNTRU()
cipher = sympy.encrypt_text(public, "Hello, мир! 😀\0")
assert torch.decrypt_text(private, loads(dumps(cipher))) == "Hello, мир! 😀\0"
save(public, "public.json")
save(private, "private.json")
```

`generate_keys` returns `(PublicKey, PrivateKey)` and verifies both inverses and the public-key identity. It accepts `random_source` and `max_attempts`. Both engines provide `encrypt_poly(key, coefficients, r=None, trace=None)`, `decrypt_poly(key, coefficients, trace=None)`, and byte/text methods. Optional arguments are keyword-only. Results at the public boundary are tuples of Python integers. `Ciphertext` stores a tuple of blocks and `Parameters`; all models are frozen dataclasses with immutable coefficient tuples.

Polynomial coefficients are ordered `[a0, a1, ...]` in `Z[x]/(x^N-1)`. Short message and blinding polynomials are padded on the right; overlong inputs are rejected. Key vectors and ciphertext blocks must have exactly N coefficients. Centering subtracts the modulus when `2*c >= modulus`. Polynomial decryption preserves -1; the byte decoder rejects nonbinary output.

Default randomness uses `secrets.SystemRandom`. For repeatable experiments use `RandomSource(seed)`, which must not be used for real secrets. To compare backends, pass the same explicit `r` to `encrypt_poly` or one `r_polys` entry per block to `encrypt_bytes`. A trace is a callback `(stage_name, coefficient_tuple)` invoked by the actual algorithm. Each block emits its own sequence of stage events.

## Formats and migration

The new JSON envelope has `version: 1`, `kind` (`PublicKey`, `PrivateKey`, or `Ciphertext`), `params`, and model coefficient fields (`h`; `f, fp, fq, g`; or `blocks`). Encryption prepends an eight-byte unsigned big-endian payload length to the bytes before bit packing. Bits are most-significant-first; final blocks are padded with zeros. The header is encrypted with the data. Decoding checks length, zero padding and binary coefficients, then text decoding applies UTF-8 to the complete byte sequence.

Old stateful classes and root-module wrappers have been retired. Replace file-based `encrypt(name, text)` with loading a key once and calling the selected engine. New key generation no longer runs factorization or sparsity heuristics as security checks.

Explicit read-only adapters in `ntru_pytorch.formats`:

```python
from ntru_pytorch.formats import (
    read_legacy_public, read_legacy_private,
    read_legacy_ciphertext, decrypt_legacy_bytes,
)
private = read_legacy_private("old.priv")
cipher = read_legacy_ciphertext(old_cipher_string, private.params)
plaintext = decrypt_legacy_bytes(TorchNTRU(), private, cipher)
```

Legacy public files do not contain df/dg; the reader uses placeholders df=1, dg=0. Those weights are irrelevant to encryption/decryption; ring compatibility uses N/p/q. Old ciphertexts lack a length header, so legacy decoding stops at the first zero byte and cannot recover trailing zero bytes. Unlike the old decoder, UTF-8 can be applied to the complete returned byte string. Legacy input uses strict integer parsing. New writes always use JSON. Existing user keys are neither migrated automatically nor included as test fixtures.

## Research method and validation

`src/ntru_pytorch/tensor.py` isolates the exact tensor kernels without SymPy calls. Encryption uses full conv1d and cyclic reduction; decryption uses circulant matrix multiplication. CPU INT64 shapes and conservative accumulation bounds are checked; unsupported ranges raise exceptions. Experimental ONNX export and a Hailo pipeline are available separately; hardware qualification applies to the fixed keys and admitted HEFs described below.

Tests include a fixed artificial-key vector captured from the original Git HEAD, independent small convolution references, randomized differential arithmetic, all four backend combinations, UTF-8 and binary payloads, legacy readers, malformed inputs and all five large preset key-generation runs. Large tests are marked `slow` and may take minutes. Synthetic identity-key fixtures are strictly for regression testing.

Benchmarks validate equality before timing, use the same operands and report raw nanoseconds, median, quartiles, parameters, seed, warmup, versions, platform and Torch thread count in JSON/CSV. Polynomial measurements compare cyclic multiplication over integers without coefficient reduction. Backend-native inputs are prepared outside the polynomial timer; both results are converted to Python integer tuples inside it. Memory measurements include message coding and seeded sampling but exclude key generation and file IO; JSON read/write are timed separately. This compares PyTorch against the SymPy symbolic implementation, not optimized native NTRU. Historical article measurements are not reproduced or replaced.


## Experimental Hailo preparation

The optional `ntru_pytorch.hailo` module exports a fixed-public-key binary Conv2d graph and checks its exact host/ONNX arithmetic. It includes a fail-closed HailoRT 4.23 adapter, artifact checksums, numerical-evidence requirements and hardware benchmark commands. N=17, N=107 and N=701 HEFs have been compiled, numerically admitted and tested for exact hardware results with their fixed experimental keys on Hailo-8L. Research tests use the user-selected configuration with Secure Boot disabled. See [the Ubuntu run record](docs/hailo-ubuntu-run.md) and [the N=701 hardware benchmark](docs/hailo-n701-benchmark.md). CPU simulation is explicitly named `SoftwareHailoEncryptor` and is never reported as hardware execution.

Install `.[hailo-export]` for ONNX tools and see [the Hailo research guide](docs/hailo-research.md) for preparation, artifact admission, commands, measured boundaries and the next compiler stage. No firmware update, DFC installation or change to the current device runtime is part of this module.

For active-coefficient packing, cached affine recovery and the four-variant hardware comparison, see [the optimization guide](docs/hailo-optimization.md).

For exact grouped reconstruction, compact LUT decoding and the optional `--host-processing optimized` mode, see [the CPU processing comparison](docs/hailo-host-processing.md). The default remains `reference`.

Desktop installation remains unchanged. For a separate lean runtime wheel, bounded-memory admission of existing HEFs and preparation for Raspberry Pi, see [the edge runtime guide](docs/edge-runtime.md).

`SparseNTRU` adds exact NumPy CPU baselines using sparse ternary multiplication and optional prepared key shifts. See [the stronger CPU comparison](docs/sparse-cpu-baseline.md): on the tested desktop N=701 workload it outperforms both the original PyTorch backend and Hailo. Existing defaults are unchanged.

Внешние CPU-ориентиры liboqs 0.16.0 добавлены отдельно от существующих backend: публичный NTRU-HRSS-701 KEM и upstream C-ядро циклического умножения для нашего N701-шифрования. Сборка локальная, профили clean/AVX2 закреплены commit и SHA256; KEM и прикладное шифрование измеряются раздельно. Инструкция, границы сравнения и подготовка AArch64: [docs/liboqs-integration.md](docs/liboqs-integration.md).
