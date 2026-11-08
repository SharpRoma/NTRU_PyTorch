# Exact Hailo polynomial encryption preparation

> Current checkpoint and continuation: [Russian handoff, 9 November 2026](CODEX_HANDOFF_2026-11-09.md). Historical sections below retain their original stage boundaries. N701 has now also passed real hardware qualification on Raspberry Pi; ARM performance and ARM liboqs are still pending.

## Status and experimental boundary

This module implements an exact host model, ONNX export, an admission contract for compiler artifacts, and a HailoRT 4.23 runtime adapter. N=17 and N=107 affine HEFs have been compiled, numerically admitted and hardware-qualified on Hailo-8L with their fixed experimental keys. The tests ran with Secure Boot disabled; loading the driver with Secure Boot enabled is not established. The historical installation and extraction results are recorded in [the Ubuntu run guide](hailo-ubuntu-run.md). Passing CPU or ONNX tests does not establish that DFC can place this graph or preserve its exact arithmetic. The successful ResNet-18 device smoke test only establishes that the installed accelerator works.

The selected first profile fixes the public key in binary weights. It accelerates only the candidate multiplication r*h; sampling, encoding, recombination, coefficient reduction and centering stay on CPU. Secret-key decryption is not implemented on Hailo. One HEF belongs to one canonical public key and ring. The format of ciphertexts and the SymPy/PyTorch implementations remain unchanged.

## Arithmetic and layouts

Let L=(q-1).bit_length(). First replace every h coefficient by its residue in [0,q-1]. Output channel l*N+k has weights W[l*N+k,i] = ((h[(k-i) mod N] >> l) & 1). Input and output channel counts are padded to multiples of 16. All padded weights are zero. No bias, activation, normalization, training or modular arithmetic occurs in the graph.

Each ternary r is split into binary positive and negative vectors. For each sign, consecutive chunks of 16 positions are packed into separate full-width input frames; only the selected chunk may be nonzero. Frame order is polynomial, positive/negative sign, chunk. Runtime frames are NHWC UINT8 [frames,1,1,Cin]. ONNX uses FLOAT NCHW [1,Cin,1,1], a single Conv with static binary weights [Cout,Cin,1,1], and output [1,Cout,1,1]. FLOAT here describes the unquantized compiler input; all exercised products and sums are exactly representable, and it is not the runtime arithmetic contract.

Every output is an integer from 0 to 16. Host INT64 reconstruction sums chunks, subtracts the negative sign, and sums bit planes with factors 2**l. A dense arbitrary ternary input has product bound N*(q-1), checked against INT64. The final ciphertext equals the CPU backends exactly. Raw products using positive h residues need only agree modulo q with products using centered h; the trace explicitly names this stage cyclic_mod_q.

## Preparation and commands

Install export dependencies independently of the vendor runtime:

```powershell
python -m pip install -e ".[hailo-export,test]"
```

Generate or load a public/private JSON key pair with the existing API. Preparation accepts only the public key and never stores a private key:

```python
from ntru_pytorch import PRESETS, generate_keys
from ntru_pytorch.formats import save
public, private = generate_keys(PRESETS["moderate"])
save(public, "public.json")
save(private, "private.json")
```

```powershell
python -m ntru_pytorch hailo-prepare --public-key public.json --artifact results/hailo --cases 1000
python -m ntru_pytorch hailo-check --artifact results/hailo --cases 1000
python -m ntru_pytorch hailo-check --artifact results/hailo --hardware
python -m ntru_pytorch hailo-demo --artifact results/hailo --private-key private.json
python -m ntru_pytorch hailo-benchmark --artifact results/hailo --private-key private.json --output results/hailo-benchmark
```

The preparation directory must be empty. It contains model.onnx, weights.npy (UINT8 logical matrix), public.json, vectors.npz, and manifest.json. Checks write last-check.json, or the path selected with --report. Commands requiring hardware return exit code 2 with a diagnostic if HEF/evidence is absent. A preparation check reports hardware_tested=false even when it passes. No fallback executes CPU under a hardware label.

Use SoftwareHailoEncryptor(public) explicitly for CPU experiments. HailoEncryptor(directory) requires an admitted compiled artifact and must be opened with a context manager. It loads HEF and streams once per session. Both expose encrypt_poly/bytes/text; encrypt_polys sends a batch with explicit r_polys. Byte encryption batches all message polynomials in one inference call. The first profile is synchronous and sessions are not thread-safe. The vendor's official cp311 Windows wheel must be installed separately for Python 3.11; hailo-runtime extra supplies NumPy, not a substitute or downloader for that wheel. DFC is a separate Linux toolchain and is not installed by this project.

## Compiled artifact contract for the next stage

The next stage must produce model.hef plus a trusted extraction of its numerical graph. A narrow DFC 3.33 affine extractor now supports the observed zero-bias binary Conv1x1 topology. It verifies the actual 16/32-bit accumulator and APU transforms; the strict unit profile still has no admission certificate. See [the Ubuntu run guide](hailo-ubuntu-run.md). Do not hand-author a success assertion as a substitute for extracting and inspecting the actual compiler representation.

Set manifest.compiled to an object with runtime_version="4.23.0", architecture="HAILO8L", and files mapping model.hef, numeric-evidence.npz and compiler-report.json to their SHA256 digests. Input/output port objects contain name, shape=[1,1,Cin or Cout], dtype="UINT8", scale=1, zero_point=0. Actual HEF stream names, shapes, UINT8 types and quantization are checked again before opening the device.

numeric-evidence.npz contains effective_weights [Cout,Cin], after compiler weight transformations, and effective_bias [Cout]. They must exactly equal the exported binary matrix and zeros. compiler-report.json binds hef_sha256 and onnx_sha256 to the corresponding artifacts and records compiler_version, operation="single_linear_conv1x1", activation="identity", accumulator_bits=32, requantization="identity", saturation=false, input_scale=output_scale=1 and input_zero_point=output_zero_point=0. These are strict profile requirements, not instructions to force unsupported quantization flags into DFC.

All checksums and report claims are consistency checks, not a cryptographic attestation that a HEF implements a given matrix. Trust in the compiler extraction tool and its provenance remains necessary. Hardware basis-vector, boundary, and random tests are mandatory after numerical admission. They support validation but do not prove all unseen inputs correct. If DFC changes scales, applies lossy transformations, cannot expose adequate evidence, or cannot allocate the graph, this profile is unsupported. Keep the failed diagnostic and research that limitation separately; there is no automatic approximate mode.

## Measurements and article implications

Measure packing, the blocking inference call including host/device transfers, reconstruction/reduction, total byte encryption and one-time HEF/session load separately. The total includes encoding and orchestration; phase sums need not equal the complete call. Compilation duration is reported as unavailable because compilation is external. Hardware internal latency is also unavailable in this synchronous adapter. No energy result is inferred from runtime or nameplate power.

The benchmark validates partial sums before measurements and validates equal ciphertexts plus cross-decryption before each batch size. It uses identical messages and explicit precomputed r for SymPy, PyTorch and Hailo; test sizes are 1,8,32,128 polynomial blocks including the encrypted length header. JSON contains raw samples and median/p95/blocks-per-second summaries; CSV carries phase samples and metadata. Reports include environment versions, seed, HEF/key hashes, compiler version and transfer byte counts. SymPy is a symbolic reference; the primary practical baseline is PyTorch CPU.

For N=701, q=8192, L=13: Cin=704, Cout=9120, 44 chunks per sign and 88 frames per polynomial. At UINT8 the virtual stream payload is 88*(704+9120)=864512 bytes per polynomial, excluding protocol overhead. Logical binary weights occupy 6420480 bytes before compiler transformations. A single polynomial has N*N*L binary matrix terms per frame, with most inputs masked. Hardware skipping behavior and allocation cannot be inferred from the host scheme. This intentionally conservative candidate may be slower than CPU even at large batch sizes.

The earlier articles establish the cyclic-convolution reformulation and motivate offloading. They do not establish exact Hailo quantization, integer accumulator export, effective INT16 TOPS, or net speedup after PCIe traffic. This implementation makes those assumptions explicit and testable. Desktop measurements do not establish Cortex-M4 performance, integrated SoC energy savings, or resistance to side channels. The fixed public weights contain no private key, but blinding inputs and intermediate values still require consideration in a later security study.

## Verification

```powershell
python -m pytest -m "not slow"
python -m pytest tests/test_hailo.py -m slow
```

The slow checks exercise 1000 random polynomials for every historical parameter set against an independent integer convolution; additional examples compare complete ciphertexts against SymPy and Torch. Fast checks include ONNX execution, signed basis vectors, partial-sum bounds, padded channels, UTF-8 and binary payloads, malformed artifacts, absent HEF and mocked SDK session cleanup. Mocked HEF/report fixtures are synthetic adapter tests only and must never be described as real hardware execution.


The opt-in pytest hardware check uses `NTRU_HAILO_ARTIFACT` and `NTRU_HAILO_PRIVATE_KEY` environment variables and `python -m pytest -m hailo_hardware`. Without both paths it is skipped explicitly. This stage has no such artifact; a skipped hardware test is not a pass on the device.


## Windows dependency note

PyHailoRT 4.23 requires NumPy <2. The hailo-runtime extra constrains NumPy accordingly; the export extra constrains ml_dtypes below 0.6 so ONNX and this runtime can coexist. The installed environment uses NumPy 1.26.4 and ml_dtypes 0.5.4. Install both extras together when preparing a new Windows environment.

The vendor wheel also declares netifaces. For Windows CPython 3.11 its installation from PyPI requires Microsoft C++ Build Tools; those tools are not installed by this preparation stage. In the current environment pip check still reports missing netifaces. The vendor uses it in ethernet_utils; PCIe imports and the previous PCIe smoke test do not require it, but full vendor dependency health is not established. The environment report records its absence. Resolve it with the official compiler prerequisites before enabling Ethernet tooling; do not fake a package or substitute another distribution to silence the requirement.


## Historical Windows and WSL checkpoint

The paragraphs below preserve the transfer checkpoint. Native Ubuntu installation, compiled N=17, the implemented affine extractor and remaining hardware work supersede its pending-stage statements; see [the current Ubuntu run guide](hailo-ubuntu-run.md). The setup script now targets Ubuntu 24.04/Python 3.12.

`hailo-build` runs the standalone DFC 3.33 worker inside Ubuntu-22.04 WSL using `/opt/ntru-dfc/bin/python`. The worker does not import the host project or require Torch. Its ClientRunner translation, optimization, inference, archive and compilation interfaces were checked against the downloaded official 3.33 wheel source, but have not yet executed in Linux. `tools/setup_dfc_ubuntu.sh` provisions the dedicated Ubuntu environment after WSL2 is operational. It preserves an existing venv by refusing to overwrite it.

```powershell
python -m ntru_pytorch hailo-build --artifact results/hailo-preview --build-dir results/build-n107-unit --quant-profile unit-v1
python -m ntru_pytorch hailo-build --artifact results/hailo-preview --build-dir results/build-n107-affine --quant-profile binary-circulant-affine-v1
python -m ntru_pytorch hailo-admit --artifact results/hailo-preview --build-dir results/build-n107-unit --admit-output results/admitted-n107
```

Each build directory must be empty. Calibration includes zeros, all basis vectors, every full/tail 16-position chunk and seeded random masked inputs. Input arrays and the ONNX are hashed. The worker saves parsed, quantized and compiled HAR files, HN dictionaries, raw translated SDK parameters, bit-exact emulator sample outputs, model script, timings and the compilation log. The unquantized DFC model must exactly match the host reference. A numeric sample error is diagnostic and is not a proof.

Important unfinished gate: the worker currently ends a successful compile as `compiled-unverified`. It does not infer effective integer weights or rounding semantics from guessed SDK parameter names. The final DFC-specific extraction adapter must be completed from observed HN/translated parameters after the first real run. Until that inspection, it emits no compiler-report or numerical-evidence certificate, `hailo-admit` refuses the build, and no NTRU hardware execution is claimed. Admission verifies source hashes and stages the new artifact in a temporary sibling; the original prepared artifact is unchanged.

An additional arithmetic profile `binary-circulant-affine-v1` is available for independently extracted numerical evidence. It retains the binary/chunk graph but has distinct stream constants and recovery rules; the unit profile is unchanged. The proof includes the decoded-zero contribution of all channels, top-16 worst-case coefficient errors, supplied internal rounding bounds, decoded output saturation limits and compiler-supplied accumulator bounds. Constants are exact rational strings or the exact rational values of binary floating-point numbers. The bound must be strictly below 1/2. A full output LSB conservatively covers truncation and nearest rounding. Recovery uses an exact rational 256-entry lookup table and does not introduce another floating-point rounding step. The evidence producer remains trusted; these checks cannot establish that a fabricated report describes its HEF.

Downloaded compiler package provenance and the preserved wheel are in `results/toolchain/dfc-3.33/`, outside Git. Package archive SHA256: `196e4af9ea55b826d2aba09a00a6dc66bad98840eebc774ce061feb527a858f4`. Wheel SHA256: `788fddd66d1efbdcb8749b37b06f082f0eefc17dac78dab467a3c96096953155`. The checked machine has 16 GB RAM, AMD-V enabled in firmware and ample free disk. Linux compiler installation, numerical extraction and hardware results remain pending; Windows servicing blocks WSL activation: after UAC approval, PowerShell reports 0x80070714 and native DISM exits 87 because CbsProvider cannot initialize. FfuProvider.dll, ImagingProvider.dll and VhdProvider.dll in the system DISM directory have zero length. Diagnostics are preserved in results/toolchain/wsl-diagnostic.json and the copied logs; no automatic system-file replacement or repair has been attempted.
