# Intel prepared native SDK

The prepared SDK runs MiniLM embeddings in process through OpenVINO. It exposes
a separate [C extension](../include/turboembed_prepared.h) and leaves the existing
`turboembed.h` ABI unchanged. Text and prepared token calls populate the same
fixed-shape execution slot. GPU execution returns a leased OpenCL output buffer;
copying results to host memory is an explicit operation.

The SDK is packaged as a versioned release archive built by a script, not from
a developer's working install. It has not been published to a hosted registry.
The [validation receipt](intel-prepared-sdk-2026-09-14.md) records the tested
GPU device, runtime and model; the
[M2 packaging receipt](m2-sdk-packaging-2026-09-17.md) records the CPU-path
package validation. The [native pilot](intel-prepared-performance-2026-09-14.md)
records matched OpenVINO/ABI measurements. Optional [Rust](rust-prepared-sdk.md)
and [JDK 25 FFM](java-ffm.md) adapters use this same SDK. Android JNI is a later
track.

This page covers the Intel prepared SDK. Provider qualification status after
roadmap M4 — NVIDIA proven on Machine A, Apple gated on Machine C — is
summarized in [nvidia-m4-qualification-2026-09-16.md](nvidia-m4-qualification-2026-09-16.md)
and [apple-m4-machine-c-checklist.md](apple-m4-machine-c-checklist.md).

Per-provider installable packages of the `turboembed.h` text ABI:

| Provider package | Build / acceptance | Status |
|---|---|---|
| NVIDIA CUDA (`turboembed-cuda-sdk-*-linux-x86_64.tar.gz`) | [`scripts/make-nvidia-sdk-release.sh`](../scripts/make-nvidia-sdk-release.sh), [`scripts/nvidia-sdk-consumer-acceptance.sh`](../scripts/nvidia-sdk-consumer-acceptance.sh) | Qualified on Machine A ([receipt](nvidia-m4-qualification-2026-09-16.md)) |
| Apple Metal (`turboembed-metal-sdk-*-macos-arm64.tar.gz`) | [`scripts/make-apple-sdk-release.sh`](../scripts/make-apple-sdk-release.sh), [`scripts/apple-sdk-consumer-acceptance.sh`](../scripts/apple-sdk-consumer-acceptance.sh) | Tooling landed; **not qualified** until built and accepted on Machine C ([checklist step 7](apple-m4-machine-c-checklist.md)) |

The Apple package stages `libTurboEmbed.dylib` with its MLX Metal kernel
libraries, `include/turboembed.h`, a CMake package, C and Swift consumer
examples, and SHA-256 pins for the qualified MiniLM MLX source; its
acceptance script enforces the fail-loud device policy (METAL/AUTO succeed
only with a Metal device; explicit CPU refuses catalog aliases; MOCK is
smoke-only). Both Apple scripts refuse to run off macOS arm64.

## Support matrix

| Component | Supported in this preview | Evidence |
|---|---|---|
| Model | `sentence-transformers/all-MiniLM-L6-v2` @ `1110a243fdf4706b3f48f1d95db1a4f5529b4d41`, self-contained ONNX, f32, mean pooling, L2 normalization, 384 dims, no prefixes | [pins](../models/manifests/prepared-sources.json), [receipt](intel-prepared-sdk-2026-09-14.md) |
| Devices | OpenVINO Intel GPU (explicit or default); OpenVINO CPU (explicit selection only, never a fallback) | GPU: Battlemage G31 on `krick-1`; CPU: [CPU-only hosts](m2-sdk-packaging-2026-09-17.md) |
| Runtime | OpenVINO archive distribution, Linux x86_64. GPU-qualified build: `2026.3.1-22476-759c5a6ab8c` (Ubuntu 26.04, krick-1). CPU package path also exercised with `2025.3.0-19807-44526285f24` (Ubuntu 24.04, hosted CI and receipt below) | receipts above, [CI](../.github/workflows/ci.yml) |
| Capabilities | Text embed, prepared i32 tokens, explicit host read; leased OpenCL result (GPU only) | [header](../include/turboembed_prepared.h) |
| Bindings | C (installed header + CMake package), Rust `turboembed` crate `prepared` feature, JDK 25 FFM | [Rust](rust-prepared-sdk.md), [Java](java-ffm.md) |
| Not in this preview | Reranking, external buffer/queue import, asynchronous submission, GPU tokenization, other model families, non-Linux targets, the `turboembed.h` text ABI as an installed artifact | see notes below |

TurboRerank is deliberately outside this first packaged preview: its native
implementation exists in-tree but has not passed the same packaging, symbol,
bundle-verification, and clean-consumer gates. It ships when it does, rather
than as a stub surface. The frozen `turboembed.h` text ABI likewise remains an
in-process crate surface; the packaged library exports only the
`turboembed_prepared_v1_*` extension and installs only its header, so the
package does not advertise symbols it cannot provide.

## Build the release archive

Prerequisites are Linux, CMake 3.20+, a C++17 compiler, OpenCL C/C++
development headers with the OpenCL loader, and Python 3. `OPENVINO_ROOT` is an
extracted OpenVINO archive distribution containing `setupvars.sh`, `runtime/`,
and its license files. Binaries are qualified only for hosts whose glibc and
C++ runtime are at least the build host's.

```bash
./scripts/make-sdk-release.sh --openvino-root /path/to/openvino-distribution --output-dir dist
```

This configures and builds `native/turboembed/sdk` in a temporary directory,
installs into a clean staging prefix, packages the selected OpenVINO runtime,
and emits `dist/turboembed-prepared-sdk-<version>-linux-x86_64.tar.gz` with a
`.sha256` sibling. The build fails unless the library's dynamic symbol table
matches the public header exactly; the resulting list is installed as
`share/turboembed/exported-symbols.txt`. `share/turboembed/sdk-manifest.json`
records the SDK version, source commit, OpenVINO build string, and a SHA-256
for every installed file, and `share/turboembed/runtime-files.json` records the
packaged runtime hashes. No model is downloaded and no inference runs.

The archive contains `lib/libturboembed_prepared.so.1` with the packaged
OpenVINO core, CPU/GPU plugins, IR/ONNX frontends and TBB (relative ELF RPATHs,
so the prefix can move), `include/turboembed_prepared.h`, the CMake package
under `lib/cmake/TurboEmbedPrepared`, provisioning tools in `bin/`, the
external consumer example, and all license texts. It does not package the
operating system's OpenCL ICD loader, Intel GPU driver, glibc or C++ runtime;
the host must provide those.

For iterative development you can still run the same CMake configure /
build / install / `scripts/package-native-runtime.py` steps by hand against a
scratch prefix; the release script is the supported way to produce an artifact.
If OpenCL development files are outside standard paths, pass
`-DOpenCL_INCLUDE_DIR=` and `-DOpenCL_LIBRARY=` through a manual configure.

## Provision a model

Inference accepts an explicit bundle directory and never downloads models.
Bundle sources are provisioned through the same hash-verified manifest tooling
as the rest of the repository: `models/manifests/prepared-sources.json` pins
the qualified upstream files, and nothing depends on a developer's private
model cache.

```bash
cargo run -p inferstream-fetch -- --prepared minilm     # one-time, network
./scripts/provision-minilm-bundle.sh /path/to/extracted-sdk /path/to/new-minilm-bundle
```

The fetch step downloads the pinned revision into `models/prepared-src/minilm/`
and verifies every byte against the committed SHA-256 pins; a mismatched or
interrupted download is deleted and reported. The provision step re-verifies
those sources, then runs the installed `prepare-native-bundle.py` with the
pinned model identity, revision and license. Everything after the fetch runs
offline.

The initial qualified source is the self-contained ONNX export of
`sentence-transformers/all-MiniLM-L6-v2`, revision
`1110a243fdf4706b3f48f1d95db1a4f5529b4d41`, with its matching tokenizer,
configuration and model card. External-data ONNX exports and other model
families have not been qualified. To provision from explicitly supplied local
files instead, call the installed tool directly:

```bash
python3 /path/to/extracted-sdk/bin/prepare-native-bundle.py \
  --source-onnx /path/to/source/onnx/model.onnx \
  --tokenizer /path/to/source/tokenizer.json \
  --config /path/to/source/config.json \
  --model-card /path/to/source/README.md \
  --model-id sentence-transformers/all-MiniLM-L6-v2 \
  --revision 1110a243fdf4706b3f48f1d95db1a4f5529b4d41 \
  --license Apache-2.0 \
  --exporter /path/to/extracted-sdk/bin/turboembed-export-model \
  --output-dir /path/to/new-minilm-bundle
```

The output directory must not exist. Provisioning stages files in a sibling
directory, records the source hash and converter version, hashes each delivered
artifact, writes `bundle.json` last, then renames the complete directory. The
native loader validates the manifest, captures and verifies the files, and
constructs the graph and tokenizer from those captured bytes. Later file edits
cannot change an already loaded model. Hashes detect disagreement with the
manifest; they do not authenticate the publisher of an untrusted bundle.

The current contract is uncased BERT WordPiece, masked mean pooling, L2
normalization, f32 output with 384 dimensions, and no query/document prefix.
The default bundle limits are batch 32 and sequence 256. Unsupported
configurations and unknown manifest options return errors.

## Discover and select devices

`turboembed_prepared_v1_device_count` and `turboembed_prepared_v1_device_info`
list the devices this extension can select on the running host: OpenVINO GPUs
in ascending ordinal order, then CPU. Each entry reports the resolved device
name, runtime version, capability bits, and (for GPUs) the OpenCL driver
version — the same identity an explicitly created context reports. Runtime
devices the extension cannot select are not listed.

Discovery informs selection; it does not perform it. Creating a context still
requires an explicit device, selecting an absent device still returns
`TE_UNAVAILABLE`, and CPU is never an automatic fallback for a missing GPU.
Enumeration reflects the runtime at call time and may be repeated.

## Use the installed C API

The installed [example](../native/turboembed/sdk/examples/embed.c) is a separate
CMake project that includes only the public C header. Extract the release
archive anywhere and build against it:

```bash
tar -xzf turboembed-prepared-sdk-1.0.0-linux-x86_64.tar.gz -C /opt
SDK=/opt/turboembed-prepared-sdk-1.0.0-linux-x86_64
cmake -S "$SDK/share/turboembed/examples" -B /tmp/te-consumer -DCMAKE_PREFIX_PATH="$SDK"
cmake --build /tmp/te-consumer
/tmp/te-consumer/turboembed_prepared_embed /path/to/new-minilm-bundle
/tmp/te-consumer/turboembed_prepared_embed /path/to/new-minilm-bundle cpu
```

The example lists the discovered devices, embeds text, then uploads the pinned
tokenizer's prepared IDs once and verifies three repeated executions against
the text output. GPU is the default; CPU must be explicitly selected. Missing
GPU support returns an error. There is no server or mandatory network
connection.

Create a context, load a model, and create a slot with fixed batch and sequence
dimensions. Each descriptor starts with its byte size and version. Write text
or row-major i32 IDs/masks/type IDs, execute, consume the result, and release
that result before reusing the slot. Padding, truncation and special tokens are
handled by the native tokenizer for text input. Prepared callers supply the
correct model token IDs themselves.

Models retain contexts, slots retain models, and results retain slots. Releasing
a parent handle therefore does not invalidate a dependent handle. Raw C callers
must coordinate each handle's release with its active calls and must not use a
released handle. Distinct slots have separate requests, queues and buffers.
An overlapping operation on one slot returns `TE_BUSY`.

GPU consumers can borrow the result's OpenCL context, in-order queue and buffer.
Treat the buffer as read-only and use that queue. Result release waits for queued
consumer work before permitting buffer reuse. Host reads are explicit blocking
copies. External queue/buffer imports and asynchronous submission are unsupported.

`slot_stats` counts adapter uploads/readbacks and the bound input/output tensor
sizes. GPU host staging occupies an additional input-tensor-sized allocation.
Tokenizer scratch and OpenVINO internal allocations/transfers are outside these
counters; this API does not claim zero process allocations or zero data movement.

## Validate changes

The standard workspace tests do not build this separate SDK. The consumer
acceptance script proves the packaged artifact in a fresh temporary directory
with a minimal environment: archive and file-manifest hashes, an external
CMake consumer build, loader resolution from the extracted prefix, an explicit
CPU run of text plus prepared tokens, the fail-loud device policy for an
absent GPU, rejection of a tampered bundle, and bounded wall-clock repeats.

```bash
python3 -m unittest scripts.tests.test_prepare_native_bundle
./scripts/sdk-consumer-acceptance.sh \
  dist/turboembed-prepared-sdk-1.0.0-linux-x86_64.tar.gz /path/to/new-minilm-bundle cpu-only
```

Pass `gpu` instead of `cpu-only` on the Intel GPU host, where the default GPU
selection must succeed. The hosted `prepared-sdk` CI job runs the whole CPU
package path — release build, pinned-source fetch, provisioning, acceptance,
the Rust bindings, and the [Java FFM contracts and consumer example](java-ffm.md)
— against the pinned OpenVINO runtime on every push.

GPU changes additionally require the native contract executable on the GPU
host; it needs both the Intel GPU and the explicit CPU reference and does not
silently skip either:

```bash
cmake -S native/turboembed/sdk -B build/native-sdk -DTE_BUILD_CONTRACT_TEST=ON
cmake --build build/native-sdk --parallel 2
timeout 180 build/native-sdk/prepared_contract_test /path/to/new-minilm-bundle
```

It covers device discovery, parity, prepared/text agreement, Unicode and NUL
input, two-row output layout, invalid arguments, result leases, concurrent
slots, parent release, captured files, and downstream OpenCL consumption.

The [2026-09-14 receipts](intel-prepared-sdk-2026-09-14.md) predate device
discovery. Discovery has been validated on CPU-only hosts — see the
[discovery](prepared-discovery-2026-09-17.md) and
[M2 packaging](m2-sdk-packaging-2026-09-17.md) receipts — and remains
hardware-unverified on an Intel GPU until the contract test, the `gpu`-mode
acceptance run, and the ignored `machine_b_gpu_discovery_receipt` Rust test are
re-run on the Machine B (`krick-1`) reference host.
