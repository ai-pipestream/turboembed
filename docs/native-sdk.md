# Intel prepared native SDK

The prepared SDK runs MiniLM embeddings in process through OpenVINO. It exposes
a separate [C extension](../include/turboembed_prepared.h) and leaves the existing
`turboembed.h` ABI unchanged. Text and prepared token calls populate the same
fixed-shape execution slot. GPU execution returns a leased OpenCL output buffer;
copying results to host memory is an explicit operation.

This is development source, not a published release. The
[validation receipt](intel-prepared-sdk-2026-09-14.md) records the tested device,
runtime and model. The [native pilot](intel-prepared-performance-2026-09-14.md)
records matched OpenVINO/ABI measurements. Optional [Rust](rust-prepared-sdk.md)
and [JDK 25 FFM](java-ffm.md) adapters use this same SDK. Android JNI is a later
track.

## Build and install

Prerequisites are Linux, CMake 3.20+, a C++17 compiler, OpenVINO development files,
OpenCL C/C++ development headers, and the OpenCL loader. Python 3 is used for
explicit model provisioning and runtime packaging. The tested build is Linux
x86_64 on Ubuntu 26.04; binaries built there are not qualified for older glibc
or C++ runtimes.

Run from the repository root. `OPENVINO_ROOT` is the extracted OpenVINO
distribution containing `setupvars.sh`, `runtime/`, and its license files.

```bash
export OPENVINO_ROOT=/path/to/openvino-distribution
source "$OPENVINO_ROOT/setupvars.sh"
cmake -S native/turboembed/sdk -B build/native-sdk \
  -DOpenVINO_DIR="$OPENVINO_ROOT/runtime/cmake" \
  -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_LIBDIR=lib
cmake --build build/native-sdk --parallel 2
cmake --install build/native-sdk --prefix /tmp/turboembed-sdk
python3 scripts/package-native-runtime.py \
  --openvino-root "$OPENVINO_ROOT" --prefix /tmp/turboembed-sdk
```

If OpenCL development files are outside standard paths, also pass
`-DOpenCL_INCLUDE_DIR=/path/to/include` and
`-DOpenCL_LIBRARY=/path/to/libOpenCL.so.1` when configuring. No model is downloaded
or inference started by this build.

Use a fresh install prefix. Runtime packaging copies the selected OpenVINO
core, CPU/GPU plugins, IR/ONNX frontends, TBB libraries and licenses. Its
`share/turboembed/runtime-files.json` records their hashes. The installed SDK
uses relative ELF RPATHs, including transitive dependency lookup, so its runtime
can move with the SDK. It does not package the operating system's OpenCL ICD
loader, Intel GPU driver, glibc or C++ runtime. The host must provide those.

## Provision a model

Inference accepts an explicit bundle directory and never downloads models.
The initial qualified source is the self-contained ONNX export of
`sentence-transformers/all-MiniLM-L6-v2`, revision
`1110a243fdf4706b3f48f1d95db1a4f5529b4d41`. Supply its matching tokenizer,
configuration and model card. External-data ONNX exports and other model
families have not been qualified.

```bash
python3 /tmp/turboembed-sdk/bin/prepare-native-bundle.py \
  --source-onnx /path/to/source/onnx/model.onnx \
  --tokenizer /path/to/source/tokenizer.json \
  --config /path/to/source/config.json \
  --model-card /path/to/source/README.md \
  --model-id sentence-transformers/all-MiniLM-L6-v2 \
  --revision 1110a243fdf4706b3f48f1d95db1a4f5529b4d41 \
  --license Apache-2.0 \
  --exporter /tmp/turboembed-sdk/bin/turboembed-export-model \
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

## Use the installed C API

The installed [example](../native/turboembed/sdk/examples/embed.c) is a separate
CMake project that includes only the public C header:

```bash
cmake -S /tmp/turboembed-sdk/share/turboembed/examples -B /tmp/te-consumer \
  -DCMAKE_PREFIX_PATH=/tmp/turboembed-sdk
cmake --build /tmp/te-consumer
/tmp/te-consumer/turboembed_prepared_embed /path/to/new-minilm-bundle
/tmp/te-consumer/turboembed_prepared_embed /path/to/new-minilm-bundle cpu
```

The example embeds text, then uploads the pinned tokenizer's prepared IDs once
and verifies three repeated executions against the text output. GPU is the
default; CPU must be explicitly selected. Missing GPU support returns
an error. There is no server or mandatory network connection.

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

The standard workspace tests do not build this separate SDK. Run provisioning
tests locally, then build the native contract executable on the GPU host:

```bash
python3 -m unittest scripts.tests.test_prepare_native_bundle
cmake -S native/turboembed/sdk -B build/native-sdk -DTE_BUILD_CONTRACT_TEST=ON
cmake --build build/native-sdk --parallel 2
timeout 180 build/native-sdk/prepared_contract_test /path/to/new-minilm-bundle
```

The contract executable requires both the Intel GPU and explicit CPU reference;
it does not silently skip either. It covers parity, prepared/text agreement,
Unicode and NUL input, two-row output layout, invalid arguments, result leases,
concurrent slots, parent release, captured files, and downstream OpenCL
consumption. Also copy the installed prefix and build/run the external C example
with `LD_LIBRARY_PATH` unset before accepting packaging changes.
