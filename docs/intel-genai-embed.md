# inferstream-intel: in-process OpenVINO GenAI embeddings

Default Intel **Embed** path is an in-process C++
[`ov::genai::TextEmbeddingPipeline`](https://docs.openvino.ai/2026/api/genai_api/_autosummary/openvino_genai.TextEmbeddingPipeline.html)
reached through a cxx bridge in `crates/backend-openvino`. Clients send
**plain strings**. Tokenization is **openvino-tokenizers** inside the
pipeline. Pooling is CLS / MEAN / LAST_TOKEN with optional L2. Device is
`CPU` / `GPU` / `NPU` (catalog default: **GPU**).

**OVMS gRPC is out of scope.** There is no `backend = "ovms"` client.
Intel Embed is GenAI only. Catalog `backend = "openvino"` constructs
`TurboEmbedBackend` and `ListModels` reports `backend=turboembed`.

This document is the build + Machine B runbook. GPU and CPU MiniLM are
**live** on Machine B (Battlemage) — receipts
[`intel-minilm.json`](../testdata/receipts/turboembed/intel-minilm.json)
and
[`intel-minilm-cpu.json`](../testdata/receipts/turboembed/intel-minilm-cpu.json).
NPU is an honest fail on this host
([`intel-npu.json`](../testdata/receipts/turboembed/intel-npu.json),
`pass=false`): no Intel NPU silicon. A future NPU prove needs a
**Core Ultra client NPU** host — not Xeon, not AWS Inferentia, not
this Battlemage box.

## Build dependencies

Host packages (Ubuntu-style names; oneAPI or the standalone toolkit):

| piece | why |
|---|---|
| OpenVINO runtime (`libopenvino.so`, GPU/NPU plugins) | compile + load IR |
| OpenVINO GenAI (`libopenvino_genai.so`, `openvino/genai/rag/text_embedding_pipeline.hpp`) | `TextEmbeddingPipeline` |
| openvino-tokenizers (`libopenvino_tokenizers.so`) | string → ids inside the pipeline |
| Level Zero + Intel compute-runtime | Battlemage GPU / NPU |
| C++17 compiler | cxx bridge (`g++` or `icpx`) |

Need **OpenVINO GenAI ≥ 2025.4** for `PoolingType::LAST_TOKEN`. CLS/MEAN
exist on earlier 2025.x lines.

```bash
# oneAPI (also used for llama.cpp-SYCL)
source /opt/intel/oneapi/setvars.sh

# or standalone OpenVINO
source /opt/intel/openvino/setupvars.sh
```

`build.rs` finds the toolkit via `pkg-config` (`openvino`, `openvino_genai`)
or `OPENVINO_DIR` / `OPENVINO_GENAI_DIR` / `INTEL_OPENVINO_DIR` /
`/opt/intel/openvino*`.

There is **no Python** on the compile or serve path. `ldd` of
`inferstream-intel` must not list `libpython`.

## Build

```bash
# SYCL generation + GenAI embeddings when OpenVINO GenAI is on the host:
scripts/build-intel.sh

# embeddings-only (no SYCL / icpx):
cargo build -p inferstream-arch-intel --release --features openvino-genai
```

Without `--features openvino-genai` the binary still compiles (CI), but
`backend = "openvino"` fails **at startup** naming the feature — the
default Intel catalog cannot silently serve stubs.

## Fetch OV-format model dirs (no Python)

```bash
make fetch-ov-genai                         # every alias in the GenAI manifest
make fetch-ov-genai ALIASES=minilm,bge-base
make verify-ov-genai
make list-ov-genai
```

Artifacts land in `models/ov/<alias>/` in the GenAI layout:

```
openvino_model.xml
openvino_model.bin
openvino_tokenizer.xml      # required to construct TextEmbeddingPipeline
openvino_tokenizer.bin
tokenizer.json              # inferstream.v1 Tokenize / Detokenize
config.json
```

SHA-256 pins: `models/manifests/ov-genai-embeddings.json`. Sources are
first-party Hugging Face repos when they already publish IR
(`sentence-transformers/*` `openvino/` folders flattened to the dest root;
`OpenVINO/bge-base-en-v1.5-fp16-ov` is a full GenAI dir).

**Tokenizer IR gap.** Several ST / intfloat / thenlper repos publish
`openvino/openvino_model.xml` but not `openvino_tokenizer.xml`. Fetch still
pulls the model IR + `tokenizer.json`. The backend **refuses to load**
until `openvino_tokenizer.xml/.bin` sit next to the model. On Machine B,
copy that pair from a prior IR export or a one-off `convert_tokenizer`
run in `contrib/offline-once/` (historical tooling; not invoked by Make).

Aliases **without** a public IR source today (`bge-small`, `bge-large`,
`nomic-embed-text`): catalog still points at `models/ov/<alias>/`. Place a
GenAI-layout directory there (one-off export) before serving.

## Smoke on Machine B (Battlemage)

```bash
source /opt/intel/oneapi/setvars.sh          # or OpenVINO setupvars.sh
scripts/build-intel.sh
make fetch-ov-genai ALIASES=minilm,bge-base  # + tokenizer IR if the repo omitted it
scripts/run-intel.sh --config config/intel.toml --listen 127.0.0.1:8473

# Process must not contain python:
tr '\0' '\n' < /proc/$(pgrep -n inferstream-intel)/cmdline
ldd target/release/inferstream-intel | grep -i python && echo FAIL || echo ok
# maps should include libopenvino.so / libopenvino_genai.so / libopenvino_tokenizers.so
# and (for GPU) Level Zero. Zero libpython.

scripts/smoke-embeddings.sh 127.0.0.1:8473 change-me minilm
scripts/prove-intel-genai.sh 127.0.0.1:8473 change-me
```

Expected: `ListModels` reports `backend=turboembed` for catalog
`minilm` (the GenAI pipeline is behind the TurboEmbed C ABI). `Embed`
returns FP32 vectors (minilm = 384-dim). GPU default; explicit CPU is
the same IR (`intel-minilm-cpu.json`).

### Ignored GPU golden (crate test)

```bash
INFERSTREAM_OV_MODEL=models/ov/minilm \
INFERSTREAM_OV_DEVICE=GPU \
INFERSTREAM_OV_GOLDEN=/abs/path/testdata/reference_embeddings/ov_genai_minilm_short.json \
cargo test -p inferstream-backend-openvino --features genai -- --ignored gpu_golden
```

The golden file is produced on Machine B after the first successful Embed
(see `testdata/reference_embeddings/README.md`). It is **not** required
for the default `cargo test --workspace`.

## TurboEmbed C ABI (same pipeline, no gRPC)

`include/turboembed.h` + `crates/turboembed --features genai` loads the
same `ov::genai::TextEmbeddingPipeline` in-process:

```bash
# source OpenVINO setupvars (or rely on rpath to /work/opt/openvino_genai)
make test-turboembed-intel
# equivalent:
cargo test -p turboembed --features genai
```

`Device::OpenVinoGpu` compiles `TextEmbeddingPipeline(..., "GPU", config)`
and **fails** if the GPU plugin is missing (no CPU swap).
`Device::OpenVinoCpu` / `Device::Cpu` compiles
`TextEmbeddingPipeline(..., "CPU", config)` and must produce real embeds.
Cosine vs `testdata/e2e/goldens/{intel,nvidia}/minilm.json` ≥ 0.99.
Receipts: `testdata/receipts/turboembed/intel-minilm.json` (GPU) and
`intel-minilm-cpu.json` (CPU). Policy (no embed):
`cargo test -p turboembed --features genai --test device_policy` —
GPU + no GPU plugin is `UNSUPPORTED_DEVICE` (never `"CPU"`). Same for NPU.

### NPU — honest defer on Machine B (Battlemage)

`Device::OpenVinoNpu` is **not** a live TurboEmbed path on this host.
`Engine::create(Device::OpenVinoNpu)` fails loud (`UNSUPPORTED_DEVICE`)
and never compiles `"CPU"` or the 8-d FNV mock.

Live probe (C++, no Python) on **Machine B** against OpenVINO GenAI
2026.3.1 (`/work/opt/openvino_genai`):

| check | result |
|---|---|
| `ov::Core::get_available_devices()` | `CPU GPU` — no `NPU` |
| `libopenvino_intel_npu_plugin.so` | **missing** from `runtime/lib/intel64` (CPU + GPU plugins only) |
| `TextEmbeddingPipeline(models/ov/minilm, "NPU")` | **FAIL** — `Device with "NPU" name is not registered in the OpenVINO Runtime` |
| PCI / `/dev` | AMD Ryzen 9 9950X + Battlemage G31 dGPU. No Intel NPU silicon, no `/dev/accel`, no `intel_npu` node |

Intel NPU is on **Core Ultra client** SoCs (Meteor / Lunar / Arrow Lake),
not on a discrete Battlemage card, not on AMD CPUs, **not Xeon**, and
**not AWS Inferentia**. Policy still accepts `"NPU"` when a future host
lists the plugin (`require_ov_device("NPU", "CPU,NPU")` → `"NPU"`);
create on Machine B cannot. Receipt:
`testdata/receipts/turboembed/intel-npu.json` (`pass=false`, `wired=false`).
Do not treat that file as a MiniLM success receipt.

## OVMS (removed)

OVMS gRPC is not a serving path. `backend = "ovms"` does not parse.
See [`docs/adding-ovms-embedding-pipelines.md`](adding-ovms-embedding-pipelines.md).
