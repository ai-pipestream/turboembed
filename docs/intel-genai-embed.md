# inferstream-intel: in-process OpenVINO GenAI embeddings

Default Intel **Embed** path is an in-process C++
[`ov::genai::TextEmbeddingPipeline`](https://docs.openvino.ai/2026/api/genai_api/_autosummary/openvino_genai.TextEmbeddingPipeline.html)
reached through a cxx bridge in `crates/backend-openvino`. Clients send
**plain strings**. Tokenization is **openvino-tokenizers** inside the
pipeline. Pooling is CLS / MEAN / LAST_TOKEN with optional L2. Device is
`CPU` / `GPU` / `NPU` (catalog default: **GPU**).

**No OVMS gRPC on the default path.** `backend = "ovms"` stays compiled in
as optional/legacy.

This document is the build + krick-1 smoke runbook. **GPU live smoke was
not run on the cloud VM that landed this code** — do that on **krick-1**
(Battlemage).

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
until `openvino_tokenizer.xml/.bin` sit next to the model. On krick-1,
copy them from the existing OVMS export (already SHA-pinned in
`models/manifests/ovms-embeddings.json`) or from a one-off
`convert_tokenizer` run in `contrib/offline-once/` (not invoked by Make).

Aliases **without** a public IR source today (`bge-small`, `bge-large`,
`nomic-embed-text`): catalog still points at `models/ov/<alias>/`. Place a
GenAI-layout directory there (one-off export) before serving.

## Smoke on krick-1 (Battlemage)

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

Expected: `ListModels` reports `backend=openvino`, `platform=openvino_genai`,
`device=GPU`. `Embed` returns FP32 vectors (minilm = 384-dim).

### Ignored GPU golden (crate test)

```bash
INFERSTREAM_OV_MODEL=models/ov/minilm \
INFERSTREAM_OV_DEVICE=GPU \
INFERSTREAM_OV_GOLDEN=/abs/path/testdata/reference_embeddings/ov_genai_minilm_short.json \
cargo test -p inferstream-backend-openvino --features genai -- --ignored gpu_golden
```

The golden file is produced on krick-1 after the first successful Embed
(see `testdata/reference_embeddings/README.md`). It is **not** required
for the default `cargo test --workspace`.

## OVMS (legacy)

To front the host Model Server instead of in-process GenAI, use an explicit
`[[models]]` with `backend = "ovms"` (see the commented block in
`config/intel.toml`) or a catalog override. Walkthrough:
[`docs/adding-ovms-embedding-pipelines.md`](adding-ovms-embedding-pipelines.md).
