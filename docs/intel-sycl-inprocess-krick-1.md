# inferstream-intel: in-process llama.cpp SYCL (krick-1)

Live Tokenize + `ModelStreamInfer` for the logical generation aliases on
the Intel Battlemage host. **In-process** llama.cpp (`GGML_SYCL`, Level
Zero) — same shape as nvidia's in-process CUDA backend. Default LLM
aliases do **not** HTTP to `vlm-server` `:8085`.

`make fetch-llms` and `scripts/smoke-llms.sh` are shell + sha256sum / jq
/ grpcurl. **No python3** on fetch, smoke, or the inferstream process.

Base tree: Origin `main` (this change). Smoke binary built on krick-1
with `--features llamacpp-sycl` and `GGML_SYCL=ON`.

## Catalog (matching GGUF, not VL-7B)

| alias | weight | path |
|---|---|---|
| `default-llm` | Qwen2.5-0.5B-Instruct Q8_0 | `models/gguf/qwen-0.5b/qwen2.5-0.5b-instruct-q8_0.gguf` |
| `qwen-0.5b` | same 0.5B Q8_0 | same |
| `qwen-7b` | Qwen2.5-7B-Instruct Q5_K_M (text, two shards) | `models/gguf/qwen-7b/qwen2.5-7b-instruct-q5_k_m-00001-of-00002.gguf` |

SHA-256 pins: `models/manifests/llms.json`. Fetch: `make fetch-llms`.

## Build

```bash
scripts/setup-llamacpp-sycl.sh
source /opt/intel/oneapi/setvars.sh
GGML_SYCL=ON CMAKE_C_COMPILER=icx CMAKE_CXX_COMPILER=icpx \
  cargo build -p inferstream-arch-intel --release --features llamacpp-sycl
scripts/run-intel.sh --config config/intel.toml --listen 127.0.0.1:8471
```

## Results

Filled after the krick-1 GPU smoke (Tokenize + StreamInfer, xe CCS / VRAM,
tok/s, process list with no python).

| Result | Detail |
|---|---|
| **pending** | smoke + GPU proof running on this host |
