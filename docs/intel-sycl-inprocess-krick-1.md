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
scripts/build-intel.sh          # setup-llamacpp-sycl + icpx rustc link; no python3
make fetch-llms                 # curl + sha256sum; no python3
scripts/run-intel.sh --config config/intel.toml --listen 127.0.0.1:8473
scripts/prove-intel-sycl.sh 127.0.0.1:8473 change-me
```

`scripts/build-intel.sh` sets `GGML_SYCL=ON` and uses `icpx -fsycl` as the
rustc linker so device images in `libggml-sycl.a` are extracted. rust-lld
alone cannot. `crates/backend-llamacpp/build.rs` then pulls in `libsycl`,
Level Zero, oneMKL SYCL BLAS, and oneDNN — static `libggml-sycl.a` does
not.

## Results

Filled after the krick-1 GPU smoke (Tokenize + StreamInfer, xe CCS / VRAM,
tok/s, process list with no python).

| Result | Detail |
|---|---|
| **pending** | smoke + GPU proof running on this host |
