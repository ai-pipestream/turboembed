# inferstream-intel: in-process llama.cpp SYCL (Machine B)

Live Tokenize + `ModelStreamInfer` for the logical generation aliases on
the Intel Battlemage host. **In-process** llama.cpp (`GGML_SYCL`, Level
Zero) — same shape as nvidia's in-process CUDA backend. Default LLM
aliases do **not** HTTP to `vlm-server` `:8085`.

`make fetch-llms` is `cargo run -p inferstream-fetch -- --llms` (Rust +
sha256). `scripts/fetch-llms.sh` is an optional curl + sha256sum
fallback. `scripts/smoke-llms.sh` is jq / grpcurl. **No python3** on
fetch, smoke, or the inferstream process. Historical OpenVINO IR export
lives in `contrib/offline-once/` only; Make defaults do not call it.
OVMS gRPC is out of scope — Intel embeds are in-process GenAI.

Host: **Machine B**, Battlemage `Intel(R) Graphics [0xe223]`, xe DRM.
Binary: `target/release/inferstream-intel` (106 MiB, `libsycl.so.8`, no
`libpython`). Listen: `127.0.0.1:8473`. PID **1122034**.

Tree at smoke: `623fe32` (`fix: link Intel OpenMP (iomp5) into the SYCL
intel binary`) on `ai-pipestream/intel-sycl-inprocess-7dff`.

## Catalog (matching GGUF, not VL-7B)

| alias | weight | path |
|---|---|---|
| `default-llm` | Qwen2.5-0.5B-Instruct Q8_0 | `models/gguf/qwen-0.5b/qwen2.5-0.5b-instruct-q8_0.gguf` |
| `qwen-0.5b` | same 0.5B Q8_0 | same |
| `qwen-7b` | Qwen2.5-7B-Instruct Q5_K_M (text, two shards) | `models/gguf/qwen-7b/qwen2.5-7b-instruct-q5_k_m-00001-of-00002.gguf` |

SHA-256 pins (`models/manifests/llms.json`, verified on disk):

| file | sha256 |
|---|---|
| `qwen2.5-0.5b-instruct-q8_0.gguf` | `ca59ca7f13d0e15a8cfa77bd17e65d24f6844b554a7b6c12e07a5f89ff76844e` |
| `qwen2.5-7b-instruct-q5_k_m-00001-of-00002.gguf` | `42f6693004793ee6cf1b2b723f0273b10f86a3bb2a949bd9128d4cda5fb866cd` |
| `qwen2.5-7b-instruct-q5_k_m-00002-of-00002.gguf` | `beba9d4f2f5a1fe7d144dcae332e68b52c26705c5310dece2e5d1997e091e134` |

`ModelMetadata.properties` for every alias: `device=Sycl`,
`mode=in-process`, `n_gpu_layers=4294967295` (`u32::MAX`), no `endpoint`.

## Build

```bash
scripts/build-intel.sh          # setup-llamacpp-sycl + icpx rustc link; no python3
make fetch-llms                 # inferstream-fetch (Rust); no python3
scripts/run-intel.sh --config config/intel.toml --listen 127.0.0.1:8473
scripts/prove-intel-sycl.sh 127.0.0.1:8473 change-me
```

`scripts/build-intel.sh` sets `GGML_SYCL=ON` and uses `icpx -fsycl` as the
rustc linker **only for the final binary** so device images in
`libggml-sycl.a` are extracted. rust-lld alone cannot. Crate build
scripts stay on the default linker. `crates/backend-llamacpp/build.rs`
then pulls in `libsycl`, Level Zero, oneMKL SYCL BLAS, oneDNN, and
Intel OpenMP (`iomp5`) — static `libggml-sycl.a` does not.

## Load (GPU, not CPU)

llama.cpp log on this host:

| model | layers | SYCL0 weights |
|---|---|---|
| Qwen2.5-0.5B Q8_0 | **25/25** offloaded | **500.84 MiB** |
| Qwen2.5-7B Q5_K_M | **29/29** offloaded | **4829.59 MiB** |

StreamInfer also reserved SYCL0 KV (48 MiB / 224 MiB) and compute
(298.50 / 304.00 MiB) buffers. `sycl-ls` after `setvars.sh`:

```
[level_zero:gpu][level_zero:0] Intel(R) oneAPI Unified Runtime over Level-Zero V2, Intel(R) Graphics [0xe223] 20.2.0 [1.14.37020]
```

`/proc/1122034/maps` includes `libsycl.so.8.0.0`, `libze_loader.so.1`,
`libze_intel_gpu.so.1`, `libmkl_sycl_blas.so.5`, `libiomp5.so`. **Zero**
`libpython` mappings.

## Smoke (Tokenize + StreamInfer)

`scripts/smoke-llms.sh 127.0.0.1:8473 change-me default-llm qwen-0.5b qwen-7b`
— 64-token prompt, no python3.

| MODEL | TOK_IDS | CHUNKS | FINAL | COLD_MS | WARM_MS | TOK_S |
|---|---:|---:|---|---:|---:|---:|
| default-llm | 5 | 64 | true | 9919 | 300 | **213.3** |
| qwen-0.5b | 5 | 64 | true | 298 | 257 | **249.0** |
| qwen-7b | 5 | 64 | true | 11725 | 920 | **69.6** |

`=== llm smoke: 3 passed, 0 failed ===`

## xe DRM / VRAM (pid 1122034, after StreamInfer)

```
drm-driver:           xe
drm-pdev:             0000:03:00.0
drm-resident-vram0:   6613920 KiB   (~6.3 GiB)
drm-cycles-ccs:       59713723      (nonzero CCS — GPU compute, not idle)
```

## No python during StreamInfer

`scripts/prove-intel-sycl.sh` sampled `/proc/<pid>` + descendants every
250 ms **while** `ModelStreamInfer` ran.

```
===== during 2026-09-11T19:03:59-04:00 pid=1122034 =====
    PID    PPID USER     CMD
1122034       … operator /work/inferstream/target/release/inferstream-intel --config config/intel.toml --listen 127.0.0.1:8473
-- children --
(no children)
-- cmdlines (self + descendants) --
  1122034: /work/inferstream/target/release/inferstream-intel --config config/intel.toml --listen 127.0.0.1:8473
```

Threads are `{inferstream-int}` only. **ok: no python in the inferstream
process tree during StreamInfer.** `ldd` of the binary has no
`libpython`.

## Results

| Result | Detail |
|---|---|
| **PASS** | in-process SYCL; `mode=in-process`; no `:8085` |
| **PASS** | 0.5B 25/25 + 7B 29/29 on SYCL0 (Level Zero `0xe223`) |
| **PASS** | Tokenize + StreamInfer all three aliases; real tok/s |
| **PASS** | xe CCS cycles + 6.3 GiB resident VRAM |
| **PASS** | zero Python on fetch-llms, smoke, binary, and process tree during StreamInfer |
