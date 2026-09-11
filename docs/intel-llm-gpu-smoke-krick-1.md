# inferstream-intel: LLM alias GPU smoke (krick-1)

Live Tokenize + `ModelStreamInfer` for the logical generation aliases on
the Intel Battlemage host. **GPU only** — same bar as the OVMS embed proof
(`docs/adding-ovms-embedding-pipelines.md`): xe DRM compute-engine cycles
attributable to the SYCL llama-server must move during StreamInfer, plus
resident VRAM and server eval tok/s.

Base tree: Origin `main` **`9073c27`** (LLM alias catalog). Smoke binary
built from that SHA on krick-1.

| Result | Detail |
|---|---|
| **PASS** | `scripts/smoke-llms.sh 127.0.0.1:8471 change-me default-llm qwen-7b` — 2 passed, 0 failed |
| `qwen-0.5b` | **not served** (`Tokenize` → `NotFound: model "qwen-0.5b" is not configured`). Honest: no 0.5B SYCL server. |

## What was smoked

`inferstream-intel --config config/intel.toml --listen 127.0.0.1:8471`
(19 models: OVMS embeds + LLM aliases). Catalog aliases:

| alias | ListModels | Tokenize `"Hello, inferstream!"` | StreamInfer |
|---|---|---|---|
| `default-llm` | `llama-cpp` ready, `hasTokenizer=true` | ids `[9707, 11, 23583, 4027, 0]` = `Hello \| , \| ␠infer \| stream \| !` | smoke: 5 chunks, `final=true`, cold 105 ms / warm 92 ms |
| `qwen-7b` | same | same ids (same llama-server vocab) | smoke: 4 chunks, `final=true`, cold 146 ms / warm 68 ms |
| `qwen-0.5b` | absent from ListModels | NotFound | not called |

`ModelMetadata` for both served aliases reports the **actual** loaded
weight, not a pretend 0.5B / text-only 7B:

```
device      = Sycl
endpoint    = http://127.0.0.1:8085
model_alias = ggml-org/Qwen2.5-VL-7B-Instruct-GGUF:Q4_K_M
model_ftype = Q4_K - Medium
model_path  = .../Qwen2.5-VL-7B-Instruct-Q4_K_M.gguf
```

## llama.cpp SYCL server

Container `vlm-server` (`ghcr.io/ggml-org/llama.cpp:server-intel`),
healthy, host `:8085`. Cmd: `-hf ggml-org/Qwen2.5-VL-7B-Instruct-GGUF:Q4_K_M
--host 0.0.0.0 --port 8080 -ngl 99 -c 8192`. `/dev/dri` passed through
(xe / Battlemage G31). oneAPI + Level Zero in the image
(`ONEAPI_ROOT=/opt/intel/oneapi`, compiler 2025.3). `/health` →
`{"status":"ok"}`.

## GPU evidence (xe DRM, same bar as OVMS)

Sampled inside the llama-server container:

```bash
docker exec vlm-server sh -c 'grep -h drm-cycles-ccs /proc/1/fdinfo/*'
docker exec vlm-server sh -c 'grep -h drm-resident-vram0 /proc/1/fdinfo/*'
```

`drm-driver: xe`, `drm-pdev: 0000:03:00.0` (Battlemage G31).

| probe | CCS cycles (pid 1, client with compute) | `drm-resident-vram0` |
|---|---|---|
| idle, before first StreamInfer | 79,202,186 | 456 KiB |
| after `default-llm` StreamInfer (32 tokens) | 120,750,882 (**+41,548,696**) | **5,064,648 KiB (~4.83 GiB)** |
| after `qwen-7b` StreamInfer (40 tokens) | 137,854,759 (**+10,473,976** from prior) | 5,070,920 KiB |

CCS only moved while StreamInfer was in flight. Weights landed in
dedicated VRAM (not a CPU decode). Idle VRAM of hundreds of KiB vs ~4.8
GiB resident during generation is the SYCL offload (`-ngl 99`) becoming
hot.

## tok/s (llama-server `slot print_timing`)

| alias | prompt eval | decode eval |
|---|---|---|
| `default-llm` (32 pred) | 11 tokens / 169.33 ms/tok cold-prompt (5.91 tok/s) | **91.08 tok/s** (10.98 ms/tok) |
| `qwen-7b` (40 pred) | 17 tokens / 8.87 ms/tok (**112.68 tok/s**) | **91.31 tok/s** (10.95 ms/tok) |

~11 ms/token decode on Qwen2.5-VL-7B Q4_K_M is Battlemage SYCL, not
CPU. Smoke-script wall times are short because the default prompt
(`Say hello in five words or fewer.`) finishes in 3–4 tokens.

## Catalog honesty (chose honesty, did not stand up a 0.5B server)

`default-llm` and `qwen-7b` on intel both forward to the host SYCL
llama-server. On krick-1 that process loads **Qwen2.5-VL-7B-Instruct
Q4_K_M**, not a text-only 7B and not a 0.5B. That is already written in
`config/catalog.toml` and the README alias table. This smoke:

* did **not** point `qwen-0.5b` at `:8085` (would be a lie);
* did **not** stand up a second 0.5B SYCL server (not present on the host);
* left `qwen-0.5b` without an intel catalog sub-table so serving it
  fails at startup / RPC with `NotAvailableOnArch` / `NotFound`;
* surfaces `model_path` + `model_alias` from llama-server `/props` on
  `ModelMetadata` so clients see the VL-7B GGUF behind the logical name.

A dedicated text-only 7B or 0.5B SYCL server would be a later host
change; until then the aliases stay honest about the VL-7B endpoint.
