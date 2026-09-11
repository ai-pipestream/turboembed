# inferstream-apple: LLM alias Metal smoke (krickert-mac)

Live Tokenize + `ModelStreamInfer` for the logical generation aliases on
the Apple M2 host via **mlx-lm on Metal** (not CPU, not GGUF-on-MLX).

Base tree: Origin `main` **`9073c27`** (LLM alias catalog). Smoke binary
`inferstream-apple` rebuilt from that SHA plus the tok/s / tokenizer /
ping-stats fixes (`e1a41ce`, later merged to `main` as `b93865e`).

| Result | Detail |
|---|---|
| **PASS** | `scripts/smoke-llms.sh` — 3 passed, 0 failed |
| Host | krickert-mac, MacBook Air Mac14,2, Apple M2, 24 GB, Metal 4 |
| Engine | persistent `python/mlx_bridge.py` → `mlx-lm` 0.32.2, `Device(gpu, 0)` |

## What was smoked

Memory was already tight (compressor + swap), so aliases were served
conservatively: **0.5B pair first**, then **7B 4-bit alone**.

| pass | config | serve |
|---|---|---|
| 1 | `/tmp/apple-llm-0.5b.toml` | `default-llm`, `qwen-0.5b` (same MLX 4-bit) |
| 2 | `/tmp/apple-llm-7b.toml` | `qwen-7b` only (~4 GB 4-bit) |

| alias | ListModels | Tokenize `"Hello, inferstream!"` | StreamInfer (`max_tokens=16`) |
|---|---|---|---|
| `default-llm` | `mlx` ready, `hasTokenizer=true` | ids=5 tokens=5 first=`Hello` `[9707, 11, 23583, 4027, 0]` | 11 chunks, `final=true`, cold 12266 ms / 0.9 t/s, warm 2061 ms / **5.3 t/s** |
| `qwen-0.5b` | same | same ids (shared Qwen tokenizer) | 11 chunks, `final=true`, cold 1348 ms / 8.2 t/s (already hot), warm 1829 ms / **6.0 t/s** |
| `qwen-7b` | `mlx` ready, `hasTokenizer=true` | same 5 ids | 9 chunks, `final=true`, cold 16999 ms / 0.5 t/s, warm 3774 ms / **2.4 t/s** |

Warm `qwen-7b` decoded `Hello, nice to meet you.` Tokenizer.json pins
matched `models/manifests/llms.json` (`sha256=c0382117…`, 7,031,645 B)
for both `qwen-0.5b` and `qwen-7b`.

Smoke-script tok/s is **wall-clock including gRPC/JSON**, on a short
completion (9–11 pieces). Cold includes the first MLX load into unified
memory. `qwen-0.5b` “cold” is already warm because it shares
`mlx-community/Qwen2.5-0.5B-Instruct-4bit` with `default-llm`.

## Metal evidence (not CPU)

Standalone bridge ping after `scripts/setup-mlx.sh`:

```json
{"mlx_version":"0.32.2","matmul_ok":true,"device":"Device(gpu, 0)",
 "metal_available":true,"active_memory":32788,"peak_memory":49244}
```

`mx.device_info()`:

```
device_name = Apple M2
architecture = applegpu_g14g
memory_size = 25769803776
max_recommended_working_set_size = 19069665280
```

`vmmap` of the live `mlx_bridge.py` worker after StreamInfer mapped
Apple GPU, not a CPU fallback:

- `/System/Library/Extensions/AGXMetalG14G.bundle` (M2 GPU)
- `Metal.framework`
- `MetalPerformanceShaders` / `MPSNeuralNetwork`
- `IOAccelerator` shared mappings

Worker RSS (process view; 4-bit weights mostly sit in unified GPU
memory, so RSS is not the full footprint):

| stage | python `mlx_bridge` RSS |
|---|---|
| idle, before first generate | ~21–32 MB |
| after `Qwen2.5-0.5B-Instruct-4bit` | 185 MB |
| after `Qwen2.5-7B-Instruct-4bit` | 661 MB |

Server log on first generate (HF cache hit after prefetch of the 7B
pin `c26a38f6…`):

```
[mlx_bridge] loading LM mlx-community/Qwen2.5-0.5B-Instruct-4bit
[mlx_bridge] loading LM mlx-community/Qwen2.5-7B-Instruct-4bit
```

`mx.default_device()` stayed `Device(gpu, 0)` for the whole run.
`metal_available=True`. There is no `Device(cpu, …)` path in this venv.

## Fixes merged

- `scripts/smoke-llms.sh` reports `COLD_T/S` / `WARM_T/S`
- `scripts/setup-mlx.sh` fetches the pinned `qwen-7b` tokenizer
- `config/apple.toml` serves `qwen-0.5b` next to `default-llm` (same 4-bit)
- `mlx_bridge` ping reports `metal_available` + active/peak memory via
  mlx 0.32 `mx.get_*_memory`
