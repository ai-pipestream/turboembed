# NVIDIA e2e embed parity goldens (Machine A)

Live capture on the NVIDIA host for cross-arch cosine. Tree is Origin
`main` **`2b8883c`** or later. No Python — `inferstream-fetch` +
`inferstream-e2e`.

## What was up

`inferstream-nvidia` on **ORT CUDA** serving `minilm` and `bge-small`
(`config/nvidia-parity.toml`). Full `config/nvidia.toml` also lists those
aliases (plus LLMs); the slim file is what this run started.

| | |
|---|---|
| listen | `0.0.0.0:8461` |
| token | `change-me` (`INFERSTREAM_E2E_TOKEN`) |
| reachable at | `<machine-a>:8461` on the LAN and over the VPN |
| GPU | RTX 4080 SUPER, CUDA EP (`device=cuda` in session log) |

`make fetch-corpus` landed SHA-pinned `testdata/corpus/tiny-shakespeare.txt`
(gitignored). STS pairs were already committed.

## Dumps

| path | notes |
|---|---|
| `testdata/e2e/goldens/nvidia/minilm.json` | 237 items, 384-d, mean+L2, ~2.0 MiB |
| `testdata/e2e/goldens/nvidia/bge-small.json` | 237 items, 384-d, CLS+L2, ~2.0 MiB |

Schema is `docs/e2e-parity.md` (`schema_version`, `items[].id/text/vector`).
These files are **committed** (not gitignored).

Write / replay:

```bash
make e2e-parity-goldens TARGET=nvidia WRITE=1 \
  INFERSTREAM_E2E_NVIDIA_ADDR=127.0.0.1:8461
make e2e-parity-goldens TARGET=nvidia \
  INFERSTREAM_E2E_NVIDIA_ADDR=127.0.0.1:8461
```

## Cosine vs self-replay

Live ORT CUDA vs the dumps just written:

| alias | min | mean | n | gate |
|---|---|---|---|---|
| `minilm` | **1.0000** | **1.0000** | 237 | 0.99 |
| `bge-small` | **1.0000** | **1.0000** | 237 | 0.99 |

`hello world` first components (live Embed after capture) sit on the
golden within ~5e-5 — fp32 session noise, not a different vector.

## `--parity-cross` from Machine A

| peer | reach | result |
|---|---|---|
| apple / Machine C | **yes** — live capture on Machine C after the pooling fix | **before:** minilm min **-0.1404** / mean **-0.0085** (BERT pooler). **after (FP + mean+L2):** minilm min **0.9795** / mean **0.9997** (n=213); bge-small min **0.9938** / mean **0.9999**. See `testdata/e2e/goldens/apple/README.md`. |
| intel / Machine B | host up, reachable on the LAN and over the VPN | **no inferstream** on `:8461` / `:8471`. Only llama-server `:8085`. Goldens not on `origin/main` yet. |

Retry when intel is serving:

```bash
INFERSTREAM_E2E_NVIDIA_ADDR=<machine-a>:8461 \
INFERSTREAM_E2E_INTEL_ADDR=<machine-b>:8461 \
INFERSTREAM_E2E_APPLE_ADDR=<machine-c>:8461 \
INFERSTREAM_E2E_TOKEN=change-me \
  make e2e-parity
```

Offline three-way once peer dumps land:

```bash
make e2e-parity \
  DUMP_NVIDIA=testdata/e2e/goldens/nvidia \
  DUMP_INTEL=testdata/e2e/goldens/intel \
  DUMP_APPLE=testdata/e2e/goldens/apple
```
