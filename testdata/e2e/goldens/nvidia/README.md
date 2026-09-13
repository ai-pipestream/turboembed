# NVIDIA goldens (Machine A)

Captured **2026-09-12** on **Machine A** (RTX 4080 SUPER) from Origin `main`
`2b8883c` (`feat: SHA-pinned test corpus and cross-arch embedding parity`).

| file | alias | pooling | dim | items |
|---|---|---|---|---|
| `minilm.json` | `minilm` | mean + L2 | 384 | 237 |
| `bge-small.json` | `bge-small` | CLS + L2 | 384 | 237 |

`mpnet` was not on the serve list for this capture.

## Server

- binary: existing `target/release/inferstream-nvidia` (ORT CUDA EP)
- config: `config/nvidia-parity.toml` (`serve = ["minilm", "bge-small"]`)
- listen: **`0.0.0.0:8461`**
- bearer: **`change-me`**
- LAN: `192.168.1.242:8461` / `192.168.1.243:8461`
- Tailscale: `100.110.72.95:8461`

`minilm` loads TEI's HF ONNX snapshot; `bge-small` loads
`models/onnx/bge-small/onnx/model.onnx`.

## Self-replay (live vs these dumps)

```
make e2e-parity-goldens TARGET=nvidia INFERSTREAM_E2E_NVIDIA_ADDR=127.0.0.1:8461
```

| alias | n | min cosine | mean cosine | gate |
|---|---|---|---|---|
| `minilm` | 237 | **1.0000** | **1.0000** | ≥ 0.99 |
| `bge-small` | 237 | **1.0000** | **1.0000** | ≥ 0.99 |

## Peer compare (intel / apple)

Dumps are **not** gitignored. Point `--parity-cross` at this directory:

```bash
# dumps only (no live GPU needed on the client)
make e2e-parity \
  DUMP_NVIDIA=testdata/e2e/goldens/nvidia \
  DUMP_INTEL=testdata/e2e/goldens/intel \
  DUMP_APPLE=testdata/e2e/goldens/apple

# live nvidia + a peer dump
cargo run -p inferstream-e2e -- --parity-cross \
  --peer nvidia=192.168.1.242:8461 \
  --dump intel=testdata/e2e/goldens/intel \
  --token change-me
```

Thresholds: nvidia ↔ intel MiniLM FP **0.99**; apple FP MLX vs nvidia
**0.97** (English MiniLM is ~1.000; min 0.9795 is CJK UNK drift). See
`docs/e2e-parity.md` and `docs/nvidia-e2e-parity-goldens-machine-a.md`.
