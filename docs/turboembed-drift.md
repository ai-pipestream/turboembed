# TurboEmbed / inferstream embedding drift

Same catalog alias + same text should stay inside the **parity cosine
floors** across nvidia (ORT), intel (OpenVINO GenAI), and apple (MLX).
Drift is that check run over the **popular-model matrix**, not just the
default parity trio (`minilm`, `bge-small`, `mpnet`).

This is scaffolding: the harness reuses `inferstream-e2e` thresholds and
capture code. It does **not** start GPUs.

## Thresholds (reuse parity — do not weaken)

Documented in [`e2e-parity.md`](e2e-parity.md) and
`crates/e2e/src/parity.rs`:

| pair | min cosine |
|---|---|
| same arch (live vs golden) | **0.99** (`SAME_ARCH_MIN`) |
| nvidia ORT FP32 ↔ intel GenAI | **0.99** (`CROSS_FP_MIN`) |
| any pair that includes apple | **0.97** (`CROSS_QUANT_MIN`) |
| mock ↔ mock | **0.99** |

A miss prints the worst text id, both arches, min/mean cosine.

## Models to track (popular × arches)

Source of truth: `testdata/e2e/matrix.json` `embeds[]`. The drift alias
list is `DEFAULT_DRIFT_ALIASES` in `crates/e2e/src/parity.rs`.

| alias | dim | nvidia | intel | apple |
|---|---|---|---|---|
| `minilm` | 384 | yes | yes | yes |
| `minilm-l12` | 384 | yes | yes | yes |
| `mpnet` | 768 | yes | yes | — |
| `bge-small` | 384 | yes | yes | yes |
| `bge-base` | 768 | yes | yes | yes |
| `bge-large` | 1024 | yes | yes | yes |
| `bge-m3` | 1024 | yes | yes | yes |
| `e5-small` | 384 | yes | yes | yes |
| `e5-base` | 768 | yes | yes | yes |
| `e5-large` | 1024 | yes | yes | yes |
| `gte-small` | 384 | yes | yes | yes |
| `gte-base` | 768 | yes | yes | yes |
| `nomic-embed-text` | 768 | yes | yes | — |

An alias that an arch does not serve (not on `ListModels`, or catalog
`NotAvailableOnArch`) **skips** that pair. `minilm` is still required
when an arch is in the peer set and serves nothing.

Do not add GGUF-Q / leftover 4-bit MLX aliases to this list; those are
not claimed at 0.99 / 0.97.

## Make target

```bash
# Same skip-if-no-addrs rule as e2e-parity (CI cloud must not start GPUs).
make e2e-drift

# Live three-way:
INFERSTREAM_E2E_NVIDIA_ADDR=krick:8461 \
INFERSTREAM_E2E_INTEL_ADDR=krick-1:8461 \
INFERSTREAM_E2E_APPLE_ADDR=krickert-mac:8461 \
  make e2e-drift

# Dumps only:
make e2e-drift DUMP_NVIDIA=testdata/e2e/goldens/nvidia \
               DUMP_INTEL=testdata/e2e/goldens/intel \
               DUMP_APPLE=testdata/e2e/goldens/apple

# Subset:
cargo run -p inferstream-e2e -- --drift --only minilm,bge-small \
  --peer nvidia=krick:8461 --peer intel=krick-1:8461
```

`make e2e-drift` is `inferstream-e2e --drift` (parity-cross + the drift
alias list). Capture goldens with the existing
`make e2e-parity-goldens TARGET=… WRITE=1`.

## Recording

Each passing pair prints `n=… min=… mean=…` — that **is** the drift
record for the run. Commit new goldens under
`testdata/e2e/goldens/<arch>/<alias>.json` when a host is first brought
up; later runs compare live (or dump) vs dump using the same floors.
