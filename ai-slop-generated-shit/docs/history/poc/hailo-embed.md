# Embeddings on the Raspberry Pi AI HAT+ (Hailo)

`Device::Hailo` runs MiniLM-class embeddings in-process through the
TurboEmbed C ABI (`include/turboembed.h`) and the HailoRT C API. There is
no server, no Python, and no intermediate framework in the inference path:

```
text ──► native/wordpiece (host) ──► word-embedding gather (host,
         embedding_tables.bin) ──► encoder body on the NPU (model.hef via
         HailoRT vstreams, incl. position/token-type add + embeddings
         LayerNorm) ──► masked pooling + L2 (host) ──► fp32 vector
```

The split exists because the public HEFs compile only the transformer
encoder body (fixed shape, batch 1); the vocab `Gather` does not fit the
Dataflow Compiler. The official Model Zoo contract (confirmed against
`hailo-ai/hailo-apps`' v2a demo): `input_layer1` takes the **raw
word-embedding rows for all 128 positions** (PAD id's row on padding),
`input_layer2` takes the **[seq,seq] additive attention bias** (0 = attend,
-10000 = masked). A `bert_embeddings` front-end (host-side
word+position+token_type + LayerNorm, zero padding) is kept for community
single-input HEFs via `config.json`'s `front_end`. Everything on the host
side is plain C++ already in this repo (`native/wordpiece`,
`native/turboembed/src/hailo.cpp`).

## Hardware lanes

| Board | Chip | Status |
|---|---|---|
| Raspberry Pi 5, AI HAT+ 26 TOPS | Hailo-8 | supported via the official Hailo Model Zoo `all_minilm_l6_v2` hailo8 HEF |
| Raspberry Pi CM5 on the CM5 IO Board + Hailo-8 M.2 M-key module | Hailo-8 | same hailo8 HEF; the CM5IO has no FFC connector — use a standalone B+M/M-key M.2 module (the A+E-key module on the AI HAT+ does not fit) |
| AI HAT+ 13 TOPS | Hailo-8L | supported via the official hailo8l HEF |
| AI HAT+ 2 (8 GB) | Hailo-10H | blocked on a DFC 5.x encoder compile — see "Hailo-10H" below |

The HEF is **architecture-locked**: a hailo8l HEF will not load on a
Hailo-8 and vice versa. It is also **HailoRT-version-locked**: the v2.19.0
zoo HEFs expect the HailoRT 4.x line (the `hailo-all` apt stack); the 10H
uses `hailo-h10-all` (HailoRT 5.x) and the two stacks conflict.

## Board bring-up (the Hailo-8 Pi 5 and the Hailo-8 CM5)

1. Raspberry Pi OS **Trixie, 64-bit**.
2. `sudo apt update && sudo apt install dkms hailo-all`
3. Verify: `hailortcli fw-control identify` prints the chip
   (`HAILO8L` / `HAILO8`). The device node is `/dev/hailo0`.
4. The AI HAT+ negotiates PCIe Gen3 automatically.

CM5 IO Board notes: the module sits in the board's **M.2 M-key slot**, and
its 3.3 V rail is gated by the CM5's `PCIE_PWR_EN` pin (default off via a
100 kΩ pull-down), so run a current bootloader EEPROM
(`sudo rpi-eeprom-update -a`) with `PCIE_PROBE=1`
(`sudo rpi-eeprom-config --edit`). If `lspci` shows no `1e60:` device after
a cold boot, reseat the module — insert at ~30°, screw flat; one reseat
fixed a persistent `brcm-pcie ... link down` during the CM5 bring-up.

## Provision the model (once per board, or rsync)

```sh
make fetch-hailo                                   # HEFs + tokenizer, hash-verified
scripts/hailo-select-hef.sh models/hailo/minilm    # model.<arch>.hef -> model.hef
python3 scripts/export-minilm-hailo-tables.py      # embedding_tables.bin + config.json
```

`export-minilm-hailo-tables.py` is pure stdlib (no pip). It extracts the
embedding tables from the pinned `sentence-transformers/all-MiniLM-L6-v2`
revision (the same pin the prepared-SDK lane uses) and needs network only
for that one download. A fully provisioned `models/hailo/minilm/` is ~90 MB
and can be rsynced between boards of the same chip.

## Build and test

```sh
cargo build -p turboembed --features hailo     # needs hailo/hailort.h (hailo-all)
cargo test -p turboembed --features hailo --test hailo_minilm \
    -- --include-ignored --nocapture --test-threads=1
```

The build links the versioned soname (`libhailort.so.4.x`) directly because
the Pi packages ship no `.so` symlink; override with `HAILORT_LIB_DIR` /
`HAILORT_INCLUDE_DIR` if your prefix differs.

The live suite reports per-text cosines against the FP32 ONNX goldens
(`testdata/reference_embeddings/ort_cuda_minilm_*.json`) without gating on
them — INT8 quantization compresses absolute cosine (0.32–0.71 measured,
mean 0.54) while ranking stays near-parity. The quality gate is Spearman ≥
0.85 on `testdata/corpus/sts-pairs.jsonl`. Never relax or overwrite the
FP32 goldens.

## Measured (the Hailo-8 Pi 5 and the Hailo-8 CM5, 2026-09-20/21)

Full receipts: [testdata/receipts/turboembed/pi5-hailo8.json](../testdata/receipts/turboembed/pi5-hailo8.json),
[testdata/receipts/turboembed/cm5-hailo8.json](../testdata/receipts/turboembed/cm5-hailo8.json).

- **Throughput**: 75.5 embeddings/s steady state (p50 13.24 ms), flat in
  text length (fixed 128-token frame), ≈10× the Pi 5 CPU lane; batch-32
  loops on the host at the same rate (the HEF is batch-1). CM5 and Pi 5
  lanes measure identically (HEF-bound, same Gen3 x1 link).
- **Quality**: Spearman 0.9371 on the 96-pair STS corpus vs 0.9438 for the
  FP32 ORT CPU reference on the same corpus — ranking is near-parity;
  absolute cosine vs FP32 is not preserved (see above). Both hosts
  reproduce the embeddings bitwise (deterministic INT8 HEF).
- **Tokenizer**: native wordpiece matches HF `tokenizers` id-for-id on the
  mixed Greek/CJK/Russian golden text.
- Stack: `hailo-all 5.1.1`, `libhailort.so.4.23.0`, firmware 4.23.0; all 5
  live tests in `crates/turboembed/tests/hailo_minilm.rs` pass.

## Policy (unchanged from every other device)

- `Device::Hailo` on a build/host without the provider or the HAT fails
  `Unavailable` / `UnsupportedDevice` with the rebuild/install guidance.
  It never falls back to CPU and never serves the 8-dim mock.
- `mock-embed` stays smoke-only; a Hailo engine rejects it.
- `EmbedOptions` are honored host-side: `pooling` (mean/cls/last),
  `normalize`, `truncate_to` (floor: 3 specials minimum).

## Hailo-10H (AI HAT+ 2, 8 GB) — pending

The genai stack (`hailo-ollama`) ships decoder LLMs only — no embedding
endpoint, no encoder models. The 10H lane needs one spike: compile the
MiniLM encoder cut (`/embeddings/Add_1` → `last_hidden_state`, fixed
seq len) with DFC 5.x on an x86 host into a `hailo10h` HEF, drop it as
`model.hailo10h.hef`, and the same provider code runs it. Known community
data point: a swept DIY 10H compile of MiniLM measured ≈ −7% nDCG@10 vs
FP32 — evaluate against the goldens before adopting. The 8 GB on-board
memory is irrelevant at MiniLM size; it matters only for the separate
decoder-LLM goal.

## Compiling other encoders (bge-small, …)

Same recipe on any x86_64 host with the Dataflow Compiler (free Hailo
Developer Zone account; DFC 3.x targets hailo8/hailo8l, 5.x targets
hailo10h): export ONNX without the embedding block, quantize with ~50–256
diverse calibration samples, compile per arch. CLS-pooling models (bge)
set `"pooling": "cls"` in `config.json`. New models get a `HAILO_SOURCES`
row and manifest entry via `make update-hailo-manifest`.

## Troubleshooting

- **`no Hailo device found`** — driver not loaded or wrong metapackage;
  check `lspci | grep -i hailo`, `ls -l /dev/hailo0`, `dmesg | grep hailo`.
- **`hailo_configure_vdevice failed`** — almost always an arch or
  HailoRT-version mismatch; re-run `scripts/hailo-select-hef.sh` and check
  `hailortcli --version` against the HEF's compile line (v2.19.0 HEFs want
  the 4.x runtime).
- **Sharing the NPU** with other processes (e.g. hailo-ollama on a 10H)
  needs HailoRT's multi-process service (`group_id`); by default one
  process owns the device.
- **Thermals** — the HATs throttle under sustained load; watch
  `hailortcli monitor` during benches and record throttling in receipts.
