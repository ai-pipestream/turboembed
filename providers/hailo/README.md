# Turbo Hailo provider

`libturbo_provider_hailo.so` serves `EMBED x TEXT` on Hailo-8 and Hailo-8L
accelerators (Raspberry Pi AI HAT+, AI HAT+ 26 TOPS, M.2 modules) through
the HailoRT 4.x C API. It is a plugin for the core: `libturbo` loads it
from the provider path and talks to it through `turbo_provider_get`.

## What runs where

The public HEFs (Hailo Model Zoo `all_minilm_l6_v2`) contain only the
transformer encoder body at one fixed shape (128 tokens, batch 1). The
vocabulary gather does not fit the Dataflow Compiler, so the provider
splits the pipeline and says so in `turbo_model_info`:

| stage | placement | done by |
|---|---|---|
| tokenize | host | native WordPiece (`native/wordpiece`) |
| word-embedding gather | host | the `hailo_tables` artifact |
| encode (position/token-type add, LayerNorm, 6 layers) | device | the HEF through vstreams |
| pool (mean, CLS, last) | host | provider |
| normalize, `output_dim` | host | provider |

`fully_accelerated` is 0. Results are `TURBO_PLACE_HOST` f32 and the device
does not advertise `TURBO_CAP_DEVICE_RESULT`; `h2d_bytes` and `d2h_bytes`
count the frames moved through HailoRT's DMA pipeline.

The HEF is quantized. The capability cell for `EMBED x TEXT` reports
`dtype = I8`, `reference_dtype = F32`, and the measured `cosine_floor`
against the FP32 reference vectors. Absolute cosine is not preserved
(0.32 to 0.71 on the reference texts); ranking stays near parity with FP32
(Spearman 0.937 against 0.944 on the STS corpus). The live suite gates
both: cosine against the floor the suite owns for this provider and dtype
(`testdata/reference_embeddings/quantized_floors.json`, 0.45, set from the
receipt; the cell's own floor must not exceed it), and Spearman over
`testdata/corpus/sts-pairs.jsonl`.

## Bundle

An embedding bundle with two artifacts and a WordPiece `tokenizer.json`:

| artifact | contents |
|---|---|
| `hef` | the encoder HEF compiled for this chip (`hailo8`, `hailo8l`); a HEF for another chip fails at load with the chip named |
| `hailo_tables` | `embedding_tables.bin`: fp32 word, position, and token-type tables plus the embeddings LayerNorm, exported by `scripts/export-hailo-tables.py` from the checkpoint the HEF was compiled from |

The contract must say `max_seq` equal to the HEF frame length and
`limits.fixed_shape` must be set; the provider checks both against the HEF
at load and refuses a mismatch.

```sh
python3 scripts/export-hailo-tables.py tables/          # embedding_tables.bin
turbo-bundle import --source all-MiniLM-L6-v2 --output minilm-hailo8 \
    --license Apache-2.0 --model-id sentence-transformers/all-MiniLM-L6-v2 \
    --artifact hef=model.hailo8.hef --artifact hailo_tables=tables/embedding_tables.bin \
    --max-seq 128 --fixed-shape --max-batch 32
```

Front end: a two-input HEF (hidden state plus the `[seq, seq]` additive
attention bias) is the official Model Zoo contract and takes the raw word
rows, PAD row on padding (`front_end=word`); it is inferred, and it is the
front end the receipts measure. A single-input HEF must name its front end
with the model option `front_end=word` or `front_end=bert_embeddings` (the
host computes the full BERT embeddings; no receipt covers it yet, so the
capability's cosine floor does not speak for it). Any other option is
rejected.

Batches are looped on the host one row at a time (the HEF is batch 1);
`max_batch` is a host-side limit only.

## Build

On Raspberry Pi OS with the `hailo-all` package (HailoRT 4.23, header in
`/usr/include/hailo`, library `/usr/lib/libhailort.so`);
[`docs/hailo-pi-setup.md`](../../docs/hailo-pi-setup.md) covers getting a
Raspberry Pi 5 AI HAT board to that point for both the Hailo-8 and the
Hailo-10H package lines:

```sh
cmake -S providers/hailo -B build/hailo
cmake --build build/hailo
```

Pass `-DHAILORT_INCLUDE_DIR` and `-DHAILORT_LIBRARY` for another install.
The library exports one symbol, links HailoRT and the in-tree WordPiece
tokenizer, and is never unloaded once loaded (HailoRT is not unload-safe).

## Test

Vtable tests (no core involved) and the Rust live suite:

```sh
export TURBO_LIVE_BUNDLE=$HOME/bundles/minilm-hailo8
./build/hailo/turbo_provider_hailo_test
TURBO_LIVE_LIB=$PWD/build/hailo/libturbo_provider_hailo.so TURBO_LIVE_PROVIDER=hailo \
    cargo test -p turbo-conformance --test live_embed -- --test-threads=1 --nocapture
```

Without a Hailo device the vtable tests skip and say so; `device_count`
reports the driver's own error when the PCIe driver is missing rather than
returning an empty list.

## Devices

One ordinal per Hailo PCIe device, kind `TURBO_DEVICE_NPU`, vendor id
`0x1e60`. The name carries the architecture, board name, and BDF
(`Hailo-8 (Hailo-8, 0001:01:00.0)`); `runtime_version` is the HailoRT
library version and `driver_version` the device firmware. A context is a
`hailo_vdevice` bound to that one device with HailoRT's scheduler on, so
several models can be configured on one context. Sessions of one model
serialize their runs on the model's mutex because vstreams are not
reentrant. A vstream write or read that fails mid-run leaves frames in
flight that nothing can drain, so the model is marked unusable and every
later run fails with `TURBO_E_INVALID_STATE` until it is reloaded.

## Status

`EXPERIMENTAL` for `EMBED x TEXT` on Hailo-8: the 14 vtable tests
(`providers/hailo/tests/provider_test.cpp`) and the 14 live embedding tests
(`crates/turbo-conformance/tests/live_embed.rs`, including the STS ranking
gate) pass on both `pi5ai1` (Raspberry Pi 5 with AI HAT+ 26 TOPS) and
`cm5ai1` (CM5 IO Board, Hailo-8 M.2 module); receipt:
`testdata/receipts/turbo/hailo-2026-09-21.json`. Throughput on `pi5ai1`
(`turbo-bench embed`, `testdata/receipts/turbo/bench/hailo-pi5ai1-embed-2026-09-21.json`):
76 rows/s at every batch and sequence length, since the HEF is batch 1 with
a fixed 128-token frame (about 13.2 ms per row). Hailo-8L is untested (no
board), Hailo-10H needs a DFC 5 HEF, and the x86_64 PCIe build is untried.
A matched-native benchmark receipt is still required before this cell can
move from `EXPERIMENTAL` to `SUPPORTED`.

## Not offered

Rerank, classification, generation (Hailo-10H generation is a separate
plan item), device buffers, token types other than 0 (the HEF folds type 0
in), and any CPU fallback.
