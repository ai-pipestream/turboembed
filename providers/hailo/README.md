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
(0.32 to 0.71 on the reference texts); ranking is (Spearman 0.94 on the
STS corpus, the same as FP32). The live suite gates both: cosine against
the floor the cell states, and Spearman over `testdata/corpus/sts-pairs.jsonl`.

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
rows, PAD row on padding (`front_end=word`). A single-input HEF is treated
as a community export that expects the full BERT embeddings from the host
(`front_end=bert_embeddings`). The model option `front_end` overrides the
inference; any other option is rejected.

Batches are looped on the host one row at a time (the HEF is batch 1);
`max_batch` is a host-side limit only.

## Build

On Raspberry Pi OS with the `hailo-all` package (HailoRT 4.23, header in
`/usr/include/hailo`, library `/usr/lib/libhailort.so`):

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
reentrant.

## Not offered

Rerank, classification, generation (Hailo-10H generation is a separate
plan item), device buffers, token types other than 0 (the HEF folds type 0
in), and any CPU fallback.
