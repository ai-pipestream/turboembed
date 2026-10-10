# Static models against other implementations

Programs that run other static-embedding libraries the way `core/tests/static_speed.rs` runs turbo. They measure speed and score accuracy against Model2Vec's vectors. None of this is part of the library, and nothing here is built by the workspace.

| Tool | Version | Program |
|---|---|---|
| [statembed](https://crates.io/crates/statembed) | 1.0.0, with and without `rayon` | `rs/src/bin/statembed.rs` |
| [model2vec-rs](https://github.com/MinishLab/model2vec-rs) | 1567eaa | `rs/src/bin/m2vrs.rs` |
| [go-potion](https://github.com/trengrj/go-potion) | v0.1.2 | `go/main.go` |
| [model2vec-zig](https://github.com/PaytonWebber/model2vec-zig) | ed5b443 (0.2.0) | `zig/src/main.zig` |
| [Model2Vec](https://github.com/MinishLab/model2vec) | 3ef2bf2 | `py/bench.py model2vec` |
| [sentence-transformers](https://github.com/huggingface/sentence-transformers) `StaticEmbedding` | 6.1.0 | `py/bench.py st` |

Each program takes a model directory, a JSON array of texts, a comma-separated list of batch sizes (or `-` for none), and optionally a file for the vectors. All use the same timing protocol: one warm pass, then three runs of at least two seconds each. A run embeds the texts in order, batch by batch, and the best run is reported in texts per second. The vectors are written as raw little-endian F32, one row per text. A text the tool refuses gives a row of NaN.

model2vec-zig reads WordPiece tables in F32 or I8 only, so it is skipped for potion-code-16M, potion-code-16M-v2 and potion-multilingual-128M. It has no batch API, so a batch larger than 1 splits the texts across one thread per CPU.

## Running

`peers.sh` takes three directories, each with one subdirectory per potion model:

- `MODELS_DIR`: the model's own `config.json`, `tokenizer.json` and `model.safetensors`.
- `GOLDEN_DIR`: `texts.json` and `golden-512.safetensors`, as written by `bundle/model2vec-golden`.
- `TURBO_BUNDLES`: bundles made by `turbo-bundle make`.

```sh
export MODELS_DIR=... GOLDEN_DIR=... TURBO_BUNDLES=...
bench/static-peers/peers.sh setup      # toolchains, builds, model copies, text sets
bench/static-peers/peers.sh accuracy   # each tool against the goldens, all nine models
bench/static-peers/peers.sh score      # the same vectors scored again, without running the tools
bench/static-peers/peers.sh timing     # texts/s per tool, model, text set and batch
```

`setup` downloads Go 1.25.1, Zig 0.16.0, the Rust, Go and Zig dependencies, and a Python virtual environment with Model2Vec and sentence-transformers. It installs nothing system-wide. Everything goes under `$W` (default `~/static-peers`), except the Zig dependency, which Zig unpacks into `zig/zig-pkg` (ignored by git). No tool downloads a model: each one reads copies of the files in `MODELS_DIR`, because statembed writes beside the tokenizer and go-potion rewrites `model.safetensors` in place.

The timing sets come from the golden texts: the texts as they are (`golden`), 2000 texts of 45 words (`t60`), and 1000 chunks of 256 words (`w256`). `TIMED` and `BATCHES` override the models and batch sizes timed. Timing means something only on an idle machine with the same load for every tool.

Each line `accuracy` prints gives: the rows equal to the golden to the bit, the texts refused, the worst cosine, the largest absolute difference, and how many rows fall under cosine 0.99999 and under 0.999. Up to ten of the rows under 0.999 follow, each with both norms and the start of its text.
