# Direct-native reference programs

The native half of the matched benchmark protocol (PLAN.md section 11):
for each provider, the runtime driven the way its own users drive it, on
the exact workload the `libturbo` run used, writing a receipt of kind
`native` that `turbo-bench compare` sets against the `libturbo` receipt.
No libturbo code is in any timed path.

The OpenVINO program also has attribution knobs, off by default:
`--static` compiles the model reshaped to the cell's `[batch, seq]`,
`--fuse` pools and normalizes inside the graph, `--i32` declares the
inputs i32 through the pre-processor. Together they reproduce the
provider's graph, so a gap between the plain loop and the provider can
be split into what the graph choices cost and what the provider adds;
on the Ryzen CPU the graph choices are a gain of about 4 percent, and
the gap that remained was the provider's.

| provider | reference | what it drives |
|---|---|---|
| `cuda` | `reference/ort-cuda` (Rust) | ONNX Runtime's CUDA execution provider through the `ort` crate, the crate the provider links; input tensors per run, hidden state read back, pooling and L2 on the host |
| `openvino` | `reference/openvino` (C++) | OpenVINO's C++ API (`ov::Core`, `compile_model`, an infer request per run) with the same LATENCY and f32 hints; pooling and L2 on the host |
| `ggml` | `reference/llama-cpp` (Rust) | llama.cpp through `llama-cpp-2`: greedy decoding with the end token suppressed for `generate`, llama.cpp's own tokenizer and pooling for `embed` |
| `hailo` | `reference/hailo/native-receipt.py` | `hailortcli benchmark` on the HEF (HailoRT's own tool); its streaming FPS is the rows per second |
| `metal` | `reference/metal` (Objective-C++) | the provider's Metal kernels driven by a plain program (Metal has no vendor runtime), so the comparison measures what the provider adds around them |

## Protocol

1. Run the `libturbo` side with a token dump:
   `turbo-bench embed ... --dump-tokens tokens.json --out turbo.json`.
   The dump holds every cell's texts, ids, mask and lengths, the bundle's
   identity (manifest and artifact hashes) and its pooling and
   normalization, so the native side runs the same rows and names the
   same bundle. A GGUF bundle has no core tokenizer: its dump carries the
   texts alone and the reference tokenizes them, as the provider does.
2. Run the reference on the dump: `reference-ort-cuda --tokens tokens.json
   --out native.json`, `reference-openvino --tokens tokens.json --device
   GPU --commit <sha> --out native.json`, `reference-llama-cpp embed
   --tokens tokens.json --out native.json`, `reference-metal --tokens
   tokens.json --commit <sha> --out native.json`; for generation,
   `reference-llama-cpp generate --gguf model.gguf --turbo-receipt
   turbo.json --out native.json`; for Hailo,
   `reference/hailo/native-receipt.py --hef model.hef --turbo-receipt
   turbo.json --commit <sha> --ssh pi5ai1 --out native.json`.
   Both sides warm up by the same rule before a cell's timed samples: at
   least `--warmup` iterations and at least 0.5 s of them
   (`turbo_bench::receipt::warm_up`; the C++ programs carry the same
   loop). A count alone left the first cells of a run measuring a GPU
   still raising its clocks, which read as a few percent against
   whichever side ran first.
3. `turbo-bench compare --turbo turbo.json --native native.json --out
   compare.json` matches the cells (the prepared-token path when both
   sides have one, else the text path; total and decode rate for
   generation), prints `libturbo` throughput as a fraction of native per
   cell, and gives the verdict: SUPPORTED when every cell reaches the
   floor (0.95) and nothing is unmatched. The comparison receipts are
   committed under `testdata/receipts/turbo/bench/compare-*.json`.
   A receipt names the commit of the build that produced it; turbo-bench
   and the Rust references will not write one from a tree with
   uncommitted changes (the commit would end in -dirty and that tree
   cannot be rebuilt), and `compare` lists such a receipt under `dirty`
   and marks the comparison EXPERIMENTAL. Run both back to back: on a passively cooled device (the
   M2) a long cell at the end of a ten-minute run is measured on a
   throttled GPU, and a reference run on a cooler one minutes later
   reads a few percent faster for that cell alone.

The Rust references are workspace members (`cargo build -p
turbo-reference-ort-cuda`, `cargo build --release -p
turbo-reference-llama-cpp --features cuda` with `CUDA_PATH` set as for the
ggml provider); the C++ ones build with cmake and make as their
directories say; the Rust ones record the build's commit through the
`turbo-bench` library, the others take `--commit`.

## Results (2026-09-22)

| pair | device | cells | libturbo versus native | verdict |
|---|---|---|---|---|
| cuda / ONNX Runtime 1.28 CUDA EP | RTX 4080 SUPER (krick) | embed, 9 | 1.04x to 2.64x | SUPPORTED |
| openvino / OpenVINO 2026.3.1 C++ | Battlemage B70 (krick-1) | embed, 9 | 1.15x to 1.55x | SUPPORTED |
| openvino / OpenVINO 2026.3.1 C++ | Ryzen 9 9950X3D CPU (krick) | embed, 9 | 1.03x to 1.30x (the first run read 0.91x on the larger cells: the provider's token writer built an error message per token, about 1 us each, fixed the same day; `compare-openvino-krick-cpu-embed-2026-09-22b.json`) | SUPPORTED |
| ggml / llama.cpp CUDA | RTX 4080 SUPER (krick) | generate, 128 tokens | 0.99x total, 1.00x decode | SUPPORTED |
| ggml / llama.cpp CUDA | RTX 4080 SUPER (krick) | embed, 9 (text path) | 0.98x to 1.90x (the first run, 0.94x on two cells, had the reference tokenizing outside its timed loop and a count-only warm-up; `compare-ggml-krick-gpu-embed-2026-09-22b.json` is the matched one) | SUPPORTED |
| hailo / hailortcli | Hailo-8 (pi5ai1) | embed, 6 | 1.00x to 1.01x (`compare-hailo-pi5ai1-embed-2026-09-22b.json`, re-run from commit 5cb4cad; the first pair named no commit) | SUPPORTED |
| metal / the same kernels | Apple M2 (krickert-mac) | embed, 9 | 0.98x to 1.00x before and 0.97x to 1.01x after the simdgroup matmul (the provider adds no measurable cost around the kernels) | SUPPORTED |

Where `libturbo` is faster than the native loop it is because the
provider keeps the hidden state on the device and pools with its own
kernels or graph, while a plain user of the runtime copies the whole
hidden state back and pools on the host; the comparison is of what a
user of each runtime would get for the same work.
