# Direct-native reference programs

The native half of the matched benchmark protocol (PLAN.md section 11):
for each provider, the runtime driven the way its own users drive it, on
the exact workload the `libturbo` run used, writing a receipt of kind
`native` that `turbo-bench compare` sets against the `libturbo` receipt.
No libturbo code is in any timed path.

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
3. `turbo-bench compare --turbo turbo.json --native native.json --out
   compare.json` matches the cells (the prepared-token path when both
   sides have one, else the text path; total and decode rate for
   generation), prints `libturbo` throughput as a fraction of native per
   cell, and gives the verdict: SUPPORTED when every cell reaches the
   floor (0.95) and nothing is unmatched. The comparison receipts are
   committed under `testdata/receipts/turbo/bench/compare-*.json`.

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
| openvino / OpenVINO 2026.3.1 C++ | Ryzen 9 9950X3D CPU (krick) | embed, 9 | 0.91x to 1.24x; four cells under the floor | EXPERIMENTAL |
| ggml / llama.cpp CUDA | RTX 4080 SUPER (krick) | generate, 128 tokens | 0.99x total, 1.00x decode | SUPPORTED |
| ggml / llama.cpp CUDA | RTX 4080 SUPER (krick) | embed, 9 (text path) | 0.94x to 1.88x; two cells under the floor by about 30 us of per-call overhead | EXPERIMENTAL |
| hailo / hailortcli | Hailo-8 (pi5ai1) | embed, 6 | 1.00x | SUPPORTED |
| metal / the same kernels | Apple M2 (krickert-mac) | embed, 9 | 0.98x to 1.00x before and 0.97x to 1.01x after the simdgroup matmul (the provider adds no measurable cost around the kernels) | SUPPORTED |

Where `libturbo` is faster than the native loop it is because the
provider keeps the hidden state on the device and pools with its own
kernels or graph, while a plain user of the runtime copies the whole
hidden state back and pools on the host; the comparison is of what a
user of each runtime would get for the same work.
