# Reference code: the system of truth for "fastest"

A benchmark receipt is only a claim about speed if it names what it was
measured against. The reference programs under `reference/` and the
capability cells they qualify compare `libturbo` with the vendor's own
stack and, where a faster implementation exists, with that. Those stacks
are read and built from pinned checkouts under `/work/reference-code`,
one per line below. A receipt that cites a reference names the checkout's
commit, so "at or above 0.95 of native" means 0.95 of a known program at a
known commit, not of a memory.

The pin matches what the tree runs where the tree pins something (the
onnxruntime the cuda provider links, the OpenVINO release on the B70, the
HailoRT releases on the two Pi boards) and the newest release otherwise.
Change a pin only together with the receipts that were measured against
it.

| Checkout | Repository | Tag | Commit | Reference for |
|---|---|---|---|---|
| `onnxruntime` | microsoft/onnxruntime | v1.28.0 | `da9b5e364c465de65c49d91e696cd6485270757f` | The runtime the cuda provider links through the `ort` crate; the `reference/ort-cuda` loop |
| `llama.cpp` | ggml-org/llama.cpp | b11130 | `183d2a04c2a666187598abb771918cfbb40dcebb` | The ggml provider's engine and GGUF generation; the `reference/llama-cpp` loop links the vendored copy in `llama-cpp-sys-2`, which carries no build tag |
| `openvino` | openvinotoolkit/openvino | 2026.3.1 | `759c5a6ab8c066af5f4bc5ebd04643706012a37d` | The openvino provider's runtime on the B70 and Intel CPUs; the `reference/openvino` loop |
| `openvino_tokenizers` | openvinotoolkit/openvino_tokenizers | 2026.3.1.0 | `77e4637387cea167c998fa27b1245af9e04d2911` | Tokenization as an OpenVINO graph op: where Intel's pipeline fuses the tokenizer |
| `openvino.genai` | openvinotoolkit/openvino.genai | 2026.3.1.0 | `56d9685302da2fc5cc7c9689cfab500fd0660a02` | Intel's own fused text-embedding pipeline: the fastest known path on Intel GPU |
| `hailort` | hailo-ai/hailort | v5.1.1 | `1adb3306b57f0bc44166a5ecb1b87c4c116ae0e7` | The Hailo-10H runtime on the AI HAT+ 2; `hailortcli run2` is the native loop |
| `hailort-4.23` | hailo-ai/hailort | v4.23.0 | `08f088d3b443c7846af067269ce998c6d5d91449` | The Hailo-8 runtime shipped in Raspberry Pi OS `hailo-all` |
| `text-embeddings-inference` | huggingface/text-embeddings-inference | v1.9.4 | `e80ef225ed0e6cb1717ce632a6a84b6cf211bb67` | The fastest known embedding server on NVIDIA GPUs (candle kernels, device pooling); the bar for the cuda provider beyond the onnxruntime loop |
| `candle` | huggingface/candle | 0.11.0 | `31f35b147389700ed2a178ee66a91c3cc25cc80d` | The kernels under text-embeddings-inference |
| `mlx` | ml-explore/mlx | v0.32.2 | `1f8e74e3f12f31365464a6867c6579f0e9b29d85` | Apple's own array framework: the fastest known path on Apple silicon, the bar for the metal provider |
| `CTranslate2` | OpenNMT/CTranslate2 | v4.8.2 | `d44d2d069eb88c7b7804da864c10c201501cb4a9` | A fused transformer encoder on CPU and CUDA with no framework in the loop; a second bar for CPU embedding |
| `TensorRT` | NVIDIA/TensorRT | v11.3 | `98adec82349b3ae22aa3f753d733b9a1a84d497d` | The open-source plugins and samples (fused BERT layers); NVIDIA's own fastest path |
| `model2vec-rs` | MinishLab/model2vec-rs | v0.3.0 | `a66e495a49fe1a22c87a87774d889301ae1cdbab` | Static embeddings: the bar for the static provider |
| `cudf` | rapidsai/cudf | v26.08.01 | `ee6d3d2564869c910f8efa93c6668cf0f50ae5bb` | `nvtext` WordPiece on the GPU: tokenization as a device stage on NVIDIA |

Present before this list and not re-pinned, recorded at the commit found:

| Checkout | Describe | Commit | Reference for |
|---|---|---|---|
| `tokenizers` | backup/train_encode_split-52-ge97fa4f8 | `e97fa4f86ae4ef1e590b45374532572a35ef1ac3` | The tokenizer the core's native WordPiece and BPE were checked against |
| `sentencepiece` | v0.2.2pre1 | `e0cce7d37b065b5140349dbe12c6bcf6192fdd78` | SentencePiece models |
| `djl` | v0.24.0-292-g700d59d48 | `700d59d48be8b596368a63b71b5790a6707e8103` | The DJL repository format (PLAN.md P11, deferred) |
| `opennlp` | opennlp-2.3.2-494-gb453b9ee3 | `b453b9ee3f2856a1dbf323ad13c8465a1203d3f0` | The chunking fallback, native-image branch in progress |
| `open_model_zoo` | 2024.6.0-41-g7cc29a914 | `7cc29a91472b4cb1289a11e655ba3e188e1d4a31` | Intel model recipes |
| `model_server` | v2023.0-1658-gbc8d8897 | `bc8d8897e4bddd53fe9c02e20cb767a5bcd02d98` | OpenVINO Model Server, a serving comparison |
| `faiss` | v1.15.0-29-g02ea14372 | `02ea14372c9983f3eaa15698512180dc62ea234d` | Vector search, not in scope |

Re-pin a checkout in this second table the first time a receipt cites it.

## How the checkouts were made

Shallow clones at the tag (`git clone --depth 1 --branch <tag>`), so the
commit is the tag's commit and the history is not present. A deeper
history is fetched on demand (`git fetch --unshallow`).
