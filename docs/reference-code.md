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

## What the source says (read 2026-09-23, at the commits above)

Read from the checkouts, not built or measured. Paths are relative to
each repository root.

### Tokenization is a host stage on every stack

- OpenVINO tokenizers are `ov::op::Op` subclasses that implement
  `evaluate()` in host C++ (`src/ov_extension.cpp:72-106`,
  `src/wordpiece_tokenizer.cpp`, `src/bpe_tokenizer.cpp`,
  `src/unigram_tokenizer.cpp`, `src/sentence_piece.cpp`); the README
  states CPU only (`README.md:170`); compiling one for the GPU plugin
  fails at `openvino/src/plugins/intel_gpu/src/plugin/program_builder.cpp:210-231`
  with no fallback. openvino.genai hardcodes the tokenizer device to CPU
  (`src/cpp/src/tokenizer/tokenizer_impl.cpp:394`).
- onnxruntime has no GPU tokenizer; its only tokenizer op is a CPU
  splitter (`onnxruntime/contrib_ops/cpu/tokenizer.cc`).
- text-embeddings-inference tokenizes on host threads with the
  `tokenizers` crate (`core/src/tokenization.rs`).
- llama.cpp, MLX, CTranslate2 and HailoRT tokenize on the host; the
  GenAI HEFs carry a tokenizer as an external resource that also runs on
  the host (`hailort/src/genai/llm/llm.cpp:201-225`).
- The one GPU tokenizer is cudf's `nvtext::wordpiece_tokenize`
  (`cpp/include/nvtext/wordpiece_tokenize.hpp:106-111`, vocabulary from
  `load_wordpiece_vocabulary`, ids are row indices, a `cuco::static_map`
  in `cpp/src/text/wordpiece_tokenize.cu:34-58`). It needs
  `normalize_characters` first (`normalize.hpp:79-154`), returns a lists
  column of int32 ids with no mask, no padding and no special tokens
  (`wordpiece_tokenize.cu:847`), and every call takes an RMM stream and
  memory resource. Its BPE returns strings, not ids.

### Pooling and normalization can sit in the graph, except on Hailo

- openvino.genai inserts pooling (CLS: Slice and Squeeze; mean: mask
  Broadcast, Multiply, ReduceSum, Divide; last token: Slice or Gather)
  and `NormalizeL2` into the encoder graph with
  `PrePostProcessor::postprocess().custom`
  (`src/cpp/src/rag/text_embedding_utils.cpp:30-122,147-162`) on GPU and
  CPU. On the NPU a dynamic model goes through the NPUW path with
  pooling in a separate CPU model and the whole hidden state on the host
  (`src/cpp/src/rag/npu/text_embedding_pipeline.cpp:44-50`). No segment
  pooling. Its reranker appends Sigmoid or Softmax
  (`text_rerank_pipeline.cpp:61-86`).
- onnxruntime's CUDA provider fuses attention, embedding layernorm, skip
  layernorm and padding removal (`onnxruntime/contrib_ops/cuda/bert/`),
  registered for in-session fusion at
  `core/optimizer/graph_transformer_utils.cc:405-418`.
- TensorRT's open-source BERT plugins (`plugin/embLayerNormPlugin`,
  `plugin/skipLayerNormPlugin`, `plugin/bertQKVToContextPlugin`) are
  deprecated in favour of `INetworkDefinition::addAttention`; the fused
  attention cubins for head size 32 exist for sm75 and sm80 only, and
  whether sm89 dispatches to one or falls back to cuBLAS
  (`mhaRunner.cu:403`) is not settled by the source. No pooling or
  normalization plugin.
- text-embeddings-inference pools on the device (`models/flash_bert.rs`:
  CLS and last token 399-426, mean 441-461) over an unpadded batch with
  `cu_seqlens` (376-391), then copies to the host (`lib.rs:685`) and
  normalizes there in f32 (`core/src/infer.rs:288-302`).
- llama.cpp pools inside the ggml graph (`src/llama-graph.cpp:3704-3754`;
  mean is a matmul with an `inp_mean` matrix filled on the host per
  sequence id, 234-274, so one sequence id per chunk is segment pooling);
  L2 is host code (`common/common.cpp:1940`).
- MLX has Metal kernels for layer norm and RMS norm
  (`mlx/backend/metal/normalization.cpp`) and fused attention for head
  sizes 64 to 256 only (`scaled_dot_product_attention.cpp:661-671`), so
  MiniLM's head size 32 takes the composed path; pooling, segment pooling
  and L2 are compositions that stay on the GPU stream.
- HailoRT's only host post-process ops are argmax, softmax, NMS and the
  detection decoders (`src/net_flow/ops/`); quantize and dequantize run on
  the host (`libhailort/src/transform/transform.cpp:35-52`). Pooling and
  normalization are host stages on Hailo.
- CTranslate2 has only BERT's CLS gather and pooler dense on the device
  (`src/models/language_model.cc:388-399`); no mean pooling, no L2.

### Keeping an output on the device for the next stage

- onnxruntime: IO binding with a device-allocated output
  (`include/onnxruntime/core/session/onnxruntime_c_api.h:2883-2970`) and
  CUDA graphs (`cuda_provider_options.h:31`).
- OpenVINO GPU: remote tensors from `cl_mem` or USM
  (`src/inference/include/openvino/runtime/intel_gpu/ocl/ocl.hpp:286-379`);
  a remote input is used without a copy
  (`intel_gpu/src/plugin/sync_infer_request.cpp:900-907`), a remote
  output is not mapped to the host (515-518), and two compiled models
  share the default context when their plugin options match
  (`remote-tensor-api-gpu-plugin.rst:160-163`). USM host memory can slow
  a discrete GPU (`sync_infer_request.cpp:932-933`).
- MLX: arrays are lazy and stay in shared-mode Metal buffers
  (`backend/metal/allocator.cpp:15-16`); a host read is a pointer, not a
  copy (`array.h:375`).
- llama.cpp: no public API keeps the output in a backend buffer; every
  result is read back with `ggml_backend_tensor_get_async`
  (`src/llama-context.cpp:1546-1603`).
- TEI: every result is copied to the host.
- CTranslate2: the output `StorageView` stays on the device.

### Zero copy and the fastest loop per stack

- HailoRT 5.1.1: `ConfiguredInferModel::run_async` (C++ only; `hailortcli
  run2` defaults to it, `run2_command.cpp:383-390`); buffers page aligned
  and pre-mapped with `VDevice::dma_map` or `dma_map_dmabuf`
  (`vdevice.hpp:210,246`, `infer_model.hpp:85-131`). HailoRT 5 on the
  main branch serves Hailo-10 and Hailo-15 only; Hailo-8 is the `hailo8`
  branch, and the 5.x C API drops the ethernet and MIPI functions and
  changes the device capability struct, so the two providers stay
  separate builds.
- Intel NPU: static shapes only (`npu-device.rst:409-411`); batching is
  batch one with concurrent requests
  (`npu-device/batching-on-npu-plugin.rst:10-28`).
- The source gives no ranking on any machine; the loops to measure are
  listed in `PLAN.md` section 0, item R3.

## How the checkouts were made

Shallow clones at the tag (`git clone --depth 1 --branch <tag>`), so the
commit is the tag's commit and the history is not present. A deeper
history is fetched on demand (`git fetch --unshallow`).
