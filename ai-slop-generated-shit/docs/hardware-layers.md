# The layers each hardware offers

What each vendor exposes between the device and a program, lowest first,
and what each layer gives for the stages of this library's pipeline:
upload, tokenize, embedding lookup with layernorm, the QKV and MLP
projections, attention, residual with layernorm, pooling, L2
normalization, segment pooling, the sigmoid and softmax heads, download.
Read from the pinned checkouts in `docs/reference-code.md` and the
vendors' documentation on 2026-09-23. Nothing here was measured; every
"needs measuring" line is an open item for R3 and R4 in `PLAN.md`
section 0. The recommendation per family is where the direct path
(section 0: the lowest layer, no framework runtime) should sit.

## NVIDIA: RTX 4080 SUPER (sm_89) and Jetson Orin Nano (sm_87)

Installed on the x86 host at the time of reading: CUDA 12.4 from
distribution packages (nvcc 12.4.131, cuBLAS and cuBLASLt 12.4.5.8,
cuDNN 9.0.0, driver 595.84); no system TensorRT. The pins in
`docs/reference-code.md` expect CUDA 13.x, and JetPack 7.2 on the Orin
ships CUDA 13.2, cuDNN 9.20 and TensorRT 10.16 (from documentation, not
checked on the board). Library sizes on disk: libcublasLt 442 MB,
libcublas 110 MB, cuDNN about 1 GB, libnvinfer 633 MB plus 177 MB of
builder resources per SM.

Layers, lowest first:

1. **Driver API and runtime.** Closed. Upload and download, streams,
   CUDA graphs that capture the whole pipeline, `cudaMallocAsync` pools
   (`cuda-samples`: `streamOrderedAllocation`, `graphMemoryNodes`). On
   the Jetson the CPU and GPU share one DRAM, so pinned or managed host
   memory removes the copy (`simpleZeroCopy`); `prop.integrated` detects
   it. Pooling, L2, segment pooling, the heads and the embedding lookup
   are small custom kernels at this layer. Unverified: memory pool
   support on the Orin (`cudaDevAttrMemoryPoolsSupported`) and pinned
   memory coherence there.
2. **cuBLAS and cuBLASLt.** Closed. The QKV, output and MLP GEMMs. Lt
   epilogues fuse bias, ReLU, GELU and GELU with bias
   (`cublasLt.h:1125-1227` in the 12.4 header). Lt cannot fuse the
   residual add, layernorm or softmax. Whether its GELU is erf (BERT)
   or tanh is not stated in the header. TEI takes this route
   (`backends/candle/src/layers/cublaslt.rs:62,103`).
3. **cuDNN 9 through cudnn-frontend.** Library closed; the frontend is
   header-only, Apache-2.0. Graph nodes for sdpa, layernorm, rmsnorm,
   matmul, pointwise, reduction, softmax (`include/cudnn_frontend/node/`).
   SDPA forward in FP16 or BF16 needs SM80 or newer and a head dim that
   is a multiple of 8, capped at 128 on SM8x
   (`sdpa_support_surface.h:276,373,378-382`), so head sizes 32 and 64
   are admitted on both GPUs; whether the backend has a fast engine for
   32 on Ada or Orin is known only once a graph is built. Padding and
   sequence masks are supported. Cost: about 1 GB of libraries and a
   graph build per shape.
4. **CUTLASS v4.8.0.** BSD-3, header-only. Arch list includes 87 and 89
   (`CMakeLists.txt:176,182`). Epilogues fuse bias, erf GELU
   (`epilogue/thread/activation.h:574-591`, `linear_combination_gelu.h`)
   and the residual add through the EVT trees (`visitor_2x.hpp:317`).
   The fused multi-head attention for SM80 and later is
   `examples/41_fused_multi_head_attention/kernel_forward.h`: head dim as
   a template argument, variable sequence lengths (`seqstart_q_ptr`),
   no PyTorch. Runtime dependency is libcudart only; the cost is nvcc
   compile time per instantiation.
5. **flash-attention v2.8.3.** BSD-3. Forward kernels for head dims 32,
   64, 96, 128, 192 and 256, built for sm_80 and binary compatible with
   sm_86, 87 and 89 (`setup.py:181`,
   `flash_fwd_launch_template.h:204`). Upstream is tied to PyTorch
   (`flash.h:12`); candle 0.11.0 carries a PyTorch-free port under
   `candle-flash-attn/kernels/` with hdim32 and hdim64, which is the
   usable C++ route. Nothing in either repository says sm_87 by name.
6. **TensorRT.** Core closed; the v11.3 plugins and parsers are
   Apache-2.0. `IAttention` takes packed `[totalTokens, heads, dimHead]`
   in FP32, FP16 or BF16 (`include/NvInfer.h:6926-6948`). A plan is
   locked to the SM and TensorRT version it was built with unless
   restricted (`NvInfer.h:10406-10433`). The BERT plugins ship fused
   attention cubins for head 64 on sm80, sm86 and sm87; head 32 exists
   only for sm75 and sm80, and the variable-length path pads 32 to 64
   (`qkvToContextPlugin.cpp:861-865`). TensorRT is a graph compiler
   with its own runtime: a framework, and the bar to measure against,
   not the direct path. Cost: about 800 MB per SM plus per-device engine
   builds, and the pinned 11.3 headers do not match the 10.16 on the
   Jetson.
7. **cudf nvtext.** Apache-2.0. The only GPU tokenizer:
   `load_wordpiece_vocabulary` plus `wordpiece_tokenize` returns a lists
   column, no padding, no truncation, normalization as a separate
   `character_normalizer`; pulls in RMM and libcudf. A planned cell, not
   the path.

Per stage, the layer the direct path uses: upload and download at the
runtime with pinned memory and graphs; tokenize on the host (the core);
embedding lookup with layernorm, residual with layernorm, pooling, L2,
segment pooling and the heads as our kernels; the projections on
cuBLASLt with the bias and GELU epilogue (or a CUTLASS EVT epilogue that
also folds the residual); attention on the flash port for head sizes 32
and 64, with cuDNN sdpa and CUTLASS 41 as the alternatives to measure.

Recommendation: on sm_89 with head size 64, cuBLASLt GEMMs with the
GELU epilogue, the flash hdim64 varlen kernel, our kernels for the rest,
all captured in one CUDA graph; this is TEI's shape without its host
copy. For head size 32 (MiniLM), flash hdim32 or CUTLASS 41. On sm_87
the same with host buffers in pinned zero-copy memory. cuBLASLt is
already a dependency through llama.cpp.

Needs measuring: cuDNN sdpa against flash hdim32 against CUTLASS 41 on
both GPUs; whether the cuBLASLt GELU matches erf; a CUTLASS epilogue
with bias, GELU and residual against cuBLASLt plus a separate kernel;
whether a cuDNN fused layernorm engine exists on sm_87; graph replay
overhead on the Orin; compile times and kernel sizes of the flash port.

## Apple: M2

Metal and its compiler are closed; the metal-cpp headers are Apache-2.0.
The checkout under `/work/reference-code/metal-cpp` is a GitHub mirror
of Apple's download (developer.apple.com/metal/cpp is the source of
truth; MLX itself fetches `metal-cpp_26.zip` from there,
`mlx/CMakeLists.txt:228-231`).

Layers, lowest first:

1. **Metal and the shading language.** `StorageModeShared` places a
   buffer in unified memory, so upload and download are a memcpy or
   nothing; the provider does this (`providers/metal/src/provider.mm:7-12,232`).
   Indirect command buffers and argument buffers cut per-dispatch cost;
   neither is used today (one command buffer per run, then
   `waitUntilCompleted`, `provider.mm:902-976`). `simdgroup_float8x8`
   multiply-accumulate is the GEMM primitive through M2; MLX's steel
   GEMM is built on it (`kernels/steel/gemm/mma.h:46,205`). Metal 4
   (macOS 26) adds `MTL4CommandBuffer`, argument tables, `MTLTensor`
   (`Metal/MTLTensor.hpp:41-76`) and a machine learning command encoder;
   MSL 4 adds tensor ops (`mpp::tensor_ops::matmul2d`). Every stage is
   reachable; every kernel is ours to write.
2. **Metal Performance Shaders.** Closed. `MPSMatrixMultiplication`,
   `MPSNDArrayMatrixMultiplication`, `MPSMatrixSoftMax`, `MPSMatrixSum`.
   No fused attention, no layernorm, no GELU.
3. **MPSGraph.** Closed. A fused `scaledDotProductAttention` since
   macOS 15 with no documented head size limit, so it is the one Apple
   op that fuses attention at head size 32. No named layernorm op; it
   is composed from mean, variance and normalize, and whether the graph
   compiler fuses that chain is not documented. `compile` builds an
   executable per feed shape; `encode` may call `commitAndContinue` when
   mixed with our own dispatches.
4. **MLX v0.32.2.** MIT, C++ without Python. The full attention kernel
   accepts head dims 64, 72, 80, 96, 128, 192 and 256, the vector kernel
   64, 96, 128, 192 and 256 (`scaled_dot_product_attention.cpp:662-665,682-687`);
   head 32 takes the unfused fallback (744-746). The newer kernels that
   use the MSL 4 tensor ops are gated on macOS 26.2 and GPU generation
   17 or later (`device.cpp:947-964`), so they never run on an M2.
   `fast::layer_norm`, `quantized_matmul` and shapeless `compile` exist.
   As a dependency it brings FetchContent of metal-cpp, json and fmt,
   Accelerate linked public, and macOS 14 or later.
5. **Accelerate, BNNS and BNNSGraph.** Closed. CPU only; nothing found
   that reaches the GPU or the Neural Engine. Useful for CPU stages.
6. **Core ML and the Neural Engine.** Closed. There is no public ANE
   API; `MLComputeUnits.all` lets the OS place ops on the ANE, per op,
   at load time, and a Metal 4 machine learning pass can do the same,
   still from a Core ML model. No custom kernels, fixed or enumerated
   shapes to stay on the ANE (from coremltools guidance, not
   re-verified in Apple's documentation), outputs as `MLMultiArray` or
   caller-provided backings, no device-resident handle. Conversion
   needs coremltools, Python, so it runs in a pinned container.
7. **Interop.** An Objective-C++ file exporting `extern "C"` is the
   simplest C ABI provider and is what exists. Swift export adds the
   Swift runtime as a dependency.

Per stage on the M2: upload and download are zero copy with shared
storage on every layer above; tokenize is host everywhere; embedding
lookup with layernorm, GELU, residual, pooling, L2, segment pooling and
the heads are our kernels (MPSGraph composes them, MLX has ops for them,
Core ML runs them inside the model with no segment pooling); the GEMMs
are simdgroup matrices (ours or steel), `MPSMatrixMultiplication`, or
MPSGraph matmul; attention at head 32 is ours or MPSGraph, at head 64
ours, MPSGraph or MLX's fused kernel.

The provider's kernels as they stand: fp32 throughout
(`kernels.inc:1-8`); the GEMM tile loads straight from device memory
with no threadgroup staging or double buffering (168-196); attention
materialises the full score matrix with one thread per dot product
(324-351); softmax is one thread per row in three passes (353-378). The
gap to the fastest path is in how these are written, not in which layer
they sit on.

Recommendation: keep our own Metal kernels and rewrite them: half or
bfloat16 weights, a threadgroup-staged simdgroup GEMM (the steel tiling
is MIT and can be ported), and a fused attention kernel for head size
32, which no Apple layer other than MPSGraph fuses. MPSGraph's attention
is the one thing to measure against before deciding. MLX is a kernel
source and a benchmark reference, not a runtime dependency. The Neural
Engine is worth one cell as a separate provider through Core ML with
enumerated sequence buckets, knowing it cannot hold a resident result or
run segment pooling, and where its ops execute has to be measured.

Needs measuring on the M2: fp16 against fp32 simdgroup GEMM throughput;
MPSGraph attention against a fused kernel of ours at head 32 and 64
over sequence 128, 256 and 512; the per-call cost of an MPSGraph
executable; end-to-end Core ML latency and where the ops land. Not
verified from Apple's own pages: whether MPSGraph fuses composed
layernorm, the ANE shape rules, whether the Metal 4 machine learning
pass ever picks the ANE on an M2.

## Intel: Arc B70 (Battlemage), Intel CPUs, the Intel NPU

Layers, lowest first:

0. **Kernel driver and firmware.** `xe` and `i915` for the GPU,
   `intel_vpu` for the NPU, both in the Linux kernel. The NPU firmware is
   a binary blob (`linux-npu-driver`: `firmware/bin/mtl_vpu_v0.0.bin`).
   No pipeline stage runs here.
1. **Level Zero loader and headers** (`level-zero`, MIT). A dispatch
   layer routing `ze*` calls to the vendor drivers: command lists and
   queues, events and fences, USM host, device and shared allocations,
   `zeModuleCreate` from SPIR-V or native zebin
   (`include/ze_api.h:8301-8302`). GPU kernels can be launched from
   here without SYCL. The NPU graph extension header lives in a separate
   repository (`intel/level-zero-npu-extensions`), not in this checkout.
   Per stage: upload and download only; every compute kernel is ours.
2. **compute-runtime (NEO)** (MIT). One codebase with OpenCL 3.0 and
   Level Zero 1.17 front ends over a shared core; Battlemage supported
   (`README.md:39`, `shared/source/xe2_hpg_core/linux/product_helper_bmg.cpp`).
   Depends on the Intel Graphics Compiler and GmmLib; `ocloc` compiles
   OpenCL C to a Battlemage zebin ahead of time. USM is available on
   OpenCL too (`cl_intel_unified_shared_memory`), so `cl_mem` and USM
   exist on either API. Per stage: none, it is the substrate.
3. **oneDNN v3.13.2** (Apache-2.0). Battlemage listed
   (`README.md:119-120`). GPU runtime OCL, SYCL or ZE; only SYCL needs
   DPC++ (`doc/build/build_options.md:416-433`); ZE is marked
   experimental (`RELEASE_NOTES.md:57`). GPU kernels are OpenCL C plus
   nGEN, which emits native ISA, so no Intel LLVM for OCL or ZE.
   Primitives on GPU and CPU: matmul with post-ops (eltwise GELU, binary
   add for the residual), layer normalization (`src/gpu/intel/lnorm`),
   softmax, reduction. Graph API fused patterns: SDPA, optimized on GPU
   for f16 and bf16 with head dim up to 512 on XMX hardware
   (`doc/graph/fusion_patterns/sdpa.md:193-199`), gated MLP, norm
   fusions. The Graph API has OCL and SYCL interop only, no ZE
   (`include/oneapi/dnnl/`), so fused SDPA through it means the OCL
   runtime. The brgemm ukernel API is CPU only. Per stage: the
   projections, GELU, residual, layernorm and attention yes; embedding
   gather has no primitive found; L2 as reduction plus divide; the heads
   as eltwise and softmax; masked mean pooling as a matmul; tokenize no.
4. **oneMath v0.9** (Apache-2.0 interfaces). SYCL only
   (`README.md:111-160`); on Intel GPU the backend is closed oneMKL or
   generic SYCL BLAS. Adds nothing over oneDNN here and forces a DPC++
   build.
5. **SYCL and DPC++.** A kernel authoring path only, not needed: OpenCL
   C through `ocloc` or SPIR-V through `zeModuleCreate` covers it.
6. **OpenVINO 2026.3.1** (Apache-2.0). The GPU plugin embeds upstream
   oneDNN for fully connected, gemm, gated MLP and reduce
   (`src/plugins/intel_gpu/src/graph/impls/onednn/`). Its SDPA is built
   on oneDNN's gemmstone microkernels (`sdpa_gen_micro.cpp:901`) and
   needs xe_hpg or newer with microkernel support
   (`ocl_v2/sdpa/sdpa_opt.cpp:180-188`); Xe2 kernels use subgroup size
   16. Layernorm through `MVNFusion` (`transformations_pipeline.cpp:706`).
   `SDPAFusion` is registered in the common pipeline
   (`moc_transformations.cpp:263`); the CPU plugin disables it and
   builds a Snippets MHA subgraph instead (`intel_cpu/.../transformation_pipeline.cpp:873,1287`).
   Whether our BERT export becomes SDPA on the GPU was not verified.
   NPU plugin: the compiler is PLUGIN (`openvino_intel_npu_compiler`)
   for NPU4000, 5010 and 5020, DRIVER otherwise (NPU3720, Meteor Lake)
   (`compiler_adapter_factory.cpp:15-20`); execution is always the Level
   Zero graph extension; static shapes only; compute FP16 with U8 and
   INT8 weights (`npu-device.rst:113-116,412`); batching as batch one
   with concurrent requests. NPUW partitions large models and is aimed
   at LLMs, not needed for an encoder.
7. **linux-npu-driver v1.38.0** (MIT). `libze_intel_npu.so` implements
   core Level Zero plus the graph extension: `zeGraphCreate`,
   `zeGraphSetArgumentValue`, `zeAppendGraphInitialize`,
   `zeAppendGraphExecute`, `zeGraphGetNativeBinary`
   (`umd/level_zero_driver/api/ext/ze_graph.cpp:135-1126`). No
   `zeModuleCreate`, no kernels of any kind. Graph formats: NATIVE, a
   precompiled ELF blob checked against the firmware's version
   (`graph.cpp:741`, `docs/overview.md:430`), and NGRAPH_LITE, a
   serialized OpenVINO model compiled by the compiler in the driver,
   which is dlopen'd (`vcl_symbols.hpp:84`) and built from OpenVINO plus
   `openvinotoolkit/npu_compiler`. Client device code exists for vpu
   37xx, 40xx and 50xx only; data-centre parts were not found in it.

Conclusions:

- **B70.** The OpenVINO GPU plugin, resident: USM or `cl_mem` in, fused
  fully connected with post-ops, micro-SDPA, pooling through the
  PrePostProcessor. Its heavy kernels already are oneDNN, so oneDNN is
  not a layer beneath it for the matmuls and attention; going below
  OpenVINO can only win on glue: embedding gather fused with layernorm,
  fused masked pooling with L2 and the head, fewer launches per layer, a
  Level Zero immediate command list instead of the OpenCL queue. Finding
  out costs a profile of the OpenVINO graph first
  (`ov::enable_profiling` and the exec graph) to see the time outside
  the fully connected and SDPA kernels, then one encoder layer written
  on oneDNN's Graph API with the OCL runtime against the same `cl_mem`;
  one to two weeks for a single-layer comparison, worth it only if the
  non-GEMM share is large.
- **Intel CPU.** The OpenVINO CPU plugin: a forked oneDNN with Snippets
  MHA, its own brgemm kernels, and MLP and QKV fusions on x64
  (`transformation_pipeline.cpp:1152-1187`). Plain oneDNN lacks the MHA
  fusion. No evidence that anything below the plugin beats it.
- **NPU.** There is one compiler, Intel's NPU compiler, reached through
  OpenVINO or the driver, and one execution path, the Level Zero graph
  extension. The OpenVINO runtime can be skipped by calling
  `zeGraphCreate` with an NGRAPH_LITE model or a NATIVE blob and
  `zeAppendGraphExecute` on USM buffers, but the model is still an
  OpenVINO-serialized model and the compiler is still OpenVINO-derived,
  static shapes, FP16 compute. No kernel can be authored for the NPU.
  The session's buckets are the shapes compiled.

Not verified: the `ze_graph_ext.h` contents; the npu_compiler licence;
whether OpenVINO's ONNX reader runs `SDPAFusion` for our model; a oneDNN
GPU gather primitive; the maturity of oneDNN's ZE runtime on Battlemage;
data-centre NPU support in the driver.

## Hailo: Hailo-8 and Hailo-10H on the Raspberry Pi 5, with the Arm host cores

Layers, lowest first:

1. **PCIe driver and firmware.** The driver is GPL-2.0 (the separate
   `hailort-drivers` repository); the ioctl header is GPL-2.0 with the
   syscall note and MIT (`hailort/drivers/common/hailo_ioctl_common.h:1`);
   the device firmware is a closed binary. The driver exposes vDMA
   channels, descriptor lists, interrupt wait (`:555-597`), buffer map,
   unmap and sync for user pointers or dmabuf fds
   (`HAILO_DMA_DMABUF_BUFFER`, `:200-224`), and contiguous allocation
   (`:426-436`). Upload and download only. The Hailo-8 has no DRAM, so a
   network larger than on-chip memory runs as several contexts reloaded
   over PCIe; the Hailo-10H board carries 8 GB of LPDDR4X, so weights
   and intermediates can live in device memory and larger HEFs run
   without reloads. That changes capacity, not which ops a HEF can hold.
2. **HailoRT** (MIT; the GStreamer element LGPL-2.1). The C API has
   vstreams and buffer mapping including `hailo_vdevice_dma_map_dmabuf`
   (`hailort.h:2786-2876,3490-3626`) and no InferModel. The C++ API has
   `VDevice::dma_map` and `dma_map_dmabuf` (`vdevice.hpp:210-258`) and
   InferModel with `set_dma_buffer` (`infer_model.hpp:131`); HailoRT
   4.23 has the same dmabuf and `run_async` surface (`hailort-4.23`
   `vdevice.hpp:246`). The device emits UINT8 or UINT16 only; FLOAT32
   output and the NHCW to NCHW reorder are host code with no NEON
   intrinsics (`src/transform/transform.cpp:870-909,1279`). The 5.x
   genai path keeps token embedding lookup as a host-side uint16 table
   and feeds mask and RoPE as HEF inputs
   (`hailort_server/genai/llm/pre_process.hpp:52-141`). Upload, download
   and dequantize.
3. **Dataflow Compiler.** Closed, Hailo EULA, x86 Linux only. Takes ONNX
   or float and 16-bit TFLite (int8 TFLite rejected; TF1 deprecated
   since 5.1.0). Unsupported ops cut the graph and the cut part runs on
   the host. Quantization is 8-bit by default with per-layer `a16_w16`;
   the zoo's text encoders promote inputs, adds and convolutions to
   16-bit (`cfg/alls/generic/clip_vit_b_16_text_encoder.alls:5-11`);
   layernorm through `layer_norm_decomposition`. Matmul, softmax,
   layernorm and full attention blocks compile to the NPU: the zoo's
   CLIP text encoders run from after the embedding add to the last
   hidden state on hailo8, 8l, 10h and 15h
   (`cfg/networks/clip_vit_b_16_text_encoder.yaml:17-23`). Float outputs
   exist only through host dequantization. The supported-layer list and
   the GELU implementation are behind the gated Developer Zone.
4. **Model zoo v5.1.0** (MIT code; model licence per entry). No MiniLM
   or BERT entry for any target. Text encoders for 10H and 15H: CLIP
   B/16, B/32, L/14, RN50x4, SigLIP2, TinyCLIP
   (`docs/public_models/HAILO10H/HAILO10H_text_image_retrieval.rst:53-139`);
   the 10H table links `hailo15h` HEF files, a documentation error or a
   shared binary, not verified. The CLIP contract keeps embedding lookup
   with positional add, EOT padding, the EOT gather, projection and L2
   on the host (`core/preprocessing/text_preprocessing.py:6-17`,
   `core/postprocessing/text_encoding_postprocess.py:8-21`). The v2.19.1
   branch has `all_minilm_l6_v2` for hailo8 and hailo8l only: inputs
   `embedding` (16x8x384, 128 tokens) and an additive `masked_fill` mask
   (16x8x1536) put into the softmax by `set_input_mask_to_softmax()`,
   output `last_hidden_state` 1x128x384, 156 FPS at batch 1 and 742 at
   batch 8 (`cfg/base/all_minilm.yaml:14-47`,
   `cfg/alls/generic/all_minilm_l6_v2.alls:1`). Its host pre and post
   processing is not in the public branch.
5. **TAPPAS** (LGPL-2.1). GStreamer vision pipelines for HailoRT 4.23
   and 5.1.0. Nothing for text.
6. **KleidiAI v1.31.0** (Apache-2.0). Matmul and depthwise convolution
   microkernels only (`kai/ukernels/`); no layernorm, softmax or GELU.
   On the Cortex-A76 (Pi 5) and A78AE (Orin Nano), both Armv8.2-A with
   FEAT_DotProd and FP16 and without I8MM, BF16 or SVE: f32 NEON MLA,
   f16 NEON, and int8 dynamic activation against int8 or int4 weights
   with `neon_dotprod`. The `neon_i8mm`, bf16, SVE and SME variants do
   not run there. Worth it for the projections in a CPU fallback; of no
   use for pooling or L2, which are memory-bound reductions plain NEON
   handles.
7. **Arm Compute Library v53.3.0** (MIT). `NEGEMMLowp`, `NEMatMul`,
   `NESoftmaxLayer`, `NEMeanStdDevNormalizationLayer`,
   `NEL2NormalizeLayer`, an activation layer with GELU
   (`arm_compute/function_info/ActivationLayerInfo.h:65`). A large
   runtime with its own tensor and allocator model; only worth it if a
   whole fallback encoder runs outside ggml.
8. **llama.cpp b11130** (MIT). KleidiAI integration present, off by
   default (`ggml/CMakeLists.txt:153`), fetching KleidiAI 1.24.0; only
   MUL_MAT and GET_ROWS go through it (`kleidiai/kleidiai.cpp:1829`); on
   dotprod-only cores only the Q4_0 and Q8_0 kernel sets apply
   (`kleidiai/kernels.cpp:487,700,917`), the f32 path needs SME. F16 and
   F32 GGUF use ggml's own NEON code.

Stage placement on the Hailo-8, following the MiniLM-L6 contract: the
NPU runs QKV, masked attention, MLP with GELU, residual with layernorm,
at a fixed sequence of 128. The host runs tokenization, the embedding
lookup with the word, position and type sums (whether the embedding
layernorm is inside the HEF is not verified), mask construction,
quantize and dequantize, mean and segment pooling, L2, and the heads.
The Hailo-10H has the same split with room for 16-bit layers and larger
encoders.

Host stages on the Pi's and Orin's cores: plain NEON f32 loops for the
embedding gather, pooling, L2 and the heads, fused with dequantization
so HailoRT's FLOAT32 transform is skipped (request UINT16 and
dequantize in our own pass). KleidiAI dotprod kernels only for the
CPU-fallback matmuls. The GGUF fallback through llama.cpp with Q8_0 or
Q4_0 and `GGML_CPU_KLEIDIAI=ON`.

A Hailo-10H MiniLM HEF: no prebuilt one exists. The route is the DFC
5.x targeting `hailo10h` with the same ONNX the v2.19 zoo uses, cut at
the embedding and mask inputs, the v2.19 alls script reused, and a
calibration set. The recipe was written for DFC 3.x and hailo8; that
it compiles for hailo10h is plausible because the CLIP text encoders
prove the attention path, and unverified. It needs an x86 machine with
the EULA-gated compiler.

Needs measuring: host dequantization for UINT16 against FLOAT32; the
PCIe Gen2 x1 transfer of a 128 by 384 frame; batch 1 against batch 8
latency; scheduler overhead with several HEFs; 8-bit against 16-bit
accuracy versus the fp32 reference; Hailo-10H against Hailo-8 for the
same model. Not verified: the compiler's supported-layer list and GELU
handling, whether the hailo15h-labelled HEF files run on the 10H, the
MiniLM host pre and post processing, where `hailort_server` executes.
