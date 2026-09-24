# Audit: providers, reference programs, benchmark harness, receipts (2026-09-23)

Read-only audit of `providers/`, `native/`, `reference/`, `crates/turbo-bench`,
`tools/turbo-bundle`, `testdata/receipts`, `scripts/` and `packaging/` on
branch `turbo-v2` at commit 7dd68aa, against the mission and the fifteen
rules in `AGENTS.md`. Delivered as written by the auditor; one personal
machine name in the evidence is replaced with a placeholder. Twelve
findings known before this audit (the three WordPiece tokenizers, the
double vtable crossing, the wrapper crates, hailo on vstreams, the
re-export crate, the capability bit and placement and provenance overlap,
the two server surfaces, Resolve and Select as two calls, the protobuf
catalog, P11, stored compare verdicts, the CPU path through ONNX Runtime)
are not repeated here.

Ranked from most costly to the philosophy to least.

1. **The CUDA SUPPORTED embed cell measures ONNX Runtime against ONNX Runtime.**
   - Evidence: testdata/receipts/turbo/bench/cuda-rtx4080-embed-2026-09-23.json:14 has `runtime_version: "onnxruntime ORT Build Info: git-branch=rel-1.28.0..."`. The native side, native-ort-cuda-rtx4080-embed-2026-09-23.json:12-14, is also ORT CUDA EP. The cell is providers/cuda/src/lib.rs:191-201.
   - The reference is also weakened. reference/ort-cuda/src/main.rs:144-200 has no IoBinding: every timed iteration clones ids, mask and types, uploads them from the host, reads [b,s,h] back and pools on the host. The ratios of 1.00 to 1.15 therefore measure "with IoBinding" against "without IoBinding", not the direct kernels section 0 promises. TensorRT, the fastest known path, was never compared.
   - Rules 3 and 7.
   - Cut: set the cell to EXPERIMENTAL until the receipt's runtime names the provider's own kernels and the reference is the fastest known path (TensorRT, or at minimum ORT with IoBinding).

2. **The Metal benchmark compares the provider with a copy of itself.**
   - Evidence: reference/metal/main.mm:40 includes providers/metal/src/kernels.inc, and lines 380-431 copy `encode_batch` (provider.mm:751-890). compare-metal-mac-embed-2026-09-22d.json names the runtime "the provider's kernels driven directly". reference/README.md says Metal "has no vendor runtime", which is false: MPSGraph, Core ML and MLX exist.
   - The kernels this validates are naive: attention materialises [b,h,s,s] with one dot product per thread (kernels.inc:324); softmax runs one thread per row (:353); LayerNorm makes three serial passes per token (:85).
   - Rule 7.
   - Cut: delete the copied forward pass, write reference/metal against MPSGraph or MLX, and set the cell to EXPERIMENTAL until then.

3. **Seven of nine conformance and precision receipts name a commit that did not run them.**
   - Evidence: cuda-2026-09-21.json:58-59, cuda-jetson:70-71, ggml:43-44, hailo:91-92, metal:86-87, openvino-minilm:59-60 and openvino-tasks:2-3 all have a `commit_note` saying the commit was filled in afterwards over a placeholder. They are hand-written, with no shared schema.
   - Every SUPPORTED cell cites one: cuda lib.rs:193, ggml lib.rs:225, openvino provider.cpp:1208 and 1215, hailo provider.cpp:1098, metal provider.mm:1074.
   - The metal one predates the simdgroup matmul now on the default path (cf43b8e, provider.mm:809), and records `status: EXPERIMENTAL` with `cosine_floor: 0.0`, while the cell reports 0.9995.
   - The hailo one also says EXPERIMENTAL.
   - openvino-b70-2026-09-22.json:81 records every model group failing with BUNDLE_NO_ARTIFACT, and it is still cited for two SUPPORTED cells.
   - Rule 7.
   - Cut: have turbo-conformance write machine-generated receipts stamped with the build commit, and demote every cell until they exist.

4. **The OpenVINO and ggml references are misconfigured, and the "wins" come from that.**
   - OpenVINO: reference/openvino/main.cpp:40-47 has `--fuse`, `--static` and i32 off by default. The native side runs a dynamic graph, reads the hidden state back and pools on the host (285-330). With i32 on, it builds `ids_copy` inside the timed loop (278). Every native-openvino receipt says "host-side mean pooling" (for example native-openvino-b70-gpu-embed-2026-09-22.json:279), and the flags are not recorded. The B70 ratios of 1.15 to 1.56 therefore measure the reference's copy.
   - ggml: reference/llama-cpp/src/main.rs:361-366 sets `n_ubatch = b*s` (8192), while the provider caps it at 4096 (providers/ggml/src/lib.rs:978, 1193). That produces the 1.89x at 32x256 (compare-ggml-rtx4080-gpu-embed-2026-09-22c.json:115-119).
   - Metal adds the reverse bias: the reference memcpys the output inside the timed loop (main.mm:434) and libturbo does not.
   - Rule 7.
   - Cut: make the fastest configuration the reference default, record the flags in the receipt, and re-issue every verdict.

5. **The Hailo native receipt is computed from one number, in Python.**
   - Evidence: reference/hailo/native-receipt.py:73-91 turns one `hailortcli` streaming FPS into every cell, so p50 = mean = min = max, and `iters` is derived rather than counted. It copies the device name, bundle identity and `live_tokens_per_row` from the turbo receipt (:79, 99-100), which defeats compare's same-device and same-bundle check (crates/turbo-bench/src/main.rs:1292-1302). Pipelined throughput is set against per-call latency.
   - The seq-32 cells equal seq-128 because the HEF frame is fixed.
   - Line 69 records `uname -n` over `--ssh <host>`.
   - compare-hailo-pi5-hailo8-embed-2026-09-22b.json lacks the `kind` and `dirty` fields.
   - Rules 7, 13 and 14.
   - Cut: write a C++ HailoRT timing loop per cell, delete the script, and drop the seq-32 cells.

6. **Results cross the bus but are reported as zero copies.**
   - Evidence: `CudaBuffer::read_to_host`, providers/cuda/src/lib.rs:411-437, copies device memory into a 1 MiB pinned bounce buffer (417-422), then `copy_nonoverlapping` into the destination (434), with a stream sync every MiB (432). `d2h` counts only `read_back` (1305-1314).
   - turbo-bench reads the result in every timed iteration (main.rs:459, 479), yet `d2h_bytes: 0.0` appears in the cuda, metal and openvino receipts (e.g. cuda-rtx4080-embed-2026-09-23.json:60, 92, 124). providers/cuda/tests/provider.rs:335 and 358 assert the zero.
   - `host_allocs` and `provider_allocs` are null in all 29 receipts (main.rs:357-365 passes them through). OpenVINO reports `UINT64_MAX` "not counted" (provider.cpp:1579).
   - Hailo counts f32 host bytes as `dma_bytes_per_row`.
   - Rules 3 and 12.
   - Cut: count every read in `d2h_bytes`, copy once into the destination, and have turbo-bench refuse a receipt with a null counter.

7. **SUPPORTED is claimed for hardware never measured.**
   - Evidence: cuda: any discrete GPU (`!p.integrated`, lib.rs:191-192). ggml: any device whose name contains "CUDA", including Jetson, which cuda itself keeps EXPERIMENTAL (lib.rs:222-223). openvino: any GPU and any CPU (provider.cpp:1204-1210); the CPU cell cites the B70 conformance receipt (:1215). metal: any Apple GPU (provider.mm:1072).
   - Rule 7 ("on a named architecture"); matching on the device name is also a heuristic (rule 6).
   - Cut: key each cell on the architecture label its receipts name.

8. **Placement says device where the host does the work.**
   - Rerank POSTPROCESS is reported DEVICE or FUSED, but `return_sorted`/`top_n` reads the scores back and `std::stable_sort`s on the host: cuda lib.rs:821-829 vs 1554-1566; openvino provider.cpp:673 vs 996-999; metal provider.mm:575 vs 1000-1003.
   - Hailo reports ENCODE as DEVICE (709) while the word and position embedding, LayerNorm and bias build run on the host (875-917).
   - Rules 3 and 12.
   - Cut: set placement per run, and add a lookup stage for Hailo.

9. **Per-call copies and allocations on every hot path.**
   - CUDA (lib.rs): tokens are written as i32 at full seq, then widened to i64 into pinned staging (1090-1092, 1121-1142); all-zero token types are uploaded every call (1151), although `DeviceMem::zero` at cuda.rs:279 is unused; `shape().to_vec()`, the outputs vector and `spans.clone()` are allocated per run (1214-1218, 1544-1571); three TensorRefMut are rewrapped per run (1158-1173); `Activation::None` does a device-to-device logits copy (1274, 1287).
   - OpenVINO (provider.cpp): each row is tokenized twice, three times on left truncation (817, 843, 851); `upload()` always writes `max_batch*max_seq` (933-946); `queue.finish()` stalls the host before a synchronous `infer()` (946), on a private queue (267); no shape buckets: 10 tokens compute 512 (744); `write_tokens` stages the ids, then `upload()` copies them again (1434).
   - Metal (provider.mm): each row is tokenized twice (687, 714/722) and the prefix re-tokenized per row (683); attention always runs at the session's `max_seq` (674-678, 755, 842); two `copyf` passes per layer (844, 861), a separate `add_bias` (823-829) and three QKV GEMMs (845-847); per-row dispatches for `output_dim` and the reranker head (920-929, 947-970); `write_tokens` memcpys per row despite HOST_PTR_IMPORT (1313-1316).
   - Hailo (provider.cpp:988-1004): per row, a 196 KB gather, the 64 KB bias rebuilt, f32 frames quantized on the host (552, 556), and a blocking read with no pipelining.
   - ggml: it loads the GGUF a second time `vocab_only` just to tokenize (lib.rs:448-515); three allocations per row (1289, 480, 514); host L2 normalization before `output_dim` cuts (1426-1446).
   - CUDA pooling kernel: thread 0 scans the mask serially and every thread rereads it per dimension (kernels.cu:48-69).
   - Rule 3 ("slower than the hardware's best is a bug").
   - Cut: tokenize once into mapped or pinned buffers at the used width, upload `rows*used_seq`, bucket shapes, fuse bias and QKV, and batch per-row dispatches.

10. **ggml ships an unreceipted second tokenizer.** `llama_tokenize` through `VocabModel` (ggml/src/lib.rs:448-515) bypasses the core tokenizer, and there is no equivalence receipt. This is separate from the three known WordPiece copies. Rule 11. Cut: tokenize through the core, or commit the equivalence receipt.

11. **Provider scaffolding is copied instead of shared.**
    - openvino provider.cpp:211-407 and hailo provider.cpp:205-409 are nearly verbatim: `checked_truncate`, `packed_desc`, `dtype_size`, `describe_into`, `make_buffer`, `import_buffer`, `Buffer`, `release<T>`.
    - `encode_row` exists three times (openvino 800-873, hailo 778-846, metal provider.mm:674).
    - The token-range check appears twice (openvino 1420-1429, hailo 1310-1319).
    - BIO span aggregation appears twice (openvino 1037-1117, cuda lib.rs:1328-1345).
    - `Bundle::read` (native/turbo_provider_common.hpp:313-364) is a second bundle.json parser.
    - safetensors is parsed four times: tools/turbo-bundle/src/safetensors.rs, native safetensors.hpp, scripts/export-hailo-tables.py, scripts/gen-reference-tasks.py.
    - "2.0.0-alpha.0" is hard-coded three times (openvino 1183 and 1597, hailo 64).
    - Rule 9.
    - Cut: move buffers and row packing into provider_common, span aggregation into the core, and pass the parsed contract through the vtable.

12. **turbo-bench compare does not enforce the receipt protocol.**
    - It accepts `commit: "unknown"`: metal-mac-embed-2026-09-22.json:10 and the metal rerank receipt.
    - It accepts a `-dirty` commit: openvino-rtx4080-cpu-embed-2026-09-22.json, committed despite `TURBO_BENCH_ALLOW_DIRTY` (receipt.rs:260).
    - The 22d metal and b70-22b compares have no `dirty` field.
    - It never checks that both sides ran the same commit or the same token ids (main.rs:1289-1302). It skips the text path when a token dump exists (1324-1327).
    - `Counter::Words` uses a 0.75 words-per-token heuristic (296-311).
    - `native_reference: "not run..."` is written into every receipt (701), and the module doc says the references are "still to be written" (40-43).
    - The `--floor` help says throughput while the code compares p50 latency.
    - Rules 6, 7, 9 and 11.
    - Cut: reject unknown or dirty commits, require a token-dump hash on both sides, compare the text path, and delete the stale field and the heuristic.

13. **Private names and paths are in committed files and history.**
    - An agent scratchpad path under `/tmp/`: native-openvino-rtx4080-cpu-embed-2026-09-22.json:279 and 22c:297, compare-openvino-rtx4080-cpu-embed-2026-09-22.json:5 and 23, native-llama-cpp-rtx4080-gpu-embed-2026-09-22c.json:232, native-ort-cuda-rtx4080-embed-2026-09-23.json:304.
    - A checkout path under `/work/`: native-llama-cpp-...-generate-22c.json:53.
    - A home-relative stash path: cuda-jetson-2026-09-21.json:14.
    - A personal laptop's host name: testdata/receipts/bench/machine-c-metal.json:8, turboembed/apple-minilm.json:5.
    - `/work/models`: models/manifests/*.
    - Commit 8d7b6dc's message lists the whole host mapping, and that commit also hand-edited 43 generated receipts.
    - scripts/gen-reference-tasks.py:486 defaults `--machine` to gethostname().
    - The label `cm5-hailo8` (hailo-2026-09-21.json:68) and the `mac` filenames are not on the rule 14 list.
    - Rule 14.
    - Cut: write basenames only, require TURBO_BENCH_MACHINE, delete the v1 receipt dirs (40 files, about 3.5 MB), and reword 8d7b6dc before the tree goes public.

14. **Dead code and tests that cannot fail.**
    - native/turbo_buffer: 3,578 lines of v1 code that no build references. Its README points at missing docs and a missing Makefile, and tools/write_intel_ze_receipt.cpp:138 writes "Machine B".
    - native/wordpiece tests are not built by any target, `return` silently on missing fixtures (wordpiece_tests.cpp:44-46, 128-129, 250-251), and read `INFERSTREAM_ROOT`.
    - The metal pipelines `zerof` and `softmax_row` are compiled but never dispatched (provider.mm:126, 132).
    - Stale "every cell EXPERIMENTAL" docs sit above Supported returns (cuda lib.rs:28-29, ggml lib.rs:29). The ggml test covers only the CPU cell (1606-1620).
    - Rules 7 and 9.
    - Cut: delete turbo_buffer and the dead pipelines, and wire the wordpiece tests to fail on a missing fixture.

15. **The static provider misreports itself.**
    - It reports "tokenizers 0.23" (providers/static/src/lib.rs:17-19, 87, 144-148) while using `turbo_core::tokenizer` (34).
    - `truncate_dims` is stored and never checked, so any `output_dim` is accepted (251, 276, 295, 344).
    - `write_tokens` panics instead of erroring on out-of-bounds input (323-339).
    - Test fixture builders are `pub` in the cdylib (423-466).
    - Rules 2, 6, 9 and 12.
    - Cut: report the core tokenizer, check the bundle's dims, and move the fixtures under cfg(test).

16. **Silent defaults.**
    - cuda lib.rs:813-814 substitutes max_seq 512 and max_batch 32 when the bundle omits them.
    - The Hailo DETERMINISTIC bit (provider.cpp:197, 1090) rests on one prose sentence.
    - Rules 2, 6 and 7.
    - Cut: fail with bundle_invalid, and clear the bit until a repeat-run field exists.

17. **Python outside the allowed places.** scripts/export-hailo-tables.py produces a bundle artifact, not a committed file, and runs on the host outside a container. Rule 13. Cut: fold it into `turbo-bundle import` on the existing safetensors reader.

Not reached: turbo-bundle import.rs internals and whether its contract parse duplicates core bundle.rs; packaging/ Dockerfile digest pinning and scripts/package.sh; the ggml generation-path allocations; the native/wordpiece encode hot path; the metal, openvino and hailo test suites beyond skip behaviour; whether the core rejects options whose capability bits a C++ provider leaves clear; whether turbo-bench's session max_seq equals the dump's seq.
