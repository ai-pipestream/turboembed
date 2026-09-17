# NVIDIA M4 qualification, 2026-09-16 (Machine A)

This receipt records roadmap M4 for the NVIDIA provider: the M0–M2 contract
gates, packaging, and the matched native-overhead measurement, all run live
on Machine A (`krick`, NVIDIA GeForce RTX 4080 SUPER, driver 595.84, ORT
1.28.0 `da9b5e3`, CUDA 13 user-space libraries). Apple is deliberately **not**
qualified here; its gates are listed in
[apple-m4-machine-c-checklist.md](apple-m4-machine-c-checklist.md).

## The prepared-path gap, stated honestly

The prepared SDK (`turboembed_prepared_v1_*`,
[native-sdk.md](native-sdk.md)) is implemented over OpenVINO and supports
Intel devices only; there is no CUDA device in its ABI. NVIDIA M4
qualification therefore runs through the frozen
[`turboembed.h`](../include/turboembed.h) text ABI over ONNX Runtime's CUDA
EP — the path all existing NVIDIA receipts use — brought through the same
gate structure as M1/M2: matched overhead against a direct native baseline,
installable packaging with a clean-consumer acceptance, explicit provisioning
pins, and fail-loud device policy. Extending the prepared ABI itself to CUDA
remains future work and is not claimed.

## Matched native overhead (M1-style pilot)

`make bench-nvidia-overhead` (`crates/bench-turbo/src/bin/bench-nvidia-overhead.rs`)
compares two implementations of the same synchronous text-to-vector request
on identical model revision (`all-MiniLM-L6-v2` @ `1110a243`, f32), tokenizer,
mean+L2 postprocessing, fixed `[batch, 256]` execution shape, and CUDA
device 0:

- **Direct baseline:** a raw `ort` 2.0.0-rc.13 consumer — CUDA EP, IoBinding,
  reusable host token buffers, preallocated host hidden output, host mean+L2.
  The stock checkpoint exposes only `last_hidden_state`, so a direct consumer
  reads back `[batch, seq, dim]`; that is recorded, not hidden.
- **ABI path:** `turboembed.h` through the safe Rust `Engine` (CUDA EP +
  IoBinding + DEVICE mean+L2 kernel + mapped PINNED `[batch, dim]` result).

Grid: batches 1/8/32 × token targets 32/128/256 × full and deterministic
mixed-length rows (18 cases), 20-execution warmup, 3 repeats per case with
alternating order, each repeat capped at 10 s or 10,000 requests. Gates were
predeclared from the [library design](library-design.md#native-performance-acceptance):
ABI p50 within 5% of direct, throughput ≥ 95%, parity max abs ≤ 5e-4 and
RMSE ≤ 1e-4.

**Result — all 54 repeats passed**
([receipt](../testdata/receipts/bench/machine-a-nvidia-overhead.json)):

- ABI p50 ranged from **3.6% to 17.9% faster** than direct ORT
  (p50 ratio 0.8211–0.9643); throughput 103.5%–122.5% of direct. The ABI wins
  because device pooling reads back 384 floats instead of the full hidden
  state.
- Numerical parity direct-vs-ABI: max abs error `8.94e-8`, max RMSE
  `1.22e-8` across all 18 cases.
- Steady-state ABI counters (arena allocs, `gpu_external_alloc`, forward
  cudaMalloc, token H2D, hidden D2H) were zero in every case.
- Sample counts ran 680–10,000 per leg; legs under 1,000 samples (the batch-32
  cases) record p99 as descriptive only, so this pilot makes no tail claim
  there.
- A two-engine concurrent observation (batch 8, mixed 128) measured
  per-engine p50 4278 µs at ~225 req/s each, symmetric across engines —
  recorded observation, not a scalability acceptance.

### Defect found and fixed by the pilot

The first pilot run **failed** its gates: the ABI was ~2× slower than direct
ORT at batch 1 (p50 1457 µs vs 760 µs). Stage bisection localized ~0.8 ms per
request to `turbo_buffer_arena_rent`, which re-ran the CUDA backend probe —
`cudaGetDeviceCount` + `cudaSetDeviceFlags` + `cudaSetDevice` +
`cudaGetDeviceProperties` — on **every** rent of the per-request PINNED
result row. The probe is now memoized
(`native/turbo_buffer/src/cuda.cpp`); device presence and mapped-host
capability are static per process and the fail-loud behavior is unchanged.
The Metal and Level Zero backends already memoize via `init_once` and were
not touched. After the fix the ordinary Machine A SOLIDIFY bench improved
from p50 1504 µs to **749 µs** for `embed_one("minilm", "hello world")`
([machine-a-cuda.json](../testdata/receipts/bench/machine-a-cuda.json),
re-run via `make bench-machine-a`, goldens unchanged). The 108 native
`turbo_buffer` tests, including the live CUDA PINNED/DEVICE rent-reuse
proofs, pass after the change.

No golden was modified to make any gate pass.

## M0 contracts re-proven live on this tree

- **CUDA allocator isolation under multiple live engines:** the previously
  ignored `ort_allocator::tests::cuda_engines_release_external_pool` ran live
  (two MiniLM engines, eight requests each from separate threads, matched
  outputs within 1e-5, pool destroyed only after the final lease and
  outstanding allocation) in 717 ms, plus the three CPU-backed ownership unit
  tests. See [cuda-allocator-isolation.md](cuda-allocator-isolation.md) for
  the design.
- **Full NVIDIA suite:** `make test-turboembed-nvidia` (all ignored tests
  included, with `TURBOEMBED_TOKENIZER_JSON` pointing at the pinned MiniLM
  tokenizer) passed: CUDA IoBinding goldens (cosine 1.000000, refreshed
  [nvidia-minilm.json](../testdata/receipts/turboembed/nvidia-minilm.json)),
  explicit CPU EP goldens
  ([nvidia-minilm-cpu.json](../testdata/receipts/turboembed/nvidia-minilm-cpu.json)),
  ORT TensorRT EP goldens
  ([nvidia-minilm-tensorrt.json](../testdata/receipts/turboembed/nvidia-minilm-tensorrt.json)),
  device policy (AUTO/CUDA/TensorRT never silently CPU), ownership, options,
  input bounds, and the native WordPiece contract suite.

## M2-style installable packaging

`scripts/make-nvidia-sdk-release.sh` builds
`dist/turboembed-cuda-sdk-<version>-linux-x86_64.tar.gz` (+ `.sha256`) from a
clean staging prefix:

- `lib/libturboembed.so.1` — the frozen `turboembed.h` ABI from the new
  `crates/turboembed-cabi` cdylib (`--features ort-cuda`; ONNX Runtime core
  statically linked), plus the dlopened ORT CUDA/TensorRT provider plugins.
- Symbol gate: all 15 header functions must export at the versioned
  `TURBOEMBED_1` node and nothing else may leak except the recorded internal
  `turboembed_ort_*` hooks (rustc force-exports `#[no_mangle]` items past the
  version script; this allowlist difference from the Intel exact-match gate
  is recorded in the packaged `exported-symbols.txt`).
- `include/turboembed.h`, a CMake package (`TurboEmbed::turboembed`), the
  external consumer example, an explicit catalog template (no developer
  cache paths), SHA-256 model pins for the qualified MiniLM source, the
  Apache-2.0 license, `README-runtime.md` naming what the host must provide
  (NVIDIA driver, libcudart, cuBLAS/cuDNN 9; TensorRT 10 only for the TRT
  device), and a hashed `sdk-manifest.json` over every installed file.

`scripts/nvidia-sdk-consumer-acceptance.sh <tarball> <model-dir> gpu <cuda-libs>`
**passed on Machine A** in a fresh temp directory with a minimal environment:
archive and manifest hashes verified; provisioned model files verified
against the packaged pins and a byte-flipped model **rejected** by that
verification (the `turboembed.h` loader does not re-hash at load time — a
recorded difference from the Intel prepared bundle loader, enforced instead
at the provision/verify step); external CMake consumer built against the
extracted prefix only; loader resolved `libturboembed.so.1` from the prefix;
explicit CPU run and default CUDA run both embedded two texts at 384 dims
with unit norms; five bounded wall-clock CUDA repeats at ~0.6 s per full
process run. `cpu-only` mode (CUDA must fail loudly) is wired for GPU-less
hosts. Model provisioning uses the existing SHA-pinned
`cargo run -p inferstream-fetch` flow (`make fetch-embeddings ALIASES=minilm`).

Make targets: `make nvidia-sdk-release`, `make nvidia-sdk-acceptance MODE=gpu`.
No artifact was published to a hosted registry.

## Model coverage: one deliberately different contract

With MiniLM green, M4 adds **bge-small-en-v1.5 (CLS pooling + L2)** on
NVIDIA — different pooling and a different (Xenova) tokenizer at the same
384 dims, against the committed CLS goldens
(`testdata/e2e/goldens/nvidia/bge-small.json`), fetched through the pinned
manifest. The ignored live test
`nvidia_bge_small::bge_small_cls_ort_cuda_matches_golden` passed on
Machine A: cosine 1.000000 on all `parity:*` items, L2 norms within 1e-3 of
1.0, and a mean-pooling request on the CLS alias returns an explicit
`NotImplemented` — no silent substitution
([receipt](../testdata/receipts/turboembed/nvidia-bge-small.json)).
Multilingual inputs and further contracts remain future expansions per the
roadmap's one-at-a-time rule.

## Support matrix after M4 (what each provider actually ships)

| Provider | Surface | Status | Evidence |
|---|---|---|---|
| Intel OpenVINO | `turboembed_prepared_v1_*` SDK archive + Rust/Java FFM adapters; `turboembed.h` GenAI path in-tree | M2 packaged preview (GPU receipts on Machine B; CPU package path in CI) | [native-sdk.md](native-sdk.md) |
| NVIDIA CUDA | `turboembed.h` over ORT CUDA EP (+ explicit CPU EP, ORT TensorRT EP); installable `turboembed-cuda-sdk` archive; MiniLM (mean+L2) and bge-small (CLS+L2) | **M4-qualified on Machine A** (this receipt): conformance, allocator isolation, matched overhead, packaging acceptance | this page |
| Apple Metal | Swift `libTurboEmbed.dylib` + MLX in-tree; M0 ownership receipts | **Gated on Machine C** — live Metal proofs, matched overhead, packaging are open gates; no qualification claimed | [checklist](apple-m4-machine-c-checklist.md) |

The prepared-token/device-result surface remains Intel-only; NVIDIA ships the
text ABI with prepared-style guarantees measured above; external buffer/queue
import and asynchronous submission remain unsupported on all providers.

## Re-run everything on Machine A

```bash
make test-turboembed-nvidia          # full suite incl. ignored live tests
make turbo-buffer-tests              # native arena tests (live CUDA)
make bench-machine-a                 # SOLIDIFY receipt
make bench-nvidia-overhead           # matched overhead pilot (~20 min)
make nvidia-sdk-release
make nvidia-sdk-acceptance MODE=gpu
make fetch-embeddings ALIASES=bge-small
cargo test -p turboembed --features ort-cuda --test nvidia_bge_small -- --ignored --nocapture
```
