# CUDA allocator ownership

ORT's external allocation callbacks receive a byte count or pointer, without
a session context. TurboEmbed previously stored the most recently loaded
engine's arena globally. Loading another engine changed where callbacks rented
and returned memory; closing an engine could leave callbacks with freed storage.

The callback pool now owns a separate CUDA arena. Each CUDA session holds a
lease, released after its ORT session and allocator fields. A synchronized
registry tracks every outstanding allocation. The arena is destroyed only
after the final session lease and final outstanding allocation are released.
Failed loads release their lease through the same ownership path. Engine input,
hidden-state, and output arenas remain independently owned.

This is the existing device-0 CUDA path. ORT 1.28 selects its default stream
when external allocator callbacks are supplied and rejects combining those
callbacks with a user compute stream. TurboEmbed's pooling kernels also use
the default stream. Other devices or external streams require their own
context and stream-aware reuse design before support is advertised.
[ORT 1.28 CUDA provider](https://github.com/microsoft/onnxruntime/blob/v1.28.0/onnxruntime/core/providers/cuda/cuda_execution_provider.cc#L317-L335),
[external allocator options](https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html#gpu_external_allocfreeempty_cache).

The pool inherits TurboBuffer's 256-slab capacity and reports allocation failure
when exhausted. Its cached memory lives as long as any session or outstanding
allocation needs the pool. Pool reuse does not establish zero process allocation
or parallel GPU execution; those require separate measurements.

## Validation on 2026-09-14

Three allocator-state unit tests passed: two-session ownership with late free
and pool recreation, release after a load without allocations, and overflow-safe
byte rounding. These use CPU-backed storage to test the ownership logic.

The live test `ort_allocator::tests::cuda_engines_release_external_pool` passed
on `krick`, NVIDIA GeForce RTX 4080 SUPER, driver 595.84, using ORT 1.28.0
(reported build commit `da9b5e3`). Two MiniLM engines ran eight requests each
from separate threads. Outputs matched the initial engine's output within
absolute error `1e-5`. The surviving engine remained usable after the first
engine closed; its retained result delayed final destruction. The test verified
zero session leases and no remaining callback pool after final result release.
The test completed in 0.492 seconds, including loading and warmup. This is a
bounded correctness observation, not a throughput or overhead benchmark.

```bash
cargo test --locked -p turboembed --features ort-cuda --lib ort_allocator \
  -- --skip cuda_engines_release_external_pool

# Set LD_LIBRARY_PATH to the installed CUDA/cuDNN library directory.
timeout --signal=TERM 180 cargo test --locked -p turboembed \
  --features ort-cuda --lib ort_allocator::tests::cuda_engines_release_external_pool \
  -- --ignored --exact --nocapture
```

The local CUDA libraries were in `/work/inferstream/.libs/nvidia/lib`. Model:
`sentence-transformers/all-MiniLM-L6-v2`, catalog snapshot
`1110a243fdf4706b3f48f1d95db1a4f5529b4d41`. SHA-256:

| File | SHA-256 |
|---|---|
| `onnx/model.onnx` | `6fd5d72fe4589f189f8ebc006442dbb529bb7ce38f8082112682524616046452` |
| `tokenizer.json` | `be50c3628f2bf5bb5e3a7f17b1f74611b2561a3a27eeab05e5aa30f411572037` |

This test checks lifecycle and numerical stability between engines, not tokenizer
conformance or independent model goldens. No existing goldens or receipts were
overwritten. TensorRT, Apple, and Intel are outside this allocator test's scope.

## Tokenizer follow-up, 2026-09-14

After native tokenizer validation was added, the optional ORT feature lint run
identified mutable token slices constructed from immutable slot references.
The private inference entry now requires a mutable session, and each token slice
borrows its distinct slot mutably. Safe engine calls already serialize access;
raw C callers remain subject to the single-engine serialization contract.
The fallback Hugging Face tokenizer is boxed once at model load to avoid a
large enum variant. No allocation was added to the inference loop by that change.

`cargo clippy --locked -p turboembed --features ort-cuda --lib -- -D warnings`
now passes. The same ignored CUDA two-engine allocator test was explicitly run
and passed again in 645.87 ms using ORT 1.28.0. The preceding invocation without
`--ignored` did not execute the hardware test and is not counted as validation.
