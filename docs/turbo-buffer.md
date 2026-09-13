# TurboBuffer — unified device-buffer arena

Frozen C ABI: [`include/turbo_buffer.h`](../include/turbo_buffer.h).
Apple copy (keep identical): `swift/Sources/TurboBufferC/include/turbo_buffer.h`.

TurboEmbed and TurboRerank **rent** i32 token rows and f32 activations
from one arena type. Engines do not `malloc` those buffers on
steady-state forward. The alloc counter is process-wide; tests reset it
before `forward` / mock `embed` and require `0`.

## Backends

| Backend | Placement | Alloc | Proof host |
|---|---|---|---|
| CPU | HOST | 64-byte `posix_memalign` | this cloud run |
| CUDA | PINNED, DEVICE | `cudaHostAlloc`, `cudaMalloc` | Machine A **LIVE** |
| ZE | HOST, SHARED, DEVICE | Level Zero USM | Machine B |
| Metal | SHARED (HOST aliases SHARED) | `MTLResourceStorageModeShared` | Machine C |

Missing compile → `NOT_IMPLEMENTED`. Compiled but no device →
`UNAVAILABLE`. Neither remaps to CPU. PINNED on a CPU arena (etc.) is
`NOT_IMPLEMENTED`, not a silent HOST.

## Rent / return

1. `turbo_buffer_arena_create(device)` — one backend per arena.
2. `turbo_buffer_arena_rent(dtype, placement, rows, cols, row_stride)` —
   reuse a free slab of the same placement when `bytes` fit. New slabs
   increment `turbo_buffer_alloc_counter`.
3. `turbo_buffer_arena_return` — double-return is `DOUBLE_FREE`.
4. Views are `[rows, cols]` with `row_stride` in **elements**.
   `row_stride == 0` pads each row to 64 bytes. TurboRerank tokens pass
   `stride == seq` so OpenVINO keeps `row_stride == seq`.

Slab table capacity is 256 (fixed). Rent after warmup does not grow it.

## What each engine rents

**TurboRerank:** engine arena at create. Load rents BERT scratch and
the work token buffer. `turborerank_buffer_alloc` rents from a process
arena of the same ABI. `forward` and `score` (after load) must not
increment the alloc counter.

**CUDA (Machine A LIVE):** PINNED token rows (`cudaHostAlloc`) and
DEVICE activation / token scratch (`cudaMalloc` at load). Steady-state
`forward` must see `turbo_buffer_alloc_counter() == 0` and
`turbo_buffer_cuda_forward_allocs() == 0`. Tests fail if a per-forward
`cudaMalloc` / `cudaHostAlloc` returns for those slots. One packed
int32 H2D per row still happens (SOLIDIFY item 2 — not claimed zero).

OpenVINO CPU without Level Zero rents a **CPU** arena for host
tensors. That is not a ZE success — `turbo_buffer_arena_create(ZE)`
is still `NOT_IMPLEMENTED` / `UNAVAILABLE` on this binary.

**TurboEmbed:** every engine owns a CPU arena for host FP32 result
rows. Mock/CPU `embed` rents `[n_texts, dim]`. Load warms a 32×8 slab
so the next embed of that shape is 0 allocs. ORT / GenAI result copies
also rent when those features are on. Device compute buffers stay with
ORT / GenAI / MLX until item (4) — the hooks (`engine->arena`,
`#include "turbo_buffer.h"`) are already in the stub.

Swift `libTurboEmbed.dylib` on Machine C does not yet link this
arena. That is a documented gap for (4), not a fake Metal success.

## Tests

```bash
make turbo-buffer-tests              # alignment, dual-rent, double-free; CUDA PINNED+DEVICE when nvcc+GPU
make turboembed-mock-arena-tests     # mock embed allocs/forward == 0
make turborerank-tests               # includes the above + Berlin band when weights exist
```

Reintroducing `posix_memalign` for Rerank scratch or work tokens fails
`turbo_buffer_arena_owns` in the live CPU score test.

## Machine A (this proof)

CUDA PINNED + DEVICE rent/return is **LIVE** (`TURBO_BUFFER_CUDA`,
nvcc + a CUDA device). `make turbo-buffer-tests` and
`make turborerank-tests` on Machine A must take the live branch — not
`NOT_IMPLEMENTED`, not a CPU stand-in, not an interim host copy of the
BERT graph.

`make test-turborerank-nvidia` refreshes
`testdata/receipts/turborerank/nvidia-minilm-l6.json` (Berlin HF band).
H2D of packed int32 ids/mask/types/pos per row is still present; do
not read this receipt as zero host-to-device bytes.
