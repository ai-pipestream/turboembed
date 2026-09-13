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
| CUDA | PINNED, DEVICE | `cudaHostAlloc`, `cudaMalloc` | Machine A (structure + fail-loud here) |
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

**TurboRerank (CPU proven here):** engine arena at create. Load rents
BERT scratch (f32 activations + WordPiece i32) and the work token
buffer. `turborerank_buffer_alloc` rents from a process arena of the
same ABI. `forward` must not increment the counter. GPU token
workspaces use the same rent path (CUDA PINNED / ZE SHARED / Metal
SHARED) when that backend is live.

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
make turbo-buffer-tests              # alignment, dual-rent, double-free, GPU fail-loud
make turboembed-mock-arena-tests     # mock embed allocs/forward == 0
make turborerank-tests               # includes the above + Berlin band when weights exist
```

Reintroducing `posix_memalign` for Rerank scratch or work tokens fails
`turbo_buffer_arena_owns` in the live CPU score test.

## Machine A follow-up

CUDA PINNED + DEVICE are implemented and compile-gated
(`TURBO_BUFFER_CUDA`). This cloud run has no GPU: create/rent is
`NOT_IMPLEMENTED`. Machine A should run `make turborerank-tests` with
nvcc and confirm CUDA token rent + DEVICE activation scratch + Berlin
goldens. Open a Machine A task only after CPU arena tests here are green.
