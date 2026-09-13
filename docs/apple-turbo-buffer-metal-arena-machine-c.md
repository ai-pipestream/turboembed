# TurboBuffer Metal SHARED arena (Machine C)

Live proof that `include/turbo_buffer.h` Metal `SHARED` rent/return is
real MTL (`MTLResourceStorageModeShared`), and that TurboRerank Metal
forward uses those arena token slots. No private `newBufferWithLength`
for tokens. No Swift-side token malloc.

| Field | Value |
|---|---|
| Host | Machine C, Apple M2, Metal |
| Tree | `origin/main` at `1f2aa93` plus this Machine C proof |
| Command | `make test-turborerank-apple` |
| Receipt | `testdata/receipts/turborerank/apple-minilm-l6.json` |

## What was proven

1. **`turbo_buffer_arena_create(METAL)`** succeeds. Probe of
   `SHARED` is `OK`. `DEVICE` / `PINNED` on a Metal arena are
   `NOT_IMPLEMENTED` (not remapped to HOST).
2. **Rent / return is real MTL.** A SHARED i32 view is 64-byte
   aligned, `turbo_buffer_metal_owns(ptr) == 1`, and
   `turbo_buffer_metal_lookup` returns a non-null `id<MTLBuffer>`.
   Caller writes through `contents`. HOST on Metal aliases SHARED
   (same unified-memory buffer).
3. **Slab reuse.** After return of two SHARED rents of the same
   shape, a second pair increments `turbo_buffer_alloc_counter` by
   **0**. Double-return is `DOUBLE_FREE`.
4. **TurboRerank Metal forward rents those slots.** Engine create
   attaches a Metal arena. Load rents work tokens (`input_ids` /
   mask / types / positions) and WordPiece scratch from that arena.
   `turbo_buffer_arena_owns` and `turbo_buffer_metal_owns` are both
   true. After warmup, `score` / `forward` see **allocs == 0**.
5. **Berlin HF band.** Identity logits match
   `testdata/reference_rerank/ms_marco_minilm_l6_berlin.json`
   within `2e-3`. Receipt `pass=true`.
6. **No silent bypass.** A CPU `posix_memalign` buffer passed to a
   Metal engine fails loud (no host copy). `metal_api.mm` looks up
   tokens only via `turbo_buffer_metal_lookup`. The old private
   `SharedRegistry` / `metal_shared_alloc_bytes` path is gone.
7. **Swift / lib.** `TurboRerankEngine.score` calls
   `turborerank_score` (engine work buffer). The Swift client has
   no token-buffer allocator. `make libturborerank-apple` archives
   `arena.cpp` + `metal.mm` into `libturborerank_apple.a`; SPM
   links that archive — it does not `@_cdecl` a second token path.

TurboEmbed `libTurboEmbed.dylib` still does **not** rent Metal
compute buffers from this arena. That is item (4), not a fake
Metal success.

## Commands

```bash
make turbo-buffer-tests          # SHARED rent/return/reuse + fail-loud
make turborerank-tests           # Metal Berlin + arena_owns + allocs/forward==0
make turborerank-tests-nometal   # Metal create fails loud
make turborerank-apple-receipt   # writes apple-minilm-l6.json
make libturborerank-apple        # Swift-linkable archive (same objects)
```
