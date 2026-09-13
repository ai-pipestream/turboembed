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
| ZE | HOST, SHARED, DEVICE | Level Zero USM | **LIVE** Machine B (`docs/turbo-buffer-ze-machine-b.md`) |
| Metal | SHARED (HOST aliases SHARED) | `MTLResourceStorageModeShared` | **LIVE** Machine C (`docs/apple-turbo-buffer-metal-arena-machine-c.md`) |

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

**TurboRerank:** engine arena at create. Load rents BERT scratch
(f32 activations + WordPiece i32) and the work token buffer.
`turborerank_buffer_alloc` rents from a process arena of the same ABI.
`forward` and `score` (after load) must not increment the counter.
GPU token workspaces use the same rent path (CUDA PINNED / ZE SHARED /
Metal SHARED) when that backend is live.

**CUDA (Machine A LIVE):** PINNED token rows (`cudaHostAlloc`) and
DEVICE activation / token scratch (`cudaMalloc` at load). Steady-state
`forward` must see `turbo_buffer_alloc_counter() == 0` and
`turbo_buffer_cuda_forward_allocs() == 0`. Tests fail if a per-forward
`cudaMalloc` / `cudaHostAlloc` returns for those slots. One packed
int32 H2D per row still happens (SOLIDIFY item 2 — not claimed zero).

**ZE (Machine B LIVE):** HOST / SHARED / DEVICE USM
(`docs/turbo-buffer-ze-machine-b.md`). OpenVINO GPU tokens are SHARED.
`ze_query` must report the requested type. DEVICE is proven with a
Level Zero memcpy, not a host stand-in.

**Metal (LIVE on Machine C):** `turbo_buffer_arena_create(METAL)` +
SHARED rent allocates `MTLResourceStorageModeShared`. HOST aliases
SHARED. TurboRerank Metal `forward` binds those MTLBuffers via
`turbo_buffer_metal_lookup` — no private token `newBufferWithLength`.
Proof: [`docs/apple-turbo-buffer-metal-arena-machine-c.md`](apple-turbo-buffer-metal-arena-machine-c.md).

OpenVINO CPU without Level Zero rents a **CPU** arena for host
tensors. That is not a ZE success — `turbo_buffer_arena_create(ZE)`
is still `NOT_IMPLEMENTED` / `UNAVAILABLE` on this binary.

**TurboEmbed:** every engine owns an arena. Mock/CPU `embed` rents
host FP32 `[n_texts, dim]` from a CPU arena (load warms 32×8). ORT /
GenAI result copies rent when those features are on. **Metal (Machine C
LIVE, SOLIDIFY 4):** `libTurboEmbed.dylib` creates a Metal arena,
rents SHARED i32 tokens + f32 last-hidden + f32 results, and wraps
those MTL contents as MLX arrays. `turbo_buffer_metal_lookup` only —
no private registry. After load, `allocs/forward == 0`. Proof:
[`docs/apple-turboembed-metal-arena-machine-c.md`](apple-turboembed-metal-arena-machine-c.md).
Device graphs inside ORT / GenAI / mlx-swift layer ops stay with those
runtimes.

## Tests

```bash
make turbo-buffer-tests              # alignment, dual-rent, double-free; CUDA/ZE when live
make turbo-buffer-intel-receipt      # Machine B ZE HOST/SHARED/DEVICE
make turboembed-mock-arena-tests     # mock embed allocs/forward == 0
make turborerank-tests               # includes the above + Berlin band when weights exist
make test-turborerank-intel          # Machine B OV + ZE receipts
make test-turborerank-apple          # Machine C: Metal SHARED live + Berlin receipt
make test-turboembed-apple           # Machine C: libTurboEmbed Metal SHARED + MiniLM receipt
```

Reintroducing `posix_memalign` for Rerank scratch or work tokens fails
`turbo_buffer_arena_owns` in the live CPU score test. Reintroducing a
private MTL token alloc on Machine C fails `turbo_buffer_metal_owns`
and `turbo_buffer_arena_owns` in the Metal score test.

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
