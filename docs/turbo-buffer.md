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
| CUDA | PINNED, DEVICE | `cudaHostAllocMapped`, `cudaMalloc` | Machine A **LIVE** (mapped tokens, 0 id H2D) |
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

**CUDA (Machine A LIVE):** PINNED mapped token rows
(`cudaHostAllocMapped`) and DEVICE activation scratch (`cudaMalloc`
at load). Host tokenize / pack writes the PINNED pages; kernels read
`turbo_buffer_cuda_mapped_device_ptr`. Steady-state `forward` must see
`turbo_buffer_alloc_counter() == 0`,
`turbo_buffer_cuda_forward_allocs() == 0`, and
`turbo_buffer_cuda_forward_h2d_bytes() == 0`. Tests fail if a
per-forward `cudaMalloc` / `cudaHostAlloc` returns, or if a token-row
`cudaMemcpy` H2D is reintroduced. Unmapped pointers fail loud — there
is no convenience H2D.

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

**TurboEmbed ORT (Machine A LIVE):** CUDA / AUTO / TensorRT engines own a
CUDA arena. Tokens are PINNED mapped i64 rows (stored as i32×2),
hidden states are DEVICE f32, result rows are PINNED. IoBinding
binds those views. The CUDA EP `gpu_external_alloc` hook rents DEVICE
slabs so ORT intermediates are arena-owned. Load warms I/O at
max batch × max seq plus a pair of `[1, dim]` result slabs (so holding
one `Embeddings` while embedding again does not malloc). After
that warmup, `turbo_buffer_alloc_counter() == 0` and
`gpu_external_alloc` calls == 0 on the next embed. Mean+L2
(mask-weighted) runs on DEVICE into the mapped PINNED result row.
Activation `d2h_hidden_bytes` is 0. The caller reads the 384-d row
from mapped PINNED (`result_host_bytes` ≪ hidden volume).

Explicit CPU EP owns a CPU arena: HOST tokens, HOST hidden, HOST
results, same IoBinding reuse. Mock still warms a 32×8 HOST slab.

**TurboEmbed GenAI (Machine B LIVE):** GPU / AUTO engines open a **ZE**
arena. Load rents i32 token rows and f32 hidden scratch as **SHARED**.
`embed` rents the FP32 result from the same arena. Infer wraps those
pointers with `ov::Tensor(..., usm)`. After warmup,
`allocs/forward == 0` for those slots. CPU / OPENVINO_CPU rents **HOST**
(ZE HOST when L0 is present, else a CPU arena — that is not a ZE GPU
success). GPU create without ZE SHARED fails loud — never a CPU arena.
See [`docs/turboembed-genai-ze-machine-b.md`](turboembed-genai-ze-machine-b.md).

**TurboEmbed Metal (Machine C LIVE, SOLIDIFY 4):** `libTurboEmbed.dylib`
creates a Metal arena, rents SHARED i32 tokens + f32 last-hidden +
f32 results, and wraps those MTL contents as MLX arrays.
`turbo_buffer_metal_lookup` only — no private registry. After load,
`allocs/forward == 0`. Proof:
[`docs/apple-turboembed-metal-arena-machine-c.md`](apple-turboembed-metal-arena-machine-c.md).

## Tests

```bash
make turbo-buffer-tests              # alignment, dual-rent, double-free; CUDA/ZE when live
make turbo-buffer-intel-receipt      # Machine B ZE HOST/SHARED/DEVICE
make turboembed-mock-arena-tests     # mock embed allocs/forward == 0
make test-turboembed-intel           # Machine B GenAI ZE SHARED + MiniLM ≥0.99
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
`compute.h2d_per_row` is `0`. The receipt writer fails if a
steady-state CUDA `score` observes any intercepted HostToDevice
bytes.
