# SOLIDIFY final-bench receipts

JSON written by a **live** `make bench-machine-*` on the named Machine.
Not mocks. Not estimated latency. Not a claimed-zero H↔D when the
plugin still host-wraps tensors. Not a copied cosine receipt.

| file | Machine | device / memory | command |
|---|---|---|---|
| `machine-a-cuda.json` | **A** (NVIDIA) | CUDA PINNED mapped + DEVICE | `make bench-machine-a` / `make bench-turbo MACHINE=A` |
| `machine-b-ov.json` | **B** | OpenVINO GPU, ZE SHARED USM | `make bench-machine-b-ov` / `make bench-turbo MACHINE=B` |
| `machine-c-metal.json` | **C** (Apple M2) | Metal **SHARED** (`MTLResourceStorageModeShared`) | `make bench-machine-c` / `make bench-turbo MACHINE=C` — **LIVE** |
| `machine-a-nvidia-overhead.json` | **A** (NVIDIA) | direct ORT CUDA vs `turboembed.h` ABI | `make bench-nvidia-overhead` |
| `machine-c-metal-overhead.json` | **C** (Apple) | direct mlx-swift Metal vs `libTurboEmbed.dylib` ABI | `make bench-apple-overhead` — harness landed, **no receipt yet**; see [checklist step 4](../../../docs/apple-m4-machine-c-checklist.md) |

Gates (all three machines): p50/p99 measured after warmup,
`allocs/forward == 0`, Berlin + MiniLM embed goldens in band,
memory-path honesty (no silent CPU fallback).

Do not hand-edit latency or counter fields. Re-run the Make target on
that Machine. `host` is provenance — map it through the root README
chart.

Machine C proof: [`docs/apple-solidify-bench-machine-c.md`](../../../docs/apple-solidify-bench-machine-c.md).
