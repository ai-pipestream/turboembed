# TurboRerank on Apple (Swift)

The frozen C ABI is [`include/turborerank.h`](../include/turborerank.h).
Apple keeps an identical copy at
`swift/Sources/TurboRerankC/include/turborerank.h`.

## Why this is not a TurboEmbed-style `@_cdecl` dylib

TurboEmbed on Mac **is** `libTurboEmbed.dylib`: Swift `@_cdecl` owns
every symbol because the Linux C++ side is a mock stub. Linking that
stub on Mac would hide a fake MiniLM.

TurboRerank is the opposite. CPU / CUDA / OpenVINO already live in
`native/turborerank` and export the frozen ABI. Replacing those
symbols with a Swift `@_cdecl` dylib on Mac would either:

1. duplicate the ABI (link errors), or
2. drop the proven CPU path that Machine C still needs.

Phase 2c therefore keeps the **C++ ABI** and implements Metal as
Objective-C++ (`metal_api.mm`) next to `cuda_api` / `ov_api`. The
Swift package is a **client** of that ABI (`TurboRerank` →
`TurboRerankC`), not a second implementation.

## Memory model (honest)

| region | where | copy |
|---|---|---|
| token / mask / type / position | `MTLResourceStorageModeShared` | **none** on `forward` — kernels bind the caller-written MTLBuffers |
| weights | mmap'd safetensors → MTL shared at **load** | **once** at load, never on the hot path |
| activations | MTL shared scratch reserved at load | none (pre-sized) |
| scores | caller `float *` | one 4-byte read of the logit buffer after GPU completion |

No `std::vector` on the token path. A CPU-allocated buffer passed to
a Metal engine fails loud (refuses a silent host copy).

## Device policy

| requested | if missing | fallback |
|---|---|---|
| `METAL` / `AUTO` (Machine C) | `UNAVAILABLE` | **none** — never CPU |
| `CUDA` / `TENSORRT` / `OPENVINO_GPU` / `NPU` | `UNAVAILABLE` / `UNSUPPORTED_DEVICE` | **none** |
| `CPU` | (explicit) | first-party FP32 CE |
| `MOCK` | (explicit smoke) | refuses catalog CE scores |

`AUTO` on Machine C (no CUDA, no OpenVINO GPU) is Metal.
