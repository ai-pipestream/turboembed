# TurboEmbed on Apple (Swift)

Apple exposes the **same C ABI** as Linux. The header is
[`include/turboembed.h`](../include/turboembed.h). The Swift package keeps
an identical copy at `swift/Sources/TurboEmbedC/include/turboembed.h`
(the `turboembed` crate tests they match).

## How Swift exports C

Three supported mechanisms — pick one per binary; do not mix two
definitions of the same symbol in one link:

1. **`@_cdecl("turboembed_engine_create")`** on a `public` Swift function
   whose signature matches the header. This is what
   `swift/Sources/TurboEmbed/ABI.swift` does today (mock stub).
2. **C++ in the same process** (`extern "C"`) — the Linux stub in
   `native/turboembed/src/stub.cpp`. Not linked into the Swift server.
3. **Bridging header / Clang module** (`TurboEmbedC`) so Swift can *call*
   the C types and, later, call into a C++ ORT/GenAI dylib if we ever
   ship a universal binary. Apple's production path is MLX, not ORT.

Swift can also call C++ directly (C++ interop). We still **export** the
C ABI so Rust and C clients stay on one header.

## Package targets

| target | role |
|---|---|
| `TurboEmbedC` | Clang module: the frozen header only |
| `TurboEmbed` | `@_cdecl` stub + safe Swift wrapper (`Engine`) |

Neither is linked into `inferstream-apple` yet. The live Apple server
keeps using `MlxEngine` in-process. Next step: `ABI.swift` calls
`MlxEngine.embed` for catalog aliases and keeps the mock for
`mock-embed`.

## Linux CI

`swift build` is not required on Linux CI. If someone builds the package
on Linux, `ABI.swift` still compiles: the stub does not import
`MlxEngine` / Metal. Real Metal work stays `#if os(macOS)` when it
lands.

```bash
# on a Mac
swift build --package-path swift --target TurboEmbed
```

## Ownership

Same rules as the header: input strings are views (Swift `String` is
copied into a temporary UTF-8 buffer for the duration of the call).
`Embeddings` owns the C result and exposes `UnsafeBufferPointer<Float>`
until it is released.
