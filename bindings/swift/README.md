# Pipestream Turbo for Swift

`PipestreamTurbo` is a SwiftPM package over `libturbo`'s C ABI:

- `CTurbo`: a clang module whose umbrella header includes
  `include/turbo/turbo.h` by relative path (SwiftPM allows no header search
  path outside the package). Nothing is declared in the package itself.
- `PipestreamTurbo`: the Swift API. `Runtime`, `Context`, `Model`,
  `Session`, `Result`, `Generation`, and `Tokenizer` are final classes
  owning one C handle each and releasing it in `deinit`; a child keeps its
  parent alive as the C contract does. Options are structs mirroring the C
  descriptors; enums mirror the `uint32_t` constants and are checked
  against the header's values once at first use. Every failing call throws
  `TurboError` with the code, the 1-based field index, and the library's
  message.
- `turbo-conformance`: the conformance cases as an executable
  (`swift run turbo-conformance`), because neither XCTest nor Swift Testing
  is available with the macOS command line tools alone.

The module is not named `Turbo`: on a case-insensitive file system the
module's own `libTurbo.a` would satisfy the linker's `-lturbo` before
`libturbo` itself.

## Building and running the conformance cases

```bash
cargo build -p turbo-shared            # target/debug/libturbo.dylib
cd bindings/swift
DYLD_LIBRARY_PATH=../../target/debug swift run turbo-conformance
```

`Package.swift` adds the repository's `target/debug` and `target/release`
to the linker search path; a package that consumes an installed archive
passes its own `-Xlinker -L<dir>`. `TURBO_BUNDLES` overrides the mock
bundle root (default `testdata/bundles/mock`).

On `krickert-mac` (Apple M2, macOS 27, Swift 6.4 command line tools) the
original nine cases pass. The five cases added for generation, the
tokenizer, the held-result BUSY case, and the cross-thread cancel case
have not yet been run on that machine.

## Generation and the tokenizer

```swift
let g = try model.createGeneration(GenerateDesc())
try g.prompt([Message.user("What is the capital of France?")])
let last = try g.drain { chunk in print(chunk.text, terminator: ""); return true }
print(" [\(last.finishReason)]")

let tok = try runtime.createTokenizer(bundlePath: "/opt/bundles/minilm-onnx")
let enc = try tok.encode(["hello world"], rowStride: 16)
print(try tok.decode(enc.row(0)))
```

`Chunk` is copied out of native memory, so it stays valid after the next
step; `drain` cancels when the closure returns `false` and the final chunk
reports `.cancelled`. `cancel()` may be called from another thread.

## Not yet in the binding

The push form `turbo_generate`, the chunk-plan functions, buffer
allocation and import, and RUN-model binding are reachable through
`CTurbo` but have no Swift wrapper yet.
