# Bindings

A binding wraps the C ABI (`include/turbo/turbo.h`) for another language.
`bindings/java` (`ai.pipestream:turbo`) and `bindings/swift`
(`PipestreamTurbo`) exist in this tree today; Android is planned (`PLAN.md`
section 10, P10).

## Java (`bindings/java`, `ai.pipestream:turbo`)

A JDK 25 binding built on the foreign function and memory API: no JNI, no
generated C. It has two layers (`bindings/java/README.md`):

- **`ai.pipestream.turbo.ffi`**: the raw C surface. One class per ABI
  struct (layout plus field accessors) and `TurboNative` with every
  function and constant, generated from `include/turbo/turbo.h` by
  jextract and committed to the repository, not written by hand.
- **`ai.pipestream.turbo`**: the safe API. `Turbo` (runtime), `Context`,
  `Model`, `Session`, `Result`, `Generation`, and `Tokenizer` are
  `AutoCloseable` handles over the C handles; option records
  (`EmbedOptions`, `RerankOptions`, `ClassifyOptions`, `GenerateDesc`,
  `EncodeOptions`) mirror the C descriptors; enums mirror the `uint32_t`
  ABI constants; every non-`TURBO_OK` status becomes a `TurboException`
  carrying the status code, the 1-based field index, and the library's
  message.

The semantics are the C contract's, unchanged: releasing a parent handle
never invalidates a live child, a session is single-owner
(`TurboException` with `TURBO_E_BUSY` under contention), a live `Result`
leases its session, and an option is honored exactly or refused naming the
field.

### Locating the library

`TurboNative` resolves `libturbo` from, in order: the `turbo.library`
system property (a path), the `TURBO_LIBRARY` environment variable, then
the loader's search for `turbo` (`java.library.path`, `LD_LIBRARY_PATH`).
Provider libraries are loaded separately, by path, through
`Turbo.create(List.of(...))`.

### Building and testing

```bash
cargo build -p turbo-shared                     # produces target/debug/libturbo.so
cd bindings/java
mvn test                                        # uses ../../target/debug/libturbo.so and testdata/bundles/mock
mvn test -Dturbo.library=/path/to/libturbo.so   # another build
```

The tests are the conformance cases run through the binding against the
`mock` provider, under `--enable-native-access=ALL-UNNAMED
--illegal-native-access=deny`. On `krick` (JDK 25.0.3, Temurin) and on
`krick-1` (JDK 25.0.4, Temurin, AMD Ryzen 9 9950X) the sixteen tests pass
in under a second. `.github/workflows/ci.yml`'s `java` job runs the
same thing on every push: `cargo build --locked -p turbo-shared` then
`cd bindings/java && mvn -q -B test` under JDK 25 (Temurin), against the
mock provider only (no OpenVINO/CUDA/ggml libraries are built in that job).

### Regenerating the raw layer

After any change to `include/turbo/turbo.h`:

```bash
JEXTRACT=~/opt/jextract/jextract-22/bin/jextract scripts/gen-java-ffi.sh
```

jextract 22 generates indexed array accessors (`shape(struct, i)`) whose
var-handle coordinates do not match JDK 25; the safe layer reads arrays
through the slice accessors (`shape(struct)`) instead, and the raw layer's
indexed forms are not used anywhere in the safe API.

### Generation and the tokenizer

`Model.createGeneration(GenerateDesc)` returns a `Generation`: `prompt`
(chat messages) or `promptTokens`, then `step()` per chunk, or `drain`
with a predicate that receives every `Chunk` and can stop the stream
(the final chunk then reports `FinishReason.CANCELLED`); `cancel()` may
be called from another thread. A `Chunk` is copied out of native memory
so it outlives the next step. `GenerateDesc` is a record with `with*`
builders; every non-default field is honored exactly or refused with the
field index, as in C. `Turbo.createTokenizer(bundlePath)` returns a
thread-safe `Tokenizer` with `encode` (rows of a caller-chosen stride,
padded with the pad id), `decode`, `count`, and `info()`.

### Not yet in the binding

The push form `turbo_generate`, the chunk-plan functions, buffer
allocation and import, and `RUN`-model binding are present in the raw
`ffi` layer (jextract generates the whole header) but have no safe
wrapper yet. The binding timings of P7 are still open.

See `bindings/java/README.md` for the full walkthrough and a worked
example, and `docs/c-api.md` for the C contract the binding wraps.

## Swift (`bindings/swift`, module `PipestreamTurbo`)

A SwiftPM package over the C ABI, with no code generation step. `Package.swift`
declares three targets: two make up the library, and the third is the
conformance runner.

- **`CTurbo`**: a clang module target with no Swift or C declarations of its
  own. Its umbrella header (`Sources/CTurbo/include/CTurbo.h`) includes the
  generated `include/turbo/turbo.h` by a relative path
  (`../../../../../include/turbo/turbo.h`, five levels up from
  `bindings/swift/Sources/CTurbo/include/` to the repository root) because
  SwiftPM allows no header search path outside the package; `shim.c` exists
  only to give the target the one compilation unit a C target needs, since
  the module is otherwise header-only.
- **`PipestreamTurbo`**: the Swift API, depending on `CTurbo`. `Runtime`,
  `Context`, `Model`, `Session`, `Result`, `Generation`, and `Tokenizer` are
  `final` classes that each own one C handle and release it in `deinit`
  (`Result` additionally exposes an idempotent `close()`, and so do
  `Generation` and `Tokenizer`); a child keeps its parent alive exactly as
  the C contract requires, so releasing a parent before a live child is
  safe. Option structs (`EmbedOptions`, `RerankOptions`, `ClassifyOptions`,
  `GenerateDesc`, `EncodeOptions`) mirror the C descriptors field for field.
  Thirteen enums (`DeviceKind`, `SelectPolicy`, `Task`, `Modality`,
  `CapStatus`, `Truncate`, `PromptRole`, `Normalize`, `Pooling`,
  `OutputDType`, `Aggregation`, `Placement`, `FinishReason`) mirror the
  ABI's `uint32_t` constants. Every failing call throws `TurboError` with
  the status code, the 1-based field index (0 when the failure names no
  field), and the library's message.

The module and product are named `PipestreamTurbo`, not `Turbo`: on a
case-insensitive file system (macOS's default), a package named `Turbo`
would build its own static library as `libTurbo.a`, and that file would
satisfy the linker's `-lturbo` request ahead of the real `libturbo` the C
ABI lives in.

### Enum values are checked once, at first use

The thirteen enums above are hand-written, not generated, with `UInt32` raw
values chosen to match the header's named constants. A private
lazily-initialized value, `constantsVerified`
(`Sources/PipestreamTurbo/PipestreamTurbo.swift`), runs a block of
`precondition` calls comparing representative cases of every enum against
the matching `TURBO_*` constants (for example
`DeviceKind.cpu.rawValue == TURBO_DEVICE_CPU`). `Runtime.init` reads that
value before doing anything else, so the check runs once, at first use, and
a header change that renumbers a constant without a matching Swift edit
traps immediately instead of silently miscompiling a wire value.

### Generation and the tokenizer

`Model.createGeneration(_:)` returns a `Generation`: `prompt(_:)` (chat
messages) or `promptTokens(_:)`, then `step()` per chunk, or `drain(_:)`
with a closure that receives every `Chunk` and can stop the stream by
returning `false` (the final chunk then reports `.cancelled`); `cancel()`
may be called from another thread. `Chunk` is a struct copied out of native
memory, so it outlives the next step. `GenerateDesc` mirrors the C
descriptor field for field; every non-default field is honored exactly or
the generation throws naming the field, as in C. `Runtime.createTokenizer(bundlePath:)`
returns a `Tokenizer` with `encode`, `decode`, `count`, and `info()`.
These mirror the Java binding's `Generation` and `Tokenizer` wrappers.

### The conformance runner is an executable, not XCTest

`turbo-conformance` (`Sources/TurboConformance/main.swift`) is an
`executableTarget` running the conformance cases — the same groups the
Java binding and the Rust conformance suite exercise, now including
generation and the tokenizer, sixteen cases in all — through the Swift
API against the mock provider. It is a plain top-level script with a
hand-rolled `expect`/`thrown` assertion helper and a list of cases run in a
loop, rather than an XCTest bundle or a Swift Testing suite: neither
framework ships with the Swift command line tools alone (both need a full
Xcode install), and the binding's conformance cases need to run on any Mac
that has just a Swift toolchain. Run it with `swift run turbo-conformance`.

### Locating the library, at link time and at run time

`Package.swift` gives all three targets the same linker settings:
`.linkedLibrary("turbo")` plus `-L<package>/../../target/debug` and
`-L<package>/../../target/release` as `unsafeFlags`, so the build links
against whichever of `target/debug/libturbo.*` or `target/release/libturbo.*`
cargo has already produced in the repository's own target directory; a
package consuming an installed archive instead supplies its own
`-Xlinker -L<dir>`. That satisfies the linker at build time only — running
the resulting binary still needs the dynamic loader to find
`libturbo.dylib`, so `DYLD_LIBRARY_PATH` has to point at the same directory
at run time too:

```bash
cargo build -p turbo-shared            # target/debug/libturbo.dylib
cd bindings/swift
DYLD_LIBRARY_PATH=../../target/debug swift run turbo-conformance
```

`TURBO_BUNDLES` overrides the mock bundle root the conformance cases load
from (default `testdata/bundles/mock`, resolved relative to `main.swift`'s
own path).

On `krickert-mac` (Apple M2, macOS 27, Swift 6.4 command line tools) all
sixteen cases pass (2026-09-22), including generation, the tokenizer, the
held-result BUSY case, and the cross-thread cancel case; see
`docs/testing.md`.

### Not yet in the binding

The push form `turbo_generate`, the chunk-plan functions, buffer
allocation and import, and `RUN`-model binding are reachable through
`CTurbo` (the whole header is exposed as a module) but have no
`PipestreamTurbo` wrapper yet.

See `bindings/swift/README.md` for the walkthrough this section summarizes.
