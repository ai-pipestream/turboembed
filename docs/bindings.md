# Bindings

A binding wraps the C ABI (`include/turbo/turbo.h`) for another language.
`bindings/java` (`ai.pipestream:turbo`) is the only one that exists in this
tree today; Swift and Android are planned (`PLAN.md` section 10, P7, P10).

## Java (`bindings/java`, `ai.pipestream:turbo`)

A JDK 25 binding built on the foreign function and memory API: no JNI, no
generated C. It has two layers (`bindings/java/README.md`):

- **`ai.pipestream.turbo.ffi`**: the raw C surface. One class per ABI
  struct (layout plus field accessors) and `TurboNative` with every
  function and constant, generated from `include/turbo/turbo.h` by
  jextract and committed to the repository, not written by hand.
- **`ai.pipestream.turbo`**: the safe API. `Turbo` (runtime), `Context`,
  `Model`, `Session`, and `Result` are `AutoCloseable` handles over the C
  handles; option records (`EmbedOptions`, `RerankOptions`,
  `ClassifyOptions`) mirror the C descriptors; enums mirror the `uint32_t`
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
--illegal-native-access=deny`. On `krick` (JDK 25.0.3, Temurin) nine tests
pass in under a second. `.github/workflows/ci.yml`'s `java` job runs the
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

### Not yet in the binding

Generation (`turbo_generation_*`, `turbo_generate`), the tokenizer and
chunk-plan functions, buffer allocation and import, and `RUN`-model
binding are present in the raw `ffi` layer (jextract generates the whole
header) but have no safe wrapper yet (`PLAN.md` P6 and P7). The Swift
package and an `nano1`/`krick-1` run of the Java suite are also open items
of P7 (`PLAN.md` section 10).

See `bindings/java/README.md` for the full walkthrough and a worked
example, and `docs/c-api.md` for the C contract the binding wraps.
