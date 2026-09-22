# Pipestream Turbo for Java

`ai.pipestream:turbo` is a JDK 25 binding for `libturbo` built on the
foreign function and memory API (no JNI, no generated C). It has two layers:

- `ai.pipestream.turbo.ffi`: the raw C surface, generated from
  `include/turbo/turbo.h` by jextract (`scripts/gen-java-ffi.sh`) and
  committed. One class per struct with layout and field accessors, and
  `TurboNative` with every function and constant.
- `ai.pipestream.turbo`: the safe API. `Turbo` (runtime), `Context`,
  `Model`, `Session`, `Result`, `Generation`, and `Tokenizer` are
  `AutoCloseable` handles over the C handles; option records
  (`EmbedOptions`, `RerankOptions`, `ClassifyOptions`, `GenerateDesc`,
  `EncodeOptions`) mirror the C descriptors; enums mirror the `uint32_t`
  constants; every non-`TURBO_OK` status is a `TurboException` carrying the
  code, the 1-based field index, and the library's message.

The semantics are the C contract's, unchanged: releasing a parent handle
never invalidates a child, a session is single-owner (`TURBO_E_BUSY` under
contention), a live `Result` leases its session, and an option is honored
exactly or refused naming the field.

## Locating the library

`TurboNative` resolves `libturbo` from, in order: the `turbo.library`
system property (a path), the `TURBO_LIBRARY` environment variable, then
the loader's search for `turbo` (`java.library.path`, `LD_LIBRARY_PATH`).
Provider libraries are loaded by path through `Turbo.create(List.of(...))`.

## Building and testing

```bash
cargo build -p turbo-shared                     # produces target/debug/libturbo.so
cd bindings/java
mvn test                                        # uses ../../target/debug/libturbo.so and testdata/bundles/mock
mvn test -Dturbo.library=/path/to/libturbo.so   # another build
```

The tests are the conformance cases run through the binding against the
mock provider, under `--enable-native-access=ALL-UNNAMED
--illegal-native-access=deny`. On `krick` (JDK 25.0.3, Temurin) and
`krick-1` (JDK 25.0.4, Temurin) the sixteen tests pass in under a second.

## Regenerating the raw layer

After any change to `include/turbo/turbo.h`:

```bash
JEXTRACT=~/opt/jextract/jextract-22/bin/jextract scripts/gen-java-ffi.sh
```

jextract 22 generates indexed array accessors (`shape(struct, i)`) whose
var-handle coordinates do not match JDK 25; the safe layer reads arrays
through the slice accessors (`shape(struct)`) instead. Do not use the
indexed forms.

## Example

```java
try (Turbo rt = Turbo.create(List.of("/opt/turbo/providers/libturbo_provider_cuda.so"));
     Context ctx = rt.createContext(rt.selectDevice());
     Model model = ctx.loadModel("/opt/bundles/minilm-onnx");
     Session s = model.createSession(8, 256)) {
    s.writeText(List.of("hello world"), EmbedOptions.defaults());
    try (Result r = s.run()) {
        float[] v = r.readFloats(0);              // [1, dim], copied from wherever it lives
        System.out.println(r.placement() + " " + v.length);
    }
}
```

## Generation and the tokenizer

```java
try (Generation g = model.createGeneration(GenerateDesc.defaults().withMaxNewTokens(64))) {
    g.prompt(List.of(Message.user("What is the capital of France?")));
    Chunk last = g.drain(c -> { System.out.print(c.text()); return true; });
    System.out.println(" [" + last.finishReason() + "]");
}
try (Tokenizer tok = rt.createTokenizer("/opt/bundles/minilm-onnx")) {
    Encoding e = tok.encode(List.of("hello world"), 16, EncodeOptions.defaults().withMaxTokens(16));
    System.out.println(tok.decode(e.row(0), true) + " " + tok.count("hello world", true));
}
```

`Chunk` is copied out of native memory, so it stays valid after the next
step; `drain` cancels when the predicate returns false and the final
chunk reports `CANCELLED`. `Tokenizer` is thread-safe.

## Not yet in the binding

The push form `turbo_generate`, the chunk-plan functions, buffer
allocation and import, and RUN-model binding are in the raw layer but have
no safe wrapper yet.
