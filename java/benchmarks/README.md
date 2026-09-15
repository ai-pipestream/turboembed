# Java binding pilot

This bounded pilot compares the installed native C ABI with the public Java FFM
adapter. It follows the bridge budget in [the library design](../../docs/library-design.md#native-performance-acceptance):
Java adds at most 5 microseconds p50 and 20 microseconds p99 in the bridge-isolation
case. Those limits are set before running this harness.

Use the same installed SDK and pinned MiniLM bundle for both executables. Run
on the Intel reference GPU with no overlapping GPU tests or benchmarks. Cover
batch/sequence pairs `(1,32)`, `(8,128)`, `(32,256)`, with each row containing
`hello world` padded to its slot length. Each path checks text/prepared parity
before measurement. This is a binding experiment; the broader native pilot
separately covers full and mixed-length input.

Each executable makes three repeats of:

- `bridge`: native `slot_stats` versus Java `slot.stats()`, including the Java
  metadata record and thread/lifetime checks. Warm up 100,000 calls; record at
  most 100,000 individual calls or three seconds.
- `prepared`: execute and release a result with inputs already uploaded and no
  host readback. Java also checks the result dimension. Warm up 500 calls;
  record at most 3,000 calls or three seconds.
- `text_host`: native tokenization, upload, execute, explicit host readback and
  release. Java includes UTF-16 to UTF-8 encoding and copying into a reused heap
  FloatBuffer. Strings and descriptor arrays in the benchmark are reused.
  Warm up 500 calls; record at most 3,000 calls or three seconds.

Alternate native-first and Java-first order between shapes. Keep all samples,
report repeat spread, and mark p99 from fewer than 1,000 observations as
insufficient. A single process per path/shape is diagnostic evidence, not a
universal JIT or tail-latency guarantee. Cap the complete run at ten minutes.
Only the bridge case has an additive language-overhead gate; inference timings
remain separate matched-workload observations.

The Java harness also measures thread-allocated Java bytes across 100 additional
calls after each timing loop. It excludes native/runtime/driver allocations and
other Java threads. Result wrappers and FFM address objects may allocate. This
counter does not measure data transfers or total process memory.

Build `binding_benchmark.cpp` through `TE_BUILD_BENCHMARK=ON`, or compile it
against the installed SDK headers/library and this repository's vendored
`nlohmann/json.hpp`. Compile `BindingBenchmark.java` with JDK 25 and the two
consumer jars on the classpath. Run:

```bash
./binding_benchmark /path/to/minilm-bundle 1 32 > native-1-32.json
java --enable-native-access=ALL-UNNAMED -cp "$TE_CLASSPATH" \
  BindingBenchmark /path/to/sdk /path/to/minilm-bundle 1 32 java-1-32.json
```

The native executable writes one JSON object on stdout. The Java executable
writes the requested output file. Keep diagnostic stderr separately.

The first run exposed repeated Java layout-path allocation in `slot.stats()`.
The follow-up caches those fixed offsets (including text descriptor offsets),
then repeats this same bounded matrix to measure the final adapter and allocation
change. It also includes the independently tested unaligned direct-buffer fix;
that fallback is outside these heap-output timing workloads. Retain both runs.
