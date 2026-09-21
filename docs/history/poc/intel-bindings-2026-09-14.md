# Intel SDK binding validation, 2026-09-14

These are local development and hardware results, not a published release or
hosted-CI result. The native library is `596e5c3`, using the exact Intel GPU,
OpenVINO runtime and pinned MiniLM bundle in the
[native SDK receipt](intel-prepared-sdk-2026-09-14.md). The Rust and Java source
changes accompany this receipt. No server was used.

## Rust

On the Intel host, Rust/Cargo 1.98.1 linked the installed prepared SDK:

- Both explicitly ignored `prepared_sdk` hardware tests were selected and
  passed, with no ignored tests in that invocation. They cover CPU/GPU numerical
  agreement, prepared/text inputs, Unicode/NUL/empty strings, output canaries,
  invalid input slices, native parent retention, OpenCL handle access, shared
  models, concurrent slots and moving slots/results between threads.
- Three documentation tests passed, including two prepared API compile-fail
  tests for slot reuse while a result is borrowed and sharing a non-Sync slot.
- The prepared C declaration layout test passed; two unrelated library tests
  were filtered by the explicit test name.
- The thread-local Rust allocator probe observed zero Rust allocation calls
  across 32 reused execute/release operations. It does not instrument OpenVINO,
  other threads, GPU drivers or native C++ allocation.

The default workspace command passed 301 tests with zero failures and nine
ignored tests. Four existing reranker-model tests return early without optional
weights and are included in Rust's passed count. The prepared feature's hardware
suite is compiled only when that feature is enabled. Workspace formatting and
default-feature Clippy passed; prepared-feature Clippy also passed. Existing
`nvcc` compiler-bindir build warnings remain.

## Java contracts and consumer

The Intel host ran Temurin JDK `25.0.4+7`. The development host compiled with
GraalVM CE JDK `25.0.3`. Maven builds use strict `-Xlint:all -Werror` compilation.
The common API targets Java 17 bytecode and the FFM implementation targets
Java 25; no older-JVM runtime implementation was exercised.

The local ordinary build passed the C-header layout test and explicitly skipped
six hardware tests. With SDK and bundle environment variables set on Intel,
all seven tests passed with zero failures, errors or skips. The final run included
unaligned direct FloatBuffers and cached descriptor offsets.

The tests cover every C descriptor's size, alignment and field offsets; pinned
model metadata; CPU/GPU vector parity at maximum error `5e-4` and RMSE `1e-4`;
and prepared/text agreement at `1e-6` and `1e-7`. They also cover empty/NUL/Unicode
strings, malformed UTF-16, invalidated inputs, missing GPU ordinal, fixed staging
positions, output bounds and canaries, heap/direct/opposite-order/unaligned
buffers, read-only output, stale result/OpenCL views, early parent close,
rejected slot close while a result is live, foreign-thread calls/buffer access,
concurrent slots created from a shared model, and two independent Java providers
with separate errors and native-library lifetimes.

A separate directory contained only `Embed.java` and the two built jars. It
compiled with JDK 25 and ran against the installed SDK under
`env -i PATH=/usr/bin:/bin`, without `LD_LIBRARY_PATH` or OpenVINO setup scripts.
It selected the Intel GPU, returned 384 floats with norm within `2e-7` of one,
and passed the pinned prepared-token/text comparison in both the initial and
final-adapter runs. The SDK still requires the system
OpenCL loader, GPU driver, glibc and C++ runtime.

The installed C example was also extended to check its pinned tokenizer, upload
prepared tokens once, and compare three executions against text output. A fresh
consumer directory compiled the installed example with strict warnings under
`env -i PATH=/usr/bin:/bin`; both GPU and explicit CPU runs passed. The original
native library binary was unchanged by this example update.

## Performance evidence

The [native pilot](intel-prepared-performance-2026-09-14.md) records the separate
OpenVINO-versus-C-ABI gates. The [binding pilot plan](../java/benchmarks/README.md)
defines Java's bridge, prepared execution and ordinary text measurements.
Both 205-second runs completed within their ten-minute bounds. The initial run
passed all nine bridge comparisons but exposed 760 Java allocation bytes per
`stats()` call. Caching fixed descriptor offsets removed that overhead at the
measured call site. The final run repeated the same shapes, gates and workload.

All nine final bridge comparisons passed. Observed Java-minus-native p50 was
0.020 microseconds; the largest p99 difference was 0.029 microseconds, below the
predeclared 5/20-microsecond limits. These values are near the measurement clock's
resolution; they are diagnostic observations, not a universal downcall guarantee.
Each bridge path recorded 100,000 calls per repeat, including native error checks
and Java thread/lifetime checks. No clock-cost subtraction was applied.

Median of the three repeat p50 values, in microseconds:

| Batch × sequence | Native prepared | Java prepared | Native text to host | Java text to host |
|---|---:|---:|---:|---:|
| 1 × 32 | 405.1 | 407.9 | 451.0 | 447.5 |
| 8 × 128 | 1725.7 | 1704.3 | 1747.7 | 1740.4 |
| 32 × 256 | 14634.5 | 14598.3 | 14862.8 | 14846.1 |

Every row contains `hello world`, padded to its declared shape. Prepared timings
exclude upload and readback; text timings include native tokenization, upload,
inference, host readback, and Java encoding/bulk copy where applicable. Small
cross-process differences in either direction are not evidence of Java
accelerating the GPU. Batch-32 timings have only 202–206 observations per repeat;
they are insufficient for a p99 claim. All raw samples and repeat variation are
preserved. No comparison to a different library or server is implied.

The final Java allocation diagnostic observed zero bytes for the benchmark's
non-escaping `stats()` record, 64 bytes per prepared execute/release, and 168–4128
bytes per text request across the tested shapes/repeats. These are calling-thread
Java allocations across 100 extra calls, not process allocations. Retaining a
metadata record can prevent escape-analysis elimination. OpenVINO/C++ and driver
allocations are excluded; the native pilot separately records substantial C++
allocation in both native and ABI paths.

Raw receipts and exact measured artifact hashes:

- [Initial samples](../testdata/receipts/bench/intel-bindings-2026-09-14-initial-json.tar.gz)
  with adjacent SHA-256 manifest.
- [Final samples](../testdata/receipts/bench/intel-bindings-2026-09-14-final-json.tar.gz)
  with adjacent SHA-256 manifest.
- [Measured jars, benchmark sources/executables, SDK library and bundle manifest](../testdata/receipts/bench/intel-bindings-2026-09-14-provenance.json).

The native benchmark targets and extracted remote-buffer probe compiled on Intel.
One source-tree linking attempt omitted the documented OpenVINO environment and
failed to resolve TBB symbols; sourcing `setupvars.sh` made that build pass.
Installed SDK consumers continued to run without that environment. No hosted CI,
artifact publication, deployment, Apple qualification or Android validation was
performed for these bindings.
