# Java M3 validation (CPU path), 2026-09-17

This receipt covers the roadmap M3 work on a GPU-less build VM: the JDK 25
FFM adapter and Java API validated against the M2 packaged SDK on the
explicit-CPU path, including device discovery, the cpu-only contract mode,
the clean consumer example, and matched native/Java CPU binding timings.
Intel GPU coverage of the M3 additions remains gated on the Machine B
(`krick-1`) reference host, as listed at the end. The
[2026-09-14 receipt](intel-bindings-2026-09-14.md) remains the Java GPU
validation and performance record for the pre-existing surface.

## Environment

- Host: cloud build VM, Linux x86_64, Ubuntu 24.04, glibc 2.39, 4 vCPU,
  no GPU, no OpenCL ICD beyond the loader.
- Java: Temurin JDK `25.0.4.1+1` (LTS), Maven 3.9.16. Native toolchain:
  GCC 13.3.0 (`CC=gcc CXX=g++`), CMake 3.28.3, Python 3.12.3.
- OpenVINO runtime: archive distribution
  `openvino_toolkit_ubuntu24_2025.3.0.19807.44526285f24_x86_64.tgz`
  (SHA-256 verified against the CI pin), reporting
  `2025.3.0-19807-44526285f24-releases/2025/3`. The GPU-qualified baseline
  on `krick-1` remains `2026.3.1-22476-759c5a6ab8c`.
- SDK: `scripts/make-sdk-release.sh` on this branch (M2 merge base
  `1f0f764`) produced `turboembed-prepared-sdk-1.0.0-linux-x86_64.tar.gz`,
  SHA-256 `9db8d6fb727a05e84819707f30a799c3c0c0f9eed16dd58f8cf4cd53d8462297`.
- Bundle: `cargo run -p inferstream-fetch -- --prepared minilm` verified the
  committed pins, and `scripts/provision-minilm-bundle.sh` provisioned the
  pinned `sentence-transformers/all-MiniLM-L6-v2` bundle offline.
- Jars under test, built by `mvn -f java/pom.xml --batch-mode verify`:
  `turboembed-api` SHA-256
  `28e01c90690fafae62455c31d784f4ac325290e0ec994fc6fa9439a1f4852686`,
  `turboembed-ffm` SHA-256
  `777c85ac0ce45cb93d99ec5caf1a12b12f64d53a138c1f596e715e907f2bdb32`.

## Contract results

- Ordinary build (no SDK environment): 8 tests — the C-header layout
  conformance test (now including `te_device_info`) passed and the 7
  hardware contract tests were skipped. Strict `-Xlint:all -Werror`
  compilation passed for both modules.
- Packaged CPU run (`TURBOEMBED_PREPARED_SDK`, `TURBOEMBED_PREPARED_BUNDLE`,
  `TURBOEMBED_PREPARED_DEVICES=cpu-only`): 8 tests passed, 0 failures,
  0 skips. This covers prepared/text agreement between two independent CPU
  contexts, pinned model metadata, empty/embedded-NUL/Unicode text, malformed
  UTF-16, invalidated inputs, output bounds/order/alignment/aliasing and
  read-only rejection, result leases and close-during-use, foreign-thread
  rejection, concurrent slots from a shared model, two independent providers
  with separate library lifetimes, and the new discovery contract.
- Fail-loud device policy on this host: discovery listed exactly one device
  (`Intel(R) Xeon(R) Processor`, CPU); requesting `AUTO` or `OPENVINO_GPU`
  returned the typed unavailable error (code 4). The CPU byte counters
  reported zero explicit GPU transfers, matching the documented stats
  semantics.
- API neutrality: `jdeps -verbose:class` on the built `turboembed-api` jar
  reported only `java.base` dependencies and zero `java.lang.foreign`
  references; its class files are Java 17 bytecode (major version 61). The
  FFM adapter jar is the only artifact with Java 25 bytecode.

## Clean consumer example

A separate temporary directory contained only `Embed.java` and the two jars.
It compiled with `javac --release 25` and ran under `env -i
PATH=/usr/bin:/bin` against the extracted release archive and provisioned
bundle only:

- Explicit CPU: printed the discovered device list, embedded `hello world`
  (384 dimensions, norm `0.99999996`), and passed the pinned
  prepared-token/text comparison at `1e-6`/`1e-7`. Exit 0.
- Default GPU: failed with `requested device is unavailable; CPU is never an
  automatic fallback`. Exit 1. No silent CPU fallback occurred.

## Matched CPU binding timings

Both binding benchmarks ran with the new explicit `cpu` argument against the
same installed SDK and bundle, alternating native-first and Java-first
between shapes, one process per path/shape, no concurrent load. Median of
the three repeat p50 values, in microseconds:

| Batch × sequence | Native prepared | Java prepared | Native text to host | Java text to host |
|---|---:|---:|---:|---:|
| 1 × 32 | 1699.6 | 1488.1 | 1663.9 | 1506.9 |
| 8 × 128 | 31819.4 | 29501.5 | 29859.5 | 29445.5 |
| 32 × 256 | 260028.7 | 251738.9 | 265955.6 | 251254.6 |

- Bridge case (`slot_stats` per call, 100,000 recorded calls per repeat):
  Java p50 0.1 µs versus native 0.0–0.1 µs at every shape, within the
  predeclared 5/20 µs p50/p99 bridge budget. Java allocation for the
  non-escaping stats record was 0 bytes per call.
- Java thread allocations: 64 bytes per prepared execute/release and
  168–4128 bytes per text request across shapes, matching the GPU-run
  behavior. These are calling-thread Java allocations only.
- Repeat-to-repeat p50 spread was up to ~17% on this shared 4-vCPU VM, and
  Java-versus-native differences at every shape are inside that spread in
  both directions; they are not evidence of Java accelerating inference.
- Sample counts: 1 × 32 recorded 1,672–1,990 inference observations per
  repeat; 8 × 128 recorded 78–103 and 32 × 256 recorded 11–12, so p99 for
  those shapes is insufficient under the plan's 1,000-observation rule.
- Bound deviation: the complete CPU matrix took about 40 minutes, exceeding
  the plan's ten-minute cap, because the fixed 500-call inference warmups
  were calibrated on the GPU reference host and CPU inference here is
  roughly 20–70× slower per call. No workload, warmup, or gate was changed.
- Raw samples:
  [tarball](../testdata/receipts/bench/java-m3-cpu-2026-09-17-json.tar.gz)
  with adjacent SHA-256 manifest.

These CPU numbers document the Java-versus-native boundary on this host.
They do not stand in for the roadmap's Intel GPU performance budget; the
[2026-09-14 GPU pilot](intel-bindings-2026-09-14.md) remains that record.

## Remaining Machine B (`krick-1`) gates

The M3 additions in this change are hardware-unverified on an Intel GPU
until run on the reference host:

1. Re-run the Java contract suite in default `gpu` mode (both Intel GPU and
   explicit CPU required), which now includes the discovery contract test
   and the mode-aware stats assertions.
2. Re-run the standalone Java example with default GPU selection against a
   `2026.3.1`-built release archive.
3. Re-run the binding benchmark matrix on the GPU to confirm the explicit
   device argument does not perturb the previously recorded GPU numbers.
