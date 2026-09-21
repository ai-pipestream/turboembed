# Native ownership validation, 2026-09-14

This receipt records bounded local validation of the native ownership and input
contracts changed on the `native-sdk-foundations` branch. It is evidence for
the tested Linux Mock path. It is not complete M0 certification: Swift/macOS,
Metal, OpenVINO, CUDA, TensorRT, and accelerator model correctness were not
validated here.

## Default workspace tests

The default parallel command was run without serializing the test harness:

```bash
cargo test --locked --workspace
```

The saved log is `/tmp/turboembed-foundations-workspace-tests.log`. Across unit,
integration, and documentation test binaries, 285 tests passed, none failed,
and 8 were ignored. The ignored tests were five llama.cpp endpoint tests, one
TurboRerank RPC weights check, one TurboRerank native weights check, and one
TurboEmbed CUDA weights check.

Four `crates/turborerank/tests/cpu_minilm.rs` tests returned successfully without
running their model assertions because the optional reranker weights were not
present. Rust reports these conditional early returns as passed, so they are
included in the 285 pass count rather than the 8 ignored count. The local
MiniLM tokenizer snapshot used by the server extension test was present; that
test was not conditionally skipped.

After the additional callback-join and native count-bound regressions, a full
parallel rerun exposed another test-isolation defect:
`config::tests::bearer_mode_requires_tokens` failed while another configuration
test changed `INFERSTREAM_API_KEYS` in the shared process. Both environment-
dependent cases now run in subprocesses with explicit environments. Production
authentication behavior is unchanged.

The configuration suite then passed 19 tests. The final ordinary workspace run
passed 288 tests with 0 failures and 8 ignored; the same four conditional
reranker-model skips remain included in the pass count. The final log is
`/tmp/turboembed-foundations-verified-workspace-tests.log`. The intervening failed
run is `/tmp/turboembed-foundations-final-workspace-tests.log`.

`cargo fmt --all -- --check` and
`cargo clippy --locked --workspace --all-targets -- -D warnings` also pass after
separate formatting and lint cleanup. The existing `nvcc` compiler-bindir
build warning remains; it is not a denied Rust lint.

## Native ASan and UBSan

The two focused integration test binaries were built in an isolated target
directory. CUDA and Level Zero were disabled because this was explicitly a
Mock ownership and string-lifetime run:

```bash
env \
  CARGO_TARGET_DIR=/tmp/turboembed-native-sanitizers-libs \
  CXX=g++ \
  TURBOEMBED_DISABLE_CUDA=1 \
  TURBOEMBED_DISABLE_ZE=1 \
  CXXFLAGS='-O1 -g -fsanitize=address,undefined -fno-omit-frame-pointer' \
  RUSTFLAGS='-C linker=g++ -C link-arg=-Wl,--no-as-needed -C link-arg=-lasan -C link-arg=-lubsan' \
  ASAN_OPTIONS='verify_asan_link_order=0:detect_leaks=1:halt_on_error=1:abort_on_error=1:strict_string_checks=1' \
  UBSAN_OPTIONS='halt_on_error=1:print_stacktrace=1' \
  cargo test --locked -p turboembed --no-default-features \
    --test ownership --test abi_smoke -- --nocapture
```

That run passed 10 `abi_smoke` tests and the five ownership tests then present,
with no AddressSanitizer, LeakSanitizer, or UndefinedBehaviorSanitizer
diagnostics. After adding
`callback_can_join_result_release_on_another_thread`, it was run with the same
environment and target directory:

```bash
cargo test --locked -p turboembed --no-default-features \
  --test ownership callback_can_join_result_release_on_another_thread \
  -- --exact --nocapture
```

The new test passed, giving 16 focused sanitizer test results in total: the
prior 15 plus this regression. The test exercises a callback joining a thread
that releases another result from the same engine.

`ldd` showed both focused test executables loading `libasan.so.8` and
`libubsan.so.1`. The native `stub.o` contained unresolved `__asan_*` and
`__ubsan_*` instrumentation references resolved by those libraries. The C++
stub, TurboBuffer arena and platform fallback sources, and WordPiece sources
compiled through `cc` were sanitizer-instrumented. The Rust crate, tests, and
Rust standard library were not compiler sanitizer-instrumented; the Rust test
executables only linked the sanitizer runtimes needed by the native objects.

Two setup attempts failed before any tests ran. Using Rust's default `rust-lld`
left the GCC sanitizer hooks unresolved. Switching the final linker to `g++`
and adding the runtimes explicitly then caused ASan's runtime-order check to
abort Rust build-script executables. The successful configuration above keeps
the detection options enabled and sets `verify_asan_link_order=0` to accommodate
Rust's link ordering. Linker warnings about `tmpnam`, `tmpnam_r`, and `tempnam`
came from `libasan.so`; they were not sanitizer findings in TurboEmbed.

## Unrun validation

No Swift compiler or Apple frameworks were available on this Linux host, so
the Swift wrapper tests and native macOS/Metal tests were not compiled or run.
No OpenVINO, CUDA, TensorRT, GPU, NPU, model-quality, benchmark, hosted CI,
publication, or deployment validation is claimed by this receipt.
Subsequent, separately scoped hardware results are recorded in the
[CUDA allocator receipt](cuda-allocator-isolation.md) and
[initial Intel GPU receipt](intel-native-baseline-2026-09-14.md).
The later [Apple ownership receipt](apple-ownership-validation-2026-09-14.md)
records compiled Swift tests and actual Metal ownership/concurrency validation
on the M2 Mac.
