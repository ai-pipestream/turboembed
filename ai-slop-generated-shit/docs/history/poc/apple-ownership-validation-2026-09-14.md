# Apple ownership validation, 2026-09-14

All 18 Swift package tests passed on the M2 Mac, including Metal GPU ownership
checks: 12 XCTest tests and six Swift Testing tests, with no failures or skips.
Validation used `e97b494` plus the changes accompanying this receipt, in an
isolated checkout.

## Environment

- SSH alias supplied by the project owner: `kristians-macbook-air`; hostname
  an Apple M2 MacBook Air (`Mac14,2`), 24 GiB memory.
- macOS 26.5.1, build `25F80`, arm64.
- Xcode 26.6, build `17F113`; Swift 6.3.3.
- `mlx-swift` 0.31.6, revision `0bb916c67f4b9e5c682cbe02a42c701c93ab5021`,
  resolved from the committed `swift/Package.resolved`.
- The existing `/Volumes/pipework/work/inferstream` checkout remains at
  `9d03af9`; its running server and files were left unchanged.
- Validation checkout:
  `$HOME/te-validation/native-foundations-20260914-e97b494`.

The ten MiniLM files were copied from the existing Mac checkout only after all
sizes and hashes matched `models/manifests/mlx.json`. Copied files were verified
again. The manifest names `sentence-transformers/all-MiniLM-L6-v2`, revision
`1110a243fdf4706b3f48f1d95db1a4f5529b4d41`. No reference goldens were modified.

## Build setup and findings

The SSH session selected Command Line Tools by default. These commands select
the installed Xcode toolchain and expose the installed Git LFS executable needed
by the Mac's Git hooks, without changing global settings:

```bash
export DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer
export PATH=/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin
make libturbo-buffer-apple libturborerank-apple
swift test --package-path swift --force-resolved-versions --jobs 2 \
  --filter WrapperOwnershipAndStringsTests
```

The first native build failed with status 141 after creating the archive.
`nm | grep -q` closed the pipe early; the measured pipeline statuses were
`141, 0`. Under the script's `pipefail`, this made a successful symbol match
fail the build. The check now consumes all `nm` output and still rejects a
missing symbol. Both native archives then built successfully.

An initial Swift dependency checkout failed because `/usr/local/bin/git-lfs`
was absent from the command's PATH. Adding the installed executable resolved
that failure. The dependency lockfile was not relaxed or updated.

The first Swift compilation then failed in a TurboRerank header test, which
compared an imported C enum directly with an integer. The test now compares
the enum's raw value and is named for the device constant it checks. SwiftPM
compiles this test target even when filtering for TurboEmbed tests.

Linking then exposed a second test setup defect: the TurboRerank tests imported
the headers without linking `libturborerank_apple`. The test target now links
that archive and its native framework dependencies.

The server target also rejected `NSLock.lock()` and `unlock()` inside its
async rerank method. A synchronous `withLock` closure now encloses the same
native call and error handling, with no suspension inside the critical section.

The final server build passed after that correction:

```bash
swift build --package-path swift --force-resolved-versions --jobs 4 \
  --product inferstream-apple
```

The linker warned that objects in `libturborerank_apple` targeted macOS 26.0
while the Swift product targeted macOS 15.0. This run establishes build and
test behavior on macOS 26.5.1 only. Aligning native deployment targets and
testing the supported minimum OS remain Apple packaging work. The server was
compiled but not started, and reranker model inference was not exercised.

The shader script produced eight kernels but failed to compile `fence.metal`
with its existing flags. The pinned MLX source uses those fence kernels only
when `MLX_METAL_FAST_SYNCH` is enabled; its default is zero and the environment
variable was unset for this validation. Fast synchronization is not validated.
The generated `mlx.metallib` is 2,855,435 bytes, SHA-256
`d2638df92d1a8631162ff1f66076e92d5db29a2332394d56f9dd6efb768b2a5e`.

The shaders were built with:

```bash
sh scripts/build-apple-metallib.sh swift/.build/arm64-apple-macosx/debug
```

Cmlx is statically linked into the XCTest bundle. The generated `mlx.metallib`
and `default.metallib` were also copied next to its executable, under
`swift/.build/arm64-apple-macosx/debug/inferstream-applePackageTests.xctest/Contents/MacOS/`.
Both copies matched the hash above.

## Test results

After the test executable linked, these commands ran it without repeating
SwiftPM's dependency builds:

```bash
export INFERSTREAM_ROOT="$PWD"
swift test --package-path swift --skip-build \
  --filter WrapperOwnershipAndStringsTests
swift test --package-path swift --skip-build
```

The focused run passed all five wrapper tests. The full run passed:

- Five wrapper tests covering exact UTF-8 spans, empty and embedded-NUL input,
  engine retention, 64 concurrent calls on one Mock engine, thread-local
  creation errors, and rejection of an unrepresentable result count.
- Two Metal arena tests covering SHARED buffer ownership and lookup, actual
  MiniLM inference, and zero additional **arena** allocations for the measured
  request after warmup. This counter does not measure process allocations or
  data movement.
- Two new Metal wrapper tests. One retains a result while releasing the caller's
  engine reference, checks that engine destruction waits for result release,
  and keeps a second engine usable throughout. The other issues 16 concurrent
  calls across two loaded engines, including Unicode text, and checks that
  retained output buffers remain unchanged. New vectors match each engine's
  initial output within an absolute tolerance of `5e-5`.
- Three TurboRerank ABI/header tests and six Inferstream codec, scratch-buffer,
  and catalog tests.

All model-dependent tests executed. The provider reported `Device(gpu, 0)`
with Metal available and produced 384-dimensional MiniLM vectors. These are
ownership and repeatability checks, not an independent embedding-quality
golden comparison. The full XCTest run took 4.209 seconds; the six Swift
Testing tests took 0.006 seconds. Those times are test-run durations, not
inference benchmarks.

The [saved test output](../testdata/receipts/turboembed/apple-ownership-2026-09-14.log)
contains both runs. No sanitizer instrumentation was enabled on the Mac.
The Mac's changed source files and package lockfile matched the local files
by SHA-256 after validation. The original Mac checkout remained clean and its
existing server process remained running.

## Scope

This work does not qualify Apple against the Intel prepared SDK performance
budget or add an Apple implementation of the prepared ABI. Apple FFM, Android
JNI, broad model qualification, hosted CI, publication, and deployment were
not exercised.
