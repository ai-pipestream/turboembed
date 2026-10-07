# Apple silicon (Metal)

The `metal` backend runs on the GPU of Apple silicon Macs through Metal.
The host side is Objective-C++ compiled with the Xcode command line
tools; the kernels are carried in the library as source and compiled by
Metal for the GPU when a context is made. Backend reference:
[../metal.md](../metal.md).

## Prerequisites

- An Apple silicon Mac (M1 or later) on macOS 14 or later.
- The Xcode command line tools (clang, the macOS SDK, `xcrun`). Xcode
  itself and its offline `metal` compiler are not needed.
- Rust, through rustup, from an arm64 shell (not under Rosetta).

```
scripts/setup/metal.sh             # check
scripts/setup/metal.sh --install   # start Apple's installer for the command line tools
```

An Intel Mac, a Mac with an AMD GPU, or macOS before 14 lists its GPU,
reports its capability cell as UNSUPPORTED with the reason, and refuses
a context.

## Build

```
cargo build --release -p turbo --features metal
```

The library is `target/release/libturbo.dylib`; `turbo_version()`
returns `0.1.0 metal cpu`. The build fails on any target but macOS.

## A bundle

Any bundle with a `FORMAT_SAFETENSORS` artifact whose `backends` lists
`metal` (every recipe's `weights-f32`): see [bundles.md](bundles.md).
The reference container runs under Docker Desktop on the Mac, or make the
bundle on Linux and copy the directory over.

## Conformance

```
export TURBO_TEST_REQUIRE_METAL=1

TURBO_TEST_DEVICE=metal cargo test --release -p turbo --features metal --test conformance -- --nocapture

TURBO_TEST_BUNDLE=models/bundles/all-minilm-l6-v2 TURBO_TEST_DEVICE=metal \
    cargo test --release -p turbo --features metal --test conformance -- --include-ignored --nocapture
TURBO_TEST_BUNDLE=models/bundles/all-minilm-l6-v2 \
    cargo test --release -p turbo --features metal --test metal -- --include-ignored --nocapture
```

## Tiers

Every tier computes in F32, so MODEL, FASTEST and EXACT give the same
vectors and are held to F32's bound (cosine 0.9999, largest absolute
difference 1e-4). A model stored in F16 or BF16 is refused at MODEL and
computes from an F32 copy at FASTEST and EXACT. Apple GPUs have no F64,
so the sums the CPU encoder takes in F64 are taken in F32 here.

The committed records for an M2 (`benchmarks/records/m2.metal.*`) cover
all-minilm-l6-v2 at each tier, measured against text-embeddings-inference
running on Metal on the same Mac.

## Environment

The backend reads no variable of its own.

## Limits

- F32 only.
- Hidden and intermediate widths must be multiples of 8; another model
  is refused at load.
- For models whose heads take the narrow attention kernel, `max_seq` is
  bounded by the GPU's threadgroup memory: about 8000 tokens.
- A session's scratch must fit one Metal buffer.

## Serving

`cargo build --release -p turbo-kserve --features metal`, then
[../grpc.md](../grpc.md).
