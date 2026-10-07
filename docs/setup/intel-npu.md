# Intel NPU (AI Boost)

The `npu` backend runs on the NPU of Intel Core Ultra processors (Meteor
Lake, Arrow Lake, Lunar Lake, Panther Lake) through the Level Zero
driver's graph extension. The driver compiles an OpenVINO IR from the
bundle on the machine itself; the library links no OpenVINO. It is a
separate backend from the Intel GPU one, and the two features may be on
together. Backend reference: [../npu.md](../npu.md).

## Prerequisites

To build: nothing beyond Rust. The loader is opened at run time.

To run:

- The Level Zero loader, 1.10 or newer (`zeInitDrivers`):
  `libze_loader.so.1` on Linux, `ze_loader.dll` on Windows.
- Intel's NPU driver. On Linux: the kernel's `intel_vpu` module (in
  mainline Linux since 6.3), a `/dev/accel/accel*` node the user can
  open (usually the `render` group), and the user-space driver
  `libze_intel_vpu.so` from the packages at
  github.com/intel/linux-npu-driver. On Windows: Intel's NPU driver
  package, which installs the loader too; Device Manager lists "Intel(R)
  AI Boost".
- A driver whose graph extension is 1.8 or newer and whose compiler is
  5.9 or newer. Model loading names either when it is older.

```
scripts/setup/intel-npu.sh            # check (Linux)
scripts/setup/intel-npu.sh --install  # the Level Zero loader from the distribution
```

The script checks the module, the device node and its permissions, the
loader's version and the NPU driver. The NPU driver is not in Ubuntu's
archive; the script names the release page instead of installing it.

On Windows, run the cargo commands below from a shell whose `PATH` finds
`ze_loader.dll`. A clone must keep `testdata/` and `bundle/recipes/*.jsonl`
byte for byte (`.gitattributes` marks them `-text`); a clone made with
`core.autocrlf` rewriting them fails the bundle hashes, and
[../npu.md](../npu.md), Windows notes, has the two commands that refresh
them.

## Build

```
cargo build --release -p turbo --features npu
```

No build variables. `turbo_version()` names `npu`.

## A bundle

The NPU loads `FORMAT_OPENVINO_IR` artifacts only: two files, the xml
and its weights, compiled at a fixed shape (`fixed_seq` and
`fixed_batch`), with `npu` in `backends`. A bundle of weights alone has
no artifact for it and is refused with `TURBO_E_BUNDLE_NO_ARTIFACT`.

- Prebuilt: the MiniLM and BGE small, base and large prereleases on the
  repository's releases page carry the `openvino-f16` IR (F16, 1 x 128
  tokens) beside the F32 weights ([bundles.md](bundles.md)).
- Made: the all-MiniLM-L6-v2 and bge-small, base and large recipes list
  `openvino-f16`, which `turbo-bundle make` converts in the reference
  container.

The MiniLM recipe also has `openvino-embeddings-f16`, an IR that starts
after the word-embedding lookup (`INPUT_EMBEDDINGS`; the host gathers
the rows). The loader takes the first artifact it can run, so a bundle
with both runs the token-id IR.

## Conformance

```
export TURBO_TEST_REQUIRE_NPU=1

cargo test -p turbo --features npu
TURBO_TEST_BUNDLE=models/bundles/all-minilm-l6-v2 \
    cargo test -p turbo --features npu --test npu -- --include-ignored
TURBO_TEST_BUNDLE=models/bundles/all-minilm-l6-v2 TURBO_TEST_DEVICE=npu \
    cargo test -p turbo --features npu --test conformance -- --include-ignored --nocapture
```

Reference cases longer than the compiled 128 tokens are checked to be
refused with `TURBO_E_CAPACITY` and are not compared.

## Tiers

| Tier | Computes in | Notes |
|---|---|---|
| MODEL | the graph's output dtype, F16 for the published IRs | |
| FASTEST | the same | MODEL and FASTEST are one path. |
| EXACT | refused | `TURBO_E_UNSUPPORTED_OPTION`, field 3: the graph is compiled in F16. |

F16 is held to cosine 0.999 against the reference.

## Environment

The backend reads no variable of its own. `TURBO_NPU_GRAPH_FORMAT` and
`TURBO_NPU_GRAPH_INPUT` appear in benchmark records as settings the run
had, not as switches.

## Limits

- F16 only; no EXACT.
- A fixed shape: the published IRs take one row of 128 tokens a frame.
  A session's `max_seq` is at most the IR's `fixed_seq`, and a text
  longer than that is refused with `TURBO_E_CAPACITY` unless the caller
  cuts it: `turbo_embed_options.max_tokens = 128`, or the session's
  `max_seq`, with the bundle's truncation side. With `max_tokens` 0 the
  core cuts at the bundle's own `max_seq` (256 or 512), not the IR's.
- Token type 0 only, for an IR with no token-type input.
- Host buffers only.
- The device wait gives up after 30 seconds and fails the run.

## Serving

`cargo build --release -p turbo-kserve --features npu`, then
[../grpc.md](../grpc.md). Set each model's `max_seq` to the IR's
`fixed_seq`, and have clients send `max_tokens` when texts may be
longer.
