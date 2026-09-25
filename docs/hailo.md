# The Hailo backend

The `hailo` backend runs embed sessions on Hailo accelerators through
HailoRT: its C API to list devices and its C++ InferModel API to run a
compiled graph (a HEF). It is C++ in `core/hailo/`, compiled by `core/build.rs` into a static
library that libturbo links against HailoRT's shared library, and the
core reaches it only through its `turbo_backend` table
(`include/turbo/turbo_backend.h`). It is off by default: the `hailo`
feature of the `turbo` crate links it, and `turbo_version()` then names
`hailo`.

It loads `FORMAT_HEF` artifacts alone (docs/bundle.md): a bundle needs a
HEF compiled for the device's architecture, and a bundle of raw weights
alone has no artifact for it (`TURBO_E_BUNDLE_NO_ARTIFACT`).

## Requirements

- Linux, with a Hailo PCIe driver loaded: `hailo1x_pci` for a Hailo-10H,
  `hailo_pci` for a Hailo-8 or Hailo-8L.
- HailoRT's headers (`include/hailo/hailort.h`) and `libhailort.so`, of
  the same version as the driver. On Raspberry Pi OS these come with
  `hailo-h10-all` (HailoRT 5.1.1) or `hailo-all` (HailoRT 4.23.0).
- A host C++17 compiler.

## Building

```
cargo build -p turbo --release --features hailo
```

| Variable | Meaning |
|---|---|
| `TURBO_HAILORT_ROOT` | The prefix HailoRT is installed under, with `include/hailo/hailort.h` and `libhailort.so` in `lib/`, `lib/<arch>-linux-gnu/` or `lib64/`. Unset: `/usr`. |
| `CXX` | The C++ compiler. Unset: `c++`. |

The library links `libhailort.so` with its directory as a run path. A
machine that runs it needs HailoRT; a build without the feature does
not. Without a driver HailoRT scans nothing, and the backend lists no
device.

## What it does

- **Devices.** One per device `hailo_scan_devices` returns, in its
  order, each opened once per process and identified with
  `hailo_identify`. `arch`, the label benchmark records are filed under,
  is the architecture identify reports: `hailo10h`, `hailo8`, `hailo8l`.
  `name` is the product name identify gives, else the board name, else
  the architecture's (`Hailo-10H`, which names no board). `kind` is
  `TURBO_DEVICE_NPU`. `memory_total` and `memory_free` are 0: HailoRT
  reports neither. `runtime_version` is HailoRT's library version, and
  `driver_version` the kernel module's and the firmware's
  (`hailo1x_pci 5.1.1, firmware 5.1.1`). A device HailoRT scans but that
  does not answer identify is left out, and the runtime's log says why.
  The list is made the first time a runtime asks for devices and kept
  for the life of the process: load the driver before the process
  starts, and restart it after a device that failed to answer is
  brought back.
- **Capability.** Embed is EXPERIMENTAL at `MODEL` and `FASTEST`,
  computing in I8, honoring every field of `turbo_embed_options`. `EXACT`
  is UNSUPPORTED: a HEF computes in the dtype it was compiled in, never
  F32.
- **Contexts.** A context is the device's HailoRT vdevice. HailoRT opens a
  device for one vdevice at a time in a process, so every context on a
  device shares one, made by the first and released with the last.
- **Buffers.** `HOST` only: memory aligned to 64 bytes, exported as a
  `TURBO_HANDLE_HOST_PTR`. `PINNED`, `DEVICE` and `SHARED` are
  `TURBO_E_UNSUPPORTED`: the device's memory is HailoRT's, reached only
  through the frames it sends.
- **Models.** A `FORMAT_HEF` artifact from `INPUT_EMBEDDINGS` to
  `OUTPUT_HIDDEN_STATES`, `compute_dtype` `DTYPE_I8`, `fixed_seq` and
  `fixed_batch` given, `fixed_batch` 1, with an F32 `word_embeddings`
  table in its `host_weights`. Anything else is refused, naming what.
  `VDevice::create_infer_model` reads the HEF from the bytes the core
  hashed, and the model is configured once. Its two inputs, the word
  rows `[fixed_seq, hidden]` and the attention bias
  `[fixed_seq, heads * fixed_seq]`, and its output, the hidden states
  `[fixed_seq, hidden]`, are read from the HEF by size, each with its
  quantization; a HEF whose streams are not these is
  `TURBO_E_BUNDLE_INVALID`, naming the stream. A HEF compiled for
  another architecture fails in HailoRT, as `TURBO_E_RUNTIME` with its
  status.
- **Sessions.** Two frames, their bindings, and the session's copy of
  the rows are allocated when the session is made. A row is one frame:
  each token's word row is gathered from the table on the host and
  quantized as it is written into the frame, and the bias is 0 for a
  key the mask keeps and -100 for one it drops, the same for every head
  and query. The frames go through `run_async` two at a time, so one
  fills on the host while the other runs, and each frame's hidden states
  are dequantized, pooled (mean over the mask, the first token, or the
  last live one), cut to `output_dim`, and normalized on the host as the
  frame comes back. Runs on one model take turns. A token type other
  than 0 is `TURBO_E_UNSUPPORTED_OPTION`, naming the row: the HEF
  computes type 0 only.
- **What a result reports.** Stages: upload and encode on the device,
  lookup, pooling and normalize on the host, and the hidden states
  downloaded; the vectors are `TURBO_PLACE_HOST`. `h2d_bytes` is each
  frame's rows and bias, `d2h_bytes` each frame's hidden states.
  `host_allocs` and `device_allocs` are 0: the backend allocates nothing
  in a run. What HailoRT allocates inside `run_async` is its own and is
  not counted.

## Bundles

The HEF is compiled for one architecture (`target`, `hailo10h`) and a
fixed frame: `fixed_seq` tokens, one row. A case longer than `fixed_seq`
is refused with `TURBO_E_CAPACITY`. On the Hailo-10H, all-MiniLM-L6-v2
compiled in I8 from the upstream ONNX cut at the word-embedding gather
and the attention mask (128 tokens) runs at 224 rows a second with two
frames in flight, the rate `hailortcli benchmark` gives the same HEF
(Raspberry Pi 5, HailoRT 5.1.1, 2026-09-25).

## Testing on a Hailo machine

From the workspace root. `TURBO_TEST_REQUIRE_HAILO=1` makes every test in
`core/tests/hailo.rs` that needs a device fail when the backend lists
none, where without it the test passes with a line saying it was
skipped; set it on a Hailo machine, so a run that found no device cannot
pass.

```
export TURBO_TEST_REQUIRE_HAILO=1

# Everything, with the Hailo-only tests in core/tests/hailo.rs:
cargo test -p turbo --features hailo

# The Hailo-only tests, with a bundle that has a HEF for the device, and
# conformance on it (docs/conformance.md):
TURBO_TEST_BUNDLE=<bundle-dir> \
    cargo test --release -p turbo --features hailo --test hailo -- --include-ignored --nocapture
TURBO_TEST_BUNDLE=<bundle-dir> TURBO_TEST_DEVICE=hailo \
    cargo test --release -p turbo --features hailo --test conformance -- --include-ignored --nocapture
```

The tests that need a bundle are ignored without `--include-ignored`,
and fail if `TURBO_TEST_BUNDLE` is not set.
The listing test compares the devices listed with what
`hailo_scan_devices` returns, and each one's `arch` with its PCI device
id in sysfs (`1e60:45c4` is a Hailo-10H; `1e60:2864` a Hailo-8 or
Hailo-8L). HailoRT writes `hailort.log` into the directory a test runs
in.
