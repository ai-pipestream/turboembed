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
  frame comes back. Runs on one model take turns. A frame that fails or
  times out on the device (10 s) fails the run with `TURBO_E_RUNTIME`
  and HailoRT's status, and the session then refuses every write and run
  with `TURBO_E_INVALID_STATE`: the device may still own the frame's
  memory. Releasing the session waits for each frame still in flight
  for 10 s at most; once one has not come back, there or in a run, the
  rest are not waited for, so on a device that stopped answering the
  session's release returns after 10 s at most instead of hanging. The
  frame sets whose frames did not come back are kept by the model, since
  the device may still write into them, and freed as soon as their jobs
  finish, checked each time a session on the model is made or released.
  When the model goes, those still running are detached from HailoRT and
  their memory is left allocated. That bound is for the session: the
  model's own last release runs HailoRT's shutdown of the configured
  model, which waits for its transfers on a timeout of its own (10 s in
  HailoRT 4.23) that this backend does not set.
  `TURBO_HAILO_FRAME_TIMEOUT_MS`, read when a session is made, sets the
  frame timeout to a whole number of ms from 0 to 600000 instead of 10 s;
  any other value refuses the session with `TURBO_E_INVALID_ARGUMENT`.
  `core/tests/hailo_release.rs` sets 0, so frames genuinely miss it on a
  device that answers, and checks that the release then returns at once,
  that the frame sets kept are freed once their frames come back, over
  timeouts and recoveries repeated, that the model released straight
  after such a session returns as soon as the device answers, and that
  the device runs a new session. A token type other
  than 0 is `TURBO_E_UNSUPPORTED_OPTION`, naming the row: the HEF
  computes type 0 only.
- **Late pooling and `cpu_threads`.** The session is made through
  `session_open`, and its vectors hold `max(max_batch, max_ranges)`
  rows. After `turbo_embed_write_ranges` the run sends every row as
  usual, and as each frame comes back the host pools every range of
  that frame's row: the mean of the dequantized hidden states of the
  range's tokens whose mask is 1, summed in double in column order, cut
  to `output_dim`, normalized when asked, and written at the range's
  index. Frames come back in any order, so a vector's place is always
  the range's index in the list, never its row. A range over a whole
  row gives that row's mean vector bit for bit. The device is not a
  CPU, so `turbo_session_desc.cpu_threads` must be 0; any other value
  is `TURBO_E_UNSUPPORTED_OPTION` naming field 6.
- **What a result reports.** Stages: upload and encode on the device,
  lookup, pooling and normalize on the host, and the hidden states
  downloaded; the vectors are `TURBO_PLACE_HOST`. `h2d_bytes` is each
  frame's rows and bias, `d2h_bytes` each frame's hidden states.
  `host_allocs` and `device_allocs` are 0: the backend allocates nothing
  in a run. What HailoRT allocates inside `run_async` is its own and is
  not counted.

## Bundles

The bundle tool makes the HEFs (bundle/README.md, step 4): the recipe's
`hef-hailo10h-s128` and `hef-hailo8-s128` artifacts are compiled from the
upstream ONNX export by Hailo's Dataflow Compiler, in a container built
locally, on the calibration texts the recipe carries
(`recipes/all-minilm-l6-v2.calibration.jsonl`). The compiler is not the
same for every chip: the Hailo-10H's is the 5.x line
(`bundle/hailo/Dockerfile`, image `turbo-hailo-dfc`), the Hailo-8's and
Hailo-8L's the 3.x line on Python 3.10 (`bundle/hailo/Dockerfile.hailo8`,
image `turbo-hailo-dfc3`). Each runs the same `hef_compile.py`, with the
artifact's `target` as the compiler's architecture. A HEF made by a 3.x
compiler loads in HailoRT 4.x; 3.34.0 makes HEFs that HailoRT 4.23.0
runs.

A HEF is compiled for one architecture (`target`) and a fixed frame:
`fixed_seq` tokens, one row. A case longer than `fixed_seq` is refused
with `TURBO_E_CAPACITY`. The backend loads the HEF whose `target` is the
device's `arch`.

On the Hailo-10H (Raspberry Pi 5, HailoRT 5.1.1, firmware 5.1.1),
all-MiniLM-L6-v2 is compiled in I8 from the upstream ONNX cut at the
word-embedding gather and the attention mask (128 tokens) by the
Dataflow Compiler 5.4.0 with
`compiler_optimization` `"max"`, with a minimum cosine of 0.937 to the
F32 reference vectors, over the tier's floor of 0.93. Frames run in
bursts of 32, so a HEF's contexts are loaded once a burst. Through the
library (one session, `TURBO_PRECISION_MODEL`, mean pooling, normalized;
one row the mean of 200 runs, 64 rows the mean of 20, at commit
a74f0b5a, 2026-10-23):

| Compile | Contexts | Compile time (32 cores) | One row | 64 rows a run | `hailortcli run2 --batch-size 1` | `--batch-size 32` |
|---|---|---|---|---|---|---|
| default | 6 | about 3 min | 6.0 ms | 670 rows/s | 224 frames/s | 783 frames/s |
| `"max"` | 4 | about 25 min | 5.1 ms | 908 rows/s | 276 frames/s | 1186 frames/s |

The two give the same cosines to the reference cases and to 60
held-out texts (minimum 0.937, mean 0.972).

On the Hailo-8 (26 TOPS, on a Raspberry Pi 5 and on a Compute Module 5,
HailoRT 4.23.0), the same model is compiled by 3.34.0 with
`compiler_optimization` `"max"`, with a minimum cosine of 0.936 to the
F32 reference vectors, over the tier's floor of 0.93. The Hailo-8 has no
memory of its own, so a HEF's contexts are loaded in turn from the host
for every burst of frames: a run of one row pays for all of them, and
rows a second climb with the batch. Through the library (one session,
`TURBO_PRECISION_MODEL`, mean pooling, normalized; one row the median of
500 runs), the same on both boards (2026-10-06):

| Compile | Contexts | Compile time (32 cores) | One row | 32 rows a run | 64 rows a run | `hailortcli run --batch-size 32` |
|---|---|---|---|---|---|---|
| default | 4 | about 6 min | 14.7 ms | 282 rows/s | 301 rows/s | 333 frames/s |
| `"max"` | 3 | 45 to 50 min | 14.1 ms | 337 rows/s | 365 rows/s | 424 frames/s |

The two give the same vectors to the last digit the conformance suite
reports: the setting moves layers between the device's resources and
changes no weight or quantization.

## Device or CPU

A Hailo-8 has no memory of its own, so a frame of one row pays for
loading every context from the host, and a frame is always `fixed_seq`
tokens however few the row holds. The host CPU's time follows the
tokens instead. So for short rows sent one a run the CPU is faster, and
the embed crate's `Calibration` (`analysis/embed/src/calibrate/`) decides
per configuration. `Calibration::measure` runs the sentence vector step
over the caller's texts at each window (tokens a row) and batch (rows a
run) asked, on the device at `Precision::Model` and on the CPU at
`Precision::Exact`, and keeps the lowest cosine of the device's vectors
to the CPU's and the fastest time of each side.
`Calibration::side` gives the device for a configuration where its
cosine clears the floor (0.93 for I8) and it is faster than the CPU, and
the CPU, at `Precision::Exact`, elsewhere; a configuration that was not
measured is an error. The batch is the session's cap on rows a run
(`max_batch`), not the rows a call sends: a call of fewer documents runs
fewer rows, on the side its cap chose, so a document's vectors do not
depend on what else a call sends. The caller's texts are what is timed,
so the short runs at the end of a call are in the times as far as the
texts have them; a caller whose calls are mostly a sentence or two
calibrates and opens the step with a batch of 1.
`Calibration::load_or_measure` keeps the measurement in a directory,
keyed by the bundle, the device, its runtime and driver, the library's
build and what was asked, and `Calibration::annotator` opens the step on
the chosen side. `TURBO_CALIBRATION_FORCE` set to `device` or `cpu`
overrides the choice; any other value is an error.

Measured on the Hailo-8 (on a Raspberry Pi 5, HailoRT 4.23.0, firmware
4.23.0) with the `"max"` HEF of the bundle above, against the Pi 5's CPU
(4 Cortex-A76 cores, the `cpu` backend at `Precision::Exact`, F32), at
commit 0ddf5820, 2026-10-23. The texts are the 12 documents of
`analysis/embed/tests/data/calibration-texts.txt` three times over (168
sentences); each time is the median of 5 runs of the whole step:

| Window | Rows a run | Min cosine | Hailo-8 | CPU | Runs on |
|---|---|---|---|---|---|
| 16 | 1 | 0.965 | 2996 ms | 1632 ms | CPU |
| 16 | 8 | 0.965 | 1239 ms | 1477 ms | Hailo-8 |
| 16 | 32 | 0.965 | 640 ms | 1394 ms | Hailo-8 |
| 32 | 1 | 0.962 | 1602 ms | 1521 ms | CPU |
| 32 | 8 | 0.962 | 671 ms | 1393 ms | Hailo-8 |
| 32 | 32 | 0.962 | 351 ms | 1385 ms | Hailo-8 |
| 64 | 1 | 0.949 | 799 ms | 1294 ms | Hailo-8 |
| 64 | 8 | 0.949 | 334 ms | 1242 ms | Hailo-8 |
| 64 | 32 | 0.949 | 176 ms | 1243 ms | Hailo-8 |
| 128 | 1 | 0.935 | 504 ms | 1150 ms | Hailo-8 |
| 128 | 8 | 0.935 | 223 ms | 1109 ms | Hailo-8 |
| 128 | 32 | 0.935 | 133 ms | 1125 ms | Hailo-8 |

The crossover is at one row a run: the CPU wins up to 32 tokens a row
and the Hailo-8 from 64. From 8 rows a run the Hailo-8 wins at every
window, by 1.2 times at 16 tokens and 8 rows and up to 8.4 times at 128
tokens and 32 rows. Every window clears the floor. The same measurement
on the Hailo-8 on a Compute Module 5 gives the same cosines, device times
within 0.2% and CPU times within 5%, and the same choice at every
configuration. Opened through the calibration, each configuration ran
on the side chosen, within 10% of that side's measured time, and gave a
document sent alone the vectors it got among the rest. One sentence sent
alone took 14.1 ms on the Hailo-8 and 7.5 ms on the CPU at every
configuration: a call that small, sent to a session whose batch chose
the Hailo-8, spends about 6.5 ms more than the CPU would.

`Dispatch` chooses per call instead of per configuration: it holds the
step on the Hailo-8 and on the CPU at one window and cap, and runs each
call whole on one side. A call the calibration timed (the whole set, its
first sentence alone or its first text alone) goes to the side measured
faster, over every cap it was timed at; any other goes to the side
whose cost is lower, fitted to the calibration's times as a cost a run,
a row and a token. The calibration keeps each side's fastest time, since
another process on the machine can only slow a run, and a call's runs,
rows and tokens are planned the way it runs. Each document's layers
record the side that made its vectors. Through one instance on the
Hailo-8 on a Raspberry Pi 5 at commit 5f46f682, 2026-10-24, the middle
of six medians each (dispatched, then the step fixed on the Hailo-8 and
on the CPU, in ms):

| Window | Cap | One sentence | One document | All 168 sentences |
|---|---|---|---|---|
| 16 | 1 | CPU 7.5 (14.1, 7.5) | CPU 46.6 (84.5, 47.0) | CPU 1619 (2999, 1617) |
| 16 | 8 | CPU 7.6 (14.1, 7.5) | CPU 41.5 (41.9, 41.5) | Hailo-8 1242 (1239, 1417) |
| 16 | 32 | CPU 7.6 (14.1, 7.6) | CPU 41.3 (41.9, 41.3) | Hailo-8 640 (640, 1424) |
| 32 | 1 | CPU 7.5 (14.1, 7.5) | Hailo-8 42.2 (42.2, 42.7) | CPU 1514 (1605, 1511) |
| 32 | 8 | CPU 7.5 (14.1, 7.5) | Hailo-8 35.2 (35.2, 40.5) | Hailo-8 675 (674, 1398) |
| 32 | 32 | CPU 7.6 (14.1, 7.6) | Hailo-8 35.2 (35.2, 40.7) | Hailo-8 351 (351, 1387) |
| 64 | 1 | CPU 7.5 (14.1, 7.5) | Hailo-8 28.2 (28.2, 36.0) | Hailo-8 802 (802, 1290) |
| 64 | 8 | CPU 7.6 (14.1, 7.6) | Hailo-8 28.2 (28.2, 35.4) | Hailo-8 337 (337, 1252) |
| 64 | 32 | CPU 7.6 (14.1, 7.6) | Hailo-8 28.2 (28.2, 35.6) | Hailo-8 176 (176, 1243) |
| 128 | 1 | CPU 7.5 (14.1, 7.5) | Hailo-8 14.1 (14.1, 31.7) | Hailo-8 507 (507, 1144) |
| 128 | 8 | CPU 7.6 (14.1, 7.6) | Hailo-8 14.1 (14.1, 32.0) | Hailo-8 223 (223, 1107) |
| 128 | 32 | CPU 7.6 (14.1, 7.6) | Hailo-8 14.1 (14.1, 32.3) | Hailo-8 133 (133, 1122) |

On both boards every call ran whole on one side, both boards chose the
same side in every one of the 36 cells, and the side was the faster
wherever the two sides' six rounds did not overlap. They overlapped in
two cells, both on the Compute Module 5, and both ran on the CPU: one
document at window 16 and cap 8 (Hailo-8 41.91 ms, 41.89 to 41.91; CPU
41.31 ms, 41.29 to 44.75) and at window 16 and cap 32 (Hailo-8 41.92 ms,
41.90 to 41.99; CPU 41.25 ms, 41.16 to 41.94). The Hailo-8's times on
the two boards were within 0.23% of each other and the CPUs' within
2.6%.

What dispatch adds is read from six pairs of the dispatched call and the
step fixed on the side it chose, timed back to back: the middle
difference of the quiet pairs, those whose two times are each within
0.5% of the fastest on their side. Planning a call took 0.001 to
0.08 ms. The reading was -0.05 to +0.72 ms on the Raspberry Pi 5 and
-0.02 to +3.90 ms on the Compute Module 5, and at most 0.35% of the
fixed time in every cell (the Compute Module 5's one sentence at
window 128 and cap 8, +0.027 ms); the largest in ms, +3.90 ms, is 0.26%
of the Compute Module 5's whole set at window 32 and cap 1, on the CPU.
One cell gave no reading: the Raspberry Pi 5's whole set at window 16
and cap 1, which ran on the CPU (dispatched 1618.9 ms; CPU 1616.7 ms,
1610.7 to 1620.5; Hailo-8 2998.7 ms, 2998.6 to 2998.7). Fewer than four
of its six pairs were quiet in each of three sets, so the test reported
it inconclusive. That cell is unverified: under
`TURBO_TEST_REQUIRE_HAILO=1` the test now fails on an inconclusive
submission, so it needs a quieter run to pass. Every other reading
was within the planning time and 0.5% of the fixed time, which the test
requires.

On the Hailo-10H (on a Raspberry Pi 5, HailoRT 5.1.1, firmware 5.1.1),
which has memory of its own, the same measurement at commit 67e1327b
(the CPU then timed at `Precision::Model`, which computes this model in
F32 as `Precision::Exact` does) chose the device at every configuration:
its slowest, 16 tokens and one row a run, took 1025 ms.

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

# A session released after its frames missed their timeout (in a binary of
# its own, since it sets TURBO_HAILO_FRAME_TIMEOUT_MS):
TURBO_TEST_BUNDLE=<bundle-dir> \
    cargo test --release -p turbo --features hailo --test hailo_release -- --include-ignored --nocapture
```

The tests that need a bundle are ignored without `--include-ignored`,
and fail if `TURBO_TEST_BUNDLE` is not set.
The listing test compares the devices listed with what
`hailo_scan_devices` returns, and each one's `arch` with its PCI device
id in sysfs (`1e60:45c4` is a Hailo-10H; `1e60:2864` a Hailo-8 or
Hailo-8L). HailoRT writes `hailort.log` into the directory a test runs
in.
