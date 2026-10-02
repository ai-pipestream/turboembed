# The npu backend

The `npu` backend runs embed sessions on Intel NPUs (the AI Boost tile
of Core Ultra parts) through the oneAPI Level Zero driver's graph
extension, the lowest level Intel exposes for the NPU: the loader's
`zeInitDrivers` with the NPU driver type, `zeDriverGetExtensionFunctionAddress`
for `ZE_extension_graph`, and the extension's own graph create,
initialize and execute. It is Rust in `core/src/npu.rs` and
`core/src/npu/`, and the core reaches it only through its
`turbo_backend` table (`include/turbo/turbo_backend.h`). It is off by
default: the `npu` feature of the `turbo` crate links it, and
`turbo_version()` then names `npu`.

It is not the `levelzero` backend. That one is the Intel GPU path: its
own kernels, compiled from OpenCL C to SPIR-V, on the loader's GPU
drivers. The two features are separate, list separate devices, and may
be on together; an Arrow Lake machine then lists its Arc GPU under
`levelzero` and its AI Boost under `npu`.

It loads `FORMAT_OPENVINO_IR` artifacts alone (docs/bundle.md): the
driver carries OpenVINO's NPU compiler and builds the graph from the
IR's xml and weights on the machine itself, so the library links no
OpenVINO and no ONNX Runtime, at build time or at run time. OpenVINO
appears in this tree only as a reference program the benchmark tool
runs in a container (docs/benchmarks.md); ONNX files in a bundle are
for the reference programs and the converters, never executed by the
core. A bundle of raw weights alone has no artifact for this backend
(`TURBO_E_BUNDLE_NO_ARTIFACT`), and nothing falls back to the CPU in
its place.

## Requirements

- To build: nothing of Level Zero. The loader is opened at run time.
- To run: the Level Zero loader (`ze_loader.dll` on Windows,
  `libze_loader.so.1` elsewhere, 1.10 or newer for `zeInitDrivers`) and
  Intel's NPU driver for it (on Windows the Intel NPU driver package;
  on Linux `intel/linux-npu-driver`'s `libze_intel_vpu.so` over the
  kernel's `intel_vpu` module and `/dev/accel`).
- A driver whose graph extension is 1.8 or newer and whose compiler is
  5.9 or newer. Model loading says plainly when either is older.
  The published graph header's current version is 1.20, and its
  function table is 33 pointers. Versions 1.17, 1.18 and 1.19 add no
  pointers, so a copy of that header that stops at 1.18 ends at
  `pfnEvict` (31 pointers, index 30). Version 1.20 appends
  `pfnGetArgumentProperties4` and `pfnGetArgumentNames`. This backend
  does not call that pair. A driver that advertises 1.17, which is
  what the Arrow Lake listing reported, is never read past the fields
  that version covers.

The loader is opened when the first runtime lists its devices, not when
the library loads, so a machine without it runs everything else as
usual and the backend lists no device.

## Building

```
cargo build -p turbo --release --features npu
```

No build variables: there is nothing to point at.

## What it does

- **Devices.** `zeInitDrivers` is asked for NPU drivers alone. For each
  driver that lists `ZE_extension_graph` among its extensions, the
  extension's function table is taken; a driver that advertises the
  extension and then refuses the table is an error, not an absence.
  Each of the driver's devices of type NPU is then probed with the
  extension's `pfnDeviceGetGraphProperties`; a device is listed only
  when the probe answers, and one that does not answer is left out with
  nothing standing in for it. The version that gates later calls is the
  one `ZE_extension_graph` advertises. On a driver of 1.6 or newer the
  probe also calls `pfnDeviceGetGraphProperties2`. A
  `graphExtensionVersion` of 0 in either struct is the field left
  unwritten, not version 0.0, and the advertised version is kept. A
  non-zero device version older than the advertised one is the one
  used. No driver, no NPU device, or no graph
  extension lists nothing, and that is not an error. `kind` is
  `TURBO_DEVICE_NPU`, `unified_memory` 1 (the NPU computes over the
  host's memory), `memory_total` what the driver reports for the
  device, `memory_free` 0 (nothing reports it). `arch`, the label
  benchmark records are filed under, is from the PCI device id:
  `mtl-npu` (0x7d1d), `arl-npu` (0xad1d), `lnl-npu` (0x643e), `ptl-npu`
  (0xb03e), `intel-npu-<id>` for an Intel id not named yet.
  `runtime_version` is the loader's version and `driver_version` the
  driver's, in Intel's packing. When NPU hardware is seen and every
  device had to be skipped (no graph extension on its driver, or a
  probe that failed), listing is an error carrying each reason, never
  a quiet empty list on a machine that has the hardware. The list, or
  that error, is made the first time a runtime asks and kept for the
  life of the process: a driver fixed underneath a running process is
  seen by the next process, not this one.
- **Capability.** The backend reports embed as EXPERIMENTAL at
  `MODEL` and `FASTEST`, honoring `normalize`, `pooling` and
  `output_dim` (the core owns `truncate`, `max_tokens` and
  `prompt_role`, and sets their bits itself, so
  `turbo_capability.options_honored` reads 0b111111). The reported
  dtype is `TURBO_DTYPE_F16`, the compute dtype the published npu
  recipes declare, which the backend knows without loading an
  artifact. `model_load` still reads the compiled graph, and
  `turbo_session_get_info` says what a session really resolved. The
  backend does not write SUPPORTED. The core does, in
  `decide_embedded` (`core/src/record.rs`, docs/benchmarks.md).
  `Record::is_for` requires `rows.kind` `ROWS_MIXED`, the device
  arch, this build's OS, the backend, the task, the precision, and
  for npu `TURBO_NPU_GRAPH_FORMAT`. A matching record that falls
  short does not promote the cell. Two ROWS_MIXED MiniLM IR records
  are committed at `bc15354`, measured at library commit `99c9264`
  (Speed, below). They match arch `arl-npu`, `os` windows, backend
  `npu`, `TASK_EMBED`, `PRECISION_MODEL` and `PRECISION_FASTEST`,
  `DTYPE_F16`, library version `0.1.0`, and
  `TURBO_NPU_GRAPH_FORMAT=NGRAPH_LITE`, so `is_for` is true. Each is
  batch 1, seq 128, cases `[0, 1, 2, 3, 4, 5, 6, 7]`, `live_tokens`
  181: eight `[1, 128]` frames. `conformance.rows` is 16. The timed count
  is `rows.cases.len()` (8). The cases that fit seq are
  `conformance.rows` minus that count (8), and the rows list those
  eight distinct cases. OpenVINO was measured on the same rows.
  `falls_short` is none. On a Windows build of `0.1.0` whose device
  arch is `arl-npu` and whose graph format is `NGRAPH_LITE`,
  `decide_embedded` sets `MODEL` and `FASTEST` to SUPPORTED.
  `benchmark` is the record's file name, `cosine_floor` is
  `conformance.min_cosine` (0.999996097954919), and `speed_ratio` is
  the record's. The bundle is not part of the cell key. The files
  were measured on `sentence-transformers/all-MiniLM-L6-v2`. The two
  ROWS_DENSE files are reference evidence. `is_for` skips them, so
  they do not bind. A cell with no mixed record for it stays
  EXPERIMENTAL and `reason` is `no benchmark record for this cell`:
  another arch (`mtl-npu`, `lnl-npu`, `ptl-npu`, `intel-npu-<id>`),
  a `NATIVE` graph, or a build whose OS is not windows. A dtype of 0
  is not used to keep a record from matching.
  `EXACT` is UNSUPPORTED: a compiled graph computes in the dtype its
  IR fixed, never F32 throughout.
- **Contexts.** A context is a Level Zero context on the device and one
  in-order immediate command list, used under the context's lock.
- **Buffers.** `HOST` only: memory the driver allocated
  (`zeMemAllocHost`, 64-byte aligned), which the device reads directly,
  exported as a `TURBO_HANDLE_HOST_PTR`. `PINNED`, `DEVICE` and
  `SHARED` are `TURBO_E_UNSUPPORTED`.
- **Models.** A `FORMAT_OPENVINO_IR` artifact to
  `OUTPUT_HIDDEN_STATES`, from `INPUT_TOKEN_IDS` or from
  `INPUT_EMBEDDINGS`. Anything else is refused, naming what. Before any
  compile is tried, the device's own graph properties are checked:
  NGRAPH_LITE must be among its graph formats, and the highest opset
  any of the IR's layers names must be within
  `maxOVOpsetVersionSupported`; either refusal says so in words. The
  IR's two verified files are then handed to the driver's compiler as
  the graph extension takes them: the `ZE_GRAPH_FORMAT_NGRAPH_LITE`
  container (the compiler's version, a block count of 2, then the xml
  and the weights, each behind its u64 size), with build flags naming
  each input's and output's precision and layout as `<name>:<value>`,
  the Parameter or Result name. OpenVINO's NPU adapter from compiler
  5.9 writes `<index>:<value>` in `get_parameters()` order. The flags
  here use the name. What those flags need, and
  nothing else, is read from the IR's xml: each Parameter's element
  type and rank, each Result's port precision and rank. The graph is
  created with `pfnCreate3` where the extension has it (1.12+), so a
  compile failure carries the compiler's own log in the
  `TURBO_E_RUNTIME` message, else `pfnCreate2`. When the device lists
  `ZE_GRAPH_FORMAT_NATIVE`, that compiled graph is not the one
  initialized. Its blob is copied from `pfnGetNativeBinary2` (extension
  1.7 or later; the driver owns the view) or from `pfnGetNativeBinary`
  (the caller owns the buffer). The `NGRAPH_LITE` graph is destroyed.
  A new graph is created with `ZE_GRAPH_FORMAT_NATIVE` from those
  bytes and an empty build-flag string, and that graph is initialized.
  An empty blob, a missing export, or a driver refusal of the blob is
  `TURBO_E_RUNTIME` or `TURBO_E_UNSUPPORTED`, and the `NGRAPH_LITE`
  graph is not kept. A device that does not list `NATIVE` initializes
  the `NGRAPH_LITE` graph and says so in the debug log, with the
  formats bitfield. That is the path the `383a7d1` receipt ran: on
  intel-npu, driver `0.15.21738`, `graphFormatsSupported` was `0x2`,
  so bit `0x1` was clear and the load stayed on `NGRAPH_LITE`. A load
  whose graph was created as `NATIVE` has not been run on intel-npu.
  Initialization (`pfnGraphInitialize`, or appended and
  synchronized, as the graph's properties ask) is the weights' move to
  the device. The
  compiled arguments are taken by name, `input_ids`, `attention_mask`
  and optionally `token_type_ids`, the names a BERT export gives; a
  graph whose input is named anything else is refused naming it, and
  no argument is ever assigned a role by position. Among several
  outputs only `last_hidden_state` is taken; a single output is the
  hidden states whatever its name. The compiled boundary is checked
  in full: ids, the mask and token types (where the graph takes them)
  are I64 or I32 of the same `[batch, seq]`, with device layout NC;
  the hidden states are FP32 or FP16 `[batch, seq, hidden]`, with
  device layout CHW. Those are the packed layouts the build flags
  asked for. A blocked or other device layout is refused, because the
  host writes packed rows. `hidden` equals the manifest's, and the
  compiled batch and seq equal `fixed_seq` and `fixed_batch`, which
  the manifest requires. The output precision is the session's
  compute dtype: FP16 is `DTYPE_F16` and FP32 is `DTYPE_F32`, and a
  manifest `compute_dtype` that names the other is refused.
  `turbo_session_get_info` reports that dtype. The NPU compiles static
  shapes: export the IR with the shape fixed and say it in the
  manifest, so the core caps sessions at it; a dynamic IR fails in the
  driver's compiler with its own message.
- **INPUT_EMBEDDINGS.** The host gathers each token's row from the
  `host_weights` artifact's `word_embeddings` table, F32
  `[vocab, hidden]`, and writes it as the graph's `word_rows` input
  `[batch, seq, hidden]`. Past the written length the frame repeats
  the first id. The mask becomes `attn_bias`
  `[batch, heads, seq, seq]`: 0 for a key the mask keeps and -100 for
  one it drops, the same value for every head and every query. Those
  are the input names the Hailo cut uses (`bundle/hailo/hef_compile.py`).
  A graph whose input is named anything else is refused, naming it.
  The device layout is CHW for the rows and NCHW for the bias, the
  packed layouts the build flags name. The host writes FP32 or FP16,
  whichever precision the compiler kept for that argument, and refuses
  any other. A token type other than 0 is
  `TURBO_E_UNSUPPORTED_OPTION`, naming the row and position, and the
  graph is not run: the cut folds type 0 in as a constant. Lookup is a
  host stage. The token-id path is unchanged.
- **The container is an implicit contract.** The graph extension's
  header defines NGRAPH_LITE's enum value and nothing about the
  buffer's layout: the bytes above are the contract between OpenVINO's
  NPU plugin (`serializeIR` in its compiler adapter) and the compiler
  in the driver, read from the plugin's source and pinned byte for
  byte in this backend's unit tests. The leading compiler version is
  what couples it: the compiler reads the buffer it is handed against
  its own version, which this backend takes from the same device probe
  the plugin uses. The accepted risk is that Intel changes the layout
  in a future compiler major version; the serializer side has been
  stable across compiler majors 4 through 7, a change would surface as
  a compile refusal carrying the compiler's own log (never a silent
  wrong answer), and the fix would be versioned here the way the
  plugin versions it.
- **Sessions.** Every byte a run touches is allocated when the session
  is made: one host buffer per graph input and one for the hidden
  states, a frame (`fixed_batch` rows of the compiled seq) each; the
  result vectors; and the session's copy of the rows. `embed_write`
  keeps the rows. A row whose token type is not 0 is
  `TURBO_E_UNSUPPORTED_OPTION`, naming the row and position, when the
  graph takes no token type input or when the artifact is
  `INPUT_EMBEDDINGS`. A row with no live token, which the core never sends,
  is `TURBO_E_INVALID_ARGUMENT` rather than a NaN from pooling over
  nothing. A run binds the session's buffers to the graph's arguments
  (argument values live on the graph, so runs on one model take turns),
  then per frame writes the inputs. A token-id graph writes each row's
  session tokens in the argument's own precision and leaves the rest of
  the compiled row zero. An embeddings graph writes the gathered word
  rows and the bias, and leaves unused rows zero. The execute is the
  whole compiled frame, including those zeros. `session_create_tuned`
  reports the format that frame was initialized as (`NGRAPH_LITE` or
  `NATIVE`) in `turbo_session_info.choices`. It does not measure
  kernels, and it does not read a cached choice or an environment
  variable: the driver's bit selects the format. The run
  appends `pfnAppendGraphExecute` on the
  context's immediate list, synchronizes, and pools each row's hidden
  states on the host (mean over the mask, the first token, or the last
  live one, summed in F64), cuts to `output_dim`, and normalizes when
  asked.
- **What a result reports.** Stages: upload, encode and download on
  the device, pooling and normalize on the host. Lookup is on the
  device for a token-id graph and on the host for an embeddings graph,
  where the host does the gather. The vectors are `TURBO_PLACE_HOST`.
  `h2d_bytes` is the frames' input buffers (the gathered rows and the
  bias, for an embeddings graph), `d2h_bytes` the frames' hidden
  states. `host_allocs` and
  `device_allocs` are 0: the backend allocates nothing in a run. What
  the driver moves inside an execute is its own and is not counted.

## Bundles

A bundle runs on this backend when it carries a `FORMAT_OPENVINO_IR`
artifact whose `backends` lists `npu`: two files, the xml then its
weights, `compute_dtype` as the conversion fixed it, `graph_input`
`INPUT_TOKEN_IDS`, `graph_output` `OUTPUT_HIDDEN_STATES`, and the shape
compiled in as `fixed_seq` and `fixed_batch`. The MiniLM recipe's
`openvino-f16` artifact is that file: `backends: ["npu"]`, `fixed_seq`
128, `fixed_batch` 1, converted from `onnx-f32`. The driver's compiler,
not OpenVINO on the machine, builds it for the device at
`turbo_model_load`. `target` may name an `arch` label to pin an IR to
one NPU generation, or stay empty for any.

OpenVINO is used only while the bundle is made. The reference image
(`bundle/reference`, `openvino==2026.3.0`) runs
`onnx_to_openvino_ir.py`: `openvino.convert_model` on the F32 export,
a reshape of every rank-2 input to `[fixed_batch, fixed_seq]`, then
`openvino.save_model` with `compress_to_fp16=True`. That pass rewrites
Constant weights. It does not change the Result, so an IR saved that
way has a FP32 hidden-state port. The build flags copy that port into
`--outputs_precisions`, and the driver's compiler reports the output
argument at that precision. `model_load` then refuses a manifest
`compute_dtype` of `DTYPE_F16` with "the graph computes in DTYPE_F32".
That check stays. The script sets each output tensor to f16 with
`PrePostProcessor` before saving, which is the step OpenVINO's NPU
compile tool takes for an FP16 output (`-op FP16`). The saved Result
port is FP16, the flag names that Result (`last_hidden_state:FP16`),
and the script exits if a Result is anything else. The ids stay integer. That is the same weight
compression `ovc` performs
(`ovc onnx/model.onnx --input "input_ids[1,128],attention_mask[1,128],token_type_ids[1,128]" --compress_to_fp16=True --output_model openvino/model.xml`),
plus the output conversion and the opset lowering. The script is what
the tool runs, because it also writes the `produced_by` report. The
library never links OpenVINO.

`convert_model` fuses attention into `ScaledDotProductAttention`, an
opset 13 layer. The Arrow Lake driver reports
`maxOVOpsetVersionSupported` 11, and `model_load` refuses an IR whose
highest layer opset is above that. The script builds the subgraph
OpenVINO's `ScaledDotProductAttentionDecomposition` pass builds (the
Python package does not wrap that pass): MatMul, scale, mask, softmax,
MatMul, ops at opset 8 or below. After `save_model` it reads every
layer `version="opsetN"` the backend reads, and exits if any is above
the cap, naming the type. The cap is `--max-opset` (default 11). It
does not rewrite a layer's version attribute. The MiniLM
recipe's `reference.produced_by.container` is the image the seal
recorded:

`turbo-reference@sha256:994f2d1b0c61669a2cf3c3d89b1b6d58fe9b82b7560602801e0da69d83b5786a`

That string is `turbo-reference@` plus `docker image inspect --format "{{.Id}}" turbo-reference`
after `docker build -t turbo-reference bundle/reference` on the
machine that sealed the MiniLM bundle. The image contains
`onnx_to_openvino_ir.py` and OpenVINO 2026.3.0. A machine converts
when that id is present. The MiniLM pin stays `994f2d1b`.

On main, the MiniLM recipe and `bge-small-en-v1.5`,
`bge-base-en-v1.5`, and `bge-large-en-v1.5` pin
`turbo-reference@sha256:c3ceae1e238e7ec3ac6ece401c1138727d4e80977bd63a1ba8ec64841ffa2934`.
`bge-m3` and `bge-m3-8192` pin
`turbo-reference@sha256:0e153c1db87c62f55ff6f918c7d8a0120c3d2e5d45d3f5a0336e81153c12dd6b`.
Those bge pins are unchanged here. Only the bge-m3 recipes pin
`0e153c1d`. Main's `bundle/reference` Dockerfile does not copy
`onnx_to_openvino_ir.py` and does not install OpenVINO, so those
pins are not this MiniLM image.

From the workspace root, on Linux or on Windows with Docker Desktop
(PowerShell; the same commands):

```
docker build -t turbo-reference bundle/reference
docker image inspect --format "{{.Id}}" turbo-reference
cargo run -p turbo-bundle -- make bundle/recipes/all-minilm-l6-v2.json <upstream-dir> <bundle-dir>
```

`<upstream-dir>` is a checkout of the model's repository at the commit
the recipe names (the tool fetches when it can; a directory already
holding those files is the offline path, bundle/README.md). The convert
step writes `openvino/model.xml` and `openvino/model.bin`, and the
embeddings cut writes `openvino/embeddings.xml` and
`openvino/embeddings.bin`. `make` then seals every conversion the
recipe names. The Hailo image
(`turbo-hailo-dfc@sha256:0972b97df2cfba9ba20abf9efa99a7f57ec675a00e6712a610f6cbd410a575b2`)
is built locally, and `bundle/hailo/Dockerfile` needs a wheel this
repository does not carry. That image was absent for the make that
sealed the other MiniLM artifacts, so the HEF was left out. That
make ran in `994f2d1b`, and `turbo-bundle verify` passed. The sealed
artifacts are `weights-f32`, `onnx-f32`, `onnx-f16`, `openvino-f16`,
and `openvino-embeddings-f16`. The reference and each conversion
record the pin above. `openvino-f16` has `produced_by.reproducible`
true. `openvino-embeddings-f16` records `cut_max_abs_diff` 0.00451
and `produced_by.reproducible` false: the two runs differed. That
seal is the bundle made here. It is not the hardware session. The
Arrow Lake session is recorded under Hardware. The weights, the
ONNX files, and both IRs are not in git.

A machine without Docker does not run the script on the host. Make
the bundle on a Linux machine that has the pinned image, and copy the
sealed bundle over. `seal` assembles a manifest from files a container
already wrote. Each conversion report names that image. A report with
no container, or with `container` `host`, is refused. `seal` opens
`<file>.report.json` first. For the token-id IR that name is
`model.xml.report.json` in the openvino directory. `report.json` in
that file's directory is used when the named report is absent. It does not leave the report in the bundle. A conversion
whose files are absent is omitted. A partial IR (the xml without the
weights, or the files without the report) is refused.

```
cargo run -p turbo-bundle -- seal bundle/recipes/all-minilm-l6-v2.json <upstream> <bundle>
```

`<bundle>` must not already hold a `manifest.json`. `seal` copies the
upstream files the remaining artifacts name, checks each recipe-local
file it still carries against its pin, fills `files` with sizes and
SHA-256, and verifies through the core.

### The embeddings cut

`openvino-embeddings-f16` is a second IR, `INPUT_EMBEDDINGS`,
`host_weights` `weights-f32`, the same frame (`fixed_seq` 128,
`fixed_batch` 1). The script cuts the export, checks the ONNX cut
against it, lowers the graph, saves the IR, and checks that saved IR
again. The container runs the same script with `--cut embeddings`. The
report `seal` opens for this artifact is:
`openvino\embeddings.xml.report.json`. It is not `openvino\report.json`.
That file is the token-id IR's report, and a seal of both files that
bound it here would be refused. The embeddings report must contain
`cut=embeddings` and `cut_max_abs_diff`. The token-id report must
contain neither.

The rows are the Hailo calibration texts plus one synthetic row with a
mask hole. The ONNX cut, before lowering, must be within 1e-4 of the
export on a kept position. After `save_model` (`compress_to_fp16`,
FP16 Results, SDPA lowered to opset 11 or below) the saved IR is
compared the same way. `cut_max_abs_diff` is that second difference.
FP16 compression is not the 1e-4 ONNX tolerance. The saved IR must be
within 0.02. The MiniLM cut measured 0.00451 on these texts. Both
checks compile with `INFERENCE_PRECISION_HINT`
`f32`. The CPU plugin's default hint lowers the fused attention, and
the static cut then disagrees with the export by about 1e-2 on a kept
position. The script exits if either check is over its tolerance, if a
layer opset is above the cap, or if a Result is not FP16. The gathered
inputs stay FP32.

The recipe lists `openvino-f16` before `openvino-embeddings-f16`, and
both name `npu`. The loader takes the first artifact it can run, so a
bundle that contains both still loads the token-id IR. To run the
gather, seal a copy of the recipe that drops the `openvino-f16`
artifact (or lists the embeddings artifact first and keeps both files).
`seal` copies `weights/model.safetensors`, which is the host table.
Point `TURBO_TEST_BUNDLE` at that directory.

Hardware records that seal. Commit `5717dc5` records the
embeddings-sealed MiniLM on intel-npu: the npu tests passed 10,
including `a_run_reports_its_frames_and_where_each_stage_ran` with
lookup on the host, and conformance passed 3. Commit `6d904e7` records
that the IR compiled on AI Boost (`cut_max_abs_diff` 0.00451,
`turbo-bundle verify` exited 0). Verify and conformance passed. The
LOOKUP stage re-run is the receipt in `5717dc5`.

## Windows notes

The machine this backend is for first (an Arrow Lake laptop with Intel
AI Boost) runs Windows: the loader is `ze_loader.dll`, installed with
Intel's driver package, and the AI Boost driver advertises
`ZE_extension_graph` 1.17 there; the device lists as
`Intel(R) AI Boost`, arch `arl-npu`. Nothing else differs: the same
tests, the same environment variables,
`cargo test -p turbo --features npu` from a shell whose PATH finds
`ze_loader.dll`. A machine where Device Manager shows "Intel(R) AI
Boost" but the backend lists nothing has a loader or NPU driver
problem, and `TURBO_TEST_REQUIRE_NPU=1` makes the tests say so instead
of passing.

The sealed test bundles under `testdata/` are verified by size and
SHA-256 on every load, so their bytes must come out of a checkout
exactly as committed: `.gitattributes` marks them `-text`, which keeps
`core.autocrlf` from rewriting their line endings. A Windows clone
made before that file existed holds rewritten copies (tokenizer.json
reads 742346 bytes against the manifest's 711661); refresh them once
with `git rm -r --cached testdata && git checkout -- testdata`, or
reclone.

The same rewrite breaks `make` and `seal` on the recipe's calibration
file. `bundle/recipes/*.jsonl` is marked `-text` for the same reason:
`stage` hashes those bytes, and the MiniLM pin
`bde646a9a523e6dbb93315343bd363890ef31510625aee4a2e8c7a6fda65c3b4` is
the LF file in the commit. A checkout that turned the line endings into
CR LF hashes to
`9e6ac4eb099b4934d86adb2af92fdc3d5817b84ecd9d7bf1445d1b832d37d940` and
is refused. A clone made before that attribute existed holds the
rewritten copy; refresh it once, from the repo root:

```
git rm --cached bundle/recipes/all-minilm-l6-v2.calibration.jsonl
git checkout -- bundle/recipes/all-minilm-l6-v2.calibration.jsonl
```

or reclone.

## Testing on an NPU machine

From the workspace root. `TURBO_TEST_REQUIRE_NPU=1` makes every test in
`core/tests/npu.rs` that needs a device fail when the backend lists
none, where without it the test passes with a line saying it was
skipped; set it on an NPU machine, so a run that found no device cannot
pass.

```
export TURBO_TEST_REQUIRE_NPU=1

# Everything, with the NPU-only tests in core/tests/npu.rs:
cargo test -p turbo --features npu

# The NPU-only tests, with a bundle that has an OpenVINO IR whose
# backends lists npu, and conformance on it (docs/conformance.md):
TURBO_TEST_BUNDLE=<bundle-dir> \
    cargo test -p turbo --features npu --test npu -- --include-ignored
TURBO_TEST_BUNDLE=<bundle-dir> TURBO_TEST_DEVICE=npu \
    cargo test -p turbo --features npu --test conformance -- --include-ignored
```

`a_run_reports_its_frames_and_where_each_stage_ran` reads the loaded
artifact. Lookup is `TURBO_STAGE_HOST` when that artifact is
`INPUT_EMBEDDINGS` and `TURBO_STAGE_DEVICE` when it is
`INPUT_TOKEN_IDS`. Upload, encode and download stay on the device.
Pooling and normalize stay on the host. The loader takes the first
`npu` artifact, so a bundle that lists `openvino-f16` first still
reports lookup on the device. A seal that drops that artifact, or
lists `openvino-embeddings-f16` first, reports lookup on the host.

## Host-only blocker

GitHub Actions does not run this backend on an NPU. The `test (npu)`
step in `.github/workflows/ci.yml` is `cargo test -p turbo --features npu`
on `ubuntu-24.04`. That machine has no Intel NPU driver and no
`/dev/accel` node. The job does not pass `--include-ignored` and does
not set `TURBO_TEST_REQUIRE_NPU`.

What that job does run: the library tests that need no device (the IR
container, the build flags, the host gather, and the native descriptor)
and the integration tests that return when the backend lists nothing.
A missing device is a skip line, not a measured session.

What it does not run: the four tests in `core/tests/npu.rs` marked
`needs an Intel NPU and TURBO_TEST_BUNDLE with an OpenVINO IR for it`.
Those stay ignored. Conformance with `TURBO_TEST_DEVICE=npu` is the
ignored test in `core/tests/conformance.rs`, and this job does not
include it. `turbo-bench` is not pointed at an NPU in that job either.

There is no self-hosted NPU runner in this workflow. A green `test (npu)`
on `ubuntu-24.04` is not a device run, and it is not coverage of the graph
the driver compiles.

## Hardware

Commit `d1007c3`, on Windows Arrow Lake. The device lists as
`Intel(R) AI Boost`, arch `arl-npu`, loader 1.28.2, driver
`0.15.21738`. An earlier listing on that machine advertised graph
extension 1.17. The bundle that ran was a host seal of the MiniLM
`openvino-f16` IR (`container` `host`, `reproducible` false). That
host path is gone: a conversion report must name the pinned image,
and the copied reference under `bundle/reference/out/` is not in
this tree.

```
set TURBO_TEST_REQUIRE_NPU=1
set TURBO_TEST_BUNDLE=<bundle>
cargo test -p turbo --features npu --test npu -- --include-ignored
```

10 passed.

`TURBO_TEST_BUNDLE` stays set from the command above.

```
set TURBO_TEST_DEVICE=npu
cargo test -p turbo --features npu --test conformance -- --include-ignored
```

3 passed. One of the three is the `FORMAT_SAFETENSORS` case, which this
backend skips. The numerical proof is the two tests that match the
reference.

Commit `5717dc5` records an embeddings-sealed MiniLM on intel-npu.
`cargo test -p turbo --features npu --test npu -- --include-ignored`
with `TURBO_TEST_REQUIRE_NPU=1` passed 10, and none failed.
`a_run_reports_its_frames_and_where_each_stage_ran` passed: lookup was
on the host. Conformance with `TURBO_TEST_DEVICE=npu` passed 3. The
tokenizer file hashed to
`be50c3628f2bf5bb5e3a7f17b1f74611b2561a3a27eeab05e5aa30f411572037`.
That run is the `ZE_GRAPH_FORMAT_NGRAPH_LITE` graph. A native load
has not been run on intel-npu.

Commit `6d904e7` records that `openvino-embeddings-f16` compiled on
AI Boost (`cut_max_abs_diff` 0.00451, `turbo-bundle verify` exited 0).

Commit `383a7d1` records the same intel-npu machine again. Arrow Lake, the
device listed as `Intel(R) AI Boost`, driver `0.15.21738`.
`TURBO_TEST_REQUIRE_NPU=1` and
`cargo test -p turbo --features npu --test npu -- --include-ignored`
passed 10, and none failed. The driver reported
`graphFormatsSupported` `0x2`, which is `ZE_GRAPH_FORMAT_NGRAPH_LITE`.
Bit `0x1`, `ZE_GRAPH_FORMAT_NATIVE`, was not set. The load initialized
the lite graph. That is the fallback `model_load` takes when the device
does not advertise `NATIVE`. The run did not create a graph with
`ZE_GRAPH_FORMAT_NATIVE`. The code that would, when a driver advertises
the bit, is in this tree and was not taken on that run.

Two ROWS_DENSE npu speed records are committed at `00aa734`,
measured at library commit `febfed530`. They are MiniLM
(`all-MiniLM-L6-v2`) `INPUT_TOKEN_IDS` F16, batch 1, seq 128, case
8, on Windows Arrow Lake. The device is `Intel(R) AI Boost`, arch
`arl-npu`. Settings are `TURBO_NPU_GRAPH_FORMAT=NGRAPH_LITE` and
`TURBO_NPU_GRAPH_INPUT=INPUT_TOKEN_IDS`. The files are
`benchmarks/records/arl-npu.npu.ngraph-lite.embed.model-dense.all-minilm-l6-v2-da08a0f9.febfed530b66.json`
and
`benchmarks/records/arl-npu.npu.ngraph-lite.embed.fastest-dense.all-minilm-l6-v2-da08a0f9.febfed530b66.json`.
`rows.kind` is `ROWS_DENSE`. `Record::is_for` requires
`ROWS_MIXED`, so these files are reference evidence and do not bind
a cell.

Two ROWS_MIXED records of the same MiniLM IR are committed at
`bc15354`, measured at library commit `99c9264`. Batch 1, seq 128,
cases 0 through 7, `live_tokens` 181. `machine.os` is windows.
Settings are the same `NGRAPH_LITE` and `INPUT_TOKEN_IDS`. The
device driver in both files is Level Zero `0.15.21738`. The files
are
`benchmarks/records/arl-npu.npu.ngraph-lite.embed.model.all-minilm-l6-v2-da08a0f9.99c92648aa9a.json`
and
`benchmarks/records/arl-npu.npu.ngraph-lite.embed.fastest.all-minilm-l6-v2-da08a0f9.99c92648aa9a.json`.
OpenVINO was measured on those eight frames. Library version is
`0.1.0` and `compute_dtype` is `DTYPE_F16`. `conformance.rows` is
16, so eight reference cases fit seq 128 and `rows.cases` lists
each of them. `is_for` matches and `falls_short` is none.
`decide_embedded` sets `MODEL` and `FASTEST` to SUPPORTED on a
Windows build of library `0.1.0` whose device arch is `arl-npu` and
whose graph format is `NGRAPH_LITE`. The numbers are in Speed,
below. `EXACT` stays UNSUPPORTED. Another arch and a `NATIVE` load
stay EXPERIMENTAL. The product path is the Level Zero graph
extension and `FORMAT_OPENVINO_IR`. `benchmark_app` is the reference
control on the static IR (`openvino/model.xml`).

Earlier intel-npu timings were one two-token text at shape
`[1, 128]` each, TEI was `not_run`, and `benchmark_app` either
exited 1 or was disabled. Those files are not in this tree. A
number with no comparison is not a record here. The output-precision
check is unchanged: FP16 is `DTYPE_F16`, FP32 is `DTYPE_F32`, and a
manifest `compute_dtype` that names the other is refused.

## Speed

Four npu speed records are committed. The harness is `turbo-bench`
with `--features npu` (docs/benchmarks.md). It measures the library
on `--device npu`. OpenVINO `benchmark_app` is the kernel reference.
TEI's CPU image is the end-to-end baseline. Both are references only.
The product path is the Level Zero graph extension and
`FORMAT_OPENVINO_IR`. The library does not run ONNX and it does not
link OpenVINO. The measured machine is intel-npu (Windows Arrow Lake,
device `Intel(R) AI Boost`, arch `arl-npu`). TEI was `--no-tei`. The
OpenVINO side is the static IR (`-m openvino/model.xml`).

The two ROWS_MIXED records, committed at `bc15354` and measured at
library commit `99c9264`, are batch 1, seq 128, cases 0 through 7,
`live_tokens` 181, `computed_tokens` 1024. They bind the `arl-npu`
windows `NGRAPH_LITE` `MODEL` and `FASTEST` cells. The two
ROWS_DENSE records, committed at `00aa734` and measured at library
commit `febfed530`, are case 8 at the same shape. They stay in the
tree as reference evidence. `Record::is_for` requires `ROWS_MIXED`,
so the dense files do not bind a cell.

The model is `sentence-transformers/all-MiniLM-L6-v2`, the one the
other optimized paths record. The published shape on the token-id seal
(`openvino-f16`) is `--batch 1 --seq 128`. That artifact's
`fixed_batch` is 1 and its `fixed_seq` is 128. One library frame is
one `benchmark_app` request of `[1, 128]`. The tool refuses any other
shape for an npu load, including the shape it would pick when
`--batch` and `--seq` are omitted. Mixed rows on that frame cycle
every reference case that fits 128, each padded into its own
`[1, 128]` request. OpenVINO times the same cases, one request each.
The two ROWS_MIXED files above are that pass: cases 0 through 7,
`live_tokens` 181, eight frames. The library p50 is the sum of those
frames, about 30 ms. A single `[1, 128]` frame of case 0 was about
3.7 ms. Those thin files were withdrawn and are not in the tree.

`PRECISION_MODEL` and `PRECISION_FASTEST` are two cells at that same
shape. `--precision model` on mixed rows is filed as
`arl-npu.npu.ngraph-lite.embed.model.`. `--precision fastest` on
mixed rows is filed as `arl-npu.npu.ngraph-lite.embed.fastest.`.
The same precisions with `--rows dense` add `-dense` after the
precision. On this IR both sessions compute in the compiled F16
graph. A record of one precision does not fill the other. `EXACT`
is `UNSUPPORTED` and is not a speed cell.

`timing.computed_tokens` for a single frame of this cell is 128: the
device executes the compiled frame, and the zeros after the live
tokens are part of that frame. The count is compiled frames times
`fixed_seq`. The mixed files committed below run every reference
case that fits 128, each in its own `[1, 128]` frame, so the count
is 8 times 128, which is 1024. `--batch` stays 1.

`library.settings` names `TURBO_NPU_GRAPH_FORMAT` and
`TURBO_NPU_GRAPH_INPUT`. On driver `0.15.21738` the format is
`NGRAPH_LITE`, because `graphFormatsSupported` is `0x2` and bit `0x1`
is clear. The file name carries `ngraph-lite` after `npu`. A driver
that advertises bit `0x1` loads `NATIVE`, writes
`TURBO_NPU_GRAPH_FORMAT=NATIVE`, and is filed as `native`. That
record is a different cell. The format is what the device selected.
Setting an environment variable does not change it.

An embeddings-sealed copy (`openvino-embeddings-f16`,
`INPUT_EMBEDDINGS`) is the host-gather path. `benchmark_app -d NPU`
compiles the static token-id IR, which is not that cut. The tool
refuses a `speed_ratio` for the embeddings artifact. A library-only
record is `--no-tei` and `--no-openvino` together, and its
`speed_ratio` stays null.

Labels. The machine in a published record is the device arch, `arl-npu`
on this part, and the device name the driver reports. Do not put a
hostname in the record; the tool rewrites host paths, and a home
directory is refused. Write-ups name the part `intel-npu` or
`arl-npu`, or Machine A, B, or C when more than one host is in view.

On Linux the container is given `/dev/accel/accel0` unless
`--openvino-accel` names another node. Several NPUs are refused,
because `-d NPU` is OpenVINO's first device and the command has no
per-device index.

`benchmark_app -d NPU` is given `openvino/model.xml` (the bin sits
beside it), not `onnx/model.onnx`. The prior blocker on intel-npu
was a compile failure: the report said IR serialized API found 8.1,
expected 8.2, and NPU-VCL returned
`ZE_RESULT_ERROR_INVALID_NULL_POINTER`. The IR file was not the
versioned party. The OpenVINO NPU plugin re-serializes the graph for
the driver, and the library already compiles this same IR through
the Level Zero graph extension. That mismatch was the OpenVINO
package and the NPU driver compiler out of step. It is resolved for
the measured cells. They were timed with OpenVINO nightly 2026.5.0
(`2026.5.0-23311-786052d995f`). The binding driver string is the
Level Zero version the records report, `0.15.21738`.
`benchmark_app` compiled the static IR (`-m openvino/model.xml`)
and wrote a kernel measurement. Passing the dynamic ONNX file
instead fails earlier, at `core.cpp:120`, and that file is not the
NPU reference input.

The full-case-set files, both `NGRAPH_LITE` and `INPUT_TOKEN_IDS`,
`DTYPE_F16`, `ROWS_MIXED`, library version `0.1.0`, `machine.os`
windows, driver `0.15.21738`. `conformance.min_cosine` is
0.999996097954919, above the F16 floor of 0.999, and OpenVINO was
measured on the same eight frames. `rows.cases` is
`[0, 1, 2, 3, 4, 5, 6, 7]` and `live_tokens` is 181.
`conformance.rows` is 16, so they cover 8 of 8 reference cases that
fit seq. `falls_short` is none. `decide_embedded` sets the `arl-npu`
windows `NGRAPH_LITE` `MODEL` and `FASTEST` cells to SUPPORTED.
The files, committed at `bc15354` and measured at library commit
`99c9264`:

- `benchmarks/records/arl-npu.npu.ngraph-lite.embed.model.all-minilm-l6-v2-da08a0f9.99c92648aa9a.json`:
  library p50 30.2244 ms, 261.2454192860806 rows/s; OpenVINO IR p50
  30.22 ms; `speed_ratio` 1.0001455989410986.
- `benchmarks/records/arl-npu.npu.ngraph-lite.embed.fastest.all-minilm-l6-v2-da08a0f9.99c92648aa9a.json`:
  library p50 29.9935 ms, 262.28463787892775 rows/s; OpenVINO IR p50
  30.259999999999998 ms; `speed_ratio` 0.9911929940515533.

On a Windows build of library `0.1.0` whose device arch is `arl-npu`
and whose graph format is `NGRAPH_LITE`, `MODEL` and `FASTEST` are
SUPPORTED. `turbo_capability.benchmark` is that file name.
`cosine_floor` is 0.999996097954919. `reason` is empty. Another
arch, a `NATIVE` graph, and a build whose OS is not windows stay
EXPERIMENTAL with `no benchmark record for this cell`.

The dense files are the same settings and the same static IR, case
8. They do not bind:

- `benchmarks/records/arl-npu.npu.ngraph-lite.embed.model-dense.all-minilm-l6-v2-da08a0f9.febfed530b66.json`:
  library p50 3.9668 ms, about 252.93 rows/s; OpenVINO IR p50 3.82 ms;
  `speed_ratio` about 1.038.
- `benchmarks/records/arl-npu.npu.ngraph-lite.embed.fastest-dense.all-minilm-l6-v2-da08a0f9.febfed530b66.json`:
  library p50 3.97 ms, about 251.27 rows/s; OpenVINO IR p50 3.87 ms;
  `speed_ratio` about 1.026.
