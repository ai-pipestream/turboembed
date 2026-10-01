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
- **Capability.** Embed is EXPERIMENTAL at `MODEL` and `FASTEST`,
  honoring `normalize`, `pooling` and `output_dim` (the core owns
  `truncate`, `max_tokens` and `prompt_role`, and sets their bits
  itself, so `turbo_capability.options_honored` reads 0b111111). The
  reported dtype is 0: no dtype is claimed before an artifact is seen,
  because the IR's compilation fixes it; `model_load` reads it from
  the compiled graph and `turbo_session_get_info` says what a session
  really resolved. A benchmark record names a dtype, so the 0 also
  backs no SUPPORTED claim. `EXACT` is UNSUPPORTED: a compiled graph
  computes in the dtype its IR fixed, never F32 throughout. The
  backend never claims SUPPORTED; only the core says that, and only
  over a benchmark record.
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
  each input's and output's precision and layout by index, exactly as
  OpenVINO's own NPU plugin serializes them. What those flags need, and
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
  formats bitfield. That is the path the `57c302f` receipt ran: on
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
  live tokens in the argument's own precision with zeros after. An
  embeddings graph writes the gathered word rows and the bias. The run
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
port is FP16, the flag is `0:FP16`, and the script exits if a Result
is anything else. The ids stay integer. That is the same weight
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
recipe's `reference.produced_by.container` is the image built from
`bundle/reference` on this tree, which contains
`onnx_to_openvino_ir.py`:

`turbo-reference@sha256:994f2d1b0c61669a2cf3c3d89b1b6d58fe9b82b7560602801e0da69d83b5786a`

That string is `turbo-reference@` plus `docker image inspect --format "{{.Id}}" turbo-reference`
after `docker build -t turbo-reference bundle/reference`. A later build
gets a new id, and the pin moves with it. The bge recipes still name
an older image, which does not contain this script.

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
step writes `openvino/model.xml` and `openvino/model.bin`. `make` then
seals every conversion the recipe names. The Hailo image
(`turbo-hailo-dfc@sha256:0972b97df2cfba9ba20abf9efa99a7f57ec675a00e6712a610f6cbd410a575b2`)
is built locally, and `bundle/hailo/Dockerfile` needs a wheel this
repository does not carry, so `make` stops after the reference, the F16
ONNX, and the IR are written, and it does not write `manifest.json`.
`turbo-bundle seal` on those files leaves the HEF out. A seal of that
kind verified here: artifacts `weights-f32`, `onnx-f32`, `onnx-f16`,
and `openvino-f16`, the reference and both conversions recorded against
the pin above, and the IR `produced_by.reproducible` true (two runs,
identical bytes). That Linux seal is not the hardware session. The
Arrow Lake session is recorded under Hardware. The weights, the
ONNX files, and the IR are not in git.

On the Arrow Lake machine, where OpenVINO 2026.3 is already installed
and Docker Desktop is not, the host script produces the same two IR
files, and `turbo-bundle seal` writes the manifest from them without
running docker.

With that install's `python` on `PATH` (`py -3.12` is the usual
launcher), after `fetch` has put the export in `<upstream>`:

```
py -3.12 -c "import openvino; print(openvino.get_version())"
mkdir <bundle>\openvino
py -3.12 <repo>\bundle\reference\onnx_to_openvino_ir.py <upstream>\onnx\model.onnx <bundle>\openvino\model.xml <bundle>\openvino\model.bin <bundle>\openvino\report.json --seq 128 --batch 1 --max-opset 11
```

The first line must print a 2026.3 version. The script refuses an input
whose rank is not 2, a layer opset above `--max-opset`, and a Result
port that is not FP16. Omitting `--max-opset` is the same cap, 11.
`openvino\report.json` is the token-id IR's `produced_by` report.
`seal` opens `<file>.report.json` first (`openvino\model.xml.report.json`),
then `report.json` in that file's directory when the named report is
absent. It does not leave the report in the bundle. The report names
no container, so the manifest records `container` `host` and
`reproducible` false: the script ran once on the machine, not twice in
the pinned image.

The loader still checks the reference file, and `seal` does not run the
container that writes it. The file this pin sealed, and the report, are
in the repo:

```
bundle/reference/out/all-minilm-l6-v2/reference.safetensors
bundle/reference/out/all-minilm-l6-v2/report.json
```

`reference.safetensors` is 23292 bytes, SHA-256
`b773a9f83f5017e5bbcb26ffd9a5087d5def4aa68e8c199598869231398b795b`.
`report.json` is that sealed bundle's `reference.produced_by`, and its
`container` is the pin above. Copy them to
`<bundle>\reference\reference.safetensors` and
`<bundle>\reference\report.json`. Keep the host-produced
`openvino\model.xml`, `openvino\model.bin`, and `openvino\report.json`.
`seal` does not fill the container in. The F16 ONNX file, the HEF,
and `openvino-embeddings-f16` are left out of the manifest when their
files are not in the bundle; they are not claimed as made. The token-id
IR is not optional: a seal without both of its files is refused. A seal that copied these two files and a
host IR report verified here: the IR is recorded as `container` `host`
and `reproducible` false, and the reference keeps the pin. Then, from
the repo root:

```
cargo run -p turbo-bundle -- seal bundle/recipes/all-minilm-l6-v2.json <upstream> <bundle>
```

`<bundle>` must not already hold a `manifest.json`. `seal` copies the
upstream files the remaining artifacts name, checks each recipe-local
file it still carries against its pin, fills `files` with sizes and
SHA-256, and verifies through the core. A partial IR (the xml without
the weights, or the files without the report `seal` opens) is refused.

### The embeddings cut

`openvino-embeddings-f16` is a second IR, `INPUT_EMBEDDINGS`,
`host_weights` `weights-f32`, the same frame (`fixed_seq` 128,
`fixed_batch` 1). The script cuts the export, checks the ONNX cut
against it, lowers the graph, saves the IR, and checks that saved IR
again. On the same OpenVINO 2026.3 install:

```
py -3.12 <repo>\bundle\reference\onnx_to_openvino_ir.py <upstream>\onnx\model.onnx <bundle>\openvino\embeddings.xml <bundle>\openvino\embeddings.bin <bundle>\openvino\embeddings.xml.report.json --seq 128 --batch 1 --max-opset 11 --cut embeddings --heads 12 --tokenizer <upstream>\tokenizer.json --calibration <repo>\bundle\recipes\all-minilm-l6-v2.calibration.jsonl
```

The fourth path is the report `seal` opens for this artifact:
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

Hardware records that seal. At tip `7cd162a` the embeddings-sealed
MiniLM passed the npu tests 10, including
`a_run_reports_its_frames_and_where_each_stage_ran` with lookup on the
host, and conformance passed 3. Tip `93f0d76` is where the IR compiled
(`cut_max_abs_diff` 0.00451, `turbo-bundle verify` exited 0) and the
npu tests passed 9, that stage assertion being the one that failed.

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

## Hardware

Commit `90d7138`, on Windows Arrow Lake. The device lists as
`Intel(R) AI Boost`, arch `arl-npu`, loader 1.28.2, driver
`0.15.21738`. An earlier listing on that machine advertised graph
extension 1.17. The bundle was a host seal of the MiniLM
`openvino-f16` IR: `container` `host`, `reproducible` false. The
reference file copied into that seal is
`bundle/reference/out/all-minilm-l6-v2/`.

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

An embeddings-sealed MiniLM at tip `7cd162a` ran on intel-npu.
`cargo test -p turbo --features npu --test npu -- --include-ignored`
with `TURBO_TEST_REQUIRE_NPU=1` passed 10, and none failed.
`a_run_reports_its_frames_and_where_each_stage_ran` passed: lookup was
on the host. Conformance with `TURBO_TEST_DEVICE=npu` passed 3. The
tokenizer file hashed to
`be50c3628f2bf5bb5e3a7f17b1f74611b2561a3a27eeab05e5aa30f411572037`.
Tip `93f0d76` is where `openvino-embeddings-f16` compiled on AI Boost
(`cut_max_abs_diff` 0.00451, `turbo-bundle verify` exited 0) and the
npu tests passed 9. The stage assertion was the failure there. The
`7cd162a` receipt is the `ZE_GRAPH_FORMAT_NGRAPH_LITE` graph. The
`NATIVE` create path was not in that commit.

At tip `57c302f` the same intel-npu machine ran again. Arrow Lake, the
device listed as `Intel(R) AI Boost`, driver `0.15.21738`.
`TURBO_TEST_REQUIRE_NPU=1` and
`cargo test -p turbo --features npu --test npu -- --include-ignored`
passed 10, and none failed. The driver reported
`graphFormatsSupported` `0x2`, which is `ZE_GRAPH_FORMAT_NGRAPH_LITE`.
Bit `0x1`, `ZE_GRAPH_FORMAT_NATIVE`, was not set. The load initialized
the lite graph. That is the fallback `model_load` takes when the device
does not advertise `NATIVE`. The run did not create a graph with
`ZE_GRAPH_FORMAT_NATIVE`. The code that would, when a driver advertises
the bit, is in this tree and was not taken. This tree has no NPU.

Capability stays `EXPERIMENTAL` at `MODEL` and `FASTEST`. The dtype
reported before a model is loaded is 0, and that is why a benchmark
record is not `SUPPORTED`. `EXACT` stays `UNSUPPORTED`. The backend
does not claim `SUPPORTED`. The output-precision check is unchanged:
FP16 is `DTYPE_F16`, FP32 is `DTYPE_F32`, and a manifest
`compute_dtype` that names the other is refused.

## Speed

No file under `benchmarks/records/` is an npu record, and none was
written here: this tree has no NPU, so there are no timings to enter.
The harness is `turbo-bench` with `--features npu`
(docs/benchmarks.md). It measures the library on `--device npu` and
the same two references levelzero uses. OpenVINO `benchmark_app` is
the kernel reference. TEI's CPU image is the end-to-end baseline.
Both are references only. The product path stays the Level Zero graph
extension.

The model is `sentence-transformers/all-MiniLM-L6-v2`, the one the
other optimized paths record. The comparable cell is the token-id
seal (`openvino-f16`): `benchmark_app` compiles that bundle's ONNX
encoder, which is the file the IR was converted from. An
embeddings-sealed copy measures the host-gather path. Its OpenVINO
reference is still the full ONNX encoder, which includes the
embedding lookup the cut removed, so that pair is not the same graph.
Leave `--batch` and `--seq` at the tool's defaults. The IR is fixed
at sequence 128 and batch 1, the model's `max_batch` is 64, and the
tool's default batch is 32, so the library runs 32 frames of 1. The
default seq is the longest reference case that fits 128. A GPU record
at sequence 256 is a different shape. `EXACT` is `UNSUPPORTED` and is
not a speed cell.

Labels. The machine in a published record is the device arch, `arl-npu`
on this part, and the device name the driver reports. Do not put a
hostname in the record; the tool rewrites host paths, and a home
directory is refused. Capability stays `EXPERIMENTAL`. The graph
format is not a field of the record. On driver `0.15.21738` a record
is a `NGRAPH_LITE` measurement: `graphFormatsSupported` is `0x2`, and
the debug log says `the graph is NGRAPH_LITE`. Do not label that
record `NATIVE`. A `NATIVE` timing waits on a driver that advertises
bit `0x1`. The same command records it, because `model_load` follows
the bit, and the debug log then says `the graph is the native blob`.

The follow-up is that command on intel-npu, from a clean pushed tree,
for the token-id MiniLM seal at `model` and `fastest`, mixed rows.
`--no-tei` is honest when Docker cannot run TEI's CPU image; the
record says TEI was not run, and the speed cell is against OpenVINO
when `benchmark_app` ran. `--no-openvino` leaves the kernel cell
empty. A disabled reference is recorded as not run. It is not filled
with a time.

## Still to land

- A load on intel-npu whose graph is created with
  `ZE_GRAPH_FORMAT_NATIVE`. The `57c302f` run on driver `0.15.21738`
  reported `graphFormatsSupported` `0x2`. `NATIVE` (bit `0x1`) was not
  advertised, so `model_load` initialized the `NGRAPH_LITE` graph and
  did not create the native one. The code path remains for a driver
  that lists the bit. That create has not been exercised on hardware.
- Speed cells. The harness and the command are in Speed, above. No
  record is committed. Timings wait on a follow-up intel-npu run. A
  record on driver `0.15.21738` is `NGRAPH_LITE`. A `NATIVE` timing
  waits on a driver that advertises bit `0x1`.
- No benchmark record marks this backend `SUPPORTED`. Capability stays
  `EXPERIMENTAL`.
