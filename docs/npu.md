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
  5.9 or newer. Model loading says plainly when either is older;
  current drivers (graph extension 1.13 through 1.20) are well past
  both.

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
  nothing standing in for it. No driver, no NPU device, or no graph
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
- **Models.** A `FORMAT_OPENVINO_IR` artifact from `INPUT_TOKEN_IDS` to
  `OUTPUT_HIDDEN_STATES`; anything else is refused, naming what (the
  host-gather path for `INPUT_EMBEDDINGS` is not built). Before any
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
  `TURBO_E_RUNTIME` message, else `pfnCreate2`; it is then initialized
  (`pfnGraphInitialize`, or appended and synchronized, as its
  properties ask), which is the weights' move to the device. The
  compiled arguments are taken by name, `input_ids`, `attention_mask`
  and optionally `token_type_ids`, the names a BERT export gives; a
  graph whose input is named anything else is refused naming it, and
  no argument is ever assigned a role by position. Among several
  outputs only `last_hidden_state` is taken; a single output is the
  hidden states whatever its name. The compiled boundary is checked:
  ids and mask as I64 or I32 `[batch, seq]`, hidden states back as
  FP32 or FP16 `[batch, seq, hidden]`, `hidden` equal to the
  manifest's, and the compiled shape equal to `fixed_seq` and
  `fixed_batch` where the manifest sets them. The NPU compiles static
  shapes: export the IR with the shape fixed and say it in the
  manifest, so the core caps sessions at it; a dynamic IR fails in the
  driver's compiler with its own message.
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
  keeps the rows; a row whose token type is not 0, on a graph with no
  token type input, is `TURBO_E_UNSUPPORTED_OPTION`, naming the row and
  position, and a row with no live token, which the core never sends,
  is `TURBO_E_INVALID_ARGUMENT` rather than a NaN from pooling over
  nothing. A run binds the session's buffers to the graph's arguments
  (argument values live on the graph, so runs on one model take turns),
  then per frame writes each row's live tokens in the argument's own
  precision with zeros after, appends `pfnAppendGraphExecute` on the
  context's immediate list, synchronizes, and pools each row's hidden
  states on the host (mean over the mask, the first token, or the last
  live one, summed in F64), cuts to `output_dim`, and normalizes when
  asked.
- **What a result reports.** Stages: upload, lookup, encode and
  download on the device, pooling and normalize on the host; the
  vectors are `TURBO_PLACE_HOST`. `h2d_bytes` is the frames' input
  buffers, `d2h_bytes` the frames' hidden states. `host_allocs` and
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
`openvino.save_model` with `compress_to_fp16=True`. That is the same
conversion the `ovc` command line performs
(`ovc onnx/model.onnx --input "input_ids[1,128],attention_mask[1,128],token_type_ids[1,128]" --compress_to_fp16=True --output_model openvino/model.xml`);
the script is what the tool runs, because it also writes the
`produced_by` report. The library never links OpenVINO. The image the
recipe currently pins was built before this script existed, so `make`
refuses the conversion until the image is built again and the pin in
`reference.produced_by.container` is the new id.

From the workspace root, on Linux or on Windows with Docker Desktop
(PowerShell; the same commands):

```
docker build -t turbo-reference bundle/reference
docker image inspect --format "{{.Id}}" turbo-reference
```

The id prints as `sha256:<64 hex>`. Put
`turbo-reference@sha256:<64 hex>` in the recipe's
`reference.produced_by.container`, then:

```
cargo run -p turbo-bundle -- make bundle/recipes/all-minilm-l6-v2.json <upstream-dir> <bundle-dir>
```

`<upstream-dir>` is a checkout of the model's repository at the commit
the recipe names (the tool fetches when it can; a directory already
holding those files is the offline path, bundle/README.md). The convert
step writes `openvino/model.xml` and `openvino/model.bin` and seals
both into `files`.

On the Arrow Lake machine, where OpenVINO 2026.3 is already installed
and Docker is not, the same script produces the two files the seal
step hashes. With that install's `python` on `PATH` (`py -3.12` is the
usual launcher), from a directory that already holds the staged
`onnx/model.onnx` (the bundle directory after `stage`, or the upstream
checkout):

```
py -3.12 -c "import openvino; print(openvino.get_version())"
py -3.12 <repo>\bundle\reference\onnx_to_openvino_ir.py onnx\model.onnx openvino\model.xml openvino\model.bin report.json --seq 128 --batch 1
```

The first line must print a 2026.3 version. The script refuses an input
whose rank is not 2. `openvino\model.xml` and `openvino\model.bin` are
what `make` would have written; `report.json` is the `produced_by`
report and is not a bundle file. Sealing still goes through
`turbo-bundle`, because only that fills `files` with the sizes and
SHA-256 the core checks, and the reference vectors still come from the
pinned container. A hand-copied IR in a bundle whose manifest was
sealed without it will not load.

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

## Still to land

- Execution proof. Device listing is proven on hardware (Arrow Lake
  Windows, `Intel(R) AI Boost`, arch `arl-npu`, loader 1.28.2, graph
  extension 1.17). The compile-and-run path has not run against a
  device: the MiniLM recipe now carries `openvino-f16` for `npu`, and
  the proof is still the ignored tests and the conformance run above,
  on that machine, with `TURBO_TEST_REQUIRE_NPU=1` and
  `TURBO_TEST_BUNDLE` set to a bundle `make` sealed from that recipe.
  Nothing in this tree claims a session has run on the NPU.
- The `INPUT_EMBEDDINGS` host-gather path, should an NPU graph ever be
  cut at the embedding gather the way the Hailo one is.
- `ZE_GRAPH_FORMAT_NATIVE`: loading a driver-precompiled blob, which
  would need its own artifact format in docs/bundle.md, decided when a
  bundle wants to carry one.
