# Benchmark records

A capability cell (`turbo_runtime_capability`) says SUPPORTED only when a
benchmark record backs it. A record is one measurement of the library on
one device, for one task and precision, on one bundle, at one commit,
next to the vendor's fastest programs on the same token rows. Records are
JSON files in `benchmarks/records/`, written by one tool, `turbo-bench`
(`bench/`), and compiled into the library by `core/build.rs`: the
library reads no file at run time. Nothing in a record is typed by hand.

## Making one

```
cargo run --release -p turbo-bench -- record --bundle <bundle-dir> [options]
```

The tool uses the library through its C interface, as any caller does:
it lists the devices, loads the bundle on the one `--device` names, makes
a session at `--precision`, writes the token rows with
`turbo_embed_write_tokens`, runs, and reads the vectors back with
`turbo_result_read`. Then it runs each reference program on the same
rows, and writes the record into `benchmarks/records/` of the working
tree the library was built from, under the name below. A record of that
name is never replaced.

A record backs the (backend, task, precision) cell on that machine and
operating system, whatever model is later loaded there: the bundle is
what was measured, and it is named in the record, but the cell does not
depend on it. So the tool refuses a bundle under the repository's
`testdata/` (compared as canonical paths, the manifest's too, so a link
or a `..` path into it is refused as well): those are test fixtures, not
models anyone runs. The guard catches a mistake, not a determined copy:
the same files copied elsewhere pass it.

| Option | Meaning |
|---|---|
| `--bundle <dir>` | The bundle. Required; not one under `testdata/`. |
| `--device <index\|backend>` | A runtime device index, or a backend name for the first device it lists. Default `cpu`. |
| `--precision model\|fastest\|exact` | The session's precision. Default `model`. |
| `--batch <n>`, `--seq <n>` | The rows' shape. Default: 32 rows, or the model's `max_batch` if fewer; the longest reference case that fits the model's `max_seq`. |
| `--warmup <n>`, `--iterations <n>` | Untimed runs, then timed runs. Default 20 and 200. |
| `--repo <dir>` | The git working tree the library was built from; its commit must be the tool's build commit (Provenance). Default: the one the tool was built in. |
| `--out <dir>` | Where the record goes. Default `<repo>/benchmarks/records`. |
| `--tei-image <name@sha256:…>`, `--tei-model <dir>` | text-embeddings-inference: its image, and the model in the upstream layout (the directory `turbo-bundle fetch` fills). |
| `--no-tei` | TEI is not run; the record says so. |
| `--tensorrt-image <name@sha256:…>` | NVIDIA's TensorRT container. |
| `--tensorrt-inputs <ids,mask,types>` | The ONNX inputs, in that order, each of `[A-Za-z0-9_.]+` and neither `.` nor `..` (each names a file and a part of trtexec's lists). Default `input_ids,attention_mask,token_type_ids`. |
| `--tensorrt-input-dtype int64\|int32` | Their element type. Default `int64`. |
| `--tensorrt-warmup-ms <n>` | trtexec's `--warmUp`. Default 1000. |
| `--trtexec <path>` | trtexec inside the image. Default `trtexec`. |
| `--no-tensorrt` | TensorRT is not run; the record says so. |
| `--openvino-image <name@sha256:…>` | An OpenVINO container with `benchmark_app` (levelzero). |
| `--openvino-inputs <ids,mask,types>`, `--openvino-input-dtype int64\|int32` | As for TensorRT, for benchmark_app. Default `input_ids,attention_mask,token_type_ids` and `int64`. |
| `--benchmark-app <path>` | benchmark_app inside the image. Default `benchmark_app`. |
| `--no-openvino` | OpenVINO is not run; the record says so. |
| `--work <dir>` | Scratch for trtexec's and benchmark_app's input files. Default the system's temporary directory. |

Every reference program the tool knows for the device's backend must be
named or disabled; leaving one out is an error, checked before anything
is measured, so a record never omits
a program silently. `turbo-bench check <record.json>...` parses records
and says whether each backs SUPPORTED for its own cell in this build.

On an RTX 4080, with the toolkit in `/usr/local/cuda` and both images
pulled, from a clean checkout of a commit pushed to origin:

```
CUDA_DEVICE_ORDER=PCI_BUS_ID TURBO_CUDA_ROOT=/usr/local/cuda cargo run --release -p turbo-bench --features cuda -- record \
    --bundle all-minilm-l6-v2/ --device cuda --precision model \
    --tei-image ghcr.io/huggingface/text-embeddings-inference@sha256:<digest of the 89-* image> \
    --tei-model upstream/ \
    --tensorrt-image nvcr.io/nvidia/tensorrt@sha256:<digest>
```

On an Intel Arc GPU, with Level Zero installed, one Intel GPU listed,
and both images pulled:

```
cargo run --release -p turbo-bench --features levelzero -- record \
    --bundle all-minilm-l6-v2/ --device levelzero --precision model \
    --tei-image ghcr.io/huggingface/text-embeddings-inference@sha256:<digest of the cpu-* image> \
    --tei-model upstream/ \
    --openvino-image openvino/ubuntu24_dev@sha256:<digest>
```

## Provenance

A record names the commit the library was built from, and the tool
refuses to make one unless that name is true.

When the tool is built, `bench/build.rs` runs `git rev-parse HEAD` and
`git status` in the tree it is built in, and compiles in the commit and
the tree's changes (the same exemption as step 2 below). It is run
again whenever HEAD, the branch HEAD names or `packed-refs` moves (found
with `git rev-parse --git-path`, so a worktree, whose `.git` is a file,
works), and whenever anything under `core/`, `include/`, `bench/` or
`benchmarks/`, or the workspace's `Cargo.toml` or `Cargo.lock`, changes,
or any path in the changes it found: what it compiled in is never older
than the binary, and a build of a dirty tree runs again when the tree
is put back. Only paths that exist are handed to cargo (a missing one
reads as changed on every call): a branch that is only in `packed-refs`
is watched through its directory, where its next commit writes it.

Then, with git, in `--repo`, before measuring and again before writing:

1. `git rev-parse --show-toplevel`: it is a working tree.
2. `git status --porcelain=v1 --untracked-files=all
   --ignore-submodules=none` lists nothing but new, untracked files
   under `benchmarks/records/`: no change, staged or not, and no
   untracked file git does not ignore. New records are left out because
   they are not what is measured, so several can be made before they are
   committed; a committed record changed or deleted is a change like any
   other.
3. `git rev-parse --verify HEAD^{commit}` is the commit.
4. `git remote get-url origin`: there is an origin.
5. `git for-each-ref --contains <commit> refs/remotes/origin/` lists at
   least one branch, other than `origin/HEAD`. These are the refs the
   last push or fetch left; the tool does not reach the remote. The
   branches are recorded in `library.pushed_to`.

The two checks, before and after, must give the same answer. Before
measuring, the commit must also be the one the tool was built from, and
the build's tree must have had no changes; otherwise the library in the
tool is not the commit the record would name. So the tool is run with
`cargo run` from the clean, pushed tree, which builds it from that
commit first.

## The record

A sample. Its figures are from a run of an earlier build of
`turbo-bench record` on the CPU backend, with the small bundle in
`testdata/`, TEI disabled, and `--repo` a scratch repository pushed to a
local origin. The tool now refuses both that bundle and that repository,
so the commit names nothing in this repository; `manifest_sha256` is the
small bundle's as it is now.

```json
{
  "record_version": 1,
  "recorded_at": "2026-09-25T00:25:23Z",
  "machine": { "arch": "x86_64", "host_cpu": "Intel(R) Xeon(R) Processor @ 2.10GHz", "os": "linux" },
  "device": {
    "backend": "cpu", "kind": "DEVICE_CPU", "name": "Intel(R) Xeon(R) Processor @ 2.10GHz",
    "vendor": "GenuineIntel", "driver_version": "", "runtime_version": "", "memory_total": 16877465600
  },
  "library": {
    "version": "0.1.0", "build": "0.1.0 cpu",
    "commit": "2547f28967c52eed949ebe432f20a2c1a71f6157", "pushed_to": ["origin/main"]
  },
  "task": "TASK_EMBED",
  "precision": "PRECISION_MODEL",
  "compute_dtype": "DTYPE_F32",
  "bundle": {
    "model_id": "sentence-transformers/all-MiniLM-L6-v2", "revision": "1",
    "manifest_sha256": "273ee3dfe946989a27a68e37d58e1dd7bc537c06b404075279f02a1b640b91f7",
    "artifact_sha256": "c258bffc3c37afc5a8b12324c0d29b81d31738fbf72183ade9c52c0d68e06180",
    "tokenizer_sha256": "da0e79933b9ed51798a3ae27893d3c5fa4a201126cef75586296df9b4d2c62a0"
  },
  "rows": {
    "batch": 32, "seq": 64, "live_tokens": 621,
    "cases": [0, 1, 2, 3, 4, 5, 6, 7, 8, 0, 1, 2, 3, 4, 5, 6, 7, 8, 0, 1, 2, 3, 4, 5, 6, 7, 8, 0, 1, 2, 3, 4],
    "sha256": "b7abcfa07e539dd88d3f5b08d06f184e065c3b6789163a9b2d0be45d3c1fedde"
  },
  "timing": {
    "warmup": 20, "iterations": 200, "p50_ms": 3.833206, "p99_ms": 3.950523,
    "mean_ms": 3.6571804100000027, "min_ms": 3.241942, "max_ms": 4.662617,
    "rows_per_second": 8749.762009891207
  },
  "conformance": { "rows": 41, "min_cosine": 0.999999999999992, "max_abs_diff": 8.940696716308594e-8 },
  "references": [
    {
      "name": "text-embeddings-inference", "role": "end_to_end", "pinned": "", "version": "",
      "commands": [], "procedure": "", "measured": null,
      "not_run": "disabled on the command line (--no-tei)"
    }
  ],
  "speed_ratio": null,
  "speed_reference": null
}
```

The tool writes it with serde_json's pretty printer: two-space indent,
every object member and every array element on a line of its own. Here
most objects and `cases` are folded onto fewer lines. Every
field is required, and an unknown one is an error.

| Field | Meaning |
|---|---|
| `record_version` | 1. |
| `recorded_at` | When the measurement finished, UTC. |
| `machine.arch` | `turbo_device_info.arch` of the device: the label the record is filed under (`rtx4080`, `x86_64`). |
| `machine.host_cpu` | The CPU device's name, for context; empty in a build that lists no CPU. |
| `machine.os` | The operating system the tool was built for (`std::env::consts::OS`): part of the cell's key. |
| `device.*` | `turbo_device_info`: backend, kind as `DEVICE_*`, name, vendor, driver and runtime versions, total memory. |
| `library.version` | The version number at the start of `turbo_version()`. |
| `library.build` | `turbo_version()` whole: the version and the backends linked. |
| `library.commit`, `.pushed_to` | See Provenance. |
| `task`, `precision` | `TASK_EMBED`; `PRECISION_*` as the session asked. |
| `compute_dtype` | `DTYPE_*` as `turbo_session_get_info` reported it. `DTYPE_I8` is accepted and has no floor. |
| `bundle.*` | `turbo_model_info`: model id and revision, and the manifest, artifact and tokenizer hashes. |
| `rows` | The shape, the live tokens, the reference case each row is, and the hash of the rows (below). |
| `timing` | The library: `warmup` untimed runs, then `iterations` timed ones, each a `turbo_embed_write_tokens`, `turbo_session_run`, `turbo_result_read` of every vector and `turbo_result_release`, timed from the host. Nearest-rank p50 and p99, mean, min, max, and rows per second over the timed runs' wall time. |
| `conformance` | The rows compared with the bundle's fp32 reference on this device, through the C interface: each distinct case alone as a batch of one at its own length, then every row of the last timed batch. The lowest cosine, in [-1, 1], and the largest absolute difference, not negative. |
| `references[]` | Each reference program the tool knows for the backend: `name` and `role`, which are `text-embeddings-inference` and `end_to_end`, `tensorrt` and `kernel`, or `openvino` and `kernel` (any other pair is refused), `pinned` (the image as `name@sha256:<64 hex>`, the name of `[a-z0-9][a-z0-9._/:-]*`; empty only when disabled before one was named), `version` (as the program reported it), `commands` (every external command, as its argv), `procedure` (what the tool did around them), and either `measured` (`iterations`, `p50_ms`, `p99_ms`, `rows_per_second`, and `min_cosine` against the reference when the program returns vectors) or `not_run` with the reason. |
| `speed_ratio` | `timing.p50_ms` over the p50 of the fastest measured reference, named in `speed_reference`; both null when none was measured. The core recomputes it and refuses a record where it differs. |

### Token rows

The rows are the bundle's reference cases (`reference.file`'s `ids` and
`lengths`), in case order, those no longer than `seq`, repeated until
`batch` rows are full; each is padded to `seq` with the tokenizer's pad
id and mask 0, and every type is 0. Their hash is SHA-256 of the bytes
`turbo-bench rows 1`, a NUL, `batch` and `seq` as little-endian u32,
then the ids, the mask and the types as little-endian i32, row-major:
exactly what `turbo_embed_write_tokens` is given. Every reference program
is given the same rows.

### Name

```
<machine>.<backend>.<task>.<precision>.<model>-<manifest>.<commit>.json
```

`machine` is the arch label; for a CPU, the arch label, a dash and the
first 8 hex of the SHA-256 of the processor's name, since a CPU record
is filed under both (`core/src/cpu.rs`). `task` and `precision` are the
enum names without their prefix, in lower case. `model` is the last part
of the model id, at most 32 characters; `manifest` the first 8 hex of the
manifest hash; `commit` the first 12 of the commit. Every part is lower
case letters, digits and single dashes. The name must fit
`turbo_capability.benchmark`, 95 bytes; the tool refuses a longer one,
and the core refuses a record whose file name is not the one its
contents give. For example
`rtx4080.cuda.embed.model.all-minilm-l6-v2-<8 hex>.<12 hex>.json`.

## Reference programs

| Backend | Program | Role | How it runs |
|---|---|---|---|
| cuda | TensorRT `trtexec` | kernel | NVIDIA's TensorRT container, on the bundle's ONNX file |
| cuda | text-embeddings-inference | end to end | its GPU image, over HTTP |
| cpu | text-embeddings-inference | end to end | its CPU image, over HTTP |
| levelzero | OpenVINO `benchmark_app` | kernel | an OpenVINO container, on the bundle's ONNX file, on the GPU |
| levelzero | text-embeddings-inference | end to end | its CPU image, over HTTP: the end-to-end baseline on that machine |
| metal | none yet | | a record of it backs nothing |
| any other | none yet | | a record of it backs nothing |

Metal has no reference program yet, so a Metal cell cannot reach
SUPPORTED.

Images are pinned as `name@sha256:<64 hex>`, the name of lower-case
letters, digits and `._/:-`, starting with a letter or digit (so never
an option); a tag is refused. The tool
never pulls (`--pull never`, and `docker image inspect` first), so what
runs is what was fetched on purpose. Each command is recorded as run.

**TensorRT** builds an engine from the bundle's `FORMAT_ONNX` artifact,
checked against its hash, and times the rows loaded from raw files. The
library never executes ONNX; a reference program may. A bundle with no
ONNX artifact gives a reference with `not_run` saying so. The command:

```
docker run --rm --pull never --network none --gpus device=<ordinal> \
    --mount type=bind,src=<bundle>,dst=/bundle,readonly \
    --mount type=bind,src=<work>,dst=/work,readonly \
    <image> trtexec --onnx=/bundle/<onnx file> \
    --shapes=input_ids:<batch>x<seq>,attention_mask:<batch>x<seq>,token_type_ids:<batch>x<seq> \
    --loadInputs=input_ids:/work/input_ids.bin,attention_mask:/work/attention_mask.bin,token_type_ids:/work/token_type_ids.bin \
    --warmUp=<ms> --iterations=<iterations> --duration=0 --percentile=99 <precision flag>
```

The precision flag is `--noTF32` for F32 (the library computes F32 in
F32), `--fp16` for F16, `--bf16` for BF16. The output must end in
`&&&& PASSED`; the version is its `TensorRT version:` line, p50 and p99
the summary's `Latency` median and `percentile(99%)` (the H2D copy, the
GPU compute and the D2H copy of one batch), the iterations its `Timing
trace has N queries`, and rows per second its `Throughput` times the
batch. Its GPU Compute Time median and p99 go in `procedure`. The ONNX
graph stops at the hidden states, so this time has no pooling or
normalization in it, and the library's has. `<ordinal>` is the CUDA
device ordinal the backend listed; docker's `--gpus device=` counts in
the driver's order, so the two agree when `CUDA_DEVICE_ORDER=PCI_BUS_ID`
is set for the tool. With more than one CUDA device listed, the tool
refuses to run a reference program on a GPU without it.

**text-embeddings-inference** serves the model directory in the upstream
layout. Its `tokenizer.json` must have the bundle tokenizer's hash and
its `model.safetensors` the loaded artifact's, and `config.json` must be
there; otherwise the reference is `not_run`. The command:

```
docker run --detach --rm --pull never --name turbo-bench-tei-<pid> [--gpus device=<ordinal>] \
    --publish 127.0.0.1::80 --env HF_HUB_OFFLINE=1 \
    --mount type=bind,src=<model dir>,dst=/model,readonly <image> \
    --model-id /model --port 80 --dtype <float32|float16> --pooling <mean|cls|last-token> \
    --max-client-batch-size <batch> --max-batch-tokens <max(batch x seq, 16384)>
```

then `docker port` for the host port, `GET /health` until it answers
(at most 600 s, and never after the container stops), and `GET /info`
for its version and dtype. TEI takes rows as token ids but decodes them
to text and encodes that again without special tokens; the tool asks it
to do exactly that (`POST /decode` with `skip_special_tokens: false`,
then `POST /tokenize` with `add_special_tokens: false`) and records
`not_run` unless every row comes back as the same ids. Then `POST /embed`
with the batch's rows as ids (no padding), the bundle's normalization
and `truncate: false`, `--warmup` times untimed and `--iterations` times
timed, each from sending the request to reading the whole response. The
last response's vectors give `min_cosine`. The container is removed
when the tool is done with it. TEI has no BF16 dtype; a BF16 session
records it as `not_run`.

The command gives no `--auto-truncate`. In TEI's router
(`router/src/main.rs`) it is a bare flag through 1.8, so a value after
it is a stray argument and the router exits; from 1.9 it takes a value
and defaults to true. Either way it only sets the default for a request
that does not say, and every `/embed` request here says `truncate:
false`, so an over-long row is an error, never cut.

**OpenVINO** compiles the bundle's `FORMAT_ONNX` artifact, checked
against its hash, for the Intel GPU with `benchmark_app`, and times the
rows loaded from raw files, one synchronous request at a time, as the
library runs. A bundle with no ONNX artifact gives a reference with
`not_run` saying so. The command:

```
docker run --rm --pull never --network none --device /dev/dri --group-add <render gid> \
    --mount type=bind,src=<bundle>,dst=/bundle,readonly \
    --mount type=bind,src=<work>,dst=/work,readonly \
    <image> benchmark_app -m /bundle/<onnx file> -d GPU -hint latency -api sync -nireq 1 \
    -niter <iterations> \
    -shape input_ids[<batch>,<seq>],attention_mask[<batch>,<seq>],token_type_ids[<batch>,<seq>] \
    -i input_ids:/work/input_ids.bin,attention_mask:/work/attention_mask.bin,token_type_ids:/work/token_type_ids.bin \
    -infer_precision <f32|f16> -latency_percentile <50|99>
```

`<render gid>` is the group of the first `/dev/dri/renderD*` node, so
the container's user may open it. benchmark_app reports one latency
percentile per run, so the command runs twice, the same but for
`-latency_percentile`, 50 then 99; both are recorded. Each run makes
one untimed first inference (benchmark_app's warm-up, which it does not
let a count be set for), then `-niter` timed ones. From its report
(`[ INFO ]` lines, as its Python and C++ forms print them): the version
is the `Build` line under `OpenVINO:`; p50 is the first run's `Median`
and p99 the second run's `99 percentile`, in the `Latency` block
(microseconds when the average is under a millisecond, converted); the
iterations its `Count`; rows per second the first run's count times the
batch over its `Duration`. Its `Average` and `Throughput` go in
`procedure`. The two runs must report the same version and count, and
the second a p99 no lower than the first's median, or the tool stops
with an error. `-infer_precision` is `f32` for F32 and `f16` for F16;
the GPU plugin has no BF16, so a BF16 session records `not_run`. With
more than one Level Zero device listed the tool refuses to run it:
benchmark_app's `GPU` is OpenVINO's first, which need not be the device
measured. On that machine TEI's CPU image is the end-to-end reference.

For the CPU backend the reference is TEI's CPU image, which runs on any
x86_64 machine. A CPU record without it measured backs nothing.

## The SUPPORTED rule

For a cell (device, task, precision), the backend first says what it
has built. If it says UNSUPPORTED, that stands. If it says EXPERIMENTAL,
the core looks at the records compiled into the build.

A record is for the cell when its `machine.arch` is the device's arch
label, `machine.os` the operating system this build is for,
`device.backend` the device's backend, and `task` and `precision` the
cell's; for a CPU, `device.kind` is `DEVICE_CPU` and `device.name` is
the processor's name too. The bundle is not part of the key: a record
backs the (backend, task, precision) cell on that machine and OS,
whatever model is later loaded there.

A record for the cell backs SUPPORTED when all of these hold:

1. `library.version` is this build's version number (the start of
   `turbo_version()`).
2. `compute_dtype` has a tolerance, and it is the dtype the backend says
   the precision computes in.
3. The conformance reaches the tolerance for that dtype:

   | Compute dtype | Lowest cosine | Largest absolute difference |
   |---|---|---|
   | F32 | 0.9999 | 1e-4 |
   | F16, BF16 | 0.999 | not bounded |
   | I8 | none: never SUPPORTED | |

4. At least one of the known reference programs,
   `text-embeddings-inference`, `tensorrt` or `openvino`, was measured.
   (A record naming any other program, or one with another's role, does
   not parse.) The tool gives each backend its own programs (Reference
   programs), so a levelzero record is backed by OpenVINO or TEI.

Of the records that back the cell, the newest by `recorded_at` (then by
name) is named: the cell is SUPPORTED, `benchmark` is its file name,
`cosine_floor` its `conformance.min_cosine`, and `speed_ratio` its
`speed_ratio`. When none backs it, the cell stays EXPERIMENTAL and
`reason` is the newest record's file name and what it lacks, or `no
benchmark record for this cell` when no record is for it.

The version rule is the simplest one that is honest: a record counts
only for the library version it was measured with. A later commit of the
same version may run different code, and the record still names the
commit it was measured at; a new version number needs new records.

`core/src/record.rs` holds the schema, the name, the checks and the
rule; the tool writes records through the same types and checks, and
`core/tests/conformance.rs` uses the same tolerances.

## Tests

`bench/tests/` runs without docker or a GPU:

- `provenance.rs`: the refusals, in git repositories made in temporary
  directories with a bare origin.
- `records.rs`: the schema, the name, the parser's refusals and the
  SUPPORTED rule, on records made from a real measurement of the CPU
  backend on the small bundle. A reference program cannot run there, so
  the tests that need a measured reference give the record one whose
  figures are derived from that measurement; such a record is never
  written where the core reads records.
- `tool.rs`: `turbo-bench record` run as a program. Its refusals, on
  trees made for the test: a tree it was not built from, a dirty tree, a
  bundle in `testdata/`, a reference neither named nor disabled, an
  unpinned image. And, only when this checkout is a clean commit on a
  branch of origin and the tool was built from it (otherwise the test
  prints why to stderr and skips, or fails when
  `TURBO_BENCH_REQUIRE_RECORD=1`, as CI sets it on pushes to main), a record made end to end on the CPU with a copy
  of the small bundle and TEI disabled, into a temporary directory:
  every field is checked against what the library, git and the bundle
  say, and the record backs nothing.
- `runners.rs`: the reference programs' commands, argument by argument;
  their output parsed from examples in the form TEI, trtexec and
  benchmark_app print it; which programs each backend gets; and the
  cases recorded as not run.

What they cannot check is the programs themselves: a TEI, trtexec or
benchmark_app run needs docker and the images, trtexec an NVIDIA GPU,
and benchmark_app an Intel one.
