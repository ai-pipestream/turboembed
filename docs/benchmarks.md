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

| Option | Meaning |
|---|---|
| `--bundle <dir>` | The bundle. Required. |
| `--device <index\|backend>` | A runtime device index, or a backend name for the first device it lists. Default `cpu`. |
| `--precision model\|fastest\|exact` | The session's precision. Default `model`. |
| `--batch <n>`, `--seq <n>` | The rows' shape. Default: 32 rows, or the model's `max_batch` if fewer; the longest reference case that fits the model's `max_seq`. |
| `--warmup <n>`, `--iterations <n>` | Untimed runs, then timed runs. Default 20 and 200. |
| `--repo <dir>` | The git working tree the library was built from. Default: the one the tool was built in. |
| `--out <dir>` | Where the record goes. Default `<repo>/benchmarks/records`. |
| `--tei-image <name@sha256:…>`, `--tei-model <dir>` | text-embeddings-inference: its image, and the model in the upstream layout (the directory `turbo-bundle fetch` fills). |
| `--no-tei` | TEI is not run; the record says so. |
| `--tensorrt-image <name@sha256:…>` | NVIDIA's TensorRT container. |
| `--tensorrt-inputs <ids,mask,types>` | The ONNX inputs, in that order. Default `input_ids,attention_mask,token_type_ids`. |
| `--tensorrt-input-dtype int64\|int32` | Their element type. Default `int64`. |
| `--tensorrt-warmup-ms <n>` | trtexec's `--warmUp`. Default 1000. |
| `--trtexec <path>` | trtexec inside the image. Default `trtexec`. |
| `--no-tensorrt` | TensorRT is not run; the record says so. |
| `--work <dir>` | Scratch for trtexec's input files. Default the system's temporary directory. |

Every reference program the tool knows for the device's backend must be
named or disabled; leaving one out is an error, so a record never omits
a program silently. `turbo-bench check <record.json>...` parses records
and says whether each backs SUPPORTED for its own cell in this build.

On an RTX 4080, with the toolkit in `/usr/local/cuda` and both images
pulled:

```
TURBO_CUDA_ROOT=/usr/local/cuda cargo run --release -p turbo-bench --features cuda -- record \
    --bundle all-minilm-l6-v2/ --device cuda --precision model \
    --tei-image ghcr.io/huggingface/text-embeddings-inference@sha256:<digest of the 89-* image> \
    --tei-model upstream/ \
    --tensorrt-image nvcr.io/nvidia/tensorrt@sha256:<digest>
```

## Provenance

A record names the commit the library was built from, and the tool
refuses to make one unless that name is true. With git, in `--repo`,
before measuring and again before writing:

1. `git rev-parse --show-toplevel`: it is a working tree.
2. `git status --porcelain=v1 --untracked-files=all
   --ignore-submodules=none` lists nothing outside `benchmarks/records/`:
   no change, staged or not, and no untracked file git does not ignore.
   Records are left out because they are not what is measured, so
   several can be made before they are committed.
3. `git rev-parse --verify HEAD^{commit}` is the commit.
4. `git remote get-url origin`: there is an origin.
5. `git for-each-ref --contains <commit> refs/remotes/origin/` lists at
   least one branch, other than `origin/HEAD`. These are the refs the
   last push or fetch left; the tool does not reach the remote. The
   branches are recorded in `library.pushed_to`.

The two checks, before and after, must give the same answer. The tool is
meant to be run with `cargo run` from that tree, which builds the library
from it first.

## The record

A sample, from `turbo-bench record` on the CPU backend with the small
bundle in `testdata/` and TEI disabled, run with `--repo` a scratch
repository pushed to a local origin (so the commit is that
repository's, as in the tool's own tests):

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
    "manifest_sha256": "c8dcbbc0d2aa9a12bd3cb8c941382028e5410bd04b39335650008b6a9a0bc39c",
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

The tool writes it one field to a line; it is folded here. Every field
is required, and an unknown one is an error.

| Field | Meaning |
|---|---|
| `record_version` | 1. |
| `recorded_at` | When the measurement finished, UTC. |
| `machine.arch` | `turbo_device_info.arch` of the device: the label the record is filed under (`rtx4080`, `x86_64`). |
| `machine.host_cpu` | The CPU device's name, for context; empty in a build that lists no CPU. |
| `machine.os` | The operating system the tool was built for. |
| `device.*` | `turbo_device_info`: backend, kind as `DEVICE_*`, name, vendor, driver and runtime versions, total memory. |
| `library.version` | The version number at the start of `turbo_version()`. |
| `library.build` | `turbo_version()` whole: the version and the backends linked. |
| `library.commit`, `.pushed_to` | See Provenance. |
| `task`, `precision` | `TASK_EMBED`; `PRECISION_*` as the session asked. |
| `compute_dtype` | `DTYPE_*` as `turbo_session_get_info` reported it. `DTYPE_I8` is accepted and has no floor. |
| `bundle.*` | `turbo_model_info`: model id and revision, and the manifest, artifact and tokenizer hashes. |
| `rows` | The shape, the live tokens, the reference case each row is, and the hash of the rows (below). |
| `timing` | The library: `warmup` untimed runs, then `iterations` timed ones, each a `turbo_embed_write_tokens`, `turbo_session_run`, `turbo_result_read` of every vector and `turbo_result_release`, timed from the host. Nearest-rank p50 and p99, mean, min, max, and rows per second over the timed runs' wall time. |
| `conformance` | The rows compared with the bundle's fp32 reference on this device, through the C interface: each distinct case alone as a batch of one at its own length, then every row of the last timed batch. The lowest cosine and the largest absolute difference. |
| `references[]` | Each reference program the tool knows for the backend: `name`, `role` (`kernel` or `end_to_end`), `pinned` (the image by digest; empty only when disabled before one was named), `version` (as the program reported it), `commands` (every external command, as its argv), `procedure` (what the tool did around them), and either `measured` (`iterations`, `p50_ms`, `p99_ms`, `rows_per_second`, and `min_cosine` against the reference when the program returns vectors) or `not_run` with the reason. |
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
| any other | none yet | | a record of it backs nothing |

Images are pinned as `name@sha256:<64 hex>`; a tag is refused. The tool
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
    --max-client-batch-size <batch> --max-batch-tokens <max(batch x seq, 16384)> --auto-truncate false
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

For the CPU backend the reference is TEI's CPU image, which runs on any
x86_64 machine. A CPU record without it measured backs nothing.

## The SUPPORTED rule

For a cell (device, task, precision), the backend first says what it
has built. If it says UNSUPPORTED, that stands. If it says EXPERIMENTAL,
the core looks at the records compiled into the build.

A record is for the cell when its `machine.arch` is the device's arch
label, `device.backend` the device's backend, and `task` and `precision`
the cell's; for a CPU, `device.kind` is `DEVICE_CPU` and `device.name`
is the processor's name too.

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

4. At least one reference program was measured.

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
- `tool.rs`: `turbo-bench record` run as a program on the CPU with TEI
  disabled, into a temporary directory: every field is checked against
  what the library, git and the bundle say, and the record backs
  nothing.
- `runners.rs`: the reference programs' commands, argument by argument;
  their output parsed from examples in the form TEI and trtexec print
  it; and the cases recorded as not run.

What they cannot check is the programs themselves: a TEI or trtexec run
needs docker and the images, and trtexec a GPU.
