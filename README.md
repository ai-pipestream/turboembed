# TurboEmbed

TurboEmbed is a native embedding and inference library for text models.
It presents one small C interface, and behind that interface each
backend is written against the lowest layer the hardware vendor gives:
CUDA kernels on NVIDIA, Level Zero on Intel GPUs, Metal on Apple
silicon, HailoRT on Hailo boards, and a plain CPU path. The aim is an
API that reads cleanly on top and, underneath, a path picked for speed
on the hardware the program is running on and the model it has loaded,
with as little hardware churn and memory copying as the job allows.

## What it does

Embedding works today: text in, vectors out, from one header, on a CPU,
an NVIDIA GPU, an Intel GPU, an Apple GPU or a Hailo accelerator. Other
tasks, such as reranking, classification, token tagging and chunking,
are added to the header when they are built, not before.

The unit of work is a task, not a tensor operation. "Embed these texts"
is one call. The backend runs the stages on the device and keeps the
data there between them; the only bytes that come back are the vectors
the caller asked for. The library counts each copy between host and
device and returns the count with the result, so a slow path shows up
as a number rather than a feeling.

A model travels as a bundle: a directory with a manifest, hashes, the
weights, the tokenizer and the model's settings, such as pooling,
normalization, sequence length, prefixes and output dimension. Point
different machines at the same bundle and they give the same answer.
Tokenization happens once, in the core, so tokens are the same on any
machine.

Selection is part of the interface. The caller names a task and, if it
wants, constraints; the library answers with the bundles on this machine
that can do it, the backend and device that will do it fastest, and the
evidence that answer rests on.

## Speed, and how it is claimed

The library adapts to the device and the model rather than running one
generic graph. On NVIDIA, for example, GEMM tiles, split strategies and
attention kernels are chosen per architecture and per shape, and an
autotuner keeps what it measured. Each backend is held to the same
standard: slower than the vendor's fastest program on the same inputs is
a bug, not a trade-off.

A claim here is a test, not a sentence. A capability is reported as
supported only when a benchmark record in `benchmarks/records/` backs
it: one measurement, on a named class of device, at a named commit,
against the vendor's fastest program at a pinned version, on the same
inputs, with the command that produced it. There are no hand-written
numbers, and a comparison that is not fair produces no number at all.
Conformance tests compare each backend's vectors with the reference
outputs the bundle carries, at each precision the backend offers.

The tests and the record tool live in the tree and run on your hardware
the same way they run on ours, so a result you read here is one you can
reproduce. A packaged suite that runs the full set on a machine of your
choosing and reports the outcome as a claim for that machine is the
next piece of this.

## Shape of the code

The header is the design. `include/turbo/turbo.h` is one hand-written
file that compiles standalone as C11 and as C++17. Changing it is a
design decision made in the open, not a build step, and nothing is
added to it that a built feature does not need.

The core is Rust. It loads bundles, tokenizes on the host, hands token
rows to a backend and hands results back. Backends are written in what
the vendor speaks: CUDA C++ on NVIDIA, C++ against Level Zero and
HailoRT, Objective-C++ on Metal. The gRPC server in `server/` speaks the
Open Inference Protocol so it runs under KServe. Language bindings sit
on the same header and add nothing to it; Java, for one, goes through
JDK 25's foreign function interface.

A few rules hold the shape:

- Nothing is called working until it has run on the real hardware.
  There is no fake device in the normal path. A test that cannot run on
  the current machine is skipped and says so.
- A stage runs where its data is. Copies between host and device are
  counted and reported.
- The model's settings live in the bundle, not in code.
- If the hardware cannot do what was asked, the call fails and says what
  it cannot do. Nothing is substituted, defaulted or clamped.
- No Python in the tree. A vendor's Python tool runs inside a pinned
  container, driven from Rust.
- Machines are named by what they are, such as `rtx4080`, `b70`,
  `orin-nano`, `pi5-hailo8` or `m2`, and nothing from a particular
  machine (host names, user names, paths) is committed.

## Reading further

- `docs/bundle.md`: the bundle format and the `turbo-bundle` tool.
- `docs/conformance.md`: how a backend is checked against a bundle's
  reference outputs.
- `docs/benchmarks.md`: what a benchmark record is, how it is made with
  `turbo-bench`, and how records back capabilities.
- `docs/autotune.md`: the tuner and its cache.
- `docs/demo.md`: a web page for trying the gRPC server from a browser.
- `docs/cpu.md`, `docs/cuda.md`, `docs/levelzero.md`, `docs/metal.md`,
  `docs/hailo.md`, `docs/npu.md`: one page per backend.
- `docs/kserve.md`: the gRPC server.

## The previous attempt

This code is a restart. An earlier attempt is kept in
`ai-slop-generated-shit/`, with its history, for reading. Its audits in
`docs/reviews/` there explain what went wrong: benchmarks that compared
the code with itself, hand-written test records, a fake default device,
and an interface with several copies of the same data on the way to the
caller. The header files were the part worth keeping and are where this
tree started. Nothing else comes back from that folder without being
read first.

## Licence

Apache-2.0. See `LICENSE`. The CUDA backend carries a subset of NVIDIA's
CUTLASS headers under `core/cuda/cutlass/`, BSD-3-Clause, with their
licence in `core/cuda/cutlass/LICENSE.txt`.
