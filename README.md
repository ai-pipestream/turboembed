# TurboEmbed

A native library that runs text models on whatever accelerator a machine
has, through one C interface, with each backend written directly against
the vendor's lowest layer. Embedding first. Reranking, classification,
token tagging, chunking and generation follow one at a time, each
landed on every machine we own before the next starts. Java, Swift,
Rust and a gRPC server sit on the C interface and add nothing to it.

This is a restart. The previous attempt is in `ai-slop-generated-shit/`,
moved there whole with its history on 2026-09-23. Its two audits
(`docs/reviews/2026-09-23-audit-*.md` in that folder) say why: the
benchmarks compared the code with itself or with a crippled reference,
the test records were hand-written, the default device was a fake, and
the "one interface" had three tokenizers, two copies of its C
conversions and five copies of the data on the way to Java. The header
files were the one part worth keeping, and they are the starting point
here. Everything else is written again, against the rules below, and
nothing from that folder comes back without being read first.

## What we are building

One library, `libturbo`, with one C header. A program calls it the same
way on every machine we own:

- an x86 box with an RTX 4080 SUPER
- an x86 box with an Intel Arc B70
- an Intel NPU (leased now, a laptop later)
- a Jetson Orin Nano
- a Raspberry Pi 5 with a Hailo-8, and one with a Hailo-10H
- an M2 Mac

For each of those, the backend is the lowest thing the vendor gives us:
CUDA kernels and cuBLASLt on NVIDIA, OpenVINO's compiled graph on Intel
GPU, CPU and NPU, our own Metal kernels on Apple, HailoRT on the Pi,
llama.cpp for GGUF models. Not ONNX Runtime, not a framework. ONNX is a
file format users bring; it is never the thing that runs.

The unit of work is a task, not an operation. "Embed these texts" or
"chunk this document and embed the chunks" is one call, and the backend
runs the whole thing on the device, keeping the data there between
stages. The only things that come back to the host are the vectors, or
the spans, that the caller asked for.

The model's settings (pooling, normalization, sequence length, prefixes,
dimension, which tokenizer) travel in a bundle: a directory with a
manifest and hashes. Point two different machines at the same bundle and
they give the same answer.

Selection is part of the interface. The caller names a task and, if it
wants, constraints; the library says which bundles on this machine can
do it and which backend and device will do it fastest, and why.

## Why

I build text pipelines and I was developing them on a Mac and deploying
them on Linux. Every accelerator has its own vocabulary, and you relearn
it with every new card. The libraries that promise to hide that either
go through a generic runtime, which is convenient and slow because the
data keeps crossing the bus, or they keep the vendor's pipeline and lose
the common interface.

I own an NVIDIA card, an Intel card, two Hailo boards and a Mac, and I
want all of them used to their limit through one API. Slow hardware is
fine. A path that is slower than what the hardware can do is a bug.

The specific thing that started this: running chunking and embedding as
separate services on one GPU makes the data go GPU, host, GPU, host for
no reason. If both stages are on the card, the card should do both in
one pass. That is what "a stage runs where its data is" means.

ONNX is built for elegance: one format, one session API, every backend
behind it. This is built for speed. When a cleaner abstraction and a
faster path disagree, the faster path wins and the abstraction is bent
to fit it.

## How

**The header is the design.** `include/turbo/turbo.h` is one file,
hand-written, cut from the previous attempt's three headers to what the
first feature needs: text embedding, tokens in and vectors out, on one
device, with a summary of where each stage ran and every byte that
crossed the bus. The other tasks and chunking are added to it when they
are built, not before. Changing it is a design decision made in the
open, not a build step. It compiles standalone as C11 and C++17.

**Rust core, vendor backends in whatever the vendor speaks.** The core
loads bundles, tokenizes once on the host, hands token rows to a
backend, and hands results back. Backends are C++ (OpenVINO, HailoRT),
Objective-C++ (Metal), or Rust with CUDA kernels. Java uses JDK 25's
foreign function interface over the C header. Swift wraps the same
header. The gRPC server speaks the Open Inference Protocol so it runs
under KServe. None of these know anything the header does not say.

**Rules.** These are short because every one of them was broken last
time by being long.

1. Nothing is called working until it has run on the real hardware.
   There is no fake device in the normal path. A test that cannot run on
   the current machine is skipped and says so; it never passes.
2. A performance claim is a number from a named machine at a named
   commit, measured against the vendor's fastest program at a pinned
   version, on the same inputs. The programs and their versions are
   listed in the tree. If the comparison is not fair, there is no
   number.
3. A stage runs where its data is. Every copy between host and device is
   counted, and the count is returned with the result.
4. The model's settings live in the bundle, never in code.
5. If the hardware cannot do what was asked, the call fails and says
   what it cannot do. Nothing is substituted, defaulted or clamped.
6. Tokens are the same on every machine. One tokenizer, in the core.
7. No Python in the tree. A vendor's Python tool runs inside a pinned
   container, driven from Rust.
8. No machine names, user names or paths from our machines in anything
   committed. Machines are named by what they are: `rtx4080`, `b70`,
   `intel-npu`, `orin-nano`, `pi5-hailo8`, `pi5-hailo10h`, `m2`.
9. Add only what the header needs. A file that exists to wrap another
   file is deleted.

**What is here now.** The header, this file, the licence. The
previous attempt, for reading. The vendor stacks and the fastest known
programs for each machine are checked out at pinned versions under a
reference directory outside the tree; the list with commits is in
`ai-slop-generated-shit/docs/reference-code.md`, and what each vendor's
layers offer per pipeline stage is in
`ai-slop-generated-shit/docs/hardware-layers.md`. Those two are the
research this restart stands on and move here once the first backend
uses them.

**Order of work.** The header is cut to embedding. Next: the Rust core
against it with the CUDA backend on the RTX 4080, measured against
TensorRT and the fastest known embedding server on the same inputs.
Then the same feature on the B70, the M2, the two Pis and the Jetson,
so the hard parts show up before any second task is added. Bindings and
the server after the core holds on every machine.

## Licence

Apache-2.0. See `LICENSE`. The CUDA backend carries a subset of NVIDIA's
CUTLASS headers under `core/cuda/cutlass/`, BSD-3-Clause, with their
licence in `core/cuda/cutlass/LICENSE.txt`.
