# Future accelerators — noted, not started

Status: **planned after the Intel NPU proof**
([intel-cloud-npu-runbook.md](intel-cloud-npu-runbook.md)). Nothing on this
page is wired, validated, or scheduled; it exists so these targets are not
forgotten. Each would be a new provider behind the existing backend
boundary, subject to the same fail-loud device policy, goldens, and
receipt discipline as the current NVIDIA / Intel / Apple paths.

- **Intel Gaudi (Habana).** Intel *datacenter* accelerator on the
  SynapseAI/Gaudi software stack. It is **not** served by the OpenVINO NPU
  plugin, so the Core Ultra NPU work does not carry over; Gaudi needs its
  own backend and its own qualification host. Kristian's decision: OpenVINO
  NPU on a Core Ultra host first, Gaudi later.
- **AMD ROCm / Instinct.** Datacenter GPU peer of the Machine A CUDA path.
  The plausible entry points are the ONNX Runtime ROCm/MIGraphX execution
  providers or a native HIP backend; either way it is a new provider with
  its own driver stack and qualification hardware.
- **AMD Ryzen AI / XDNA.** *Client* NPU peer of the Intel Core Ultra NPU,
  but a different execution provider entirely (Ryzen AI SW / Vitis AI EP,
  XDNA driver) — the OpenVINO NPU plugin does not drive it.
- **Raspberry Pi AI HAT+ 2** (not the earlier AI HAT / AI HAT+): official
  Raspberry Pi product (~$200) carrying a **Hailo-10H NPU with 8 GB of
  dedicated on-board LPDDR** on the HAT itself. Stack is HailoRT with
  compiled HEF models — a separate provider from OpenVINO NPU, Gaudi, and
  ROCm; neither OpenVINO nor an ONNX Runtime EP drives it, and it needs its
  own ARM host build. Status: parked until after the Intel Cloud OpenVINO
  NPU proof; attractive later as a cheap edge Machine for live receipts.

Clarification to avoid a false "AMD is covered" reading: Machine B pairs an
**AMD CPU** with an **Intel Battlemage dGPU** driven by the OpenVINO GPU
plugin. That proves nothing about AMD accelerators — no ROCm, no XDNA.

## CUDA Rust (NVIDIA kernel frontends)

Status: **documented interest only** — no dependency, no build change, not
started. Unlike the entries above, this is not a new provider or execution
path; it is a possible future way to *author* the small custom device
kernels we already ship on the NVIDIA path.

In September 2026 NVIDIA announced [CUDA Rust](https://developer.nvidia.com/blog/introducing-cuda-rust-two-tracks-for-writing-gpu-kernels/),
two tracks for writing GPU kernels natively in Rust:

- **`cuda-oxide`** — the SIMT track. A custom `rustc` codegen backend that
  compiles `#[kernel]` functions to PTX. Early alpha; requires a pinned
  nightly toolchain, clang/libclang, and CUDA 12.x+.
- **`cutile-rs`** — the Tile track. Kernels are JIT-compiled through CUDA
  Tile IR; the compiler owns thread mapping and memory layout. Further
  along: published on crates.io, runs on **stable Rust 1.89+** with CUDA
  13.3 and no custom LLVM, and is already used outside NVIDIA (Hugging
  Face's Grout inference engine, mistral.rs).

What it is and is not for us:

- Both tracks are for **writing GPU kernels in Rust**. Neither replaces
  the ONNX Runtime CUDA / TensorRT execution providers; the MiniLM/BGE
  forward pass stays on ORT/TensorRT with IoBinding, exactly as validated
  on Machine A.
- The plausible TurboEmbed use is an optional future spike on **our own
  device kernels** — for example the CUDA pooling kernel
  ([`native/turboembed/src/pool_cuda.cu`](../native/turboembed/src/pool_cuda.cu))
  or a TurboRerank on-device score path — behind an experimental gate,
  compared against the existing `.cu` implementation with the same goldens
  and receipts.
- If we do a first spike, prefer **`cutile-rs`** (stable Rust, no nightly,
  no custom LLVM); drop to `cuda-oxide` only if the kernel needs full SIMT
  control over threads and shared memory.

Constraints, per NVIDIA's own framing: Linux only, compute capability
≥ 8.0 (Machine A's RTX 4080 SUPER qualifies), APIs will move, and neither
track is production-ready today.
