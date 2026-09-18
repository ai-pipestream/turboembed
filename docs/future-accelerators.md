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
- **Hailo-10H (Raspberry Pi AI HAT+ 2, 8 GB).** Edge NPU on the HailoRT
  stack (compiled HEF models), attached to a Raspberry Pi host. A distinct
  runtime and toolchain from all of the above — neither OpenVINO nor an
  ONNX Runtime EP drives it — so it would be its own provider with its own
  ARM host build and qualification hardware.

Clarification to avoid a false "AMD is covered" reading: Machine B pairs an
**AMD CPU** with an **Intel Battlemage dGPU** driven by the OpenVINO GPU
plugin. That proves nothing about AMD accelerators — no ROCm, no XDNA.
