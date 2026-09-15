# Intel remote-buffer probe, 2026-09-14

The [standalone direct OpenVINO probe](../native/turboembed/bench/README.md)
passed on the Intel Battlemage G31 reference host with OpenVINO 2026.3.1 and the
[pinned MiniLM bundle](intel-native-baseline-2026-09-14.md#environment-and-source).

The probe reshapes the model to batch 1, sequence 32, adds masked mean pooling
and L2 normalization before compilation, and wraps program-owned OpenCL buffers
as remote inputs and the final embedding output. OpenVINO uses the shared
in-order command queue. A following OpenCL kernel doubles the embedding into
another GPU buffer; only that consumed result is read back. It is compared,
after dividing by two, with the same graph run on explicit CPU.

Observed result:

```text
gpu=Intel(R) Graphics [0xe223] (dGPU)
max_abs_error=1.93715e-07
rmse=6.01531e-08
host_input_bytes=384
explicit_readback_bytes=1536
PASS: remote inputs, GPU pooled output, downstream OpenCL consumer, explicit readback
```

The numerical gates were set before execution: maximum absolute error at most
`5e-4` and RMSE at most `1e-4`. Remote output identity was also checked against
the caller-created OpenCL buffer. No GPU embedding readback occurred before the
second kernel. The byte counts are the probe's explicit transfers; they do not
measure provider-internal traffic or prove zero copies or zero allocations.
This single case is not the roadmap's workload/performance baseline.

Build/run used CMake Release and GCC 15.2.0 in
`/work/bench/turboembed-openvino-probe-clean`. OpenCL headers were extracted from
Ubuntu packages into `/work/bench/turboembed-opencl-headers-20260914/root`:

- `opencl-c-headers` `3.0~2025.07.22-2build1`, package SHA-256
  `82d0947da094a2fb8b961caccaab18c9ca203eb8916efbeb09ced6367ab70e85`.
- `opencl-clhpp-headers` `3.0~2025.07.22-1ubuntu2`, package SHA-256
  `8cee41130fc5a5a6923a89ed8402163230855be3d8d3a63729b2364f2fc723bd`.

The existing loader was `/usr/lib/x86_64-linux-gnu/libOpenCL.so.1`. No system
packages were installed. The final clean build linked `openvino::runtime` and `OpenCL::OpenCL`.
`setupvars.sh` supplied OpenVINO and its bundled TBB library discovery during
both build and execution. That environment dependency
must be handled by SDK packaging before a clean-consumer install can pass.
The run was bounded by a 90-second timeout. Intel-host logs:
`/work/bench/turboembed-openvino-remote-probe-build.log` and
`/work/bench/turboembed-openvino-remote-probe.log`.

During development, direct linking and the first CMake build failed to resolve
TBB in an unsourced environment; an initial run without the OpenVINO environment
could not load TBB. Explicit TBB linking was tested and then removed after a
clean build succeeded with the documented environment. The
first graph attempt also attached pooling after a Result node and failed CPU
compilation. The final graph uses the Result input, and the final build/runtime
commands above passed. These setup failures did not produce validation results.

The [prepared ABI design](prepared-abi-design.md) uses this verified memory path
as its Intel starting point. Versioned native handles, lifetime enforcement,
prepared/text API unification, full numerical/workload coverage, matched timing,
SDK packaging, and FFM remain to be implemented and verified.
