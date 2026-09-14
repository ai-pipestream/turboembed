# Direct OpenVINO remote-buffer probe

This standalone C++ program checks the first prepared execution path on an
Intel GPU. It reads a MiniLM OpenVINO bundle, constructs the prepared IDs for
`hello world`, and compares GPU output with an explicit CPU reference.

The graph includes masked mean pooling and L2 normalization. Three input buffers
and the final `[1, 384]` output use OpenCL remote tensors. A second OpenCL kernel
consumes the embedding on the shared queue before an explicit host readback.
The probe checks buffer identity, finite values, maximum absolute error
(`5e-4`), and RMSE (`1e-4`). Its sequence length is 32.

Prerequisites: a C++17 compiler, CMake 3.20+, an Intel GPU with its OpenCL runtime,
OpenCL C/C++ headers and loader development library, and the OpenVINO runtime SDK
with its matching TBB package. The verified environment is OpenVINO 2026.3.1 on
Linux x86_64. Set `OPENVINO_ROOT` to that SDK installation, then run from the
repository root:

```bash
source "$OPENVINO_ROOT/setupvars.sh"
cmake -S native/turboembed/bench -B build/openvino-probe \
  -DCMAKE_BUILD_TYPE=Release \
  -DOpenVINO_DIR="$OPENVINO_ROOT/runtime/cmake"
cmake --build build/openvino-probe -j2
build/openvino-probe/openvino_remote_probe /path/to/minilm/bundle
```

For separately extracted OpenCL headers or a nonstandard loader location,
supply `OpenCL_INCLUDE_DIR` and `OpenCL_LIBRARY` to CMake. The probe performs no
model download and does not modify bundles, fixtures, or receipts.

This is a correctness probe, not the finished SDK or a performance benchmark.
It does not measure allocation counts, provider-internal copies, or native/ABI
call overhead. The reported transfer bytes count the program's explicit OpenCL
commands only. The measured result and exact environment are recorded in the
[remote-buffer receipt](../../../docs/intel-remote-buffer-probe-2026-09-14.md).
