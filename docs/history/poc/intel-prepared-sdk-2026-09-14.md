# Intel prepared SDK validation, 2026-09-14

This receipt covers the additive prepared C API, model-bundle provisioning and
relocatable native packaging. It does not establish the roadmap's native-overhead
budget or qualify Rust/Java bindings. No artifact was published or deployed.

## Environment and model

- Host: `krick-1`, Linux x86_64, Ubuntu 26.04.1, kernel `7.0.0-31-generic`.
- GPU: Intel Battlemage G31, PCI `8086:e223`, reported by OpenVINO as
  `Intel(R) Graphics [0xe223] (dGPU)`.
- CPU reference: AMD Ryzen 9 9950X, explicitly selected.
- Compiler/toolchain: GCC 15.2, CMake 4.2.3, glibc 2.43.
- OpenVINO runtime and exporter both reported
  `2026.3.1-22476-759c5a6ab8c-releases/2026/3`. This is the observed build string
  for this SDK run; use it rather than a build string from an older receipt.
- Source: `sentence-transformers/all-MiniLM-L6-v2`, revision
  `1110a243fdf4706b3f48f1d95db1a4f5529b4d41`, self-contained `onnx/model.onnx`.
  The [pinned model card](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/blob/1110a243fdf4706b3f48f1d95db1a4f5529b4d41/README.md)
  declares Apache-2.0.

The exporter read that ONNX and serialized new OpenVINO IR. Existing historical
IR files were not relabeled with invented conversion provenance.

| Artifact | SHA-256 |
|---|---|
| Source ONNX | `6fd5d72fe4589f189f8ebc006442dbb529bb7ce38f8082112682524616046452` |
| Exported XML | `651149767db8997eac4375252f874776ec3ef762e3578b4e4ec9778c4dfa7ba4` |
| Exported BIN | `727b4613da4f7edff48f7bb5a849670025edfc344b80fde66c6a73a1b0493f4c` |
| Tokenizer JSON | `be50c3628f2bf5bb5e3a7f17b1f74611b2561a3a27eeab05e5aa30f411572037` |
| Config JSON | `953f9c0d463486b10a6871cc2fd59f223b2c70184f49815e7efbcab5d8908b41` |
| Model card | `dcd602d2fd35c203a247304a06fec6654a12f7941b739f9221a064fe8dc3b7f0` |

## Native contracts

The standalone shared library compiled and the
[contract executable](../native/turboembed/sdk/contract_test.cpp) passed on both
explicit CPU and Intel GPU. No hardware path was skipped. Tested shapes were
`[1,32]` and `[2,32]`, with 384-dimensional normalized output.

The numerical gates were maximum absolute error `5e-4` and RMSE `1e-4` against
the CPU graph. Release-build CPU/GPU comparison for prepared `hello world`
measured maximum absolute error `2.01166e-7`, RMSE `6.00436e-8`. Repeated GPU
executions and text/prepared agreement stayed within the tighter `1e-6` maximum
and `1e-7` RMSE gates used for those comparisons.

The test also exercised:

- Invalid descriptors, oversized shapes, invalid token IDs, insufficient output
  capacity and execution without valid inputs.
- Empty strings, a null zero-length span, embedded NUL, Unicode and invalid UTF-8.
- Independent concurrent execution slots and two-row output layout.
- A live result preventing writes/re-execution, then permitting reuse on release.
- A result remaining readable after slot, model and context handles are released.
- Corrupt artifacts, duplicate/unknown manifest fields, NUL metadata and missing
  bundles returning errors.
- A loaded model continuing to work after its on-disk tokenizer is changed.
- A downstream OpenCL kernel reading the leased GPU output on the borrowed queue.
  Result release waited for that work, and adapter readback counters did not
  increase. The test explicitly read the downstream kernel's separate output
  for comparison; this is not a claim of zero total test readback.

The ABI owns three i32 input tensors and one f32 result tensor per slot. GPU
slots also own equally sized host input staging. These tests do not measure
OpenVINO internal allocation/copy behavior or large-batch performance.

## Sanitizers

The SDK, native tokenizer and contract executable were rebuilt with
`-fsanitize=address,undefined -fno-omit-frame-pointer` and the corresponding
linker flags. The first execution stopped because oneTBB's affinity-library
loading used `RTLD_DEEPBIND`, which AddressSanitizer rejects.

The same executable passed its full CPU/GPU contracts with:

```bash
TBB_ENABLE_SANITIZERS=1 ASAN_OPTIONS=detect_leaks=0 UBSAN_OPTIONS=halt_on_error=1 \
  timeout 180 prepared_contract_test /path/to/minilm-bundle
```

The TBB setting is its
[documented sanitizer compatibility setting](https://uxlfoundation.github.io/oneTBB/main/intro/limitations.html).
No production code or affinity policy was changed to obtain the pass. Leak
detection was disabled; vendor runtime binaries were not rebuilt with sanitizer
instrumentation. This establishes neither leak freedom nor instrumented GPU
kernel coverage.

## Installation and relocation

The SDK was installed, packaged with selected OpenVINO/TBB runtime files and
licenses, and copied to another prefix. The installed C example was configured
as a separate project against that copied prefix. Configuration, compilation,
linking and execution used `env -i PATH=/usr/bin:/bin`.

Both GPU and CPU runs returned 384 dimensions and norm `1.000000`. `ldd` resolved
TurboEmbed, OpenVINO and TBB from the copied prefix. The OpenCL ICD loader,
standard C++ runtime and glibc resolved from the operating system. `nm -D`
showed only the 17 intended `turboembed_prepared_v1_*` functions and their ELF
version definition. GPU/CPU plugin loading succeeded without `setupvars.sh` or
`LD_LIBRARY_PATH` in the consumer environment.

An initial run command used `embed` instead of the target's actual name,
`turboembed_prepared_embed`, and failed before launching the program. The
corrected commands passed; no loader workaround was needed.

## Workspace and provisioning checks

- `cargo test --locked --workspace`: 301 passed, 0 failed, 9 ignored. Four
  existing conditional reranker-model tests are included in the pass count and
  return early without their external model artifacts, as in the prior baseline.
- Formatting and default-feature workspace Clippy passed. Existing `nvcc`
  compiler-bindir warnings remain build-script output, not Rust lint failures.
- Native tokenizer regression target: 10 passed, 1 model-dependent test ignored.
  Its first invocation used the nonexistent test target `wordpiece`; the correct
  target is `wordpiece_contract`.
- Model provisioning: 8 Python unit tests passed, covering complete publication,
  hashes, refusal to overwrite, symlink destinations, failed export cleanup,
  malformed identity/configuration and source changes during export.

The existing text ABI's Apple/Swift and NVIDIA provider qualification was not
rerun for this new Intel-only extension. Hosted CI, publication, a managed
adapter and matched native performance measurements remain unexecuted here.
