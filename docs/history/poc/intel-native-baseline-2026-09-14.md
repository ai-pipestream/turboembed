# Intel native baseline preliminary receipt — 2026-09-14

This is a single-case GPU smoke receipt, not a performance baseline. It does
not establish prepared GPU-resident output, transfer behavior, allocation
behavior, latency, throughput, or native/ABI/binding overhead. The full pilot
and its acceptance criteria remain those in [the roadmap](../ROADMAP.md).

## Environment and source

- Host: an x86_64 host with an Intel Arc B70 (Battlemage), Linux
  `7.0.0-31-generic`.
- GPU: Intel Battlemage G31 (`8086:e223`), using the `xe` kernel driver.
- Runtime: `/work/opt/openvino_genai`, a symlink to
  `/work/opt/openvino_genai_ubuntu26_2026.3.1.0_x86_64`; runtime version
  `2026.3.1-22476-56d9685302d-releases/2026/3`.
- Toolchains: Rust/Cargo 1.98.1, CMake 4.2.3, GCC/G++ 15.2.0, and clang++
  21.1.8. Cargo was supplied by `$HOME/.cargo/bin`.
- Source: isolated clone `/work/bench/turboembed-baseline-20260914`, commit
  `d476e8ce3027ed0409cb27a4b35c86188f524a2e`, branch
  `native-sdk-foundations`. The source checkout was clean at capture time.
- Model directory: the clone's `models/ov` is a symlink to the existing
  `/work/inferstream/models/ov`; the inherited checkout and its model bundle
  were not modified.

MiniLM bundle hashes:

| File | SHA-256 |
|---|---|
| `openvino_model.xml` | `f87dd1482b2a745f8c699b81ddd9cbcad666a193be4693abcea44b7ac8c67c1e` |
| `openvino_model.bin` | `8b86cab4722e2aefab310cf96d4d5a9eb3b187f7d9670a082afc55c7fa0d392a` |
| `tokenizer.json` | `be50c3628f2bf5bb5e3a7f17b1f74611b2561a3a27eeab05e5aa30f411572037` |

The bundle contains `config.json`, `tokenizer.json`, and
`tokenizer_config.json`, but no bundle-local provenance, manifest, license, or
README file. Reproducible SDK packaging must add explicit bundle provenance and
license metadata.

## Command and result

The command sourced the OpenVINO environment and ran under a 600-second bound:

```bash
source /work/opt/openvino_genai/setupvars.sh
cargo test --locked -p turboembed --features genai --test intel_genai_gpu \
  minilm_c_abi_embed_one_on_gpu -- --exact --nocapture
```

`/work/bench/turboembed-intel-initial-gpu.log` records a 2.82-second build and
a 0.27-second test run: 1 passed, 0 failed, 0 ignored, 5 filtered out.

The exact test creates an engine through the C ABI with
`TURBOEMBED_DEVICE_OPENVINO_GPU`, loads the `minilm` alias, and calls
`turboembed_embed_one` for `"hello world"`. It verifies successful GPU engine
creation and model load, a non-null result/value pointer, `dim == 384`,
`count == 1`, and cosine similarity to the Intel MiniLM golden vector at or
above the test constant `COSINE_FLOOR = 0.99`. It then frees the result and
destroys the engine.

The result proves only that this one direct C-ABI MiniLM call passed on this
host and runtime. It is not evidence of the roadmap's prepared execution path
or a GPU-resident output contract.
