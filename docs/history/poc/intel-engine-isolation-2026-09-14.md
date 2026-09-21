# Intel engine isolation, 2026-09-14

The repaired native tokenizer and option validation passed bounded Intel
execution checks on the [recorded reference host and MiniLM bundle](intel-native-baseline-2026-09-14.md#environment-and-source):
Intel Battlemage G31, OpenVINO 2026.3.1, Linux x86_64.

Source was `78a88d8` plus the isolation test introduced with this receipt and
removal of the unused `genai.cpp` workspace-vocabulary fallback helper. The test
file SHA-256 was
`257d7e8fb4364128b84796587b1baef4f571c6f1601b414e181e6b2111d2c97e`.
The isolated hardware checkout was `/work/bench/turboembed-baseline-20260914`;
its existing model files and the inherited `/work/inferstream` checkout were
not changed.

```bash
source /work/opt/openvino_genai/setupvars.sh
cargo test --locked -p turboembed --features genai \
  --test intel_engine_isolation -- --nocapture
cargo test --locked -p turboembed --features genai \
  --test intel_genai_gpu minilm_c_abi_embed_one_on_ -- --nocapture
```

Each command ran under a 180-second timeout. Results:

- Isolation/option tests: 2 passed, 0 failed, 0 ignored, 0.47 seconds.
- Existing C ABI model checks: 2 passed, 0 failed, 0 ignored, 4 filtered,
  0.53 seconds. These exercised GPU and explicit CPU and compared `hello world`
  with the unchanged Intel golden at the existing cosine floor of 0.99.

The isolation test creates two GPU engines and runs eight requests on each
concurrently. It keeps an earlier result alive while later requests execute,
checks that the retained values do not change, releases the first engine and
its last result, and runs another request on the survivor. It also verifies
that results remain valid after dropping their Rust engine owner. The separate
option test checks unsupported truncation and normalization, then executes a
valid request on the same engine. No receipt-writing tests or golden updates
were run.

Logs on the Intel host:
`/work/bench/turboembed-intel-isolation.log` and
`/work/bench/turboembed-intel-tokenizer-smoke.log`.

The CUDA two-engine allocator test also passed again after tokenizer changes:
ORT 1.28.0, NVIDIA RTX 4080 SUPER, 1 passed in 0.64 seconds. Its scope remains
that of the [allocator receipt](cuda-allocator-isolation.md); this rerun is
recorded in `/tmp/turboembed-tokenizer-cuda-isolation.log` on the development host.

These checks establish the exercised ownership and option behavior. They do
not establish prepared GPU-resident output, transfer counts, native/ABI timing,
SDK installation, FFM, or Apple validation. The existing Intel text path still
uses host-accessible hidden output and CPU pooling.
