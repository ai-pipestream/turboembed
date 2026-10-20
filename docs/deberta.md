# Native DeBERTa encoder foundation

Part of #391, #714 and #715. The Rust `turbo::deberta` API executes the
shared-key DeBERTa-v3 encoder used by GLiNER2. It returns hidden states for
extraction heads without Python, ONNX or an external inference engine.

This first CPU implementation establishes the equations and the checkpoint
boundary. It does not add a C session task, tokenizer/prompt construction,
GLiNER span heads, GLiREL relation heads, GLiNER2 schema decoding, SaT,
quantization, GPU execution or a throughput acceptance claim. Those issues
stay open. In particular, the separate relation-layer work in #784 does not
complete the neural portion of #714.

## Supported computation

- Word embeddings, embedding LayerNorm and input masking.
- Shared query/key projections of LayerNorm-normalized relative embeddings.
- Content-to-content, content-to-position and position-to-content attention,
  with signed logarithmic distance buckets and the common `sqrt(3 * head_dim)`
  scale. The position-to-content gather is transposed as in Transformers.
- Stable masked softmax, value aggregation, residual LayerNorm, GELU feedforward
  and output residual LayerNorm for each layer. Dropout is disabled at inference.
- Explicit token and prompt position selection through checked immutable linear
  projections. Both use the same encoder output; padding positions are errors.
  Callers supply model-specific heads. These projections are not substitutes
  for a checkpoint's nonlinear span or schema heads.

The encoder uses bounded per-call workspace and immutable model weights.
`encode` retains only the final output; `encode_layers` also retains embedding
and intermediate outputs for parity tests. Attention uses one score row at a
time. The current linear execution is scalar CPU code with existing native
GELU; packed GEMM, batching and device backends remain later work.

## Configuration and package contract

`core/src/deberta/config.rs` defines the supported profile and named limits.
The converter accepts F32 safetensors for `deberta-v2` configuration with
`share_att_key=true`, both `c2p` and `p2c`, relative embedding LayerNorm,
GELU, no absolute input positions, and no type embeddings. It rejects
convolution, separate positional keys, differing embedding width, unsupported
attention-head width and recurrent `z_steps`. It does not reinterpret such
checkpoints as BERT or discard unsupported architecture settings.

Each required tensor must have the exact shape and finite values. Configurations,
token IDs (including padding IDs), mask lengths and context limits are checked
before indexing. Empty/all-masked sequences are errors. Hidden values for masked
query rows follow upstream behavior, including uniform attention when the whole
query row is masked; projection selection still excludes padding.

`bundle/src/deberta/mod.rs` converts tensor names to a canonical sorted F32
safetensors file. Prefix removal is explicit: `encoder.` for GLiNER2, `-` for
an unprefixed DebertaV2Model checkpoint. Conversion copies only encoder weights.
Other task heads remain in the source checkpoint and can be loaded explicitly
through `TokenPromptProjections::from_safetensors` when they are linear.

The package contains `deberta.json` and `weights.safetensors`. Metadata includes
format version, validated configuration, source repository, exact 40-character
revision, licence and weight SHA-256. It is a separate encoder package for the
Rust API, not an existing libturbo session manifest. The loader verifies the
digest before constructing an encoder. Metadata is limited to 64 KiB, weights
to 2 GiB. Conversion requires a new output directory and never overwrites an
existing package. A write failure leaves an incomplete directory; loading it
fails until the complete manifest and matching weights are present.

```
turbo-bundle deberta-convert encoder_config.json model.safetensors encoder. fastino/gliner2-base-v1 f9634218e53580c56edf0de97ca1a7d3f1c2354e Apache-2.0 native-encoder
turbo-bundle deberta-verify native-encoder
```

The caller obtains token IDs and masks, loads `PackageManifest::load`, calls
`Encoder::encode`, and passes visible positions to the token/prompt heads.
Future extraction tasks can share that single encoder result.

## Independent numerical reference

The fixture in `core/tests/fixtures/deberta` is extracted from
[fastino/gliner2-base-v1 at f9634218](https://huggingface.co/fastino/gliner2-base-v1/tree/f9634218e53580c56edf0de97ca1a7d3f1c2354e).
Its model card declares Apache-2.0. The fixture is modified from those weights:
2 layers, hidden width 32, 2 heads, intermediate width 48, 16 vocabulary rows
(original IDs 1000..1015), and 16 relative rows (original rows 248..263).
Other tensor dimensions use prefix slices. No random weight remains after
strict state-dictionary loading. This is a reduced numerical fixture, not a
usable GLiNER checkpoint or evidence of extraction quality.

The reference is Hugging Face Transformers 4.57.1 `DebertaV2Model` in evaluation
mode with CPU PyTorch. `provenance.json` records the immutable source revision,
original tensor shapes, row offsets and SHA-256 of fetched byte ranges.
`bundle/reference/deberta/build_fixture.py` reproduces the slices and expected
outputs; its adjacent requirements file pins the reference environment.
Install the CPU PyTorch wheel from its official CPU index, then the other
requirements, and run the script with a new output directory. Only fixture
production uses Python. Tests and inference are native Rust and offline.

Five sequences cover singleton input, bidirectional positions, padding,
noncontiguous masks and distances beyond the linear bucket region. Tests compare
embedding output, each encoder layer and linear token/prompt outputs. Current
maximum encoder absolute error is below 4e-7; tests allow 2e-5 for CPU numerical
variation. This does not replace full-width, full-depth checkpoint validation
before shipping an extraction model. The next integration should add that
record plus tokenizer/prompt, span and schema-head parity.

Run:

```
cargo test --locked -p turbo --test deberta -- --nocapture
cargo test --locked -p turbo-bundle --test deberta
```

Boundary tests reject invalid dimensions, missing/truncated tensors, non-finite
weights, digest changes, unsupported profiles, bad token IDs, masks and projection
positions. Conversion is deterministic, package reload preserves encoder output,
and existing output directories remain untouched.
