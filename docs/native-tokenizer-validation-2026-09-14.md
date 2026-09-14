# Native tokenizer validation, 2026-09-14

The [native BERT contract](native-tokenizer-contract.md) now gates model
configuration before selecting WordPiece. The Intel embedding provider requires
its bundle's tokenizer; it no longer substitutes a vocabulary from a different
model. ORT retains its existing Hugging Face fallback for unsupported native
configurations. TurboRerank's model path also requires tokenizer metadata;
explicit low-level TXT loads retain the uncased BERT preset.

The five initial reference regressions failed against the previous code:
Unicode substitutions changed IDs, BPE configuration was accepted as WordPiece,
malformed UTF-8 was accepted, and output truncation returned internal errors for
single sequences and pairs. All five pass after the repair. Additional tests
cover count-only calls, added-token configuration, i64 rows and stride canaries,
duplicate JSON keys, missing special tokens, and malformed TXT vocabulary bytes.
No reference vectors were changed.

## Local checks

- `cargo test --locked --workspace`: 299 passed, 0 failed, 9 ignored, before the
  final two tokenizer regressions were added. Four reranker-model conditional
  skips remain included in the pass count. Log:
  `/tmp/turboembed-tokenizer-workspace-final.log`.
- Final targeted `wordpiece_contract`: 10 passed, 0 failed, 1 ignored. The
  ignored case requires an explicitly supplied actual MiniLM tokenizer.
  Log: `/tmp/turboembed-tokenizer-contract-final.log`.
- Default workspace Clippy with `-D warnings` and workspace formatting passed.
  Existing `nvcc` compiler-bindir build warnings remain.
- Standalone native C++ ASan/UBSan tests: 51 assertions passed, 0 failed; one
  conditional MiniLM vocabulary case was skipped. No sanitizer diagnostics.
- Native-instrumented Rust test executables: 3 option tests and the then-present
  8 tokenizer tests passed. The actual MiniLM corpus passed separately under the
  same instrumentation. The last two i64/duplicate-key regressions were added
  afterward and have ordinary test coverage. Log:
  `/tmp/turboembed-wordpiece-options-sanitizers.log`.

The sanitizer environment is the one documented in the
[ownership receipt](native-ownership-validation-2026-09-14.md#native-asan-and-ubsan),
with `--test wordpiece_contract --test options`. C++ objects contain ASan and
UBSan instrumentation. Rust code and the Rust standard library are not compiler
sanitizer-instrumented. The test binaries link the sanitizer runtimes for native
objects. No model downloads were performed.

## Actual model tokenizer comparison

Model: `sentence-transformers/all-MiniLM-L6-v2`, revision
`1110a243fdf4706b3f48f1d95db1a4f5529b4d41`.
Tokenizer SHA-256:
`be50c3628f2bf5bb5e3a7f17b1f74611b2561a3a27eeab05e5aa30f411572037`.
Reference: locked Rust `tokenizers` 0.23.2.

```bash
TURBOEMBED_TOKENIZER_JSON=/path/to/pinned/tokenizer.json \
  cargo test --locked -p turboembed --test wordpiece_contract \
  actual_minilm_unicode_corpus_matches_reference -- --ignored --exact
```

The test disables reference padding/truncation and compares exact token IDs.
It covers named Unicode/special-token examples and every Unicode scalar between
ASCII `a` and `b`, grouped into bounded requests. The ordinary run passed in
10.61 seconds; the native-instrumented run passed in 10.68 seconds. These are
correctness test durations, not tokenizer throughput measurements. This corpus
does not exhaust every possible sequence of Unicode characters.

Native Unicode decomposition/lowercasing comes from pinned utf8proc. Category
predicates match the reference dependency's tables and are reproducibly generated.
Sources, licenses, and hashes are in [third_party](../third_party/README.md).
Neither dependency adds a shared runtime requirement. JSON parsing occurs at
load; inference uses the frozen vocabulary and bounded stack scratch.

This receipt does not establish GPU numerical parity, native performance,
packaging, FFM, or Apple validation. Those remain separate delivery gates.
