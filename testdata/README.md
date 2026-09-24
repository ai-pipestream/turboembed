# Test data

- `all-minilm-l6-v2/tokenizer.json`: the upstream tokenizer file of
  sentence-transformers/all-MiniLM-L6-v2 at commit
  `c9745ed1d9f207416be6d2e6f8de32d1f16199bf`, unchanged: 466247 bytes,
  SHA-256 `be50c3628f2bf5bb5e3a7f17b1f74611b2561a3a27eeab05e5aa30f411572037`.
  Taken from the previous attempt's `testdata/bundles/minilm-tokenizer/`.
- `tokenizer-texts.jsonl`: texts the core's tokenizer is compared with
  upstream `tokenizers` on: scripts, spacing, control characters, emoji,
  special-token strings. Taken from the previous attempt's
  `testdata/corpus/multilingual.jsonl`, text only.
- `tiny-bert-reference/reference.safetensors`: the reference file
  `bundle/reference/reference.py` wrote for the cases of
  `bundle/recipes/all-minilm-l6-v2.json` with `max_seq` 64 and the query
  prefix `"query: "`, on the small BERT `bundle/reference/tiny_bert.py`
  writes from the tokenizer above. `produced_by.json` is what the script
  reported for that run. Running both scripts again with the pinned
  versions gave the same bytes. Its vectors are real outputs of the
  upstream pipeline for that model, not of MiniLM; the bundle tool's
  tests use it to check sealing and verification.
- `tiny-bert-bundle/`: the bundle `turbo-bundle` sealed and verified
  from that model: its manifest, the tokenizer file, the weights
  `tiny_bert.py` wrote (`weights/model.safetensors`, 4049552 bytes,
  SHA-256 `c258bffc3c37afc5a8b12324c0d29b81d31738fbf72183ade9c52c0d68e06180`)
  and the reference file above. The core's tests check numerics against
  it: `core/tests/conformance.rs` compares every reference case with the
  upstream vectors (docs/conformance.md), and `core/tests/sessions.rs`
  compares the encoder's options with the same arithmetic written
  plainly in f64.
