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
  `bundle/recipes/all-minilm-l6-v2.json`, with `max_seq` 64, on a
  randomly initialised two-layer BERT (hidden 32, four heads, seed 0)
  saved by sentence-transformers 6.1.0 with mean pooling and L2
  normalization, and the tokenizer above. The query prefix was
  `"query: "`. Its vectors are real outputs of the upstream pipeline for
  that model, not of MiniLM; the bundle tool's tests use it to check
  sealing and verification, never numerics.
