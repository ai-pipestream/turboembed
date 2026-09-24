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
