# Test data

- `all-minilm-l6-v2/tokenizer.json`: the upstream tokenizer file of
  sentence-transformers/all-MiniLM-L6-v2 at commit
  `c9745ed1d9f207416be6d2e6f8de32d1f16199bf`, unchanged: 466247 bytes,
  SHA-256 `be50c3628f2bf5bb5e3a7f17b1f74611b2561a3a27eeab05e5aa30f411572037`.
  Taken from the previous attempt's `testdata/bundles/minilm-tokenizer/`.
- `bge-m3/tokenizer.json`: the upstream tokenizer file of BAAI/bge-m3 at
  commit `5617a9f61b028005a4858fdac845db406aefb181` (17098108 bytes, SHA-256
  `21106b6d7dab2952c1d496fb21d5dc9db75c28ed361a05f5020bbba27810dd08`) cut
  down for the tests: of its 250002 Unigram pieces it keeps, in the
  file's order, the first 16384, every piece of one character, and
  `<mask>`, 28233 in all, with ids renumbered to that order (`<mask>` is
  28232; the added token and the post-processor's ids follow); the
  normalizer with its precompiled character map, the pre-tokenizer, the
  post-processor, the decoder, `unk_id` and `byte_fallback` are unchanged.
  Read as JSON it equals the upstream file except in `model.vocab` and
  those ids. 1152616 bytes, SHA-256
  `c71be2d7663b73c2ab5d00585dfa25e34f43d1d2da983592ef1153c32e28a26b`,
  written with compact separators and no ASCII escaping. The core's
  Unigram is compared with upstream `tokenizers` on it
  (`core/tests/tokenizer.rs`), on the parity texts and the corners of
  the SentencePiece pipeline; a full-vocabulary check is what every
  BGE-M3 bundle's reference cases give on load. Under MIT, the licence of
  BAAI/bge-m3.
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
  from that model: its manifest, its tokenizer file, the weights
  `tiny_bert.py` wrote (`weights/model.safetensors`, 4049552 bytes,
  SHA-256 `c258bffc3c37afc5a8b12324c0d29b81d31738fbf72183ade9c52c0d68e06180`)
  and the reference file above. The core's tests check numerics against
  it: `core/tests/conformance.rs` compares every reference case with the
  upstream vectors (docs/conformance.md), and `core/tests/sessions.rs`
  compares the encoder's options with the same arithmetic written
  plainly in f64.
  Its `tokenizer.json` is not the upstream file above byte for byte: it
  is the MiniLM tokenizer as transformers wrote it again when
  `tiny_bert.py` saved the model (`save_pretrained`), 711661 bytes,
  SHA-256 `da0e79933b9ed51798a3ae27893d3c5fa4a201126cef75586296df9b4d2c62a0`.
  Read as JSON it equals the upstream file except that
  `truncation.direction` is present, as `"Right"`. Like the upstream
  file it is under Apache-2.0, the licence of
  sentence-transformers/all-MiniLM-L6-v2.

- `tiny-static-recipe/`: the recipe `turbo-bundle distill` made the
  next bundle from: a 16-wide static model distilled from
  `tiny-bert-bundle/`, with the quality texts of
  `bundle/recipes/static-quality.jsonl` (docs/static.md).
- `tiny-static-bundle/`: that static bundle, sealed: its table
  (`weights/static.safetensors`, F16, 1037908 bytes, SHA-256
  `1e3014311b0931a58b4fe9a56a8fb3e064ae81f938362edea39ed19d3d566f16`),
  the tokenizer file of `tiny-bert-bundle/`, the quality texts, and a
  reference written by Model2Vec 0.9.0's `StaticModel`
  (`bundle/reference/static_reference.py`) in a virtual environment with
  the versions the manifest names, not in the pinned container: it is
  test data. Its base model has random weights, so its quality numbers
  say nothing about a real model. `core/tests/static_model.rs` checks
  the static path against it.

Everything here is about 8.2 MB, 4 MB of it the tiny
BERT bundle's weights, 1.8 MB the static bundle and 1.1 MB the cut-down
BGE-M3 tokenizer.
