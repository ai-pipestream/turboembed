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
- `tiny-bpe/tokenizer.json`: a byte-level BPE in RoBERTa's form, 1200
  pieces and 939 merges, that upstream `tokenizers` 0.23.2 trained on the
  texts the tokenizer tests encode, with GPT-2's byte alphabet, RoBERTa's
  special tokens (`<mask>` taking the whitespace before it),
  pre-tokenizer, post-processor and decoder (`bpe::make_tiny_bpe` in
  `core/tests/tokenizer.rs`, which writes the same bytes again). 28901
  bytes, SHA-256
  `ea6c2eeefe26149a84e65d6694da4838d0051667e9e8cd6fe0d97e6d205d771e`. The
  core's byte-level BPE is compared with upstream on it, on those texts
  and on texts drawn at random from their characters, so that merges
  stop part way.
- `tiny-bpe-split/tokenizer.json`: the same in Llama-3's form, 1200
  pieces, that upstream `tokenizers` 0.23.2 trained on
  `tiny-bpe/corpus.jsonl` with Llama-3's pre-tokenizer (a Split with its
  pattern, then a ByteLevel that splits nothing), GPT-2's byte alphabet
  and RoBERTa's special tokens, post-processor and decoder
  (`make_tiny_bpe_split` in `core/tests/bpe_split.rs`, which writes the
  same bytes again). 29373 bytes, SHA-256
  `d4c7c9dcccc43c4ba18dc80435cc5c2560f01d63746263547d89a1fd599e6b90`. The
  core is compared with upstream on it with Llama-3's pattern, and with
  Qwen2's pattern and NFC normalizer written into it.
- `tiny-bpe/corpus.jsonl`: the 232 texts that tokenizer was trained on
  (`bpe::corpus` in `core/tests/tokenizer.rs`, the texts the tests
  encode, twice over), one JSON string per line, written by
  `bpe::write_corpus_file`; `bpe::the_corpus_file_is_the_corpus` checks
  it. `analysis/bpe-trainer` trains on it and writes the tokenizer file
  again byte for byte (docs/bpe-trainer.md).
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

- `tiny-sat-recipe/`: the recipe of a small SaT model (wtpsplit's
  `SubwordXLMForTokenClassification`, XLM-R with two layers of width 32,
  four heads and one label) with random weights, which
  `analysis/sat/reference/sat_reference.py tiny` writes over the
  tokenizer file of `bge-m3/` above; the bundle's `FAMILY_ROBERTA`
  encoder is its `roberta.` tensors, and the classifier
  (`classifier.weight`, `classifier.bias`) is in the same file, where
  `turbo-sat` reads it. Its reference cases are the bge-m3 recipe's,
  with no prompt.
- `tiny-sat-bundle/`: that bundle, sealed: the weights
  (`weights/model.safetensors`, 3695516 bytes, SHA-256
  `5eb6e13e665dd248b60afbc5594fa60130ce6de3d488802072335f2d9400b1c4`),
  the tokenizer file of `bge-m3/` unchanged, and the reference file the
  `reference` command wrote (the mean of wtpsplit's encoder states over
  each case). `analysis/sat/README.md` says how both were made and
  checked; `analysis/sat/tests` compare the SaT step with wtpsplit on
  it. Under MIT, the licence of wtpsplit and of BAAI/bge-m3's tokenizer.

- `tiny-static-recipe/`: the recipe `turbo-bundle distill` made the
  next bundle from: a 16-wide static model distilled from
  `tiny-bert-bundle/`, with the quality texts of
  `bundle/recipes/static-quality.jsonl` (docs/static.md).
- `tiny-static-bundle/`: that static bundle, sealed: its table
  (`weights/static.safetensors`, F16, 1037908 bytes, SHA-256
  `1e3014311b0931a58b4fe9a56a8fb3e064ae81f938362edea39ed19d3d566f16`),
  the tokenizer file of `tiny-bert-bundle/`, the quality texts, and the
  reference the tool wrote. Its ids and vectors are byte for byte the
  ones Model2Vec's `StaticModel` gives for the same table and
  tokenizer. Its base model has random weights, so its quality numbers
  say nothing about a real model. `core/tests/static_model.rs` checks
  the static path against it.

- `sentencepiece/`: SentencePiece `.model` files and what Google's
  sentencepiece library (the Python package 0.2.2, with protobuf
  7.36.1) gives for them, which `core/tests/sentencepiece.rs` and
  `analysis/embed/tests/subwords.rs` compare the core with.
  `goldens.py` wrote all of it (its docstring has the commands):
  - `t5.model`: T5's model, `cc_all.32000/sentencepiece.model` of the
    public `t5-data` bucket
    (https://storage.googleapis.com/t5-data/vocabs/cc_all.32000/sentencepiece.model),
    unchanged: 791656 bytes, SHA-256
    `d60acb128cf7b7f2536e8f38a5b18a05535c9e14c7a355904270e15b0945ea86`.
    Unigram with the `nmt_nfkc` character map. Under Apache-2.0, the
    licence of T5.
  - `unigram-identity`, `unigram-bytes`, `unigram-suffix`,
    `unigram-rules`, `bpe-identity`, `bpe-bytes` (`.model`, 7 to 12 KB
    each): small models `goldens.py train` trained on
    `tokenizer-texts.jsonl`, the texts of `goldens.py` and `botchan.txt`
    of the sentencepiece repository's `data/` (tag v0.2.2, commit
    `e0cce7d37b065b5140349dbe12c6bcf6192fdd78`; a public domain novel,
    not kept here), to cover what T5 does not: user-defined, control and
    UNUSED pieces, byte fallback, whitespace as a suffix, spaces kept, a
    custom character map with rules that share a prefix and a deletion,
    whitespace not escaped, and a denormalizer. Training twice gives the
    same bytes. Made for this repository, under its licence.
  - `<model>.goldens.jsonl`: for each text, the normalized text, its
    `norm_to_orig`, the ids, each piece's offsets and the decoded text,
    and the pieces of one long text by count and SHA-256; the header
    line names the model and the versions. `xlmr.goldens.jsonl` is for
    XLM-RoBERTa's model as `goldens.py` and the test rebuild it from
    `bge-m3/tokenizer.json` (cut down, so not the published file).
    `mt5`, `umt5` and `llama2.goldens.jsonl` are for published models
    that are not kept here: mT5's and umT5's models from the same
    bucket (Apache-2.0) and Llama 2's from karpathy/llama2.c (Meta's
    Llama 2 licence). Their addresses and SHA-512 are in
    `core/tests/sentencepiece.rs`, which checks them when
    `TURBO_SENTENCEPIECE_MODELS` names a directory holding them as
    `mt5.model`, `umt5.model` and `llama2.model`.
  - `variants.goldens.jsonl`: for each edit of `goldens.py`'s
    `VARIANTS` to a committed model (its whitespace switches set
    otherwise, or user-defined pieces appended), the SHA-256 of what
    the library gives on every text, one line per text as the
    docstring of `goldens.py` writes it. `core/tests/sentencepiece.rs`
    makes the same edits.

Everything here is about 9.8 MB, 4 MB of it the tiny
BERT bundle's weights, 2 MB the SentencePiece models and goldens,
1.8 MB the static bundle and 1.1 MB the cut-down BGE-M3 tokenizer.
