# Optional CoNLL-2003 comparison

The `conll_eval` example compares the native name finder and an upstream
prediction file against the same local gold tokens. It writes aggregate counts,
per-kind precision/recall/F1, prediction agreement and content hashes. A model
card's published F1 is not a paired comparison and is not an input to this tool.

## Inputs and permission

Under the decision recorded in [#1599](https://github.com/ai-pipestream/turboembed/issues/1599#issuecomment-6468295141):

- The user obtains Reuters RCV1 text under the Reuters agreement and prepares a
  local English CoNLL-2003 `eng.testb`. This repository supplies a format reader,
  not the Reuters text. Obtain the data through the
  [NIST distribution procedure](https://trec.nist.gov/data/reuters/reuters.html).
- The user may download `dslim/bert-base-NER` after reading and accepting the
  model and dataset terms. The recipe pins source commit
  `d1a3e8f13f8c3566299d95fcfc9a8d2382a9affc` and file digests. The model card's MIT
  declaration does not grant rights to redistribute Reuters text.
- Neither these weights nor the corpus is bundled, packaged, installed by
  default, or used to train a model shipped by this project. Keep the local
  bundle, corpus and upstream predictions outside version control.

The existing bundle fetch command requires explicit acceptance for this
recipe's notice. Run the following only after accepting the applicable terms;
replace the relative directory names with private locations chosen by the user:

```sh
cargo run --release -p turbo-bundle -- fetch --accept-terms \
  bundle/recipes/dslim-bert-base-ner.json local-input/dslim
```

The fetcher reuses cached files only when their pinned hashes match. It records
the acceptance and hashes in the download directory. Without `--accept-terms`,
the recipe's notice stops the command before download. The evaluator itself
does not fetch anything and does not assert acceptance on the user's behalf.

Seal and verify a local native token-classification bundle using the existing
bundle tooling and the accepted upstream files:

```sh
cargo run --release -p turbo-bundle -- reference \
  bundle/recipes/dslim-bert-base-ner.json local-input/dslim local-input/dslim-bundle
cargo run --release -p turbo-bundle -- verify local-input/dslim-bundle
```

The reference step needs the recipe's pinned reference image. If that image is
unavailable, bundle preparation remains incomplete: do not substitute an
unpinned image or present another model's results as dslim parity. This command
does not install Python on the user's host. Changes to reference images and
bundle conversion belong to the bundle implementation, not this evaluator.

## Upstream predictions

Use an upstream inference implementation with the same checkpoint commit and
the exact gold token sequence. Save predictions as a private UTF-8 file with
four space-separated CoNLL columns: token, POS placeholder, chunk placeholder,
and predicted IOB tag. Empty lines separate samples. Keep optional DOCSTART
markers in CoNLL-2003 form. This is a prediction file, not a fifth column added
to the gold file. The accepted types are PER, ORG, LOC and MISC, plus O; IOB1
and IOB2 are interpreted by the existing CoNLL-2003 reader.

Record the upstream engine/version, inference command, tokenizer revision,
long-input handling and decoding choices with the result. The command checks
that the supplied upstream commit matches the native bundle's source commit,
but cannot authenticate how an external prediction file was produced. The
prediction file hash identifies the exact bytes compared. Project-authored
test predictions exercise the scorer and are never evidence of dslim parity.

## Run and interpret

```sh
git rev-parse HEAD
cargo run --release -p turbo-namefind-dl --example conll_eval -- \
  local-input/dslim-bundle local-input/eng.testb local-input/upstream.conll \
  d1a3e8f13f8c3566299d95fcfc9a8d2382a9affc word
```

Use `piece` in place of `word` for the separate piece-decoding comparison;
record that choice and ensure the upstream predictions use the intended policy.
The native run explicitly uses CPU Exact precision and two CPU threads. It
processes at most 16 samples per inference submission. Input files are read
into memory, but text is not copied to a scratch file and no predictions are
dumped. Standard output contains only aggregate results and hashes. Errors
from the corpus reader omit source excerpts.

Sentence counts and every token must align before inference. Gold entities
match predictions only when token start, token end and type all match.
Native byte spans project to the tokens they overlap, with repeated identical
token/type spans counted once, matching the existing optional test protocol.
Per-kind counts include empty kinds; precision/recall/F1 are zero when the
relevant denominator is zero. The receipt reports:

- Native and upstream accuracy against gold, both overall and by entity kind.
- F1 difference, which can be zero even when predictions differ.
- Entity agreement and the number of samples with different entity sets.
- `parity=exact_entity_agreement` only if every entity set matches; otherwise
  `parity=different_entities`.
- Native manifest, gold-file and upstream-file SHA-256 digests, source commit,
  decoding, device and precision. Keep the repository commit printed above
  with the receipt as the implementation revision.

Exit zero means a valid comparison completed, not that parity or an accuracy
gate passed. Do not turn it into a CI acceptance gate without an agreed
threshold. No actual dslim/Reuters paired result is claimed by this change.

## Existing opt-in test and native regression tests

The existing ignored integration test and its accuracy floors are unchanged:

```sh
TURBO_NER_BUNDLE=local-input/dslim-bundle \
TURBO_CONLL2003_TEST=local-input/eng.testb \
cargo test --release -p turbo-namefind-dl --test conll \
  conll2003_english_test_set -- --ignored --exact --nocapture
```

It requires the user's accepted model and data. Its historical measurements
are accuracy evidence, not a fresh upstream comparison.

The example's ordinary native tests need neither input:

```sh
cargo test -p turbo-namefind-dl --example conll_eval
```

They exercise the real reader, span projection, exact scorer and actual CPU
inference with the repository's generated tiny BERT fixture. That fixture is
for deterministic software tests, not a trained NER checkpoint or peer result.
