# Transformer name finder

`turbo-namefind-dl` (`analysis/namefind-dl`) finds names with a BERT
token-classification model, the way OpenNLP's `NameFinderDL` does, on
turboembed's own encoder instead of ONNX Runtime. `NameFinderDl` is an
analysis step: it reads the text, its `Tokens` and, when present, its
`Sentences`, and writes `opennlp:entities`.

## What it computes

For each sentence (the whole text when there is no sentence layer):

1. The sentence is cut at Unicode whitespace into words, and the words
   into chunks of `split` words (250), consecutive chunks sharing
   `overlap` words (50), as `AbstractDL.chunkRanges` cuts them.
2. Each chunk's words are rejoined with single spaces, as `AbstractDL`
   rejoins them, tokenized by the bundle's WordPiece tokenizer and read
   as one row, `[CLS] pieces [SEP]`. A separator WordPiece would drop as
   a control character (form feed, line tabulation, next line) still
   parts two words.
3. The token classification head scores every position of the row, the
   special tokens too. A position's label is the first highest score
   that is not NaN, and its probability that label's softmax
   probability, with OpenNLP's handling of infinities
   (`NameFinderDL.labelProbability`).
4. A `B-<type>` position starts an entity that runs over the
   `I-<type>` positions after it; its probability is the lowest of
   theirs.
5. The candidates of the whole document are resolved as
   `mergeOverlappingSpans` resolves them: longest first, in UTF-16 units
   as Java measures a span, then most probable, each kept unless it
   overlaps one kept before it.

An entity's token range is the tokens its bytes overlap, by their
fragments, in any order the token layer holds them; when those tokens
are not consecutive in the layer, the step fails rather than name the
wrong ones.

Where it differs from `NameFinderDL`, by design:

- An entity is the bytes its pieces cover. `NameFinderDL` rebuilds the
  entity's text from its pieces and searches the chunk's text for it,
  which drops an entity whose pieces do not spell the text (an `[UNK]`
  piece, say).
- A chunk with more pieces than a row holds is read in consecutive rows.
  `NameFinderDL` hands the whole chunk to ONNX Runtime, which fails past
  the model's 512 positions. An entity across two rows ends at the
  first, since the next row starts on an `I-` piece.
- Sentences come from the document's `Sentences` layer rather than a
  `SentenceDetector`, and the text is not folded first (whitespace and
  dash folding are off by default in `InferenceOptions` too).

## The model

The finder runs any bundle whose classifier is `CLASSIFIER_TOKEN` with
`B-`/`I-` labels. `bundle/recipes/dslim-bert-base-ner.json` makes one
from `dslim/bert-base-NER`, the model OpenNLP's `NameFinderDL`
evaluation uses; its licence and class are in docs/licences.md. Given
the upstream files in a directory:

```
turbo-bundle reference bundle/recipes/dslim-bert-base-ner.json <upstream-dir> <bundle-dir>
```

The reference container writes the model's token logits next to its
vectors, and the bundle tool checks them against the labels of the
upstream `config.json`.

## Parity with OpenNLP

`analysis/namefind-dl/reference/NameFinderDlGolden.java` runs OpenNLP's `NameFinderDL` on
the model's ONNX export, with the default inference options and each
line of `analysis/namefind-dl/tests/data/jvm-cases.txt` as one sentence, and writes
`analysis/namefind-dl/tests/data/jvm-golden.json`. It was run against OpenNLP at
`b453b9ee3f2856a1dbf323ad13c8465a1203d3f0` with ONNX Runtime 1.29.0:

```
java -cp <opennlp-dl and its dependencies> NameFinderDlGolden.java \
    <model.onnx> <vocab.txt> tests/data/jvm-cases.txt tests/data/jvm-golden.json
```

The cases include accented and non-Latin text, an emoji, a text of 620
words read in three chunks, and an entity across the first chunk
boundary. `analysis/namefind-dl/tests/jvm.rs` (ignored; set `TURBO_NER_BUNDLE` to the
bundle) checks that the native finder gives every entity the same
bytes and type and a probability within 1e-5 of OpenNLP's. On the CPU
backend all 116 entities match, the largest probability difference
being 3.9e-6.

`analysis/namefind-dl/tests/offline.rs` runs the finder on the small BERT in `testdata`
with a head that labels every piece `B-PER`, so the result is known
without a reference: every piece is found once however the text is
chunked, rows are split or runs are batched, a sentence layer limits
what is read, and each entity maps to its token.
