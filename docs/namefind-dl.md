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
   parts two words. A chunk of more pieces than a row holds is read in
   rows overlapping by half a row; each piece takes its scores from the
   row where it sits furthest from a cut edge (the middle half of each
   row), and the chunk is decoded as one run, the special tokens between
   rows left out.
3. The token classification head scores every position of the row, the
   special tokens too. A position's label is the first highest score
   that is not NaN, and its probability that label's softmax
   probability, with OpenNLP's handling of infinities
   (`NameFinderDL.labelProbability`).
4. By default (`Decoding::Word`) each word takes its first piece's
   label and probability, as the model was trained, and covers all its
   pieces. The words are the tokenizer's own, as BERT's basic
   pre-tokenization cuts them (`turbo::tokenizer::pre_words`): parted by
   whitespace, with each punctuation character and each CJK ideograph a
   word of its own. So `(Paris)` is `(`, `Paris`, `)`, a name before a
   full stop or comma ends before it, and unspaced ideographs are read
   one by one. A `B-<type>` word starts an entity that runs over the
   `I-<type>` words after it; its probability is the lowest of theirs.
   `Decoding::Piece` reads every position this way instead, the special
   tokens too, as `NameFinderDL` does; the OpenNLP parity test runs in
   that mode.
5. The candidates of the whole document are resolved as
   `mergeOverlappingSpans` resolves them: longest first, in UTF-16 units
   as Java measures a span, then most probable, each kept unless it
   overlaps one kept before it.

An entity's token range is the tokens its bytes overlap, by their
fragments, in any order the token layer holds them; when those tokens
are not consecutive in the layer, the step fails rather than name the
wrong ones.

Where it differs from `NameFinderDL`, by design:

- Words are labelled by their first piece by default; `NameFinderDL`
  labels every piece (`Decoding::Piece`).
- An entity is the bytes its pieces cover. `NameFinderDL` rebuilds the
  entity's text from its pieces and searches the chunk's text for it,
  which drops an entity whose pieces do not spell the text (an `[UNK]`
  piece, say).
- A chunk with more pieces than a row holds is read in overlapping rows
  and decoded as one run, so an entity across a row boundary is whole.
  `NameFinderDL` hands the whole chunk to ONNX Runtime, which fails past
  the model's 512 positions.
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
bundle) runs the finder with `Decoding::Piece` and checks that it gives every entity the same
bytes and type and a probability within 1e-5 of OpenNLP's. On the CPU
backend all 116 entities match, the largest probability difference
being 3.9e-6.

`analysis/namefind-dl/tests/offline.rs` runs the finder on the small BERT in `testdata`
with a head that labels every piece `B-PER`, so the result is known
without a reference: every piece is found once however the text is
chunked, rows are split or runs are batched, a sentence layer limits
what is read, and each entity maps to its token. By words, each word is
one entity, whole, even when its pieces fall in different rows.
`analysis/namefind-dl/tests/rows.rs` (ignored; set `TURBO_NER_BUNDLE`)
checks on `dslim/bert-base-NER` that rows of 10 to 23 pieces, which cut
every name of more than one piece somewhere, give the entities one row
gives.

## CoNLL-2003

`analysis/namefind-dl/tests/conll.rs` (ignored; set `TURBO_NER_BUNDLE`
to the bundle and `TURBO_CONLL2003_TEST` to a local copy of the English
test set, `eng.testb`) scores the finder as OpenNLP scores a name
finder: the samples of `Conll03NameSampleStream`
(`turbo_formats::conll::Conll03Reader`) and the counts of `FMeasure`, a
name right when its tokens and type both are. Each sentence is one
text, its tokens joined by single spaces as `NameFinderDL.find` joins
them. A name covers the tokens its bytes overlap, and names with the
same tokens and type count once: when the model labels two pieces of
one word `B-` each, both names cover that word.

On `dslim/bert-base-NER` with the CPU backend, the 3,453 test sentences
hold 5,648 names. With the default, each of the tokenizer's words
labelled by its first piece, the finder gives 5,945, of which 5,189 are
right:

| Type | Precision | Recall | F1 |
| --- | --- | --- | --- |
| all | 0.8728 | 0.9187 | 0.8952 |
| person | 0.8362 | 0.9567 | 0.8924 |
| organization | 0.8893 | 0.9091 | 0.8991 |
| location | 0.9281 | 0.9287 | 0.9284 |
| misc | 0.8008 | 0.8305 | 0.8154 |

The test set's tokens are joined by spaces, so whitespace words there are
its tokens; labelling those instead scored 0.9157, but on running text
it kept punctuation in a name (`Macron.`) and lost a name behind a
bracket or quote. The words the model was trained on part a token such
as `U.N.` or `1.` at each full stop, and a `B-` the model gives one of
those parts is a name of its own, which costs precision here.

The model card reports a test F1 of 0.913 from its own evaluation, which
scores its own word-level labels rather than names over these tokens.

With `Decoding::Piece` (`TURBO_NER_DECODING=piece`), as `NameFinderDL`
decodes, a `B-` the model gives a piece inside a word starts a name
there; the finder gives 6,360 names, of which 5,188 are right:

| Type | Precision | Recall | F1 |
| --- | --- | --- | --- |
| all | 0.8157 | 0.9186 | 0.8641 |
| person | 0.6976 | 0.9573 | 0.8071 |
| organization | 0.8770 | 0.9097 | 0.8930 |
| location | 0.9214 | 0.9281 | 0.9247 |
| misc | 0.7873 | 0.8276 | 0.8069 |
