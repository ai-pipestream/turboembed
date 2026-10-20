# Keyword and keyphrase extraction

`turbo-keywords` extracts ranked keywords and multi-word keyphrases
from one document with zero external model dependencies: YAKE from five
statistical features, TextRank from co-occurrence graph centrality. One
document, no model, no corpus, no download, any language with a stop
word list. Both read the layers every pipeline already has
(`opennlp:sentences`, `opennlp:tokens`, `turbo:terms`, and for TextRank
`opennlp:pos`) and write one shared layer, `turbo:keywords`.

The algorithms follow the papers. YAKE is Campos et al., "YAKE! Keyword
extraction from single documents using multiple local features",
Information Sciences 509 (2020) 257-289; the default parameter values
follow Spark NLP's `YakeParams` (Apache-2.0), and the design notes
deliberately do not copy that implementation's scoring defects. TextRank
is Mihalcea and Tarau, "TextRank: Bringing order into texts", EMNLP
2004.

## Crate

| Crate | Path | Contents |
|---|---|---|
| `turbo-keywords` | `analysis/keywords/` | The `Keywords` layer, the YAKE extractor and annotator, the TextRank extractor and annotator, the dedup and stopword seams, the document graph codec |

It depends on `turbo-analysis`, `turbo-analysis-layers` and
`turbo-stopword` (the bundled lists), nothing else.

## The layer

`turbo:keywords` is positional: one item per occurrence, so a phrase
that appears twice is two items at two spans carrying the same phrase
and score. Items are in document order and may overlap, because
deduplication removes near-duplicates, not every phrase that shares a
word with another ("machine learning" and "learning competitions" can
both survive). Columns beside the span:

- `phrase`: the normalized (folded) keyphrase, terms joined by one
  space.
- `score`: the salience in `[0, 1]`, higher more salient.
- `algorithm`: which extractor produced the item (`yake` or
  `textrank`).
- `terms`: how many terms the keyphrase holds.

One column needs one direction, so the score is always higher-is-better:
TextRank's PageRank centrality already lies in `[0, 1]`, and YAKE's raw
score (`S(kw)`, lower better) is normalized across the survivors so the
best scores 1 and the worst 0. A caller ranking by the layer sorts by
`score` descending; the raw YAKE score stays available on the
extractor's candidates for anyone who wants the paper's own scale.

Both annotators write the same layer type. A pipeline running both binds
them to two instance names, exactly as a gold and a predicted layer are
two instances:

```rust
let analyzer = Analyzer::builder()
    .given(Key::<Sentences>::DEFAULT)
    .given(Key::<Tokens>::DEFAULT)
    .given(Key::<PosTags>::DEFAULT)
    .given(Key::<Terms>::DEFAULT)
    .step(Bound::new(YakeAnnotator::new(Stopwords::bundled(StopwordLanguage::English))))
    .step(Bound::new(TextRankAnnotator::new()).write(0, "textrank"))
    .build()?;
```

## YAKE

Five passes, no regex, following the paper:

1. **Occurrence fold.** One merge-walk of tokens against sentences
   assigns each token its sentence index and in-sentence position, and
   folds the term table (through `turbo:terms`, so term identity is the
   pipeline's own folded form): frequency, the sentence index of every
   occurrence, and acronym or mid-sentence-capitalized counts from a
   code-point tag classifier (`d` number, `u` unknown, `a` acronym,
   `n` mid-sentence capital, `p` plain).
2. **Co-occurrence.** Per sentence, the terms in the window of
   `window_size` tokens to each side, stop words included: they are
   context, only candidates get excluded.
3. **Term scores** over the candidate vocabulary (content-tagged terms
   that are not stop words; the frequency moments are taken over that
   vocabulary, not over every token, where stop word frequencies would
   flatten `TFNorm`):

   - `TCase(t) = max(aCount, nCount) / (1 + ln TF(t))`
   - `TPosition(t) = ln(ln(3 + median(sentence indices)))`, the double
     logarithm of the paper
   - `TFNorm(t) = TF(t) / (meanTF + stdTF)` (population standard
     deviation), `TF(t)` when the denominator is zero
   - `TSentence(t) = distinct sentence indices / sentence count`
   - `DL` and `DR`: distinct window neighbors over total, each side
   - `TRel(t) = 1 + (DL + DR) * TF(t) / maxTF`
   - `TScore(t) = TPosition * TRel / (TCase + TFNorm/TRel + TSentence/TRel)`
4. **Candidates.** Every `min_n..=max_n` token window per sentence whose
   tokens are all content-tagged and whose first and last terms are not
   stop words, keyed by their token list (never a joined string, so a
   token holding a comma is never re-split). Each is scored
   `S(kw) = prod(S(w)) / (KF * (1 + sum(S(w))))`, lower better, where an
   interior stop word contributes the bigram probability
   `rightCo(prev)[s]/TF(prev) * rightCo(s)[next]/TF(s)` and a candidate
   whose denominator does not hold together is dropped rather than
   ranked on a NaN.
5. **Deduplication.** Sort ascending by score, ties by earlier first
   occurrence; drop a candidate contained in a survivor, containing one,
   or whose similarity with a survivor reaches the threshold; cut to
   `top_n`; normalize the raw scores into the layer's salience.

The stopword filter is configuration, not a layer:
`Stopwords::bundled(StopwordLanguage::English)` is one of OpenNLP's
eleven bundled lists, `Stopwords::custom(&["..."])` the caller's own
folded forms. Defaults: window 3, n-grams 1..=3, top 30, no score cut,
token-overlap dedup at 0.8.

## TextRank

Content tokens (`NOUN`, `PROPN`, `ADJ`) become graph vertices, one per
folded term; two vertices gain a unit of edge weight for every window
(of `window`, default 2) they share within a sentence, never a
self-loop. PageRank iterates
`PR(v) = (1 - d) + d * sum_u w(u,v)/out(u) * PR(u)` with `d = 0.85`
until the L1 delta is below `1e-4` (at most 100 iterations), and the
scores normalize to centralities in `[0, 1]`. The top `top_n` vertices
are marked; in each sentence, maximal runs of adjacent marked content
tokens collapse into one keyphrase scored by the sum of its members'
centralities, with the exact span of its first and last token.
Adjacency is in the token stream: a token the POS filter passed over
breaks a run, and a single vertex with no neighbor still converges to
`1 - d`.

## The document graph codec

`turbo_keywords::export` registers the layer's codec on a `Codecs` set
(`register_codecs` adds it to an existing set, `codecs()` returns the
standard codecs plus this one). The payload is the span columns, the
algorithm tag set, and one `(phrase, score, algorithm tag, term count)`
row per item. Import is strict, like every codec's: a truncated body,
trailing bytes, a row count that disagrees with the spans, a tag outside
the tag set, a score outside `[0, 1]`, an empty phrase, a term count of
zero, or a span that does not slice the view text on UTF-8 boundaries is
refused with a typed `ExportError`, never clamped. Round-trips are
byte-identical: a document exported twice through one codec set gives
the same bytes, so the record a search index holds is exactly the record
the pipeline wrote.

## Tests

No mocks anywhere; the extractors run on real text with hand-built
layers, and the literals are the values of an independent reference
computation. `analysis/keywords/src/yake/features.rs` pins every term
feature of a three-sentence fixture against values worked from the
paper's formulas to 1e-9, including a window-size test the reference
implementation cannot pass (it declares the window and never reads it).
`analysis/keywords/tests/` runs both annotators end to end: the
hand-verified ranking, the design's worked example, exact occurrence
spans for repeated phrases, comma tokens, interior stop words,
deduplication of nested candidates, empty and single-token and
stopword-only documents, and the codec's round-trip and its refusals.
