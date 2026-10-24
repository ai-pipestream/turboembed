# Lexical cohesion and discourse boundaries

`LexicalBoundaries` and `DiscourseBoundaries` write separate named
`Boundaries` instances. `WeighTextBoundaries` adds either to a base signal.
Both weights default to zero. No existing pipeline changes its cuts.

## Signals

Lexical cohesion reads `View`, `Sentences`, `Tokens` and the term analyzer's
`Terms`. Choose `LexicalForm::Folded` (default), `Stem` or `Lemma`; a missing
column is an error, with no fallback. Tokens must be disjoint, ordered,
contained in one sentence, and in the same view. Invalid UTF-8 offsets,
column lengths and token-to-sentence indexes are errors. Punctuation-only
forms do not contribute. There is no additional stop list or IDF.

The cosine compares term-frequency bags in adjacent full sentence windows.
Counts move incrementally with the windows. Empty windows are unscored.
The cosine is rounded to millionths and uses the same integer smoothing,
TextTiling depth and cost conversion as embedding drift. Defaults are a
three-sentence window, smoothing radius one, depth scoring, floor 10,000,
scale 2,000 and TopicShift threshold 100,000. End gaps without full windows
cost zero. As in embedding drift, smoothing runs over scored gaps in order.

Discourse markers lower the gap before a matching sentence. The inventories
below were authored in this repository under Apache-2.0, without importing a
third-party lexicon. Language selection is explicit:

| Language | Markers |
| --- | --- |
| English | however; in contrast; on the other hand; nevertheless; meanwhile; first; second; finally; in conclusion; to summarize |
| Spanish | sin embargo; en cambio; por otro lado; no obstante; mientras tanto; primero; segundo; finalmente; en conclusión; en resumen |
| Dutch | echter; daarentegen; aan de andere kant; niettemin; ondertussen; ten eerste; ten tweede; ten slotte; tot slot; samenvattend |

Matching uses full Unicode case folding and the shared Unicode White_Space
predicate. Initial and repeated whitespace are allowed; a marker must end
before a non-word character or at the sentence end. Letters, marks, numbers
and connector punctuation prevent a partial-word match. Quotes are not
skipped and accents are not removed. These connectives can also occur
within one topic, so a match is only a soft cost. A zero discount produces
neither a cost nor a TopicShift reason.

The sentence anchor is retained. Neither step rewrites text or offsets.
Tests project derived-view sentence spans to source and back, including
length-changing case folding. Configuration hashes include all scoring
settings, the selected term column, and the actual marker inventory.

## Pipeline use

Given sentences and tokens, use the existing term analyzer and bind:

```rust,ignore
builder
    .add(TermAnnotator::for_language("en").unwrap())
    .step(Bound::new(StructureBoundaries { costs: Costs::default() })
        .write(0, STRUCTURE))
    .step(LexicalBoundaries::default().bound())
    .step(DiscourseBoundaries {
        language: DiscourseLanguage::English,
        discount: 1000,
    }.bound())
    .step(WeighTextBoundaries {
        base: Weight::ONE,
        lexical: Weight::ONE,
        discourse: Weight(0),
    }.bound())
    .add(Chunker { budget });
```

The base input can instead be another named, already weighed signal:
`weigher.bound().read(1, "combined")`. Missing lexical or
discourse inputs are allowed only at weight zero. The existing integer
weighted-sum routine combines costs and preserves stronger structural
reasons. This is a typed pipeline API; no new CLI preset is enabled.

## Qasper test split ablation

[Qasper v0.3](https://github.com/allenai/qasper) supplies the real paper
texts, section layout, questions and evidence. Its data licence is CC BY
4.0. The [aggregate receipt](receipts/chunker-950/qasper.json) records input
hashes, configuration hashes, counts and all retrieval cutoffs.

All 416 test papers and 1,451 questions were used, without parameter tuning.
There are 76,342 detected sentences, 1,506,101 word tokens and 5,663 reference
section boundaries. A pinned English OpenNLP sentence model runs separately
on each paragraph or heading. Abstracts are separate reference segments.
These section starts are a **silver boundary reference**, not independently
annotated topic changes. There were 2,713 sentence-initial marker matches.

The optimizer uses a 254-word maximum, 200-word target, size weight 1,000
and zero overhead. These are actual word-token counts, not simulated
subword counts. It reads plain-text structure costs and no cohesion term;
this experiment uses the public optimizer without structural hard-limit
runs. The lexical defaults above and a discourse discount of 1,000 are
fixed across arms. Each enabled signal has weight 1,000.

Retrieval uses the production corpus-wide BM25 posting index, English
terms, k1=1.2 and b=0.75. It uses no embedding model. All arms use the same
questions, evidence and retrieval settings. Evidence metrics average over
1,310 questions with placed evidence; the reader reports 21 unplaced
annotations. Pk, WindowDiff and boundary F1 are macro means over papers.

| Signals | Chunks | Pk (lower) | WindowDiff (lower) | Boundary F1 | Document nDCG@10 | Document Recall@10 | Evidence byte recall@5 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Base | 7,386 | 0.480092 | 0.503606 | 0.186644 | 0.381822 | 0.481737 | 0.169252 |
| Base + lexical | 8,087 | 0.440093 | 0.470934 | 0.229918 | 0.382699 | 0.487250 | 0.172889 |
| Base + discourse | 7,535 | 0.501456 | 0.523331 | 0.139056 | 0.374744 | 0.483804 | 0.166524 |
| Base + both | 8,252 | 0.456371 | 0.488679 | 0.199936 | 0.381518 | 0.487250 | 0.170125 |

Lexical cohesion changed cuts in 415 papers, discourse in 384, and both in
415. Lexical improved agreement with the section reference, but the
retrieval difference is small and evidence recall@1 fell from 0.056793 to
0.050495. Discourse worsened the boundary metrics and nDCG at this weight.
These observations do not establish a general retrieval gain or validate
Spanish/Dutch quality. Both signals remain opt-in. No significance claim
or dense-retrieval claim is made.

## Reproduce

Fetch the Qasper test archive from the URL in
`analysis/eval/src/sets/qasper.rs`, and the English UD EWT sentence 1.3 model
identified in the receipt. The example validates both SHA-256 digests.
It writes all per-paper cuts and per-query scores locally; it does not
write corpus text to the report. The committed receipt retains aggregates.

```sh
cargo run --locked --release -p turbo-eval --example text_signals -- \
  "$QASPER_ARCHIVE" "$SENTENCE_MODEL" qasper-signals.json
```

The recorded `segmentation_ms` times only weighing and optimization after
signal computation, not end-to-end analysis or retrieval. It is not a
performance comparison between signals.
