# Quality consumers

The quality consumers depend on TextQuality/SpanQuality (#1272, issue #1208)
and the corrected span joins in #1251. Their temporary prerequisites branch
combines those two exact heads; it contains no consumer implementation. The
consumer PR targets that branch so its diff remains focused. Retarget it to
main after both prerequisite PRs land. This does not claim completion of the
remaining #1198 projection work.

## Conditional steps and execution reports

`quality_gate(view, min_score, forbidden)` checks the default TextQuality's
view, composite floor and forbidden bits. `lexical_quality_gate` checks density
and MTLD floors. The `_on` forms accept a named quality key. Direct predicate
evaluation returns false on missing quality, a mismatched view or invalid
thresholds. `ConditionalStep::from_predicate` automatically declares the quality
key as a required input and includes the rule, view, thresholds and flags in
the configuration hash. Existing closure constructors retain their behavior.

```rust,ignore
use turbo_analysis::{Bound, ConditionalStep, ViewId};
use turbo_analysis_layers::{QualityFlags, quality_gate};

let gated_step = ConditionalStep::from_predicate(
    "clean-enough",
    quality_gate(ViewId::SOURCE, 0.6,
        QualityFlags::MOJIBAKE_SUSPECT | QualityFlags::NAV_BOILERPLATE),
    Bound::new(expensive_annotator),
);
```

Add TextQualityAnnotator to the same Analyzer, or declare the appropriate
quality key with `given` when quality arrives on input documents.
`Analyzer::run_with_report` runs once and returns each document's ordinary
result alongside an ExecutionReport. It lists output keys and their actual
Complete, Skipped or Failed statuses in pipeline stage order. None explicitly
means no output record. It does not infer timings. Integration tests execute
real rule sentence-detector steps behind both gates and check skipped output,
named dependencies, configuration identity and failed-input propagation.

## Filtering chunks and retrieval candidates

`QualityChunkFilter::new(min_score, forbidden)` reads Chunks, SpanQuality and
Sentences. Bind its output to a different Chunks instance:

```rust,ignore
Bound::new(QualityChunkFilter::new(0.5, QualityFlags::NAV_BOILERPLATE)?)
    .write(0, "clean")
```

A chunk is rejected if any nonempty actual fragment overlaps a failing quality
row. Fragment gaps do not count. Uncovered text and empty chunks remain eligible;
missing coverage is not evidence of poor quality. Every retained column stays
aligned. Parent indices are remapped to the nearest retained ancestor, or None
if every ancestor was dropped. Sentence ranges continue to refer to the input
Sentences instance; bind input slot 2 when that instance is named.

`Cascade::retain_by_quality` applies the same rule before either stage-1
shortlist consumes its budget and before contextual reranking. Repeated calls
intersect eligibility; an invalid call leaves existing eligibility unchanged.
The quality mask uses `Spans::join_overlapping` and prefix counts of bad rows,
so each returned overlap range takes constant time without enumerating all
pairs. Arbitrarily ordered query fragments use stable radix sorting. Total work
and memory are O(F + S + C), for F fragments, S quality rows and C candidates.

## Fractional content term counts

`ContentTermCounter::<L, B>::new().with_quality()` adds SpanQuality as a required
input and writes DensityTermStats<L>. The ordinary counter continues to write
the original integer TermStats<L>. A token in boilerplate contributes zero;
otherwise its quality row's information_density is its contribution. Region
membership follows the existing term-counter policy: the token's start must
fall in the region. Uncovered tokens contribute one. Zero-density terms do not
enter tf, effective length or df. Fractional weights are never rounded away.

DensityTermStats has floating tf and effective lengths, integer df for positive
rows, and standard graph codecs at View, Sentences, Structure, Chunks and Levels
scope. Import validates sparse columns, vocabulary bounds and exact alignment.
`DensityCollection::new(stats, vocabulary)` computes N, mean effective length and
df without counting empty effective units. Its online mean avoids overflow when
large finite lengths are averaged. Use the existing Query constructor with
this collection, then `stats.score(&query, unit)` to retain fractions in BM25.
Single-unit scoring validates only that row rather than rescanning every unit.
The score uses scaled division to avoid overflowing a finite denominator.

The Cascade's existing lexical input remains its original TermStats-based
space; density scoring is an explicit separate path. Tests cover actual
indexing/counter pipelines, 0/0.25/0.5/1 contributions, boilerplate exclusion,
effective df and collection length, large finite values, BM25 results, byte-exact
graph roundtrips and malformed fractional counts. Contextual-stage control-flow
tests use deterministic vector inputs and make no claim of trained model
accuracy or hardware speed.
