# Long-document evidence gate (#1486)

The four sets were selected for document structure before retrieval scores:
Qasper test, GutenQA, CUAD v1 test and ContractNLI test. Selection is frozen;
a failure is a result, not a reason to replace a set or change a threshold.
The native English rule sentencer and plain structure are used throughout.

## Recovered structural screen

The earlier screen used source `a5e5b079c6cbc06a1e4c16dd9413f4cdd13f7df3`,
all-MiniLM-L6-v2 manifest
`0f17adc4b795ed27bbe59ec88d35d37bf10f25af15f07b09f3249e5c1001b6d3`,
v1 max/target 254/200, structure costs, no cohesion, and sentence-windows-5.
It preceded the current clause-splitting configuration. These are all-document
cut statistics, not retrieval results, quality records, or oracle bounds.

| Set | Documents | v1 / windows total chunks | Different full cuts | Tokens p50 / p95 |
|---|---:|---:|---:|---:|
| Qasper | 416 | 10727 / 15405 | 416 | 4802 / 8976 |
| GutenQA | 100 | 62843 / 89505 | 100 | 105522 / 477345 |
| CUAD | 102 | 5070 / 5635 | 101 | 5111 / 31065 |
| ContractNLI | 123 | 1368 / 1282 | 122 | 1903 / 4773 |

CUAD and ContractNLI together took 5.04 seconds and 415628 KiB peak RSS in
that screen. This is not embedding throughput. The first Qasper/GutenQA run
timed out before its SQuAD control; there is no complete-run timing for it.
The private screen's projected document JSON was not the original dataset.

## Sources and adaptation

[CUAD](https://www.atticusprojectai.org/cuad/) (Hendrycks et al., 2021) and
[ContractNLI](https://stanfordnlp.github.io/contract-nli/) (Koreeda and Manning,
2021) publish CC BY 4.0 data. ContractNLI also supplies TERMS. Obtain the
archives from their official pages under those terms, then place each
original test.json in the named cache directory. The production readers
verify the following member hashes even for manually supplied files.
No automatic download or terms bypass is added.

| Source | SHA-256 |
|---|---|
| CUAD archive | `f8161d18bea4e9c05e78fa6dda61c19c846fb8087ea969c172753bc2f45b999a` |
| CUAD test member | `007b6a40b0c65247f881627375c3d2e9b6aeeb5dfa89957494feed0765a1a073` |
| ContractNLI archive | `e03fc77bbf8b53e2976a250e81d8a294bc3d5e5fb014521e477dee9340d6287b` |
| ContractNLI test member | `460267b56052a2dc5aead98eb35eadef9e6734d5723d37b4a9790e410f812387` |

Each full contract is a document. CUAD has one query per exact question
template, with grade-1 document and span judgements wherever answers exist.
Impossible questions supply no positive judgement. Template IDs retain the
source category; inconsistent text under one category is rejected. The test
split contains 102 documents, 41 templates, 1244 positive document-template
pairs and 2643 evidence spans.

ContractNLI has one query per hypothesis. Entailment and contradiction
annotations both supply evidence about the hypothesis; NotMentioned supplies
none. The 123 test documents yield 17 hypotheses, 1188 positive pairs and 2404
spans. This is corpus-wide evidence search, not the original contract-scoped
NLI classification task. No 2091-query independence claim is made. Both
readers preserve contract IDs, exact source text and code-point-to-byte span
conversion. Invalid offsets, missing hypotheses and inconsistent labels fail.
The Price Restrictions template has no positive CUAD test judgement, so 40
of the 41 templates are scored. These adaptations can have much lower recall@10 because one template judges
many contracts. They are not numerically comparable to published extraction
or NLI accuracy.

`cargo run -p turbo-eval --example contract_audit -- "$CACHE"` verifies the
members and prints corpus/query/evidence counts without embedding anything.

## Frozen quality protocol

The candidate is v1 with max/target 254/200, structure costs, no cohesion and
`--clauses`. Baselines are sentence-windows-5, fixed-tokens-200 with no
overlap, and the existing semantic splitter at its defaults. All use the
same all-MiniLM bundle, native rule sentences, plain structure, ignored
titles and hybrid retrieval: exact dense cosine plus BM25, weighted fusion,
BM25 weight 0.35, k1 1.2, b 0.75 and folded terms. Every quality record must
come from one clean pushed source commit, one device and one precision tier.
The record stores all resolved parameters and raw per-query scores.

The policy from #172 comment 6466746203, implemented by #1608, is unchanged:
on eligible evidence sets, the 95% paired interval lower bounds for recall@10
and contained@10 must exceed zero, and fewer judged spans must be split.
Eligibility uses the smaller share of judged documents with at most two
chunks under the candidate and sentence windows, and requires that share
below 0.5. At least one evidence set must be eligible. Every set's nDCG@10
interval upper bound against its best baseline must be at least -0.5 points.
The old +2 document-ranking gate remains a separate diagnostic.

`analysis/eval/examples/long_document_report.rs` accepts the cache, candidate
record directory, windows directory, then other baseline directories. It
retains the ordinary chunk-gate output alongside grouped uncertainty,
per-query deltas and the oracle gain from choosing the better chunker for
each query. It rejects different data, commits, bundles, devices, analyzers
or retrieval settings. Qasper questions are clustered by source paper and
GutenQA questions by book; resampling whole clusters preserves their
query-weighted mean. Contracts resample their unique templates. All use
10000 paired draws, seed 154 and 95% percentile intervals. These intervals
are conditional on the fixed corpus; they do not estimate uncertainty over
new documents or establish template independence beyond the source grouping.
Both raw and grouped verdicts are retained, including failures. The results
below apply this protocol without changing its datasets or thresholds.


## Frozen results

**FAIL: two of four evidence sets pass; all four document-ranking floors pass.**
The ordinary query bootstrap and the properly grouped report agree on that
verdict. ContractNLI fails contained evidence; CUAD fails both evidence
intervals. Qasper and GutenQA pass. Lower split rates alone do not pass the
gate. No dataset or configuration was replaced after observing these results.

All sixteen final records were measured at source
`cee65da41953c054a1d142a1da63afe9daeb0c07`, with the pinned manifest above,
CPU, model precision and eight CPU threads. The final records are under
`analysis/eval/records/long-document-1486/cpu8/`; arm directories `v1`,
`windows`, `fixed` and `semantic` use the frozen configurations above.
Machine identity and resolved parameters are in each record. Build metadata,
binary hashes, source identities, resource accounting and all paired query
deltas are under `docs/receipts/long-document-1486/`.

The following are differences in percentage points, with **grouped** 95%
intervals. The floor compares nDCG@10 with the best baseline for that set.

| Set | Recall@10 vs windows | Contained@10 vs windows | Split share v1 / windows | Floor vs best baseline | Evidence / floor |
|---|---|---|---|---|---|
| ContractNLI | +1.88 (+0.17..+4.15) | +0.56 (-1.13..+2.82) | 5.32% / 7.15% | -1.85 (-4.94..+0.89), fixed | FAIL / pass |
| CUAD | -0.47 (-2.73..+1.70) | +0.53 (-1.71..+2.73) | 4.28% / 4.69% | -1.04 (-3.89..+1.65), fixed | FAIL / pass |
| Qasper | +5.68 (+4.41..+7.02) | +13.78 (+11.97..+15.61) | 7.7% / 51.0% | -1.31 (-2.72..+0.12), semantic | pass / pass |
| GutenQA | +10.80 (+8.42..+13.26) | +12.05 (+9.54..+14.54) | 5.82% / 8.97% | +0.01 (-0.47..+0.47), fixed | pass / pass |

The raw `chunk-gate` output and its exit status 1 are retained in
`docs/receipts/long-document-1486/chunk-gate.txt`; the grouped intervals and
every candidate/reference score, signed delta and oracle gain are in
`docs/receipts/long-document-1486/paired.json`. The smaller short-document
shares are 1.63% for ContractNLI, 1.96% for CUAD and zero for the other sets,
so all four are eligible for the evidence gate.

### Document-ranking diagnostic, kept separate

Every preselected set has a per-query oracle uplift above the old +2-point
bar against sentence windows. That establishes reachability in principle,
not an achieved candidate gain or a selection rule applied after scoring.
The old document-ranking gate fails against each of the three baselines;
its raw outputs are retained as the three `gate-*.txt` receipts.

| Set | Scored queries with different nDCG@10 | Oracle uplift vs windows | Actual v1 uplift vs windows |
|---|---:|---:|---:|
| ContractNLI | 8 / 17 | +3.3782 | +1.5992 |
| CUAD | 29 / 40 | +3.9777 | +0.5304 |
| Qasper | 359 / 1451 | +3.1750 | -0.9263 |
| GutenQA | 402 / 3000 | +3.7357 | +1.7139 |

### Coverage and exclusions

`docs/receipts/long-document-1486/coverage.json` was computed through the
frozen library's pinned readers; its small audit consumer is included as
`docs/receipts/long-document-1486/coverage.rs`. The document and evidence
query denominators were independently checked against all sixteen records.
There are **zero absent-document judgements on every set**.

| Set | Loaded queries | Scored document / evidence queries | Document / evidence groups | Placed span judgements | Unplaced judgements | Queries without placed evidence |
|---|---:|---:|---:|---:|---:|---:|
| Qasper | 1451 | 1451 / 1310 | 416 / 408 papers | 3395 | 21 | 141 |
| GutenQA | 3000 | 3000 / 2888 | 100 / 100 books | 2888 | 112 | 112 |
| CUAD | 41 | 40 / 40 | 40 / 40 templates | 2643 | 0 | 1 |
| ContractNLI | 17 | 17 / 17 | 17 / 17 hypotheses | 2404 | 0 | 0 |

Only CUAD excludes a query from document scoring: Price Restrictions has no
positive test judgement. Qasper's existing reader also excludes 459 raw
`FLOAT SELECTED` evidence occurrences, representing 322 distinct
query/evidence-string pairs, and zero empty evidence strings. These filter
counts are separate from the 21 unplaced judgements and from query counts;
they must not be added together as if they counted the same unit. Qasper's
141 evidence-unscored questions include questions with no usable placed
evidence; their document judgements remain scored. All omissions apply
identically to every arm. The contract readers placed every annotated span.

### Structural and resource receipts

The current, clause-enabled screen is distinct from the historical screen:

| Set | Current v1 / windows chunks | Different document cuts | Documents with at most two chunks, v1 / windows |
|---|---:|---:|---:|
| Qasper | 10761 / 15405 | 416 / 416 | 0 / 0 |
| GutenQA | 71645 / 89505 | 100 / 100 | 0 / 0 |
| CUAD | 5181 / 5635 | 101 / 102 | 4 / 2 |
| ContractNLI | 1424 / 1282 | 122 / 123 | 2 / 3 |

The full current screen is in `docs/receipts/long-document-1486/current-screen.jsonl`.
A replay of the recovered private executable reproduced every historical
cut total and token percentile, and recovered its omitted short-document
counts (the same 0/0, 0/0, 4/2 and 2/3). That replay is labeled by executable
and source-file hashes, not claimed as a newly source-verified build. Its
legacy CUAD `queries=4182` counts repeated annotation pairs, not independent
queries. Neither structural screen is a retrieval result.

Twelve completed two-thread CPU probe records are retained separately in
`analysis/eval/records/long-document-1486/cpu2-probe/`. Every full metric,
including every per-query value, matches its eight-thread repeat exactly.
The initial two-thread GutenQA v1 attempt was terminated with status 143
before writing a record so the entire matrix could restart with a fixed
thread configuration. Its cancellation reason and resource receipt remain;
it is not a failed quality score or a passing run. All sixteen final runs
completed with exit status zero. The quality gate nevertheless exits 1.

Resource measurements were collected on a shared host and are not isolated
throughput comparisons. The completed runs, interruption and legacy replay
are accounted for in the receipts. `docs/receipts/long-document-1486/SHA256SUMS`
hashes all published records and receipts using repository-relative paths.
