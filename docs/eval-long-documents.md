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
These adaptations can have much lower recall@10 because one template judges
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
Both raw and grouped verdicts must be shown, including failures. No held-out
retrieval measurements have yet been recorded for these new adapters.
