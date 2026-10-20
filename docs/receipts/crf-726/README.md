# CRF evidence

MASC SHA-256: `445449f535cac62e74b2712ad975d7b727e72b8276720dd27492266565ae0afd`.
The fixture follows the existing document split and Twitter exclusions,
sampling 2,048 training and 512 held-out sentences evenly across genres.
It contains 439/105 person entities. Cutoff=2, iterations=100; CRF L2=1.0.
The Maxent baseline retains normal adaptive features.

| Codec | CRF entity F1 | Maxent entity F1 |
| --- | --- | --- |
| BIO | 0.188976378 | 0.052631579 |
| BILOU | 0.177419355 | 0.053571429 |

Low absolute scores limit this to a fixture comparison. An initial prefix
fixture returned zero F1 for both models, prompting genre-spread sampling.
No parameters were tuned on held-out labels. Logs contain counts and timings.

Finite differences on real training sample 31 (10 tokens) covered 405 BIO
and 685 BILOU parameters: maximum absolute errors 2.033e-10 and 2.317e-10.
Synthetic tests enumerate legal paths of lengths 1 through 4 and compare
partitions, node/span marginals and top-20 scores. Payload truncation,
invalid counts/labels, NaN and signed numeric features are also tested.

MASC 3.0.0 attribution: American National Corpus project (https://anc.org/),
Manually Annotated Sub-Corpus, CC BY 3.0 US. Dataset bytes are not committed.

## CPU comparison

Reference: ai-pipestream/opennlp commit
`cdf33c653d5d61d5af7d37ffee189a83bd8392ea`, unchanged postag CrfScorer.java and
CrfScoreTable.java. Both harnesses calculate marginals and best paths.
32 tokens, 9 BILOU unit labels (all transitions legal), 200 warmups and 1,000
repetitions per process. CPU-pinned ABBA medians: Rust 0.107481407 seconds,
Java 0.261199685 seconds, a 2.43x ratio. Paths match and checked first-position
marginals differ by less than 1e-12.

Java preserves exact path-score sums; Rust uses f64. This bounded workload
does not prove extreme-score parity or a general speed advantage.
