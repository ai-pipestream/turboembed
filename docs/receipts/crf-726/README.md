# CRF evidence

MASC SHA-256: `445449f535cac62e74b2712ad975d7b727e72b8276720dd27492266565ae0afd`.
The fixture follows the existing document split and Twitter exclusions,
sampling 2,048 training and 512 held-out sentences evenly across genres.
It contains 439/105 person entities. Cutoff=2, iterations=100; CRF L2=1.0.
The Maxent baseline retains normal adaptive features.

| Codec | CRF entity F1 | Maxent entity F1 |
| --- | --- | --- |
| BIO | 0.204724409 | 0.052631579 |
| BILOU | 0.177419355 | 0.053571429 |

Low absolute scores limit this to a fixture comparison. An initial prefix
fixture returned zero F1 for both models, prompting genre-spread sampling.
No parameters were tuned on held-out labels. Logs contain counts and timings.

Finite differences on real training sample 31 (10 tokens) covered 405 BIO
and 685 BILOU parameters: maximum absolute errors 2.417e-10 and 1.713e-10.
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
repetitions per process. CPU-pinned ABBA medians: Rust 0.114110841 seconds,
Java 0.293013037 seconds, a 2.57x ratio. Paths match and checked first-position
marginals differ by less than 1e-12.

Java preserves exact path-score sums; Rust uses f64. This bounded workload
does not prove extreme-score parity or a general speed advantage.

CPU: AMD Ryzen 9 9950X3D, x86_64. Runs use one pinned core.

## Red/green and gates

`red-numeric.log` shows a failing common-score regression on the initial
implementation: probabilities were [1, 1], expected [0.5, 0.5]. Centering
start, end and transition scores fixes it; `green-numeric.log` records the
passing exhaustive and numeric tests. Unreachable spans remain zero.

Final checks after merging main:
- `cargo fmt --all --check`: pass.
- `cargo clippy --locked -p turbo-sequence -p turbo-train -p turbo-train-components -p turbo-namefind -p turbo-train-recipes -p turbo-bench --all-targets -- -D warnings`: pass.
- `cargo test --locked -p turbo-sequence -p turbo-train -p turbo-train-components -p turbo-namefind`: 380 passed, 0 failed, 2 existing ignored tests.
- `cargo test --locked -p turbo-bench --test tree`: 3 passed.

The ignored tests are the pre-existing QN NER fixture (unreproduced sums)
and the opt-in published UD model reproduction (external treebanks).
CRF training is slower than the Maxent baseline; the speed comparison above
measures decoding only. No claim of faster training is made.
