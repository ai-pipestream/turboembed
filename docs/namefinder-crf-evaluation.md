# Full MASC CRF name-finder evaluation

The full-corpus run for #923 rejects publication of all 4 CRF models.
The CRF's micro F1 is 44.66 across person, location, organization and date,
compared with 51.71 for maxent trained on the same MASC split and 56.03 for
the published OpenNLP 1.5 models. On the 3 types supported by namefind-dl,
the scores are 44.24, 53.96 and 80.12 respectively. No model or zoo entry
is published by this change.

## Data and training

The pinned MASC 3.0.0 corpus digest is
`445449f535cac62e74b2712ad975d7b727e72b8276720dd27492266565ae0afd`.
The standard recipe split has 351 training documents and 39 held-out
documents. The 2 Twitter documents remain excluded. Training uses 24958
samples and 431637 tokens, with no training-sample cap. Evaluation uses
6384 samples and 85292 tokens; 1 malformed-token held-out sentence is
skipped. The recipe checks the pinned release, training and held-out
counts, and reference model hashes before writing outputs.

Training is BIO, feature cutoff 2, L2 1.0 and 100 optimizer iterations.
Model ZIP timestamps are fixed at zero. The matched maxent baseline uses
GIS, cutoff 0, 300 iterations and 1 thread on the same training documents.
Each type has its own model and gold annotation set. Both inference paths
use the existing OpenNLP feature generators. CRF previous-label features
receive constant `other` history and adaptive state resets each sentence;
maxent preserves its normal document-adaptive behavior. This evaluates
the current shipped recipes, not a controlled comparison of optimizers
with the same features or a search over hyperparameters.

The local transformer comparison uses `dslim/bert-base-NER`, upstream
revision `d1a3e8f13f8c3566299d95fcfc9a8d2382a9affc`, through the actual
`NameFinderDl` CPU backend at `Precision::Exact`, 4 threads and batches
of 8 samples. The bundle manifest SHA-256 is
`867431b1ddfd21d61f157948ee77d37311a4af74c95444f02cc64a00eace4d9c`.
Its existing default window configuration is used, including long-sentence
windows. MASC tokens are joined with one space. Predicted byte spans map
to their overlapping token range, then exact typed token spans are scored.
The comparison verifies the prediction-input hash against the pinned split.
PER/LOC/ORG map to person/location/organization; MISC is excluded. The
checkpoint has no date label, so no date result is invented for it. Only
local evaluation uses the checkpoint; no weights or corpus text are added.

## Accuracy

All values below are percentages. The [machine-readable receipt](receipts/crf-923/report.json)
records exact counts, unrounded scores, model and sample hashes, omitted
annotation counts, gate settings and intervals. Evaluation requires an
exact span and type match. The overall rows are micro scores from summed
counts, not averages of per-type F1.

| Type | CRF precision | CRF recall | CRF F1 | Matched maxent F1 | Published 1.5 F1 | Namefind-dl F1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Person | 75.48 | 23.04 | 35.30 | 55.33 | 60.82 | 87.14 |
| Location | 80.55 | 54.13 | 64.75 | 56.39 | 64.10 | 82.06 |
| Organization | 70.97 | 34.27 | 46.22 | 44.83 | 42.54 | 55.92 |
| Date | 67.54 | 36.03 | 46.99 | 37.73 | 42.44 | unsupported |
| All 4 types | 74.79 | 31.84 | 44.66 | 51.71 | 56.03 | unsupported |
| Shared 3 types | 76.39 | 31.13 | 44.24 | 53.96 | not scored here | 80.12 |

Gold counts are 1363 person, 436 location, 321 organization and 358 date.
Training name counts are 4894, 3944, 2934 and 2714 respectively. The MASC
reader omits annotations outside sentences, annotations in dropped
sentences and overlapping annotations according to the existing recipe.
Those omissions are recorded separately for every type and side in the
receipt; these figures do not claim to score every raw MASC annotation.

## Gate decisions

The existing model gate uses 10000 paired bootstrap resamples, seed 154,
and a noninferiority margin of 0.5 F1 points. A pass requires the lower
95% bound of CRF minus reference to be at least -0.5. Units are held-out
sentences; the interval does not account for document-level correlation.

| Comparison | Difference | 95% interval | Decision |
| --- | ---: | --- | --- |
| Person vs matched maxent | -20.03 | [-23.73, -16.25] | FAIL |
| Location vs matched maxent | +8.36 | [4.69, 12.09] | PASS |
| Organization vs matched maxent | +1.39 | [-7.28, 9.78] | FAIL |
| Date vs matched maxent | +9.27 | [3.37, 15.36] | PASS |
| All 4 vs matched maxent | -7.05 | [-9.62, -4.49] | FAIL |
| All 4 vs published 1.5 | -11.36 | [-14.08, -8.73] | FAIL |
| Shared 3 vs namefind-dl | -35.89 | [-38.70, -33.07] | FAIL |

Every per-type comparison with published 1.5 fails its gate, as does every
shared-type comparison with namefind-dl. The receipt includes each of
those intervals and the matched maxent versus namefind-dl results. Passing
one comparison does not authorize publishing a model that fails another.

## Reproduction and optimizer correction

The first full location training exposed a generic QN error-handling bug:
an invalid trial point left a cached error even after Armijo backtracking
recovered to a finite point. Successful evaluations now clear that stale
error. A separate guard rejects a backtracking step that underflows to
zero instead of reporting false convergence. The event-training objective
and its JVM arithmetic are unchanged. Tests use real bounded differentiable
objectives, including an unrecoverable domain and a nonfinite trial gradient.
The recovery test fails on the original implementation.

After adding the zero-step guard, all 4 models were trained again on the
full corpus. Every model SHA-256 matched the earlier completed run. The
CRF recipe exits with failure after writing all per-type and overall gate
records because the accuracy gates fail. This is a measured rejection,
not a training crash or a unit-test failure.

Use the examples with caller-supplied local inputs and output directories:

```sh
cargo run --locked --release -p turbo-train-recipes --example crf_namefind -- MASC_DIRECTORY REFERENCES OUTPUT
cargo run --locked --release -p turbo-namefind-dl --example masc_predict -- BUNDLE OUTPUT/person/held-out.json PREDICTIONS_JSON
cargo run --locked --release -p turbo-train-recipes --example crf_compare_dl -- MASC_DIRECTORY OUTPUT PREDICTIONS_JSON
```

`REFERENCES` contains the digest-checked OpenNLP 1.5 English person,
location, organization and date models. The first command refuses an
existing per-type output directory. The prediction command refuses an
existing output file. The comparison command writes reports and returns
success when report generation succeeds, even when a reported gate fails;
publication must inspect the gate decisions. Keep model files, held-out
inputs and prediction records local. The committed receipt contains
aggregate evidence and corpus-relative document identifiers only.
