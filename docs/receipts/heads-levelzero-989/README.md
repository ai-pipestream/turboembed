# Embedding heads on Level Zero (#989)

Base: `09d5577d` (origin/main at branch time). Device: Intel Arc Pro B70
through the Level Zero backend, f32 at both precisions. `tests.log` is
the run: three release runs at each of `TURBO_TEST_PRECISION=model` and
`exact`, with `TURBO_TEST_REQUIRE_LEVELZERO=1`, so a run that did not
reach the device fails. The B70 CI runner containers had no cargo, test
or nextest process when it started.

| Head | Test | Against the CPU | Largest difference |
|---|---|---|---|
| `ClassifierHead`, label centroids | `analysis/embed/tests/heads.rs` `heads_and_centroids_cpu_and_device_parity` | scores within 1e-5 | within 1e-5 (the test prints the bound only) |
| `LearnedAspectSentiment` | `analysis/embed/tests/heads.rs` `learned_and_rule_aspect_sentiment_extract_triplets_and_match_cpu_device` | the same targets, categories and opinions; each polarity within 1e-5; the device backend not `cpu` | within 1e-5 |
| `DenseClassifier` (logistic softmax, GELU MLP sigmoid with calibrated thresholds) | `analysis/embed/tests/heads_device.rs` | each sentence vector and probability within 1e-4 (1e-2 at f16), the predicted tags the same | vectors 1.2e-7, probabilities 5.7e-8 |
| `BundleNliScorer` | `analysis/embed/tests/heads_device.rs` | each pair's logits within 1e-4 (1e-2 at f16) | 8.9e-8 |

The bounds are `docs/conformance.md`'s F32 and F16 score bounds; every
difference is far inside them, and the three runs at each precision
printed the same numbers. `heads.rs` reads no `TURBO_TEST_PRECISION`, so
its runs are at the model's precision each time; that file is 1954
lines, and a precision switch there waits on splitting it.

Command, per run:

```
TURBO_TEST_DEVICE=levelzero TURBO_TEST_REQUIRE_LEVELZERO=1 TURBO_TEST_PRECISION=model \
  cargo test --release --features turbo/levelzero -p turbo-analysis-embed \
  --test heads_device --test heads -- --nocapture
```

The BERT-NER finder's device test (namefind-dl) is not here: it waits on
a sealed dslim/bert-base-NER bundle.
