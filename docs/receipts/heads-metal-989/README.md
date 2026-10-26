# Embedding heads on Metal (#989)

Code: `8d9acddbff992badcd6e875731e6697748cb1256` (the commit each run in the log
names, with its full command). Device: Apple M2 through the Metal backend,
F32 at both precisions. `tests.log` is the run, unfiltered but for cargo's
build lines: twenty release runs at each of `TURBO_TEST_PRECISION=model`
and `exact`, eighty test binaries in all, as #1416's Level Zero receipt
ran them. Every one passed, and every repeat printed the same numbers,
the NLI test's per-pair lines included.

These are correctness runs, not timings. The M2 is a desktop in use,
and other work was running: the one-minute load average was between
7.0 and 48.2 (the log records it before each run).

F16 is not covered: every run here is F32, at MODEL and at EXACT, since
the bundle's FASTEST is not one these tests ask for.

| Head | Test | Against the CPU | Largest difference |
|---|---|---|---|
| `ClassifierHead`, label centroids | `analysis/embed/tests/heads.rs` `heads_and_centroids_cpu_and_device_parity` | scores within 1e-5 | within 1e-5 (the test prints the bound only) |
| `LearnedAspectSentiment` | `analysis/embed/tests/heads.rs` `learned_and_rule_aspect_sentiment_extract_triplets_and_match_cpu_device` | the same targets, categories and opinions; each polarity within 1e-5 | within 1e-5 |
| `DenseClassifier` (logistic softmax, GELU MLP sigmoid with calibrated thresholds) | `analysis/embed/tests/heads_device.rs` | each sentence vector and probability within 1e-4, the predicted tags the same | vectors 6.0e-8, probabilities 6.6e-8 |
| `BundleNliScorer` | `analysis/embed/tests/heads_device.rs` | each pair's logits within 1e-4, the label each pair picks the same | 9.5e-6; the pairs pick contradiction, entailment and neutral |

How each test reaches the device:

- `heads_device.rs` runs on the device `TURBO_TEST_DEVICE` names. It now
  reads `TURBO_TEST_REQUIRE_METAL` as it reads the CUDA and Level Zero
  variables: with it set, a run that names no device, or names `cpu`,
  fails ("metal is required but TURBO_TEST_DEVICE names no device"), and
  so does one whose session's backend is not `metal`.
- `heads.rs` reads `TURBO_TEST_DEVICE` alone. Its centroid check prints
  the session's execution (`metal`, the Apple M2, above), and its aspect
  check asserts the device backend is not `cpu`. It reads neither
  `TURBO_TEST_REQUIRE_METAL` nor `TURBO_TEST_PRECISION`, so its runs are
  at the model's precision each time, as on Level Zero.

The bounds are `docs/conformance.md`'s F32 score bounds; every
difference is far inside them. The NLI head is the test's own, centred on
its six pairs (docs/receipts/heads-levelzero-989 says how). Each per-pair line
in the log gives the CPU's label and margin, every run: pair 0 contradiction by 0.047,
pair 1 neutral by 0.378, pair 2 entailment by 0.057, pair 3
contradiction by 0.018, pair 4 contradiction by 0.0043 and pair 5
entailment by 0.0079, each margin over twice the 1e-4 bound, so the
test held every pair's Metal label to the CPU's, and it passed each time.

Command, per run:

```
TURBO_TEST_DEVICE=metal TURBO_TEST_REQUIRE_METAL=1 TURBO_TEST_PRECISION=model \
  cargo test --release --features turbo/metal -p turbo-analysis-embed \
  --test heads_device --test heads -- --nocapture
```

The BERT-NER finder's device test (namefind-dl) is not here: it waits on
a sealed dslim/bert-base-NER bundle, as on Level Zero.
