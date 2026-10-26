# Embedding heads on Level Zero (#989)

Base: `09d5577d` (origin/main at branch time). Device: Intel Arc Pro B70
through the Level Zero backend, f32 at both precisions. `tests.log` is
the run: twenty release runs at each of `TURBO_TEST_PRECISION=model` and
`exact`, eighty test binaries in all, every one passing and every repeat
printing the same numbers.

These are correctness runs, not timings. The host was busy with other
work throughout: the one-minute load average was between 23.7 and 27.5
(the log records it before each run) on a 32-thread host. The B70 CI
runner containers had no cargo, test or nextest process when it began.

| Head | Test | Against the CPU | Largest difference |
|---|---|---|---|
| `ClassifierHead`, label centroids | `analysis/embed/tests/heads.rs` `heads_and_centroids_cpu_and_device_parity` | scores within 1e-5 | within 1e-5 (the test prints the bound only) |
| `LearnedAspectSentiment` | `analysis/embed/tests/heads.rs` `learned_and_rule_aspect_sentiment_extract_triplets_and_match_cpu_device` | the same targets, categories and opinions; each polarity within 1e-5 | within 1e-5 |
| `DenseClassifier` (logistic softmax, GELU MLP sigmoid with calibrated thresholds) | `analysis/embed/tests/heads_device.rs` | each sentence vector and probability within 1e-4 (1e-2 at f16), the predicted tags the same | vectors 1.2e-7, probabilities 5.7e-8 |
| `BundleNliScorer` | `analysis/embed/tests/heads_device.rs` | each pair's logits within 1e-4 (1e-2 at f16), and the label each pair picks the same | 7.2e-6 |

How each test reaches the device:

- `heads_device.rs` runs on the device `TURBO_TEST_DEVICE` names and,
  with `TURBO_TEST_REQUIRE_LEVELZERO=1`, fails unless the session's
  backend is `levelzero`.
- `heads.rs` reads `TURBO_TEST_DEVICE` alone. Its centroid check prints
  the session's execution (`levelzero`, the B70, above), and its aspect
  check asserts the device backend is not `cpu`. It reads neither
  `TURBO_TEST_REQUIRE_LEVELZERO` nor `TURBO_TEST_PRECISION`, so its runs
  are at the model's precision each time. That file is 1954 lines, and
  those switches there wait on splitting it.

The NLI head is built for the test. The tiny encoder's pooled state
moves by about 1e-3 between inputs, so a fixed head picks one label for
every pair. The test centres its head on its six pairs instead (a CPU
pass reads three pooled dimensions; each label's logit is its dimension
less the pairs' mean, times 200), and the pairs then pick all three
labels: contradiction, neutral, entailment, contradiction,
contradiction, entailment, by margins of 4.3e-3 to 0.38. A pair whose
CPU margin is within twice the bound is not held to a label, and the
pairs that are must pick at least two. On the B70 every pair is held,
and every label is the CPU's.

The bounds are `docs/conformance.md`'s F32 and F16 score bounds; every
difference is far inside them.

Command, per run:

```
TURBO_TEST_DEVICE=levelzero TURBO_TEST_REQUIRE_LEVELZERO=1 TURBO_TEST_PRECISION=model \
  cargo test --release --features turbo/levelzero -p turbo-analysis-embed \
  --test heads_device --test heads -- --nocapture
```

The BERT-NER finder's device test (namefind-dl) is not here: it waits on
a sealed dslim/bert-base-NER bundle.
