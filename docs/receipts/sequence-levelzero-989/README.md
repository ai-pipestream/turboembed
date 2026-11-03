# BundleSequenceScorer on Level Zero (#989)

Base: `29214c70` (origin/main at branch time), plus
`analysis/embed/tests/sequence_device.rs`, which this receipt's commit
adds. Device: Intel Arc Pro B70 through the Level Zero backend, f32 at
both precisions, release builds, with `TURBO_TEST_REQUIRE_LEVELZERO=1`,
so a run that did not reach the device fails. `tests.log` is the run;
the B70 had no other compute client and its CI runner containers no
cargo, test or nextest process when it started.

## Parity

Twenty runs at each of `TURBO_TEST_PRECISION` `model` and `exact` of

```
TURBO_TEST_DEVICE=levelzero TURBO_TEST_REQUIRE_LEVELZERO=1 TURBO_TEST_PRECISION=model \
  cargo test --release --features turbo/levelzero -p turbo-analysis-embed --test sequence_device -- --nocapture
```

40 test binaries, every one passing, each printing the precision it ran
at (`f32 (model)` or `f32 (exact)`) and the same numbers as every other
run:

| Scorer | Largest logit difference from the CPU | Labels the same as the CPU's |
|---|---|---|
| `BundleSequenceScorer::score_chunks` | 5.4e-6 (bound 1e-4) | all seven texts; billing, hardware and search all picked |

The bundle is a sequence-classification bundle of
`testdata/tiny-bert-bundle`'s encoder and a head made for the test,
through `nli_support::make_nli_bundle_with`. The tiny encoder's pooled
state moves by about 1e-3 between inputs, so a fixed head picks one
label for every text; the test centres its head on its seven texts (a
CPU pass reads three pooled dimensions, each logit is its dimension less
the texts' mean, times 200), and they then pick all three labels, by CPU
margins of 2.3e-3 to 0.053. A text whose CPU margin is within twice the
bound is not held to a label, and the texts that are must pick at least
two; on the B70 every text is held. The x200 head scales the device's
difference with it.

The host's one-minute load was between 19.6 and 21.8 over the run (the
log records it before each); these are correctness runs, not timings.

## Not done

- namefind-dl (the BERT-NER finder) on Level Zero: it needs a sealed
  `dslim/bert-base-NER` bundle, which docs/licences.md refuses by the
  class rule for its CoNLL-2003 training data, so none is built here.
