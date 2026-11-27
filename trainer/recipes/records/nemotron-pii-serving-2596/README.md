# Nemotron PII: catalog versus refit at their own temperatures (#2596)

This is a comparison of two pinned artifacts, **not a model promotion**.
Both were run on CPU through the production `PiiTagger` and analyzer, at
an admission threshold of 0.50. The catalog model uses its released
calibration. The unreleased refit has no embedded calibration; this
test supplies the candidate's dev-only fit explicitly.

| Artifact | Manifest SHA-256 | Temperature |
| --- | --- | ---: |
| Catalog v1 | `19d3fb87daac08f0361390809b747af638bc2dbb93e447695c3f5609fc735b6d` | 1.093758 |
| Refit v2 | `dd73330bc8bef5bc0f1b9c7b6c004cfbf6713fa1dadcc03a6238a159b6ddc3b7` | 1.0807142 |

The inputs and calibration records are described in
[`nemotron-pii-1888`](../nemotron-pii-1888) and
[`nemotron-pii-names-2398`](../nemotron-pii-names-2398). To reproduce,
set `TURBO_NEMOTRON_PII_BUNDLE` to v1 and
`TURBO_NEMOTRON_PII_REFIT_BUNDLE` to v2, then run:

```sh
cargo test --locked -p turbo-tagger --test pii_shared_name_sentences -- --ignored --nocapture
```

## Exact-span readings

| Cases | Catalog head | Refit head | Catalog + Places | Refit + Places |
| --- | ---: | ---: | ---: | ---: |
| Eight originally reported city subjects | 0/8 | 6/8 | 0/8 | 7/8 |
| All 22 city sentences | 4/22 | 19/22 | 7/22 | 20/22 |
| 20 standalone person sentences | 10/20 | 9/20 | 10/20 | 9/20 |

| Person memo (true / false / missed cities) | Catalog head | Refit head | Catalog + Places | Refit + Places |
| --- | ---: | ---: | ---: | ---: |
| 1 | 12 / 12 / 1 | 12 / 12 / 1 | 13 / 4 / 0 | 13 / 4 / 0 |
| 2 | 5 / 2 / 3 | 6 / 3 / 2 | 8 / 5 / 0 | 8 / 3 / 0 |
| 3 | 4 / 2 / 4 | 4 / 3 / 4 | 8 / 5 / 0 | 7 / 4 / 1 |

The two person memos in #2410 do not regress with Places, but v2 misses
a city on memo 3 (`back from Vienna`) that v1 kept. It also loses one
standalone person reading. Using the refit's own temperature does not
remove either deficit at 0.50. The runnable test still checks the #2410
floors and prints the additional #2596 results; passing it is **not**
evidence that the broader #2596 quality requirements are satisfied.

## Frozen corpus scores (historical training receipts)

These are exact-span precision/recall/F1 from the pinned Nemotron-PII
test split, not from applying the production 0.50 threshold or Places.
The two receipts have the same gold counts for these classes. This
comparison uses their historical decoded labels; the serving
temperature test above did not rescore this split.

| Type | Gold spans | v1 P / R / F1 | v2 P / R / F1 |
| --- | ---: | ---: | ---: |
| All | 848,257 | 0.5299 / 0.7176 / 0.6096 | 0.5432 / 0.7180 / 0.6185 |
| city | 18,328 | 0.7012 / 0.7794 / 0.7382 | 0.6918 / 0.7699 / 0.7287 |
| company_name | 54,826 | 0.4792 / 0.6250 / 0.5425 | 0.4582 / 0.5777 / 0.5111 |
| last_name | 59,565 | 0.8907 / 0.8904 / 0.8905 | 0.8787 / 0.8958 / 0.8872 |

Overall F1 conceals both city and company precision/recall losses. The
last-name loss comes from lower precision despite a small recall gain.
For a new variant, select the checkpoint and fit temperature on dev
only. Then run the same untouched corpus split and paired analyzer
checks. Compare both false and missed city counts on each memo, rather
than trading recall for precision inside one aggregate score.

## Bounded dev-only pilot: smaller lexical table (rejected)

This CPU pilot reused the layer-22 feature cache from the refit record.
Both fits used the same 22,500 publisher training documents (zero-based
indices divisible by four among the 90,000 publisher training rows),
plus all 6,000 paired synthetic training documents. They kept complete
documents and the layer-cache reader checked the full source dimensions
and preserved source row offsets. Neither fit read the test feature
file. Both selected epoch 6 on the unchanged 10,600-row dev split;
seed 1, batch 128, learning rate 0.004, patience 4, six CPU workers.
The sole comparison was 128 vs 64 dimensions in the hashed word table.

| Dev exact-span F1 | Width 128 | Width 64 |
| --- | ---: | ---: |
| All, full dev | 0.5805 | 0.5821 |
| City, full dev | 0.7541 | 0.7507 |
| Company name, full dev | 0.7712 | 0.7752 |
| Last name, full dev | 0.8270 | 0.8328 |
| City, paired synthetic dev only | 0.8617 | 0.8625 |
| First name, paired synthetic dev only | 0.8594 | 0.8563 |

The narrower table raises overall dev F1 but lowers city F1 on full dev
and person F1 on the paired portion. It was rejected before calibration
or an evaluation on the held-out inputs. The 600 paired dev documents
are separate draws from the training generator, with the same template
and name pools; this pilot does not demonstrate transfer to unseen
names or contexts. No pilot weights entered the bundle or zoo.

## Bounded dev-only pilot: lower synthetic share (rejected)

The follow-on held lexical width at 128 and total training documents
at 28,500. It kept the 22,500 publisher control rows, added 3,000 rows
evenly spaced from the indices whose remainder modulo four is one,
and retained every second synthetic row, starting with the first.
For the extra publisher rows, choose group `g` from 0 through 22,499
when integer division of `(g + 1) * 3000 / 22500` is larger than
`g * 3000 / 22500`, then take row `4*g + 1`. Both fits kept the same
seed, batch, number of steps per epoch, and original dev split,
selected epoch 6, and never scored the test split. Corpus parsing
validated the pinned publisher release; only train/dev feature files
were present in the pilot cache.

The 3,000 retained synthetic rows contain 7,135 city and 7,024 person
spans; 2,598 rows contain both roles, and all 48 shared names appear
in both roles across those rows.

| Dev exact-span F1 | 6,000 synthetic rows | 3,000 synthetic rows |
| --- | ---: | ---: |
| All, full dev | 0.5805 | 0.5833 |
| City, full dev | 0.7541 | 0.7544 |
| Company name, full dev | 0.7712 | 0.7737 |
| Last name, full dev | 0.8270 | 0.8253 |
| City, paired synthetic dev only | 0.8617 | 0.8324 |
| First name, paired synthetic dev only | 0.8594 | 0.8273 |

The preregistered dev comparison required no drop on the full-dev and
paired metrics before using any holdouts. The lower synthetic share
fails that requirement: it further worsens last names and loses about
three F1 points on each paired class. No calibration, serving export,
or test run was attempted for this pilot.

These two bounded pilots do not establish a safe model fix. A different
data intervention would keep the paired-example budget and architecture
fixed while replacing some generated rows with balanced context
contrasts and replay of original organization and surname contexts.
The current synthetic family assigns only city and first-name labels;
it cannot directly rehearse organization names or surnames. Include
the same unfamiliar token in city subjects, ordinary human actions,
multi-word full names, and company names (e.g. a municipal subject,
an employee with a family name, and a business suffix), including
matched clauses after `from` and `to`. Build those contrasts with
new name pools, independent authored templates, and no copied memo or
probe strings. Split names **and** template families before generation,
verify exact text and normalized-template disjointness from the reserved
probes, and gate on this separate synthetic dev and publisher dev
before trying the frozen tests. Keep the previously
measured city gains, all three memo error counts, standalone person
results, and corpus class scores as separate acceptance gates.

## Registered continuation: balanced same-token contrast

This is a matched bounded experiment, not a shipment. Hold the ModernBERT
layer-22 encoder, 128-dimensional hashed word table, BIO head, seed 1,
batch 128, learning rate 0.004, six epochs, and 28,500 training-document
budget fixed. Use the same 22,500 pinned publisher rows in each arm plus
6,000 project-authored documents: the original generator for control,
or the four-role same-token generator for the candidate. The new example
generator labels city, given name, multi-word surname, and organization
in matched subject, `from`, and `to` contexts. Draw candidate dev from
a separate name pool and separate template families. Use the same
publisher and synthetic feature sources in both arms except for the
intentional replacement of the synthetic training documents. The
publisher prefix is reused; all synthetic arms use the same pinned CPU
encoder, first-subword pooling and settings.

Choose each arm's checkpoint on **publisher dev only**. The preregistered
dev gate requires no decrease in publisher micro span F1 or its city,
first-name, last-name, and company-name F1, no decrease in historical
synthetic-dev city/person F1, and no decrease in any of the four roles
on the new synthetic dev, with at least 0.02 absolute improvement in
the four-role macro span F1. A failing candidate is not calibrated,
assembled or scored on holdout. This gate is purposely not a serving
pass. A survivor would still need an independent full fit, dev-only
calibration and the previously reserved serving and corpus acceptance
checks in #2596, including all three memo counts.

`pii_contrast_cache` writes only the synthetic train and dev files.
`pii_contrast_fit` checks the archived encoder, head spec, and publisher
cache metadata digests, loads only the publisher training Parquet for
the publisher train/dev split, and refuses test feature reads. Set
`ENCODER`, `EXPERIMENT`, `PUBLISHER_CORPUS`, `PUBLISHER_CACHE` and
`REFIT_HEAD_SPEC` to the pinned inputs described above and the archived
v2 fit. The output directory is a fresh scratch directory.

```sh
cargo build --locked --release -p turbo-trainer-recipes \
  --example pii_contrast_cache --example pii_contrast_fit
target/release/examples/pii_contrast_cache "$ENCODER" "$EXPERIMENT/legacy" legacy
target/release/examples/pii_contrast_cache "$ENCODER" "$EXPERIMENT/contrast" contrast
target/release/examples/pii_contrast_fit \
  "$PUBLISHER_CORPUS" "$PUBLISHER_CACHE" "$REFIT_HEAD_SPEC" "$EXPERIMENT"
```

Exit status 2 means the matched run finished and the dev gate rejected
the candidate. Inspect the per-arm `receipt.json` files and
`dev-gate.json`. Do not substitute a more favorable dev split or retune
after a holdout result. Neither the example programs nor their outputs
change the model used by the serving path.

### Objective accounting

The batches and optimizer schedule match (223 updates per epoch), but the
examples change the token mix. The head receives one supervised loss row
per SimpleTokenizer word, including `O`; it sums per-word gradients
and divides by that batch's word count. The publisher contribution is
4,401,114 word rows in either arm. Total rows rise from 4,611,099 to
4,717,633 (+2.31%); synthetic rows rise from 209,985 to 316,519,
including 28,375 vs 34,498 synthetic BIO-labelled rows.

`pii_contrast_mix` replays the exact initialization and seeded batch
order without fitting. Across the six epochs, synthetic word rows
account for 4.610% of all gradient coefficients in the control and
6.783% in the candidate; BIO-labelled synthetic word coefficients
account for 0.623% and 0.739%, respectively. These are shares of the
loss denominator, not claims about the magnitude of the gradients or
AdamW updates. This is a material dose difference even with the same
document budget; a positive dev result would need confirmation at a
matched token budget before opening the holdouts. The preregistered
dev checks are unchanged.

```sh
cargo run --locked --release -p turbo-trainer-recipes --example pii_contrast_mix -- \
  "$PUBLISHER_CORPUS" "$REFIT_HEAD_SPEC"
```

### Matched contrast pilot result (rejected)

The bounded CPU comparison ran the preregistered fit at
`4c8ea58186bfacf4b3f099a807517647c74002ac`. Both selected epoch 6
and wrote 6,201,308-byte head files. SHA-256:

- Control: `3130d49ed775838a4107f0fb463ec63904a4189ebcde8dec5e10d53b282e71c6`
- Candidate: `cdf1b3cbb4bb5819611042a758bf2dcada5dba184592e251d4a79b660fb4d063`

| Dev set and exact-span F1 | Control | Candidate |
| --- | ---: | ---: |
| Publisher micro | 0.5725 | 0.5709 |
| Publisher city | 0.6787 | 0.6715 |
| Publisher first name | 0.8778 | 0.8740 |
| Publisher last name | 0.8270 | 0.8158 |
| Publisher company name | 0.7712 | 0.7635 |
| Historical synthetic city | 0.8617 | 0.5505 |
| Historical synthetic first name | 0.8594 | 0.6547 |
| New name- and template-disjoint dev city | 0.3730 | 0.3460 |
| New dev first name | 0.2140 | 0.2979 |
| New dev multiword surname | 0.0000 | 0.0000 |
| New dev company name | 0.0259 | 0.0516 |

The new-dev four-role macro rose by 0.0207, but city fell and neither
arm matched any of the 600 exact multiword surname spans. The candidate
predicted more surname spans (710 vs 404) without getting an exact
match. Every publisher target class and both historical paired classes
fell. The command exited 2 as designed: three dev checks failed and
`holdout_scored` was false. Neither head was calibrated, assembled,
promoted, or evaluated on the reserved corpus, short sentences or memos.

This rejects the tested generator and replacement, not the general
hypothesis that token-matched contrasts could improve the model. The
document count matched but the synthetic loss weight did not. No
token-budget confirmation was run after this rejection.

### Why the synthetic surname score was zero

A dev-only replay used the frozen control and rejected candidate, the
same cached feature rows, and the production core tagger's decoder.
It matched the receipt scorer exactly on predictions and on the gold
BIO tags. Every gold surname had `B-last_name I-last_name`. These
are 600 scored instances from 128 distinct surname clauses, not 600
independent surname constructions.

| Gold surname coverage on the new synthetic dev | Control | Candidate |
| --- | ---: | ---: |
| Exact type and boundary | 0 | 0 |
| Right type, wrong boundary | 199 | 478 |
| Right boundary, wrong type | 30 | 10 |
| Overlap with both wrong type and boundary | 321 | 98 |
| No overlapping span | 50 | 14 |

The candidate tagged the name token as a surname 478 times (381
`I-last_name`, 97 `B-last_name`), but never tagged the preceding
`del` or `von` as a surname. Both heads emitted only single-token
surname spans on this set. The synthetic train families used
`de`, `Van` and `St.`; the dev families intentionally use different
particles as well as different names and contexts. On 12 fresh CPU
probes holding context and name fixed across particle swaps, the
candidate got 3/6 exact with `de` or `Van`, and 0/6 with `del` or
`von`; the control got 0/12. This is a small diagnostic, not a score
gate or proof the particle is the sole cause.

The regression is not limited to those unfamiliar particles. On
publisher dev, exact multiword surname matches went from 108/198 to
90/198, and same-type partials rose from 74 to 89. Particle-led exact
matches went from 99/165 to 82/165 (the `al` stratum was 81/129 to
65/129). Publisher dev contains no gold multiword surname starting
with `del` or `von`. This supports a learned boundary regression,
not a bug in how the decoder joins well-formed BIO tags.

```sh
cargo run --locked --release -p turbo-trainer-recipes --example pii_surname_audit -- \
  "$EXPERIMENT" "$ENCODER"
```

### Registered continuation: narrow, row-matched overlay

Replace exactly 600 of the 6,000 legacy synthetic documents; retain
the other 5,400 and their encoded rows byte-for-byte. A deterministic
selection distributes edits proportionally within eligible legacy
strata of city/first-name counts, word length, and non-`O` word
count. Every new row preserves its original index, word count and
count of non-`O` BIO words. Thus each seeded batch has the same
number of documents, total word rows and BIO-labelled synthetic
loss rows in both arms. The label and span-type distribution can
change intentionally.

The interventions are short four-role subject, `from` and `to`
contrasts, with two- or three-word surnames (particles and other
compound surnames), a mix of title and no-title contexts, and
ordinary unlabelled uses of particle-like words. The earlier
`del` and `von` dev targets and the reserved examples remain out of
training. Use the same publisher subset, frozen control head, hash
width, model, seed, batch/update schedule and dev sets and gates.
Check additionally that the exact publisher-dev counts on the
198 multiword surnames and the 165 particle-led subset do not
regress. Report surname false positives on 12 fresh neutral particle
probes separately; these do not select the checkpoint, replace a
dev set or open a holdout. The runner asserts the cache and row
equivalence before fitting.

```sh
cargo build --locked --release -p turbo-trainer-recipes \
  --example pii_overlay_cache --example pii_contrast_fit
target/release/examples/pii_overlay_cache "$ENCODER" "$EXPERIMENT/legacy" "$EXPERIMENT/overlay"
target/release/examples/pii_contrast_fit \
  "$PUBLISHER_CORPUS" "$PUBLISHER_CACHE" "$REFIT_HEAD_SPEC" "$EXPERIMENT" --overlay
```
