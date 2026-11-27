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

### Row-matched overlay result (rejected)

The CPU run at `b124714490b4e7f1ed8a51299b9233d152725b6b` selected
epoch 6 and wrote a 6,201,308-byte head (SHA-256
`b75a793608ca52e2b8815361c2186c5648e86f58537857e8f31fa7c694bfa981`).
The training runner verified all 5,400 unedited encoded rows
byte-for-byte and all 600 edits' word and non-`O` counts per index.
The independent loss accounting agreed exactly by seeded batch:
209,985 synthetic words, 28,375 synthetic BIO-labelled words and
223 steps per epoch. The label distribution moved as intended:
synthetic city spans 14,288 to 13,221, first-name spans 14,087
to 13,069, and company and surname spans each 0 to 600.

| Dev or diagnostic | Frozen control | Overlay |
| --- | ---: | ---: |
| Publisher micro F1 | 0.5725 | 0.5724 |
| Publisher city F1 | 0.6787 | 0.6796 |
| Publisher first name F1 | 0.8778 | 0.8768 |
| Publisher last name F1 | 0.8270 | 0.8175 |
| Publisher company name F1 | 0.7712 | 0.7710 |
| Publisher multiword surnames, exact | 108/198 | 103/198 |
| Publisher particle-led surnames, exact | 99/165 | 94/165 |
| Historical synthetic city F1 | 0.8617 | 0.8590 |
| Historical synthetic first-name F1 | 0.8594 | 0.8571 |
| New disjoint synthetic city F1 | 0.3730 | 0.3728 |
| New disjoint synthetic first-name F1 | 0.2140 | 0.2284 |
| New disjoint synthetic company F1 | 0.0259 | 0.0686 |
| New disjoint synthetic surname F1 | 0.0000 | 0.0000 |
| Neutral particle probes with a surname false positive | 1/12 | 4/12 |

The overlay restores much of the historical city/person and publisher
surname performance lost by wholesale replacement, but the frozen
control still does better on those safeguards. New four-role macro
F1 improved by 0.0143, below the existing 0.02 bar; loss of real
publisher multiword spans and additional false positives are distinct
from the artificial unseen-particle stress. The same dev gate again
exited 2. No serving calibration, holdout scoring, or promotion ran.

A better next direction would draw varied real multiword surname
positives and same-token context negatives from publisher **training**
and match class support rather than adding synthetic particles alone.
Measure exact boundaries on the frozen publisher dev strata and
neutral-particle errors first; the multiword decoder itself should
not be changed to force a match.

### Registered bounded chooser probe

The separate type probe considers **only** an already decoded city or
first-name span where the opposite type is in the same refit head's
two highest grouped scores at those exact boundaries. Same-head inputs
are grouped probabilities and fixed projections of context states,
not names, token lists, or gold type features. It fits a small logistic
chooser for each direction. All boundaries and all other typed
spans must be identical to the native refit.

The fit uses 30,000 publisher training rows; calibration uses 15,000
other rows. Both locales of a publisher persona stay together.
A direction is disabled unless held-in calibration reaches at least
90% correct and at least ten exact fixes. After locking these choices,
report raw publisher dev TP/FP/FN exactly once. Both city and
first-name F1 must meet **both** native references and unrelated
outputs cannot change. This is a partial probe, not a reason to open
the unchanged serving holdout or claim company or surname recovery.

### Publisher boundary and role-support ceilings

A deterministic oracle uses gold on publisher **dev** only to count
possible exact repairs. It is not a serving rule. A strict repair may
extend an existing surname by one token only if the added token's
native tag is `O`: it cannot overwrite another typed span. This is
an optimistic ceiling at the fixed raw prediction count. The actual
catalog and previously refitted bundles were loaded by their pinned
manifests; the pilot heads were loaded by their pinned file hashes.

| Head | Raw surname F1 | Strict left gains | Strict right gains | F1 ceiling, strict either direction |
| --- | ---: | ---: | ---: | ---: |
| Catalog v1 | 0.8857 | 2 | 3 | 0.8869 |
| Refit v2 | 0.8814 | 1 | 2 | 0.8821 |
| Bounded control | 0.8270 | 1 | 5 | 0.8284 |
| Four-role replacement | 0.8158 | 0 | 4 | 0.8167 |
| Row-matched overlay | 0.8175 | 0 | 4 | 0.8185 |

Even ignoring typed collisions, refit left extension recovers just
11 matches and reaches 0.8840, below the catalog's native result.
Relaxed right extension can numerically cross the catalog's raw F1 but
would overwrite existing typed tokens, violating the intervention
constraint. No boundary classifier was fitted. The artificial dev
overlay would have 180 strict left-extension opportunities; its
publisher counterpart has zero. Both are raw BIO tests, not results
after serving calibration, score admission or Places rules.

A separate oracle scores the actual packaged heads' native spans and
calibrated mean grouped BIO probabilities at those **same** boundaries.
`O` occupies a ranking position but is never emitted. Adding ranks is
an oracle support set, not a threshold or promotion. Refit's missed
gold cases on publisher dev:

| Type | Right refit boundary, wrong type | Of those, in top 2 | In top 3 | No exact refit boundary |
| --- | ---: | ---: | ---: | ---: |
| City | 197 | 124 | 152 | 246 |
| First name | 310 | 200 | 258 | 147 |
| Last name | 279 | 142 | 207 | 205 |
| Company name | 28 | 15 | 25 | 1,055 |

Native union alone is not a safe chooser: city F1 is 0.7541 on the
union compared with 0.7581 for the refit; company is 0.8156 vs
0.8256, surname 0.8815 vs catalog 0.8857. Emitting every refit
top-2 alternative adds 2,443 city false positives for 124 true
positives and 1,687 first-name false positives for 200 true positives.
For refit-native first-name spans with city in top 2, only 18 of
600 should switch, while 549 are already correct. For the reverse
direction, 29 of 293 should switch and 239 are already correct.
The resulting ideal selector has only narrow headroom. Company
mostly needs better boundaries, not the city/person selector.

```sh
cargo run --locked --release -p turbo-trainer-recipes --example pii_surname_audit -- \
  "$EXPERIMENT" "$ENCODER" "$PUBLISHER_CORPUS" "$PUBLISHER_CACHE" "$CATALOG_BUNDLE" "$REFIT_BUNDLE"
cargo run --locked --release -p turbo-trainer-recipes --example pii_role_oracle -- \
  "$PUBLISHER_CORPUS" "$PUBLISHER_CACHE" "$CATALOG_BUNDLE" "$REFIT_BUNDLE" --probe
```

### Bounded chooser result (rejected)

The frozen probe at `f05afd44be0745c6b297b97ec20fa2d750a580e2`
could not lock a direction on its untouched training-persona
calibration partition:

| Direction | Fit eligible / true fixes | Calibration eligible / true fixes | Locked threshold |
| --- | ---: | ---: | --- |
| First name to city | 1,949 / 25 | 980 / 12 | Disabled |
| City to first name | 823 / 33 | 400 / 17 | Disabled |

Neither direction found at least ten correct flips with 90% precision.
As designed, the dev replay made zero changes. TP / FP / FN for city
stayed 1,562 / 554 / 443 and first name stayed 5,243 / 506 / 457,
the exact native-refit values. Every other type and every boundary were
unchanged. City F1 remained below the catalog; the probe did not pass.
No serving or reserved holdout evaluation ran.

The actionable data gap is reliable, exact-boundary role ambiguity:
more real city versus person cases where the frozen head's alternative
really is correct, with counterexamples where the native type must
stay. For another selection experiment, label these cases on a new
publisher-training or separately approved corpus, reserve whole name
and context families for calibration, and leave independent dev/test
unchanged. Company and surname also need separate real span-boundary
data. These probes do not support changing the current decoder or
admitting more alternatives by default.

### Catalog-preserving additive city audit (no eligible additions)

The catalog+Places analyzer stays untouched. A proposed extra span
must be an exact city in all three signals: independent Places,
the calibrated refit head at the 0.50 serving bar, and that refit
with Places. Any intersection with any catalog mention, including
person, company or city, rules it out. Standalone mode has no
proposal path. This experiment does not install an inference policy.

The bundles could share encoder states in principle. The SHA-256 of
137 identical non-head tensors (sorted name/dtype/shape and exact
payloads) is
`6827f1649bcfdffc4e75da97fa001aadb32542bef05c81da6127784d811885c2`.
Both tokenizers have digest
`9fd55248d51d33976b324fc11592e28071da7d41e0e9401dfb7082e30574b7b1`.
Both use the same ModernBERT architecture, layer 22, width 768,
first-subword pooling and no layer mix. The measured audit still ran
the existing CPU analyzers separately; it did not implement sharing.

The full 10,000-row publisher dev cache (2,005 gold cities) yielded a
broad shortlist of 294 proposal spans in 234 rows. This cached
prescreen used 0.49 for prospective refit cities and omitted clear
catalog city duplicates (0.55 or above); it is not an exhaustive
uncached serving pass. Exact independent Places rejected 251 spans:
20 had overlapping but different ranges, and 231 had no matching
Places span (60 of those were gold). The real CPU analyzers replayed
the 43 survivors in 38 rows at their own temperatures and the 0.50
serving bar. One gold candidate failed neural admission; catalog+Places
already emitted all other 42 (36 gold and six not gold). Thus the
audited gate adds **0 true and 0 false** city spans. Of 16 fresh
ambiguous and neutral examples, five had independent Places support;
four were again catalog duplicates and one did not pass the refit
neural bar. No new city or protected-span changes occurred.

An acceptor cannot learn to select positives that never enter this
gate. The same independent Places support is already used by the
catalog, so requiring it is not an independent rescue for these
cases. The next dataset would need genuine ambiguous city mentions
that Places currently does not recognize, with human-labelled
non-city lookalikes and existing person/company spans preserved. It
would require a separately authorized acceptance design, not just a
looser overlap or confidence rule. The original probes, all memos,
and publisher test remained closed.

```sh
cargo run --locked --release -p turbo-trainer-recipes --example pii_city_shortlist -- \
  "$PUBLISHER_CORPUS" "$PUBLISHER_CACHE" "$CATALOG_BUNDLE" "$REFIT_BUNDLE" > "$CITY_SHORTLIST"
cargo run --locked --release -p turbo-tagger --example pii_city_rescue_audit -- \
  "$CATALOG_BUNDLE" "$REFIT_BUNDLE" "$CITY_SHORTLIST"
```

### What the pinned publisher training corpus actually supports

This inventory uses publisher `TRAIN` alone. The dataset is itself a
published synthetic corpus, but these are its existing gold labels,
not examples invented for this experiment. Duplicate local contexts
are normalized over five words on either side, masking entities
and numbers. Both locales of a persona count once where diversity
is reported. For shared single-token surfaces, bipartite matching
requires a distinct persona and context for each selected occurrence
within each role and surface; the capacities are sums of per-surface
minima, **not** a global cross-role/cross-surface persona-disjoint split.

| Shared normalized surface | Any support | At least two in each role | Balanced pairs at that floor | At least five in each role | Pairs at that floor |
| --- | ---: | ---: | ---: | ---: | ---: |
| City vs first name | 168 | 61 | 234 | 14 | 114 |
| City vs last name | 286 | 77 | 225 | 5 | 43 |
| City vs either person type | 385 | 119 | 418 | 19 | 162 |

There are 1,864 multiword surnames across 1,252 personas and 1,486
masked contexts. The `al` particle has 1,213 gold surname uses across
808 personas and 978 contexts, and 112 unlabelled uses across 64
personas and 60 contexts. This supports an `al` boundary experiment
with genuine negatives. It does not support an assertion about
generalizing to `del` and `von`: there are only five gold `del`
instances across two personas, and no gold `von` instances. The
reserved unseen evaluation and its gate stay unchanged.

For company boundaries, the frozen refit ran over 15,000 other
publisher training rows, grouped by persona and excluded from the
chooser's training and calibration slices. There were 9,873 gold
multiword company spans and 955 misses: 928 same-type partials, two
other-type overlaps and 25 uncovered. The misses span 693 personas
and 939 masked contexts. The company start token also appears as a
gold single-token first name for 31 surfaces and 48 missed-company
instances, with 288 first-name negatives. Only three surfaces have
at least two independent persona/context marginals in both roles
(upper capacity 13 pairs); only one has five (upper capacity eight).
There is substantial ordinary boundary training support, but the
common-name/company contrast is too sparse to treat as a robust
independent class-disambiguation objective.

The next candidate worth testing would warm-start both arms from
the same catalog checkpoint. Keep the full pinned publisher
objective and its negatives in the final fit; add only a small,
separately accounted rehearsal term for authentic same-token city vs
person contexts, attested multiword boundaries (especially `al` with
neutral and typed negatives), and ordinary partial company spans.
Do not replace the publisher mix to manufacture contrasts. Select
the training objective in internal publisher TRAIN folds: reserve
whole persona groups and normalized ambiguity surfaces and exclude
their target occurrences from both arms before calibration; audit
cross-fold intersections and recompute actual support after splitting.
Use the same replay, bounded streamed batches, schedule and budget
in both arms. A matching control still has to meet the shipped
baseline. Keep the existing dev and serving/test comparisons and
protected-class gates. This is a proposal, not a new fit or passing
candidate; it needs bounded streamed replay rather than the current
full-cache materialization.

```sh
cargo run --locked --release -p turbo-trainer-recipes --example pii_publisher_train_inventory -- "$PUBLISHER_CORPUS"
cargo run --locked --release -p turbo-trainer-recipes --example pii_company_train_inventory -- \
  "$PUBLISHER_CORPUS" "$PUBLISHER_CACHE" "$REFIT_BUNDLE"
```

### Registered bounded warm-start replay

One paired experiment starts both arms from the identical pinned
catalog weights. Each sees every original publisher TRAIN row once.
The only difference is matched supplementary examples: at most 24
pairs in each of six attested cases (city, first name, clipped
multiword `al` surname, neutral unlabelled `al`, partial full
company, and first-name contexts with a false company mention).
Preflight requires at least eight pairs per case, globally unique
source personas among supplementary rows, and exact word and
non-`O) counts within every pair. Each supplemental row is repeated
four times and interleaved at the same indices in both arms. The
ordinary control rows share the case's gold role or neutral channel
and exact loss denominators; target rows are selected from pinned
TRAIN labels and original catalog errors, never publisher dev.

Schedule: one pass per arm, 128 documents/update, fixed seed 2596
over sequential cache chunks, AdamW with initial learning rate
0.00008, decay 0.01, clipping 5, no warmup. Six low-priority CPU
workers and a 32 GiB cap; read one batch from the layer-22 feature
file rather than expanding the full corpus. No early stopping or
learning-rate selection. The TRAIN inventory is descriptive only:
full replay sees its rows, so none of those checks is called an
independent dev partition. The independent gate remains publisher
dev. Native raw exact-span precision and recall (micro and each
of city, first name, surname and company) must not drop below
either catalog or the matched replay control. City must improve
over control and neutral particle/company false positives may not
increase relative to either. No gates are supplanted: a survivor
would still require the previously reserved original serving
checks and corpus gates before changing anything in production.

The initial one-sixth TRAIN prescreen supplied only two exact-matched
city pairs, so it stopped before fitting. The frozen preflight scans
all shared city/person and rare `al` TRAIN rows, retaining the fixed
persona sixth for companies. No match constraint or dev gate changed.

The expanded preflight found 133 word/BIO-matched pairs (22 city,
24 given name, 15 `al` surname, 24 neutral `al`, 24 partial company,
24 false company). Both arms therefore have 90,000 original rows
and 532 supplementary rows, with 708 matched updates.

```sh
cargo run --locked --release -p turbo-trainer-recipes --example pii_warm_replay -- \
  "$PUBLISHER_CORPUS" "$PUBLISHER_CACHE" "$CATALOG_BUNDLE" "$REPLAY_PLAN" --plan
cargo run --locked --release -p turbo-trainer-recipes --example pii_warm_replay -- \
  "$PUBLISHER_CORPUS" "$PUBLISHER_CACHE" "$CATALOG_BUNDLE" "$REPLAY_PLAN" --fit "$REPLAY_OUTPUT"
```
