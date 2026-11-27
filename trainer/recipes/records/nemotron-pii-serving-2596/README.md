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
contrasts and replay of original organization and surname contexts. Use
a dev slice that holds out names **and** sentence templates, alongside
publisher dev, before trying the frozen tests. Keep the previously
measured city gains, all three memo error counts, standalone person
results, and corpus class scores as separate acceptance gates.
