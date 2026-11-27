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
