# Nemotron-PII head refitted with shared-name sentences (#2398)

A candidate head, `turbo-en-nemotron-pii-modernbert-base-2`, **not
released**. It is the #1888 recipe (`../nemotron-pii-1888`) on the corpus
`nemotron-pii-names`: `Nemotron-PII` at revision `b70ffaf5` plus `6,000`
train and `600` dev project-written documents on names a person and a
city share (`trainer/recipes/src/nemotron_pii/shared_names.rs`,
Apache-2.0). Run on 2026-11-18 at commit
`d91effb51d7bd26f868d1b762a6ad908218912d6` on the 4080 machine.

## Commands

`$TRAIN_HEAD` is `turbo-train-head` built with `--release --features
cuda`; `corpora/nemotron-pii` is the dataset checkout and
`bundles/modernbert-base` the `answerdotai/ModernBERT-base` bundle
(manifest `65a07202...`); `$C` is the commit above.

```
$TRAIN_HEAD cache --corpus nemotron-pii-names --dir corpora/nemotron-pii --bundle bundles/modernbert-base --out cache-v2/nemotron-pii-names --device cuda --pooling first --layers 1 --max-batch 8 --commit $C
$TRAIN_HEAD fit-cache --corpus nemotron-pii-names --dir corpora/nemotron-pii --cache cache-v2/nemotron-pii-names --out runs-v2/nemotron-pii-names --layer 22 --hash-embed 128 --batch 128 --lr 0.004 --epochs 20 --patience 4 --threads 16 --commit $C
$TRAIN_HEAD calibrate --corpus nemotron-pii-names --dir corpora/nemotron-pii --cache cache-v2/nemotron-pii-names --run runs-v2/nemotron-pii-names --out runs-v2/nemotron-pii-names/calibration.json --layer 22 --commit $C
$TRAIN_HEAD assemble --bundle bundles/modernbert-base --run runs-v2/nemotron-pii-names --out assembled-v2/turbo-en-nemotron-pii-modernbert-base-2 --zoo-id turbo-en-nemotron-pii-modernbert-base-2 --component COMPONENT_PII_DETECTOR --language en --version 2 --license-terms https://www.apache.org/licenses/LICENSE-2.0 --url-prefix https://github.com/ai-pipestream/turboembed/releases/download --zoo-out assembled-v2/zoo-entry.json
```

## Run

| | |
|---|---|
| train / dev / test sentences | `96,000` / `10,600` / `100,000` |
| train / dev / test words | `17,841,931` / `1,990,960` / `18,710,937` |
| feature cache | layer `22`, `630.3` s on cuda |
| fit | seed `1`, batch `128`, lr `0.004`, `20` epochs, patience `4`; `1,795.8` s |
| best epoch | `20` of `20`, dev span F1 `0.6387` (still rising at the cap) |
| temperature | `1.0807142165565866` (dev NLL `0.08559` to `0.08507`, entity ECE `0.04112` to `0.03281`) |
| `fit.json` | `8,602` B, SHA-256 `8e3259db69958afd0befaef4aa27b22f9bd0d7ae77c1d4e8abf37224858f4370` |
| `calibration.json` | `1,648` B, SHA-256 `63598b6d76d92fe9ae37a821bcc0d470f561d96d8aeb9ac2632ed641eee4f3c0` |
| `head.safetensors` | `6,201,372` B, SHA-256 `d944a8559d8156bbb3c35c23e5b028a59bef28a91057a9cb12c6c674ec625b42` |
| bundle manifest | `100,191` B, SHA-256 `dd73330bc8bef5bc0f1b9c7b6c004cfbf6713fa1dadcc03a6238a159b6ddc3b7` (`analysis/pins/models.txt`) |

The receipts themselves stay with the run, as they record the paths of
the machine; their hashes are above.

## Scores

`Nemotron-PII` test spans, against the released head of #1888:

| | released | refit |
|---|---|---|
| precision | `0.5299` | `0.5432` |
| recall | `0.7176` | `0.7180` |
| F1 | `0.6096` | `0.6185` |
| `first_name` F1 | `0.9243` | `0.9258` |
| `last_name` F1 | `0.8905` | `0.8872` |
| `city` F1 | `0.7382` | `0.7287` |
| `state` F1 | `0.8490` | `0.8609` |
| `company_name` F1 | `0.5425` | `0.5111` |

The person-name memos of `turbo-pii` and the single sentences
(`analysis/tagger/tests/pii_places_person.rs`,
`pii_shared_name_sentences.rs`, run at the tests' temperature
`1.093758`), as city true, false, missed:

| | released | refit |
|---|---|---|
| `places-person-names` | `13`, `4`, `0` | `13`, `4`, `0` |
| `places-person-names-2` | `8`, `5`, `0` | `8`, `3`, `0` |
| `places-person-names-3` (rules alone `8`, `6`, `0`) | not run | `7`, `4`, `1` |
| `22` sentences on a city, head / with rules | `4` / `7` | `19` / `20` |
| `20` sentences on a person, head / with rules | `10` / `10` | `9` / `9` |

A first refit on an earlier mix (each name drawn alone, no joined
sentences; candidate manifest `f2784204...`) scored F1 `0.6166` and
lost three city uses of the memos after `from` and `to`; it was
dropped.
