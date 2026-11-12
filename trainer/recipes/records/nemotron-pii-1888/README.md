# Nemotron-PII 55-label BIO PII head, test split (#1888)

Receipts for the default-tier English contextual PII head trained in Rust
(`turbo-trainer`) on `nvidia/Nemotron-PII` at revision
`b70ffaf5ff39e079776134c5bf4381f00a9fd1ed` (`nemotron_pii::PARQUET_RELEASE`:
90,000 train, 10,000 dev carved by persona `uid` ordinal so `us` and `intl`
locale twins stay together, and 100,000 test documents; 18,710,937 test words
and 848,257 gold test spans across 55 PII and PHI types), all produced by
`turbo-train-head` at commit `735f516c7ca50c0db8cd25470bf3d70f239ab25a`:

| row | receipt | precision | recall | micro span F1 | vs majority (paired bootstrap, 95%) |
|---|---|---|---|---|---|
| gold | | 1.0000 | 1.0000 | 1.0000 | |
| majority baseline (each word's commonest training BIO tag) | `majority.json` | 0.1208 | 0.2740 | 0.1677 | |
| spaCy 3.8.7 `en_core_web_lg` 3.8.0 (`--map` on 4 shared types) | `spacy-lg.json` | 0.2028 | 0.0978 | 0.1319 | -0.0357 [-0.0376, -0.0340] |
| Stanza 1.10.1 `en` OntoNotes NER (`--map` on 4 shared types) | `stanza-en.json` | 0.4082 | 0.1126 | 0.1765 | +0.0088 [+0.0068, +0.0108] |
| our head (`answerdotai/ModernBERT-base` layer 22, first-piece pooling, 128-wide hashed word features) | `head.json`, `fit.json` | 0.5299 | 0.7176 | 0.6096 | +0.4419 [+0.4404, +0.4434] |

## Shared peer types (`ORG`, `DATE`, `TIME`, `LANGUAGE`)

Off-the-shelf OntoNotes NER models (`spaCy` `en_core_web_lg` and `Stanza`
`1.10.1-en`) share four non-ambiguous entity types with `Nemotron-PII`'s 55
labels (`ORG -> company_name`, `DATE -> date`, `TIME -> time`,
`LANGUAGE -> language`; OntoNotes `PERSON` spans full names while
`Nemotron-PII` splits `first_name` and `last_name`, and OntoNotes `GPE`/`LOC`
merges `city`, `county`, `state`, and `country`). On those 161,289 gold test
spans, our head outperforms the majority baseline and both peers on every
shared type:

| type | gold spans | majority F1 | spaCy `en_core_web_lg` F1 | Stanza `1.10.1-en` F1 | our head F1 |
|---|---:|---:|---:|---:|---:|
| `company_name` (`ORG`) | 54,826 | 0.1552 | 0.2708 | 0.5209 | 0.5425 |
| `date` (`DATE`) | 73,337 | 0.0246 | 0.3234 | 0.4809 | 0.6621 |
| `time` (`TIME`) | 24,489 | 0.0140 | 0.0990 | 0.2740 | 0.5323 |
| `language` (`LANGUAGE`) | 8,637 | 0.8216 | 0.7080 | 0.8157 | 0.9002 |
| **4-type micro subset** | **161,289** | **0.0809** | **0.2909** | **0.4832** | **0.6127** |

## Coarse group span F1 (`head.json`)

| coarse group | gold spans | found | precision | recall | F1 | majority F1 |
|---|---:|---:|---:|---:|---:|---:|
| `person` | 159,386 | 164,190 | 0.8666 | 0.8927 | 0.8794 | 0.7857 |
| `financial` | 54,772 | 63,533 | 0.7191 | 0.8341 | 0.7723 | 0.1143 |
| `location` | 99,084 | 117,826 | 0.7100 | 0.8443 | 0.7714 | 0.4815 |
| `temporal_demographic` | 177,612 | 231,509 | 0.5877 | 0.7661 | 0.6652 | 0.1283 |
| `medical_phi` | 38,573 | 50,528 | 0.5779 | 0.7570 | 0.6554 | 0.0235 |
| `organization_work` | 140,562 | 167,279 | 0.5211 | 0.6202 | 0.5664 | 0.1495 |
| `credential_network` | 32,515 | 77,595 | 0.2658 | 0.6344 | 0.3747 | 0.0063 |
| `government_id` | 23,629 | 49,208 | 0.2742 | 0.5710 | 0.3705 | 0.0070 |
| `contact` | 122,124 | 227,050 | 0.2650 | 0.4928 | 0.3447 | 0.0014 |

Note on `contact`, `credential_network`, and `government_id`: `SimpleTokenizer`
splits punctuation-heavy identifiers (`email`, `url`, `ssn`, `ipv6`) into
multiple word pieces where single-token boundary errors miss exact span match,
whereas `PiiDetector::pack(Pack::AllStructured)` scans raw UTF-8 bytes with
format and checksum verification and wins on overlap when paired via
`PiiTagger::with_rules`.

## All 55 per-type span F1s (`head.json`)

| Nemotron-PII type | gold | found | precision | recall | F1 | majority F1 |
|---|---:|---:|---:|---:|---:|---:|
| `account_number` | 16,693 | 24,192 | 0.4511 | 0.6537 | 0.5338 | 0.0369 |
| `age` | 7,135 | 8,346 | 0.7832 | 0.9162 | 0.8445 | 0.0000 |
| `api_key` | 4,667 | 16,790 | 0.1530 | 0.5505 | 0.2395 | 0.0000 |
| `bank_routing_number` | 8,354 | 7,699 | 0.9534 | 0.8786 | 0.9145 | 0.1531 |
| `biometric_identifier` | 11,378 | 12,473 | 0.8266 | 0.9061 | 0.8645 | 0.0217 |
| `blood_type` | 5,539 | 5,672 | 0.9362 | 0.9587 | 0.9473 | 0.0037 |
| `certificate_license_number` | 3,002 | 5,911 | 0.3206 | 0.6312 | 0.4252 | 0.0057 |
| `city` | 18,328 | 20,371 | 0.7012 | 0.7794 | 0.7382 | 0.6198 |
| `company_name` | 54,826 | 71,509 | 0.4792 | 0.6250 | 0.5425 | 0.1552 |
| `coordinate` | 7,659 | 12,396 | 0.4030 | 0.6522 | 0.4981 | 0.0000 |
| `country` | 23,332 | 23,941 | 0.9470 | 0.9718 | 0.9592 | 0.9028 |
| `county` | 8,357 | 10,159 | 0.6893 | 0.8380 | 0.7564 | 0.4196 |
| `credit_debit_card` | 12,867 | 14,683 | 0.8233 | 0.9395 | 0.8776 | 0.1714 |
| `customer_id` | 20,502 | 22,413 | 0.7741 | 0.8462 | 0.8085 | 0.1202 |
| `cvv` | 4,843 | 4,861 | 0.9459 | 0.9494 | 0.9477 | 0.0000 |
| `date` | 73,337 | 98,155 | 0.5784 | 0.7741 | 0.6621 | 0.0246 |
| `date_of_birth` | 18,079 | 23,943 | 0.5620 | 0.7442 | 0.6404 | 0.0000 |
| `date_time` | 13,221 | 22,552 | 0.2922 | 0.4984 | 0.3684 | 0.0000 |
| `device_identifier` | 2,507 | 12,231 | 0.0996 | 0.4858 | 0.1653 | 0.2941 |
| `education_level` | 9,345 | 10,108 | 0.7451 | 0.8059 | 0.7743 | 0.0918 |
| `email` | 53,928 | 90,252 | 0.2881 | 0.4822 | 0.3607 | 0.0000 |
| `employee_id` | 8,878 | 12,249 | 0.5808 | 0.8013 | 0.6735 | 0.1831 |
| `employment_status` | 10,941 | 12,247 | 0.7328 | 0.8203 | 0.7741 | 0.3396 |
| `fax_number` | 6,422 | 15,247 | 0.1452 | 0.3448 | 0.2043 | 0.0000 |
| `first_name` | 84,021 | 81,100 | 0.9410 | 0.9083 | 0.9243 | 0.8749 |
| `gender` | 7,645 | 7,431 | 0.9748 | 0.9475 | 0.9610 | 0.9576 |
| `health_plan_beneficiary_number` | 10,558 | 18,739 | 0.3706 | 0.6577 | 0.4740 | 0.0093 |
| `http_cookie` | 4,891 | 17,142 | 0.0953 | 0.3341 | 0.1483 | 0.0000 |
| `ipv4` | 6,078 | 7,923 | 0.6426 | 0.8376 | 0.7272 | 0.0000 |
| `ipv6` | 3,139 | 7,473 | 0.2197 | 0.5231 | 0.3095 | 0.0000 |
| `language` | 8,637 | 8,855 | 0.8891 | 0.9115 | 0.9002 | 0.8216 |
| `last_name` | 59,565 | 59,546 | 0.8907 | 0.8904 | 0.8905 | 0.7197 |
| `license_plate` | 4,078 | 5,425 | 0.5058 | 0.6729 | 0.5775 | 0.0116 |
| `mac_address` | 4,250 | 6,282 | 0.5498 | 0.8127 | 0.6559 | 0.0000 |
| `medical_record_number` | 11,098 | 13,644 | 0.4824 | 0.5931 | 0.5321 | 0.0420 |
| `national_id` | 2,847 | 5,615 | 0.1124 | 0.2216 | 0.1491 | 0.0371 |
| `occupation` | 36,070 | 38,753 | 0.3013 | 0.3237 | 0.3121 | 0.1188 |
| `password` | 6,983 | 9,754 | 0.5135 | 0.7173 | 0.5986 | 0.0485 |
| `phone_number` | 23,927 | 40,651 | 0.3715 | 0.6312 | 0.4677 | 0.0078 |
| `pin` | 6,456 | 5,617 | 0.8798 | 0.7655 | 0.8187 | 0.0422 |
| `political_view` | 6,339 | 8,384 | 0.5490 | 0.7261 | 0.6253 | 0.4236 |
| `postcode` | 6,277 | 6,388 | 0.8186 | 0.8330 | 0.8257 | 0.4040 |
| `race_ethnicity` | 9,229 | 10,429 | 0.7759 | 0.8768 | 0.8233 | 0.7572 |
| `religious_belief` | 6,240 | 6,712 | 0.7878 | 0.8474 | 0.8166 | 0.6018 |
| `sexuality` | 3,261 | 3,438 | 0.9398 | 0.9908 | 0.9646 | 0.9193 |
| `ssn` | 6,062 | 12,586 | 0.2633 | 0.5467 | 0.3554 | 0.0000 |
| `state` | 18,280 | 18,249 | 0.8497 | 0.8482 | 0.8490 | 0.6721 |
| `street_address` | 16,851 | 26,322 | 0.4480 | 0.6997 | 0.5462 | 0.0386 |
| `swift_bic` | 5,559 | 6,481 | 0.7142 | 0.8327 | 0.7689 | 0.0000 |
| `tax_id` | 1,302 | 2,846 | 0.1729 | 0.3779 | 0.2372 | 0.0095 |
| `time` | 24,489 | 33,264 | 0.4621 | 0.6277 | 0.5323 | 0.0140 |
| `unique_id` | 1,777 | 10,284 | 0.0532 | 0.3078 | 0.0907 | 0.0387 |
| `url` | 37,847 | 80,900 | 0.2033 | 0.4345 | 0.2769 | 0.0000 |
| `user_name` | 15,800 | 23,544 | 0.3594 | 0.5356 | 0.4302 | 0.0344 |
| `vehicle_identifier` | 4,561 | 6,541 | 0.5742 | 0.8235 | 0.6766 | 0.0001 |

## Temperature calibration (`calibration.json`)

Post-hoc temperature scaling fitted on the 10,000-document held-out `dev` split
(1,969,973 tokens, 380,348 non-`O` entity tokens; `test` is never touched):

- Fitted temperature: `T = 1.093758` (`turbo_tagger::NEMOTRON_PII_TEMPERATURE`)
- Token NLL: `0.09079 -> 0.09005`
- Overall 15-bin ECE: `0.00246 -> 0.00197`
- Entity-token 15-bin ECE: `0.04808 -> 0.03848`
- Token accuracy (invariant under positive `T`): `0.97505`

## Reproduce

```sh
turbo-train-head cache \
  --corpus nemotron-pii --dir corpora/nemotron-pii \
  --bundle bundles/modernbert-base --out cache/nemotron-pii \
  --device cuda --pooling first --layers 1 --max-batch 8 \
  --commit 735f516c7ca50c0db8cd25470bf3d70f239ab25a

turbo-train-head baseline \
  --corpus nemotron-pii --dir corpora/nemotron-pii \
  --out runs/majority \
  --commit 735f516c7ca50c0db8cd25470bf3d70f239ab25a

turbo-train-head fit-cache \
  --corpus nemotron-pii --dir corpora/nemotron-pii \
  --cache cache/nemotron-pii --out runs/nemotron-pii \
  --layer 22 --hash-embed 128 \
  --batch 128 --lr 0.004 --epochs 20 --patience 4 --threads 16 \
  --commit 735f516c7ca50c0db8cd25470bf3d70f239ab25a

turbo-train-head calibrate \
  --corpus nemotron-pii --dir corpora/nemotron-pii \
  --cache cache/nemotron-pii --run runs/nemotron-pii \
  --out runs/nemotron-pii/calibration.json --layer 22 \
  --commit 735f516c7ca50c0db8cd25470bf3d70f239ab25a

turbo-train-head export \
  --corpus nemotron-pii --dir corpora/nemotron-pii --out test.bio.tsv
# spaCy en_core_web_lg 3.8.0 and Stanza 1.10.1-en with nemotron-pii-raw:
MAP="B-ORG=B-company_name,I-ORG=I-company_name,B-DATE=B-date,I-DATE=I-date,B-TIME=B-time,I-TIME=I-time,B-LANGUAGE=B-language,I-LANGUAGE=I-language"
turbo-train-head peer \
  --corpus nemotron-pii --dir corpora/nemotron-pii \
  --peer spacy-3.8.7-en_core_web_lg-3.8.0 --predictions peers.spacy-lg-raw.tsv \
  --out runs/peer-spacy-lg --column bio --map "$MAP" \
  --commit 735f516c7ca50c0db8cd25470bf3d70f239ab25a
turbo-train-head peer \
  --corpus nemotron-pii --dir corpora/nemotron-pii \
  --peer stanza-1.10.1-en --predictions peers.stanza-en-raw.tsv \
  --out runs/peer-stanza-en --column bio --map "$MAP" \
  --commit 735f516c7ca50c0db8cd25470bf3d70f239ab25a
```

## In the zoo

`head.safetensors` is 6,201,372 bytes with SHA-256
`b8cb82e3708629e101990487add3dea807b2d56fb2cda15cbcfd06dcb054f9e6`, pinned in
`analysis/pins/models.txt`. `turbo-train-head assemble` packs the head into the
`answerdotai/ModernBERT-base` bundle (`manifest.json` SHA-256
`65a07202...`) as `turbo-en-nemotron-pii-modernbert-base-1` in
`analysis/zoo/index/zoo.json` (`KIND_HEAD`, `COMPONENT_PII_DETECTOR`, `en`,
`CLASS_PERMISSIVE`), assembled manifest SHA-256
`19d3fb87daac08f0361390809b747af638bc2dbb93e447695c3f5609fc735b6d`, with
`weights/model.safetensors` (604,836,336 bytes, SHA-256 `942c5aca...`) holding
the encoder weights and `heads.pii` tensors together.

```sh
turbo-train-head assemble \
  --bundle bundles/modernbert-base \
  --run runs/nemotron-pii \
  --out assembled/turbo-en-nemotron-pii-modernbert-base-1 \
  --zoo-id turbo-en-nemotron-pii-modernbert-base-1 \
  --component COMPONENT_PII_DETECTOR \
  --language en --version 1 \
  --license-terms https://www.apache.org/licenses/LICENSE-2.0 \
  --url-prefix https://github.com/ai-pipestream/turboembed/releases/download \
  --zoo-out assembled/zoo-entry.json
```

Each file in `zoo.json` points at its flat asset name under the release
`turbo-en-nemotron-pii-modernbert-base-1`.
