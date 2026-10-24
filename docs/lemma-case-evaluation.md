# Case-preserving lemma recipes

Issue #954 changes newly trained lemma models to retain annotated case. Published
OpenNLP models still use their original decoder. `LemmatizerFactory::opennlp`
selects that format explicitly for reproduction and JVM goldens; `new` selects
`turbo.lemmatizer.CaseSensitiveV1`. A JVM without that factory refuses the native
model rather than decoding its classes incorrectly.

The existing `models-ud-2.16-1` release and its recipe ids retain the
OpenNLP factory and original model bytes. The native models measured below
are local evaluation artifacts, not replacements for that release. Publishing
native UD models requires new model ids, a release tag and index entries.

Native classes learn either `K:` (original token) or `L:` (lowercase token), then
apply case-sensitive UTF-16 edits. Decimal edit indices end with `:`, including
indices above nine. The classifier learns case from the annotated lemma; there
is no capitalization rule based on POS. This permits both unseen proper names
and ordinary words at the start of a sentence to use an identity class.

## Method

The comparison uses the complete pinned UD r2.16 train and test splits, the
existing recipe sample reader, and gold POS tags for lemmatization. Multiword
rows are combined as in that reader and empty nodes are omitted. No samples
were skipped. Native and lowercase baselines both use GIS, 300 iterations,
cutoff 5 and 4 training partitions, with the same feature generator and beam.
The baseline is the code at `62dce738e7e4e81bc77d66c883cd3d562229591b`.
The published references are OpenNLP UD 1.3 models (2.5.4).

These measurements deliberately retain 4 partitions on both sides to isolate
the class-format change. The separate UD 1.3 reproduction fix in #953 uses
12 partitions for that protocol; the native-v1 comparison and the existing
UD 2.16 release recipes retain their recorded four-partition settings.
No held-out examples were added to training and no gate thresholds changed.

Full accuracy below includes all test tokens. Capital accuracy counts lemmas
that differ from the shared Java lowercase function. The proper-name subset
also requires gold POS `PROPN`. The evaluation example checks encode/decode
round trips for all annotated train and test pairs before scoring.

## Full held-out accuracy

| Treebank | Native correct / total | Native % | Lowercase baseline % | Published % |
| --- | ---: | ---: | ---: | ---: |
| Spanish AnCora | 51,947 / 52,454 | 99.0334 | 91.3181 | 90.5441 |
| Catalan AnCora | 57,230 / 57,894 | 98.8531 | 89.7416 | 89.7416 |
| Portuguese Porttinari | 30,942 / 31,451 | 98.3816 | 91.2753 | 81.1198 |
| Finnish FTB | 14,348 / 16,261 | 88.2357 | 88.2357 | 83.0576 |
| Italian MarkIT | 8,166 / 9,654 | 84.5867 | 82.3907 | 83.2608 |

| Treebank | Native capital correct / total | Native capital % | Native capital PROPN correct / total | PROPN % |
| --- | ---: | ---: | ---: | ---: |
| Spanish AnCora | 4,053 / 4,093 | 99.0227 | 3,991 / 4,018 | 99.3280 |
| Catalan AnCora | 5,295 / 5,323 | 99.4740 | 5,280 / 5,305 | 99.5287 |
| Portuguese Porttinari | 2,235 / 2,247 | 99.4660 | 2,189 / 2,199 | 99.5452 |
| Finnish FTB | 0 / 1 | 0.0000 | 0 / 0 | n/a |
| Italian MarkIT | 214 / 1,170 | 18.2906 | 208 / 210 | 99.0476 |

The lowercase baseline and published reference get zero capitalized lemmas
correct in each language. A subset with no tokens has no estimated accuracy.

## Existing paired gates

The reference-coverage mask is unchanged. In particular, capitalized lemmas
remain in the gate even though the old decoder cannot give them. The gate uses
10,000 paired sentence bootstrap resamples, seed 154, a 95% difference interval
and the existing -0.5 point noninferiority margin. Full accuracy above is also
shown so the coverage exclusions cannot conceal errors.

| Treebank | Excluded tokens | Native % | Published % | Difference points | 95% difference interval | Gate |
| --- | ---: | ---: | ---: | ---: | --- | --- |
| Spanish AnCora | 57 | 99.0916 | 90.6426 | +8.4490 | +8.0708 to +8.8326 | PASS |
| Catalan AnCora | 18 | 98.8838 | 89.7695 | +9.1143 | +8.6548 to +9.5876 | PASS |
| Portuguese Porttinari | 100 | 98.4913 | 81.3786 | +17.1127 | +16.5923 to +17.6190 | PASS |
| Finnish FTB | 177 | 88.8709 | 83.9716 | +4.8993 | +4.3473 to +5.4521 | PASS |
| Italian MarkIT | 33 | 84.7729 | 83.5464 | +1.2265 | +0.6012 to +1.8519 | PASS |

## Italian gate diagnosis

The lemma gate changes from FAIL (82.57% versus 83.55%, difference interval
-1.44 to -0.53 points) to PASS. The native model gets 208 of 210 capitalized
proper-name lemmas correct. It still misses most capitalized punctuation
labels: MarkIT uses `[PUNCT]` for 3 of 1,896 training punctuation tokens but
927 of 960 test punctuation tokens. Three occurrences are below cutoff 5.
All three models get only 33 of the 960 test punctuation lemmas correct. The
new encoding round-trips those gold labels, so this is not an inability to
represent their case. The annotation distribution is reported without changing
the gold data or fitting a rule to the test labels.

The POS model is unchanged and its gate remains FAIL: 89.96% versus 92.17%,
difference -2.21 points with interval -2.92 to -1.49. Over all 9,654 test tokens,
the native tagger gets 8,682 correct and the reference 8,895. Of the net 213 lost
correct tags, 123 are on the 1,335 tokens absent from the training vocabulary:
native 1,003/1,335 (75.13%), reference 1,126/1,335 (84.34%). On seen tokens the
counts are 7,679/8,319 (92.31%) and 7,769/8,319 (93.39%).

The native tagger reaches 18,677/18,759 (99.56%) on its training tokens, compared
with the reference's 17,408/18,759 (92.80%). This supports a generalization and
lexical-coverage explanation on the small 611-sample MarkIT training split.
It does not isolate data volume from domain differences or prove that tuning
would close the gap. No feature or scoring defect was found in this check.
The POS gate stays closed; the lemma fix does not justify publishing that POS
model.

## Reproduction and provenance

The in-tree `ud_native_lemmas` example reproduces the native-v1 protocol:
GIS, 300 iterations, cutoff 5 and four fixed partitions, with the native
case-preserving factory. It accepts canonical source lemma recipe ids and
writes distinct `lemmas-native-v1-local` output ids. The existing `ud`
command and every released recipe id retain compatibility training. Neither
the native ids nor their cards are registered as published recipes.

Fetch through the existing licence-aware zoo command. The native command
checks every selected treebank's licence and file digests, reference digest,
sample coverage and output directory before training. It runs the same
held-out/reference gate with unchanged limits and retains its report even
when it fails. The card records the actual four-partition protocol. Use a
saved lowercase baseline model directory for the comparison.

```sh
cargo run --locked --release -p turbo-train-recipes -- fetch-ud --home data
cargo run --locked --release -p turbo-train-recipes --example ud_native_lemmas -- \
  data/artifacts references native-models turbo-es-ud-ancora-2.16-lemmas-1
cargo run --locked --release -p turbo-train-recipes --example ud_case_eval -- \
  data/artifacts native-models lowercase-models references es

# For the Italian diagnosis, train its native lemma and existing POS recipe.
cargo run --locked --release -p turbo-train-recipes --example ud_native_lemmas -- \
  data/artifacts references native-models turbo-it-ud-markit-2.16-lemmas-1
cargo run --locked --release -p turbo-train-recipes -- ud \
  --treebanks data/artifacts --references references --out native-models \
  --recipe turbo-it-ud-markit-2.16-pos-1
cargo run --locked --release -p turbo-train-recipes --example ud_case_eval -- \
  data/artifacts native-models lowercase-models references it
```

The Italian POS command exits unsuccessfully because its gate fails,
while retaining the gate report. `ud_case_eval` emits aggregate counts and
SHA-256 values only. No corpus text, model binaries or raw per-sample records
are committed here. The Portuguese text rights remain unsettled and its
artifact publication guard remains in place. Italian VIT reference weights are
used locally for evaluation under their existing restricted terms.

The in-tree native command was rerun on the complete Spanish AnCora split:
14,287 training samples, 442,892 events and 1,721 held-out samples, with no
skipped sentences. Its 1,923,528-byte package reproduces the recorded
`cc96098192bfce38e11d67cd186b98c7595c94ed9590f52ab6efd70c5f822cd0` digest.
The updated evaluator reads the distinct local output id and reproduces
51,947/52,454 correct lemmas (99.03343882%), including 4,053/4,093 capital
lemmas and 3,991/4,018 capital proper-noun lemmas. The unchanged gate omits
the same 57 reference-class exclusions and passes at 99.09155104% candidate
versus 90.64259404% reference accuracy. Those gate scores are separate from
the all-token figures. The other four language receipts below were not
regenerated for this reproduction-path correction.

The published Spanish recipe was also retrained through the existing `ud`
command. Its 1,828,462-byte package matches the released
`e00c951c6f60b5907f15ffa475a6a013483ba2b8c370cfc261c1c70114e70c61`
digest and passes its unchanged gate. The release registry is unchanged.

| Treebank | Train samples / tokens | Test samples / tokens | Successful round trips |
| --- | ---: | ---: | ---: |
| Spanish AnCora | 14,287 / 442,892 | 1,721 / 52,454 | 495,346 |
| Catalan AnCora | 13,123 / 417,117 | 1,846 / 57,894 | 475,011 |
| Portuguese Porttinari | 5,893 / 110,566 | 1,683 / 31,451 | 142,017 |
| Finnish FTB | 14,981 / 127,359 | 1,867 / 16,261 | 143,620 |
| Italian MarkIT | 611 / 18,759 | 340 / 9,654 | 28,413 |

SHA-256 values identify the complete input files and model packages.

### Spanish AnCora

Source recipe: `turbo-es-ud-ancora-2.16-lemmas-1`.
Local native id: `turbo-es-ud-ancora-2.16-lemmas-native-v1-local`.

- Train: `79e7463700e75bdf921c85b6e04495af118462a2b032bbff590fb82aa6ad8d5d`
- Test: `2a24316503a56e171f07b38cf2ec13c22e3b62a337c6dbf2d9d1a54ada2b012b`
- Native model: `cc96098192bfce38e11d67cd186b98c7595c94ed9590f52ab6efd70c5f822cd0`
- Legacy model: `e00c951c6f60b5907f15ffa475a6a013483ba2b8c370cfc261c1c70114e70c61`
- Reference model: `d9970825f87e95fbc4c54e7c35cb2af141a29fd65f21908d86e29ef9af2da914`

### Catalan AnCora

Source recipe: `turbo-ca-ud-ancora-2.16-lemmas-1`.
Local native id: `turbo-ca-ud-ancora-2.16-lemmas-native-v1-local`.

- Train: `bb5d06fb14336bae2c3aa64aa7e65afe183295a4669ce4f45bf0ed57c18ccccd`
- Test: `f68ff102b9bf8bc197be3262a1caf25682683c4db4c648663c7a791bfbfa115f`
- Native model: `d680446b4d783397402a990683eaa6de8fe69b122f8dfa876107166852ca517b`
- Legacy model: `86f07881a6f325253b7e285b915aecbf88d7b2fc1d280b6d4032ea9689ff3cf1`
- Reference model: `b8d537720f11a79b3429e1957404dc1bf7fcaf838ca79cc097d6d634e388f094`

### Portuguese Porttinari

Source recipe: `turbo-pt-ud-porttinari-2.16-lemmas-1`.
Local native id: `turbo-pt-ud-porttinari-2.16-lemmas-native-v1-local`.

- Train: `d9d3d76ac9ac8cb569c42a698cc60721e088db7c7e90401582aab68aa9ca85d3`
- Test: `08e51c3687f2800ea45445cb48c7030832984c316b1c39757606499b8e7577b9`
- Native model: `b161884b824a46202405e5e0858c08db194bf8c25b8cb4eb977de81623ffef47`
- Legacy model: `65c23e16a75c48d33d96d1311432a86218a3ba3cff7eabb75eb2bcb1752d1d91`
- Reference model: `279f23d9d0a2ed0828d13e812f2db1220070b8f6692ee82af44f7e0d6c2fcf6c`

### Finnish FTB

Source recipe: `turbo-fi-ud-ftb-2.16-lemmas-1`.
Local native id: `turbo-fi-ud-ftb-2.16-lemmas-native-v1-local`.

- Train: `0cc04932f18613c4491ace06f597b7b8caed36e557168b9b6ef154da7ae2d71b`
- Test: `3aca7fbd727846c0955e9db581571a0fca89d46f825ee7f2a32fab1ea0bc31ba`
- Native model: `b174f29c45e860f127318bba3484861dfc291f14c09b659b6e1f9b50d8ace75a`
- Legacy model: `9650fcdeb459cea1565175cdf37ee1a8d90306414f287b5205e1e25b421b6278`
- Reference model: `d9ef61db782abc1a60285445a228b4aacda7dee7b853aae02740fe1979e4d268`

### Italian MarkIT

Source recipe: `turbo-it-ud-markit-2.16-lemmas-1`.
Local native id: `turbo-it-ud-markit-2.16-lemmas-native-v1-local`.

- Train: `4f6fc3423acb00d49e1e167bd21b11ebbc23e22275fd6f5eeb4d75cfbbccc5c6`
- Test: `9048ed0064e3989e2c37377755a28c25c7e462b1ad916e5808dc0a7ce95fc3fc`
- Native model: `41f4ad8d45199e8e515050b72d15964c036d03200c436576eea8dfc400d67983`
- Legacy model: `5ba6ea4a0310397f68e92ef3a86dbd03ebad3f0dbb096f4737122c5c0a68c69a`
- Reference model: `10b25bc4506af3a5f06e56b9e1058aecccb3021e5d08902a1fa1ee33606510c7`

Dataset releases and terms: [Spanish AnCora](https://github.com/UniversalDependencies/UD_Spanish-AnCora/tree/r2.16),
[Catalan AnCora](https://github.com/UniversalDependencies/UD_Catalan-AnCora/tree/r2.16),
[Portuguese Porttinari](https://github.com/UniversalDependencies/UD_Portuguese-Porttinari/tree/r2.16),
[Finnish FTB](https://github.com/UniversalDependencies/UD_Finnish-FTB/tree/r2.16),
and [Italian MarkIT](https://github.com/UniversalDependencies/UD_Italian-MarkIT/tree/r2.16).
The recipe registry pins each file size and digest; its existing licence checks
apply before training or artifact publication.
