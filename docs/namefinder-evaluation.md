# Local NameFinderME evaluation (#177)

The local CoNLL-2003 English test evaluation below records the corpus
comparison for #177. It uses the approved local test set and existing
1.5 English reference models; no training or model release is involved. No
Reuters text or annotation records are distributed. The MASC recipes and
their accuracy gates (#427) are unchanged; the FineWeb model publication
step (#429) is separate.

## Evaluation protocol

Set `MODEL` and `CORPUS` to files the caller is entitled to use. `TYPES` is
`all`, `per`, `org`, `loc` or `misc`. This filters gold labels only, matching
the CoNLL reader; model predictions of other types still count as
false positives. Use a matching label inventory for meaningful accuracy.
No label remapping or merge of independent finders is performed.

Records include SHA-256 of the complete model and corpus bytes, selected
gold types, token counts, document reset flags, gold spans, predicted spans
and probabilities. They contain no source text or workstation paths. The
record still discloses annotation positions and labels; treat results in
accordance with the input data agreement. Corpus digests are sensitive to
line endings and byte order marks, deliberately preventing comparison of
different inputs under one claimed run.

The comparator checks model and corpus identity, sample order, reset flags,
gold labels and token counts, then requires identical predicted spans and
types and absolute probability error at most `1e-9`. Equal aggregate F1
alone cannot pass. Non-finite scores, invalid probabilities, empty corpora,
empty or out-of-range spans, overlapping predictions and duplicate mentions
are rejected. Adaptive features carry across sentences and reset at document
boundaries as requested by `Conll03NameSampleStream`. The F1 sentinel
`-1` is preserved when precision and recall are both zero.


## CoNLL-2003 English test results

Run on 2026-10-22 with the locally supplied English test split (`eng.testb`,
stored as `test.txt`). Each of the three model runs read the same **3,453
sentences and 46,435 tokens**, with **231 adaptive-data reset flags** from
`Conll03NameSampleStream`. Document markers are not scored as tokens.

The native harness was built in release mode from TurboEmbed commit
`ad2afc6acf43310c60bbcd9f2944f35205bb79a0`, with `--locked`.

### Input identity

| Input | SHA-256 |
|---|---|
| CoNLL-2003 English test bytes | `dc4ac7f3d6d01a6bec59261814561da3b287b6b493b97d1f69c580b52ec1831b` |
| `en-ner-person.bin` | `687a9263d96b37fced707c9f2ac0560f9edaf54658856395555901924f64dbe4` |
| `en-ner-organization.bin` | `0136c12afe1ac357142260c39bb879b7c9d121e41024114db5a6455b4fd5ba00` |
| `en-ner-location.bin` | `8fe39e48633f4a86c4132d9c54b396a2d8e0460c1d71e3562dacf976984f447b` |

### Exact-match entity scores

Each model was evaluated independently with its matching gold-type filter:
`per`, `org`, or `loc`. Precision, recall and F1 below are micro scores on a
0-to-1 scale, calculated from each runtime's prediction record by the same
harness evaluator. A correct entity must match both token boundaries and
type. Counts make the ratios reproducible: precision is correct/predicted,
recall is correct/gold, and F1 is twice correct/(predicted + gold).

| Type | Runtime | Gold | Predicted | Correct | Precision | Recall | F1 |
|---|---|---:|---:|---:|---:|---:|---:|
| `per` | TurboEmbed | 1617 | 719 | 618 | 0.8595271210013908 | 0.3821892393320965 | 0.5291095890410958 |
| `per` | Reference | 1617 | 719 | 618 | 0.8595271210013908 | 0.3821892393320965 | 0.5291095890410958 |
| `org` | TurboEmbed | 1661 | 494 | 251 | 0.5080971659919028 | 0.15111378687537627 | 0.23294663573085847 |
| `org` | Reference | 1661 | 494 | 251 | 0.5080971659919028 | 0.15111378687537627 | 0.23294663573085847 |
| `loc` | TurboEmbed | 1668 | 1167 | 911 | 0.7806341045415596 | 0.5461630695443646 | 0.6426807760141093 |
| `loc` | Reference | 1668 | 1167 | 911 | 0.7806341045415596 | 0.5461630695443646 | 0.6426807760141093 |

For all three runs, the comparator passed on every sample: model and corpus
hashes, gold types, sentence order, token counts, reset flags, gold spans,
and predicted spans/types matched. Maximum absolute entity-probability
error was **0** for `per`, **0** for `org`, and **0** for `loc`, below the
required `1e-9`. Agreement is checked before the native result is written;
it is not inferred from matching aggregate F1.

These are accuracy measurements for the three classical single-type models,
not a combined four-type CoNLL score. No `misc` model, cross-finder merge,
label remapping, training, tuning or threshold selection was performed.
The low recall, especially for organizations, occurs in both runtimes.
Parity establishes faithful execution of these models, not their suitability
as a new accuracy baseline.

### Reproduction and data handling

Use the commands above once per model, with `TYPES=per`, `org` and `loc`,
and a distinct output record for each run. The corpus and model digests
must match the table before comparing results. Preserve document boundaries
so adaptive features reset in the same places.

Local evaluation was approved under the CoNLL-2003 entry in
[licences.md](licences.md). The corpus and the classical model binaries remain
external inputs. Only hashes, counts and aggregate metrics are included in
this report; no Reuters text, gold/predicted span records, or workstation
paths are committed. The raw comparison records are local scratch output,
not publication artifacts. This evaluation grants no redistribution or
training permission for the corpus or models.
