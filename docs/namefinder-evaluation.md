# Historical NameFinderME evaluation (#177)

The archive-based evaluation example and its JVM comparator test were removed
under #1511. The model loader rejects OpenNLP archives before decoding. These
results describe an earlier implementation and cannot validate a replacement
entity head. Do not run the historical JVM/model procedure as an acceptance
test for current builds.

The retained numerical entity decoder is described in
`analysis/sequence/README.md`; it accepts typed states and native score arrays.
Native document-level entity integration is tracked separately in #1540 and
replacement model heads in #1524. The historical scores below are preserved
with their input identities and data-use limits.

## CoNLL-2003 English test results

Run on 2026-10-22 with the locally supplied English test split (`eng.testb`,
stored as `test.txt`). Each of the three model runs read the same **3,453
sentences and 46,435 tokens**, with **231 adaptive-data reset flags** from
`Conll03NameSampleStream`. Document markers are not scored as tokens.

The native harness was built in release mode from TurboEmbed commit
`ad2afc6acf43310c60bbcd9f2944f35205bb79a0`, with `--locked`. The JVM producer
ran on OpenNLP revision `cdf33c653d5d61d5af7d37ffee189a83bd8392ea`, verified
from the container's commit file, in
`turbo-goldens-java:opennlp-cdf33c653d5d-addons-33e95a4956f1` with network
access disabled. The local image ID was
`sha256:69a8e4967d23de7a2580658246e48cb4b078eae594b2a2cdd9e01003154aab64`.
Both runtimes used the same unmodified model and corpus bytes.

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
| `per` | OpenNLP JVM | 1617 | 719 | 618 | 0.8595271210013908 | 0.3821892393320965 | 0.5291095890410958 |
| `org` | TurboEmbed | 1661 | 494 | 251 | 0.5080971659919028 | 0.15111378687537627 | 0.23294663573085847 |
| `org` | OpenNLP JVM | 1661 | 494 | 251 | 0.5080971659919028 | 0.15111378687537627 | 0.23294663573085847 |
| `loc` | TurboEmbed | 1668 | 1167 | 911 | 0.7806341045415596 | 0.5461630695443646 | 0.6426807760141093 |
| `loc` | OpenNLP JVM | 1668 | 1167 | 911 | 0.7806341045415596 | 0.5461630695443646 | 0.6426807760141093 |

For all three runs, the comparator passed on every sample: model and corpus
hashes, gold types, sentence order, token counts, reset flags, gold spans,
and predicted spans/types matched. Maximum absolute entity-probability
error was **0** for `per`, **0** for `org`, and **0** for `loc`, below the
required `1e-9`. Agreement is checked before the native result is written;
it is not inferred from matching aggregate F1.

These are accuracy measurements for the three legacy single-type models,
not a combined four-type CoNLL score. No `misc` model, cross-finder merge,
label remapping, training, tuning or threshold selection was performed.
The low recall, especially for organizations, occurs in both runtimes.
Parity establishes faithful execution of these models, not their suitability
as a new accuracy baseline.

### Historical data handling

Reproduction requires the historical implementation and the exact inputs
identified above. No runnable archive-loading recipe is supplied by the
current build. These results have not been rerun with a replacement head.

Local evaluation was approved under the CoNLL-2003 entry in
[licences.md](licences.md). The corpus and the legacy model binaries remain
external inputs. Only hashes, counts and aggregate metrics are included in
this report; no Reuters text, gold/predicted span records, or workstation
paths are committed. The raw comparison records are local scratch output,
not publication artifacts. This evaluation grants no redistribution or
training permission for the corpus or models.
