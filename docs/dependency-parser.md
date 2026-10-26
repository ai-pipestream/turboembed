# Dependency parsing

`turbo-depparse` implements OpenNLP 3.x arc-standard parsing, training,
UAS/LAS evaluation and document annotation (inventory 68, 69, 72).
Eight permissive UD 2.16 dependency parser models (four languages x MaxEnt and
Feedforward) are listed in the built-in model zoo under release
`models-ud-depparse-2.16-1`.

## Graph and alignment

`DependencyGraph<A>` uses `Ix<A>` for nodes, heads, children and path steps.
`Tokens` is the default anchor; `Words` supports UD syntactic words.
`WordOrTokenAnchor::Pos` selects token or word POS at compile time.

`new` requires a nonempty single-root tree; `from_layer` accepts forests.
`Dependencies<A>::check()` validates column lengths, relation tags and head
indices, including self-loop rejection, without allocating. Multi-node cycles
are allowed at that boundary, including direct `Document::add` insertion.
`check_forest()` adds O(n) cycle rejection without recursion and validates
structural inputs itself. Graph import, annotation and wire decoding call it
after the structural check; callers needing a forest can opt in directly.
Navigation includes `roots`, `root`, `arc`, `children`, `shortest_path` and
`lowest_common_ancestor`. Path steps retain labels and `ToHead`/`ToDependent`
directions. Separate trees have no connecting path. Invalid indexes return
errors. Child lookup uses an O(n) index; ancestor/path queries take O(height
+ path length). Projectivity uses an O(n²) crossing check including root arcs.

`to_layer` preserves tag IDs, tag-set name/order and unused labels. Layers
store heads and relations, so text alignment travels in `DependencySample<A>`:
`from_conllu` -> `graph().to_layer()` -> `with_layer` -> `to_conllu`.
This preserves MWT ranges, view ID, word byte spans and source fields.
`source_text` retains original comments, columns and empty nodes as provenance,
even after replacing a tree. It does not serialize edited dependencies.

`ConlluDependencySample::read_sentence` accepts one basic-dependency block.
Initial spans address source FORM fields. For reconstructed text, supply the
view ID and word spans explicitly; expanded words may share surface spans.
Conversion without alignment metadata fails.

## Model and pipeline

The public transition/state/oracle/context APIs reproduce 37 ordered Java
features. Parsing uses 2n transitions. Training skips non-projective samples
before emitting events and reports the skip count. The greedy decoder picks
the first highest-scoring legal outcome and rejects malformed inventories
or non-finite scores.

`DependencyModel::open` loads component `DependencyParserME` with artifact
`depparse.model`; `from_raw` loads classifier bytes. `train` uses the shared
trainer and returns a reloadable archive accessible through `bytes`.
Loaded model provenance includes SHA-256. `from_maxent` requires caller identity.

The native `turbo-depparse` executable supports `inspect`, `convert`, `parse`
and explicit-parameter `train` for both linear and feedforward models.
It needs no Python installation and does not overwrite source or existing
output files. See the [CLI commands and training limits](feedforward-dependency-parser.md#native-command-line).
Existing raw and packaged model APIs remain directly loadable; conversion is
optional. CLI training requires explicit projective-only acceptance and
reports input, retained and excluded sentence and word counts. Its
`train_projective` Rust APIs reject crossing trees and retain literal relation
labels with `Plain` encoding. Default `train` APIs retain non-projective trees
through pseudo-projectivization and use `EscapedV1` labels.

Evaluation reports token-weighted UAS/LAS, with optional punctuation exclusion
(default POS `PUNCT`). LAS requires a matching head and label. Cross-validation
uses round-robin held-out partitions and a training callback; partition count
must be in 2..=sample_count. Failed evaluation preserves accumulated totals.

`DependencyAnnotator<A>` consumes view, sentences, surface tokens, `A` and
`A::Pos`. Token mode binds surface and anchor slots to the same instance.
Word mode resolves expanded forms through surface tokens. UTF-8, view, POS
and segmentation alignment are checked before parsing. Output heads use
document indexes. Empty input produces an empty forest; uncovered words fail.
Provenance records version `1` and model identity.

See `analysis/depparse/src/lib.rs`, `analysis/depparse/tests/data/README.md`
and `docs/receipts/depparse-688/README.md` for API, parity data and receipts.

## Feedforward model

The neural parser, trainer, both feature layouts, model formats and annotation
integration are documented in [feedforward dependency parsing](feedforward-dependency-parser.md).

## Pseudo-projective handling and UD 2.16 models

Non-projective trees are handled via Nivre and Nilsson (2005) pseudo-projective
lifting (`DependencyGraph::projectivize`) and step-by-step decoding
(`DependencyGraph::deprojectivize`). During recipe training (`turbo-train-recipes -- depparse`),
every non-projective sentence in `train` is projectivized with directional
`base^<step1^>step2...` path labels instead of being discarded, and both
`DependencyParserME` and `FeedforwardDependencyParser` decode directional
`^<` / `^>` suffixes back into non-projective dependencies according to the
model's explicit `LabelEncoding` mode (while `LabelEncoding::Plain` and
`LabelEncoding::EscapedV1` preserve ordinary custom labels such as `root^literal`
unchanged). Note that Nivre and Nilsson's directional label-path scheme is an
approximate encoding when multiple siblings under the same lifted ancestor share
the same relation label and direction (for example, gold encode/decode over all
1,867 `UD_Finnish-FTB` test sentences alters 8 heads across 8 sentences out of
16,286 words due to same-label governor ambiguity).

Each recipe gates its full-budget pseudo-projective candidate (`100` GIS
iterations for MaxEnt, `5` AdaGrad epochs for Feedforward across all training
sentences) against a single-step projective-only internal baseline
(`<model_id>-projective-baseline`: `1` GIS iteration for MaxEnt, `1` AdaGrad
epoch for Feedforward trained on projective sentences only). Because Apache
OpenNLP publishes no pre-trained UD dependency parser models in the zoo, the
paired bootstrap gate (`gate.json` on LAS, `gate-uas.json` on UAS) verifies
end-to-end learning and non-projective retention over that single-step internal
baseline rather than comparing against an equal-budget external parser (such as
MaltParser or Stanza).

Ten dependency parser models are trained across the five CC BY 4.0 UD 2.16
treebanks (`UD_Spanish-AnCora`, `UD_Catalan-AnCora`, `UD_Portuguese-Porttinari`,
`UD_Finnish-FTB`, and `UD_Italian-MarkIT`), with eight shippable models published
in release `models-ud-depparse-2.16-1` and registered in `analysis/zoo/index/zoo.json`
(Portuguese Porttinari remains recipe-only because newspaper text rights are not settled):

| Treebank (`r2.16`) | Train Sentences (Non-Proj) | Test Sentences (Tokens) | Model ID | Parser | Size (bytes) | SHA-256 | UAS (all / no-punct) | LAS (all / no-punct) | LAS Gate (`diff`, 95% CI) | UAS Gate (`diff`, 95% CI) |
|---|---:|---:|---|---|---:|---|---:|---:|---|---|
| `UD_Spanish-AnCora` | 14,287 (840) | 1,721 (53,622) | `turbo-es-ud-ancora-2.16-depparse-maxent-1` | MaxEnt | 17,406,430 | `6d399e33ee858d0ab87ebc6fa9438a445f614b6729f8a609ab5ffeabfa538f63` | 83.45 / 85.76 | 78.81 / 80.51 | PASS (+52.91, [+52.33, +53.50]) | PASS (+43.31, [+42.69, +43.93]) |
| `UD_Spanish-AnCora` | 14,287 (840) | 1,721 (53,622) | `turbo-es-ud-ancora-2.16-depparse-feedforward-1` | Feedforward | 900,875 | `68aea2dfec51c015ce8ac99ff4c15875d84a364ef9bfe9fe3b5a4976c0b0f8a7` | 83.99 / 85.72 | 79.35 / 80.45 | PASS (+3.74, [+3.38, +4.10]) | PASS (+3.02, [+2.68, +3.38]) |
| `UD_Catalan-AnCora` | 13,123 (749) | 1,846 (59,610) | `turbo-ca-ud-ancora-2.16-depparse-maxent-1` | MaxEnt | 15,339,008 | `33e2ec6844fabfa173bf0e0eafca62eaf259f95622e1c149a2f15197782c27f6` | 85.58 / 87.29 | 81.62 / 82.86 | PASS (+53.61, [+53.05, +54.17]) | PASS (+40.98, [+40.38, +41.59]) |
| `UD_Catalan-AnCora` | 13,123 (749) | 1,846 (59,610) | `turbo-ca-ud-ancora-2.16-depparse-feedforward-1` | Feedforward | 816,124 | `8af62c85bd4f3ab42498ee8b217911ff55101df2caab554391120ab2dbb45455` | 85.69 / 86.97 | 81.59 / 82.38 | PASS (+3.45, [+3.11, +3.79]) | PASS (+2.75, [+2.44, +3.07]) |
| `UD_Portuguese-Porttinari` | 5,893 (0) | 1,683 (33,580) | `turbo-pt-ud-porttinari-2.16-depparse-maxent-1` | MaxEnt (local) | 5,405,678 | `422d8ff7c4d0a3513ce1518a5e65b8fe49832acffcc665cf9b419e7d4f3557a9` | 85.91 / 87.75 | 82.74 / 84.12 | PASS (+50.98, [+50.22, +51.73]) | PASS (+40.04, [+39.27, +40.80]) |
| `UD_Portuguese-Porttinari` | 5,893 (0) | 1,683 (33,580) | `turbo-pt-ud-porttinari-2.16-depparse-feedforward-1` | Feedforward (local) | 423,296 | `a18b65db4ba0332ab5c31686cd21f131ba014dea776143f8d2a104c58d023147` | 87.63 / 88.58 | 84.71 / 85.22 | PASS (+6.18, [+5.71, +6.66]) | PASS (+4.96, [+4.49, +5.43]) |
| `UD_Finnish-FTB` | 14,981 (1,602) | 1,867 (16,286) | `turbo-fi-ud-ftb-2.16-depparse-maxent-1` | MaxEnt | 5,757,374 | `755ef066e10c65adc3d17f93bb15b35ebee78fbd189fd56297f12409bf5f782d` | 81.70 / 79.18 | 74.10 / 70.33 | PASS (+53.26, [+52.31, +54.24]) | PASS (+53.75, [+52.75, +54.77]) |
| `UD_Finnish-FTB` | 14,981 (1,602) | 1,867 (16,286) | `turbo-fi-ud-ftb-2.16-depparse-feedforward-1` | Feedforward | 564,886 | `3df4a7db92fb89759aa964153cc7a55154779cf25a56250cc8b2b2924ce9870d` | 79.59 / 76.73 | 69.39 / 64.84 | PASS (+6.96, [+6.30, +7.63]) | PASS (+5.45, [+4.82, +6.07]) |
| `UD_Italian-MarkIT` | 611 (0) | 340 (10,196) | `turbo-it-ud-markit-2.16-depparse-maxent-1` | MaxEnt | 1,364,091 | `2f8539971c1d4434bde2b5e16aff8348c8ef7841d2a5576b180ed23e8f70da93` | 78.56 / 82.09 | 71.58 / 74.43 | PASS (+47.32, [+45.79, +48.90]) | PASS (+41.86, [+40.16, +43.64]) |
| `UD_Italian-MarkIT` | 611 (0) | 340 (10,196) | `turbo-it-ud-markit-2.16-depparse-feedforward-1` | Feedforward | 258,169 | `397fc29dd5ee2ac576400f38742a38023d4b23abe09df351dd18c449094f2b3e` | 77.29 / 80.78 | 69.16 / 71.84 | PASS (+31.61, [+30.15, +33.10]) | PASS (+23.63, [+22.16, +25.18]) |


### Dependency label encoding compatibility

Relation text does not select a decoder. Models loaded from old raw, Java or
unmarked native containers use `LabelEncoding::Plain`, preserving custom
carets, angle brackets and backslashes. New training escapes every relation
consistently, including projective sentences. MaxEnt stores
`Turbo-Dependency-Label-Encoding` in its manifest; native neural ZIP version 3
stores `label_encoding` in its configuration. Unknown modes are refused, and
legacy Java export refuses encoded models because that format has no mode field.
Native versions 1 and 2 remain readable; conversion is optional.

The eight original `models-ud-depparse-2.16-1` artifacts predate these fields.
Their complete SHA-256 values, from their published cards, are listed in
`analysis/depparse/src/encoding.rs`. Only those exact files automatically use
`LegacyPseudo` decoding. A repackaged or modified unmarked file defaults to
plain labels; it is not identified by its transitions, filename or language.
The published bytes and their evaluated predictions are unchanged. New model
serialization records the selected mode explicitly, including when converting
one of these historical neural models to native version 3.

`projectivize` marks its returned graph as encoded. `deprojectivize` decodes
that graph and otherwise preserves plain graphs. Transition reconstruction or
layer import discards graph metadata; callers must then use
`deprojectivize_with` with the encoding from the model. Re-projectivizing an
already encoded graph is refused to prevent double escaping.
