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
optional. Current training requires explicit projective-only acceptance and
reports excluded sentence and word counts.

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
every non-projective sentence in `train` is projectivized with `+` composite
governor labels (up to `MAX_PROJECTIVE_LIFTS` steps) instead of being discarded,
and both `DependencyParser` and `FeedforwardDependencyParser` decode composite
labels back into non-projective dependencies after transition decoding.

Ten dependency parser models are trained across the five CC BY 4.0 UD 2.16
treebanks (`UD_Spanish-AnCora`, `UD_Catalan-AnCora`, `UD_Portuguese-Porttinari`,
`UD_Finnish-FTB`, and `UD_Italian-MarkIT`), with eight shippable models published
in release `models-ud-depparse-2.16-1` and registered in `analysis/zoo/index/zoo.json`
(Portuguese Porttinari remains recipe-only because newspaper text rights are not settled):

| Treebank (`r2.16`) | Train Sentences (Non-Proj) | Test Sentences (Tokens) | Model ID | Parser | Size (bytes) | SHA-256 (prefix) | UAS (all / no-punct) | LAS (all / no-punct) | LAS Gate (`diff`, 95% CI) | UAS Gate (`diff`, 95% CI) |
|---|---:|---:|---|---|---:|---|---:|---:|---|---|
| `UD_Spanish-AnCora` | 14,287 (840) | 1,721 (53,622) | `turbo-es-ud-ancora-2.16-depparse-maxent-1` | MaxEnt | 17,406,430 | `6d399e33ee858d0a` | 83.45 / 85.76 | 78.81 / 80.51 | PASS (+52.91, [+52.33, +53.50]) | PASS (+43.31, [+42.69, +43.93]) |
| `UD_Spanish-AnCora` | 14,287 (840) | 1,721 (53,622) | `turbo-es-ud-ancora-2.16-depparse-feedforward-1` | Feedforward | 900,875 | `68aea2dfec51c015` | 83.99 / 85.72 | 79.35 / 80.45 | PASS (+3.74, [+3.38, +4.10]) | PASS (+3.02, [+2.68, +3.38]) |
| `UD_Catalan-AnCora` | 13,123 (749) | 1,846 (59,610) | `turbo-ca-ud-ancora-2.16-depparse-maxent-1` | MaxEnt | 15,339,008 | `33e2ec6844fabfa1` | 85.58 / 87.29 | 81.62 / 82.86 | PASS (+53.61, [+53.05, +54.17]) | PASS (+40.98, [+40.38, +41.59]) |
| `UD_Catalan-AnCora` | 13,123 (749) | 1,846 (59,610) | `turbo-ca-ud-ancora-2.16-depparse-feedforward-1` | Feedforward | 816,124 | `8af62c85bd4f3ab4` | 85.69 / 86.97 | 81.59 / 82.38 | PASS (+3.45, [+3.11, +3.79]) | PASS (+2.75, [+2.44, +3.07]) |
| `UD_Portuguese-Porttinari` | 5,893 (0) | 1,683 (33,580) | `turbo-pt-ud-porttinari-2.16-depparse-maxent-1` | MaxEnt (local) | 5,405,678 | `422d8ff7c4d0a351` | 85.91 / 87.75 | 82.74 / 84.12 | PASS (+50.98, [+50.22, +51.73]) | PASS (+40.04, [+39.27, +40.80]) |
| `UD_Portuguese-Porttinari` | 5,893 (0) | 1,683 (33,580) | `turbo-pt-ud-porttinari-2.16-depparse-feedforward-1` | Feedforward (local) | 423,296 | `a18b65db4ba0332a` | 87.63 / 88.58 | 84.71 / 85.22 | PASS (+6.18, [+5.71, +6.66]) | PASS (+4.96, [+4.49, +5.43]) |
| `UD_Finnish-FTB` | 14,981 (1,602) | 1,867 (16,286) | `turbo-fi-ud-ftb-2.16-depparse-maxent-1` | MaxEnt | 5,757,374 | `755ef066e10c65ad` | 81.70 / 79.18 | 74.10 / 70.33 | PASS (+53.26, [+52.31, +54.24]) | PASS (+53.75, [+52.75, +54.77]) |
| `UD_Finnish-FTB` | 14,981 (1,602) | 1,867 (16,286) | `turbo-fi-ud-ftb-2.16-depparse-feedforward-1` | Feedforward | 564,886 | `3df4a7db92fb8975` | 79.59 / 76.73 | 69.39 / 64.84 | PASS (+6.96, [+6.30, +7.63]) | PASS (+5.45, [+4.82, +6.07]) |
| `UD_Italian-MarkIT` | 611 (0) | 340 (10,196) | `turbo-it-ud-markit-2.16-depparse-maxent-1` | MaxEnt | 1,364,091 | `2f8539971c1d4434` | 78.56 / 82.09 | 71.58 / 74.43 | PASS (+47.32, [+45.79, +48.90]) | PASS (+41.86, [+40.16, +43.64]) |
| `UD_Italian-MarkIT` | 611 (0) | 340 (10,196) | `turbo-it-ud-markit-2.16-depparse-feedforward-1` | Feedforward | 258,169 | `397fc29dd5ee2ac5` | 77.29 / 80.78 | 69.16 / 71.84 | PASS (+31.61, [+30.15, +33.10]) | PASS (+23.63, [+22.16, +25.18]) |

