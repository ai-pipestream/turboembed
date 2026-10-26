# Dependency parsing

`turbo-depparse` implements OpenNLP 3.x arc-standard parsing, training,
UAS/LAS evaluation and document annotation (inventory 68, 69, 72).
No pretrained language model is bundled.

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
features on projective models. Projective parsing uses 2n transitions. The
native SWAP extension supports crossing trees without changing source indexes;
its worst-case action bound is n(n+1). Training includes all valid trees, and
projective oracle sequences remain unchanged. The greedy decoder picks
the first highest-scoring legal outcome and rejects malformed inventories
or non-finite scores.

`DependencyModel::open` loads component `DependencyParserME` with artifact
`depparse.model`; `from_raw` loads classifier bytes. `train` uses the shared
trainer and returns a reloadable archive accessible through `bytes`.
Loaded model provenance includes SHA-256. `from_maxent` requires caller identity.

The native `turbo-depparse` executable supports `inspect`, `convert`, `parse`
`evaluate`, and explicit-parameter `train` for both linear and feedforward models.
It needs no Python installation and does not overwrite source or existing
output files. See the [CLI commands and training limits](feedforward-dependency-parser.md#native-command-line).
Existing raw and packaged model APIs remain directly loadable; conversion is
optional. Training retains non-projective trees by default. An optional
`--projective-only` flag reports all excluded sentence and word counts.
SWAP models use native transition semantics; Java neural export refuses them.
See the [pinned Finnish-FTB recipe](../analysis/train-recipes/dependency/fi-ftb-r2.14/README.md).

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

`HeldOutEvaluation` and the CLI retain failed/invalid predictions as zero-credit
words, split projective and non-projective totals, and refuse empty populations
or any parse failure. The caller supplies finite UAS/LAS floors before evaluation;
the report is written even if that gate fails. This is distinct from the legacy
`DependencyEvaluator`, which returns errors without adding the failed sample.
