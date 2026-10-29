# Dependency parsing

`turbo-depparse` provides the typed dependency graph (`DependencyGraph`),
CoNLL-U sample alignment (`DependencySample`), and UAS/LAS evaluators
(`DependencyEvaluator`, `DependencyCrossValidator`, `HeldOutEvaluation`).
Classical arc-standard MaxEnt (`DependencyParserME`) and Chen & Manning
feedforward transition parsers were retired under `#1529`; dependency parsing
models and document annotation are covered by the biaffine parser head on the
shared encoder (`turbo-core` / `turbo-tagger`, `#1514`).

## Graph and alignment

`DependencyGraph<A>` uses `Ix<A>` for nodes, heads, children and path steps.
`Tokens` is the default anchor; `Words` supports UD syntactic words.
`WordOrTokenAnchor::Pos` selects token or word POS at compile time.

`new` requires a nonempty single-root tree; `from_layer` accepts forests.
`Dependencies<A>::check()` validates column lengths, relation tags and head
indices, including self-loop rejection, without allocating. Multi-node cycles
are allowed at that boundary, including direct `Document::add` insertion.
`check_forest()` adds O(n) cycle rejection without recursion and validates
structural inputs itself. Graph import calls it after the structural check;
callers needing a forest can opt in directly.
Navigation includes `roots`, `root`, `arc`, `children`, `shortest_path` and
`lowest_common_ancestor`. Path steps retain labels and `ToHead`/`ToDependent`
directions. Separate trees have no connecting path. Invalid indexes return
errors. Child lookup uses an O(n) index; ancestor/path queries take O(height
+ path length). Projectivity uses an O(n^2) crossing check including root arcs.

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

## Evaluation and cross-validation

`DependencyEvaluator` reports token-weighted UAS and LAS, with optional
punctuation exclusion (default POS `PUNCT`). LAS requires a matching head and
relation label. `DependencyCrossValidator` uses round-robin held-out partitions
and a training callback over the `DependencyParser<A>` trait; partition count
must be in `2..=sample_count`. Failed evaluation preserves accumulated totals.

`HeldOutEvaluation` scores projective, non-projective, and combined sentence
populations while retaining missing or invalid predictions in every relevant
word denominator.

## Biaffine parser and annotator (`#1514`)

Dependency parsing models and pipeline annotation live in `turbo-core`
(`BiaffineParser` with Chu-Liu-Edmonds MST and Eisner projective tree decoding)
and `turbo-tagger` (`DependencyParser`, writing `Dependencies<Tokens>` or
`Dependencies<Words>` on the shared encoder). See
[neural-annotators.md](neural-annotators.md) for bundle layout and pipeline
usage.
