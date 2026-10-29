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
Navigation includes `roots`, `root`, `arc`, `children`, `incoming_arcs`, `subtree`,
`subtree_span`, `resolve_arcs`, `shortest_path` and `lowest_common_ancestor`.
`resolve_arcs` and `incoming_arcs` resolve each node and arc against the anchor
layer `A`, optional `A::Pos` tags, and document `View` text into
`ResolvedDependencyNode<'a, A>` and `ResolvedDependencyArc<'a, A>` with UTF-8
byte `Span`, `Utf16Span`, `CodePointSpan`, `Word<'a>`, and POS tag metadata.
Path steps retain labels and `ToHead`/`ToDependent`
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

`DependencyModel::from_bytes` and `DependencyModel::to_bytes` serialize the
linear model in the deterministic native `TURBO-DEPPARSE-LIN-V1` binary wire
format (`open` and `save` read and write files directly). `train`, `train_projective`,
and `train_swap` use the shared linear trainer and compute a canonical SHA-256
digest for provenance tracking.

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

See `analysis/depparse/src/lib.rs` and `analysis/depparse/tests/data/README.md`
for API and CoNLL-U fixture details.

## Feedforward model

The neural parser, trainer, both feature layouts, native model format and annotation
integration are documented in [feedforward dependency parsing](feedforward-dependency-parser.md).

## Pseudo-projective handling and SWAP training

Non-projective trees are handled via Nivre and Nilsson (2005) pseudo-projective
lifting (`DependencyGraph::projectivize`) and step-by-step decoding
(`DependencyGraph::deprojectivize`), or via the Nivre (2009) non-projective
`SWAP` transition oracle (`SwapOracle`).

Relation text does not select a decoder. Models record an explicit `LabelEncoding`
(`LabelEncoding::Plain` for projective-only and SWAP models, `LabelEncoding::EscapedV1`
for pseudo-projective models). `projectivize` marks its returned graph as encoded.
`deprojectivize` decodes that graph and otherwise preserves plain graphs. Transition
reconstruction or layer import discards graph metadata; callers must then use
`deprojectivize_with` with the encoding from the model. Re-projectivizing an
already encoded graph is refused to prevent double escaping.

`DependencyModel::train_swap`, `FeedforwardDependencyTrainer::train_swap`, and
`DependencyEventStream::generate_swap` retain crossing trees with native SWAP
transitions and ordinary `Plain` labels. `train` remains pseudo-projective;
`train_projective` remains the strict projective-only API. SWAP moves parser
buffer indexes, not source tokens or spans. The monotonic pair rule and an
`n(n+1)` action bound ensure termination.
