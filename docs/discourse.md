# Discourse units and claim links

Two annotators build a document's argument graph (#1205): one splits
sentences into elementary discourse units (EDUs) and links them with
rhetorical relations; the other links chunks (or sentences) whose
claims entail, contradict or repeat each other, scored by any NLI model.

## Layers

In `turbo_analysis_layers::rhetoric`:

| Layer | ID | Rows |
|---|---|---|
| `Edus` | `turbo:edus` | disjoint, ordered unit spans and the sentence of each |
| `Discourse` | `Edges<Edus, Edus>` | one edge per relation, kind `DiscourseEdgeLabel` |
| `ClaimLinks` | `Edges<Chunks, Chunks>` | kind `ClaimRelationKind` |
| `SentenceClaimLinks` | `Edges<Sentences, Sentences>` | the same over sentences |

`DiscourseRel` has eight values (elaboration, contrast, cause-result,
condition, temporal-sequence, concession, evidence, joint) and
`Nuclearity` three (`NucleusToSatellite`, `SatelliteToNucleus`,
`MultiNuclear`). An edge kind is a tag of a fixed 24-label tag set named
`relation/nuclearity` (`contrast/nn`, `concession/sn`, ...);
`DiscourseEdgeLabel::{tag, from_tag}` convert. `ClaimRelationKind` is
`Entails`, `Contradicts` or `Redundant` in the tag set `turbo:claims`.
Every edge runs from the earlier unit to the later one, except an
`Entails` edge, which runs from premise to hypothesis. All four layers
round-trip through the layers codecs.

The rhetoric module is named apart from `discourse.rs`, which holds the
quotation layers.

## DiscourseAnnotator

`turbo_chunker::DiscourseAnnotator` reads `Sentences`, `Tokens` and,
when present, `Dependencies`, and writes `(Edus, Discourse)`.
Connectives come from a fixed lexicon (`because`, `although`, `however`,
`for example`, `and`, ...) compiled into a `turbo-matcher` phrase
automaton; each is a subordinator, an adverbial or a coordinator.

With dependencies, a unit is the run of tokens owned by one clause head:
the root, `advcl`, `ccomp`, `parataxis`, and a `conj` that has its own
subject, object, auxiliary or marker (so `Sales and profits rose` stays
one unit). Punctuation joins the unit before it. A clause links to the
clause that governs it; a sentence that opens with an adverbial or
coordinating connective links to the previous sentence's main clause.
Unmarked links are elaborations at score 0.5; marked ones take the
connective's relation at 0.9.

Without dependencies, the fallback cuts before a subordinator, before an
adverbial or coordinator that follows a comma or semicolon, after a
semicolon, and after the first comma of a sentence that opens with a
subordinator. It scores marked links 0.7.

Contrast and joint coordination are multinuclear; otherwise the unit
holding the connective is the satellite.

### Eval

Twelve hand-parsed documents (`analysis/chunker/tests/discourse_eval.rs`),
ten gold boundaries and ten gold edges; an edge counts only when both
units, the relation and the nuclearity match.

| Mode | Boundaries (P / R) | Edges (P / R) |
|---|---|---|
| Dependencies | 10/10 / 10/10 | 8/10 / 8/10 |
| Connective fallback | 8/9 / 8/10 | 6/9 / 6/10 |

The dependency mode gets the nuclearity wrong for a `while` contrast
(RST treats it as multinuclear) and for attribution (`Acme said | profits
rose`), which has no relation of its own here and comes out as an
elaboration with the reported clause as satellite. The fallback also
misses a comma-less verb conjunct and a complement clause, and cuts
before a prepositional `after`. The set is small and was written
alongside the annotator, so treat it as a regression check, not an
accuracy estimate.

## ClaimVerificationAnnotator

`turbo_doccat::ClaimVerificationAnnotator<S: NliScorer, L = Chunks>`
reads `(View, L, Option<Entities>, Option<Vectors<L>>)` and writes
`Edges<L, L>`: `ClaimLinks` over chunks, `SentenceClaimLinks` with
`L = Sentences`. `BundleNliScorer` (turbo-embed) or any other
`NliScorer` scores the pairs.

A pair of units, neither containing the other, is a candidate when they
share an entity (the same lowercase surface of an `Entities` mention),
when their vector cosine lies in `[min_similarity, max_similarity]`
(default `[0.5, 1.0]`; lower the maximum to skip near duplicates), or
when their content words overlap by at least `min_word_overlap` (Jaccard,
default 0.25, `None` to turn off). Candidates are ranked by shared
entities, similarity, then overlap, and the first `max_pairs` (64) are
scored in both directions in a single `score_pairs` call. Pairs over the
scorer's token budget are skipped.

Both directions entailing at `entail_threshold` (0.8) is `Redundant`;
one direction is `Entails`; either direction's contradiction at
`contradict_threshold` (0.8) is `Contradicts`. A two-way head
(`entailment`, `not_entailment`) never yields a contradiction; a head
with no `entailment` outcome is refused.

The entity signal reads the sentence-level `Entities` layer only. A
document-level entity layer (#1200) can be added as a further signal
once it lands; nothing here depends on it.

### Pruning eval

`analysis/doccat/tests/claim_verify_eval.rs` runs a twelve-sentence
document through a rule-based scorer (word-set entailment and
negation-flip contradiction). That scorer is not a model, so the numbers
measure only what pruning costs against scoring every pair with the
same scorer (5 edges from 132 directed pairs):

| Candidates | Edges kept | Directed pairs scored |
|---|---|---|
| Word overlap (default, no entities) | 3/5 | 6/132 |
| Entities only | 5/5 | 14/132 |
| Overlap and entities | 5/5 | 14/132 |

Overlap alone misses the two entailments where a long premise shares
few words with a short hypothesis; the entity signal recovers them. No
trained NLI checkpoint was run for this change: the bundle test runs the
verifier on the untrained tiny model and checks only that its edge
scores equal the softmaxed logits of each direction.
