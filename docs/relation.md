# Dependency-path relation extraction (`turbo-relation`)

Declarative relation extraction over dependency graphs: a
`RelationPattern` is a path like `<nsubj >obj` (up steps from the
subject's head word, then down steps to the object's), an optional
pivot trigger, and optional restrictions on the entity kinds and
the pivot's part of speech or lemma. `RelationAnnotator` matches
every pattern against every ordered entity pair whose head words
(the first overlapping word whose dependency head leaves the
entity) lie in one sentence, following the shortest dependency
path through their lowest common ancestor.

It writes two layers: `Relations`, one row
per mention in extraction order, covering both entity spans and
keeping duplicate pattern registrations as duplicate mentions;
and `Edges<Entities, Entities>` with each (subject, object, type)
triple once at its best confidence. Both cross the document
graph boundary (`turbo_relation::export::register_codecs`).

Negation scope, epistemic factuality and deontic modality over the
same dependency trees (`ModalityAnnotator`) are in
[modality.md](modality.md).

Patterns validate as the reference records they port do: the
path splits on the project whitespace predicate, up steps precede
down steps, and a trigger must already be in the project's simple
per-character lowercase form.

## Discovery, defaults, and mining

`RelationPathAnnotator` writes `turbo:relation-paths`: for every
ordered entity pair whose heads connect within one sentence, the
subject and object, both head words, the pivot (lowest common
ancestor), and the walked steps. A row's signature (the steps in
pattern syntax) parses unchanged as a `RelationPattern` path, so
discovery output reads exactly like the rules it suggests.
`RelationPathExplorer` folds those rows over a corpus into
ranked `(subject kind, signature, pivot POS, pivot lemma, object
kind)` rows with up to three example form walks each, and any row
exports to a pattern. `DistantSupervisionMiner` instead starts
from seed entity-pair facts and ranks the signatures that
produced them by support and precision.

The default libraries (`turbo_relation::defaults`) are seven UD
shape rules (`subject-object`, `passive-agent` over both `>obl`
and `>obl:agent`, `apposition`, `possession`,
`prepositional-attachment`, `attribute`) with no triggers, and
eight semantic starting points (`works_for`,
`located_in`, `founded_by`, `headquartered_in`, `member_of`,
`acquired_by`, `born_in`, `spouse_of`) combining a shape, a pivot
lemma, and entity kinds on both ends. Both are starting points
to narrow against a corpus with the explorer, not trained
extractors.

## Measured defaults

The default libraries are measured on an in-tree fixture,
`analysis/relation/tests/fixtures/defaults/eval.conllu`: 26
sentences, 187 words, 61 entities and 39 gold (head span, tail span,
label) triples, written for this repository and licensed with it
(Apache-2.0), so nothing depends on a corpus that cannot be
redistributed. Every sentence carries a gold UD v2 parse (form,
lemma, UPOS, head, label), gold entity spans and kinds (`person`,
`organization`, `location`, `date` and a few others), and the gold
triples. Each pair carries the most specific relation the sentence
states, so `headquartered in` is `headquartered_in`, not also
`located_in`. The shape labels are read semantically: `passive-agent`
is a passive subject and its agent, not any oblique; `attribute` is a
property, class or role of the subject, not a location or origin
predicated with a copula. The fixture holds sentences each pattern
should match, paraphrases it should also cover, and near misses it
should not fire on.

`analysis/relation/tests/defaults_eval.rs` runs `RelationAnnotator`
with each library over the gold layers, counts a predicted triple once
however many patterns produced it, scores each library against the
gold triples of its own labels, and asserts the numbers below
(true positives / predicted / gold; `-` is undefined, no prediction).

Shape rules (`english_defaults`):

| label | TP/pred/gold | precision | recall | F1 |
|---|---|---|---|---|
| apposition | 2/2/2 | 1.000 | 1.000 | 1.000 |
| attribute | 2/4/2 | 0.500 | 1.000 | 0.667 |
| passive-agent | 3/8/3 | 0.375 | 1.000 | 0.545 |
| possession | 2/2/2 | 1.000 | 1.000 | 1.000 |
| prepositional-attachment | 3/3/3 | 1.000 | 1.000 | 1.000 |
| subject-object | 6/6/6 | 1.000 | 1.000 | 1.000 |
| overall (micro) | 18/25/18 | 0.720 | 1.000 | 0.837 |

Semantic starting set (`english_semantic_defaults`):

| label | TP/pred/gold | precision | recall | F1 |
|---|---|---|---|---|
| acquired_by | 2/2/3 | 1.000 | 0.667 | 0.800 |
| born_in | 1/1/2 | 1.000 | 0.500 | 0.667 |
| founded_by | 1/1/2 | 1.000 | 0.500 | 0.667 |
| headquartered_in | 1/1/1 | 1.000 | 1.000 | 1.000 |
| located_in | 0/0/4 | - | 0.000 | 0.000 |
| member_of | 0/1/2 | 0.000 | 0.000 | 0.000 |
| spouse_of | 0/0/2 | - | 0.000 | 0.000 |
| works_for | 1/2/5 | 0.500 | 0.200 | 0.286 |
| overall (micro) | 6/8/21 | 0.750 | 0.286 | 0.414 |

What misfires and what is missed, on gold parses:

- `passive-agent` over `>obl` fires on every passive oblique (`born
  in Paris`, `acquired in 2019`, `headquartered in Boston`); only the
  `>obl:agent` registration is precise on a parser that emits
  `obl:agent`.
- `attribute` (`<nsubj`) fires on copular location and origin
  predicates (`Acme is in Boston`, `Smith is from Boston`).
- The semantic patterns are active-voice `<nsubj >obl` shapes, so they
  miss the common phrasings: `located_in` and `spouse_of` need the
  passive (`is located in`, `was married to`) or a direct object
  (`married Doe`); `member_of` needs `joined Acme` (`>obj`) and fires
  on `joined with Acme`; `works_for` fires on `works with Acme`.
  `founded_by` and `acquired_by` miss the active `Smith founded Acme`
  and `Acme acquired Bolt` (the reversed `<nsubj >obj` with swapped
  ends), and `born_in` misses the reduced relative (`Dupont, born in
  Paris`).

The libraries are unchanged by this measurement; narrowing them (the
`>obl` passive agent, the copular attribute, the voice and object
variants above) changes what they extract and is left to its own
change, measured on this fixture.

## Progressive document graph preview

`AnalyzeDocumentProgressive` accepts `turbo:zero-shot-relations` with an
`OpenRelationSchema` and a separate `relation_model`. This lets one request use
a converted GLiNER package for entities and GLiNER2 for relations. Labels carry
optional thresholds, symmetry, and domain/range entity-kind constraints. The
step writes relation mentions, deduplicated `Edges<Entities, Entities>`, and
`Evidence<Relations>`. Graph export preserves entity links and original source
offsets. The demo exposes separate entity and relation model selectors and
label inputs over binary protobuf Connect.

A servable GLiNER2 package is the upstream checkpoint directory with its native
DeBERTa manifest and weights in an encoder subdirectory. It also contains
`config.json`, `added_tokens.json`, `model.safetensors`, and `spm.model`.
`LoadBundle` advertises it as `TASK_RELATION`.
It remains subject to the existing acceptance and local-only distribution
rules. GLiNER2 is the preview relation scorer. GLiREL remains an opt-in native
scorer outside this serving integration; the dependency-path semantic defaults
remain starting rules, not a high-quality general relation model.

The real GLiNER2 check in `analysis/relation/tests/gliner2_real.rs` evaluates
three known positive triples and one same-sentence negative example. At
threshold 0.5 it records TP=3, FP=0, FN=0, precision=recall=F1=1.0. This is a
small release-labelled smoke corpus, not evidence of broad relation accuracy.
The progressive test also checks a real two-model graph, evidence, model
digests, deduplicated edges, and binary graph round-trip. A larger external
relation benchmark is still needed before calling the stage generally
production-quality.

Stage status for the full document graph:

- Included in the preview and served: the GLiNER2 scorer, as the
  `turbo:zero-shot-relations` step of `AnalyzeDocumentProgressive`.
- Included in the preview, in process only: the dependency-path rules
  (`RelationAnnotator`, with the default libraries as starting rules). No
  serving step runs them.
- Experimental: a trained relation classifier head. It is served, as the
  `turbo-tagger:relations` step of `AnalyzeDocumentProgressive` for a bundle
  that carries `head.relations` (see [grpc-analysis.md](grpc-analysis.md)),
  but no release corpus has measured its quality yet.
- Experimental and in process only: GLiREL. No serving step runs it.

`analysis/relation/tests/release_acceptance.rs` runs the rules and the GLiREL
scorer in one analyzer over one shared `Entities` layer, the scorer writing
named instances. It checks each stage's `Relations` mentions (subject,
object, type and confidence read back as entity text), its deduplicated
edges at the best confidence, a rule that matches nothing, and each stage's
provenance and model reference. It then exports and re-imports the graph
over text with multi-byte characters. Every link reads the same entity text,
spans keep their byte offsets, and the re-export is byte-identical. The
GLiREL weights there are the seeded reduced fixture, so that test shows
wiring, not quality.

The served steps are measured on the Re-DocRED test split with gold
entity mentions given, at document scope, by the `relations_corpus`
example of `turbo-kserve`: `turbo-tagger:relations` reaches micro F1
0.4492, and `turbo:zero-shot-relations` (GLiNER2, ten property labels)
0.0971 on those ten properties. GLiNER2 refuses a document longer than
its 512-piece encoder. The record is
`analysis/eval/records/relations-1952/README.md`.

Predicate-argument events with N-ary, anchored arguments are in
[events.md](events.md).
