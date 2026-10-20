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

It writes two layers: `opennlp:relations` (`Relations`), one row
per mention in extraction order, covering both entity spans and
keeping duplicate pattern registrations as duplicate mentions;
and `Edges<Entities, Entities>` with each (subject, object, type)
triple once at its best confidence. Both cross the document
graph boundary (`turbo_relation::export::register_codecs`).

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

The default libraries (`turbo_relation::defaults`) are six UD
shape rules (`subject-object`, `passive-agent`, `apposition`,
`possession`, `prepositional-attachment`, `attribute`) with no
triggers, and eight semantic starting points (`works_for`,
`located_in`, `founded_by`, `headquartered_in`, `member_of`,
`acquired_by`, `born_in`, `spouse_of`) combining a shape, a pivot
lemma, and entity kinds on both ends. Both are starting points
to narrow against a corpus with the explorer, not trained
extractors.
