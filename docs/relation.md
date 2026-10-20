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
