# Document entity graph

`DocumentEntities` (`turbo:document-entities`, in `turbo-analysis-layers`)
is one row per real-world entity of a document, consolidated across the
layers that mention it. `EntityConsolidator` (in `turbo-entity-linker`)
builds it.

## What each entity carries

| Field | From |
| --- | --- |
| `canonical_name` | the best knowledge-base link's name, else the longest name |
| `kind` | the entity tag with the most score among its names |
| `kb_id` | the best `LinkedEntities` link, if any |
| `geo_coord`, `country_code` | the most confident `Locations` resolution over one of its names |
| `mentions` | every name and coreference mention, in text order, with its `Entities` row, the representative flag and its head role |
| `role_counts` | subject, object, modifier and other heads from `Dependencies` (UD relations, subtype ignored) |
| `mean_sentiment`, `sentiment_mentions` | mean `positive - negative` of the explicit `Aspects` on its mentions |
| `confidence` | the best name score or link score, clamped to `[0, 1]` |

`relations` lifts every `Edges<Entities, Entities>` row to the entities its
arguments belong to, so a relation found on `Lincoln` or on a pronoun's
name is a relation of `Abraham Lincoln`. Rows of one subject, object and
kind merge into one, with the best score and the number of rows as
`support`. An edge within one entity is dropped.

## How names become entities

Only `Entities` is required; every other input is optional and recorded in
the provenance only when present. Names and mentions are merged by:

1. coreference chains (`Groups<Mentions>`, `Groups<Entities>`);
2. the same knowledge-base id;
3. the same folded surface (case and whitespace) and kind;
4. overlapping spans of the same kind;
5. a one-word name and the only longer name of its kind it ends
   (`Obama` with `Barack Obama`, but not when `Michelle Obama` is there
   too).

Rules 3 to 5 are `ConsolidatorConfig::surface_merge` and `alias_merge`,
both on by default. A coreference group with no name (pronouns only) is
not an entity.

## Evaluation

`analysis/entity-linker/tests/consolidate_eval.rs` scores pairwise
precision, recall and F1 of which names are one entity, against gold
clusters, on six short documents (13 gold pairs), one with a tagger error
(a person tagged as a place) the surface merge cannot undo.

| Inputs | P | R | F1 |
| --- | --- | --- | --- |
| no merges | 1.000 | 0.000 | 0.000 |
| coreference chains only | 1.000 | 0.385 | 0.556 |
| name merges only | 0.857 | 0.462 | 0.600 |
| default (both) | 0.929 | 1.000 | 0.963 |

The fixture is small and hand-built; it shows the merges complement each
other, not how they do on any corpus.
