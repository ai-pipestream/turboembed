# Entity salience

`EntitySalienceAnnotator` (`turbo-entity-linker`, id
`turbo:entity-salience`) scores each consolidated entity in
`DocumentEntities` (see [entity-graph.md](entity-graph.md)) by how
central it is to the document and writes `EntitySalience`: one
`EntitySalienceScore` per entity, in entity order, with the blended
`salience`, its `rank` (0 is the most salient) and the four features.

It reads `View` and `DocumentEntities`, and `Sentences` and `Structure`
when present. Every feature is in `[0, 1]`:

| Feature | Score |
| --- | --- |
| position | `exp(-position_decay * s)` for `s` the sentence of the first mention (1 minus the byte fraction of the first mention without `Sentences`), plus `heading_bonus` when a mention lies in a `Structure` heading, capped at 1 |
| syntax | `(2 * subjects + objects + 0.5 * modifiers) / (2 * mentions with a role)`, 0 with no roles |
| frequency | `ln(1 + m) / ln(1 + max m)` for `m` the entity's mentions |
| centrality | weighted PageRank over edges between entities sharing a sentence (weight 1 per sentence) and relation edges (weight `support`), scaled so the top entity is 1; 0 for all with no edges |

`salience` is the weighted mean with `SalienceConfig` weights (defaults
position 0.3, syntax 0.3, frequency 0.2, centrality 0.2; decay 0.3,
heading bonus 0.5, damping 0.85, 30 iterations). Ties rank to the lower
entity index, so output is deterministic. `config_hash` covers the
version and every config field. `EntitySalience::top_k(k)` returns the
`k` most salient rows; the layer round-trips through the document graph
codec, and `check` refuses rows out of entity order, scores outside
`[0, 1]`, ranks that are not a permutation and salience that rises with
rank.

PageRank buffers live in a reusable per-thread `Scratch`; a few small
per-document vectors (sentence starts, heading spans, the rank order) are
still allocated per call.

## Chunk headers

`ChunkEntityHeaders` (`turbo-chunker`) takes a `salience:
SalienceHeaders` field, off by default:

- `order_by_salience` orders a chunk's own entries by the best rank of
  the document entity their key belongs to;
- `top_document_entities = k` appends the document's `k` most salient
  entities to every header that does not list them yet, after the
  carried entries, with zero mentions, so they render under "in scope".

Both need `DocumentEntities` and `EntitySalience` on the same view; with
either absent, or with both options off, headers and the config hash are
exactly as before. The `ChunkEntities` format does not change.

## Evaluation

`analysis/entity-linker/tests/salience_eval.rs` ranks the entities of
eight short hand-labelled news-style documents, each with one gold
central entity, and reports top-1 accuracy and mean reciprocal rank of
the gold entity. Mentions and their head roles are labelled by hand; no
tagger, parser or consolidator is in the loop.

| Ranking | Top-1 | MRR |
| --- | --- | --- |
| first mention | 0.375 | 0.688 |
| mention count | 0.625 | 0.812 |
| salience (default config) | 0.750 | 0.875 |

The two misses are kept as written: a topic named in a section heading
and then mostly as an object, and a topic that is mostly an object while
a regulator is the repeated subject. Eight documents is a regression
check on the scoring, not a measure of quality on any corpus, and the
weights were not tuned on it.
