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

## Validation and transport

The scorer's producer version is 2. Configuration weights must be finite,
nonnegative, and have a positive total. Position decay and heading bonus
must be finite and nonnegative, PageRank damping must be in `[0, 1]`, and
iterations must be positive. Weight normalization uses a wider accumulator
so small and large common weight scales preserve the ranking. The public
`score` method takes the source text and returns a `Result`, validating
mention coordinates and configuration before scoring.

Optional sentence and structure layers must have the same view as the
consolidated entities. Heading bonuses use actual heading fragments;
mentions in a heading envelope's gap receive no bonus. Enabled salience
headers reject mismatched document-entity views or incomplete scores.

`EntitySalience` validates one score for every consolidated entity,
including at binary import. `DocumentEntities` validates its view, source
UTF-8 coordinates and each linked source entity's view and exact envelope.
The mention decoder returns an error for reversed coordinates. Named
source references resolve through the document's recorded dependencies.

`analysis/entity-linker/tests/salience_validation.rs` runs the production
sentence detector, tokenizer, dictionary name finder, consolidator,
salience scorer, bundle tokenizer, chunker and salience-aware header step.
It verifies the resulting header text, full graph/provenance equality,
malformed score components/ranks, incomplete rows, source/view/reference
corruption, heading gaps and configuration limits. These are functional
regressions and make no timing or allocation-free claim.
