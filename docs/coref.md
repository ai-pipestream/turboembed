# Coreference (`turbo-coref`)

`analysis/coref/` ports OpenNLP's coreference resolver
(`opennlp.tools.coref`): mention detection, the sieve resolver, the
pair classifier and mention ranker, word and token vectors, the
trainers and the scorer.

## Layers

`CorefAnnotator` reads `View`, `Sentences`, `Tokens`, `PosTags` and
`Entities`, and when present `PhraseChunks`, `Phrases` and `Speakers`.
It writes three layers under the default keys:

| Layer | Id | Content |
|---|---|---|
| `Mentions` | `opennlp:mentions` | One item per mention in text order: its span (fragments kept for a discontinuous entity), sentence, token range, head token, the entity it was lifted from, its kind (proper, nominal, pronominal, demonstrative), its origin (entity, noun phrase, pronoun, gold) and its packed agreement (number, gender, animacy, person) |
| `Groups<Mentions>` | `turbo:groups<opennlp:mentions>` | The chains, singletons included, in first-mention order |
| `Edges<Mentions, Mentions>` | `turbo:edges<opennlp:mentions, opennlp:mentions>` | One edge per link that joined two chains, from anaphor to antecedent, tagged in the `coref-sieve` tag set with the sieve that decided it; the score is 1 for a sieve and the model probability for the ranker or classifier |

`Speakers` (`opennlp:speakers`) is an input: speaker turns labelled with
the speaker, used by the speaker sieve. Every chain of `n` mentions has
`n - 1` edges, so the edge count is the mention count minus the chain
count. `turbo_coref::codecs()` adds the four codecs to the shared layer
codecs for the document graph.

## Models and vectors

`CorefModel::read` opens a `CorefModel` file (component
`CorefAnnotator`, entry `coref.model`, properties `coref.ranking` and
`coref.tokenVectorDimension`); `CorefModel::package` writes one. A
pairwise model scores each antecedent candidate and links the best
above the threshold; a ranking model also scores the new-chain option.
`WordVectors` supplies head similarity features; `TokenVectors` supplies
contextual span features with real values. A model trained with an
encoder requires an encoder of the same dimension.

## Training and scoring

`train::train` trains the pair classifier with any `turbo-train`
algorithm; `train::train_ranking` trains the ranker with AdaGrad, a
`java.util.Random(17)` shuffle and L2. Gold chains are `GoldMention`
spans, or `train::gold_from_layers` from a gold `Mentions` layer and its
groups. `CorefScorer` computes MUC, B-cubed, CEAF-m, CEAF-e, mention
detection scores and the CoNLL average.

## Deviations from OpenNLP

- An entity whose span covers no token is dropped instead of becoming a
  mention without tokens.
- Chains include singletons as groups of one; OpenNLP's `CHAINS` layer
  lists every mention with its chain id, which the groups reproduce.

## Tests

`analysis/coref/tests/` ports `CorefResolutionTest`,
`CorefAnnotatorTest`, `CorefTrainerTest` and `CorefScorerTest`, and
adds an analyzer run (`pipeline.rs`) and document graph round trips
(`graph_export.rs`).
