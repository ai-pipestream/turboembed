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

`CorefInput` can be built outside a `Document`; before anything indexes
it, every layer's own check, the part-of-speech alignment, every token
and sentence index and every span and fragment against the text are
validated, and a failure is an error. `config_hash` covers the entity
types, the threshold, the model digest and the dimension and
`config_hash` of each vector provider; `WordVectors` and `TokenVectors`
implementations supply their own.

## Models and vectors

`CorefModel::read` opens a `CorefModel` file (component
`CorefAnnotator`, entry `coref.model`, properties `coref.ranking` and
`coref.tokenVectorDimension`); `CorefModel::package` writes one. A
pairwise model scores each antecedent candidate and links the best
above the threshold; a ranking model also scores the new-chain option.
`WordVectors` supplies head similarity features; `TokenVectors` supplies
contextual span features with real values. A model trained with an
encoder requires an encoder of the same dimension. `link_probability`
refuses feature values that are not one finite value per feature and
subtracts the largest logit when the plain exponential sum overflows.

## Training and scoring

`train::train` trains the pair classifier with any `turbo-train`
algorithm; `train::train_ranking` trains the ranker with AdaGrad, a
`java.util.Random(17)` shuffle and L2. Gold chains are `GoldMention`
spans, or `train::gold_from_layers` from a gold `Mentions` layer and its
groups. `CorefScorer` computes MUC, B-cubed, CEAF-m, CEAF-e, mention
detection scores and the CoNLL average.

## Word vectors from a corpus

`word2vec::Word2VecTrainer` trains `WordVectors` from tokenized
sentences with skip-gram (`Word2VecArch::SkipGram`) or CBOW
(`Word2VecArch::Cbow`), both with negative sampling. The vocabulary
keeps words seen `min_count` times, ordered by count descending then
byte order, optionally cut to `max_vocab`; tokens are lowercased as
`String.toLowerCase(Locale.ROOT)` lowercases unless `lowercase` is off.
An occurrence of a word of corpus frequency `f(w)` is dropped with
probability `1 - sqrt(t / f(w))` (`subsample` is `t`; 0 keeps all), noise
words are drawn from counts raised to 0.75, each position draws its
context reach from `1..=window`, and the learning rate decays linearly
to `min_learning_rate`. Training is single-threaded with a SplitMix64
generator seeded by `seed`, so the same parameters and corpus give the
same vectors. `Word2VecParameters::validate` and `train` return a
`Word2VecError` for a zero dimension, window, negative count, epoch
count or `max_vocab`, a dimension above `MAX_DIMENSION` (65536), training
matrices past `MAX_MATRIX_BYTES` (16 GiB each, checked before allocating),
a weight that stops being finite (checked after every epoch), a
learning rate that is not finite and positive, a floor out of
`0..=learning_rate`, a threshold that is not finite and non-negative,
an empty corpus or an empty vocabulary.
`Word2VecVectors` is a `WordVectors` provider whose `config_hash` covers
the parameters, the vocabulary and every corpus token; it writes the
word2vec and GloVe text formats, and `word2vec::read_text` reads either
into `MapWordVectors`.

## Deviations from OpenNLP

- An entity whose span covers no token is dropped instead of becoming a
  mention without tokens.
- Chains include singletons as groups of one; OpenNLP's `CHAINS` layer
  lists every mention with its chain id, which the groups reproduce.

## Tests

`analysis/coref/tests/` ports `CorefResolutionTest`,
`CorefAnnotatorTest`, `CorefTrainerTest` and `CorefScorerTest`, and
adds an analyzer run (`pipeline.rs`), document graph round trips
(`graph_export.rs`) and word2vec training (`word2vec.rs`, `word2vec_limits.rs`).
