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

## Contextual token vectors from an encoder bundle (`BundleTokenVectors`)

`turbo_analysis_embed::BundleTokenVectors` implements `TokenVectors` on top of
`VectorAnnotator`'s per-subword contextual vectors (`#709`), aligning subword
pieces to word tokens through the `Subwords` layer (`#960`):

1. `BundleTokenVectors::open(dir, &VectorOptions)` (or `BundleTokenVectors::new(Arc<VectorAnnotator>)`)
   exposes the encoder's output width (`VectorAnnotator::token_dim()`,
   accounting for an optional `ProjectionHead`) and a `config_hash` covering
   the bundle's `ModelRef` and `VectorAnnotator::token_config()`.
2. For each sentence passed to `TokenVectors::vectors` (or a document's
   `Tokens` and `Subwords` layers passed to
   `BundleTokenVectors::embed_document_tokens` or `pool_subword_vectors`),
   subword pieces are aligned to word tokens with `turbo_analysis_embed::align`
   and indexed per word token through `Subwords::of`.
3. Each word token's vector is the componentwise `f64` mean of its aligned
   subword vectors from `VectorAnnotator::embed_subword_tokens` (and
   L2-normalized when the bundle normalizes its vectors). When a byte-level BPE
   piece crosses adjacent word tokens so `Subwords::of` is empty for the
   trailing token, `pool_word_tokens` binary-searches the subword span endpoints
   to pool the pieces overlapping the token's byte span; if the tokenizer
   dropped all characters of a token so no piece overlaps, it falls back to the
   single piece at the insertion point (`a.min(subwords.len() - 1)`).

### Synthetic end-to-end ranking check (`analysis/embed/tests/coref_token_vectors.rs`)

Trained with `train_ranking` (`epochs = 20`, `learning_rate = 0.05`,
`l2 = 0.001`) on 24 project-written template training documents (and static
`Word2Vec` trained on those 24 training documents only) and scored with
`CorefScorer` on 12 held-out template documents whose test sentences differ from
the training split only in the final verb (`"expanded"` and `"grew"` in
training, `"succeeded"` in test). Each template document introduces two
same-type candidate antecedents followed by a modifier-disambiguated nominal
mention (`"The cloud division"` vs `"The bio division"`, `"The solar group"` vs
`"The lunar group"`, etc.):

| Feature set | Bundle | MUC F1 | B-cubed F1 | CEAF-e F1 | CoNLL average F1 |
|---|---|---|---|---|---|
| Surface and syntactic features only | none | 0.5000 | 0.8333 | 0.8333 | 0.7222 |
| Static `WordVectors` (`Word2Vec` skip-gram, `dim = 32`, `epochs = 25`, `seed = 42`) | none | 0.5000 | 0.8333 | 0.8333 | 0.7222 |
| Contextual `TokenVectors` (`BundleTokenVectors`) | `testdata/tiny-bert-bundle` (`dim = 32`) | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| Static `WordVectors` + contextual `TokenVectors` | `testdata/tiny-bert-bundle` (`dim = 32`) | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| Contextual `TokenVectors` (`BundleTokenVectors`) | `sentence-transformers/all-MiniLM-L6-v2` (`c9745ed1d9f207416be6d2e6f8de32d1f16199bf`, `dim = 384`) | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| Static `WordVectors` + contextual `TokenVectors` | `sentence-transformers/all-MiniLM-L6-v2` (`c9745ed1d9f207416be6d2e6f8de32d1f16199bf`, `dim = 384`) | 1.0000 | 1.0000 | 1.0000 | 1.0000 |

Static `WordVectors` compares only the single head noun of non-pronoun
mentions (`"division"`, `"group"`, `"unit"`, `"center"`), so two candidate
antecedents in the same document receive the same head similarity bucket,
whereas `BundleTokenVectors` pools contextualized token representations across
the full mention span (`ctx`, `ctxCluster`, and dense `vp`/`vd`/`vn` features)
and resolves all held-out chains (`CoNLL F1 = 1.0000` vs `0.7222`).

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

## Quotations

`QuotationAnnotator` (`turbo:quotation_annotator`) writes `Quotations`
(`turbo:quotations`, in `turbo-analysis-layers`): each quoted or reported
statement with its content span, speaker span, reporting verb span, the
row of the `Groups<Mentions>` chain the speaker belongs to, its kind
(`Direct`, `Indirect` or `Mixed`) and a confidence. It needs `Tokens`;
`Dependencies`, `Mentions` and `Groups<Mentions>` are used when present.
Quote scanning is shared with the speaker sieve.

Version 2 rejects supplied mentions from a different text view instead of
silently ignoring them. Cue distances larger than the document search to
its ends without overflowing. `Document::add` validates speaker and cue
spans against the quotation view, including UTF-8 boundaries. The speaker
chain remains a documented unchecked raw row number.

| Sieve | Finds | Confidence |
| --- | --- | --- |
| dependencies | a verb of speech outside quotes with an arc into the quote; its `nsubj` is the speaker | 0.9 |
| `ccomp` | indirect speech, or mixed when the complement quotes part of itself | 0.8, 0.7 |
| surface | a verb of speech up to 3 tokens after the closing mark or before the opening one | 0.75 |
| dialog | an unattributed quote after quotes of two chains takes the chain before last | 0.5 |
| none | a quote with no attribution | 0.3 |

A cue after a quote that ends a sentence and leads into the next quote
(`"Hi." Bob said, "Hello."`) is left to the next quote, and a cue before a
quote with a sentence end between them is not used.

`analysis/coref/tests/quotation_eval.rs` is a small hand-built check, not
a corpus score: without a parse, 11 of 11 quotes on eight news and dialog
lines get their gold speaker, and turn alternation gives 4 of 4 dialog
turns the right chain against 2 of 4 without it.
