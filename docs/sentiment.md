# Sentiment

`turbo-doccat` (`analysis/doccat/`) ports OpenNLP 3.x's sentiment
detector, `SentimentME` in `opennlp.tools.sentiment`, at prediction, and
`turbo-train-components` (`analysis/train-components/`) ports its training.
A model trained in Rust is the JVM's model byte for byte, and its
probabilities are the JVM's bit for bit.

## What it does

`SentimentME` is a maxent document classifier. Its context is the
document's tokens, each one a feature exactly as written, with no prefix.
The factory in the model's manifest decides what else goes in:

- `SentimentFactory`: nothing; the context is the tokens.
- `EmojiSentimentFactory`: after the tokens, for each token in turn, the
  features OpenNLP's `EmojiAnnotator.collectFeatures` gives it:
  `emojiSentiment=<score>`, `emojiType=<type>`,
  `emojiCategory=<group>` and `emojiRegion=<region>`, those that apply.
  A token must start with a character beyond ASCII and be one emoji (a
  presentation selector is ignored, so `❤️` and `❤` are one symbol); a
  flag gives `emojiType=FLAG`, `emojiCategory=FLAGS` and its region. The
  lookups are `turbo_normalize::emoji::features`
  (`analysis/normalize/src/emoji/annotate.rs`).

The `emojiSentiment` scores come from OpenNLP's `emoji-annotations.txt`,
which its project wrote, -2 to 2 per pictograph. That file states that the
Emoji Sentiment Ranking (Kralj Novak et al. 2015, CC BY-SA 4.0) is not
used in any form, and neither this port nor its test data uses it.

`turbo_doccat::sentiment` has:

- `SentimentModel::open`, which reads a `SentimentModel` file: component
  `SentimentME`, the classifier in `sentiment.model`, and `factory` naming
  one of the two factories. A file without `factory` is refused: OpenNLP
  loads it and then fails in the `SentimentME` constructor, since
  `SentimentModel` has no default factory. A model whose outcome holds
  half of a surrogate pair is refused too, a limit of this port, since
  labels are handed out as `&str`. The manifest's `sentiment.tokenizer`
  names the tokenizer `predict(String)` uses; this port takes tokens and
  does not read it.
- `SentimentFactory::context`, the context of a token list.
- `SentimentMe`: `probabilities` (`probabilities(String[])`, by outcome
  index), `predict`, `best_sentiment` (`getBestSentiment`, the first
  highest probability) and `rank` (best first by `Double.compare`, ties in
  outcome order).

As an annotator (`turbo-doccat:sentiment`) `SentimentMe` reads
`opennlp:tokens` and writes `turbo:categories`: the best outcome and
every outcome's probability, ranked best first by `Categories::ranked`. That is the layer a document categorizer writes,
so no layer type of its own is needed. `SentimentMe::bound` binds the
output to the `sentiment` instance (`turbo_doccat::sentiment::SENTIMENT`),
so a pipeline can run a topic categorizer into the default instance and
sentiment beside it. The layer describes the whole token list it is given,
as `SentimentME.predict(String[])` does; scoring each sentence or span is a
matter of running it on a view of that sentence.

```rust
use turbo_doccat::sentiment::{Scratch, SentimentMe, SentimentModel};
use turbo_maxent::ModelFile;

let model = SentimentModel::open(ModelFile::read("sentiment.bin")?)?;
let detector = SentimentMe::new(model);
let mut scratch = Scratch::new();
let label = detector.predict(&["a", "lovely", "garden", "\u{1F49A}"], &mut scratch);
```

## Training

`turbo_train_components::sentiment` is `SentimentME.train`:

- The samples are read as `SentimentSampleStream` reads them: one per
  line, the sentiment and then the tokens, split by `WhitespaceTokenizer`.
  It is the document category format, refusing a line with fewer than two
  tokens with the same message, so `turbo_formats::native::DocumentReader`
  reads them.
- `sentiment_events` is `SentimentEventStream`: one event per sample, its
  sentiment as the outcome and the factory's context as the predicates.
- `train_sentiment_model` trains with the event trainer the parameters
  name (`turbo_train::trainer::train_events`: GIS, L-BFGS or the
  perceptron), putting nothing into the parameters, and writes the model
  file: the manifest with the caller's language, the trainer's entries and
  `factory`, and the `sentiment.model` entry.

OpenNLP publishes no sentiment model, so there is no published file to
compare with; the parity comes from training on the same samples on both
sides.

## Parity

`SentimentGolden.java` in `analysis/goldens-tool` trains five models on
`analysis/goldens/data/inputs/sentiment-train/feedback.txt`, 56 feedback
lines written for this project with emoji, flags and a character outside
the Basic Multilingual Plane: the token factory with GIS and cutoff 1, and
the emoji factory with GIS and cutoff 1, with the command line trainer's
defaults (GIS, cutoff 5), with L-BFGS and with the perceptron. It writes
the events, the digest of the classifier entry, the manifest and the zip
entries of each model, and for each of the 12 documents in
`analysis/goldens/data/inputs/sentiment` the tokens, the context and the
`turbo:categories` layer.

`analysis/train-components/tests/sentiment.rs` trains each model in Rust
and checks the events, the classifier entry by SHA-256, the manifest and
the entries, then opens the file it wrote and checks every document's
context and probabilities bit for bit. It skips, naming
`cargo run -p turbo-goldens-tool -- generate`, while
`analysis/goldens/data/sentiment` is absent.

`analysis/doccat/tests/sentiment.rs` covers the contexts, the model
checks and the annotator on small models trained in the test.

## Lexicon and negation sentiment baseline (`LexiconSentiment`, #320)

`turbo_doccat::sentiment::lexicon` (`LexiconSentiment`, `VaderLexicon`,
`VaderScore`, `SentenceSentiment`, `NegationMode`) provides a deterministic,
explainable rule- and lexicon-based sentiment baseline that produces exact
token evidence spans (`turbo:opinions`) and sentence-level polarity
(`turbo:categories<opennlp:sentences>`):

- **Lexicon (`VaderLexicon`)**: bundles the MIT-licensed VADER lexicon
  (`vader_lexicon.txt`, 7,506 entries, Copyright (c) 2016 C.J. Hutto) and
  also reads caller-supplied TSV tables (`VaderLexicon::from_tsv`).
- **Emoji features**: Unicode emoji tokens that are not in the ASCII VADER
  lexicon fall back to OpenNLP's Apache-2.0 `emoji-annotations.txt` coarse
  polarity scores (`-2..=2`, via `turbo_normalize::emoji::opennlp_annotation`
  or the document's `turbo:emoji` layer when present). The Emoji Sentiment
  Ranking (CC BY-SA 4.0) is never used.
- **VADER-compatible mode (`LexiconSentiment::score_text`)**: reproduces
  Python `vaderSentiment`'s `SentimentIntensityAnalyzer.polarity_scores`:
  punctuation stripping that preserves short emoticons (`:)`, `:D`),
  ALLCAPS differential emphasis (`C_INCR = 0.733`), degree boosters and
  dampeners (`B_INCR = 0.293`, `B_DECR = -0.293`) with distance damping
  (`1.0`, `0.95`, `0.9`), negation scope (`N_SCALAR = -0.74`), special-case
  idioms (`SPECIAL_CASES`, `never so/this`, `without doubt`, `least`),
  contrastive `"but"` weighting (`0.5` before `"but"`, `1.5` after `"but"`),
  exclamation/question punctuation emphasis, and `score / sqrt(score^2 + 15)`
  compound normalization. Tested against all 28 sentences of
  `vaderSentiment.py`'s reference suite.
- **Evidence spans and sentence categories (`Annotator` for `LexiconSentiment`)**:
  reads `(View, Sentences, Tokens, Option<Dependencies>, Option<Emoji>)` and
  writes `(Opinions, Categories<Sentences>)`:
  - Every token with a non-zero effective valence emits an `Opinion` item in
    `turbo:opinions` pointing at its exact token span (`IxRange<Tokens>`),
    with its prior `polarity`, non-negative `intensity`, and `negated` flag
    such that `opinion.effective_polarity()` equals the token's final
    adjusted valence.
  - Each sentence gets a `Categories<Sentences>` row over the tag set
    `["positive", "negative", "neutral"]` with VADER's `[pos, neg, neu]`
    proportions and the best category selected by the standard `+/- 0.05`
    compound threshold.

### How dependency negation mode differs from VADER's three-token window

`NegationMode::Auto` (the default) uses `opennlp:dependencies` when present
on the document and falls back to VADER's three-token preceding window when
absent, recording the active inputs in the layer's `Provenance.reads` and
the mode in `ModelRef` (`vader:lexicon:auto`, `vader:lexicon:window`, or
`vader:lexicon:dependency`). Dependency mode differs from the fixed
three-token window in two documented ways:

1. **Long-distance negation (`> 3` tokens away)**: in sentences where prepositional
   or adverbial material separates the negator from its predicate head (for
   example, *"The service is not in any way good."* where `"not"` is four
   tokens before `"good"`), VADER's three-token window misses `"not"` and
   scores `"good"` as positive, whereas dependency mode sees `"not"` attached
    directly to `"good"` (`neg` / `advmod`) and negates it.
2. **Clause and constituent boundaries within 3 tokens**: in sentences where a
   negation word appears within three tokens of a sentiment word that belongs
   to a different constituent whose head is not negated (for example, *"Not
   saying everything good."* where `"Not"` modifies `"saying"` while `"good"`
   modifies `"everything"`), VADER's three-token window blindly negates
   `"good"`, whereas dependency mode checks whether the sentiment token or its
   governing complement head has a negation dependent and leaves `"good"`
   unnegated.

## Embedding classifier heads and label centroids (`turbo_analysis_embed`, `#321`)

`turbo_analysis_embed::centroids` and `turbo_analysis_embed::heads` classify
document, sentence, chunk, opinion, and aspect vectors produced by the
turboembed encoder (`#262` and `#263`):

- **Centroids (`centroid`, `CentroidNorm`, `category_sentence_centroids`)**:
  generalizes `org.apache.opennlp.grpc.chunk.Centroids` and
  `ChunkEmbedProcessor.buildCategoryGroup` from `opennlp-sandbox`
  (`OPENNLP-1833-grpc-uber-server`). `centroid` computes the element-wise
  arithmetic mean of a non-empty slice of equal-length `f32` vectors in `f64`,
  with optional post-mean L2 normalization (`CentroidNorm::None` or
  `CentroidNorm::L2`), and is tested against all five cases of
  `CentroidsTest.java`. `category_sentence_centroids` groups a document's
  sentences by their `Categories<Sentences>` label and averages each category's
  sentence vectors into one prototype vector.
- **Prototype classification (`LabelCentroids`)**: each label in a `TagSet` is
  defined by a set of example sentences (`LabelCentroids::from_text_examples`)
  or example vectors (`LabelCentroids::from_vector_examples`) averaged into one
  centroid vector. Inference computes one `f64` dot product per label followed
  by temperature-scaled softmax (`score_single_label`) or scaled sigmoid
  (`score_multi_label`). Centroids serialize to and from a plain typed binary
  format (`TCENT001`) with no pickle and no JSON trickery.
- **Fixed-weight linear and 2-layer MLP heads (`ClassifierHead`)**: runs a
  1-layer linear (`y = W x + b`) or 2-layer MLP (`y = W2 * sigma(W1 * x + b1) + b2`,
  with `HiddenActivation::Relu`, `Gelu`, or `Tanh`) forward pass in
  deterministic `f64` arithmetic, with softmax (`HeadMode::SingleLabel`) or
  independent logistic sigmoid (`HeadMode::MultiLabel`). Heads serialize to and
  from a plain typed binary format (`THEAD001`) and can be fitted
  deterministically from labeled embeddings via full-batch L2-regularized
  cross-entropy gradient descent (`ClassifierHead::fit_single_label` and
  `ClassifierHead::fit_multi_label`).
- **Vector and span-pooling annotators (`ScoringHead`, `VectorHeadAnnotator`,
  `SentenceSpanHead`, `ChunkSpanHead`, `OpinionSpanHead`, `AspectSpanHead`,
  `IncivilityHeadAnnotator`, `PragmaticsHeadAnnotator`)**:
  - `VectorHeadAnnotator<L>` reads `(L, Vectors<L>)` (`opennlp:embeddings<L>`)
    and writes `Categories<L>` (`turbo:categories<L>`).
  - `SentenceSpanHead`, `ChunkSpanHead`, `OpinionSpanHead`, and
    `AspectSpanHead` pool exact subword ranges from `Subwords`
    (`opennlp:subwords`) through the stage 2 range write (`#263`,
    `VectorAnnotator::embed_spans`). Because `OpinionSpanHead` and
    `AspectSpanHead` pool only the subword range of each opinion expression or
    aspect term within its enclosing sentence window, two opinion spans or two
    aspect terms with opposite polarity inside the same sentence receive
    distinct span-pooled embeddings and distinct polarity predictions.
  - `IncivilityHeadAnnotator` and `PragmaticsHeadAnnotator` score
    `Vectors<Sentences>` with a multi-label `ClassifierHead` to populate
    `Incivility` (`turbo:incivility`) and `Pragmatics` (`turbo:pragmatics`).

### Sentiment classification comparison and CPU/GPU parity

`analysis/embed/tests/heads.rs` compares `LexiconSentiment` (`#320`),
`LabelCentroids` (`#321`), and `ClassifierHead` (`#321`) on a self-contained
12-sentence evaluation benchmark (`testdata/tiny-bert-bundle`, 32-dim) and on
471 MPQA subjective polarity spans from MASC 3.0.0 (`CC BY 3.0 US`, 376 train /
95 test spans) embedded with `sentence-transformers/all-MiniLM-L6-v2` (384-dim,
mean-pooled, L2-normalized):

| Dataset | Embedding model | `LexiconSentiment` (`#320`) | `LabelCentroids` (`#321`) | `ClassifierHead` (`#321`) |
| --- | --- | --- | --- | --- |
| Self-contained benchmark (12 test sentences) | `tiny-bert-bundle` (32-dim) | `12 / 12` (`100.0%`) | `10 / 12` (`83.3%`) | `10 / 12` (`83.3%`) |
| MASC 3.0.0 MPQA polarity spans (376 train, 95 test) | `sentence-transformers/all-MiniLM-L6-v2` (384-dim) | `87 / 95` (`91.6%`) | `81 / 95` (`85.3%`) | `84 / 95` (`88.4%`) |

CPU and CUDA parity (`heads_and_centroids_cpu_and_device_parity` in
`analysis/embed/tests/heads.rs`) was verified on `NVIDIA GeForce RTX 4080 SUPER`
(`Execution { backend: "cuda", device: "NVIDIA GeForce RTX 4080 SUPER", precision: "f32 (model)" }`):
repeated GPU runs produce bit-for-bit identical probability vectors, and CPU
and CUDA `LabelCentroids` and `ClassifierHead` scores agree within `1e-5` with
identical predicted labels.

## Sarcasm and irony detection (`PragmaticsAnnotator`, `#324`)

`turbo_doccat::sentiment::pragmatics` (`PragmaticsAnnotator`,
`PragmaticsVectorScorer`, `DEFAULT_MIN_POSITIVE_POLARITY`,
`DEFAULT_MIN_NEGATIVE_POLARITY`, `DEFAULT_PRAGMATICS_THRESHOLD`) detects
sentence-level sarcasm and irony and writes `Pragmatics` (`turbo:pragmatics`),
aligned to `Sentences`. Scores are never folded into `turbo:categories`
polarity, so a literal positive sentence's polarity is never flipped by the
sarcasm detector; downstream consumers inspect `turbo:pragmatics` alongside
`turbo:categories` and `turbo:opinions` to decide whether to flip or discount
a positive polarity score.

`PragmaticsAnnotator` (`turbo-doccat:pragmatics`) reads
`(View, Sentences, Tokens, Option<Opinions>, Option<Categories<Sentences>>, Option<Emoji>, Option<Vectors<Sentences>>)`
and combines three signals:

1. **Riloff et al. (EMNLP 2013) incongruity (`Some(Incongruity)`)**: contrasts
   a positive sentiment expression (`Opinions` with effective polarity
   `>= min_positive_polarity`, sarcastic positive framing cues such as
   `"oh great"`, `"love waiting"`, `"just what I needed"`, `"thrilled to"`, or
   positive lexicon spans) against a negative situation phrase (`"waiting in
   line"`, `"on hold"`, `"stuck in traffic"`, `"flight cancelled"`, `"server
   crashed"`, `"working on saturday"`, from a fixed English phrase table) or
   a negative opinion span in the same sentence. Clauses separated by an adversative conjunction (`"but"`,
   `"however"`, `"although"`, `"though"`, `"yet"`, `"whereas"`, `"while"`,
   `"except"`) without an explicit sarcastic cue are treated as literal mixed
   sentiment (for example, *"The hotel room was lovely, but the flight was
   delayed."*) and excluded from incongruity triggering.
2. **Text against emoji polarity mismatch (`Some(Incongruity)`)**: contrasts
   positive text opinions against negative emoji (`turbo:emoji` from `#183`,
   scored via OpenNLP's `opennlp_annotation` and Unicode 18.0 CLDR short names,
   such as `😭`, `😒`, `🙄`, `💔`, `😡`), or negative text opinions and
   situations against positive emoji (`😍`, `🎉`, `🥳`, `😂`, `👍`), recording
   the positive and negative token spans in `Incongruity { positive, negative }`.
3. **Sentence embedding head (`evidence: None`)**: scores `Vectors<Sentences>`
   with a `PragmaticsVectorScorer` (such as a `ClassifierHead` or
   `LabelCentroids` head from `#321`) or reads `sarcasm`/`irony` probabilities
   from `Categories<Sentences>` to catch implicit irony with no explicit
   surface contrast. When the embedding head fires without explicit contrast,
   `PragmaticsItem.evidence` is `None`.

### Tests and license posture

Per `#326`, external sarcasm datasets with Twitter/Reddit platform terms or
unspecified licenses (`iSarcasm`, `SARC`, `SemEval-2018 Task 3`) are not
redistributed in the repository, so there is no measured accuracy for this
annotator yet. What exists is a 32-sentence project-written regression
fixture (`pragmatics_benchmark_f1_and_evidence_invariants` in
`analysis/doccat/src/sentiment/pragmatics.rs`): 18 sarcastic and 14 literal
sentences written beside the cue and situation tables, with the
embedding-head cases scored from hand-set two-dimensional vectors. It pins
the behaviour of the rules and the evidence invariants (every incongruity
and emoji-mismatch positive carries non-empty, non-overlapping
`Some(Incongruity { positive, negative })` token spans inside its sentence;
head-only positives carry `evidence: None`). It says nothing about how the
rules generalise, and the scores are rule weights on a `0.0..=1.0` scale,
not calibrated probabilities. `analysis/embed/tests/heads.rs`
(`pragmatics_annotator_combines_incongruity_emoji_and_embedding_head`) runs
the three signals end to end over `tiny-bert-bundle` with a fitted
`ClassifierHead` as the sentence scorer and a `LabelCentroids` head written
beside it, and compares CPU with a device when `TURBO_TEST_DEVICE` is set.

