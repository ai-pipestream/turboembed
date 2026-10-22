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
  `vaderSentiment.py`'s reference suite plus a handful of sentences where an
  earlier port diverged (`"NO good"`: a `"no"` that negates the next word
  keeps its `0.0` valence through the ALLCAPS and booster rules). A random
  differential run of 5,000 sentences built from lexicon words, boosters,
  negators, idioms and punctuation against `vaderSentiment` 3.3.2 on
  Python 3.13 matched on all but three, which differ only in floating
  point: `sum()` is compensated since Python 3.12, so a valence list that
  cancels to exactly `0.0` here can leave a residue of about `1e-16` there
  and pick up the punctuation emphasis, and a proportion that lands within
  `1e-12` of a rounding half rounds to even here. Those are not chased.
- **`score_sentence` and the "but" rule**: `vaderSentiment` rescales the
  first token carrying a repeated valence (`sentiments.index(sentiment)`)
  rather than the token at that position; `score_text` keeps that quirk for
  parity, `score_sentence` scales by token position so that the evidence
  spans point at the tokens that were actually rescaled.
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
`analysis/doccat/src/sentiment/pragmatics/tests/benchmark.rs`): 18 sarcastic and 14 literal
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

## Aspect and opinion extraction (`RuleAspectSentiment` and `LearnedAspectSentiment`, `#322`)

Aspect-based sentiment analysis (ABSA) is implemented through two annotator
paths that write the same `(Opinions, Aspects)` layers (`turbo:opinions` and
`turbo:aspects`) while recording their distinct annotator IDs in provenance:

1. **Rule path (`turbo_doccat::sentiment::aspect::RuleAspectSentiment`,
   `ID = "turbo-doccat:aspect-sentiment-rule"`)**:
   - Reads `(View, Sentences, Tokens, Option<Dependencies>, Option<PhraseChunks>, Option<PosTags>)`
     and writes `(Opinions, Aspects)`.
   - **Qiu, Liu, Bu, and Chen (2011) double propagation (`R1`..`R4`)**: when
     `opennlp:dependencies` (`Dependencies`) is present on the document,
     propagates seed opinion words from the VADER lexicon and the seed
     aspect terms across dependency arcs for up to three passes or until
     nothing new is found. Relation labels are matched on their part before
     any `:` subtype, so `nsubj:pass` counts as `nsubj`:
     - `R1`: a known opinion extracts its target: the opinion's governor
       when the opinion hangs off it by `amod`, `acl`, `advmod`, `nmod`,
       `dep`, `xcomp`, `acomp` or `ccomp`; a child of the opinion by
       `nsubj`, `obj`, `dobj`, `iobj`, `obl` or `nmod`; or a sibling under a
       shared governor when the opinion is a complement (`acomp`, `xcomp`,
       `ccomp`, `attr`, `advmod`, `amod`, `obj`, `dobj`).
     - `R2`: with `opennlp:pos` present, a known aspect extracts its `amod`
       children that carry an adjectival POS tag; the opinion takes the
       lexicon prior when the lexicon has the word and `+1.0` otherwise.
     - `R3`: a known aspect extracts conjoined aspects over `conj` and `appos`,
       expanding noun compounds (`compound`, `nn`, `flat`) to the left.
     - `R4`: a known opinion extracts conjoined opinions over `conj`, inverting
       the polarity when an adversative conjunction (`"but"`, `"however"`,
       `"yet"`, `"although"`, `"though"`, `"whereas"`) sits between them.
     Without POS tags any word in those positions can become a target, so
     verbs and pronouns are extracted as aspects on bare dependency input;
     `opennlp:pos` restricts targets to nouns.
   - **Clause-aware fallback**: when `opennlp:dependencies` is absent (or
     alongside it), extracts candidate aspect spans from `opennlp:chunks`
     (`PhraseChunks` `NP` spans), `opennlp:pos` (`PosTags` noun sequences),
     and the English aspect phrase table, and links each aspect to the closest
     opinion(s) in the same clause (or dependency arc when available).
   - **Implicit aspects (`AspectTarget::Implicit`)**: when a sentence contains
     opinion expressions (`"Way too expensive and completely overpriced."`)
     and no explicit aspect term, emits one sentence-anchored
     `AspectTarget::Implicit { sentence }` item linking the sentence's opinion
     indices.
   - **Aspect categories**: assigns a coarse category `Tag` from the aspect
     layer's `TagSet` (`DEFAULT_ASPECT_CATEGORIES`: `"general"`, `"battery"`,
     `"display"`, `"performance"`, `"build"`, `"camera"`, `"food"`,
     `"service"`, `"price"`, `"ambience"`, `"software"`, `"connectivity"`)
     using an optional `AspectCategoryScorer` over `Vectors<Aspects>` (such as
     `LabelCentroids` or `ClassifierHead`) with surface-keyword fallback.

2. **Learned path (`turbo_analysis_embed::aspect::LearnedAspectSentiment`,
   `ID = "turbo:aspect-sentiment-learned"`)**:
   - Reads `(View, Sentences, Tokens, Subwords, Option<Vectors<Subwords>>, Option<Dependencies>, Option<PhraseChunks>)`
     and writes `(Opinions, Aspects)`.
   - **Contextual subword vectors (`SubwordVectors`, `VectorAnnotator::embed_subwords`)**:
     extracts each subword piece's contextual hidden state (`Vectors<Subwords>`)
     from its enclosing sentence window via the stage 2 range write (`#263`).
   - **Subword BIO tagging mapped to `Tokens`**: scores each subword piece with
     a 5-class `ClassifierHead` (`["O", "B-ASP", "I-ASP", "B-OP", "I-OP"]`),
     maps subword piece probabilities back to `Tokens` through `Subwords`
     alignments (`subwords.token` or byte-span overlap), and decodes
     clause-bounded `IxRange<Tokens>` spans for aspects and opinions (expanding
     compound noun phrases when `PhraseChunks` is present).
   - **Span-pooled polarity and category heads (`#263`)**: pools the exact
     subword range of each decoded opinion span (`Vectors<Opinions>`) and
     aspect span (`Vectors<Aspects>`) via `VectorAnnotator::embed_spans` and
     scores them with the polarity `ClassifierHead` and optional
     `LabelCentroids` category head, so two aspects with opposite opinions in
     the same sentence (`"battery life is great but the screen scratches"`)
     receive distinct span-pooled embeddings and opposite polarities.

### Tests and license posture

Per `#326`, SemEval ABSA XML datasets (`SemEval-2014 Task 4`, `SemEval-2015 Task 12`,
`SemEval-2016 Task 5`) are non-commercial or carry unspecified redistribution
terms and are not committed to the repository.

- `analysis/doccat/src/sentiment/aspect/mod.rs` tests `RuleAspectSentiment` in
  six unit tests: `"battery life is great but the screen scratches"`,
  double propagation over `Dependencies` extracting the conjoined aspect
  `"tiramisu"`, an aspect and an opinion under one governor (`"The battery
  works well"`), implicit sentence-anchored aspects, negation written with
  a curly apostrophe (`"isn’t"`), and exact-span
  `evaluate_absa_corpus` metrics (`aspect_f1`, `opinion_f1`, `triplet_f1`)
  on a 10-sentence regression fixture written beside the rules. The
  `PhraseChunks` and `PosTags` extraction paths and the
  `AspectCategoryScorer` hook have no test yet.
- `analysis/embed/tests/heads.rs`
  (`learned_and_rule_aspect_sentiment_extract_triplets_and_match_cpu_device`)
  trains `LearnedAspectSentiment::fit` on a 9-sentence fixture over
  `tiny-bert-bundle` (32-dim) and runs both `RuleAspectSentiment` and
  `LearnedAspectSentiment` on the 9 training sentences and on a 4-sentence
  held-out set that recombines the training aspect and opinion words (so it
  checks generalisation across sentences, not across vocabulary), checking
  that both annotators write the same `(Opinions, Aspects)` layer types with
  distinct `Provenance.annotator` IDs and that CPU and device (`TURBO_TEST_DEVICE` /
  `TURBO_TEST_REQUIRE_CUDA=1`) outputs agree within `1e-5`.

## Incivility and toxicity annotator (`IncivilityAnnotator`, `#323`)

`turbo_doccat::sentiment::incivility` (`IncivilityAnnotator`,
`IncivilityLexicon`, `IncivilityVectorScorer`, `IncivilityUnit`,
`DEFAULT_INCIVILITY_THRESHOLD`, `evaluate_incivility_corpus`) scores
sentences, retrieval chunks, or structure blocks across the six multi-label
dimensions of `Incivility` (`turbo:incivility` from `#319`), following the
Oasis (ICDMW 2019) and HKBU (ICA 2026) label set:

- `IncivilityLabel::Insult` (`"insult"`)
- `IncivilityLabel::Profanity` (`"profanity"`)
- `IncivilityLabel::Threat` (`"threat"`)
- `IncivilityLabel::IdentityAttack` (`"identity_attack"`)
- `IncivilityLabel::Intolerance` (`"intolerance"`)
- `IncivilityLabel::SevereToxicity` (`"severe_toxicity"`)

`IncivilityAnnotator` (`turbo-doccat:incivility`) reads
`(View, Tokens, Option<Sentences>, Option<Chunks>, Option<Structure>, Option<Vectors<Sentences>>, Option<Vectors<Chunks>>)`
and writes `Incivility`:

1. **Lexicon and phrase matching (`IncivilityLexicon`)**: deterministic,
   zero-regex token and multi-token phrase matching over `opennlp:tokens`
   that assigns per-label weights in `0.0..=1.0`, guards against negated mild
   adjectives (`"not stupid"`, `"not dumb"`), and records the exact
   `IxRange<Tokens>` evidence span for every matched expression. Each
   `IncivilityLexicon` computes a canonical SHA-256 hex digest over its sorted
   `(phrase, label, weight)` entries for `ModelRef::sha256`.
2. **Multi-label embedding head (`IncivilityVectorScorer`)**: scores
   `Vectors<Sentences>` (`IncivilityUnit::Sentences`) or `Vectors<Chunks>`
   (`IncivilityUnit::Chunks`) through an `IncivilityVectorScorer`. Neither
   head type implements the trait itself: a small wrapper calls
   `ClassifierHead::score_incivility_vector` or
   `LabelCentroids::score_incivility_vector` from `#321` (the adapters in
   `analysis/embed/tests/heads.rs` show both). A scorer that names its
   encoder makes the annotator refuse vectors from any other encoder. A
   unit with no tokens is scored by the head but gets no evidence span,
   since there is nothing to attribute it to. When the embedding
   head predicts a positive label (`score >= threshold`) that had no direct
   lexicon match in that unit, the annotator attributes evidence to the unit's
   highest-salience content token span so every positive label carries at
   least one non-empty `IxRange<Tokens>` evidence span.
3. **Co-occurrence rule for `SevereToxicity`**: when `Threat` co-occurs with
   `Insult`, `Profanity`, `IdentityAttack`, or `Intolerance` at or above
   `threshold` (`0.5` by default), `SevereToxicity` is elevated and inherits
   the supporting token evidence spans so multiple labels can fire on the same
   unit with their own evidence spans.
4. **Optional reply-context conflict scoring (`with_reply_context`)**:
   following Opinion Conflicts (Zhang et al., 2018, arXiv:1809.00289), when
   `use_reply_context` is enabled and `turbo:structure` (`Structure`) is
   present with a parent block or a preceding `Block::Quote` block, scores
   antagonistic second-person reply patterns against the quoted or parent
   block. `ModelRef::id` ends in `:reply-context` or `:no-reply-context`
   after the setting (`oasis-incivility:lexicon:no-reply-context` or
   `<head>+lexicon:no-reply-context`); it does not record whether the
   document had a `Structure` layer.

### The learned vector head (`IncivilityVectorHead`)

`IncivilityVectorHead` is an `IncivilityVectorScorer` of its own: for each
of the six labels, in `IncivilityLabel::ALL` order, the probability
`sigmoid(w · v + b)` of a unit's vector `v`, from a `[6, dim]` weight
matrix and six biases. A head made `with_context` also has a `[6, dim]`
context matrix and adds `w_ctx · p` for a context vector `p`, which
`IncivilityAnnotator::with_vector_context` chooses:
`IncivilityVectorContext::Preceding` (the unit before, none for the
first or after a unit with no vector) or `Parent` (one vector for every
unit, the post the document replies to; `with_parent_vector` sets it). A
head without a context matrix ignores the context vector. The weights
come from the caller; `IncivilityVectorHead::fit` fits them to labelled
vectors (logistic regression per label, full-batch gradient descent from
zero, deterministic), with context vectors when given.

`to_bytes` writes the head as the magic `TIVH`, the version 1, the
dimension, a flag for the context matrix (all `u32` little-endian), then
the biases, the weight matrix and the context matrix as little-endian
`f32`. `from_bytes` checks each part and the exact length, and refuses a
dimension of 0 or a value that is not finite, with an
`IncivilityHeadError` and never a panic.

`IncivilityAnnotator::with_stage` picks the signals:
`IncivilityStage::Lexicon` (the lexicon and the reply rules alone),
`Learned` (the head alone: it needs a head and the unit layer's vectors,
a unit with no vector fails the document, and its scores are reported
as they are, without the `SevereToxicity` rule) or `Fused` (both, the
default, as above). The stage and the vector context are in the
annotator's configuration hash, and the model id names them:
`<head>:learned` for the learned stage and
`<head>+lexicon:<reply context>` for the fused one, where a head with no
encoder named is `incivility-vector-head`, and a `:preceding-vector` or
`:parent-vector` suffix with a vector context.

### Tests and license posture

Per `#326`, external toxicity corpora with custom platform terms (`Jigsaw`,
`Civil Comments`, `Social Bias Frames`) are not redistributed in the
repository.

- `analysis/doccat/src/sentiment/incivility/tests/mod.rs` tests `IncivilityAnnotator`
  on project-authored regression fixtures covering canonical lexicon SHA-256
  digests, multi-label co-occurrence (`Profanity` + `Insult` + `Threat` +
  `SevereToxicity` on one sentence with per-label `IxRange<Tokens>` evidence
  spans), chunk-level `IncivilityUnit::Chunks` scoring with a vector head,
  reply-context conflict scoring (`with_reply_context(false)` vs
  `with_reply_context(true)` over `Block::Quote` + `Block::Paragraph`), and
  per-label precision, recall, F1, and evidence-span coverage via
  `evaluate_incivility_corpus` on a 10-sentence regression fixture written
  beside the lexicon rules. As with `#324`, this fixture pins the rule and
  layer invariants rather than claiming generalisation accuracy on external
  benchmarks.
- `analysis/doccat/src/sentiment/incivility/tests/head.rs` tests the head's
  arithmetic, its context matrix, every cut of its bytes and every bad
  header field, and `fit` on labelled vectors, with and without context.
  `analysis/doccat/src/sentiment/incivility/tests/learned.rs` gives sentence vectors as data, each value a known
  property of its sentence: the three stages disagree where the lexicon
  and the vectors do, the preceding and parent vectors move a reply's
  score through the context matrix, neutral sentences that mention an
  identity ("She is a Muslim software engineer from Chicago.") stay below
  the threshold on every label while attacks on the same groups cross it,
  in the learned and the fused stage, for a configured head and for one
  fitted to rows where the identity term and hostility vary apart (a head
  keyed on the identity term fails the same check), and the learned layer
  round-trips through the document graph.
- `analysis/embed/tests/heads.rs`
  (`incivility_annotator_combines_lexicon_embedding_head_and_reply_context_cpu_device`)
  fits a multi-label `ClassifierHead` (`HeadMode::MultiLabel`) and a 6-label
  `LabelCentroids` head on `tiny-bert-bundle` (32-dim), runs
  `IncivilityAnnotator` end to end with lexicon matching, embedding-head
  scoring, and reply-context conflict detection, verifies that every positive
  prediction carries a non-empty `IxRange<Tokens>` evidence span, and checks
  that CPU and a device (`TURBO_TEST_DEVICE` / `TURBO_TEST_REQUIRE_CUDA=1`)
  agree within `1e-5` with identical evidence spans. The one sentence the
  head must catch without a lexicon hit ("Outsiders have no right to
  participate in our civic institutions.") is in the head's eight training
  rows, so the test shows the head firing on a sentence it was fitted to,
  not generalisation.

## Sentiment and aspect shift as a boundary signal (`SentimentBoundaries` and `ChunkFacets`, `#325`)

`turbo_chunker` generalizes `ChunkEmbedProcessor` and `Centroids` from
`opennlp-sandbox` (`OPENNLP-1833-grpc-uber-server`): instead of forcing a hard
chunk cut whenever the sentence sentiment label changes, sentiment, aspect, and
incivility shifts write a soft boundary signal on `turbo:boundaries`
(`Boundaries`, instance `SENTIMENT = "sentiment"`) that is weighed alongside
document structure (`STRUCTURE`) and embedding drift (`DRIFT`).

1. **Sentiment, aspect, and incivility shift scoring (`SentimentShift`,
   `sentiment_shift`, `SentimentBoundaries`, `SentimentBoundariesProvider`)**:
   - `SentimentBoundaries` (`ID = "turbo:sentiment-boundaries"`, provider name
     `"sentiment-boundaries"`) reads
     `(Sentences, Option<View>, Option<Tokens>, Option<Categories<Sentences>>, Option<Aspects>, Option<Incivility>)`
     and writes `Boundaries` bound to the `SENTIMENT` instance.
   - Across each sentence gap `g` (`0 <= g < n - 1`), `sentiment_shift` pools a
     window of `window` sentences on each side (`default = 1`) and computes up
     to three shift components in millionths (`MICRO = 1_000_000`):
     - **Polarity shift (`polarity_weight`, default `1000`)**: signed polarity
       change across the gap from `Categories<Sentences>` (`pos - neg` or best
       category score) with fallback to `Aspects` compound polarity scores.
     - **Aspect shift (`aspect_weight`, default `1000`)**: whether the two sides
       of the gap discuss disjoint aspect categories/terms or flip polarity on
       the same aspect category/term.
     - **Incivility shift (`incivility_weight`, default `500`)**: onset/offset of
       flagged `Incivility` labels or score jumps across the gap (omitted from
       the active weight denominator at gaps where both windows have zero
       incivility, so civil sentences do not dilute polarity or aspect shifts).
   - Shifts at or above `floor` (`100_000` millionths by default) reduce the gap
     cost by `-(shift * scale / 1_000_000)` (in `Boundaries::UNIT` thousandths),
     and shifts at or above `shift` (`500_000` millionths by default) mark the
     gap's reason as `Cut::TopicShift`.
2. **Weighing with structure and drift (`weigh_with_sentiment`,
   `WeighAllBoundaries`, `WeighProvider`, `Signals::StructureSentiment`)**:
   - `weigh_with_sentiment` combines `(structure, drift, sentiment)` with weights
     `(ws, wd, wsent)` in `i128` arithmetic rounded down via `div_euclid(1000)`
     and saturated to `i32`.
   - `WeighAllBoundaries::default()` sets `structure: Weight::ONE`,
     `drift: Weight::ONE`, and `sentiment: Weight(0)`, so the sentiment boundary
     signal is **off by default** unless a non-zero `sentiment` weight is
     configured in `WeighAllBoundaries`, `WeighProvider` (`"weigh-boundaries"`
     with `sentiment`), or `turbo_eval::chunkers::Signals::StructureSentiment`.
3. **Per-chunk facets (`ChunkFacets` and `ChunkFacetBuilder`, `ID = "turbo:chunk-facet-builder"`)**:
   - `ChunkFacets` (`ID = "turbo:chunk-facets"`, `OF = Some(Chunks::NAME)`,
     `Anchor::aligned::<Chunks>()`) is aligned 1-to-1 with `Chunks` and stores:
     - `polarity: Vec<Polarity>` and `polarity_score: Vec<f64>` in `-1.0..=1.0`
       for each chunk.
     - Sparse-row aspect facets (`aspect_row`, `aspect: Vec<Ix<Aspects>>`,
       `aspect_label: Strs`, `aspect_polarity: Vec<Polarity>`), accessed via
       `aspect_ixs`, `aspect_polarities`, and `aspects_of`.
     - Sparse-row incivility facets (`incivility_row`,
       `incivility_label: Vec<IncivilityLabel>` in strictly ascending label
       order per chunk, and `incivility_max: Vec<IncivilityScores>`), accessed
       via `incivility_labels` and `incivility_scores`.
4. **Boundary Similarity and span-containment evaluation (`boundary_similarity`,
   `chunk_boundary_gaps`)**:
   - `boundary_similarity(gold_gaps, pred_gaps, near_distance)` computes
     Fournier (2013) Boundary Similarity ($B \in [0.0, 1.0]$), pairing exact
     internal sentence-gap matches at weight `1.0` and near-miss transpositions
     within `near_distance` gaps at weight `0.5`.
   - In `analysis/chunker/src/sentiment.rs` and `analysis/chunker/tests/sentiment.rs`,
     on multi-aspect polarity-flip and incivility-shift review fixtures where
     sentence lengths are uniform (`6` sentences x `10` or `100` subwords under a
     `4`-sentence budget), keeping `sentiment = 0` (off by default) cuts at gap
     `1` (`B = 0.50`, splitting the first aspect's 3-sentence evidence span
     across two chunks so single-chunk span containment is `1 / 2 = 50.0%`),
     whereas enabling `sentiment = 1000` moves the cut to the aspect and
     polarity transition at gap `2` with `Cut::TopicShift` (`B = 1.00`,
     single-chunk span containment `2 / 2 = 100.0%`). Because standard
     single-topic narrative retrieval corpora do not pivot on sentiment between
     chunks, the weight remains `Weight(0)` by default.



### Scorer input boundaries

Lexicon sentiment, pragmatics, and incivility validate UTF-8 span boundaries,
layer columns, token ordering, and sentence assignments before scoring. These
scorers require contiguous input spans; discontinuous units return an error
instead of including the text between their fragments. Split such units before
scoring. Incivility evaluates overlapping and nested chunks independently.
Aligned vector inputs must contain one finite row per target. Invalid threshold
configurations and head scores return errors instead of being silently clamped.
Finite builder thresholds retain their documented clamping behavior.

Custom lexicon insertion ignores empty keys and non-finite weights, preserving
an existing valid entry. Incivility lexicon digests include the exact floating
point weights, so changes smaller than six decimal places invalidate provenance.
The three annotators use version `1` for these boundary and scoring fixes.
The raw `score_text` path retains the existing VADER rules. The low-level
`score_sentence` method requires prevalidated spans and token ranges; the
annotator is the checked document entry point.
