# Sentiment

`turbo-doccat` (`analysis/doccat/`) ports OpenNLP 3.x's sentiment
detector, `SentimentME` in `opennlp.tools.sentiment`, at prediction, and
`turbo-train-components` (`analysis/train-components/`) ports its training.
A model trained in Rust is the JVM's model byte for byte, and its
probabilities are the JVM's bit for bit.

Dataset, lexicon and checkpoint permissions are recorded in the
[complex sentiment licence audit](licences.md#complex-sentiment-licence-audit-326).
It distinguishes shipped material, local evaluation and candidates awaiting
source permissions.

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
  nine unit tests: `"battery life is great but the screen scratches"`,
  double propagation over `Dependencies` extracting the conjoined aspect
  `"tiramisu"`, an aspect and an opinion under one governor (`"The battery
  works well"`), implicit sentence-anchored aspects, negation written with
  a curly apostrophe (`"isn’t"`), `PhraseChunks` `NP` extraction of
  non-lexicon noun phrases with leading determiner and opinion trimming,
  `PosTags` noun-sequence extraction and compound expansion, custom
  `AspectCategoryScorer` scoring of explicit and implicit aspects with
  fallback, and exact-span `evaluate_absa_corpus` metrics (`aspect_f1`,
  `opinion_f1`, `triplet_f1`) on a 10-sentence regression fixture written
  beside the rules.
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
`DEFAULT_INCIVILITY_THRESHOLD`, `DEFAULT_INCIVILITY_FUSION_GATE`,
`evaluate_incivility_corpus`, `evaluate_incivility_corpus_with_thresholds`,
`char_span_to_utf8_span`, `char_span_to_token_range`,
`evaluate_toxic_spans`) scores sentences, retrieval chunks, or structure blocks
across the six multi-label dimensions of `Incivility` (`turbo:incivility` from
`#319`), following the Oasis (ICDMW 2019) and HKBU (ICA 2026) label set:

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
3. **Co-occurrence rule for `SevereToxicity` (`with_severe_toxicity_rule`)**:
   when enabled (the default) and `Threat` co-occurs with `Insult`,
   `Profanity`, `IdentityAttack`, or `Intolerance` at or above the label's
   active threshold (`with_threshold` or `with_thresholds`, `0.5` by default),
   `SevereToxicity` is elevated and inherits the supporting token evidence
   spans so multiple labels can fire on the same unit with their own evidence
   spans. When both the lexicon and the vector head score a label at or above
   `with_fusion_gate` (`DEFAULT_INCIVILITY_FUSION_GATE = 0.25`), the fused
   stage combines them probabilistically (`1 - (1 - l)(1 - h)`); otherwise it
   takes `max(l, h)` so sub-gate background head probabilities do not inflate
   lexicon scores (`#963`).
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
dimension and flags (bit 0 a context matrix, bit 1 an encoder; all
`u32` little-endian), then the biases, the weight matrix and the context
matrix as little-endian `f32`, then, for a head bound to an encoder
(`with_model`), the encoder's id, revision and SHA-256, each a `u32`
length and UTF-8 bytes. `with_model` refuses a string over 4096 bytes
with `IncivilityHeadError::Value`, so a head it accepts reads back. A
head read back keeps that encoder, so the annotator still refuses
vectors from any other. `from_bytes` checks each part and the exact
length, and refuses a dimension of 0 or a value that is not finite,
with an `IncivilityHeadError` and never a panic.

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

### Identity-term bias metrics (`evaluate_incivility_bias`)

`evaluate_incivility_bias` measures whether the scores treat text that
mentions an identity group differently, after Borkan et al., "Nuanced
Metrics for Measuring Unintended Bias with Real Data for Text
Classification" (WWW 2019 Companion). Each `GoldBiasUnit` gives a unit's
positive labels and the identity subgroups it mentions. For each label
the result has the ROC AUC over all units and, for each subgroup:

- subgroup AUC, over the units that mention the subgroup;
- BPSN AUC (background positive, subgroup negative): positives that do
  not mention the subgroup against negatives that do. A low value means
  neutral mentions score above real positives elsewhere;
- BNSP AUC (background negative, subgroup positive): positives that
  mention the subgroup against negatives that do not. A low value means
  real positives that mention the subgroup score below neutral text
  elsewhere;
- the false positive rate on neutral mentions: of the units that mention
  the subgroup and are negative for the label, the share scored at or
  above the threshold.

An AUC with no positive or no negative unit is `None`; ties count one
half (`roc_auc`, the Mann-Whitney statistic). A count mismatch or a score
that is not finite is an error.

`IdentityTerms` finds the subgroups a unit mentions from
`(subgroup, term)` pairs. A term of several words matches that run of
tokens. Tokens and terms are compared by the lookup key the emotion
lexicon uses (`analysis/doccat/src/sentiment/token_key.rs`): typographic
apostrophes read as `'`, Unicode punctuation trimmed, and full Unicode
case folding, so "MUSLIM." and "Muslim" match alike.

### Precision, recall and identity-term bias on real data

`analysis/embed/examples/incivility_bias` measures the three stages on
two sets:

- **Civil Comments**, the WILDS release with identity columns
  (`all_data_with_identities.csv`, 448,000 comments, SHA-256
  `403e638c83a225d738a937ff98b61fd0631e30f710d57928c7766d413526b77f`,
  from the CC0 mirror
  [shlomihod/civil-comments-wilds at 3fbfeca](https://huggingface.co/datasets/shlomihod/civil-comments-wilds/tree/3fbfeca80bad0f3aec37e72fa07eff222b6e752f);
  the TFDS download address answers 403). The head is fitted to the
  269,038 train comments, a threshold is chosen on the 45,180 val
  comments, and the numbers are on the 133,782 test comments. A label is
  positive at a rater fraction of 0.5 and above; an identity is
  mentioned at 0.5 and above, for the nine identities Borkan et al.
  report.
- **Jigsaw sentence templates** ("bias madlibs"),
  `sentence_templates/en_sentence_templates.csv` (76,564 sentences,
  SHA-256 `8ed7d07f61000a95a2e18747b8f1fbf43d75257cd342cddef1f3533254374633`)
  and the neutral identity terms of `words.csv` (SHA-256
  `1cca05c17d7ed990125cbdccca6ca15d469dffd1daa363cecd87de6b26305e47`),
  from [conversationai/unintended-ml-bias-analysis at 1244018](https://github.com/conversationai/unintended-ml-bias-analysis/tree/1244018d92bed14bd0a2b0197292add67dbac97f),
  CC BY 4.0, Jigsaw / Conversation AI. Each identity term is its own
  subgroup. The set marks a sentence toxic or not; a toxic sentence
  counts as an insult, the label its templates read as.

Each comment or sentence is one unit with one vector: the encoder's
pooled vector of its text, cut to the model's longest row (256 tokens
for MiniLM, 512 for BGE-small). The head is `IncivilityVectorHead::fit`
over the train comments with 0 and 1 targets (a rater fraction at 0.5
and above is 1), 500 epochs, learning rate 16 and L2 1e-4. Fitted to
the rater fractions with the default rate of 0.5, the head ranks worse
(insult AUC 0.797 against 0.862 on MiniLM) and scores almost nothing
at or above 0.5. Civil
Comments has no intolerance column, so that label's target is 0 and it
has no numbers here. Units are scored with `score_unit_in_context`, and
the records are in `docs/receipts/incivility-bias-889/`. A run takes
about three and a half minutes per encoder on an RTX 4080 SUPER, most
of it fitting the head on one CPU thread.

#### Precision and recall

Per label on the Civil Comments test split, at the annotator's default
threshold of 0.5 and at the threshold with the best micro F1 on the val
split (from 0.05 to 0.5 in steps of 0.05). Each cell is precision /
recall; the lexicon stage reads no vector, so it has one row. At the
default threshold, a stage that predicts nothing for a label has
precision 1 by the convention of `evaluate_incivility_corpus`.

| Stage | Encoder | Threshold | insult | profanity | threat | identity_attack | micro P / R / F1 |
|---|---|---|---|---|---|---|---|
| lexicon | none | 0.5 | 0.646 / 0.306 | 0.223 / 0.520 | 0.102 / 0.034 | 0.113 / 0.003 | 0.498 / 0.230 / 0.315 |
| learned | all-MiniLM-L6-v2 | 0.5 | 0.771 / 0.090 | none | none | 0.593 / 0.016 | 0.755 / 0.062 / 0.115 |
| learned | all-MiniLM-L6-v2 | 0.15 | 0.339 / 0.545 | none | none | 0.291 / 0.397 | 0.326 / 0.461 / 0.382 |
| learned | bge-small-en-v1.5 | 0.5 | 0.774 / 0.085 | none | none | 0.667 / 0.009 | 0.769 / 0.057 / 0.107 |
| learned | bge-small-en-v1.5 | 0.15 | 0.344 / 0.565 | none | none | 0.328 / 0.337 | 0.341 / 0.457 / 0.390 |
| fused | all-MiniLM-L6-v2 | 0.5 | 0.645 / 0.344 | 0.223 / 0.520 | 0.102 / 0.034 | 0.366 / 0.019 | 0.510 / 0.259 / 0.344 |
| fused | all-MiniLM-L6-v2 | 0.2 | 0.434 / 0.566 | 0.223 / 0.520 | 0.102 / 0.034 | 0.327 / 0.267 | 0.387 / 0.470 / 0.425 |
| fused | bge-small-en-v1.5 | 0.5 | 0.646 / 0.347 | 0.223 / 0.520 | 0.102 / 0.034 | 0.304 / 0.012 | 0.512 / 0.259 / 0.344 |
| fused | bge-small-en-v1.5 | 0.2 | 0.444 / 0.582 | 0.223 / 0.520 | 0.102 / 0.034 | 0.369 / 0.208 | 0.404 / 0.464 / 0.432 |

"none" is a label the head predicts for no test comment at that
threshold. The lexicon stage's best val threshold is 0.05, where its
test numbers equal those at 0.5 (its weights are at or above 0.5). The
test split has 9,278 insults, 848 obscene comments, 323 threats, 3,898
identity attacks and one severely toxic comment, so the severe toxicity
numbers (precision 0.02 and recall 1 for the fused stage at 0.2) do not
measure anything. Every positive prediction in every row carries an
evidence span.

#### Identity-term bias

ROC AUC per label on the Civil Comments test split:

| Stage | Encoder | insult | profanity | threat | identity_attack |
|---|---|---|---|---|---|
| lexicon | none | 0.647 | 0.756 | 0.517 | 0.501 |
| learned | all-MiniLM-L6-v2 | 0.862 | 0.814 | 0.923 | 0.923 |
| learned | bge-small-en-v1.5 | 0.877 | 0.824 | 0.900 | 0.922 |
| fused | all-MiniLM-L6-v2 | 0.888 | 0.917 | 0.925 | 0.923 |
| fused | bge-small-en-v1.5 | 0.898 | 0.917 | 0.902 | 0.921 |

Insult, fused stage, per identity on the Civil Comments test split:
subgroup, BPSN and BNSP AUC, and the false positive rate on neutral
mentions at the tuned threshold of 0.2.

| Identity | Comments | Insults | MiniLM sub | MiniLM BPSN | MiniLM BNSP | MiniLM FPR | BGE sub | BGE BPSN | BGE BNSP | BGE FPR |
|---|---|---|---|---|---|---|---|---|---|---|
| black | 4,872 | 600 | 0.821 | 0.850 | 0.872 | 0.074 | 0.837 | 0.852 | 0.892 | 0.077 |
| christian | 13,361 | 587 | 0.882 | 0.935 | 0.809 | 0.024 | 0.894 | 0.937 | 0.836 | 0.026 |
| female | 16,449 | 1,234 | 0.862 | 0.875 | 0.879 | 0.069 | 0.876 | 0.881 | 0.897 | 0.069 |
| homosexual_gay_or_lesbian | 3,810 | 332 | 0.813 | 0.893 | 0.814 | 0.040 | 0.819 | 0.902 | 0.824 | 0.039 |
| jewish | 2,674 | 169 | 0.848 | 0.895 | 0.841 | 0.040 | 0.869 | 0.901 | 0.868 | 0.044 |
| male | 14,295 | 1,227 | 0.861 | 0.865 | 0.887 | 0.074 | 0.873 | 0.876 | 0.899 | 0.071 |
| muslim | 6,982 | 583 | 0.833 | 0.863 | 0.869 | 0.056 | 0.844 | 0.877 | 0.876 | 0.056 |
| psychiatric_or_mental_illness | 1,676 | 273 | 0.861 | 0.772 | 0.941 | 0.160 | 0.882 | 0.848 | 0.922 | 0.087 |
| white | 7,969 | 1,008 | 0.811 | 0.797 | 0.906 | 0.123 | 0.825 | 0.817 | 0.912 | 0.109 |

The lowest BPSN, for `psychiatric_or_mental_illness` and `white`,
means neutral comments that mention them score above insults
elsewhere, and their neutral mentions are flagged most often: the bias
Borkan et al. describe. The lowest BNSP, for `christian` and
`homosexual_gay_or_lesbian`, means insults that mention them score
below neutral comments elsewhere. At the default threshold of 0.5 the
fused stage flags 0.9% to 2.7% of neutral mentions per identity.

Insult on the madlibs, with the lowest BPSN and BNSP over the 50 terms:

| Stage | Encoder | AUC | lowest BPSN | lowest BNSP |
|---|---|---|---|---|
| lexicon | none | 0.562 | 0.562 | 0.562 |
| learned | all-MiniLM-L6-v2 | 0.957 | 0.819 (homosexual) | 0.836 (older) |
| learned | bge-small-en-v1.5 | 1.000 | 0.998 (homosexual) | 0.999 (older) |
| fused | all-MiniLM-L6-v2 | 0.958 | 0.822 (homosexual) | 0.837 (older) |
| fused | bge-small-en-v1.5 | 1.000 | 0.998 (homosexual) | 0.999 (older) |

The lexicon finds few of the templates' toxic adjectives, so its scores
barely depend on the identity term and every term has the same AUC.

### Civil Comments soft-target and parent-context heads, fusion tuning, and Toxic Spans evidence evaluation (`#888`, `#963`, `#890`)

**Evaluation correction pending:** the tables and JSON receipts below are
historical outputs from the original evaluation code, not verification of the
current code. That code used ASCII-only keys instead of production Unicode
keys and reused threshold-dependent scores while selecting thresholds. The
current harness shares the production key, rescoring each candidate and tuning
SevereToxicity after the other labels. The training owner must rerun Civil
Comments and Toxic Spans evaluation before these numbers support #888, #963 or
#890 completion. Published head bytes and historical receipts are unchanged.
No new quality result is claimed here.

Threshold selection first scores each uniform candidate on development. The
five non-severe labels then select their own F1 thresholds; SevereToxicity is
selected last with those five thresholds fixed, and the final configuration
is scored again for the reported dev macro/micro F1. Test uses that same
configuration without selection on test.

The regression `tuned_dev_metrics_match_the_selected_configuration_on_published_head`
requires `TURBO_INCIVILITY_HEAD` to name the published MiniLM `.tivh` file and
checks its SHA-256. `training_source_matches_the_verified_wilds_file` requires
`TURBO_CIVIL_COMMENTS` to name the directory from `civil-comments-wilds`.
Run those tests explicitly with `cargo test --locked -p turbo-analysis-embed
--example incivility_eval -- --include-ignored --skip score_unit_interleaved_timing_receipt`.
The head test uses a finite adversarial vector to verify threshold-dependent
scoring, not a new held-out accuracy estimate.

`analysis/embed/examples/incivility_eval` trains and evaluates four
redistributable `IncivilityVectorHead` artifacts on the zoo-pinned Civil
Comments WILDS dataset (`civil-comments-wilds`, `all_data_with_identities.csv`, 269,038
train / 45,180 val / 133,782 test comments, CC0-1.0) and evaluates evidence
spans on `ipavlopoulos/toxic_spans` (`toxic-spans`, revision
`beaf170538df8275c3447b2e26070f2f82e55272`, CC0-1.0). All four heads are
published under the `models-civil-comments-1` release and loadable via
`IncivilityVectorHead::from_zoo`:

| Zoo constant / entry ID | Encoder | Context matrix | Size (bytes) | SHA-256 | Fit time (s) |
|---|---|---|---|---|---|
| `CIVIL_COMMENTS_MINILM_HEAD` (`turbo-en-civil-comments-minilm-l6-v2-head-1`) | `sentence-transformers/all-MiniLM-L6-v2` (384-dim) | No | 9,371 | `26c1147d5431901e41ca6847ee482d0534c63b0ddcf63e9e99f2b22e472997ba` | 309.79 |
| `CIVIL_COMMENTS_MINILM_CONTEXT_HEAD` (`turbo-en-civil-comments-minilm-l6-v2-context-head-1`) | `sentence-transformers/all-MiniLM-L6-v2` (384-dim) | Yes (`[6, 384]`) | 18,587 | `43f80c1aee79f005cbd058905c73649dfed5536708657172966032dbe36e9b4d` | 387.44 |
| `CIVIL_COMMENTS_BGE_SMALL_HEAD` (`turbo-en-civil-comments-bge-small-en-v1.5-head-1`) | `BAAI/bge-small-en-v1.5` (384-dim) | No | 9,355 | `378ccd113df231a0db4cdb6e3dbe634a094151f8a4f0c02e71a596d172742fa3` | 311.35 |
| `CIVIL_COMMENTS_BGE_SMALL_CONTEXT_HEAD` (`turbo-en-civil-comments-bge-small-en-v1.5-context-head-1`) | `BAAI/bge-small-en-v1.5` (384-dim) | Yes (`[6, 384]`) | 18,571 | `dfd34e2cca4cbe81d53ee6015ede61621950a10938c5e8f2c4a90d606bc6a5f7` | 298.73 |

Training runs 500 epochs of full-batch logistic regression (`learning_rate = 16.0`,
`l2 = 1e-4`) directly on the continuous rater fractions in `0.0..=1.0` (soft
targets), with hard-target (`>= 0.5`) heads fitted alongside for comparison
(MiniLM hard no-context 310.62 s, hard parent-context 407.23 s; BGE-small hard
no-context 212.61 s, hard parent-context 291.96 s). Across all four fits on
269,038 train comments (41,658 with a parent comment vector present in the
dataset; 6,951 in val and 20,702 in test), process peak RSS (`VmHWM`) stays at
2,166,120 KB (`2.07 GiB`) for MiniLM and 2,166,456 KB (`2.07 GiB`) for
BGE-small, with zero heap growth above the pre-fit embedding buffer. Full
JSON receipts are committed in
`docs/receipts/incivility-888-963-890/all-minilm-l6-v2.json` and
`docs/receipts/incivility-888-963-890/bge-small-en-v1.5.json`. As in
`goemotions_eval`, `incivility_eval` retains original tokenizer UTF-8 byte
spans per comment and verifies every evidence span against the borrowed source
comment and production `Analyzer`.

#### Civil Comments held-out test P / R / F1 with per-label dev thresholds (`#888`)

Per-label and macro/micro precision, recall, and F1 on the 133,782 test
comments using per-label thresholds tuned on the 45,180 val comments (`0.05`
to `0.50` in steps of `0.05`), plus uniform-threshold micro F1:

| Stage & Target | Encoder | Context | Uniform thresh (micro P / R / F1) | `insult` P / R / F1 | `profanity` P / R / F1 | `threat` P / R / F1 | `identity_attack` P / R / F1 | Per-label macro P / R / F1 | Per-label micro P / R / F1 |
|---|---|---|---|---|---|---|---|---|---|
| `Lexicon` | none | none | 0.05 (0.4985 / 0.2299 / 0.3146) | 0.6461 / 0.3056 / 0.4149 | 0.2228 / 0.5200 / 0.3120 | 0.1019 / 0.0341 / 0.0510 | 0.1134 / 0.0028 / 0.0055 | 0.1807 / 0.3104 / 0.1306 | 0.4985 / 0.2299 / 0.3146 |
| `Learned` (soft) | `all-MiniLM-L6-v2` | none | 0.20 (0.2963 / 0.4844 / 0.3677) | 0.3274 / 0.5316 / 0.4052 | 0.0999 / 0.1710 / 0.1261 | 0.0556 / 0.3808 / 0.0970 | 0.2406 / 0.5177 / 0.3285 | 0.4539 / 0.4335 / 0.3261 | 0.2662 / 0.5031 / 0.3481 |
| `Learned` (soft) | `all-MiniLM-L6-v2` | parent | 0.20 (0.2969 / 0.4836 / 0.3679) | 0.3269 / 0.5318 / 0.4049 | 0.0984 / 0.1722 / 0.1253 | 0.0546 / 0.3777 / 0.0954 | 0.2422 / 0.5144 / 0.3293 | 0.4537 / 0.4327 / 0.3258 | 0.2660 / 0.5023 / 0.3478 |
| `Learned` (soft) | `bge-small-en-v1.5` | none | 0.20 (0.3647 / 0.3622 / 0.3634) | 0.3618 / 0.5264 / 0.4289 | 0.1010 / 0.1321 / 0.1145 | 0.1225 / 0.2477 / 0.1639 | 0.2613 / 0.4772 / 0.3377 | 0.4744 / 0.3972 / 0.3408 | 0.3099 / 0.4834 / 0.3777 |
| `Learned` (soft) | `bge-small-en-v1.5` | parent | 0.15 (0.2961 / 0.4764 / 0.3652) | 0.4015 / 0.4601 / 0.4288 | 0.0998 / 0.1368 / 0.1154 | 0.1169 / 0.2508 / 0.1594 | 0.2632 / 0.4715 / 0.3378 | 0.4802 / 0.3865 / 0.3403 | 0.3238 / 0.4394 / 0.3728 |
| `Fused` (soft) | `all-MiniLM-L6-v2` | none | 0.25 (0.3664 / 0.4850 / 0.4175) | 0.5102 / 0.4766 / 0.4928 | 0.2228 / 0.5200 / 0.3120 | 0.1446 / 0.1115 / 0.1259 | 0.2388 / 0.5187 / 0.3271 | 0.1877 / 0.6045 / 0.2128 | 0.3555 / 0.4824 / 0.4094 |
| `Fused` (soft) | `all-MiniLM-L6-v2` | parent | 0.25 (0.3668 / 0.4856 / 0.4179) | 0.5106 / 0.4782 / 0.4939 | 0.2228 / 0.5200 / 0.3120 | 0.1502 / 0.1176 / 0.1319 | 0.2403 / 0.5151 / 0.3277 | 0.1889 / 0.6052 / 0.2140 | 0.3571 / 0.4826 / 0.4105 |
| `Fused` (soft) | `bge-small-en-v1.5` | none | 0.20 (0.3726 / 0.5046 / 0.4287) | 0.5645 / 0.4447 / 0.4975 | 0.2228 / 0.5200 / 0.3120 | 0.1185 / 0.2755 / 0.1657 | 0.2587 / 0.4777 / 0.3356 | 0.1947 / 0.6197 / 0.2197 | 0.3725 / 0.4543 / 0.4093 |
| `Fused` (soft) | `bge-small-en-v1.5` | parent | 0.20 (0.3827 / 0.4762 / 0.4244) | 0.5251 / 0.4893 / 0.5066 | 0.2228 / 0.5200 / 0.3120 | 0.1138 / 0.2786 / 0.1616 | 0.2605 / 0.4720 / 0.3357 | 0.1876 / 0.6267 / 0.2205 | 0.3682 / 0.4817 / 0.4174 |
| `Learned` (hard) | `all-MiniLM-L6-v2` | none | 0.15 (0.3261 / 0.4606 / 0.3819) | 0.4153 / 0.4278 / 0.4215 | 1.0000 / 0.0000 / 0.0000 | 1.0000 / 0.0000 / 0.0000 | 0.2905 / 0.3971 / 0.3356 | 0.7843 / 0.3042 / 0.2928 | 0.3707 / 0.3845 / 0.3775 |
| `Learned` (hard) | `bge-small-en-v1.5` | none | 0.15 (0.3409 / 0.4569 / 0.3905) | 0.4277 / 0.4435 / 0.4354 | 1.0000 / 0.0000 / 0.0000 | 1.0000 / 0.0000 / 0.0000 | 0.2555 / 0.5300 / 0.3448 | 0.7805 / 0.3289 / 0.2967 | 0.3491 / 0.4308 / 0.3856 |
| `Fused` (hard) | `all-MiniLM-L6-v2` | none | 0.20 (0.3874 / 0.4702 / 0.4248) | 0.5494 / 0.4512 / 0.4955 | 0.2228 / 0.5200 / 0.3120 | 0.1019 / 0.0341 / 0.0510 | 0.2867 / 0.3984 / 0.3335 | 0.1969 / 0.5673 / 0.2053 | 0.4080 / 0.4316 / 0.4195 |
| `Fused` (hard) | `bge-small-en-v1.5` | none | 0.20 (0.4035 / 0.4642 / 0.4317) | 0.5120 / 0.5099 / 0.5109 | 0.2228 / 0.5200 / 0.3120 | 0.1019 / 0.0341 / 0.0510 | 0.2532 / 0.5305 / 0.3428 | 0.1847 / 0.5991 / 0.2088 | 0.3709 / 0.5054 / 0.4278 |

Fitting to soft rater fractions with per-label thresholds recovers `profanity`
(F1 `0.1261` on MiniLM, `0.1145` on BGE-small) and `threat` (F1 `0.0970` on
MiniLM, `0.1639` on BGE-small) in the `Learned` stage, whereas hard-target
heads never exceed `0.05` on those rare labels (`0` positive predictions). Adding
the parent comment context vector (`CIVIL_COMMENTS_*_CONTEXT_HEAD`) lifts
per-label tuned `Fused` micro F1 from `0.4094` to `0.4105` on MiniLM (`threat`
F1 `0.1259 -> 0.1319`) and from `0.4093` to `0.4174` on BGE-small (`insult` F1
`0.4975 -> 0.5066`).

#### Fusion gate and `SevereToxicity` co-occurrence rule tuning (`#963`)

`incivility_eval` sweeps `with_fusion_gate(g)` across
`g in {0.05, 0.10, 0.15, 0.20, 0.25, 0.35, 1.0}` and
`with_severe_toxicity_rule(true | false)` on Civil Comments val and test.
Because every built-in lexicon entry has weight `>= 0.72` (`0.0` when absent),
`1 - (1 - l)(1 - h) >= l >= 0.72 > t` whenever `l > 0` and `0.0 + h = h`
whenever `l == 0`; threshold decisions across `0.05..=0.50` are therefore
invariant across `g`, while `DEFAULT_INCIVILITY_FUSION_GATE = 0.25` prevents
sub-gate head noise (`h < 0.25`) from inflating raw probability calibration or
custom low-weight lexicon entries.

| Encoder | `fusion_gate` | `use_severe_toxicity_rule` | Val uniform micro F1 | Val per-label macro / micro F1 | Test per-label macro F1 | Test per-label micro F1 | Test `severe_toxicity` TP / FP / FN (P / R / F1) |
|---|---|---|---|---|---|---|---|
| `all-MiniLM-L6-v2` | `0.25` (or `0.05..=1.0`) | `true` (default) | 0.4201 | 0.2120 / 0.4179 | 0.2128 | 0.4094 | 1 / 102 / 0 (0.0097 / 1.0000 / 0.0192) |
| `all-MiniLM-L6-v2` | `0.25` (or `0.05..=1.0`) | `false` | 0.4203 | 0.2120 / 0.4182 | 0.2096 | 0.4103 | 0 / 22 / 1 (0.0000 / 0.0000 / 0.0000) |
| `bge-small-en-v1.5` | `0.25` (or `0.05..=1.0`) | `true` (default) | 0.4334 | 0.2251 / 0.4230 | 0.2197 | 0.4093 | 1 / 261 / 0 (0.0038 / 1.0000 / 0.0076) |
| `bge-small-en-v1.5` | `0.25` (or `0.05..=1.0`) | `false` | 0.4337 | 0.2251 / 0.4233 | 0.2185 | 0.4124 | 0 / 22 / 1 (0.0000 / 0.0000 / 0.0000) |

Enabling `with_severe_toxicity_rule(true)` captures the held-out `SevereToxicity`
positive (`R = 1.0000`, macro F1 `0.2128` vs `0.2096` on MiniLM and `0.2197` vs
`0.2185` on BGE-small), while `with_severe_toxicity_rule(false)` is available
when callers prefer raw head + lexicon `SevereToxicity` precision (`FP = 22`).
The historical configuration-only timing test uses the same changed binary
for both gates and a four-dimensional test head. It is not a comparison
against main; that required receipt is pending. In its warmed 7-round
interleaved release run over 10,000 units
(`analysis/embed/examples/incivility_eval/tests.rs::score_unit_interleaved_timing_receipt`),
`IncivilityAnnotator::score_unit` runs in `min = 19.91 ms`, `median = 29.14 ms`
(`2.91 us / unit`), `max = 39.26 ms` at `fusion_gate = 0.25` against
`min = 22.98 ms`, `median = 33.38 ms`, `max = 50.95 ms` at `fusion_gate = 0.0`.

#### Toxic Spans character-to-byte/token projection and evidence span evaluation (`#890`)

`char_span_to_utf8_span` projects a half-open Unicode codepoint interval
`[char_start, char_end)` from `ipavlopoulos/toxic_spans` (`beaf170538df8275c3447b2e26070f2f82e55272`,
zoo entry `"toxic-spans"`) to a UTF-8 byte `Span` on `ViewId::SOURCE` via
`View::span_from_code_point`. `char_span_to_token_range` maps that byte span to
the overlapping `IxRange<Tokens>` in `opennlp:tokens`. `evaluate_toxic_spans`
computes per-unit macro token P / R / F1 (Following Pavlopoulos et al., SemEval
2021 Task 5, where a unit with both gold and predicted empty sets scores `1.0`),
gold-positive macro token and character-offset P / R / F1, corpus-wide micro
token P / R / F1, and the fraction of predicted-positive units whose evidence
overlaps at least one gold toxic token:

| Split | Stage & Thresholds | Encoder | Predicted-positive units | Gold-pos char macro P / R / F1 | Gold-pos token macro P / R / F1 | All-units token macro P / R / F1 | Micro token P / R / F1 | Gold overlap rate |
|---|---|---|---|---|---|---|---|---|
| `ACL2022` (11,006 units, 5,642 gold-pos, 310 non-ASCII) | `Lexicon` (`0.50`) | none | 5,796 | 0.5491 / 0.5490 / 0.5348 | 0.5485 / 0.5512 / 0.5365 | 0.5654 / 0.5668 / 0.5593 | 0.4910 / 0.4768 / 0.4838 | 0.9320 |
| `ACL2022` (11,006 units, 5,642 gold-pos, 310 non-ASCII) | `Learned` soft (`0.50`) | `all-MiniLM-L6-v2` | 1,234 | 0.0339 / 0.0285 / 0.0302 | 0.0339 / 0.0276 / 0.0294 | 0.4665 / 0.4632 / 0.4642 | 0.1548 / 0.0252 / 0.0433 | 0.2349 |
| `ACL2022` (11,006 units, 5,642 gold-pos, 310 non-ASCII) | `Learned` soft (per-label) | `all-MiniLM-L6-v2` | 6,713 | 0.1214 / 0.1046 / 0.1099 | 0.1214 / 0.1018 / 0.1074 | 0.2758 / 0.2658 / 0.2687 | 0.1020 / 0.0903 / 0.0958 | 0.1851 |
| `ACL2022` (11,006 units, 5,642 gold-pos, 310 non-ASCII) | `Learned` soft (per-label) | `bge-small-en-v1.5` | 6,721 | 0.1303 / 0.1117 / 0.1176 | 0.1303 / 0.1086 / 0.1147 | 0.2858 / 0.2747 / 0.2778 | 0.1094 / 0.0969 / 0.1027 | 0.1951 |
| `ACL2022` (11,006 units, 5,642 gold-pos, 310 non-ASCII) | `Fused` soft (`0.50`) | `all-MiniLM-L6-v2` | 6,173 | 0.5654 / 0.5639 / 0.5501 | 0.5648 / 0.5658 / 0.5514 | 0.5591 / 0.5596 / 0.5523 | 0.4779 / 0.4893 / 0.4836 | 0.9036 |
| `ACL2022` (11,006 units, 5,642 gold-pos, 310 non-ASCII) | `Fused` soft (`0.50`) | `bge-small-en-v1.5` | 5,991 | 0.5567 / 0.5568 / 0.5421 | 0.5563 / 0.5586 / 0.5437 | 0.5621 / 0.5633 / 0.5557 | 0.4838 / 0.4838 / 0.4838 | 0.9162 |
| `ACL2022` (11,006 units, 5,642 gold-pos, 310 non-ASCII) | `Fused` soft (per-label) | `all-MiniLM-L6-v2` | 7,497 | 0.5813 / 0.5879 / 0.5676 | 0.5832 / 0.5891 / 0.5704 | 0.4955 / 0.4985 / 0.4889 | 0.4142 / 0.5103 / 0.4573 | 0.8296 |
| `ACL2022` (11,006 units, 5,642 gold-pos, 310 non-ASCII) | `Fused` soft (per-label) | `bge-small-en-v1.5` | 7,478 | 0.5834 / 0.5925 / 0.5698 | 0.5862 / 0.5934 / 0.5732 | 0.5018 / 0.5055 / 0.4951 | 0.4158 / 0.5152 / 0.4602 | 0.8318 |
| `ACL2022` (11,006 units, 5,642 gold-pos, 310 non-ASCII) | `Fused` hard (per-label) | `bge-small-en-v1.5` | 7,981 | 0.5904 / 0.6021 / 0.5773 | 0.5939 / 0.6027 / 0.5812 | 0.4816 / 0.4861 / 0.4751 | 0.3985 / 0.5231 / 0.4524 | 0.8010 |
| `raw` (16,100 units, 9,658 gold-pos, 459 non-ASCII) | `Lexicon` (`0.05` / `0.50`) | none | 8,482 | 0.5070 / 0.4804 / 0.4756 | 0.5078 / 0.4826 / 0.4782 | 0.5307 / 0.5155 / 0.5129 | 0.5368 / 0.3615 / 0.4320 | 0.9204 |
| `raw` (16,100 units, 9,658 gold-pos, 459 non-ASCII) | `Learned` soft (per-label) | `all-MiniLM-L6-v2` | 9,843 | 0.1313 / 0.1027 / 0.1109 | 0.1318 / 0.0981 / 0.1066 | 0.2592 / 0.2390 / 0.2441 | 0.1293 / 0.0794 / 0.0984 | 0.2020 |
| `raw` (16,100 units, 9,658 gold-pos, 459 non-ASCII) | `Learned` soft (per-label) | `bge-small-en-v1.5` | 9,867 | 0.1398 / 0.1103 / 0.1188 | 0.1404 / 0.1054 / 0.1143 | 0.2694 / 0.2484 / 0.2537 | 0.1374 / 0.0846 / 0.1047 | 0.2117 |
| `raw` (16,100 units, 9,658 gold-pos, 459 non-ASCII) | `Fused` soft (per-label) | `all-MiniLM-L6-v2` | 10,964 | 0.5429 / 0.5184 / 0.5097 | 0.5462 / 0.5189 / 0.5123 | 0.4902 / 0.4737 / 0.4698 | 0.4584 / 0.3916 / 0.4224 | 0.7973 |
| `raw` (16,100 units, 9,658 gold-pos, 459 non-ASCII) | `Fused` soft (per-label) | `bge-small-en-v1.5` | 10,945 | 0.5438 / 0.5226 / 0.5113 | 0.5480 / 0.5227 / 0.5144 | 0.4929 / 0.4777 / 0.4727 | 0.4600 / 0.3953 / 0.4252 | 0.8037 |

Because `Learned` alone has no lexicon spans, it falls back to the single
longest content token in each predicted-positive unit (`select_salient_token_span`),
yielding `0.1099` to `0.1188` gold-positive character F1. Combining the lexicon's
exact phrase spans with the head and salient-token fallback in `Fused` raises
gold-positive character macro F1 from `0.5348` (`Lexicon`) to `0.5676`
(`MiniLM`) and `0.5698` (`BGE-small`) on `ACL2022/data/toxic_spans.csv` and
from `0.4756` to `0.5097` (`MiniLM`) and `0.5113` (`BGE-small`) on the 16,100
raw majority-vote comments.

### Tests and license posture

The [#326 audit](licences.md#complex-sentiment-licence-audit-326) approves
the published Civil Comments / Jigsaw Unintended Bias release under CC0,
including its comment text. It is eligible for redistributable training
and bundles. The actual WILDS CSV is pinned as `civil-comments-wilds`
at mirror revision `3fbfeca80bad0f3aec37e72fa07eff222b6e752f`,
SHA-256 `403e638c83a225d738a937ff98b61fd0631e30f710d57928c7766d413526b77f`.
It is distinct from the Google Parquet `civil-comments` entry; the harness
verifies the CSV size and both digests before training and writes that source
identity into new receipts. The four Civil Comments `IncivilityVectorHead` entries
(`models-civil-comments-1`) and the `toxic-spans` dataset (`CC0-1.0`) are
pinned in `analysis/zoo/index/zoo.json`. Other Jigsaw releases and Social
Bias Frames require separate source-permission checks. Project-authored
incivility rules and fixtures use the repository licence.

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
- `analysis/doccat/src/sentiment/incivility/tests/bias.rs` checks
  `roc_auc` on fixed scores with known values (including ties and empty
  sides), `IdentityTerms` on case, typographic quotes, trailing
  punctuation and a two-word term, and `evaluate_incivility_bias` on a
  nine-unit set whose overall, subgroup, BPSN and BNSP AUCs and neutral
  false positive rates are worked out by hand in the test.
- `analysis/doccat/src/sentiment/incivility/tests/spans.rs` tests
  `char_span_to_utf8_span`, `char_span_to_token_range`, and
  `evaluate_toxic_spans` on ASCII and multi-byte non-ASCII text (Spanish/French
  diacritics, CJK, and emoji), partial-token and multi-token spans, and
  comparison across `Lexicon`, `Learned`, and `Fused` stages (`#890`), while
  `analysis/embed/examples/incivility_eval/tests.rs` checks UTF-8, UTF-16, and
  Unicode codepoint span projection and production `Analyzer` parity on
  corpus comments.
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

## Emotion annotator (`EmotionAnnotator`, `#855`)

`turbo_doccat::sentiment::emotion` (`EmotionAnnotator`,
`EmotionVectorHead`, `EmotionVectorScorer`, `EmotionStage`,
`EmotionUnit`, `evaluate_emotion_corpus`) scores each sentence or chunk on
the 28 labels of `EmotionLayer` (`turbo:emotion`): GoEmotions' 27
emotions and neutral (Demszky et al., ACL 2020), in the dataset's label
order (`EmotionDimension`).

`EmotionAnnotator` (`turbo-doccat:emotion`) reads
`(View, Tokens, Option<Sentences>, Option<Chunks>, Option<Vectors<Sentences>>, Option<Vectors<Chunks>>)`
and writes `EmotionLayer` in one of three stages:

1. **Lexicon** (the default): a seed list of cue words and emoji written
   for this crate (`analysis/doccat/src/sentiment/emotion/lexicon.rs`), a few per label with a weight
   each. A token is looked up with typographic apostrophes (U+2019,
   U+02BC) read as `'`, Unicode punctuation trimmed from both ends but
   for the apostrophe, and the rest case-folded (`turbo-unicode`); an
   emoji as itself. A cue adds its weight to its label by a noisy-or and
   records its token as evidence, unless a negator ("not", "never",
   "hardly", ...) is among the three tokens before it in the unit.
   Neutral is one minus the strongest emotion.
2. **Learned**: an `EmotionVectorScorer` over the unit's vector, such as
   `EmotionVectorHead`, `sigmoid(w · v + b)` per label from a `[28, dim]`
   matrix and 28 biases. The head's scores are the unit's; a positive
   emotion's evidence is the unit's token range, the text the head read.
3. **Hybrid**: each emotion the noisy-or of the lexicon's and the head's
   scores, with the lexicon's cue tokens as evidence and the unit's range
   where the lexicon found none; neutral the smaller of the head's
   neutral and one minus the strongest fused emotion.

The learned and hybrid stages need a head, and a vector for every unit:
a unit with no vector fails the document. A head bound to an encoder
(`with_model`) refuses vectors from any other; the lexicon stage reads
no vectors and checks no encoder. Neutral never carries evidence. The
model id is `goemotions-seed:<stage>`, or `<encoder>:emotion-<stage>`
for a head that names its encoder (its revision then ends in the head's
own hash, so two heads on one encoder differ), and the configuration hash covers
the lexicon, the unit, the stage, the threshold and the head.

`EmotionVectorHead::to_bytes` writes the magic `TEVH`, the version 1,
the dimension and flags (bit 1 an encoder; all `u32` little-endian), the
biases and the weights as little-endian `f32`, then, for a head bound to
an encoder, its id, revision and SHA-256 as `u32`-length-prefixed UTF-8.
`from_bytes` checks each part and the exact length, and a head read back
keeps its encoder. `EmotionVectorHead::fit` fits the 28-label linear head
deterministically by full-batch L2-regularized logistic regression from zero
(`EmotionFitOptions`, default 500 epochs, learning rate `0.5`, L2 `1e-4`), and
`EmotionVectorHead::from_zoo` loads a `KIND_HEAD` / `FORMAT_HEAD` (`TEVH`)
artifact from `turbo-zoo` (`GOEMOTIONS_MINILM_HEAD = "turbo-en-goemotions-minilm-l6-v2-head-1"`
and `GOEMOTIONS_BGE_SMALL_HEAD = "turbo-en-goemotions-bge-small-en-v1.5-head-1"`).

`EmotionAnnotator::with_thresholds` configures per-label activation thresholds
in `EmotionDimension::ALL` order (controlling which emotions receive token
evidence spans and `Provenance` hash differentiation), and
`evaluate_emotion_corpus` / `evaluate_emotion_corpus_with_thresholds` compute
per-label precision, recall and F1, macro and micro averages, and how many
active emotions carry evidence.

### Trained GoEmotions heads and evaluation (`#926`)

`analysis/embed/examples/goemotions_eval` trains and evaluates
`EmotionVectorHead` on the GoEmotions filtered split (`goemotions` in
`analysis/zoo/index/zoo.json`, `google-research/google-research/goemotions` at
`7de3624754e2b0f2f505ae496db3ba194bb40be9`, CC BY 4.0: 43,410 train, 5,426 dev,
and 5,427 test Reddit comments across 27 emotions plus `neutral`). Each comment
is embedded as one unit vector (up to 256 tokens for
`sentence-transformers/all-MiniLM-L6-v2` and 512 tokens for
`BAAI/bge-small-en-v1.5`), and `EmotionVectorHead::fit` is run on the 43,410
train comments with `EmotionFitOptions { epochs: 500, learning_rate: 16.0, l2: 1e-4 }`.
The two resulting `.tevh` heads are published in the `models-goemotions-1`
release under `Apache-2.0` and registered in `analysis/zoo/index/zoo.json` as
`KIND_HEAD` (`FORMAT_HEAD`, `COMPONENT_DOCUMENT_CATEGORIZER`, `CLASS_PERMISSIVE`):

- `turbo-en-goemotions-minilm-l6-v2-head-1` (`43,251` bytes, SHA-256
  `9b280043bff1293280808a38d35d0cf0489cde4c6f92c08423d5c88fc5021723`, bound to
  `sentence-transformers/all-MiniLM-L6-v2` rev `2`)
- `turbo-en-goemotions-bge-small-en-v1.5-head-1` (`43,235` bytes, SHA-256
  `f82b1beb02642576d12e8932b6f3cf0b8487d3f150c0066b576c50011665123a`, bound to
  `BAAI/bge-small-en-v1.5` rev `1`)

Full evaluation receipts are committed in
`docs/receipts/goemotions-926/all-minilm-l6-v2.json` and
`docs/receipts/goemotions-926/bge-small-en-v1.5.json`. Across all stages and
threshold regimes on the 5,427 test comments, every active non-neutral emotion
prediction carries a non-empty token evidence span (`active == active_with_evidence`). The
evaluator retains original tokenizer byte spans rather than one-byte token
positions. Each comment has a byte base in the corpus address space;
subtracting that base recovers the original source span without constructing
a joined string. Evidence is checked against the borrowed original comment,
and empty comments receive no synthetic tokens. Permanent tests cover
UTF-8/UTF-16/code-point projection and parity with the production annotator
on the original GoEmotions comments `eecwqtt` and `ed5f85d`.

#### Macro and micro precision, recall, and F1 on the GoEmotions test split (5,427 comments)

Uniform and per-label thresholds are selected on the 5,426 dev comments (from
`0.05` to `0.95` in steps of `0.05` to maximize dev F1 per label or micro F1
uniformly) and evaluated once on the 5,427 test comments:

| Stage | Encoder | Threshold regime | Macro P / R / F1 | Micro P / R / F1 | Active emotions (with evidence) |
| --- | --- | --- | --- | --- | ---: |
| `lexicon` | none | default (`0.50`) | `0.652 / 0.267 / 0.301` | `0.497 / 0.448 / 0.471` | `1,678 (1,678)` |
| `lexicon` | none | per-label dev tuned | `0.661 / 0.262 / 0.304` | `0.501 / 0.447 / 0.472` | `1,615 (1,615)` |
| `learned` | `all-MiniLM-L6-v2` | default (`0.50`) | `0.836 / 0.103 / 0.127` | `0.498 / 0.343 / 0.406` | `682 (682)` |
| `learned` | `all-MiniLM-L6-v2` | uniform dev tuned (`0.15`) | `0.468 / 0.313 / 0.291` | `0.387 / 0.576 / 0.463` | `4,629 (4,629)` |
| `learned` | `all-MiniLM-L6-v2` | per-label dev tuned | `0.403 / 0.395 / 0.314` | `0.332 / 0.576 / 0.421` | `7,949 (7,949)` |
| `hybrid` | `all-MiniLM-L6-v2` | default (`0.50`) | `0.640 / 0.284 / 0.322` | `0.557 / 0.446 / 0.495` | `1,980 (1,980)` |
| `hybrid` | `all-MiniLM-L6-v2` | uniform dev tuned (`0.30`) | `0.591 / 0.324 / 0.350` | `0.512 / 0.513 / 0.512` | `2,693 (2,693)` |
| `hybrid` | `all-MiniLM-L6-v2` | per-label dev tuned | `0.504 / 0.407 / 0.396` | `0.446 / 0.555 / 0.494` | `5,351 (5,351)` |
| `learned` | `bge-small-en-v1.5` | default (`0.50`) | `0.786 / 0.089 / 0.111` | `0.534 / 0.321 / 0.401` | `624 (624)` |
| `learned` | `bge-small-en-v1.5` | uniform dev tuned (`0.20`) | `0.557 / 0.260 / 0.273` | `0.434 / 0.529 / 0.477` | `3,336 (3,336)` |
| `learned` | `bge-small-en-v1.5` | per-label dev tuned | `0.477 / 0.370 / 0.331` | `0.413 / 0.564 / 0.476` | `5,680 (5,680)` |
| `hybrid` | `bge-small-en-v1.5` | default (`0.50`) | `0.643 / 0.282 / 0.326` | `0.584 / 0.435 / 0.498` | `1,992 (1,992)` |
| `hybrid` | `bge-small-en-v1.5` | uniform dev tuned (`0.30`) | `0.601 / 0.338 / 0.370` | `0.526 / 0.526 / 0.526` | `2,899 (2,899)` |
| `hybrid` | `bge-small-en-v1.5` | per-label dev tuned | `0.486 / 0.440 / 0.410` | `0.446 / 0.597 / 0.511` | `5,539 (5,539)` |

The `hybrid` stage consistently improves over both `lexicon` and `learned`
alone: with per-label dev thresholds, `hybrid` reaches `0.396` macro F1 / `0.494`
micro F1 on `all-MiniLM-L6-v2` (versus `0.304` macro F1 for `lexicon` and
`0.314` macro F1 for `learned`) and `0.410` macro F1 / `0.511` micro F1 on
`bge-small-en-v1.5` (versus `0.331` macro F1 for `learned`), and with a single
uniform dev-tuned threshold (`0.30`) reaches `0.512` micro F1 on
`all-MiniLM-L6-v2` and `0.526` micro F1 on `bge-small-en-v1.5`.

#### Per-label precision, recall, and F1 on the GoEmotions test split

Each cell is `precision / recall / F1` on the 5,427 test comments (`lexicon` at
`0.50`, `learned` and `hybrid` at their per-label dev-tuned thresholds):

| Label | Gold test | `lexicon` (`0.50`) | `learned` (`MiniLM`) | `hybrid` (`MiniLM`) | `learned` (`BGE-small`) | `hybrid` (`BGE-small`) |
| --- | ---: | --- | --- | --- | --- | --- |
| `admiration` | 504 | 0.872 / 0.067 / 0.125 | 0.491 / 0.520 / 0.505 | 0.498 / 0.538 / 0.517 | 0.493 / 0.589 / 0.537 | 0.549 / 0.526 / 0.537 |
| `amusement` | 264 | 0.739 / 0.674 / 0.705 | 0.692 / 0.723 / 0.707 | 0.713 / 0.830 / 0.767 | 0.557 / 0.644 / 0.598 | 0.704 / 0.803 / 0.750 |
| `anger` | 198 | 0.471 / 0.040 / 0.074 | 0.398 / 0.333 / 0.363 | 0.409 / 0.364 / 0.385 | 0.386 / 0.247 / 0.302 | 0.390 / 0.278 / 0.324 |
| `annoyance` | 320 | 0.867 / 0.041 / 0.078 | 0.264 / 0.294 / 0.278 | 0.272 / 0.306 / 0.288 | 0.244 / 0.319 / 0.276 | 0.251 / 0.331 / 0.285 |
| `approval` | 351 | 0.478 / 0.094 / 0.157 | 0.183 / 0.484 / 0.266 | 0.185 / 0.499 / 0.269 | 0.281 / 0.333 / 0.305 | 0.289 / 0.373 / 0.325 |
| `caring` | 135 | 0.348 / 0.059 / 0.101 | 0.233 / 0.393 / 0.293 | 0.292 / 0.311 / 0.301 | 0.240 / 0.370 / 0.292 | 0.241 / 0.407 / 0.303 |
| `confusion` | 153 | 0.765 / 0.085 / 0.153 | 0.112 / 0.686 / 0.193 | 0.531 / 0.170 / 0.257 | 0.259 / 0.510 / 0.344 | 0.261 / 0.523 / 0.349 |
| `curiosity` | 284 | 0.556 / 0.035 / 0.066 | 0.333 / 0.356 / 0.344 | 0.331 / 0.363 / 0.346 | 0.382 / 0.500 / 0.433 | 0.382 / 0.511 / 0.437 |
| `desire` | 83 | 0.238 / 0.410 / 0.301 | 0.382 / 0.157 / 0.222 | 0.525 / 0.253 / 0.341 | 0.432 / 0.193 / 0.267 | 0.511 / 0.277 / 0.359 |
| `disappointment` | 151 | 0.778 / 0.046 / 0.087 | 0.091 / 0.464 / 0.152 | 0.180 / 0.192 / 0.186 | 0.168 / 0.265 / 0.206 | 0.172 / 0.272 / 0.210 |
| `disapproval` | 267 | 0.455 / 0.019 / 0.036 | 0.192 / 0.438 / 0.267 | 0.195 / 0.464 / 0.275 | 0.222 / 0.614 / 0.326 | 0.221 / 0.640 / 0.329 |
| `disgust` | 123 | 0.815 / 0.179 / 0.293 | 0.459 / 0.366 / 0.407 | 0.462 / 0.390 / 0.423 | 0.353 / 0.398 / 0.374 | 0.359 / 0.423 / 0.388 |
| `embarrassment` | 37 | 0.500 / 0.108 / 0.178 | 0.286 / 0.054 / 0.091 | 0.500 / 0.108 / 0.178 | 0.500 / 0.027 / 0.051 | 0.444 / 0.108 / 0.174 |
| `excitement` | 103 | 0.870 / 0.194 / 0.317 | 0.278 / 0.340 / 0.306 | 0.295 / 0.379 / 0.332 | 0.286 / 0.291 / 0.288 | 0.312 / 0.340 / 0.326 |
| `fear` | 78 | 0.679 / 0.244 / 0.358 | 0.481 / 0.487 / 0.484 | 0.500 / 0.564 / 0.530 | 0.565 / 0.500 / 0.531 | 0.537 / 0.564 / 0.550 |
| `gratitude` | 352 | 0.971 / 0.847 / 0.904 | 0.662 / 0.807 / 0.727 | 0.956 / 0.861 / 0.906 | 0.747 / 0.781 / 0.764 | 0.965 / 0.861 / 0.910 |
| `grief` | 6 | 1.000 / 0.000 / 0.000 | 1.000 / 0.000 / 0.000 | 1.000 / 0.000 / 0.000 | 1.000 / 0.000 / 0.000 | 1.000 / 0.000 / 0.000 |
| `joy` | 161 | 0.560 / 0.404 / 0.469 | 0.246 / 0.528 / 0.336 | 0.451 / 0.509 / 0.478 | 0.380 / 0.453 / 0.414 | 0.552 / 0.460 / 0.502 |
| `love` | 238 | 0.716 / 0.697 / 0.706 | 0.560 / 0.651 / 0.602 | 0.710 / 0.739 / 0.724 | 0.624 / 0.580 / 0.601 | 0.649 / 0.807 / 0.719 |
| `nervousness` | 23 | 0.375 / 0.130 / 0.194 | 0.000 / 0.000 / 0.000 | 0.429 / 0.130 / 0.200 | 1.000 / 0.000 / 0.000 | 0.429 / 0.130 / 0.200 |
| `optimism` | 186 | 0.756 / 0.349 / 0.478 | 0.392 / 0.441 / 0.415 | 0.528 / 0.505 / 0.516 | 0.275 / 0.597 / 0.377 | 0.518 / 0.473 / 0.494 |
| `pride` | 16 | 0.667 / 0.375 / 0.480 | 1.000 / 0.000 / 0.000 | 0.667 / 0.375 / 0.480 | 1.000 / 0.000 / 0.000 | 0.667 / 0.375 / 0.480 |
| `realization` | 145 | 0.722 / 0.090 / 0.160 | 0.088 / 0.290 / 0.135 | 0.609 / 0.097 / 0.167 | 0.264 / 0.131 / 0.175 | 0.333 / 0.200 / 0.250 |
| `relief` | 11 | 1.000 / 0.091 / 0.167 | 1.000 / 0.000 / 0.000 | 1.000 / 0.091 / 0.167 | 1.000 / 0.000 / 0.000 | 1.000 / 0.091 / 0.167 |
| `remorse` | 56 | 0.517 / 0.804 / 0.629 | 0.443 / 0.482 / 0.462 | 0.511 / 0.804 / 0.625 | 0.481 / 0.446 / 0.463 | 0.511 / 0.804 / 0.625 |
| `sadness` | 156 | 0.696 / 0.205 / 0.317 | 0.405 / 0.436 / 0.420 | 0.415 / 0.500 / 0.453 | 0.460 / 0.558 / 0.504 | 0.456 / 0.596 / 0.517 |
| `surprise` | 141 | 0.425 / 0.241 / 0.308 | 0.130 / 0.496 / 0.206 | 0.408 / 0.298 / 0.344 | 0.262 / 0.191 / 0.221 | 0.402 / 0.305 / 0.347 |
| `neutral` | 1787 | 0.422 / 0.950 / 0.584 | 0.484 / 0.821 / 0.609 | 0.536 / 0.756 / 0.627 | 0.495 / 0.822 / 0.618 | 0.507 / 0.831 / 0.630 |

### Tests and license posture

The common emotion lexicons (NRC Emotion Lexicon, LIWC) carry non-commercial
terms and are not used; the seed lexicon is project-authored under the
repository licence. GoEmotions (`google-research/google-research/goemotions`,
CC BY 4.0) is approved in [docs/licences.md](licences.md#complex-sentiment-licence-audit-326)
and pinned in `analysis/zoo/index/zoo.json` (`goemotions` and `goemotions-raw`),
and the two trained linear heads (`turbo-en-goemotions-minilm-l6-v2-head-1` and
`turbo-en-goemotions-bge-small-en-v1.5-head-1`) are published in the
`models-goemotions-1` release under `Apache-2.0` with CC BY 4.0 attribution.

- `analysis/doccat/src/sentiment/emotion/tests/mod.rs`: cues with their
  token evidence and the noisy-or; negated cues that do not count, and a
  negator out of reach that does not negate; text with no cue that is
  neutral with no evidence; an emoji cue; the three stages on vectors
  given as data, with their model ids and hashes; per-label thresholds
  (`with_thresholds`, `evaluate_emotion_corpus_with_thresholds`); the errors for a
  missing head, a missing vector and another encoder; the layer through
  the document graph; and the evaluation counts.
- `analysis/doccat/src/sentiment/emotion/tests/head.rs`: the head's
  arithmetic, its bytes with and without an encoder, every cut of them
  and every bad header field, `EmotionVectorHead::fit` on multi-label vectors,
  and `EmotionVectorHead::from_zoo` validation (`Kind::Head`, `Format::Head`,
  SHA-256/SHA-512 verification).
- `analysis/layers/src/sentiment/emotion_tests.rs`: the taxonomy, the
  checks on each pushed item, and `EmotionLayer::check` on broken
  layers.

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


### Python whitespace parity

Raw-text VADER uses `turbo_unicode::whitespace::is_python_space`, the shared
predicate also used by GLiREL, GLiNER2, SaT, matcher attributes and encoding
repair. It recognizes Unicode White_Space plus U+001C through U+001F. Repeated
and leading/trailing separators do not create empty tokens. For example,
`good\u{001F}movie` now has exactly the score of `good movie`, including the
positive compound score 0.44043357076016854. Format characters such as U+200B
remain inside words. The shared predicate is tested against all Unicode code
points; the expected 29-character set was also enumerated with Python 3.14.8.

A release timing comparison against main `b1a8a030` used the same Rust compiler,
same source probes, one fixed logical CPU and six alternating before/after
runs. The probes call the actual relation word splitters and SaT boundary
postprocessor, not a replacement implementation. Timings are total milliseconds
per run; the median is descriptive on a shared host, not a throughput guarantee.

| Path | Calls per run | Before median ms | After median ms |
| --- | ---: | ---: | ---: |
| GLiREL word split | 20,000 | 80.214 | 77.033 |
| GLiNER2 word split | 20,000 | 311.777 | 301.912 |
| SaT cut points | 100,000 | 77.599 | 61.295 |

No slowdown was observed in these probes. They isolate text preprocessing;
encoder inference is not timed. The relation probe repeats multilingual text,
URLs, punctuation and a legacy separator; SaT uses mixed ASCII/Unicode spaces
at sentence boundaries. Run the checked-in probes with:

```sh
cargo run --locked --release -p turbo-relation --example relation_whitespace_timing
cargo run --locked --release -p turbo-sat --example sat_whitespace_timing
```

Retain separately built binaries of main and the change and alternate their
execution order when comparing. Raw six-run samples (milliseconds):

| Path | Before | After |
| --- | --- | --- |
| GLiREL | 93.634, 83.627, 80.103, 78.713, 80.324, 79.540 | 78.568, 78.589, 80.977, 72.234, 75.498, 74.043 |
| GLiNER2 | 364.053, 310.985, 319.985, 312.569, 302.608, 303.863 | 303.214, 351.832, 301.616, 300.673, 302.207, 298.797 |
| SaT | 77.445, 81.898, 81.034, 74.445, 76.735, 77.753 | 62.754, 62.958, 58.943, 64.465, 58.403, 59.835 |
