# Document categorization

`turbo-doccat` (`analysis/doccat/`) implements document, chunk, block, and
sentence categorization, zero-shot natural-language-inference (NLI)
classification, intra-document claim verification, probability calibration, and
the lexicon, rule, and vector-head sentiment and polarity suite described in
[sentiment.md](sentiment.md).

## Architecture and module layout

The crate is organized around typed layers in `turbo-analysis-layers` and
backend-agnostic scorer traits:

- **Shared category layer (`Categories`, `SpanCategorizable`)**:
  `Categories` (`turbo:categories`) stores the ranked category distribution for
  a whole document, while `Categories<Sentences>`, `Categories<Structure>`, and
  `Categories<Chunks>` store one category distribution per sentence, structure
  block, or retrieval chunk.
- **Vector-scorer traits (`turbo_doccat::scorers`)**:
  `PolarityVectorScorer`, `PragmaticsVectorScorer`, `EmotionVectorScorer`,
  `IncivilityVectorScorer`, and `AspectCategoryScorer` decouple rule and lexicon
  annotators from the neural embedding heads in `turbo-analysis-embed`, and
  re-export the sentiment and pragmatics layer types (`Opinions`, `Aspects`,
  `EmotionLayer`, `Incivility`, `Pragmatics`, `Polarity`, `PolarityScores`).
- **Document and sentence polarity (`turbo_doccat::sentiment::polarity`)**:
  `PolarityAnnotator` (`turbo-doccat:polarity`) and
  `SentencePolarityAnnotator` (`turbo-doccat:sentence-polarity`) score
  `positive`, `negative`, and `neutral` polarity across `Lexicon`, `Learned`,
  and `Hybrid` stages using `LexiconSentiment` and `PolarityVectorHead` (trained
  with `turbo-trainer`).
- **Multi-label linear categorization (`turbo_doccat::multilabel`)**:
  one-vs-rest linear heads (`MultiLabelDoccatModel`, `MultiLabelCategorizer`,
  `MultiLabelSpanCategorizer`) with per-label threshold calibration
  (`ThresholdCalibrator`) and clean abstention (`Tag::NONE`).
- **Zero-shot NLI categorization (`turbo_doccat::zeroshot`)**:
  `ZeroShotCategorizer`, `ZeroShotAnnotator`, and `ZeroShotSpanAnnotator` over
  any `NliScorer` (such as `turbo_analysis_embed::BundleNliScorer`).
- **Intra-document claim verification (`turbo_doccat::claim_verify`)**:
  `ClaimVerificationAnnotator` and `ClaimVerifyConfig`, writing
  `Edges<L, L>` (`ClaimLinks` for chunks) with `Entails`, `Contradicts`, and
  `Redundant` edges scored by an `NliScorer`.
- **Sequence-classification categorization (`turbo_doccat::dl`)**:
  `DocumentCategorizerDL` and `DlSpanCategorizer` over any `DlSequenceScorer`
  (such as `turbo_analysis_embed::BundleSequenceScorer`).
- **Feature extraction, feature selection, and calibration (`features`,
  `selection`, `calibrated`)**:
  `BagOfWordsFeatureGenerator`, `NGramFeatureGenerator`,
  `EmojiFeatureGenerator`, chi-square and mutual-information feature selection
  (`FeatureSelection`, `SelectionMetric::Chi2`,
  `SelectionMetric::MutualInformation`), and post-hoc probability calibration
  (`CalibratedCategorizer` over
  `turbo_calibration::Transform::{Temperature, Platt, Isotonic}` and
  `turbo_calibration::metrics::reliability`).

## Multi-label categorization (`MultiLabelDoccatModel`)

`turbo_doccat::multilabel` scores every known label independently so a document
or span can receive zero, one, or many categories:

- `MultiLabelDoccatModel` holds one linear head per label (`LinearHead`, weights
  and bias), trained one-vs-rest by `train_multilabel` with full-batch gradient
  descent over `DoccatContextGenerator` features. A logistic sigmoid maps each
  head's linear score to an independent probability in `0.0..=1.0`; scores do
  not sum to `1.0`.
- Each label carries its own decision threshold in `(0.0, 1.0)` (default `0.5`).
  `MultiLabelCategorizer` emits every label whose score reaches its threshold,
  optionally capped by `max_labels`, floored by `min_score` (a hard floor below
  which a label never emits), and back-filled up to `min_labels` with the
  highest-scoring labels that clear `min_score`. When no label clears its
  threshold and `min_labels` is `None`, the prediction is empty: the annotator
  writes `Tag::NONE` (clean abstention rather than a forced argmax) while
  retaining all label scores on the item. `MultiLabelSpanCategorizer<L>` applies
  the same logic per sentence, structure block, or chunk.
- `ThresholdCalibrator` tunes per-label thresholds on a held-out validation set:
  for each label it sweeps validation scores as candidate cut points, selects
  the threshold maximizing $F_\beta$ ($\beta$ configurable, default `1.0`,
  breaking ties toward the highest threshold), and optionally enforces a
  minimum-precision floor (falling back to the highest-precision candidate when
  none reaches the floor). Labels with no validation positives retain the
  default threshold.
- `multilabel::metrics::evaluate` computes micro-, macro-, and sample-averaged
  precision, recall, and F1, plus Exact Match (the fraction of documents whose
  predicted label set equals the gold set), under the crate's division
  conventions (predicting nothing is precision `1.0`, no gold positive is recall
  `1.0`, both `0` is an F of `0.0`).
- `MultiLabelDoccatModel::write` and `MultiLabelDoccatModel::open` serialize the
  model, feature generator chain, cardinality bounds, and calibrated thresholds
  with a trailing CRC-32C checksum (format version 2; version 1 payloads open
  with `min_labels = None`). Truncated or corrupted bytes return
  `MultiLabelError` without panicking. Archive wrapping through `ModelFile` is
  disabled; use the native entry codec.

### Testing

- `analysis/doccat/tests/multilabel.rs` tests independent scoring, `Tag::NONE`
  abstention in `Categories`, threshold calibration sweeps ($\beta$, precision
  floor, tie-breaking, zero-positive labels), and training on an imbalanced
  corpus where calibrated thresholds strictly improve macro-F1 over a flat
  `0.5` threshold.
- `analysis/doccat/tests/multilabel_extra.rs` tests cardinality bounds
  (`min_labels`, `max_labels`, `min_score`), format v1/v2 backward
  compatibility, corruption rejection, deterministic `config_hash` values, and
  evaluation metrics against hand-computed values.

## Zero-shot categorization (`ZeroShotCategorizer`)

Zero-shot categorization requires no task-specific training data: the caller
supplies candidate labels at runtime, and each label becomes the hypothesis of
an NLI pair whose premise is the input text (default template
`"This example is {}."`, validated to contain a single `{}` placeholder with
surrounding text). The sequence-pair model sits behind the `NliScorer` trait,
which declares its outcome names, pair token budget, and token counter.

- Outcome names resolve case-insensitively at construction to either a 3-way NLI
  head (`entailment`, `neutral`, `contradiction`) or a 2-way head
  (`entailment`, `not_entailment`). Empty label lists, blank or duplicate
  labels, and hypotheses that exceed the pair token budget are rejected with
  `ZeroShotError`.
- Single-label mode applies a softmax over entailment logits across candidate
  labels (scores sum to `1.0`). Multi-label mode applies an independent two-way
  softmax over `(entailment, contradiction)` for each label.
- Premises longer than the pair budget (minus the longest hypothesis and the
  model's pair special tokens: 3 for BERT-style templates, 4 for RoBERTa-style)
  are split into word windows measured with `NliScorer::count_tokens`, with
  optional word overlap. Per-window logits combine by max, mean, or
  premise-length-weighted mean before the mode's softmax runs.
- `ZeroShotAnnotator` writes document-level `Categories`, and
  `ZeroShotSpanAnnotator<S, L>` writes `Categories<L>` per sentence, structure
  block, or chunk. Blank spans receive zero scores rather than failing the
  document.

### Native bundle NLI scorer (`turbo_analysis_embed::BundleNliScorer`)

`turbo_analysis_embed::BundleNliScorer` implements `NliScorer` on a
sequence-classification bundle (`TASK_EMBED` with a `CLASSIFIER_SEQUENCE` head
in `head.bin`):

1. `BundleNliScorer::open(dir, &EncoderOptions)` validates the bundle's
   `PairTemplate`, loads the `CLASSIFIER_SEQUENCE` head, and opens the `Encoder`
   session. `BundleNliScorer::open_categorizer` and
   `BundleNliScorer::open_annotator` construct a `ZeroShotCategorizer` or
   `ZeroShotAnnotator` directly from a bundle directory, candidate labels, and
   `ZeroShotConfig`.
2. `NliScorer::count_tokens` encodes text without special tokens so hypotheses
   and long premises are budgeted using exact subword counts.
3. `NliScorer::score_pairs` encodes each `(premise, hypothesis)` pair with
   `Tokenizer::encode_pair` and batches pairs into chunks of `max_batch` rows
   through `Encoder::run_cls` and `Head::logits_into`, recording backend
   `Execution` provenance on the output `Categories` layer.
4. `bundle/recipes/nli-minilm2-l6-h768.json` packages
   [`cross-encoder/nli-MiniLM2-L6-H768`](https://huggingface.co/cross-encoder/nli-MiniLM2-L6-H768)
   (revision `b95119ce93d3e065de6214e38cd4a97b0f2f2c6d`, 6-layer 768-dim
   `RobertaForSequenceClassification`, zoo entry
   `cross-encoder-nli-minilm2-l6-h768`). In `analysis/embed/tests/zeroshot.rs`,
   the bundle matches PyTorch `AutoModelForSequenceClassification` on all 8
   reference pairs within `3.81e-6` (`< 1e-4` gate) and categorizes the 4 topic
   examples (`food` `0.9660`, `finance` `0.3593`, `sports` `0.9858`, `travel`
   `0.9870`) in one batched forward pass per document (`12.24 ms/doc` on CPU).

## Intra-document claim verification (`ClaimVerificationAnnotator`)

`turbo_doccat::claim_verify::ClaimVerificationAnnotator<S, L>` scores
entailment, contradiction, and redundancy edges (`Edges<L, L>`, aliased as
`ClaimLinks` for `Chunks`) between units of a document using any `NliScorer`:

1. **Candidate pruning (`ClaimVerifyConfig`)**: to avoid scoring all $O(n^2)$
   unit pairs, a pair of non-overlapping units becomes a candidate when the two
   units share a named-entity surface (`Entities`), when their `Vectors<L>`
   cosine similarity lies in `[min_similarity, max_similarity]`, or when their
   content-word Jaccard overlap is at least `min_word_overlap`.
2. **Ranking and budgeting**: pairs that exceed the scorer's token budget are
   dropped upfront; remaining candidates are ranked by shared entity count,
   vector similarity, and lexical overlap, and the top `max_pairs` (default
   `64`) are scored in both directions in a single `score_pairs` batch.
3. **Edge emission**: each direction's logits are softmaxed. Mutual entailment
   at or above `entail_threshold` (default `0.8`) emits `ClaimRelationKind::Redundant`;
   one-way entailment emits `ClaimRelationKind::Entails` from premise to
   hypothesis; contradiction in either direction at or above
   `contradict_threshold` (default `0.8`) emits `ClaimRelationKind::Contradicts`.
   Unit tests in `analysis/doccat/src/claim_verify.rs` cover candidate pruning,
   bidirectional NLI scoring, and edge emission.

## Transformer sequence classification (`DocumentCategorizerDL`)

`turbo_doccat::dl` classifies documents and spans with a sequence-classification
transformer model (`DlSequenceScorer`):

1. `DocumentCategorizerConfig::from_json` (or `read_categories_from_json`)
   parses the `id2label` map from a model `config.json` into contiguous
   indices `0..N` and non-empty category names.
2. `DlInferenceOptions` configures text preprocessing and sliding-window
   chunking:
   - `lower_case` (default `true`), `normalize_whitespace` (default `false`),
     and `normalize_dashes` (default `false`).
   - `document_split_size` (default `250`) and `split_overlap_size` (default
     `50`), with step `document_split_size - split_overlap_size`.
   - `chunking_mode`: `DlChunkingMode::WhitespaceWords` (sliding windows over
     whitespace-delimited words) or `DlChunkingMode::Subwords` (sliding windows
     directly over pre-tokenized `Subwords` IDs on the view).
3. `DocumentCategorizerDL<S, C>` runs its `DlSequenceScorer` (`S`) over each
   window, applies a numerically stable `f64` softmax to each window's `f32`
   logits (rejecting non-finite logits with `DlDoccatError`), and aggregates
   window probability distributions through `ClassificationScoringStrategy`
   (`AverageClassificationScoringStrategy` by default).
4. As an annotator, `DocumentCategorizerDL` writes document-level `Categories`,
   and `DlSpanCategorizer<S, L, C>` writes `Categories<L>` for `Chunks`,
   `Structure`, or `Sentences`.
5. `turbo_analysis_embed::BundleSequenceScorer` implements `DlSequenceScorer` on
   native backends (CPU, Metal, CUDA, Vulkan, HIP) using `Encoder` with CLS
   pooling and the bundle's `CLASSIFIER_SEQUENCE` head (`head.bin`). Tested in
   `analysis/doccat/src/dl/mod.rs` and `analysis/embed/tests/doccat_dl.rs`.

## Feature extraction, selection, and probability calibration

- **Feature generators (`turbo_doccat::features`)**:
  `DoccatContextGenerator` runs a chain of `BagOfWordsFeatureGenerator`
  (`bow=<token>`, with optional all-letter filtering via `PatternMode`),
  `NGramFeatureGenerator` (`ng=:<t_1>:...:<t_k>`), and `EmojiFeatureGenerator`
  (`emoji=<token>`).
- **Feature selection (`turbo_doccat::selection`)**:
  `FeatureSelection` and `SelectedFeatures` rank candidate features over
  `FeatureDocument` instances by `SelectionMetric::Chi2` or
  `SelectionMetric::MutualInformation` before model fitting, tested in
  `analysis/doccat/src/selection.rs` and `analysis/doccat/tests/selection.rs`.
- **Probability calibration and provenance evidence (`turbo_doccat::calibrated`)**:
  `CalibratedCategorizer` wraps any `CalibratableCategorizer` (`PolarityAnnotator`
  or `DocumentCategorizerDL`) with `turbo_calibration::Calibration`
  (`Transform::Temperature`, `Transform::Isotonic`, or `Transform::Platt`),
  accompanied by `turbo_calibration::metrics::reliability` returning
  `Reliability` (`ece` and `brier`). Tested in
  `analysis/doccat/tests/calibrated.rs` and `analysis/doccat/tests/evidence.rs`.

