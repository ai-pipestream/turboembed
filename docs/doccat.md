# Document categorization

`turbo-doccat` (`analysis/doccat/`) ports the document categorizers of
OpenNLP 3.x (`DocumentCategorizerME` and `DoccatModel` in
`opennlp.tools.doccat.*`, `DocumentCategorizerSVM` in the
`opennlp-ml-libsvm` module, and `DocumentCategorizerDL` in
`opennlp.dl.doccat.*`), along with `SpanCategorizer<L>` and
`DlSpanCategorizer<S, L, C>` for sentence-, structure-, and chunk-level
categorization and `turbo_train_components::doccat` for training and
packaging maxent and perceptron `DoccatModel` archives.

The crate also holds OpenNLP's maxent sentiment detector, `SentimentME`,
which [sentiment.md](sentiment.md) describes, its own multi-label
categorizer, [`MultiLabelDoccatModel`](#multi-label-categorization-multilabeldoccatmodel),
and zero-shot natural-language-inference categorization,
[`ZeroShotCategorizer`](#zero-shot-categorization-zeroshotcategorizer).

It depends on `turbo-analysis`, `turbo-analysis-layers`, `turbo-jvm`
(Java's `Double.toString`, `Double.parseDouble`, `Integer.parseInt`,
`String.hashCode`, `Math.log` and `StrictMath.tanh`), `turbo-maxent`
(SHA-256, for the model reference, and the maxent/perceptron classifier),
`turbo-normalize` (the emoji features), `turbo-sequence` (`StringPattern`
for letter-only token checks) and `turbo-train` (event streams and
indexing), and on nothing outside the workspace.

## Maximum entropy and perceptron categorization (`DocumentCategorizerME`)

`DocumentCategorizerME` scores tokenized text with an OpenNLP `DoccatModel`
(`.bin` zip archive containing `doccat.model` and `manifest.properties`):

1. `DoccatContextGenerator` runs the configured `FeatureGenerator` chain over
   the tokens and optional `extra_information` map:
   - `BagOfWordsFeatureGenerator`: emits `bow=<token>` for every token (or,
     with `BagOfWordsFeatureGenerator::only_all_letter_tokens()`, only tokens
     where `StringPattern::recognize` reports `ALL_LETTERS`; by default it
     checks UTF-16 code units as the JVM does, with `PatternMode::CodePoints`
     available as an opt-in).
   - `NGramFeatureGenerator`: emits `ng=:<t_i>:...:<t_{i+n-1}>` for sliding
     token windows of length `min_gram..=max_gram` (default `2..=3`).
   - `EmojiFeatureGenerator`: emits `emoji=<token>` for emoji code points.
2. `categorize` (or `categorize_with_extra`) evaluates the underlying
   `MaxentModel` on the extracted context features and returns the outcome
   probability distribution.
3. `best_category` returns the first highest-scoring category, `get_index` /
   `get_category` map between category names and indices, `score_map` groups
   categories by probability into a `BTreeMap<OrderedFloat, BTreeSet<String>>`,
   and `sorted_score_map` orders categories by ascending score.
4. As an annotator (`opennlp:DocumentCategorizerME`), `DocumentCategorizerME`
   reads `opennlp:tokens` from the view and writes a single-item
   `turbo_analysis_layers::Categories` (`turbo:categories`) layer anchored to
   the document.
5. `SpanCategorizer<L>` (`turbo:SpanCategorizer`) wraps a
   `DocumentCategorizerME` and categorizes each span of a target layer `L:
   SpanCategorizable` (`Sentences`, `Structure`, or `Chunks`) using the tokens
   contained within that span, writing `Categories<L>` (`turbo:categories<L>`)
   aligned 1-to-1 with `L`.
6. `turbo_train_components::doccat::train` (and `train_from_samples` /
   `package`) trains a `DoccatModel` from `DocumentSample`s using GIS, QN
   (L-BFGS), Naive Bayes, or Perceptron and writes a byte-compatible OpenNLP
   `.bin` model archive.

`DoccatGolden.java` in `analysis/goldens-tool` trains 8 maxent/perceptron/naive
bayes models on the UCI SMS Spam Collection and records the features and
category probabilities on 12 test documents (`analysis/goldens/data/doccat`);
`analysis/doccat/tests/maxent_goldens.rs` and
`analysis/train-components/tests/doccat_goldens.rs` verify bit-for-bit feature,
model binary, and probability parity against the JVM.

## Multi-label categorization (`MultiLabelDoccatModel`)

`turbo_doccat::multilabel` (this crate's own design, not an OpenNLP port)
extends categorization beyond mutually exclusive labels: a document or
span scores every known label independently and receives zero, one, or
many of them.

- `MultiLabelDoccatModel` holds one linear head per label (weights and a
  bias, as `LinearHead`), trained one-vs-rest by `train_multilabel` with
  full-batch gradient descent over the `DoccatFactory` feature
  generators. The sigmoid turns each head's linear score into an
  independent score in `0.0..=1.0`; the scores do not sum to 1.0.
- Each label carries its own threshold in the open interval `(0.0, 1.0)`
  (default 0.5). `MultiLabelCategorizer` emits every label whose score
  reaches its threshold, optionally capped by `max_labels`, floored by
  `min_score` (a hard floor: a label under it never emits), and, where
  fewer than `min_labels` would emit, back-filled up to `min_labels`
  with the highest-scoring labels that pass `min_score`. When no label
  clears its threshold and `min_labels` is `None`, the prediction is
  empty: the annotators write the item with best category `Tag::NONE`,
  the clean abstention, never a forced argmax, and every score is still
  in the item. `MultiLabelSpanCategorizer<L>` does the same per
  sentence, structure block, or chunk.
- `ThresholdCalibrator` fits the thresholds on a held-out validation set:
  for each label it sweeps the validation scores as candidate cut points,
  keeps the threshold maximizing that label's F_beta (beta configurable,
  default 1.0), breaking ties toward the highest threshold, and can
  restrict candidates to a minimum-precision floor (when no candidate
  reaches the floor, the fallback is the candidate of the highest
  precision). A label with no positive in the validation set keeps the
  default threshold.
- `multilabel::metrics::evaluate` scores a run: micro-, macro-, and
  sample-averaged precision, recall, and F1, plus Exact Match (the
  fraction of documents whose predicted set equals the gold set), under
  the crate's division conventions (predicting nothing is precision 1.0,
  no gold positive is recall 1.0, both 0 is an F of 0.0).
- The model serializes as a `.bin` package (`package_multilabel_model` /
  `open_package` in the format `turbo_train::ModelPackage` writes, with
  the calibrated thresholds inside the model entry and a trailing
  CRC-32C). The entry is format version 2, which added `min_labels`;
  version 1 entries still open with `min_labels` `None`. Truncated or
  corrupted bytes fail with `MultiLabelError`, never panic.

`analysis/doccat/tests/multilabel.rs` covers the scoring, the abstention
into `Categories`, the calibration sweep (beta, floor, ties, zero
positives), and trains on an imbalanced fixture whose rare label's
optimal threshold is far from 0.5, proving the calibrated thresholds
strictly improve macro-F1 over a flat 0.5.
`analysis/doccat/tests/multilabel_extra.rs` covers the cardinality
bounds, the format's backward compatibility, and the evaluation metrics
against hand-computed values.

## SVM categorization (`SvmCategorizer`)

OpenNLP's SVM categorizer turns a document's tokens into a sparse vector and
asks LIBSVM for it. `turbo-doccat` follows it step by step:

1. The feature generators run over the tokens, in the model's order:
   `BagOfWordsFeatureGenerator` (`bow=` and the token) and
   `NGramFeatureGenerator` (`ng=` and the tokens of each n-gram, each
   after a `:`). `turbo_doccat::FeatureGenerator` is both.
2. Each feature in the vocabulary is counted; the others are dropped.
3. Each count is weighted by the model's `TermWeightingStrategy`: `BINARY`
   (1), `TERM_FREQUENCY` (the count), `TF_IDF` (the count times the
   feature's inverse document frequency) or `LOG_NORMALIZED_TF`
   (`1 + Math.log(count)`, with `turbo-jvm`'s port of `Math.log`).
4. With scaling on, each weight is scaled to the model's range from the
   feature's training minimum and maximum, clamped first, as
   `buildFeatureVector` does, and a feature whose minimum and maximum are
   equal gets the lower end.
5. A value that is zero by `Double.compare` is dropped, both after
   weighting and after scaling, so -0.0 stays. The features go to LIBSVM
   in index order, as OpenNLP's `TreeMap` gives them.
6. LIBSVM's `svm_predict_probability`: the kernel against every support
   vector, the decision value of each class pair, Platt's sigmoid on each
   (`sigmoid_predict`, with its 1e-7 floor) and the pairwise coupling
   (`multiclass_probability`). Each probability goes to the category whose
   index is the LIBSVM label, and a category without a label scores 0.

`svm::libsvm` is a port of `svm.java` 3.35, the Java LIBSVM that zlibsvm
3.0.0 wraps: `k_function` (the linear, polynomial, RBF and sigmoid kernels,
with LIBSVM's `powi` and its merge order for the RBF distance),
`svm_predict_values`, `svm_predict_probability`, the one-class density
marks and the regression Laplace parameter. The categorizer takes C-SVC and
nu-SVC models; the reader takes every kind but the precomputed kernel.

`SvmCategorizer` is the categorizer: `categorize`, `best_category`
(`getBestCategory`, the first highest score), `index` (`getIndex`, `None`
for -1), `category`, `num_categories` and `all_results` (`getAllResults`,
scores written as `Double.toString` writes them). As an annotator
(`turbo-doccat:svm`) it reads `opennlp:tokens` and writes
the document's `turbo:categories`: the best category and every
category's probability, which `Categories::ranked` lists best first,
ties in category order. The tokens are the strings the categorizer gets,
so the annotator gives what `categorize` gives for those tokens.

```rust
use std::path::Path;
use turbo_doccat::{Scratch, SvmCategorizer, SvmDoccatModel};

let model = SvmDoccatModel::open(Path::new("model.txt"), Path::new("sidecar.txt"))?;
let categorizer = SvmCategorizer::new(model)?;
let mut scratch = Scratch::new();
let scores = categorizer.categorize(&["heavy", "rain", "tonight"], &mut scratch);
let best = categorizer.best_category(scores);
```

The probabilities decide the category, not the pairwise vote. The two can
disagree, and do on some of the golden documents; the decision values and
the vote are there for comparison (`SvmDoccatModel::decision_values`).

A model's `ModelRef` is `opennlp:SvmDoccatModel:<language>` (`und` without
one), revision `1`, and the SHA-256 of the LIBSVM file's bytes followed by
the sidecar's.

### OpenNLP SVM behaviour worth knowing

- A model trained without probability estimates cannot categorize: zlibsvm
  hands LIBSVM empty `probA` and `probB` arrays, and OpenNLP throws
  `ArrayIndexOutOfBoundsException` on every document of a model with two
  or more categories. `SvmCategorizer::new` refuses such a model
  (`ModelError::NoProbabilityEstimates`).
- A document with no token in the vocabulary is an empty vector, and its
  decision values are the negated `rho` values, not zero.
- Feature selection (`FeatureSelectionStrategy`) only chooses the
  vocabulary at training; prediction never looks at it. The sidecar
  records it for the reader.

## The SVM model files

OpenNLP saves an `SvmDoccatModel` with Java serialization, which this port
does not read. The golden generator writes each model as two text files
instead, and `SvmDoccatModel::from_text` and `SvmDoccatModel::open` read
them:

- The LIBSVM model as `svm.svm_save_model` writes it. The reader is strict:
  each keyword must be exact and given once, `nr_class` comes before the
  records sized by it, counts must add up, each support vector's indexes
  ascend, and each kernel's parameters must be given. Numbers are read with
  Java's grammar, so the shortest decimals the Java writer prints and the
  `%.17g` of the C library both read back to the bits written.
- The sidecar, a line based format that holds the rest: the language, the
  weighting, the selection, the scaling range, the generators, the
  categories and, for each feature index, its inverse document frequency,
  its training minimum and maximum and its string. `svm::sidecar`
  documents the format; `Sidecar::write` writes it and reads back to the
  same model.

```text
svm-doccat 1
language	eng
weighting	TF_IDF
selection	NONE	-1
scaling	0.0	1.0
generator	bag-of-words
category	0	weather
category	1	cooking
category	2	sports
category	3	finance
feature	1	2.2617630984737906	0.0	2.2617630984737906	bow=for
feature	2	3.871201010907891	0.0	3.871201010907891	bow=up
```

(the start of the sidecar of the `topics-linear-tfidf-none` run)

Fields are separated by one tab, numbers are written as `Double.toString`
writes them, and a name is the last field, escaped as golden files escape
a string.

## SVM parity with the JVM

### The goldens

`SvmDoccatGolden.java` in `analysis/goldens-tool` trains 32 models with
`DocumentCategorizerSVM.train` on two corpora written for this project
(`inputs/svm-doccat-train`, 48 and 12 sentences): every term weighting
with every feature selection on the linear kernel, then the RBF,
polynomial and sigmoid kernels, nu-SVC, no scaling, the range -1 to 1,
n-grams and a higher cost. It exports each model as the two files above
and runs `categorize` on 12 documents (`inputs/svm-doccat`). Each golden
records the tokens, the feature vector, the decision values, the vote and
the categories with their probabilities.
`analysis/goldens-tool/README.md` ("SVM document categorizer goldens")
lists the runs and layers.

`analysis/doccat/tests/goldens.rs` reads each run's model from its
training golden and compares every document. The goldens are checked in
under `analysis/goldens/data/svm-doccat` and can be regenerated with:

```
cargo run -p turbo-goldens-tool -- generate
```

Across the 32 runs and 384 documents, 2,183 of 2,184 decision values match
the JVM's bits (every linear, polynomial and sigmoid decision value is
exact); the remaining RBF value is 1.8e-15 away.

### The wider check

A scratch program on the same JDK trained 114 models on generated corpora
of 2, 3 and 5 categories (every weighting with every selection on the
linear kernel; the RBF, polynomial and sigmoid kernels with every
weighting; nu-SVC with the linear and RBF kernels; no scaling, the range
-1 to 1, n-grams and a higher cost) and categorized 300 documents with
each, 34,200 in all, empty documents and unknown tokens among them:

| What | Same bits as the JVM | Largest difference |
|---|---|---|
| Feature vectors | 34,200 of 34,200 | none |
| Votes and best categories | 34,200 of 34,200 | none |
| Decision values, linear kernel | 100,800 of 100,800 | none |
| Decision values, polynomial kernel | 16,800 of 16,800 | none |
| Decision values, RBF kernel | 25,033 of 25,200 | 1.42e-14 |
| Decision values, sigmoid kernel | 16,759 of 16,800 (before `strict_math::tanh`) | 2.2e-16 |
| Probabilities | 113,294 of 114,000 | 4.94e-15 |

`getAllResults` gave the JVM's string wherever the probabilities had the
JVM's bits. The wider scratch check above was recorded when the sigmoid
kernel still called `f64::tanh`; with `k_function` calling
`turbo_jvm::strict_math::tanh` (verified bit-exact across all committed
sigmoid golden predictions in `topics-sigmoid-tfidf`), the sigmoid kernel
matches JDK 21's `Math.tanh` and `StrictMath.tanh` to the bit.

### Tolerances

- Exact: the feature vector, the vote, the order of the categories and,
  for the linear, polynomial and sigmoid kernels, the decision values.
  These are sums and products in LIBSVM's order, which the port keeps,
  and the sigmoid kernel calls `turbo_jvm::strict_math::tanh`.
- Decision values of the RBF kernel: within 1e-12. The RBF kernel calls
  `Math.exp`, and the port calls the platform's `exp`. On 2,000,000
  arguments, `f64::exp` (glibc) differed from HotSpot's `Math.exp` in the
  last place on 4,915. A last place error in one kernel value moves a
  decision value by about the coefficient times 2.2e-16 per support
  vector; the largest seen was 1.42e-14, so 1e-12 leaves room for other
  platforms' `exp` without hiding a wrong kernel, which moves values by
  far more.
- Probabilities: within 1e-9. Platt's sigmoid calls `Math.exp` for every
  kernel, and the pairwise coupling iterates on its results until a
  stopping test, so a last place difference can carry through a different
  number of iterations. The largest difference seen was 4.94e-15. 1e-9
  leaves that room and still fails on a wrong sigmoid or coupling step,
  which moves a probability by far more.

`Math.exp` on HotSpot is an intrinsic, not fdlibm: `StrictMath.exp`
(fdlibm, which `turbo-jvm` ports) differed from it on 192,922 of the same
arguments, so the fdlibm port would be further from the JVM than the
platform's `exp`. On JDK 21, HotSpot's `Math.tanh` delegates to
`StrictMath.tanh` and agreed with it on every tested argument, so
`turbo-jvm`'s port of fdlibm's `tanh` makes the sigmoid kernel exact.

## Why an SVM port and not linfa-svm

The SVM predictor is a port of LIBSVM's prediction code, a few hundred lines.
Running a model OpenNLP trained with the JVM's numbers needs three things:
a reader for LIBSVM's model file, LIBSVM's one-against-one vote and
pairwise probability coupling, and LIBSVM's order of `f64` operations.
General-purpose Rust SVM crates such as linfa-svm and smartcore train and
store their own models, and any of them would be a new dependency, which
the workspace does not allow. Bindings to the C LIBSVM would bring a C
build, and the C library's `exp` is the platform's too.

The port reads the file OpenNLP's LIBSVM writes, keeps LIBSVM's order of
operations, so the linear, polynomial and sigmoid kernels give the JVM's
bits, and depends on nothing new. It predicts only; SVM training stays on
the JVM.

## What the SVM categorizer does not do

- SVM training and feature selection: SVM models are trained by OpenNLP and
  exported (whereas maxent, perceptron, and Naive Bayes `DoccatModel`s are
  trained directly by `turbo_train_components::doccat`).
- Java serialized `SvmDoccatModel` files: only the exported text files.
- SVM feature generators other than `BagOfWordsFeatureGenerator()` and
  `NGramFeatureGenerator`; the letters only variant of the bag of words
  generator is package private in OpenNLP's SVM module, and the sidecar has no
  form for it.
- The precomputed kernel, whose input is kernel values rather than
  features.
- Bit for bit RBF decision values and probabilities, for the reasons under
  "Tolerances".

## Zero-shot categorization (`ZeroShotCategorizer`)

Zero-shot categorization needs no per-label training: the caller supplies
candidate labels at run time, and each label becomes the hypothesis of a
natural-language-inference pair whose premise is the text (template:
`"This example is {}."` by default, validated to hold exactly one `{}`
with text around it). The model sits behind the `NliScorer` trait, so
this crate owns the framing and the scoring modes without assuming a
sequence-pair runtime; a scorer declares its outcome names, its pair
token budget, and its own token counting.

Outcome names resolve case-insensitively at construction: either the
3-outcome head (entailment, neutral, contradiction) or the 2-outcome
head (entailment, not_entailment); anything else is refused, as are an
empty label list, a blank or duplicate label, and a hypothesis that
already fills the pair budget. Single-label mode takes a softmax over
the entailment logits across labels (scores sum to 1); multi-label
mode takes an independent two-way softmax over (entailment,
contradiction) per label. The neutral logit never enters either mode.

A premise longer than the pair budget less the longest hypothesis
and the scorer's pair special tokens (3 for BERT-style templates, 4
for RoBERTa-style) is windowed by words (measured with the scorer's
token counter, with an optional word overlap), every window
carrying the full hypothesis;
per-window logits combine by max, mean, or premise-length-weighted
mean before the mode's softmax runs once. `ZeroShotAnnotator` writes
the document's `Categories` and `ZeroShotSpanAnnotator` writes
`Categories<L>` per sentence, block, or chunk, with the scorer's model
as provenance and the template, labels, mode, aggregation, and overlap
in the annotator config hash. Blank items score zero everywhere rather
than failing the document.

## Transformer sequence classification (`DocumentCategorizerDL`)

`turbo_doccat::dl` ports OpenNLP's `opennlp.dl.doccat.DocumentCategorizerDL`,
`DocumentCategorizerConfig`, and `scoring.ClassificationScoringStrategy` /
`AverageClassificationScoringStrategy`:

1. `DocumentCategorizerConfig::from_json` (or `read_categories_from_json`)
   parses the `id2label` map from a Hugging Face `config.json` (or standalone
   `id2label` object) into non-negative `u32` indices `0..N` and category names,
   validating that indices are contiguous and names are non-empty.
2. `DlInferenceOptions` configures OpenNLP DL's text preprocessing and
   chunking:
   - `lower_case` (default `true`): folds text with `.to_lowercase()` before
     splitting, matching `InferenceOptions.isLowerCase()`.
   - `normalize_whitespace` and `normalize_dashes` (default `false`): collapse
     whitespace runs to single ASCII spaces and normalize Unicode dashes to `-`.
   - `document_split_size` (default `250`) and `split_overlap_size` (default
     `50`): sliding window step `document_split_size - split_overlap_size`,
     rejected when `document_split_size == 0` or `split_overlap_size >= document_split_size`.
   - `chunking_mode`: `DlChunkingMode::WhitespaceWords` (default, matching
     OpenNLP's `split("\\s+")` word windowing in `Inference.split`) or
     `DlChunkingMode::Subwords` (windowing directly over pre-tokenized
     `opennlp:subwords` IDs from the view).
3. `DocumentCategorizerDL<S, C>` runs its `DlSequenceScorer` (`S`) over the
   document's chunks, computes a numerically stable `f64` `softmax` over each
   chunk's raw `f32` class logits (refusing non-finite logits), and combines
   chunk probability distributions through `ClassificationScoringStrategy`
   (`AverageClassificationScoringStrategy` by default, which takes the elementwise
   mean across chunks). It exposes `categorize` (joining tokens with a space, as
   `DocumentCategorizerDL.categorize(String[])` does), `categorize_text`,
   `categorize_subwords`, `best_category`, `best_index`, `get_index`,
   `get_category`, `number_of_categories`, `score_map`, `sorted_score_map`, and
   `all_results`.
4. As an annotator (`opennlp:doccat-dl`), `DocumentCategorizerDL` reads the view
   (and optional `opennlp:subwords` when `DlChunkingMode::Subwords` is selected
   and preprocessing is disabled) and writes a single-item `Categories`
   (`turbo:categories`) layer anchored to the document. `DlSpanCategorizer<S, L, C>`
   (`opennlp:doccat-dl-spans`) categorizes each span of `L: SpanCategorizable`
   (`Chunks`, `Structure`, or `Sentences`) and writes `Categories<L>`.
5. `turbo_analysis_embed::BundleSequenceScorer` implements `DlSequenceScorer` on
   turboembed's native backends (CPU, Metal, CUDA, Vulkan, HIP) without ONNX
   Runtime: it opens the bundle's `Encoder` with `TURBO_POOLING_CLS` and
   `TURBO_NORMALIZE_NONE`, loads the `CLASSIFIER_SEQUENCE` head from `head.bin`
   (`turbo::head::Head::from_bundle`, applying the optional `pooler.dense` +
   `tanh` and linear `classifier`), and scores either text chunks (via the
   bundle's `Tokenizer`) or pre-tokenized subword windows (`Rows::from_ids`).
