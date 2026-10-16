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
`opennlp:tokens` and writes `turbo:categories`: every outcome with its
probability, best first. That is the layer a document categorizer writes,
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
