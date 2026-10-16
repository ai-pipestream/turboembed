//! OpenNLP's sentiment detector, `opennlp.tools.sentiment`: `SentimentME`
//! at prediction, the `SentimentModel` file it runs, and the context
//! generators of `SentimentFactory` and `EmojiSentimentFactory`.
//!
//! `SentimentME` is a maxent document classifier. Its context is the
//! document's tokens as they are, with no prefix; the emoji factory's
//! context generator appends, for each token in turn, the features
//! OpenNLP's `EmojiAnnotator.collectFeatures` gives it
//! (`turbo_normalize::emoji::features`): `emojiSentiment=`,
//! `emojiType=`, `emojiCategory=` and `emojiRegion=`. The sentiment
//! scores behind `emojiSentiment=` are OpenNLP's own, from its
//! project-authored `emoji-annotations.txt`.
//!
//! [`SentimentMe`] is an annotator that reads `opennlp:tokens` and writes
//! `turbo:categories`: every outcome of the model, best first, with its
//! probability. The layer is a document categorizer's, so a pipeline that
//! runs both a topic categorizer and this one binds this one's output to
//! the [`SENTIMENT`] instance ([`SentimentMe::bound`]).
//!
//! Training is in `turbo-train-components` (`sentiment`), which builds
//! its events with [`SentimentFactory::context`].

mod detector;
pub mod lexicon;
mod model;

pub use detector::{SENTIMENT, Scratch, SentimentMe};
pub use lexicon::{
    B_DECR, B_INCR, C_INCR, COMPOUND_THRESHOLD, DEFAULT_ALPHA, LexiconSentiment, N_SCALAR, NegationMode,
    SENTENCE_SENTIMENT, SentenceSentiment, VaderLexicon, VaderScore,
};
pub use model::{
    COMPONENT, EMOJI_FACTORY, MODEL_ENTRY, SentimentFactory, SentimentModel, SentimentModelError, TOKEN_FACTORY,
};
