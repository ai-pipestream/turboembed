//! Training the sentiment detector: the events of `SentimentEventStream`
//! and the model file `SentimentME.train` returns as a `SentimentModel`.
//!
//! `SentimentSampleStream` reads one sample per line, the sentiment and
//! then the tokens, all split by `WhitespaceTokenizer`, and refuses a line
//! with fewer than two tokens with the message `DocumentSampleStream`
//! uses; the samples are `turbo_formats::DocumentSample`s read by
//! `turbo_formats::native::DocumentReader`. `SentimentEventStream` makes
//! one event per sample: the sentiment as the outcome, and as the context
//! what the factory's context generator gives for the tokens
//! ([`turbo_doccat::SentimentFactory::context`]).
//!
//! `SentimentME.train` trains with the event trainer the parameters name,
//! putting nothing into them, and builds the model with the caller's
//! language and the trainer's manifest entries; `BaseModel` adds
//! `factory`, the factory's class name. Neither factory has artifacts or
//! manifest entries of its own.

use turbo_doccat::SentimentFactory;
use turbo_doccat::sentiment::{COMPONENT, MODEL_ENTRY};
use turbo_formats::DocumentSample;
use turbo_train::{
    Event, JavaString, ModelPackage, TrainedModel, TrainingConfiguration, TrainingParameters, TrainingReport, trainer,
};

use crate::error::Result;
use crate::package::{FACTORY_KEY, Stamp, add_artifacts};

/// `SentimentEventStream` over a factory's context generator.
#[derive(Debug)]
pub struct SentimentEventStream {
    factory: SentimentFactory,
    extra: Vec<String>,
}

impl SentimentEventStream {
    pub fn new(factory: SentimentFactory) -> SentimentEventStream {
        SentimentEventStream { factory, extra: Vec::new() }
    }

    /// The event of a sample: its sentiment as the outcome and
    /// `getContext(tokens)` as the predicates.
    pub fn event(&mut self, sentiment: &str, tokens: &[&str]) -> Event {
        self.factory.extra_features(tokens, &mut self.extra);
        let mut predicates = Vec::with_capacity(tokens.len() + self.extra.len());
        predicates.extend(tokens.iter().map(|t| JavaString::new(t)));
        predicates.extend(self.extra.iter().map(|f| JavaString::new(f)));
        Event::new(JavaString::new(sentiment), predicates)
    }

    /// `createEvents(sample)`: appends the sample's one event to `out`.
    pub fn events(&mut self, sample: &DocumentSample<'_>, out: &mut Vec<Event>) {
        let tokens: Vec<&str> = sample.tokens.iter().map(|t| t.text.as_ref()).collect();
        out.push(self.event(&sample.category.text, &tokens));
    }
}

/// The events of `SentimentEventStream` over `samples`.
pub fn sentiment_events<'a, 's>(
    samples: impl IntoIterator<Item = &'s DocumentSample<'a>>,
    factory: SentimentFactory,
) -> Vec<Event>
where
    'a: 's,
{
    let mut stream = SentimentEventStream::new(factory);
    let mut out = Vec::new();
    for sample in samples {
        stream.events(sample, &mut out);
    }
    out
}

/// A trained sentiment model: the classifier, what the trainer reports,
/// and the `.bin` file.
#[derive(Debug, Clone)]
pub struct TrainedSentimentModel {
    pub model: TrainedModel,
    pub report: TrainingReport,
    pub package: ModelPackage,
}

/// `new SentimentModel(language, model, manifestInfoEntries, factory)`
/// serialized: the manifest with the trainer's entries and `factory`, and
/// the `sentiment.model` entry holding `model_bytes`.
pub fn package_sentiment_model(
    factory: SentimentFactory,
    language: &str,
    report: &TrainingReport,
    model_bytes: Vec<u8>,
    stamp: Stamp<'_>,
) -> Result<ModelPackage> {
    let mut manifest = stamp.manifest(COMPONENT, language);
    manifest.set_training(report);
    manifest.set(FACTORY_KEY, factory.class());
    let mut package = ModelPackage::new(manifest);
    add_artifacts(&mut package, vec![(MODEL_ENTRY, model_bytes)])?;
    Ok(package)
}

/// `SentimentME.train(language, samples, params, factory)`: the events of
/// [`sentiment_events`], the event trainer the parameters name, and the
/// model file stamped with `stamp`.
pub fn train_sentiment_model<'a, 's>(
    language: &str,
    samples: impl IntoIterator<Item = &'s DocumentSample<'a>>,
    factory: SentimentFactory,
    params: &TrainingParameters,
    config: &mut TrainingConfiguration,
    stamp: Stamp<'_>,
) -> Result<TrainedSentimentModel>
where
    'a: 's,
{
    let events = sentiment_events(samples, factory);
    train_sentiment_model_from_events(language, &events, factory, params, config, stamp)
}

/// [`train_sentiment_model`] from events already made.
pub fn train_sentiment_model_from_events(
    language: &str,
    events: &[Event],
    factory: SentimentFactory,
    params: &TrainingParameters,
    config: &mut TrainingConfiguration,
    stamp: Stamp<'_>,
) -> Result<TrainedSentimentModel> {
    let (model, report) = trainer::train_events(events, params, config)?;
    let bytes = model.write()?;
    let package = package_sentiment_model(factory, language, &report, bytes, stamp)?;
    Ok(TrainedSentimentModel { model, report, package })
}

#[cfg(test)]
mod tests {
    use super::*;
    use turbo_formats::native::DocumentReader;
    use turbo_unicode::WhitespaceMode;

    fn samples(text: &str) -> Vec<DocumentSample<'_>> {
        DocumentReader::new(text, WhitespaceMode::Unicode).unwrap().map(|s| s.unwrap()).collect()
    }

    fn strings(e: &Event) -> Vec<String> {
        e.context().iter().map(JavaString::to_string_lossy).collect()
    }

    #[test]
    fn the_context_is_the_tokens_then_their_emoji_features() {
        let s = samples("positive good \u{1F600} day\nnegative bad \u{1F1EC}\u{1F1E7}\n");
        let plain = sentiment_events(&s, SentimentFactory::Tokens);
        assert_eq!(*plain[0].outcome(), "positive");
        assert_eq!(strings(&plain[0]), ["good", "\u{1F600}", "day"]);
        let emoji = sentiment_events(&s, SentimentFactory::Emoji);
        assert_eq!(
            strings(&emoji[0]),
            ["good", "\u{1F600}", "day", "emojiSentiment=2", "emojiType=FACE", "emojiCategory=SMILEYS_AND_EMOTION"]
        );
        assert_eq!(
            strings(&emoji[1]),
            ["bad", "\u{1F1EC}\u{1F1E7}", "emojiType=FLAG", "emojiCategory=FLAGS", "emojiRegion=GB"]
        );
    }

    #[test]
    fn the_model_file_names_the_factory() {
        let report = TrainingReport { event_hash: Some(1), trainer_type: turbo_train::TrainerType::Event };
        let package = package_sentiment_model(SentimentFactory::Emoji, "en", &report, vec![1], Stamp::at(0)).unwrap();
        assert_eq!(package.entry_names(), ["manifest.properties", "sentiment.model"]);
        assert_eq!(package.manifest().get("factory"), Some("opennlp.tools.sentiment.EmojiSentimentFactory"));
        assert_eq!(package.manifest().component_name(), "SentimentME");
        assert_eq!(package.manifest().language(), "en");
    }
}
