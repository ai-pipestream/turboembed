//! The sentiment detector on models trained here: the two factories'
//! contexts, what the model file must name, and the annotator. Parity
//! with the JVM is `analysis/train-components/tests/sentiment.rs`.

use turbo_analysis::{Analyzer, Document, Key, Span, Spans, ViewId};
use turbo_analysis_layers::{Categories, Tokens};
use turbo_doccat::sentiment::{
    EMOJI_FACTORY, MODEL_ENTRY, SENTIMENT, Scratch, SentimentFactory, SentimentMe, SentimentModel, SentimentModelError,
    TOKEN_FACTORY,
};
use turbo_maxent::ModelFile;
use turbo_train::{Event, Key as Param, Manifest, ModelPackage, TrainingConfiguration, TrainingParameters, trainer};

const SAMPLES: &[(&str, &[&str])] = &[
    ("positive", &["good", "day", "\u{1F600}"]),
    ("positive", &["lovely", "\u{2764}\u{FE0F}"]),
    ("negative", &["bad", "day", "\u{1F622}"]),
    ("negative", &["awful", "\u{1F494}"]),
    ("neutral", &["the", "bus", "\u{1F697}"]),
];

/// A model file trained on [`SAMPLES`] with `factory`'s context, whose
/// manifest has `component` and, when given, `factory_class`.
fn model_file(factory: SentimentFactory, component: &str, factory_class: Option<&str>) -> Vec<u8> {
    let mut extra = Vec::new();
    let events: Vec<Event> =
        SAMPLES.iter().map(|(o, t)| Event::from_strs(*o, &factory.context(t, &mut extra))).collect();
    let mut params = TrainingParameters::new();
    params.put(Param::ALGORITHM, "MAXENT");
    params.put(Param::ITERATIONS, 50);
    params.put(Param::CUTOFF, 0);
    let (model, _) = trainer::train_events(&events, &params, &mut TrainingConfiguration::default()).unwrap();
    let mut manifest = Manifest::new(component, "en", 1_700_000_000_000);
    if let Some(class) = factory_class {
        manifest.set("factory", class);
    }
    let mut package = ModelPackage::new(manifest);
    package.add_entry(MODEL_ENTRY, model.write().unwrap());
    package.to_bytes()
}

fn open(bytes: Vec<u8>) -> Result<SentimentModel, SentimentModelError> {
    SentimentModel::open(ModelFile::open(bytes).unwrap())
}

fn detector(factory: SentimentFactory) -> SentimentMe {
    SentimentMe::new(open(model_file(factory, "SentimentME", Some(factory.class()))).unwrap())
}

#[test]
fn the_token_context_is_the_tokens_as_they_are() {
    let mut extra = Vec::new();
    let tokens = ["good", "\u{1F600}", "caf\u{E9}"];
    assert_eq!(SentimentFactory::Tokens.context(&tokens, &mut extra), tokens);
    assert!(extra.is_empty());
}

#[test]
fn the_emoji_context_appends_each_token_s_features_in_turn() {
    let mut extra = Vec::new();
    let tokens = ["\u{1F622}", "good\u{1F600}", "\u{1F1EF}\u{1F1F5}", "\u{2764}\u{FE0F}", "\u{1F600}\u{1F600}"];
    assert_eq!(
        SentimentFactory::Emoji.context(&tokens, &mut extra),
        [
            "\u{1F622}",
            "good\u{1F600}",
            "\u{1F1EF}\u{1F1F5}",
            "\u{2764}\u{FE0F}",
            "\u{1F600}\u{1F600}",
            "emojiSentiment=-2",
            "emojiType=FACE",
            "emojiCategory=SMILEYS_AND_EMOTION",
            "emojiType=FLAG",
            "emojiCategory=FLAGS",
            "emojiRegion=JP",
            "emojiSentiment=2",
            "emojiType=HEART",
            "emojiCategory=SMILEYS_AND_EMOTION",
        ]
    );
    // A token that starts with ASCII, or holds two emoji, has none.
    assert_eq!(SentimentFactory::Emoji.context(&["good\u{1F600}"], &mut extra), ["good\u{1F600}"]);
}

#[test]
fn the_factory_names_its_class() {
    assert_eq!(SentimentFactory::from_class(TOKEN_FACTORY), Some(SentimentFactory::Tokens));
    assert_eq!(SentimentFactory::from_class(EMOJI_FACTORY), Some(SentimentFactory::Emoji));
    assert_eq!(SentimentFactory::from_class("opennlp.tools.doccat.DoccatFactory"), None);
    assert_eq!(SentimentFactory::Emoji.class(), EMOJI_FACTORY);
}

#[test]
fn the_model_file_must_name_the_component_and_a_known_factory() {
    let tokens = SentimentFactory::Tokens;
    let wrong = open(model_file(tokens, "DocumentCategorizerME", Some(TOKEN_FACTORY))).unwrap_err();
    assert!(matches!(&wrong, SentimentModelError::Component(c) if c == "DocumentCategorizerME"), "{wrong}");
    let none = open(model_file(tokens, "SentimentME", None)).unwrap_err();
    assert!(matches!(none, SentimentModelError::NoFactory), "{none}");
    let other = open(model_file(tokens, "SentimentME", Some("x.Factory"))).unwrap_err();
    assert!(matches!(&other, SentimentModelError::Factory(c) if c == "x.Factory"), "{other}");
    assert!(other.to_string().contains(EMOJI_FACTORY));

    let model = open(model_file(tokens, "SentimentME", Some(EMOJI_FACTORY))).unwrap();
    assert_eq!(model.factory(), SentimentFactory::Emoji);
    assert_eq!(model.model_ref().id, "opennlp:SentimentME:en");
    assert_eq!(model.model_ref().revision, "3.0.0-SNAPSHOT");
    assert_eq!(model.model_ref().sha256.len(), 64);
}

#[test]
fn the_emoji_features_reach_tokens_the_model_never_saw() {
    // A heart the corpus has not seen still scores through
    // `emojiSentiment=2`, which the emoji model learned from other hearts.
    let mut s = Scratch::new();
    let emoji = detector(SentimentFactory::Emoji);
    let tokens = detector(SentimentFactory::Tokens);
    assert_eq!(emoji.predict(&["\u{1F499}"], &mut s), "positive");
    let p = tokens.probabilities(&["\u{1F499}"], &mut s).to_vec();
    assert!(p.iter().all(|&x| x == p[0]), "the token model knows nothing of {p:?}");
    assert_eq!(emoji.predict(&["\u{1F62D}"], &mut s), "negative");
}

#[test]
fn ties_keep_the_outcome_order() {
    let d = detector(SentimentFactory::Emoji);
    let mut s = Scratch::new();
    let c = d.categories(&["zebra"], &mut s);
    let labels: Vec<&str> = c.iter().map(|(l, _)| l).collect();
    let outcomes: Vec<&str> = (0..3).map(|i| d.model().label(i)).collect();
    assert_eq!(labels, outcomes);
    assert_eq!(d.best_sentiment(&[1.0 / 3.0; 3]), outcomes[0]);
    assert!(c.iter().all(|(_, p)| p == 1.0 / 3.0));
}

fn tokens(text: &str) -> Tokens {
    let mut spans = Spans::new(ViewId::SOURCE);
    let mut at = 0;
    for word in text.split(' ') {
        spans.push(Span::new(at as u32, (at + word.len()) as u32));
        at += word.len() + 1;
    }
    Tokens { spans, sentence: None }
}

#[test]
fn the_annotator_writes_the_sentiment_instance_best_first() {
    let d = detector(SentimentFactory::Emoji);
    let text = "a lovely day \u{1F600}";
    let mut doc = Document::new(text);
    doc.add(&Key::<Tokens>::DEFAULT, tokens(text), turbo_analysis::Provenance::by("test")).unwrap();
    let analyzer = Analyzer::builder().given(Key::<Tokens>::DEFAULT).step(d.clone().bound()).build().unwrap();
    let mut docs = [doc];
    for r in analyzer.run(&mut docs) {
        r.unwrap();
    }
    let c = docs[0].get_key(&SENTIMENT).unwrap();
    assert_eq!(c.best().unwrap().0, "positive");
    assert!(c.score.windows(2).all(|w| w[0] >= w[1]));
    let mut s = Scratch::new();
    let words: Vec<&str> = text.split(' ').collect();
    assert_eq!(c.score, d.categories(&words, &mut s).score);
    assert!(docs[0].get::<Categories>().is_none());
    let p = docs[0].provenance(SENTIMENT.id()).unwrap();
    assert_eq!(p.annotator, "turbo-doccat:sentiment");
    assert_eq!(p.model.as_ref(), Some(d.model().model_ref()));
}
