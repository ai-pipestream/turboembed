//! The real tagged-text fixture through training, packaging and both
//! inference entry points. Compatibility goldens remain in `fixtures`.

use std::sync::Arc;
use turbo_analysis::{Analyzer, Document, Key, Provenance, Span, Spans, View};
use turbo_analysis_layers::{Lemmas, PosTags, Strs, Tag, TagSet, Tokens};
use turbo_formats::native::LemmaReader;
use turbo_lemmatizer::{CASE_SENSITIVE_FACTORY, LemmaEncoding, LemmatizerModel, MaxentLemmatizer, TokenTag};
use turbo_maxent::ModelFile;
use turbo_train::{Key as TrainKey, ModelPackage, TrainingConfiguration, TrainingParameters};
use turbo_train_components::{LemmatizerFactory, Stamp, train_lemmatizer_model};

fn package(factory: &LemmatizerFactory) -> ModelPackage {
    let path = turbo_goldens::data_dir().join("inputs/tagged/lemmas.txt");
    let text = std::fs::read_to_string(path).unwrap();
    let samples: Vec<_> = LemmaReader::new(&text).unwrap().map(Result::unwrap).collect();
    assert_eq!(samples.len(), 20);
    let mut params = TrainingParameters::new();
    params.put_str(TrainKey::ALGORITHM, "MAXENT");
    params.put(TrainKey::ITERATIONS, 100);
    params.put(TrainKey::CUTOFF, 0);
    let model = train_lemmatizer_model(&samples, factory, &params, &mut TrainingConfiguration::default(), Stamp::at(0))
        .unwrap();
    model.package
}

#[test]
fn native_training_keeps_names_and_compatibility_keeps_lowercase() {
    let words = [
        TokenTag { token: "Mr.", tag: "NNP" },
        TokenTag { token: "Smith", tag: "NNP" },
        TokenTag { token: "arrived", tag: "VBD" },
        TokenTag { token: "in", tag: "IN" },
        TokenTag { token: "Paris", tag: "NNP" },
        TokenTag { token: "on", tag: "IN" },
        TokenTag { token: "Monday", tag: "NNP" },
        TokenTag { token: ".", tag: "." },
    ];
    for (factory, encoding, expected) in [
        (
            LemmatizerFactory::new("en"),
            LemmaEncoding::CaseSensitiveV1,
            ["Mr.", "Smith", "arrive", "in", "Paris", "on", "Monday", "."],
        ),
        (
            LemmatizerFactory::opennlp("en"),
            LemmaEncoding::OpenNlp,
            ["mr.", "smith", "arrive", "in", "paris", "on", "monday", "."],
        ),
    ] {
        let model = LemmatizerModel::open(ModelFile::open(package(&factory).to_bytes()).unwrap()).unwrap();
        assert_eq!(model.encoding(), encoding);
        if encoding == LemmaEncoding::CaseSensitiveV1 {
            assert_eq!(model.file().manifest().get("factory"), Some(CASE_SENSITIVE_FACTORY));
            assert!(model.model_ref().id.starts_with("turbo:"));
        }
        let lemmatizer = MaxentLemmatizer::new(Arc::new(model));
        let mut lemmas = Strs::default();
        let mut probs = Vec::new();
        assert!(lemmatizer.lemmatize(&words, &mut lemmas, &mut probs).unwrap());
        assert_eq!((0..lemmas.len()).map(|i| lemmas.get(i)).collect::<Vec<_>>(), expected);
        assert_eq!(probs.len(), words.len());

        let text = "Mr. Smith arrived in Paris on Monday .";
        let mut doc = Document::new(text);
        let view = doc.get::<View>().unwrap().id();
        let mut spans = Spans::new(view);
        for (a, b) in [(0, 3), (4, 9), (10, 17), (18, 20), (21, 26), (27, 29), (30, 36), (37, 38)] {
            spans.push(Span::new(a, b));
        }
        let tokens = Tokens { spans, sentence: None };
        let tags = PosTags {
            tagset: TagSet::new("penn", &["NNP", "VBD", "IN", "."]),
            tags: vec![Tag(0), Tag(0), Tag(1), Tag(2), Tag(0), Tag(2), Tag(0), Tag(3)],
            prob: None,
        };
        doc.add(&Key::<Tokens>::DEFAULT, tokens, Provenance::by("test")).unwrap();
        doc.add(&Key::<PosTags>::DEFAULT, tags, Provenance::by("test")).unwrap();
        let analyzer = Analyzer::builder()
            .given(Key::<Tokens>::DEFAULT)
            .given(Key::<PosTags>::DEFAULT)
            .add(lemmatizer)
            .build()
            .unwrap();
        assert_eq!(analyzer.run(std::slice::from_mut(&mut doc)), [Ok(())]);
        let layer = doc.get::<Lemmas>().unwrap();
        assert_eq!((0..layer.lemma.len()).map(|i| layer.lemma.get(i)).collect::<Vec<_>>(), expected);
    }
}

#[test]
fn an_unknown_native_version_is_refused() {
    let mut package = package(&LemmatizerFactory::new("en"));
    package.manifest_mut().set("factory", "turbo.lemmatizer.CaseSensitiveV2");
    let model = LemmatizerModel::open(ModelFile::open(package.to_bytes()).unwrap());
    assert!(matches!(model, Err(turbo_lemmatizer::ModelError::Factory(_))));
}
