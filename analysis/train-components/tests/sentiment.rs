//! `SentimentME` against the JVM's goldens under
//! `analysis/goldens/data/sentiment/<run>/`, the goldens of
//! `SentimentGolden.java` in `analysis/goldens-tool`.
//!
//! For each run, the golden of the corpus (`sentiment-train/feedback`)
//! gives the events of `SentimentEventStream` (outcome and predicates, in
//! order), the SHA-256 and length of the `sentiment.model` entry, the
//! manifest without `Timestamp` and the zip entries. The model is trained
//! here with the run's factory and parameters and must match all of them:
//! the classifier entry byte for byte. The model file written here is then
//! opened as `turbo-doccat` opens one, and for every document golden
//! (`sentiment/<id>`) the context of the golden's tokens must be the JVM's
//! and the `sentiment` instance of `turbo:categories` the JVM's outcomes
//! and probabilities, bit for bit.

mod common;

use std::path::Path;

use common::{config, describe_difference, golden_files, java_string, layer};
use turbo_analysis::{Analyzer, Document, Key};
use turbo_analysis_layers::{Categories, Tokens};
use turbo_doccat::sentiment::{MODEL_ENTRY, SENTIMENT, SentimentFactory, SentimentMe, SentimentModel};
use turbo_formats::DocumentSample;
use turbo_formats::native::DocumentReader;
use turbo_goldens::{
    ColumnSpec, ColumnType, Golden, GoldenAnchor, Shape, Tolerance, data_dir, known_shapes, load_layer,
    matches_golden_key,
};
use turbo_maxent::{ModelFile, Properties};
use turbo_train::package::MANIFEST_ENTRY;
use turbo_train::{Event, Key as Param, TrainingConfiguration, TrainingParameters};
use turbo_train_components::Stamp;
use turbo_train_components::sentiment::{sentiment_events, train_sentiment_model};
use turbo_unicode::WhitespaceMode;

/// The command that writes the goldens.
const GENERATE: &str = "cargo run -p turbo-goldens-tool -- generate";

const LAYERS: &[(&str, &[ColumnSpec])] = &[
    (
        "turbo:sentiment-events",
        &[ColumnSpec::new("outcome", ColumnType::Str), ColumnSpec::new("predicates", ColumnType::U32)],
    ),
    (
        "turbo:sentiment-predicates",
        &[ColumnSpec::new("event", ColumnType::U32), ColumnSpec::new("predicate", ColumnType::Str)],
    ),
    (
        "turbo:sentiment-model",
        &[ColumnSpec::new("sha256", ColumnType::Str), ColumnSpec::new("length", ColumnType::U32)],
    ),
    ("turbo:sentiment-context", &[ColumnSpec::new("feature", ColumnType::Str)]),
];

fn shapes() -> Vec<Shape> {
    let mut shapes = known_shapes();
    shapes.extend(common::shapes());
    shapes.extend(LAYERS.iter().map(|(id, columns)| Shape {
        id: id.to_string(),
        anchor: GoldenAnchor::Document,
        columns,
    }));
    shapes
}

fn read(path: &Path, shapes: &[Shape]) -> Golden {
    Golden::read_with(path, shapes).unwrap_or_else(|e| panic!("{}: {e}", path.display()))
}

fn config_of<'g>(at: &str, g: &'g Golden, key: &str) -> &'g str {
    config(g, key).unwrap_or_else(|| panic!("{at}: no config {key}"))
}

/// The parameters the run called `SentimentME.train` with; the config
/// records name no others.
fn parameters(at: &str, g: &Golden) -> TrainingParameters {
    let mut p = TrainingParameters::new();
    p.put(Param::ALGORITHM, config_of(at, g, "Algorithm"));
    p.put(Param::ITERATIONS, config_of(at, g, "Iterations").parse::<i32>().unwrap());
    p.put(Param::CUTOFF, config_of(at, g, "Cutoff").parse::<i32>().unwrap());
    for c in &g.header.config {
        let known = ["Algorithm", "Iterations", "Cutoff", "run", "corpus", "language", "factory"];
        assert!(known.contains(&c.key.as_str()), "{at}: config {} is not read here", c.key);
    }
    p
}

/// The events the golden lists, read back through [`java_string`].
fn golden_events(g: &Golden) -> Vec<Event> {
    let events = layer(g, "turbo:sentiment-events");
    let (outcomes, sizes) = (events.strs("outcome").unwrap(), events.u32s("predicates").unwrap());
    let predicates = layer(g, "turbo:sentiment-predicates");
    let (owner, text) = (predicates.u32s("event").unwrap(), predicates.strs("predicate").unwrap());
    let mut next = 0;
    (0..events.len())
        .map(|i| {
            let n = sizes[i] as usize;
            let context = (next..next + n)
                .map(|k| {
                    assert_eq!(owner[k] as usize, i, "predicate {k} belongs to event {i}");
                    java_string(text.get(k))
                })
                .collect();
            next += n;
            Event::new(java_string(outcomes.get(i)), context)
        })
        .collect()
}

/// Counts over every run.
#[derive(Debug, Default)]
struct Totals {
    runs: usize,
    events: usize,
    emoji_features: usize,
    documents: usize,
    probabilities: usize,
}

/// Trains the run's model, checks it against the corpus golden, and
/// returns the detector over the model file written here.
fn train(at: &str, g: &Golden, factory: SentimentFactory, totals: &mut Totals) -> SentimentMe {
    let text = g.input_text(data_dir()).unwrap_or_else(|e| panic!("{at}: {e}"));
    let samples: Vec<DocumentSample<'_>> = DocumentReader::new(&text, WhitespaceMode::Unicode)
        .unwrap()
        .map(|s| s.unwrap_or_else(|e| panic!("{at}: {e}")))
        .collect();
    let events = sentiment_events(&samples, factory);
    let expected = golden_events(g);
    assert!(events == expected, "{at}: {}", describe_difference(&expected, &events));
    totals.events += events.len();
    totals.emoji_features +=
        events.iter().flat_map(|e| e.context()).filter(|p| p.to_string_lossy().starts_with("emoji")).count();

    let params = parameters(at, g);
    let language = config_of(at, g, "language");
    let trained = train_sentiment_model(
        language,
        &samples,
        factory,
        &params,
        &mut TrainingConfiguration::default(),
        Stamp::at(1_700_000_000_000),
    )
    .unwrap_or_else(|e| panic!("{at}: {e}"));
    let package = &trained.package;
    let (_, model) = package.entries().find(|(n, _)| *n == MODEL_ENTRY).unwrap();
    let written = layer(g, "turbo:sentiment-model");
    assert_eq!(turbo_maxent::sha256(model).to_hex(), written.strs("sha256").unwrap().get(0), "{at}: model entry");
    assert_eq!(model.len(), written.u32s("length").unwrap()[0] as usize, "{at}: model entry");

    let l = layer(g, "turbo:manifest");
    let (keys, values) = (l.strs("key").unwrap(), l.strs("value").unwrap());
    let expected: Vec<(String, String)> =
        (0..l.len()).map(|i| (keys.get(i).to_owned(), values.get(i).to_owned())).collect();
    let text = package.manifest().to_properties();
    let properties = Properties::load(text.as_bytes()).unwrap();
    let mut ours: Vec<(String, String)> =
        properties.iter().filter(|(k, _)| *k != "Timestamp").map(|(k, v)| (k.to_owned(), v.to_owned())).collect();
    ours.sort();
    assert_eq!(expected, ours, "{at}: manifest");

    // The entries in order; the manifest's length too, both timestamps
    // having thirteen digits and both dated in UTC.
    let l = layer(g, "turbo:package-entries");
    let (names, lengths) = (l.strs("name").unwrap(), l.u32s("length").unwrap());
    assert_eq!((0..l.len()).map(|i| names.get(i)).collect::<Vec<_>>(), package.entry_names(), "{at}: entries");
    for (i, name) in package.entry_names().into_iter().enumerate() {
        let len = match name {
            MANIFEST_ENTRY => text.len(),
            _ => package.entries().find(|(n, _)| *n == name).unwrap().1.len(),
        };
        assert_eq!(lengths[i] as usize, len, "{at}: the length of {name}");
    }

    let file = ModelFile::open(package.to_bytes()).unwrap_or_else(|e| panic!("{at}: {e}"));
    let model = SentimentModel::open(file).unwrap_or_else(|e| panic!("{at}: {e}"));
    assert_eq!(model.factory(), factory, "{at}");
    SentimentMe::new(model)
}

fn check_document(at: &str, g: &Golden, detector: &SentimentMe, totals: &mut Totals) {
    let text = g.input_text(data_dir()).unwrap_or_else(|e| panic!("{at}: {e}"));
    let mut doc = Document::new(text.as_str());
    load_layer(&mut doc, g, &Key::<Tokens>::DEFAULT).unwrap_or_else(|e| panic!("{at}: {e}"));
    let tokens = doc.get::<Tokens>().unwrap();
    let words: Vec<&str> = tokens.spans.iter().map(|(_, s)| &text[s.range()]).collect();

    let mut extra = Vec::new();
    let context = detector.model().factory().context(&words, &mut extra);
    let golden = layer(g, "turbo:sentiment-context").strs("feature").unwrap();
    let golden: Vec<String> = golden.iter().map(|f| java_string(f).to_string_lossy()).collect();
    assert_eq!(context, golden, "{at}: context");

    let analyzer = Analyzer::builder()
        .given(Key::<Tokens>::DEFAULT)
        .step(detector.clone().bound())
        .build()
        .unwrap_or_else(|e| panic!("{at}: {e}"));
    let mut docs = [doc];
    for r in analyzer.run(&mut docs) {
        r.unwrap_or_else(|e| panic!("{at}: {e}"));
    }
    if let Err(m) = matches_golden_key(&docs[0], &SENTIMENT, g, Tolerance::exact()) {
        panic!("{at}: {m}");
    }
    let ours = docs[0].get_key(&SENTIMENT).unwrap();
    let theirs = layer(g, "turbo:categories").f64s("score").unwrap();
    let bits = |xs: &[f64]| xs.iter().map(|x| x.to_bits()).collect::<Vec<_>>();
    assert_eq!(bits(&ours.score), bits(theirs), "{at}: probabilities bit for bit");
    assert!(docs[0].get::<Categories>().is_none(), "{at}: the default instance is left for a topic categorizer");
    totals.documents += 1;
    totals.probabilities += ours.score.len();
}

fn check_run(run: &Path, shapes: &[Shape], totals: &mut Totals) {
    let name = run.file_name().unwrap().to_string_lossy().into_owned();
    let train_goldens = golden_files(&run.join("sentiment-train"));
    assert_eq!(train_goldens.len(), 1, "{name}: one training corpus golden");
    let g = read(&train_goldens[0], shapes);
    assert!(g.header.error.is_none(), "{name}");
    assert_eq!(config_of(&name, &g, "run"), name);
    let class = config_of(&name, &g, "factory");
    let factory = SentimentFactory::from_class(class).unwrap_or_else(|| panic!("{name}: factory {class}"));
    let detector = train(&name, &g, factory, totals);

    let documents = golden_files(&run.join("sentiment"));
    assert!(!documents.is_empty(), "{name}: no document goldens");
    for path in documents {
        let at = format!("{name}/{}", path.file_stem().unwrap().to_string_lossy());
        let g = read(&path, shapes);
        assert!(g.header.error.is_none(), "{at}");
        assert_eq!(config_of(&at, &g, "factory"), class, "{at}");
        check_document(&at, &g, &detector, totals);
    }
    totals.runs += 1;
}

#[test]
fn sentiment_training_and_scoring_are_the_jvms() {
    let root = data_dir().join("sentiment");
    if !root.is_dir() {
        eprintln!("skipped: {} is absent; write the goldens with `{GENERATE}`", root.display());
        return;
    }
    let shapes = shapes();
    let mut runs: Vec<_> = std::fs::read_dir(&root).unwrap().map(|e| e.unwrap().path()).collect();
    runs.sort();
    let mut totals = Totals::default();
    for run in &runs {
        check_run(run, &shapes, &mut totals);
    }
    eprintln!("{totals:?}");
    assert!(totals.runs > 0, "{}: no runs", root.display());
    assert!(totals.emoji_features > 0, "{totals:?}");
}
