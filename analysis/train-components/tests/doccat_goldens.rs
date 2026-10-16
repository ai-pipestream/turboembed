//! `DocumentCategorizerME` against the JVM's goldens under
//! `analysis/goldens/data/doccat/<run>/`, the goldens of `DoccatGolden.java`
//! in `analysis/goldens-tool`.
//!
//! Each run trains a model on the SMS Spam Collection followed by
//! `inputs/doccat-train/emoji.txt`. The SMS corpus is CC BY 4.0 and never
//! checked in: `turbo-goldens-tool fetch-models` downloads it into
//! `corpora/` of the models directory (`TURBO_GOLDENS_MODELS`), and the
//! test skips without it unless `TURBO_TEST_REQUIRE_OPENNLP_MODELS=1`, as
//! in CI. Its SHA-512 must be the one the goldens were made from.
//!
//! For each run, the golden of the corpus (`doccat-train/emoji`) gives the
//! samples per category, the SHA-256 and length of the `doccat.model`
//! entry, the manifest without `Timestamp` and the zip entries. The model
//! is trained here with the run's parameters and must match all of them:
//! the classifier entry byte for byte. The model file written here is then
//! opened as a user of the file opens it, and for each of the 12 document
//! goldens (`doccat/<id>`) the context of the golden's tokens must be the
//! JVM's and `turbo:categories` the JVM's categories, in the JVM's order,
//! with the JVM's probabilities bit for bit. The context alone is also
//! checked without the corpus, in `turbo-doccat`'s `tests/maxent_goldens.rs`.

mod common;

use std::path::Path;

use common::{config, golden_files, java_string, layer};
use turbo_analysis::{Analyzer, Document, Key};
use turbo_analysis_layers::{Categories, Tokens};
use turbo_doccat::{
    BagOfWordsFeatureGenerator, BuiltinFeatureGenerator, DoccatFactory, DoccatModel, DocumentCategorizerME,
    EmojiFeatureGenerator, MODEL_ENTRY, NGramFeatureGenerator,
};
use turbo_formats::DocumentSample;
use turbo_formats::native::DocumentReader;
use turbo_goldens::{
    ColumnSpec, ColumnType, Golden, GoldenAnchor, Shape, Tolerance, data_dir, known_shapes, load_layer, matches_golden,
    published_model,
};
use turbo_maxent::{ModelFile, Properties};
use turbo_train::package::MANIFEST_ENTRY;
use turbo_train::{Key as Param, TrainingConfiguration, TrainingParameters};
use turbo_train_components::Stamp;
use turbo_train_components::doccat::train_doccat;
use turbo_unicode::WhitespaceMode;

/// The command that writes the goldens.
const GENERATE: &str = "cargo run -p turbo-goldens-tool -- generate";

/// The downloaded corpus, in the models directory.
const CORPUS: &str = "corpora/sms-spam-collection.tsv";

const LAYERS: &[(&str, &[ColumnSpec])] = &[
    (
        "turbo:doccat-samples",
        &[ColumnSpec::new("category", ColumnType::Str), ColumnSpec::new("samples", ColumnType::U32)],
    ),
    ("turbo:doccat-model", &[ColumnSpec::new("sha256", ColumnType::Str), ColumnSpec::new("length", ColumnType::U32)]),
    ("turbo:doccat-features", &[ColumnSpec::new("feature", ColumnType::Str)]),
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

/// The generators of every run, as `DoccatGolden.generators()` builds them.
fn factory() -> DoccatFactory {
    DoccatFactory::with_feature_generators(vec![
        BuiltinFeatureGenerator::BagOfWords(BagOfWordsFeatureGenerator::new()),
        BuiltinFeatureGenerator::NGram(NGramFeatureGenerator::default()),
        BuiltinFeatureGenerator::Emoji(EmojiFeatureGenerator::new()),
    ])
}

/// The parameters the run called `DocumentCategorizerME.train` with; the
/// config records name no others.
fn parameters(at: &str, g: &Golden) -> TrainingParameters {
    let mut p = TrainingParameters::new();
    p.put(Param::ALGORITHM, config_of(at, g, "Algorithm"));
    p.put(Param::ITERATIONS, config_of(at, g, "Iterations").parse::<i32>().unwrap());
    p.put(Param::CUTOFF, config_of(at, g, "Cutoff").parse::<i32>().unwrap());
    for c in &g.header.config {
        let known = ["Algorithm", "Iterations", "Cutoff", "run", "corpus", "language", "generators"];
        assert!(known.contains(&c.key.as_str()), "{at}: config {} is not read here", c.key);
    }
    p
}

/// The SMS corpus, checked against the SHA-512 in the golden's `model`
/// record, or `None` when it is not downloaded and not required.
fn corpus(g: &Golden) -> Option<String> {
    let path = published_model(CORPUS)?;
    let bytes = std::fs::read(&path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    let record = g.header.models.iter().find(|m| CORPUS.ends_with(&m.file)).expect("the corpus's model record");
    assert!(turbo_goldens::sha512(&bytes) == record.sha512, "{}: not the corpus of the goldens", path.display());
    Some(String::from_utf8(bytes).expect("the corpus is UTF-8"))
}

/// The samples of `text`, one per line, as `DocumentSampleStream` reads
/// them over `PlainTextByLineStream`.
fn samples<'a>(at: &str, text: &'a str) -> Vec<DocumentSample<'a>> {
    DocumentReader::new(text, WhitespaceMode::Unicode)
        .unwrap()
        .map(|s| s.unwrap_or_else(|e| panic!("{at}: {e}")))
        .collect()
}

/// Counts over every run.
#[derive(Debug, Default)]
struct Totals {
    runs: usize,
    models: usize,
    documents: usize,
    features: usize,
    probabilities: usize,
}

/// Trains the run's model, checks it against the corpus golden, and
/// returns the categorizer over the model file written here.
fn train(at: &str, g: &Golden, sms: &str, totals: &mut Totals) -> DocumentCategorizerME {
    let emoji = g.input_text(data_dir()).unwrap_or_else(|e| panic!("{at}: {e}"));
    let mut all = samples(at, sms);
    all.extend(samples(at, &emoji));

    // The categories in order of first appearance, with their counts.
    let mut counts: Vec<(&str, u32)> = Vec::new();
    for s in &all {
        match counts.iter_mut().find(|(c, _)| *c == s.category.text) {
            Some((_, n)) => *n += 1,
            None => counts.push((&s.category.text, 1)),
        }
    }
    let l = layer(g, "turbo:doccat-samples");
    let (categories, numbers) = (l.strs("category").unwrap(), l.u32s("samples").unwrap());
    let expected: Vec<(&str, u32)> = (0..l.len()).map(|i| (categories.get(i), numbers[i])).collect();
    assert_eq!(counts, expected, "{at}: samples per category");

    let factory = factory();
    let language = config_of(at, g, "language");
    let trained = train_doccat(
        language,
        &all,
        &parameters(at, g),
        &factory,
        &mut TrainingConfiguration::default(),
        Stamp::at(1_700_000_000_000),
    )
    .unwrap_or_else(|e| panic!("{at}: {e}"));
    let package = &trained.package;
    let (_, model) = package.entries().find(|(n, _)| *n == MODEL_ENTRY).unwrap();
    let written = layer(g, "turbo:doccat-model");
    assert_eq!(model.len(), written.u32s("length").unwrap()[0] as usize, "{at}: model entry length");
    assert_eq!(turbo_maxent::sha256(model).to_hex(), written.strs("sha256").unwrap().get(0), "{at}: model entry");
    totals.models += 1;

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

    // Read back from the bytes with the generators the manifest names, as
    // the JVM reads the model it serialized.
    let file = ModelFile::open(package.to_bytes()).unwrap_or_else(|e| panic!("{at}: {e}"));
    let model = DoccatModel::open(file).unwrap_or_else(|e| panic!("{at}: {e}"));
    let generators = config_of(at, g, "generators").split(',').collect::<Vec<_>>();
    let classes = model.feature_generators().iter().map(|f| f.class_name()).collect::<Vec<_>>();
    assert_eq!(classes, generators, "{at}: generators");
    DocumentCategorizerME::new(model)
}

fn check_document(at: &str, g: &Golden, categorizer: &DocumentCategorizerME, totals: &mut Totals) {
    let text = g.input_text(data_dir()).unwrap_or_else(|e| panic!("{at}: {e}"));
    let mut doc = Document::new(text.as_str());
    load_layer(&mut doc, g, &Key::<Tokens>::DEFAULT).unwrap_or_else(|e| panic!("{at}: {e}"));
    let tokens = doc.get::<Tokens>().unwrap();
    let words: Vec<&str> = tokens.spans.iter().map(|(_, s)| &text[s.range()]).collect();

    let context = categorizer.context_generator().context(&words, &Default::default());
    let golden = layer(g, "turbo:doccat-features").strs("feature").unwrap();
    let golden: Vec<String> = golden.iter().map(|f| java_string(f).to_string_lossy()).collect();
    assert_eq!(context, golden, "{at}: context");
    totals.features += context.len();

    let analyzer = Analyzer::builder()
        .given(Key::<Tokens>::DEFAULT)
        .add(categorizer.clone())
        .build()
        .unwrap_or_else(|e| panic!("{at}: {e}"));
    let mut docs = [doc];
    for r in analyzer.run(&mut docs) {
        r.unwrap_or_else(|e| panic!("{at}: {e}"));
    }
    if let Err(m) = matches_golden::<Categories>(&docs[0], g, Tolerance::exact()) {
        panic!("{at}: {m}");
    }
    let ours = docs[0].get::<Categories>().unwrap();
    let theirs = layer(g, "turbo:categories").f64s("score").unwrap();
    let ours: Vec<f64> = ours.ranked().into_iter().map(|(_, p)| p).collect();
    let bits = |xs: &[f64]| xs.iter().map(|x| x.to_bits()).collect::<Vec<_>>();
    assert_eq!(bits(&ours), bits(theirs), "{at}: probabilities bit for bit");
    totals.documents += 1;
    totals.probabilities += ours.len();
}

fn check_run(run: &Path, shapes: &[Shape], sms: &str, totals: &mut Totals) {
    let name = run.file_name().unwrap().to_string_lossy().into_owned();
    let train_goldens = golden_files(&run.join("doccat-train"));
    assert_eq!(train_goldens.len(), 1, "{name}: one training corpus golden");
    let g = read(&train_goldens[0], shapes);
    assert!(g.header.error.is_none(), "{name}");
    assert_eq!(config_of(&name, &g, "run"), name);
    let categorizer = train(&name, &g, sms, totals);

    let documents = golden_files(&run.join("doccat"));
    assert_eq!(documents.len(), 12, "{name}: document goldens");
    for path in documents {
        let at = format!("{name}/{}", path.file_stem().unwrap().to_string_lossy());
        let g = read(&path, shapes);
        assert!(g.header.error.is_none(), "{at}");
        assert_eq!(config_of(&at, &g, "run"), name, "{at}");
        check_document(&at, &g, &categorizer, totals);
    }
    totals.runs += 1;
}

#[test]
fn doccat_training_and_categorizing_are_the_jvms() {
    let root = data_dir().join("doccat");
    if !root.is_dir() {
        eprintln!("skipped: {} is absent; write the goldens with `{GENERATE}`", root.display());
        return;
    }
    let shapes = shapes();
    let mut runs: Vec<_> = std::fs::read_dir(&root).unwrap().map(|e| e.unwrap().path()).collect();
    runs.sort();
    let first = golden_files(&runs[0].join("doccat-train"));
    let Some(sms) = corpus(&read(&first[0], &shapes)) else {
        eprintln!(
            "skipped: {CORPUS} is not downloaded; fetch it with `cargo run -p turbo-goldens-tool -- fetch-models <dir>` and set TURBO_GOLDENS_MODELS to <dir>"
        );
        return;
    };
    let mut totals = Totals::default();
    for run in &runs {
        check_run(run, &shapes, &sms, &mut totals);
    }
    eprintln!("{totals:?}");
    assert_eq!(totals.runs, 4, "{}: runs", root.display());
    assert_eq!(totals.models, totals.runs);
}
