//! The context of `DocumentCategorizerME` against the JVM's goldens under
//! `analysis/goldens/data/doccat/<run>/doccat`, the goldens of
//! `DoccatGolden.java` in `analysis/goldens-tool`.
//!
//! The goldens do not hold the JVM's model file, only the SHA-256 of its
//! classifier, and the model is trained on the SMS Spam Collection, which is
//! downloaded and never checked in. So this test needs no corpus and checks
//! what needs no model: for each of the 12 documents of every run, the
//! features `BagOfWordsFeatureGenerator`, `NGramFeatureGenerator` and
//! `EmojiFeatureGenerator` give for the golden's tokens, exactly, and the
//! generators the run names. The model, the probabilities and the
//! categories are compared in `turbo-train-components`'
//! `tests/doccat_goldens.rs`, which trains the model on the corpus.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use turbo_analysis::{Document, Key};
use turbo_analysis_layers::Tokens;
use turbo_doccat::{
    BagOfWordsFeatureGenerator, BuiltinFeatureGenerator, DoccatFactory, EmojiFeatureGenerator, NGramFeatureGenerator,
};
use turbo_goldens::{ColumnSpec, ColumnType, Golden, GoldenAnchor, GoldenLayer, Shape, known_shapes, load_layer};

const FEATURES: &[ColumnSpec] = &[ColumnSpec::new("feature", ColumnType::Str)];

/// The command that writes the goldens.
const GENERATE: &str = "cargo run -p turbo-goldens-tool -- generate";

fn shapes() -> Vec<Shape> {
    let mut shapes = known_shapes();
    shapes.push(Shape { id: "turbo:doccat-features".into(), anchor: GoldenAnchor::Document, columns: FEATURES });
    shapes
}

fn layer<'g>(g: &'g Golden, id: &str) -> &'g GoldenLayer {
    g.layer(id).unwrap_or_else(|| panic!("no layer {id}"))
}

fn config<'g>(g: &'g Golden, key: &str) -> &'g str {
    g.header.config.iter().find(|c| c.key == key).map(|c| c.value.as_str()).unwrap_or_else(|| panic!("no config {key}"))
}

/// The files in `dir`, sorted.
fn sorted(dir: &Path) -> Vec<PathBuf> {
    let mut out: Vec<PathBuf> = std::fs::read_dir(dir)
        .unwrap_or_else(|e| panic!("{}: {e}", dir.display()))
        .map(|e| e.unwrap().path())
        .collect();
    out.sort();
    out
}

/// A feature as `TrainGolden.lossless` writes it: a backslash doubled. A
/// Rust string holds no lone surrogate, the other thing it escapes.
fn lossless(feature: &str) -> String {
    feature.replace('\\', "\\\\")
}

#[test]
fn every_document_has_the_jvms_features() {
    let root = turbo_goldens::data_dir().join("doccat");
    if !root.is_dir() {
        eprintln!("skipped: {} is absent; write the goldens with `{GENERATE}`", root.display());
        return;
    }
    let shapes = shapes();
    // The generators of every run, as `DoccatGolden.generators()` builds
    // them: the public constructors, the ones the manifest restores.
    let factory = DoccatFactory::with_feature_generators(vec![
        BuiltinFeatureGenerator::BagOfWords(BagOfWordsFeatureGenerator::new()),
        BuiltinFeatureGenerator::NGram(NGramFeatureGenerator::default()),
        BuiltinFeatureGenerator::Emoji(EmojiFeatureGenerator::new()),
    ]);
    let classes: Vec<&str> = factory.feature_generators().iter().map(|f| f.class_name()).collect();
    let extra = BTreeMap::new();
    let (mut runs, mut documents, mut features, mut emoji) = (0, 0, 0, 0);
    for run in sorted(&root) {
        let name = run.file_name().unwrap().to_string_lossy().into_owned();
        let goldens = sorted(&run.join("doccat"));
        assert_eq!(goldens.len(), 12, "{name}: document goldens");
        for path in goldens {
            let at = format!("{name}/{}", path.file_stem().unwrap().to_string_lossy());
            let g = Golden::read_with(&path, &shapes).unwrap_or_else(|e| panic!("{e}"));
            assert!(g.header.error.is_none(), "{at}");
            assert_eq!(config(&g, "run"), name, "{at}");
            assert_eq!(config(&g, "generators").split(',').collect::<Vec<_>>(), classes, "{at}: generators");

            let text = g.input_text(turbo_goldens::data_dir()).unwrap_or_else(|e| panic!("{e}"));
            let mut doc = Document::new(text.as_str());
            load_layer(&mut doc, &g, &Key::<Tokens>::DEFAULT).unwrap_or_else(|e| panic!("{at}: {e}"));
            let tokens = doc.get::<Tokens>().unwrap();
            let words: Vec<&str> = tokens.spans.iter().map(|(_, s)| &text[s.range()]).collect();

            let ours: Vec<String> =
                factory.context_generator().context(&words, &extra).iter().map(|f| lossless(f)).collect();
            let theirs: Vec<&str> = layer(&g, "turbo:doccat-features").strs("feature").unwrap().iter().collect();
            assert_eq!(ours, theirs, "{at}: features");
            features += ours.len();
            emoji += ours.iter().filter(|f| f.starts_with("emoji")).count();
            documents += 1;
        }
        runs += 1;
    }
    eprintln!("{runs} runs, {documents} documents, {features} features ({emoji} emoji features)");
    assert_eq!(runs, 4, "{}: runs", root.display());
    assert!(emoji > 0, "no document gave emoji features");
}
