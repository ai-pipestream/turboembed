//! Evaluates the GIS, QN and perceptron models OpenNLP trained and wrote
//! (fixtures/models, made by fixtures/src/MakeModelFixtures.java) and
//! compares the probabilities with the JVM's, and opens the tokenizer
//! model file OpenNLP serialized.

mod common;

use common::{doubles, modified_utf8, unescape};
use turbo_maxent::{Dictionary, MaxentModel, ModelFile, ModelKind};

/// One expected evaluation: the features as modified UTF-8 and the
/// probabilities the JVM gave.
type Case = (Vec<Vec<u8>>, Vec<f64>);

/// The outcome names and expected evaluations for one model.
fn expected(model: &str) -> (Vec<String>, Vec<Case>) {
    let text = std::fs::read_to_string(common::fixtures().join("models/expected.txt")).unwrap();
    let mut outcomes = Vec::new();
    let mut cases = Vec::new();
    for line in text.lines() {
        if let Some(rest) = line.strip_prefix(&format!("# {model} ")) {
            let names = rest.split_once("outcomes: ").unwrap().1;
            outcomes = names.split(' ').map(str::to_string).collect();
            continue;
        }
        let mut f = line.split('\t');
        if f.next() != Some(model) {
            continue;
        }
        let features = f.next().unwrap();
        let features: Vec<Vec<u8>> = if features.is_empty() {
            Vec::new()
        } else {
            features.split(' ').map(|s| modified_utf8(&unescape(s))).collect()
        };
        cases.push((features, doubles(f.next().unwrap())));
    }
    (outcomes, cases)
}

fn check(model: &MaxentModel, name: &str) {
    let (outcomes, cases) = expected(name);
    assert_eq!(model.outcomes().collect::<Vec<_>>(), outcomes, "{name}");
    assert!(!cases.is_empty());
    let mut out = vec![0.0; model.num_outcomes()];
    let mut exact = 0;
    let mut total = 0;
    for (features, want) in &cases {
        let context: Vec<&[u8]> = features.iter().map(Vec::as_slice).collect();
        model.eval(&context, &mut out);
        for (got, want) in out.iter().zip(want) {
            assert!((got - want).abs() <= 1e-9, "{name} {features:?}: {got} vs {want}");
            // Probabilities are positive, so their bit patterns order like
            // their values and the difference counts units in the last place.
            let ulps = (got.to_bits() as i64 - want.to_bits() as i64).abs();
            assert!(ulps <= 1, "{name} {features:?}: {got} vs {want} differ by {ulps} ulps");
            exact += usize::from(got == want);
            total += 1;
        }
        let best = want.iter().enumerate().fold(0, |b, (i, &p)| if want[b] < p { i } else { b });
        assert_eq!(model.best_outcome(&out), best);
    }
    // f64::exp and f64::ln are the platform's, not StrictMath's fdlibm, so a
    // probability can differ from the JVM's by one unit in the last place
    // (on the machine the fixtures were checked on, 11 of 100 do). Most
    // agree exactly.
    assert!(exact * 2 > total, "{name}: only {exact} of {total} probabilities are bit-identical");
}

fn read_model(name: &str) -> MaxentModel {
    MaxentModel::read(&std::fs::read(common::fixtures().join("models").join(name)).unwrap()).unwrap()
}

#[test]
fn gis_probabilities_match_the_jvm() {
    let m = read_model("gis.model");
    assert_eq!(m.kind(), ModelKind::Gis);
    check(&m, "gis.model");
}

#[test]
fn qn_probabilities_match_the_jvm() {
    let m = read_model("qn.model");
    assert_eq!(m.kind(), ModelKind::Qn);
    check(&m, "qn.model");
}

#[test]
fn perceptron_probabilities_match_the_jvm() {
    let m = read_model("perceptron.model");
    assert_eq!(m.kind(), ModelKind::Perceptron);
    check(&m, "perceptron.model");
}

#[test]
fn predicates_are_keyed_by_modified_utf8() {
    let m = read_model("gis.model");
    // NUL is C0 80 and U+1F600 is two three-byte surrogate sequences.
    assert!(m.has_predicate(b"w=\xC0\x80"));
    assert!(!m.has_predicate(b"w=\0"));
    assert!(m.has_predicate(b"w=\xED\xA0\xBD\xED\xB8\x80"));
    assert!(!m.has_predicate("w=\u{1F600}".as_bytes()));
    assert!(m.has_predicate("w=été".as_bytes()));
}

#[test]
fn the_tokenizer_model_file_opennlp_wrote() {
    let path = common::fixtures().join("models/tokenizer.bin");
    let file = ModelFile::read(&path).unwrap();
    assert_eq!(file.bytes(), std::fs::read(&path).unwrap().as_slice());
    assert_eq!(file.archive().bytes(), file.bytes());
    assert_eq!(file.component_name(), "TokenizerME");
    assert_eq!(file.language(), "en");
    assert_eq!(file.opennlp_version(), "3.0.0-SNAPSHOT");
    assert_eq!(file.manifest().get("alphaNumericPattern"), Some("^[A-Za-z0-9]+$"));
    assert_eq!(file.manifest().get("useAlphaNumericOptimization"), Some("false"));
    assert_eq!(
        file.manifest().get("serializer-class-abbreviations.dictionary"),
        Some("opennlp.tools.util.model.DictionarySerializer")
    );

    let model = file.maxent("token.model").unwrap();
    assert_eq!(model.kind(), ModelKind::Gis);
    check(&model, "tokenizer.bin");

    let abbreviations: Dictionary = file.dictionary("abbreviations.dictionary").unwrap().unwrap();
    assert!(!abbreviations.is_case_sensitive());
    assert_eq!(abbreviations.len(), 3);
    assert!(abbreviations.contains("dr."));
    assert!(abbreviations.contains("E.G."));
    assert!(abbreviations.contains("U.S."));
    assert!(!abbreviations.contains("Mr."));
    assert!(file.dictionary("missing.dictionary").unwrap().is_none());
    assert!(file.dictionary_entries("abbreviations.dictionary").unwrap().unwrap().len() == 3);
}

#[test]
fn errors_name_the_entry() {
    let file = ModelFile::read(common::fixtures().join("models/tokenizer.bin")).unwrap();
    let err = file.maxent("abbreviations.dictionary").unwrap_err().to_string();
    assert!(err.starts_with("entry \"abbreviations.dictionary\": "), "{err}");
    let err = file.maxent("nope.model").unwrap_err().to_string();
    assert_eq!(err, "the archive has no entry \"nope.model\"");
}
