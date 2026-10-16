//! The categorizer on hand-written models: the vector it builds, the
//! scores, OpenNLP's `DocumentCategorizer` methods, the annotator, and the
//! cases of OpenNLP's `DocumentCategorizerSVMTest` and
//! `DocumentCategorizerSVMEdgeCaseTest` that do not train.

use turbo_analysis::{Analyzer, Document, Key, Provenance, Span, Spans, ViewId};
use turbo_analysis_layers::{Categories, Tokens};
use turbo_doccat::svm::{ModelError, Scratch, SvmCategorizer, SvmDoccatModel};

/// Two categories, a linear kernel: `a` scores `bow=x` and `bow=y`, `b`
/// scores `bow=p`; the decision value is `x + y - p - 0.25` on the
/// vector.
const MODEL: &str = "svm_type c_svc
kernel_type linear
nr_class 2
total_sv 2
rho 0.25
label 0 1
probA -2.0
probB 0.0
nr_sv 1 1
SV
1.0 1:1.0 2:1.0
-1.0 3:1.0
";

const SIDECAR: &str = "svm-doccat 1
language\teng
weighting\tBINARY
selection\tNONE\t-1
scaling\tnone
generator\tbag-of-words
category\t0\ta
category\t1\tb
feature\t1\t0.6931471805599453\t0.0\t1.0\tbow=x
feature\t2\t1.3862943611198906\t0.0\t1.0\tbow=y
feature\t3\t0.6931471805599453\t0.0\t1.0\tbow=p
";

fn model(libsvm: &str, sidecar: &str) -> SvmDoccatModel {
    SvmDoccatModel::from_text(libsvm, sidecar).unwrap_or_else(|e| panic!("{e}"))
}

fn categorizer(sidecar: &str) -> SvmCategorizer {
    SvmCategorizer::new(model(MODEL, sidecar)).unwrap()
}

fn vector(c: &SvmCategorizer, tokens: &[&str]) -> Vec<(i32, f64)> {
    let mut s = Scratch::new();
    c.model().vectorize(tokens, &mut s);
    let (i, v) = s.vector();
    i.iter().copied().zip(v.iter().copied()).collect()
}

/// The two-class probabilities of decision value `d`, with probA -2.
fn probabilities(d: f64) -> [f64; 2] {
    let f = d * -2.0;
    let p = if f >= 0.0 { (-f).exp() / (1.0 + (-f).exp()) } else { 1.0 / (1.0 + f.exp()) };
    [p, 1.0 - p]
}

#[test]
fn binary_weighting_gives_each_known_feature_once_in_index_order() {
    let c = categorizer(SIDECAR);
    assert_eq!(vector(&c, &["p", "x", "x", "zzz"]), [(1, 1.0), (3, 1.0)]);
    assert_eq!(vector(&c, &[]), []);
}

#[test]
fn term_frequency_counts_repeats() {
    let c = categorizer(&SIDECAR.replace("BINARY", "TERM_FREQUENCY"));
    assert_eq!(vector(&c, &["y", "x", "y", "y"]), [(1, 1.0), (2, 3.0)]);
}

#[test]
fn tf_idf_multiplies_the_count_by_the_idf() {
    let c = categorizer(&SIDECAR.replace("BINARY", "TF_IDF"));
    assert_eq!(vector(&c, &["x", "y", "y"]), [(1, std::f64::consts::LN_2), (2, 2.0 * 1.3862943611198906)]);
    // A feature whose idf is 0 weighs 0 and is dropped.
    let zero = categorizer(&SIDECAR.replace("BINARY", "TF_IDF").replace("1\t0.6931471805599453", "1\t0.0"));
    assert_eq!(vector(&zero, &["x", "y"]), [(2, 1.3862943611198906)]);
}

#[test]
fn log_normalized_tf_is_one_plus_the_log_of_the_count() {
    let c = categorizer(&SIDECAR.replace("BINARY", "LOG_NORMALIZED_TF"));
    assert_eq!(vector(&c, &["x", "y", "y"]), [(1, 1.0), (2, 1.0 + std::f64::consts::LN_2)]);
}

#[test]
fn scaling_maps_the_training_range_onto_the_configured_one() {
    let sidecar = SIDECAR
        .replace("BINARY", "TERM_FREQUENCY")
        .replace("scaling\tnone", "scaling\t-1.0\t1.0")
        .replace("1\t0.6931471805599453\t0.0\t1.0", "1\t0.6931471805599453\t0.0\t4.0")
        .replace("2\t1.3862943611198906\t0.0\t1.0", "2\t1.3862943611198906\t1.0\t1.0");
    let c = categorizer(&sidecar);
    // x: 1 of 0..4 -> -0.5; y: equal bounds -> the lower end; p: 3 clamped to 1 -> 1.
    assert_eq!(vector(&c, &["x", "y", "p", "p", "p"]), [(1, -0.5), (2, -1.0), (3, 1.0)]);
    // x: 2 of 0..4 scales to 0.0, which is dropped.
    assert_eq!(vector(&c, &["x", "x"]), []);
}

#[test]
fn ngrams_and_bag_of_words_share_one_vocabulary() {
    let sidecar = SIDECAR
        .replace("generator\tbag-of-words\n", "generator\tbag-of-words\ngenerator\tngram\t2\t2\n")
        .replace("bow=p", "ng=:x:y");
    let c = categorizer(&sidecar);
    assert_eq!(vector(&c, &["x", "y"]), [(1, 1.0), (2, 1.0), (3, 1.0)]);
    assert_eq!(vector(&c, &["y", "x"]), [(1, 1.0), (2, 1.0)]);
}

#[test]
fn categorize_gives_each_category_its_probability() {
    let c = categorizer(SIDECAR);
    let mut s = Scratch::new();
    assert_eq!(c.categorize(&["x", "y"], &mut s), probabilities(2.0 - 0.25));
    assert_eq!(c.categorize(&["p"], &mut s), probabilities(-1.25));
    let vote = c.model().decision_values(&["p"], &mut s);
    assert_eq!(vote.category, 1);
    assert_eq!(s.decision_values(), [-1.25]);
}

// DocumentCategorizerSVMEdgeCaseTest: testClassifyWithUnknownTokens and
// testClassifyWithMixOfKnownAndUnknown.
#[test]
fn unknown_tokens_still_give_valid_probabilities() {
    let c = categorizer(SIDECAR);
    let mut s = Scratch::new();
    let unknown = c.categorize(&["unknown1", "unknown2", "unknown3"], &mut s).to_vec();
    assert_eq!(unknown, probabilities(-0.25));
    assert_eq!(c.best_category(&unknown), "b");
    let mixed = c.categorize(&["x", "zzz"], &mut s).to_vec();
    assert_eq!(mixed, probabilities(0.75));
    assert_eq!(c.best_category(&mixed), "a");
}

// DocumentCategorizerSVMTest: testGetCategoryAndIndex; the edge case
// tests: testGetIndexNonExistent and testGetAllResultsFormat.
#[test]
fn categories_are_named_and_indexed_as_in_opennlp() {
    let c = categorizer(SIDECAR);
    assert_eq!(c.num_categories(), 2);
    assert_eq!(c.index("a"), Some(0));
    assert_eq!(c.index("b"), Some(1));
    assert_eq!(c.index("nonexistent_category"), None);
    assert_eq!((c.category(0), c.category(1)), ("a", "b"));
    assert_eq!(c.all_results(&[0.25, 0.75]), "a[0.25] b[0.75]");
    assert_eq!(c.all_results(&[1e-7, 0.9999999]), "a[1.0E-7] b[0.9999999]");
    assert_eq!(c.all_results(&[]), "");
    assert_eq!(c.all_results(&[0.5]), "");
    assert_eq!(c.all_results(&[0.2, 0.3, 0.5]), "");
    // The first highest score wins a tie.
    assert_eq!(c.best_category(&[0.5, 0.5]), "a");
    assert_eq!(c.best_category(&[0.1, 0.2, 0.9]), "");
}

/// Three categories; the second support vector's category scores `p`.
const THREE: &str = "svm_type c_svc
kernel_type linear
nr_class 3
total_sv 3
rho 0.0 0.0 0.0
label 0 1 2
probA -1.0 -1.0 -1.0
probB 0.0 0.0 0.0
nr_sv 1 1 1
SV
1.0 1.0 1:1.0
-1.0 1.0 3:1.0
-1.0 -1.0 2:1.0
";

// DocumentCategorizerSVMEdgeCaseTest: testThreeCategories.
#[test]
fn three_categories_sum_to_one() {
    let sidecar = SIDECAR.replace("category\t1\tb\n", "category\t1\tb\ncategory\t2\tc\n");
    let c = SvmCategorizer::new(model(THREE, &sidecar)).unwrap();
    let mut s = Scratch::new();
    for tokens in [&["x"][..], &["p"], &["y"], &[], &["x", "p", "y"]] {
        let p = c.categorize(tokens, &mut s).to_vec();
        assert_eq!(p.len(), 3);
        assert!(p.iter().all(|&v| v >= 0.0), "{p:?}");
        assert!((p.iter().sum::<f64>() - 1.0).abs() < 0.05, "{p:?}");
    }
    let p = c.categorize(&["x"], &mut s).to_vec();
    assert_eq!(c.best_category(&p), "a");
    let p = c.categorize(&["p"], &mut s).to_vec();
    assert_eq!(c.best_category(&p), "b");
}

#[test]
fn labels_place_probabilities_by_category_index() {
    // LIBSVM met category 1 first.
    let swapped = MODEL.replace("label 0 1", "label 1 0");
    let c = SvmCategorizer::new(model(&swapped, SIDECAR)).unwrap();
    let mut s = Scratch::new();
    let [p, q] = probabilities(1.75);
    assert_eq!(c.categorize(&["x", "y"], &mut s), [q, p]);
}

fn model_error(libsvm: &str, sidecar: &str) -> String {
    match SvmDoccatModel::from_text(libsvm, sidecar).and_then(SvmCategorizer::new) {
        Err(e) => e.to_string(),
        Ok(_) => panic!("the model was accepted"),
    }
}

#[test]
fn a_model_the_categorizer_cannot_run_is_refused() {
    assert_eq!(
        model_error(&MODEL.replace("probA -2.0\nprobB 0.0\n", ""), SIDECAR),
        "LIBSVM model: no probA and probB, so it gives no probability estimates"
    );
    assert_eq!(
        model_error(&MODEL.replace("label 0 1", "label 0 2"), SIDECAR),
        "LIBSVM model: the label 2 is not a category index given once (0 to 1)"
    );
    assert_eq!(
        model_error(&MODEL.replace("label 0 1", "label 0 0"), SIDECAR),
        "LIBSVM model: the label 0 is not a category index given once (0 to 1)"
    );
    let one_class = "svm_type one_class\nkernel_type linear\nnr_class 2\ntotal_sv 1\nrho 0.5\nSV\n2.0 1:1.0\n";
    assert_eq!(
        model_error(one_class, SIDECAR),
        "LIBSVM model: a one_class model gives no class labels to categorize with"
    );
    assert!(matches!(
        SvmDoccatModel::open(std::path::Path::new("/nonexistent/model.txt"), std::path::Path::new("x")),
        Err(ModelError::Io { .. })
    ));
}

#[test]
fn the_model_reference_hashes_both_files() {
    let a = model(MODEL, SIDECAR);
    let b = model(MODEL, &SIDECAR.replace("1.3862943611198906", "1.0"));
    assert_eq!(a.model_ref().id, "opennlp:SvmDoccatModel:eng");
    assert_eq!(a.model_ref().sha256.len(), 64);
    assert_ne!(a.model_ref().sha256, b.model_ref().sha256);
    let und = model(MODEL, &SIDECAR.replace("language\teng\n", ""));
    assert_eq!(und.model_ref().id, "opennlp:SvmDoccatModel:und");
}

/// Tokens at the spaces of `text`.
fn document(text: &str) -> Document {
    let mut spans = Spans::new(ViewId::SOURCE);
    let mut start = 0;
    for piece in text.split(' ').filter(|_| !text.is_empty()) {
        spans.push(Span::new(start as u32, (start + piece.len()) as u32));
        start += piece.len() + 1;
    }
    let mut doc = Document::new(text);
    doc.add(&Key::DEFAULT, Tokens { spans, sentence: None }, Provenance::by("test")).unwrap();
    doc
}

#[test]
fn the_annotator_writes_every_category_best_first() {
    let analyzer =
        Analyzer::builder().given(Key::<Tokens>::DEFAULT).add(categorizer(SIDECAR)).build().expect("the analyzer");
    let mut docs = vec![document("p zzz"), document("x y"), document("")];
    for r in analyzer.run(&mut docs) {
        r.unwrap();
    }
    let rows = |d: &Document| {
        d.get::<Categories>().unwrap().ranked().into_iter().map(|(c, p)| (c.to_string(), p)).collect::<Vec<_>>()
    };
    let [a, b] = probabilities(-1.25);
    assert_eq!(rows(&docs[0]), [("b".to_string(), b), ("a".to_string(), a)]);
    let [a, b] = probabilities(1.75);
    assert_eq!(rows(&docs[1]), [("a".to_string(), a), ("b".to_string(), b)]);
    assert_eq!(docs[1].get::<Categories>().unwrap().best(), Some(("a", a)));
    let [a, b] = probabilities(-0.25);
    assert_eq!(rows(&docs[2]), [("b".to_string(), b), ("a".to_string(), a)]);

    let bad_doc = Document::new("é");
    let mut bad_spans = Spans::new(ViewId::SOURCE);
    bad_spans.push(Span::new(0, 1));
    let bad_tokens = Tokens { spans: bad_spans, sentence: None };
    assert!(
        turbo_analysis::Annotator::annotate(
            &categorizer(SIDECAR),
            (bad_doc.get::<turbo_analysis::View>().unwrap(), &bad_tokens)
        )
        .is_err()
    );
}
