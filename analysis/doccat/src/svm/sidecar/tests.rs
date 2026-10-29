use super::*;

const SIDECAR: &str = "svm-doccat 1\n\
    language\teng\n\
    weighting\tTF_IDF\n\
    selection\tCHI_SQUARE\t40\n\
    scaling\t-1.0\t1.0\n\
    generator\tbag-of-words\n\
    generator\tngram\t2\t3\n\
    category\t0\tsports\n\
    category\t1\ttech\\tnology\n\
    feature\t1\t0.6931471805599453\t0.0\t2.0794415416798357\tbow=goal\n\
    feature\t2\t1.0E-5\t-0.0\t3.0\tng=:a:b\n";

#[test]
fn a_sidecar_reads_and_writes_back_the_same_text() {
    let s = Sidecar::read(SIDECAR).unwrap();
    assert_eq!(s.language.as_deref(), Some("eng"));
    assert_eq!(s.weighting, TermWeighting::TfIdf);
    assert_eq!((s.selection, s.max_features), (FeatureSelection::ChiSquare, 40));
    assert_eq!(s.scaling, Scaling::Range { lower: -1.0, upper: 1.0 });
    assert_eq!(s.generators, [FeatureGenerator::BagOfWords, FeatureGenerator::NGram { min: 2, max: 3 }]);
    assert_eq!(s.categories, ["sports", "tech\tnology"]);
    assert_eq!(s.vocabulary.len(), 2);
    assert_eq!(s.vocabulary.index("bow=goal"), Some(1));
    assert_eq!(s.vocabulary.index("ng=:a:b"), Some(2));
    assert_eq!(s.vocabulary.index("bow=other"), None);
    assert_eq!(s.vocabulary.idf(1), std::f64::consts::LN_2);
    assert_eq!(s.vocabulary.idf(2), 1e-5);
    assert_eq!(s.vocabulary.range(2).0.to_bits(), (-0.0f64).to_bits());
    assert_eq!(s.write(), SIDECAR);
}

#[test]
fn no_scaling_and_no_language_are_allowed() {
    let text = SIDECAR.replace("scaling\t-1.0\t1.0\n", "scaling\tnone\n").replace("language\teng\n", "");
    let s = Sidecar::read(&text).unwrap();
    assert_eq!(s.scaling, Scaling::None);
    assert_eq!(s.language, None);
    assert_eq!(s.write(), text);
}

fn error(text: &str) -> String {
    Sidecar::read(text).unwrap_err().to_string()
}

#[test]
fn a_malformed_sidecar_names_the_line() {
    assert_eq!(error("svm-doccat 2\n"), "SVM doccat sidecar line 1: the first line is \"svm-doccat 2\"");
    assert_eq!(error(""), "SVM doccat sidecar line 1: the first line is \"\"");
    assert_eq!(
        error(&SIDECAR.replace("TF_IDF", "TFIDF")),
        "SVM doccat sidecar line 3: weighting: \"TFIDF\" does not parse"
    );
    assert_eq!(
        error(&SIDECAR.replace("category\t1\t", "category\t2\t")),
        "SVM doccat sidecar line 9: category: index 2 is not above the one before it"
    );
    assert_eq!(
        error(&SIDECAR.replace("feature\t2\t", "feature\t3\t")),
        "SVM doccat sidecar line 11: feature: index 3 is not above the one before it"
    );
    assert_eq!(
        error(&SIDECAR.replace("ng=:a:b", "bow=goal")),
        "SVM doccat sidecar line 11: feature: \"bow=goal\" does not parse"
    );
    assert_eq!(error(&format!("{SIDECAR}weighting\tBINARY\n")), "SVM doccat sidecar line 12: weighting is given twice");
    assert_eq!(error(&format!("{SIDECAR}idf\t1\n")), "SVM doccat sidecar line 12: unknown keyword \"idf\"");
    assert_eq!(
        error(&SIDECAR.replace("scaling\t-1.0\t1.0", "scaling\t1.0\t1.0")),
        "SVM doccat sidecar line 5: scaling: \"1.0\\t1.0\" does not parse"
    );
    assert_eq!(
        error(&SIDECAR.replace("ngram\t2\t3", "ngram\t3\t2")),
        "SVM doccat sidecar line 7: generator: \"3\\t2\" does not parse"
    );
    assert_eq!(error(&SIDECAR.replace("selection\tCHI_SQUARE\t40\n", "")), "SVM doccat sidecar: no selection record");
    assert_eq!(
        error(&SIDECAR.replace("tech\\tnology", "sports")),
        "SVM doccat sidecar: the category \"sports\" is given twice"
    );
    assert_eq!(
        error(&SIDECAR.replace("\tbow=goal", "\tbow=goal\textra")),
        "SVM doccat sidecar line 10: feature: 6 values expected, 7 found"
    );
}

#[test]
fn names_escape_controls_and_backslashes() {
    let names = ["plain", "tab\there", "line\nend", "cr\rhere", "back\\slash", "bell\u{7}", "emoji \u{1f600}", ""];
    for n in names {
        assert_eq!(unescape(&escape(n)).as_deref(), Some(n), "{n:?}");
    }
    assert_eq!(escape("a\u{1}b"), "a\\u0001b");
    assert_eq!(unescape("\\ud83d\\ude00").as_deref(), Some("\u{1f600}"));
    assert_eq!(unescape("\\ud83d"), None);
    assert_eq!(unescape("\\ude00"), None);
    assert_eq!(unescape("\\ud83dx"), None);
    assert_eq!(unescape("\\x"), None);
    assert_eq!(unescape("\\u12"), None);
    assert_eq!(unescape("raw\ttab"), None);
    assert_eq!(unescape("trailing\\"), None);
}
