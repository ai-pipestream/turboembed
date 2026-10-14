//! Opens the Spanish and French TokenizerME models of the OpenNLP UD 1.3
//! release and compares their evaluation with the JVM's.
//!
//! The models are about 540 KB each and are not checked in: `turbo-goldens
//! fetch-models` downloads them (`turbo_goldens::published_model`). Without
//! them these tests say so and pass, unless
//! `TURBO_TEST_REQUIRE_OPENNLP_MODELS=1`. The expected values were printed
//! by fixtures/src/RealModelValues.java (see fixtures/README.md).

mod common;

use turbo_maxent::{ModelFile, ModelKind, best_outcome};

type Case = (&'static [&'static str], [f64; 2]);

/// The model file, or None after saying the test is skipped.
fn model(name: &str) -> Option<ModelFile> {
    let path = turbo_goldens::published_model(name)?;
    Some(ModelFile::read(&path).unwrap_or_else(|e| panic!("{}: {e}", path.display())))
}

/// Checks the manifest, the model's shape and the ten contexts.
fn check(file: &ModelFile, language: &str, pattern: &str, predicates: usize, cases: &[Case]) {
    assert_eq!(file.component_name(), "TokenizerME");
    assert_eq!(file.language(), language);
    assert_eq!(file.opennlp_version(), "2.5.4");
    assert_eq!(file.manifest().get("alphaNumericPattern"), Some(pattern));
    assert_eq!(file.manifest().get("useAlphaNumericOptimization"), Some("false"));
    assert!(file.dictionary("abbreviations.dictionary").unwrap().is_none());

    let model = file.maxent("token.model").unwrap();
    assert_eq!(model.kind(), ModelKind::Gis);
    assert_eq!(model.outcomes().collect::<Vec<_>>(), ["F", "T"]);
    assert_eq!(model.outcome_index("T"), Some(1));
    assert_eq!(model.num_predicates(), predicates);

    let mut out = [0.0; 2];
    for (context, want) in cases {
        model.eval(context, &mut out);
        assert_eq!(&out, want, "{language} {context:?}");
        let best = if want[0] < want[1] { 1 } else { 0 };
        assert_eq!(best_outcome(&out), best, "{language} {context:?}");
    }
    assert_eq!(cases.len(), 10);
}

const ES_CASES: [Case; 10] = [
    (
        &["p=Hola", "s=,", "p1=a", "p1_alpha", "p2=l", "p2_alpha", "p21=la", "p1f1=a,", "f1=,", "f2=bok"],
        [2.75205937022738E-6, 0.9999972479406298],
    ),
    (
        &["p=dijo", "s=:", "p1=o", "p1_alpha", "p2=j", "p2_alpha", "p21=jo", "p1f1=o:", "f1=:", "f2=bok"],
        [0.002105810667993851, 0.9978941893320061],
    ),
    (
        &[
            "p=(",
            "s=España)",
            "p1=(",
            "p1_lp",
            "p2=bok",
            "p1f1=(E",
            "f1=E",
            "f1_alpha",
            "f1_caps",
            "f2=s",
            "f2_alpha",
            "f12=Es",
        ],
        [7.119305259361377E-7, 0.999999288069474],
    ),
    (
        &[
            "p=l'", "s=homme", "p1='", "p1_quote", "p2=l", "p2_alpha", "p21=l'", "p1f1='h", "f1=h", "f1_alpha", "f2=o",
            "f2_alpha", "f12=ho",
        ],
        [0.5541179856074162, 0.4458820143925839],
    ),
    (
        &[
            "p=aujourd'",
            "s=hui",
            "p1='",
            "p1_quote",
            "p2=d",
            "p2_alpha",
            "p21=d'",
            "p1f1='h",
            "f1=h",
            "f1_alpha",
            "f2=u",
            "f2_alpha",
            "f12=hu",
        ],
        [0.9620725612469668, 0.03792743875303313],
    ),
    (
        &["p=3,5", "s=%", "p1=5", "p1_num", "p2=,", "p21=,5", "p1f1=5%", "f1=%", "f2=bok"],
        [0.9731618369455728, 0.026838163054427235],
    ),
    (
        &[
            "p=\"Sí\"",
            "s=.",
            "p1=\"",
            "p1_quote",
            "p2=í",
            "p2_alpha",
            "p21=í\"",
            "p1f1=\".",
            "f1=.",
            "f1_eos",
            "f2=bok",
        ],
        [2.556417617919241E-7, 0.9999997443582381],
    ),
    (
        &["p=etc", "s=.", "p1=c", "p1_alpha", "p2=t", "p2_alpha", "p21=tc", "p1f1=c.", "f1=.", "f1_eos", "f2=bok"],
        [0.7770862403674753, 0.22291375963252472],
    ),
    (&["no-such-predicate", "p=H"], [0.9783595344568641, 0.021640465543135905]),
    (&[], [0.5, 0.5]),
];

const FR_CASES: [Case; 10] = [
    (
        &["p=Hola", "s=,", "p1=a", "p1_alpha", "p2=l", "p2_alpha", "p21=la", "p1f1=a,", "f1=,", "f2=bok"],
        [0.003163751105026393, 0.9968362488949736],
    ),
    (
        &["p=dijo", "s=:", "p1=o", "p1_alpha", "p2=j", "p2_alpha", "p21=jo", "p1f1=o:", "f1=:", "f2=bok"],
        [0.97904439730036, 0.020955602699640075],
    ),
    (
        &[
            "p=(",
            "s=España)",
            "p1=(",
            "p1_lp",
            "p2=bok",
            "p1f1=(E",
            "f1=E",
            "f1_alpha",
            "f1_caps",
            "f2=s",
            "f2_alpha",
            "f12=Es",
        ],
        [1.3690281992981205E-5, 0.999986309718007],
    ),
    (
        &[
            "p=l'", "s=homme", "p1='", "p1_quote", "p2=l", "p2_alpha", "p21=l'", "p1f1='h", "f1=h", "f1_alpha", "f2=o",
            "f2_alpha", "f12=ho",
        ],
        [4.943102021083119E-4, 0.9995056897978918],
    ),
    (
        &[
            "p=aujourd'",
            "s=hui",
            "p1='",
            "p1_quote",
            "p2=d",
            "p2_alpha",
            "p21=d'",
            "p1f1='h",
            "f1=h",
            "f1_alpha",
            "f2=u",
            "f2_alpha",
            "f12=hu",
        ],
        [0.964183822576723, 0.03581617742327695],
    ),
    (
        &["p=3,5", "s=%", "p1=5", "p1_num", "p2=,", "p21=,5", "p1f1=5%", "f1=%", "f2=bok"],
        [0.003256965849364411, 0.9967430341506357],
    ),
    (
        &[
            "p=\"Sí\"",
            "s=.",
            "p1=\"",
            "p1_quote",
            "p2=í",
            "p2_alpha",
            "p21=í\"",
            "p1f1=\".",
            "f1=.",
            "f1_eos",
            "f2=bok",
        ],
        [6.811541050229762E-8, 0.9999999318845895],
    ),
    (
        &["p=etc", "s=.", "p1=c", "p1_alpha", "p2=t", "p2_alpha", "p21=tc", "p1f1=c.", "f1=.", "f1_eos", "f2=bok"],
        [0.9177756287599724, 0.0822243712400275],
    ),
    (&["no-such-predicate", "p=H"], [0.9914142913953286, 0.00858570860467142]),
    (&[], [0.5, 0.5]),
];

#[test]
fn spanish_tokenizer_model() {
    let Some(file) = model("opennlp-es-ud-gsd-tokens-1.3-2.5.4.bin") else { return };
    let pattern =
        "^[0-9a-z\u{E1}\u{E9}\u{ED}\u{F3}\u{FA}\u{FC}\u{FD}\u{F1}A-Z\u{C1}\u{C9}\u{CD}\u{D3}\u{DA}\u{DD}\u{D1}]+$";
    check(&file, "es", pattern, 49355, &ES_CASES);
}

#[test]
fn french_tokenizer_model() {
    let Some(file) = model("opennlp-fr-ud-gsd-tokens-1.3-2.5.4.bin") else { return };
    let pattern = "^[a-zA-Z0-9\u{E0}\u{E2}\u{E4}\u{E8}\u{E9}\u{EA}\u{EB}\u{EE}\u{EF}\u{F4}\u{153}\u{F9}\u{FB}\u{FC}\u{FF}\u{E7}\u{C0}\u{C2}\u{C4}\u{C8}\u{C9}\u{CA}\u{CB}\u{CE}\u{CF}\u{D4}\u{152}\u{D9}\u{DB}\u{DC}\u{178}\u{C7}]+$";
    check(&file, "fr", pattern, 48440, &FR_CASES);
}
