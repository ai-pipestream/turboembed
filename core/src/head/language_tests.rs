use crate::head::TokenTagger;
use crate::head::tagger::tensors;
use crate::manifest::{Decode, HeadKind, HeadSpec, LanguageInput, Scheme, WordPooling};

fn spec(languages: &[&str]) -> HeadSpec {
    HeadSpec {
        name: "upos".into(),
        layer: "turboembed:pos".into(),
        kind: HeadKind::TokenTagger,
        labels: vec!["A".into(), "B".into()],
        scheme: Scheme::Tags,
        word_pooling: WordPooling::First,
        adapter: None,
        decode: Decode::Argmax,
        biaffine: None,
        pos_input: None,
        hash_embed: None,
        language: Some(LanguageInput {
            languages: languages.iter().map(|language| (*language).into()).collect(),
            dim: 2,
        }),
        seen: None,
        span: None,
        coref: None,
        seen_lookup: None,
        relation: None,
        layer_mix: None,
        license_class: None,
        artifact: "weights-f32".into(),
        tensor_prefix: "heads.upos".into(),
    }
}

fn weights(suffix: &str) -> Option<Vec<f32>> {
    match suffix {
        "language.embedding" => Some(vec![1.0, 0.0, 0.0, 1.0]),
        "out.weight" => Some(vec![0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0]),
        "out.bias" => Some(vec![0.0, 0.0]),
        _ => None,
    }
}

#[test]
fn the_selected_language_changes_scores_and_decode() {
    let spec = spec(&["es", "ca"]);
    let shapes = tensors(&spec, 2);
    assert_eq!(shapes[0], ("language.embedding".into(), vec![2, 2]));
    assert_eq!(shapes[1], ("out.weight".into(), vec![2, 4]));
    let head = TokenTagger::from_tensors(&spec, 2, &weights).unwrap();
    let (mut scores, mut decoded, mut work) = (Vec::new(), Vec::new(), Default::default());
    for (language, expected, label) in [("es", [1.0, 0.0], 0), ("ca", [0.0, 1.0], 1)] {
        head.scores_language(&[5.0, 7.0], &[0], None, Some(language), &mut scores, &mut work).unwrap();
        assert_eq!(scores, expected, "{language}");
        head.decode(&scores, 1, &mut decoded).unwrap();
        assert_eq!(decoded, [label], "{language}");
    }
    let error = head.scores(&[5.0, 7.0], &[0], &mut scores, &mut work).unwrap_err();
    assert!(error.message.contains("needs a language"), "{error}");
    let error = head.scores_language(&[5.0, 7.0], &[0], None, Some("it"), &mut scores, &mut work).unwrap_err();
    assert!(error.message.contains("unknown language"), "{error}");
}

#[test]
fn one_language_defaults_and_absent_input_refuses_a_selector() {
    let one = spec(&["es"]);
    let head = TokenTagger::from_tensors(&one, 2, &|suffix| match suffix {
        "language.embedding" => Some(vec![1.0, 0.0]),
        _ => weights(suffix),
    })
    .unwrap();
    let (mut scores, mut work) = (Vec::new(), Default::default());
    head.scores(&[5.0, 7.0], &[0], &mut scores, &mut work).unwrap();
    assert_eq!(scores, [1.0, 0.0]);

    let mut plain = one.clone();
    plain.language = None;
    let head = TokenTagger::from_tensors(&plain, 2, &|suffix| match suffix {
        "out.weight" => Some(vec![0.0; 4]),
        "out.bias" => Some(vec![0.0; 2]),
        _ => None,
    })
    .unwrap();
    let error = head.scores_language(&[5.0, 7.0], &[0], None, Some("es"), &mut scores, &mut work).unwrap_err();
    assert!(error.message.contains("without a language input"), "{error}");
}

#[test]
fn language_specs_require_unique_nonempty_names() {
    let mut spec = spec(&["es", "es"]);
    assert!(spec.validate("head").is_err());
    spec.language.as_mut().unwrap().languages.clear();
    assert!(spec.validate("head").is_err());
}
