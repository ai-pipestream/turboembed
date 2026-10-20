#[path = "qa/common.rs"]
mod common;
use std::{collections::BTreeMap, path::Path};
use turbo::{
    bundle::Bundle,
    qa::{Answer, DecodeOptions, ExtractiveQa, windows},
    safetensors::{Dtype, File},
    tokenizer::Tokenizer,
};
fn fixture() -> common::Fixture {
    common::Fixture::new(Path::new(env!("CARGO_MANIFEST_DIR")).parent().unwrap())
}

#[test]
fn joint_inputs_and_token_logits_match_transformers_reference() {
    let f = fixture();
    let bundle = Bundle::open(&f.dir).unwrap();
    let mut qa = ExtractiveQa::open(&f.dir, 1).unwrap();
    let bytes = bundle.read_verified(&bundle.manifest.reference.file).unwrap();
    let reference = File::parse("qa", &bytes).unwrap().get("logits", Dtype::F32, 2).unwrap().f32s();
    let mut actual = Vec::new();
    for case in &bundle.manifest.reference.cases {
        let context = case.context.as_deref().unwrap();
        for w in qa.logits(&case.text, context).unwrap() {
            for span in w.context.iter().flatten() {
                assert!(context.get(span.clone()).is_some());
            }
            for (start, end) in w.start.into_iter().zip(w.end) {
                actual.extend([start, end]);
            }
        }
        assert!(matches!(
            qa.answer(&case.text, context, DecodeOptions { null_threshold: 1e6, ..Default::default() }).unwrap(),
            Answer::None { .. }
        ));
        let answer =
            qa.answer(&case.text, context, DecodeOptions { null_threshold: -1e6, ..Default::default() }).unwrap();
        if let Answer::Spans { answers, null_probability, .. } = answer {
            for a in &answers {
                assert!(context.get(a.bytes.clone()).is_some());
            }
            assert!((answers.iter().map(|a| a.probability).sum::<f64>() + null_probability - 1.0).abs() < 1e-9);
        } else {
            assert!(context.is_empty());
        }
    }
    assert_eq!(actual.len(), reference.len());
    for (i, (&a, &b)) in actual.iter().zip(&reference).enumerate() {
        assert!((a - b).abs() < 1e-4, "logit {i}: {a} vs {b}");
    }
}

#[test]
fn windows_cover_all_context_and_own_each_start_once() {
    let f = fixture();
    let bundle = Bundle::open(&f.dir).unwrap();
    let tok = Tokenizer::load(&bundle).unwrap();
    let text = "alpha ".repeat(40);
    let q = bundle.manifest.qa.as_ref().unwrap();
    let ws = windows(&tok, turbo::manifest::Family::Bert, q, &"question ".repeat(30), &text).unwrap();
    assert!(ws.len() > 2);
    let mut counts = BTreeMap::new();
    for w in ws {
        assert!(w.ids.len() <= q.max_seq as usize);
        assert_eq!(w.types[..8], [0; 8]);
        for (span, owns) in w.context.iter().zip(w.max_context) {
            if let Some(s) = span {
                *counts.entry((s.start, s.end)).or_insert(0) += usize::from(owns);
            }
        }
    }
    assert_eq!(counts.len(), 40);
    assert!(counts.values().all(|&v| v == 1));
    let mut bad = q.clone();
    bad.doc_stride = u32::MAX;
    assert!(windows(&tok, turbo::manifest::Family::Bert, &bad, "q", &text).is_err());
}

#[test]
fn malformed_task_heads_and_budgets_are_rejected() {
    let mut f = fixture();
    let valid = f.manifest.clone();
    for (path, value) in [
        ("/classifier/labels", serde_json::json!(["end", "start"])),
        ("/classifier/kind", serde_json::json!("CLASSIFIER_SEQUENCE")),
        ("/qa/doc_stride", serde_json::json!(u32::MAX)),
        ("/qa/max_query_tokens", serde_json::json!(0)),
        ("/qa/max_seq", serde_json::json!(3)),
    ] {
        f.manifest = valid.clone();
        *f.manifest.pointer_mut(path).unwrap() = value;
        f.save();
        assert!(Bundle::open(&f.dir).is_err(), "{path}");
    }
}

#[test]
fn long_window_length_cannot_override_interior_context() {
    let f = fixture();
    let bundle = Bundle::open(&f.dir).unwrap();
    let mut tok = Tokenizer::load(&bundle).unwrap();
    tok.max_seq = 512;
    let mut q = bundle.manifest.qa.as_ref().unwrap().clone();
    q.max_seq = 321;
    q.max_query_tokens = 1;
    q.doc_stride = 128;
    let text = std::iter::repeat_n("alpha", 318).collect::<Vec<_>>().join(" ");
    let ws = windows(&tok, turbo::manifest::Family::Bert, &q, "alpha", &text).unwrap();
    assert_eq!(ws.len(), 2);
    assert_eq!(ws[0].context.iter().flatten().count(), 317);
    let answer_start = 316 * 6;
    let mut logits = Vec::new();
    for w in ws {
        let mut start = vec![-10.; w.ids.len()];
        let mut end = start.clone();
        start[0] = 0.;
        end[0] = 0.;
        for (i, span) in w.context.iter().enumerate() {
            if let Some(span) = span {
                if span.start == answer_start {
                    start[i] = 10.;
                }
                if span.end == text.len() {
                    end[i] = 10.;
                }
            }
        }
        logits.push(turbo::qa::WindowLogits { start, end, context: w.context, max_context: w.max_context, cls: 0 });
    }
    let Answer::Spans { answers, .. } = turbo::qa::decode(&text, &logits, DecodeOptions::default()).unwrap() else {
        panic!("the final two tokens must remain extractable");
    };
    assert_eq!(answers[0].bytes, answer_start..text.len());
}
