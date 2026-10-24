#[path = "../../../core/tests/qa/common.rs"]
mod common;
use std::path::Path;
use turbo::qa::{Answer, DecodeOptions, SpanAnswer};
use turbo_analysis::*;
use turbo_analysis_embed::{Answers, ExtractiveQaAnnotator, Question};

fn fixture() -> common::Fixture {
    common::Fixture::new(Path::new(env!("CARGO_MANIFEST_DIR")).join("../..").as_path())
}

#[test]
fn annotator_scores_utf8_context_and_records_configuration() {
    let f = fixture();
    assert!(turbo_analysis_embed::Encoder::open(&f.dir, &Default::default()).is_err());
    let answerer =
        ExtractiveQaAnnotator::open(&f.dir, 1, DecodeOptions { null_threshold: -1000.0, ..Default::default() })
            .unwrap();
    let abstainer =
        ExtractiveQaAnnotator::open(&f.dir, 1, DecodeOptions { null_threshold: 1000.0, ..Default::default() }).unwrap();
    assert_ne!(answerer.config_hash(), abstainer.config_hash());
    let mut doc = Document::new("The café is in Montréal.");
    doc.add(&Key::DEFAULT, Question("Where is the café?".into()), Provenance::by("test")).unwrap();
    let view = doc.get::<View>().unwrap();
    let question = doc.get::<Question>().unwrap();
    let (answers,) = answerer.annotate((view, question)).unwrap();
    assert!(!answers.is_empty());
    for i in 0..answers.len() {
        let span = answers.spans.get(Ix::new(i as u32));
        assert!(view.text().get(span.start as usize..span.end as usize).is_some());
    }
    let (empty,) = abstainer.annotate((view, question)).unwrap();
    assert!(empty.is_empty());
    assert_eq!(empty.null_probability, 1.0);
    doc.add(&Key::DEFAULT, answers, Provenance::by("test")).unwrap();
    let model = answerer.model().unwrap();
    assert!(!model.sha256.is_empty());
}

#[test]
fn answer_projection_uses_the_checked_view_path() {
    let mut doc = Document::new("Cafe 東京");
    let mut b = ViewBuilder::new(doc.source());
    b.keep(3);
    b.replace(4, "é");
    let (text, alignment) = b.finish();
    let view = View::derived(doc.get::<View>().unwrap(), text, alignment);
    doc.add(&Key::named("accent"), view, Provenance::by("test")).unwrap();
    let decoded = Answer::Spans {
        answers: vec![SpanAnswer { bytes: 0..5, score: 2.0, probability: 0.8 }],
        null_score: 0.0,
        null_probability: 0.2,
    };
    let answers = Answers::from_decoded(ViewId(1), decoded).unwrap();
    let source = answers.project(&doc, ViewId::SOURCE).unwrap();
    let span = source.spans.get(Ix::new(0));
    assert_eq!(&doc.source()[span.start as usize..span.end as usize], "Cafe");
    assert_eq!(source.probabilities, vec![0.8]);
    assert!(answers.project(&doc, ViewId(u16::MAX)).is_err());
    let mut invalid = answers;
    invalid.probabilities[0] = f64::NAN;
    assert!(invalid.check().is_err());
}

#[test]
fn answer_projection_cannot_turn_inserted_text_into_an_empty_answer() {
    let mut doc = Document::new("a");
    let mut builder = ViewBuilder::new(doc.source());
    builder.replace(0, "new");
    let (text, alignment) = builder.finish();
    let view = View::derived(doc.get::<View>().unwrap(), text, alignment);
    doc.add(&Key::named("inserted"), view, Provenance::by("test")).unwrap();
    let mut answers = Answers::from_decoded(
        ViewId(1),
        Answer::Spans {
            answers: vec![SpanAnswer { bytes: 0..3, score: 2., probability: 1. }],
            null_score: 0.,
            null_probability: 0.,
        },
    )
    .unwrap();
    assert!(answers.project(&doc, ViewId::SOURCE).is_err());
    answers.spans = Spans::new(ViewId(1));
    answers.spans.push(Span::new(1, 1));
    assert!(answers.check().is_err());
    assert!(Answers::from_decoded(ViewId(1), Answer::None { null_score: 0. }).unwrap().check().is_ok());
}

#[test]
fn qa_batch_matches_single_for_multiple_windows_and_empty_contexts() {
    let f = fixture();
    let a = ExtractiveQaAnnotator::open(&f.dir, 1, DecodeOptions { null_threshold: -1000.0, ..Default::default() })
        .unwrap();
    let docs = [
        Document::new("Paris is in France."),
        Document::new("The café is in Montréal. ".repeat(20)),
        Document::new(""),
    ];
    let questions = [Question("Where?".into()), Question("Where is the café?".into()), Question("What?".into())];
    let inputs: Vec<_> = docs.iter().zip(&questions).map(|(d, q)| (d.get::<View>().unwrap(), q)).collect();
    let expected: Vec<_> = inputs.iter().map(|&i| a.annotate(i).unwrap().0).collect();
    for (actual, expected) in a.annotate_batch(inputs).into_iter().zip(expected) {
        let actual = actual.unwrap().0;
        assert_eq!(actual.spans.columns(), expected.spans.columns());
        assert_eq!(actual.scores, expected.scores);
        assert_eq!(actual.probabilities, expected.probabilities);
        assert_eq!(actual.null_score, expected.null_score);
        assert_eq!(actual.null_probability, expected.null_probability);
    }
    assert!(a.annotate_batch(Vec::new()).is_empty());
}
