use std::sync::Arc;

use turbo_analysis::{Annotator, Document, IxRange, Span, Spans, View, ViewId};
use turbo_analysis_layers::{IncivilityLabel, IncivilityScores, Sentences, Tokens};
use turbo_doccat::sentiment::{
    IncivilityAnnotator, IncivilityLexicon, IncivilityVectorScorer, LexiconSentiment, PragmaticsAnnotator, VaderLexicon,
};

fn spans<L>(start: u32, end: u32) -> Spans<L> {
    let mut spans = Spans::new(ViewId::SOURCE);
    spans.push(Span::new(start, end));
    spans
}

#[test]
fn incivility_digest_distinguishes_sub_micro_weights() {
    let mut a = IncivilityLexicon::empty();
    let mut b = IncivilityLexicon::empty();
    a.insert("rude", IncivilityLabel::Insult, 0.5000001);
    b.insert("rude", IncivilityLabel::Insult, 0.5000002);
    assert_ne!(a.sha256(), b.sha256());
}

#[test]
fn invalid_incivility_ranges_return_errors() {
    let ann = IncivilityAnnotator::new().with_reply_context(true);
    let words = vec!["hello".to_owned()];
    for (start, end, parent) in [(0, 2, None), (1, 0, None), (0, 1, Some((0, 2))), (0, 1, Some((1, 0)))] {
        assert!(ann.score_unit(&words, start, end, parent, None).is_err());
    }
}

#[test]
fn invalid_incivility_thresholds_return_errors() {
    for threshold in [f64::NAN, f64::INFINITY, -0.1, 1.2] {
        let mut ann = IncivilityAnnotator::new();
        ann.threshold = threshold;
        assert!(ann.score_unit(&["hello".into()], 0, 1, None, None).is_err());
    }
}

struct BadHead(f64);
impl IncivilityVectorScorer for BadHead {
    fn score_incivility(&self, _: &[f32]) -> Result<IncivilityScores, String> {
        Ok(IncivilityScores { insult: self.0, ..IncivilityScores::default() })
    }
}

#[test]
fn incivility_rejects_invalid_head_scores() {
    for value in [f64::NAN, f64::INFINITY, -0.1, 1.1] {
        let ann = IncivilityAnnotator::new().with_vector_scorer(Arc::new(BadHead(value)));
        assert!(ann.score_unit(&["hello".into()], 0, 1, None, Some(&[1.0])).is_err());
    }
}

#[test]
fn pragmatics_rejects_non_finite_configuration() {
    let doc = Document::new("hello");
    let view = doc.get::<View>().unwrap();
    let sentences = Sentences { spans: spans(0, 5) };
    let tokens = Tokens { spans: spans(0, 5), sentence: None };
    for value in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        for ann in [
            PragmaticsAnnotator::new().with_positive_threshold(value),
            PragmaticsAnnotator::new().with_polarity_thresholds(value, 0.4),
        ] {
            assert!(ann.score_document(view, &sentences, &tokens, None, None, None, None).is_err());
        }
    }
}

#[test]
fn lexicon_annotator_rejects_split_utf8_token() {
    let doc = Document::new("é");
    let sentences = Sentences { spans: spans(0, 2) };
    let tokens = Tokens { spans: spans(1, 2), sentence: None };
    assert!(LexiconSentiment::new().annotate((doc.get::<View>().unwrap(), &sentences, &tokens, None, None)).is_err());
}

#[test]
fn pragmatics_rejects_split_utf8_token() {
    let doc = Document::new("é");
    let sentences = Sentences { spans: spans(0, 2) };
    let tokens = Tokens { spans: spans(1, 2), sentence: None };
    assert!(
        PragmaticsAnnotator::new()
            .score_document(doc.get::<View>().unwrap(), &sentences, &tokens, None, None, None, None)
            .is_err()
    );
}

#[test]
fn incivility_rejects_split_utf8_token() {
    let doc = Document::new("é");
    let sentences = Sentences { spans: spans(0, 2) };
    let tokens = Tokens { spans: spans(1, 2), sentence: None };
    assert!(
        IncivilityAnnotator::new()
            .annotate((doc.get::<View>().unwrap(), &tokens, Some(&sentences), None, None, None, None))
            .is_err()
    );
}

#[test]
fn invalid_lexicon_updates_preserve_valid_entries() {
    let mut vader = VaderLexicon::from_tsv("good\t2").unwrap();
    let before = vader.sha256_hex().to_owned();
    vader.insert("good", f64::NAN);
    vader.insert("", 1.0);
    assert_eq!(vader.sha256_hex(), before);
    let mut incivility = IncivilityLexicon::empty();
    incivility.insert("rude", IncivilityLabel::Insult, 0.5);
    let before = incivility.sha256().to_owned();
    incivility.insert("rude", IncivilityLabel::Insult, f64::NAN);
    assert_eq!(incivility.sha256(), before);
}

#[test]
fn large_finite_vader_score_normalizes_without_overflow() {
    assert_eq!(turbo_doccat::sentiment::lexicon::normalize_score(1e200, 15.0), 1.0);
    assert_eq!(turbo_doccat::sentiment::lexicon::normalize_score(-1e200, 15.0), -1.0);
}

#[test]
fn valid_custom_lexicon_updates_keep_last_value() {
    let mut lex = VaderLexicon::from_tsv("good\t1\ngood\t2").unwrap();
    lex.insert("good", 3.0);
    assert_eq!(lex.get("good"), Some(3.0));
    let score = LexiconSentiment::with_lexicon(Arc::new(lex));
    let tokens = Tokens { spans: spans(0, 4), sentence: None };
    assert!(
        score.score_sentence("good", Span::new(0, 4), &tokens, IxRange::new(0, 1), None, None).score.compound > 0.0
    );
}

#[test]
fn pragmatics_rejects_misaligned_sentence_column() {
    let doc = Document::new("good bad");
    let sentences = Sentences { spans: spans(0, 4) };
    let tokens = Tokens { spans: spans(5, 8), sentence: Some(vec![turbo_analysis::Ix::new(0)]) };
    assert!(
        PragmaticsAnnotator::new()
            .score_document(doc.get::<View>().unwrap(), &sentences, &tokens, None, None, None, None)
            .is_err()
    );
}

#[test]
fn incivility_rejects_short_vector_layer() {
    use turbo_analysis::ModelRef;
    use turbo_analysis_layers::Vectors;
    let doc = Document::new("hello");
    let sentences = Sentences { spans: spans(0, 5) };
    let tokens = Tokens { spans: spans(0, 5), sentence: None };
    let vectors = Vectors::<Sentences>::new(
        2,
        vec![],
        vec![],
        ModelRef { id: "test".into(), revision: "1".into(), sha256: "00".into() },
    );
    assert!(
        IncivilityAnnotator::new()
            .annotate((doc.get::<View>().unwrap(), &tokens, Some(&sentences), None, None, Some(&vectors), None))
            .is_err()
    );
}

#[test]
fn nested_chunks_score_their_own_tokens() {
    use turbo_analysis::{Ix, Layer};
    use turbo_analysis_layers::{Chunks, Cut};
    use turbo_doccat::sentiment::IncivilityUnit;
    let doc = Document::new("idiot");
    let tokens = Tokens { spans: spans(0, 5), sentence: None };
    let mut chunk_spans = spans(0, 5);
    chunk_spans.push(Span::new(0, 5));
    let chunks = Chunks {
        spans: chunk_spans,
        level: vec![1, 0],
        parent: vec![None, Some(Ix::new(0))],
        sentences: vec![IxRange::new(0, 0); 2],
        cut: vec![Cut::Sentence; 2],
        cost: vec![0; 2],
        model_tokens: vec![1; 2],
        over_budget: vec![false; 2],
    };
    chunks.check().unwrap();
    let (out,) = IncivilityAnnotator::new()
        .with_unit(IncivilityUnit::Chunks)
        .annotate((doc.get::<View>().unwrap(), &tokens, None, Some(&chunks), None, None, None))
        .unwrap();
    assert_eq!(out.len(), 2);
    assert_eq!(out.scores(Ix::new(0)).insult, out.scores(Ix::new(1)).insult);
}

#[test]
fn discontinuous_sentences_are_not_scored_as_their_hull() {
    let doc = Document::new("a idiot b");
    let mut sentence_spans = Spans::new(ViewId::SOURCE);
    sentence_spans.push_fragments(&[Span::new(0, 1), Span::new(8, 9)]);
    let sentences = Sentences { spans: sentence_spans };
    let tokens = Tokens { spans: spans(2, 7), sentence: None };
    let view = doc.get::<View>().unwrap();
    assert!(IncivilityAnnotator::new().annotate((view, &tokens, Some(&sentences), None, None, None, None)).is_err());
    assert!(LexiconSentiment::new().annotate((view, &sentences, &tokens, None, None)).is_err());
    assert!(PragmaticsAnnotator::new().score_document(view, &sentences, &tokens, None, None, None, None).is_err());
}
