use super::*;
use crate::sentiment::{LexiconSentiment, SENTENCE_SENTIMENT};
use turbo_analysis::{Analyzer, Bound, Document, Key, Provenance, Spans, ViewId};
use turbo_normalize::emoji::EmojiAnnotate;

fn annotate_sentences_and_tokens(doc: &mut Document) {
    let text = doc.source().to_owned();
    let mut s_spans = Spans::new(ViewId::SOURCE);
    let mut t_spans = Spans::new(ViewId::SOURCE);
    let mut t_sent = Vec::new();

    // Split sentences on `. `, `! `, `? ` or end of string.
    let mut sent_start = 0usize;
    let bytes = text.as_bytes();
    let mut s_idx = 0u32;
    let mut i = 0usize;
    while i < bytes.len() {
        let is_end =
            matches!(bytes[i], b'.' | b'!' | b'?') && (i + 1 == bytes.len() || bytes[i + 1].is_ascii_whitespace());
        if is_end || i + 1 == bytes.len() {
            let sent_end = i + 1;
            s_spans.push(Span::new(sent_start as u32, sent_end as u32));
            push_tokens_in_range(&text, sent_start, sent_end, s_idx, &mut t_spans, &mut t_sent);
            s_idx += 1;
            i = sent_end;
            while i < bytes.len() && bytes[i].is_ascii_whitespace() {
                i += 1;
            }
            sent_start = i;
        } else {
            i += 1;
        }
    }

    doc.add(&Key::DEFAULT, Sentences { spans: s_spans }, Provenance::by("test")).unwrap();
    doc.add(&Key::DEFAULT, Tokens { spans: t_spans, sentence: Some(t_sent) }, Provenance::by("test")).unwrap();
}

fn push_tokens_in_range(
    text: &str,
    start: usize,
    end: usize,
    sent_idx: u32,
    t_spans: &mut Spans<Tokens>,
    t_sent: &mut Vec<Ix<Sentences>>,
) {
    let slice = &text[start..end];
    let mut idx = 0usize;
    while idx < slice.len() {
        let ch = slice[idx..].chars().next().unwrap();
        let ch_len = ch.len_utf8();
        if ch.is_whitespace() {
            idx += ch_len;
            continue;
        }
        if ch.is_ascii_punctuation() && ch != '\'' {
            t_spans.push(Span::new((start + idx) as u32, (start + idx + ch_len) as u32));
            t_sent.push(Ix::new(sent_idx));
            idx += ch_len;
            continue;
        }
        if !ch.is_ascii() {
            // Keep non-ASCII emoji sequence together until whitespace or ASCII.
            let tok_start = idx;
            idx += ch_len;
            while idx < slice.len() {
                let c2 = slice[idx..].chars().next().unwrap();
                if c2.is_whitespace() || c2.is_ascii() {
                    break;
                }
                idx += c2.len_utf8();
            }
            t_spans.push(Span::new((start + tok_start) as u32, (start + idx) as u32));
            t_sent.push(Ix::new(sent_idx));
            continue;
        }
        let tok_start = idx;
        idx += ch_len;
        while idx < slice.len() {
            let c2 = slice[idx..].chars().next().unwrap();
            if c2.is_whitespace() || (c2.is_ascii_punctuation() && c2 != '\'') || !c2.is_ascii() {
                break;
            }
            idx += c2.len_utf8();
        }
        t_spans.push(Span::new((start + tok_start) as u32, (start + idx) as u32));
        t_sent.push(Ix::new(sent_idx));
    }
}

#[derive(Debug)]
struct FixedMapVectorScorer {
    weights: Vec<f32>,
    bias: f32,
}

impl PragmaticsVectorScorer for FixedMapVectorScorer {
    fn score_sentence_vector(&self, vec: &[f32]) -> Result<f64, String> {
        if vec.len() != self.weights.len() {
            return Err(format!("dim mismatch: got {}, expected {}", vec.len(), self.weights.len()));
        }
        let mut dot = self.bias as f64;
        for (&x, &w) in vec.iter().zip(&self.weights) {
            dot += (x as f64) * (w as f64);
        }
        Ok(1.0 / (1.0 + (-dot).exp()))
    }

    fn config_hash(&self) -> u64 {
        0x5A2C_A501
    }
}

#[test]
fn riloff_incongruity_detects_positive_sentiment_with_negative_situation_and_stores_spans() {
    let mut doc = Document::new(
        "I just love waiting in line for three hours. Oh great, the server crashed again. The food was great, but the wait was long.",
    );
    annotate_sentences_and_tokens(&mut doc);

    let pipe = Analyzer::builder()
        .given(Key::<Sentences>::DEFAULT)
        .given(Key::<Tokens>::DEFAULT)
        .add(EmojiAnnotate::new())
        .step(LexiconSentiment::new().bound())
        .add(PragmaticsAnnotator::new())
        .build()
        .unwrap();

    let mut docs = [doc];
    pipe.run(&mut docs).remove(0).unwrap();
    let doc = &docs[0];

    let tokens = doc.get::<Tokens>().unwrap();
    let prag = doc.get::<Pragmatics>().unwrap();
    assert_eq!(prag.len(), 3);

    // Sentence 0: "I just love waiting in line for three hours."
    let s0 = prag.get(Ix::new(0));
    assert!(s0.score >= 0.65, "expected high sarcasm score on s0, got {}", s0.score);
    let (pos_sp0, neg_sp0) = prag.evidence_spans(Ix::new(0), tokens).expect("s0 evidence spans");
    let text = doc.source();
    assert!(
        text[pos_sp0.range()].contains("love"),
        "expected positive evidence on 'love', got {:?}",
        &text[pos_sp0.range()]
    );
    assert!(
        text[neg_sp0.range()].contains("waiting in line"),
        "expected negative evidence on 'waiting in line', got {:?}",
        &text[neg_sp0.range()]
    );

    // Sentence 1: "Oh great, the server crashed again."
    let s1 = prag.get(Ix::new(1));
    assert!(s1.score >= 0.65, "expected high sarcasm score on s1, got {}", s1.score);
    let (pos_sp1, neg_sp1) = prag.evidence_spans(Ix::new(1), tokens).expect("s1 evidence spans");
    assert!(
        text[pos_sp1.range()].to_ascii_lowercase().contains("great"),
        "expected positive evidence on 'Oh great', got {:?}",
        &text[pos_sp1.range()]
    );
    assert!(
        text[neg_sp1.range()].contains("crashed"),
        "expected negative evidence on 'server crashed', got {:?}",
        &text[neg_sp1.range()]
    );

    // Sentence 2: "The food was great, but the wait was long." (adversative 'but' -> not sarcasm)
    let s2 = prag.get(Ix::new(2));
    assert_eq!(s2.score, 0.0, "balanced clause with 'but' must not score as sarcasm");
    assert!(s2.evidence.is_none());

    // Verify polarity in `Categories<Sentences>` is untouched (`turbo:pragmatics`
    // is never folded into polarity).
    let pol = doc.get_key::<Categories<Sentences>>(&SENTENCE_SENTIMENT).unwrap();
    assert_eq!(pol.best_category(0), "positive");
}

#[test]
fn text_against_emoji_polarity_mismatch_records_evidence_spans() {
    let mut doc =
        Document::new("This new update is just fantastic 🙄. My flight was cancelled and I lost my luggage 😂.");
    annotate_sentences_and_tokens(&mut doc);

    let pipe = Analyzer::builder()
        .given(Key::<Sentences>::DEFAULT)
        .given(Key::<Tokens>::DEFAULT)
        .add(EmojiAnnotate::new())
        .step(LexiconSentiment::new().bound())
        .step(Bound::new(PragmaticsAnnotator::new()).read(4, SENTENCE_SENTIMENT.name))
        .build()
        .unwrap();

    let mut docs = [doc];
    pipe.run(&mut docs).remove(0).unwrap();
    let doc = &docs[0];

    let tokens = doc.get::<Tokens>().unwrap();
    let prag = doc.get::<Pragmatics>().unwrap();
    let text = doc.source();

    // Sentence 0: Positive text ("just fantastic") + negative emoji ("🙄").
    let s0 = prag.get(Ix::new(0));
    assert!(s0.score >= 0.65, "s0 score {}", s0.score);
    let (pos0, neg0) = prag.evidence_spans(Ix::new(0), tokens).unwrap();
    assert!(text[pos0.range()].contains("fantastic"), "got {:?}", &text[pos0.range()]);
    assert_eq!(&text[neg0.range()], "🙄");

    // Sentence 1: Negative situation ("flight was cancelled" / "lost my luggage") + positive emoji ("😂").
    let s1 = prag.get(Ix::new(1));
    assert!(s1.score >= 0.65, "s1 score {}", s1.score);
    let (pos1, neg1) = prag.evidence_spans(Ix::new(1), tokens).unwrap();
    assert_eq!(&text[pos1.range()], "😂");
    assert!(
        text[neg1.range()].contains("cancelled") || text[neg1.range()].contains("luggage"),
        "unexpected neg1 span {:?}",
        &text[neg1.range()]
    );
}

#[test]
fn embedding_head_scores_implicit_irony_with_none_evidence() {
    let mut doc = Document::new("How predictably bureaucratic. Everything went according to plan.");
    annotate_sentences_and_tokens(&mut doc);

    // Sentence 0 vector points along the sarcasm direction; sentence 1 opposite.
    let vecs = Vectors::<Sentences>::new(
        2,
        vec![2.5f32, 1.0, -2.0, -1.0],
        Vec::new(),
        ModelRef { id: "test".into(), revision: "0".into(), sha256: "0".into() },
    );
    doc.add(&Key::DEFAULT, vecs, Provenance::by("test")).unwrap();

    let scorer = Arc::new(FixedMapVectorScorer { weights: vec![1.0, 1.0], bias: 0.0 });
    let annotator = PragmaticsAnnotator::new().with_vector_scorer(scorer);

    let pipe = Analyzer::builder()
        .given(Key::<Sentences>::DEFAULT)
        .given(Key::<Tokens>::DEFAULT)
        .given(Key::<Vectors<Sentences>>::DEFAULT)
        .add(annotator)
        .build()
        .unwrap();

    let mut docs = [doc];
    pipe.run(&mut docs).remove(0).unwrap();
    let prag = docs[0].get::<Pragmatics>().unwrap();

    // Sentence 0: scored positive by the embedding head alone (`evidence == None`).
    let s0 = prag.get(Ix::new(0));
    assert!(s0.score > 0.90, "expected head score > 0.90, got {}", s0.score);
    assert!(s0.evidence.is_none(), "head-only positive must have None evidence");

    // Sentence 1: scored negative by the head.
    let s1 = prag.get(Ix::new(1));
    assert!(s1.score < 0.10, "expected head score < 0.10, got {}", s1.score);
    assert!(s1.evidence.is_none());
}

#[test]
fn an_empty_vector_row_gets_no_head_score() {
    let mut doc = Document::new("Meeting notes.");
    annotate_sentences_and_tokens(&mut doc);
    let vecs = Vectors::<Sentences>::new(
        2,
        vec![0.0f32, 0.0],
        vec![Ix::new(0)],
        ModelRef { id: "test".into(), revision: "0".into(), sha256: "0".into() },
    );
    doc.add(&Key::DEFAULT, vecs, Provenance::by("test")).unwrap();
    // A biased scorer that says 0.99 to anything.
    let scorer = Arc::new(FixedMapVectorScorer { weights: vec![0.0, 0.0], bias: 4.6 });
    let pipe = Analyzer::builder()
        .given(Key::<Sentences>::DEFAULT)
        .given(Key::<Tokens>::DEFAULT)
        .given(Key::<Vectors<Sentences>>::DEFAULT)
        .add(PragmaticsAnnotator::new().with_vector_scorer(scorer))
        .build()
        .unwrap();
    let mut docs = [doc];
    pipe.run(&mut docs).remove(0).unwrap();
    let prag = docs[0].get::<Pragmatics>().unwrap();
    assert_eq!(prag.get(Ix::new(0)).score, 0.0);
    assert!(prag.get(Ix::new(0)).evidence.is_none());
}

#[derive(Debug)]
struct NamedScorer(ModelRef);

impl PragmaticsVectorScorer for NamedScorer {
    fn score_sentence_vector(&self, _vec: &[f32]) -> Result<f64, String> {
        Ok(0.99)
    }
    fn config_hash(&self) -> u64 {
        1
    }
    fn model_ref(&self) -> Option<ModelRef> {
        Some(self.0.clone())
    }
}

#[test]
fn vectors_from_another_encoder_are_refused() {
    let mut doc = Document::new("Meeting notes.");
    annotate_sentences_and_tokens(&mut doc);
    let vecs = Vectors::<Sentences>::new(
        2,
        vec![1.0f32, 1.0],
        Vec::new(),
        ModelRef { id: "encoder-B".into(), revision: "0".into(), sha256: "b".into() },
    );
    doc.add(&Key::DEFAULT, vecs, Provenance::by("test")).unwrap();
    let scorer = Arc::new(NamedScorer(ModelRef { id: "encoder-A".into(), revision: "0".into(), sha256: "a".into() }));
    let pipe = Analyzer::builder()
        .given(Key::<Sentences>::DEFAULT)
        .given(Key::<Tokens>::DEFAULT)
        .given(Key::<Vectors<Sentences>>::DEFAULT)
        .add(PragmaticsAnnotator::new().with_vector_scorer(scorer))
        .build()
        .unwrap();
    let mut docs = [doc];
    let err = pipe.run(&mut docs).remove(0).unwrap_err();
    assert!(err.to_string().contains("encoder-A"), "{err}");
}

mod benchmark;
