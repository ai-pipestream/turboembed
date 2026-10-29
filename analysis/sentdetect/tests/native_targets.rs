//! Our boundary supervision contract: exact spans and end labels, no model oracle.

#[path = "support/native_targets.rs"]
mod targets;

use turbo_analysis::{Analyzer, Annotator, Document, Key, OffsetTable, Span, Status};
use turbo_analysis_layers::Sentences;
use turbo_sentdetect::{RuleScratch, RuleSentenceDetector, RuleSplitter};
use turbo_unicode::props::white_space;

fn expected(target: &targets::Target) -> Vec<Span> {
    let mut cursor = 0;
    let mut spans = Vec::new();
    for part in target.sentences {
        assert!(!part.is_empty());
        let start = cursor + target.text[cursor..].find(part).expect(target.name);
        assert!(target.text[cursor..start].chars().all(white_space), "{}", target.name);
        let end = start + part.len();
        spans.push(Span::new(start as u32, end as u32));
        cursor = end;
    }
    assert!(target.text[cursor..].chars().all(white_space), "{}", target.name);
    spans
}

#[derive(Debug, PartialEq, Eq)]
enum Boundary {
    Continue,
    End,
}

fn labels(text: &str, spans: &[Span]) -> Vec<Boundary> {
    text.char_indices()
        .map(|(byte, ch)| {
            if spans.iter().any(|s| s.end as usize == byte + ch.len_utf8()) {
                Boundary::End
            } else {
                Boundary::Continue
            }
        })
        .collect()
}

#[test]
fn project_targets_match_direct_and_pipeline_boundaries() {
    for target in targets::TARGETS {
        let expected = expected(target);
        let mut got = Vec::new();
        RuleSplitter::new(target.language).split(target.text, &mut got);
        assert_eq!(got, expected, "{}", target.name);
        assert_eq!(labels(target.text, &got), labels(target.text, &expected), "{}", target.name);

        let detector = RuleSentenceDetector::new(target.language);
        let hash = detector.config_hash();
        let analyzer = Analyzer::builder().add(detector).build().unwrap();
        // Interleave empty and different documents to exercise per-thread state reuse.
        let mut docs = vec![Document::new("Other text!"), Document::new(""), Document::new(target.text)];
        assert!(analyzer.run(&mut docs).iter().all(Result::is_ok));
        let doc = &docs[2];
        let sentences = doc.get::<Sentences>().unwrap();
        assert_eq!(sentences.spans.iter().map(|(_, s)| s).collect::<Vec<_>>(), expected, "{}", target.name);
        assert_eq!(doc.status(Key::<Sentences>::DEFAULT.id()), Some(&Status::Complete));
        let provenance = doc.provenance(Key::<Sentences>::DEFAULT.id()).unwrap();
        assert_eq!(provenance.config, hash);
        assert!(provenance.model.is_none());
        let offsets = OffsetTable::build(target.text);
        for span in expected {
            assert_eq!(offsets.span_from_utf16(target.text, offsets.span_to_utf16_typed(target.text, span)), Ok(span));
            assert_eq!(
                offsets.span_from_code_point(target.text, offsets.span_to_code_point_typed(target.text, span)),
                Ok(span)
            );
        }
    }
}

#[test]
fn padding_translates_targets_without_changing_labels() {
    let mut scratch = RuleScratch::default();
    for target in targets::TARGETS {
        for padding in [" ", "\t", "\u{a0}", "\u{3000}", "\r\n"] {
            let text = format!("{padding}{}{padding}", target.text);
            let expected: Vec<_> = expected(target)
                .into_iter()
                .map(|s| Span::new(s.start + padding.len() as u32, s.end + padding.len() as u32))
                .collect();
            let mut got = Vec::new();
            RuleSplitter::new(target.language).split_with(&text, &mut scratch, &mut got);
            assert_eq!(got, expected, "{} padding {padding:?}", target.name);
        }
    }
}
