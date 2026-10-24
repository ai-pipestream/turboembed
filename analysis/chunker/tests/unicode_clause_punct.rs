//! Unicode clause punctuation with text only (#877): curly and
//! guillemet quotes close, fullwidth and Arabic semicolons are
//! hard clause marks.

use turbo_analysis::{Document, Span, Spans, View, ViewId};
use turbo_analysis_layers::{Sentences, Tokens};
use turbo_chunker::{ClauseBreak, ClauseCosts, clause_cuts};

fn cuts(text: &str) -> Vec<ClauseBreak> {
    let doc = Document::new(text);
    let view: View = doc.get::<View>().unwrap().clone();
    let mut s_spans = Spans::new(ViewId::SOURCE);
    s_spans.push(Span::new(0, text.len() as u32));
    let sentences = Sentences { spans: s_spans };
    // Whitespace tokens, punctuation attached as written.
    let mut t_spans = Spans::new(ViewId::SOURCE);
    let mut at = None;
    for (i, c) in text.char_indices().chain([(text.len(), ' ')]) {
        match (c.is_whitespace(), at) {
            (false, None) => at = Some(i),
            (true, Some(s)) => {
                t_spans.push(Span::new(s as u32, i as u32));
                at = None;
            }
            _ => {}
        }
    }
    let n = t_spans.len();
    let tokens = Tokens { spans: t_spans, sentence: Some(vec![turbo_analysis::Ix::new(0); n]) };
    clause_cuts(&sentences, &tokens, None, None, None, None, Some(&view), ClauseCosts::default())
        .sentences
        .into_iter()
        .next()
        .unwrap()
        .breaks
}

fn kinds(breaks: &[ClauseBreak]) -> Vec<&'static str> {
    breaks
        .iter()
        .map(|b| match b {
            ClauseBreak::Clause(_) => "clause",
            ClauseBreak::Phrase(_) => "phrase",
            ClauseBreak::Token(_) => "token",
            ClauseBreak::Held => "held",
        })
        .collect()
}

#[test]
fn curly_and_guillemet_quotes_behave_like_ascii_quotes() {
    let ascii = kinds(&cuts("\"He said\" then left"));
    let curly = kinds(&cuts("\u{201C}He said\u{201D} then left"));
    assert_eq!(curly, ascii, "curly quotes differ from ASCII quotes");
    let french = kinds(&cuts("\u{00AB}Il dit\u{00BB} puis parti"));
    assert_eq!(french, ascii, "guillemets differ from ASCII quotes");
}

#[test]
fn fullwidth_and_arabic_semicolons_are_hard_clause_marks() {
    let ascii = kinds(&cuts("yes but ; no more"));
    assert_eq!(kinds(&cuts("yes but \u{FF1B} no more")), ascii);
    assert_eq!(kinds(&cuts("yes but \u{061B} no more")), ascii);
    assert!(ascii.contains(&"clause"), "semicolon is a clause cut: {ascii:?}");
}
