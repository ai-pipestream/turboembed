//! The units a tokenizer runs over, and the `Tokens` layer it fills.
//!
//! Every tokenizer annotator reads a view and, when the document has one,
//! a `Sentences` layer. With sentences, each sentence is tokenized on its
//! own and every token records its sentence; without, the whole view is
//! one unit and the layer has no sentence column. This module is the one
//! place that does this, so every tokenizer handles sentences the same
//! way.

use turbo_analysis::{Error, Ix, Span, Spans, View};
use turbo_analysis_layers::{Sentences, Tokens};

/// Runs `tokenize` over each unit of `view` (each sentence, or the whole
/// text) and collects the spans it reports into a `Tokens` layer.
///
/// `tokenize` is handed the unit's text, the byte offset of the unit in
/// the view, and a buffer to append spans to. The spans it appends are
/// relative to the unit's text; they are moved to view offsets here. The
/// buffer is cleared before each unit and reused, so nothing is allocated
/// per unit. `annotator` names the caller in an error.
///
/// Sentences in another view than `view` are refused: their offsets do
/// not index this text.
pub fn tokenize_units<F>(
    annotator: &'static str,
    view: &View,
    sentences: Option<&Sentences>,
    mut tokenize: F,
) -> Result<Tokens, Error>
where
    F: FnMut(&str, u32, &mut Vec<Span>),
{
    let text = view.text();
    let mut scratch = Vec::new();
    let mut spans = Spans::<Tokens>::with_capacity(view.id(), text.len() / 6);
    let Some(sentences) = sentences else {
        tokenize(text, 0, &mut scratch);
        for &s in &scratch {
            debug_assert!(s.end as usize <= text.len());
            spans.push(s);
        }
        return Ok(Tokens { spans, sentence: None });
    };
    if sentences.spans.view() != view.id() {
        return Err(Error::Annotator {
            annotator,
            detail: format!(
                "the sentences are in view {}, not in view {} that is tokenized",
                sentences.spans.view().0,
                view.id().0
            ),
        });
    }
    let mut column: Vec<Ix<Sentences>> = Vec::with_capacity(spans.len());
    for (ix, unit) in sentences.spans.iter() {
        scratch.clear();
        tokenize(&text[unit.range()], unit.start, &mut scratch);
        for &s in &scratch {
            debug_assert!(s.end <= unit.len());
            spans.push(Span { start: s.start + unit.start, end: s.end + unit.start });
            column.push(ix);
        }
    }
    Ok(Tokens { spans, sentence: Some(column) })
}

#[cfg(test)]
mod tests {
    use super::*;
    use turbo_analysis::Document;

    /// A tokenizer that reports every maximal run of ASCII letters.
    fn letters(text: &str, _base: u32, out: &mut Vec<Span>) {
        let mut start = None;
        for (i, b) in text.bytes().enumerate() {
            match (b.is_ascii_alphabetic(), start) {
                (true, None) => start = Some(i as u32),
                (false, Some(s)) => {
                    out.push(Span::new(s, i as u32));
                    start = None;
                }
                _ => {}
            }
        }
        if let Some(s) = start {
            out.push(Span::new(s, text.len() as u32));
        }
    }

    fn spans(tokens: &Tokens) -> Vec<(u32, u32)> {
        tokens.spans.iter().map(|(_, s)| (s.start, s.end)).collect()
    }

    #[test]
    fn whole_view_without_sentences() {
        let doc = Document::new("ab cd. ef");
        let view = doc.get::<View>().unwrap();
        let tokens = tokenize_units("test", view, None, letters).unwrap();
        assert_eq!(spans(&tokens), [(0, 2), (3, 5), (7, 9)]);
        assert!(tokens.sentence.is_none());
    }

    #[test]
    fn per_sentence_with_offsets_and_column() {
        let doc = Document::new("ab cd. ef gh");
        let view = doc.get::<View>().unwrap();
        let mut s = Spans::new(view.id());
        s.push(Span::new(0, 6));
        s.push(Span::new(7, 12));
        let sentences = Sentences { spans: s };
        let mut bases = Vec::new();
        let tokens = tokenize_units("test", view, Some(&sentences), |text, base, out| {
            bases.push(base);
            letters(text, base, out)
        })
        .unwrap();
        assert_eq!(bases, [0, 7]);
        assert_eq!(spans(&tokens), [(0, 2), (3, 5), (7, 9), (10, 12)]);
        let column: Vec<usize> = tokens.sentence.as_ref().unwrap().iter().map(|i| i.get()).collect();
        assert_eq!(column, [0, 0, 1, 1]);
        assert_eq!(tokens.of(Ix::new(1)), turbo_analysis::IxRange::new(2, 4));
    }

    #[test]
    fn text_outside_sentences_is_not_tokenized() {
        let doc = Document::new("skip ab skip");
        let view = doc.get::<View>().unwrap();
        let mut s = Spans::new(view.id());
        s.push(Span::new(5, 7));
        let tokens = tokenize_units("test", view, Some(&Sentences { spans: s }), letters).unwrap();
        assert_eq!(spans(&tokens), [(5, 7)]);
    }

    #[test]
    fn sentences_in_another_view_are_refused() {
        let doc = Document::new("ab");
        let view = doc.get::<View>().unwrap();
        let sentences = Sentences { spans: Spans::new(turbo_analysis::ViewId(3)) };
        let err = tokenize_units("test", view, Some(&sentences), letters).err().unwrap();
        assert_eq!(
            err,
            Error::Annotator {
                annotator: "test",
                detail: "the sentences are in view 3, not in view 0 that is tokenized".into()
            }
        );
    }
}
