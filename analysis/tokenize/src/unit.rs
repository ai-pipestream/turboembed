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

/// One token as a unit tokenizer reports it: a span relative to the
/// unit's text and the value that goes with it, if any.
///
/// A bare [`Span`] carries no value (`()`), so a rule tokenizer appends
/// plain spans and [`tokenize_units_with`] has nothing extra to move per
/// token. `(Span, V)` carries a `V`, such as the maxent tokenizer's
/// probability.
pub trait UnitToken: Copy {
    /// The value that comes with each token.
    type Value;
    /// The token's span and its value.
    fn into_parts(self) -> (Span, Self::Value);
}

impl UnitToken for Span {
    type Value = ();
    #[inline]
    fn into_parts(self) -> (Span, ()) {
        (self, ())
    }
}

impl<V: Copy> UnitToken for (Span, V) {
    type Value = V;
    #[inline]
    fn into_parts(self) -> (Span, V) {
        self
    }
}

/// Runs `tokenize` over each unit of `view` (each sentence, or the whole
/// text) and collects the spans it reports into a `Tokens` layer.
///
/// This is [`tokenize_units_with`] for a tokenizer that reports plain
/// spans and no value per token.
pub fn tokenize_units<F>(
    annotator: &'static str,
    view: &View,
    sentences: Option<&Sentences>,
    tokenize: F,
) -> Result<Tokens, Error>
where
    F: FnMut(&str, u32, &mut Vec<Span>),
{
    tokenize_units_with(annotator, view, sentences, tokenize, |()| {})
}

/// Runs `tokenize` over each unit of `view` (each sentence, or the whole
/// text), collects the spans it reports into a `Tokens` layer and hands
/// each token's value to `value`, in token order.
///
/// `tokenize` is handed the unit's text, the byte offset of the unit in
/// the view, and a buffer to append tokens to. The spans it appends are
/// relative to the unit's text; they are moved to view offsets here. The
/// buffer is cleared before each unit and reused, so nothing is allocated
/// per unit. `value` is called once per token, after its span is pushed,
/// so a caller can fill a column aligned to the tokens. `annotator` names
/// the caller in an error.
///
/// Sentences in another view than `view` are refused: their offsets do
/// not index this text.
pub fn tokenize_units_with<T, F, G>(
    annotator: &'static str,
    view: &View,
    sentences: Option<&Sentences>,
    mut tokenize: F,
    mut value: G,
) -> Result<Tokens, Error>
where
    T: UnitToken,
    F: FnMut(&str, u32, &mut Vec<T>),
    G: FnMut(T::Value),
{
    let text = view.text();
    let mut scratch: Vec<T> = Vec::new();
    let mut spans = Spans::<Tokens>::with_capacity(view.id(), text.len() / 6);
    let Some(sentences) = sentences else {
        tokenize(text, 0, &mut scratch);
        for &t in &scratch {
            let (s, v) = t.into_parts();
            debug_assert!(s.end as usize <= text.len());
            spans.push(s);
            value(v);
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
    let mut column: Vec<Ix<Sentences>> = Vec::with_capacity(text.len() / 6);
    for (ix, unit) in sentences.spans.iter() {
        let slice = text.get(unit.range()).ok_or_else(|| Error::Annotator {
            annotator,
            detail: format!(
                "sentence {} span {}..{} is out of bounds or not on a UTF-8 char boundary (text len {})",
                ix.get(),
                unit.start,
                unit.end,
                text.len()
            ),
        })?;
        scratch.clear();
        tokenize(slice, unit.start, &mut scratch);
        for &t in &scratch {
            let (s, v) = t.into_parts();
            debug_assert!(s.end <= unit.len());
            spans.push(Span { start: s.start + unit.start, end: s.end + unit.start });
            column.push(ix);
            value(v);
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
    fn values_follow_their_tokens() {
        // Each token's value is its length; the values come out in token
        // order, across sentences, aligned to the spans.
        let doc = Document::new("ab cde. f gh");
        let view = doc.get::<View>().unwrap();
        let mut s = Spans::new(view.id());
        s.push(Span::new(0, 7));
        s.push(Span::new(8, 12));
        let sentences = Sentences { spans: s };
        let mut values = Vec::new();
        let tokens = tokenize_units_with(
            "test",
            view,
            Some(&sentences),
            |text, base, out: &mut Vec<(Span, f64)>| {
                let mut found = Vec::new();
                letters(text, base, &mut found);
                out.extend(found.into_iter().map(|s| (s, f64::from(s.len()))));
            },
            |v| values.push(v),
        )
        .unwrap();
        assert_eq!(spans(&tokens), [(0, 2), (3, 6), (8, 9), (10, 12)]);
        assert_eq!(values, [2.0, 3.0, 1.0, 2.0]);
        let column: Vec<usize> = tokens.sentence.as_ref().unwrap().iter().map(|i| i.get()).collect();
        assert_eq!(column, [0, 0, 1, 1]);
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

    #[test]
    fn sentences_outside_view_text_are_refused() {
        let doc = Document::new("a\u{e9}b");
        let view = doc.get::<View>().unwrap();
        // Out of bounds of the 4-byte view text.
        let mut s = Spans::new(view.id());
        s.push(Span::new(0, 10));
        let err = tokenize_units("test", view, Some(&Sentences { spans: s }), letters).err().unwrap();
        assert_eq!(
            err,
            Error::Annotator {
                annotator: "test",
                detail: "sentence 0 span 0..10 is out of bounds or not on a UTF-8 char boundary (text len 4)".into()
            }
        );
        // Inside the 2-byte UTF-8 sequence of U+00E9 at bytes 1..3.
        let mut mid = Spans::new(view.id());
        mid.push(Span::new(0, 2));
        let err = tokenize_units("test", view, Some(&Sentences { spans: mid }), letters).err().unwrap();
        assert_eq!(
            err,
            Error::Annotator {
                annotator: "test",
                detail: "sentence 0 span 0..2 is out of bounds or not on a UTF-8 char boundary (text len 4)".into()
            }
        );
    }
}
