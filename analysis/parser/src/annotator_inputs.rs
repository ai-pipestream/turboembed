//! Input validation and configuration hashing for [`crate::ParserAnnotator`].
//!
//! The annotator walks sentences and tokens with a single forward cursor and
//! slices each token's bytes out of the view, so both layers must be in the
//! annotator's view, in text order without overlaps, and every token span
//! must be a valid UTF-8 range of the text. These helpers check that up front
//! and return a description of the first problem instead of letting the walk
//! skip, repeat, or panic.

use turbo_analysis::{Layer, Span, View};
use turbo_analysis_layers::{Sentences, Tokens};

use crate::model::ParserType;

/// Checks the views, layer invariants and ordering of the parser's inputs.
pub(crate) fn check_inputs(view: &View, sentences: &Sentences, tokens: &Tokens) -> Result<(), String> {
    if tokens.spans.view() != view.id() {
        return Err(format!("the tokens are in view {}, not in view {}", tokens.spans.view().0, view.id().0));
    }
    if sentences.spans.view() != view.id() {
        return Err(format!("the sentences are in view {}, not in view {}", sentences.spans.view().0, view.id().0));
    }
    sentences.check().map_err(|e| format!("invalid sentences: {e}"))?;
    tokens.check().map_err(|e| format!("invalid tokens: {e}"))?;
    if !sentences.spans.is_ordered() {
        return Err("the sentences are not in text order without overlaps".into());
    }
    if !tokens.spans.is_ordered() {
        return Err("the tokens are not in text order without overlaps".into());
    }
    let text = view.text();
    let (starts, ends) = tokens.spans.columns();
    for (i, (&s, &e)) in starts.iter().zip(ends).enumerate() {
        token_text(text, i, Span::new(s, e))?;
    }
    Ok(())
}

/// The bytes of token `i`, or a description of why `span` is not a valid
/// UTF-8 range of `text` (inverted, out of bounds, or inside a character).
pub(crate) fn token_text(text: &str, i: usize, span: Span) -> Result<&str, String> {
    text.get(span.start as usize..span.end as usize).ok_or_else(|| {
        format!("token {i} span {}..{} is not a valid UTF-8 range of a {}-byte text", span.start, span.end, text.len())
    })
}

/// FNV-1a over everything besides the model that changes the parses: the
/// parser type, the beam size and the advance percentage. The percentage is
/// hashed by its bits after adding `0.0`, so `-0.0` and `0.0` agree.
pub(crate) fn config_hash(parser_type: ParserType, beam_size: usize, advance_percentage: f64) -> u64 {
    const OFFSET: u64 = 0xcbf2_9ce4_8422_2325;
    const PRIME: u64 = 0x0000_0100_0000_01b3;
    let kind: u8 = match parser_type {
        ParserType::Chunking => 1,
        ParserType::TreeInsert => 2,
    };
    let advance = (advance_percentage + 0.0).to_bits();
    let mut h = OFFSET;
    for &b in [kind].iter().chain(&(beam_size as u64).to_le_bytes()).chain(&advance.to_le_bytes()) {
        h ^= u64::from(b);
        h = h.wrapping_mul(PRIME);
    }
    h
}

#[cfg(test)]
mod tests {
    use turbo_analysis::{Document, Key, Spans, ViewId};

    use super::*;

    fn spans<L>(view: u16, items: &[(u32, u32)]) -> Spans<L> {
        let mut s = Spans::new(ViewId(view));
        for &(a, b) in items {
            s.push(Span::new(a, b));
        }
        s
    }

    fn sents(view: u16, items: &[(u32, u32)]) -> Sentences {
        Sentences { spans: spans(view, items) }
    }

    fn toks(items: &[(u32, u32)]) -> Tokens {
        Tokens { spans: spans(0, items), sentence: None }
    }

    #[test]
    fn config_hash_tracks_every_search_parameter() {
        let base = config_hash(ParserType::Chunking, 20, 0.95);
        assert_ne!(base, 0);
        assert_eq!(base, config_hash(ParserType::Chunking, 20, 0.95));
        assert_ne!(base, config_hash(ParserType::Chunking, 21, 0.95));
        assert_ne!(base, config_hash(ParserType::Chunking, 20, 0.9));
        assert_ne!(base, config_hash(ParserType::TreeInsert, 20, 0.95));
        assert_eq!(config_hash(ParserType::Chunking, 20, -0.0), config_hash(ParserType::Chunking, 20, 0.0));
    }

    const GOOD_SENTS: [(u32, u32); 2] = [(0, 10), (11, 19)];

    #[test]
    fn inputs_are_refused_before_the_parse_walk() {
        let text = "Zoë runs. It ends.";
        let doc = Document::new(text);
        let view = doc.require(&Key::<View>::DEFAULT).unwrap();
        let good_sents = sents(0, &GOOD_SENTS);
        let good_toks = toks(&[(0, 4), (5, 9), (9, 10), (11, 13), (14, 18), (18, 19)]);
        assert_eq!(check_inputs(view, &good_sents, &good_toks), Ok(()));

        let cases: [(&str, Sentences, Tokens); 7] = [
            ("sentences in another view", sents(1, &[(0, 10)]), toks(&[(0, 4)])),
            ("overlapping sentences", sents(0, &[(0, 10), (5, 19)]), toks(&[(0, 4)])),
            ("unordered sentences", sents(0, &[(11, 19), (0, 10)]), toks(&[(0, 4)])),
            ("overlapping tokens", sents(0, &GOOD_SENTS), toks(&[(0, 4), (2, 9)])),
            ("unordered tokens", sents(0, &GOOD_SENTS), toks(&[(5, 9), (0, 4)])),
            ("token past the end", sents(0, &GOOD_SENTS), toks(&[(0, 4), (14, 40)])),
            ("token inside a character", sents(0, &GOOD_SENTS), toks(&[(0, 3), (5, 9)])),
        ];
        for (what, s, t) in cases {
            assert!(check_inputs(view, &s, &t).is_err(), "{what} was accepted");
        }
        assert!(token_text(text, 0, Span { start: 9, end: 5 }).is_err(), "an inverted span was accepted");
    }
}
