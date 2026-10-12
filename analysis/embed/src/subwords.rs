//! The subword pieces of a text view, from the embedding model's own
//! tokenizer.

use std::path::Path;

use turbo::bundle::Bundle;
use turbo::manifest::{PromptRole, Truncation};
use turbo::tokenizer::{Encode, Tokenizer};
use turbo_analysis::annotate::Step;
use turbo_analysis::registry::{Provider, Spec};
use turbo_analysis::{Annotator, Bound, Error, Ix, ModelRef, Refs, Span, Spans, View};
use turbo_analysis_layers::{Subwords, Tokens};

/// Writes `opennlp:subwords` over one text view with a bundle's tokenizer.
///
/// The whole view is encoded in one row, never truncated, so the layer
/// covers all of it however long it is, past the model's `max_seq`. The
/// row holds only the text's own pieces: no special tokens from the
/// template and no prompt prefix, since neither is in the text. What they
/// add to every row the model sees is `overhead`, for the chunker's
/// budget.
///
/// A byte-level BPE piece holding part of a character spans the whole
/// character, as the tokenizer gives it, so two pieces can share a span
/// and every span stays on character boundaries.
///
/// When the document has word tokens in the same view, each piece also
/// gets the token it belongs to (`align` says which).
pub struct SubwordAnnotator {
    tokenizer: Tokenizer,
    model: ModelRef,
}

impl SubwordAnnotator {
    /// The bundle at `dir`, its tokenizer checked against the bundle's
    /// reference ids as the core checks it on every load.
    pub fn open(dir: &Path) -> Result<SubwordAnnotator, turbo::status::Error> {
        let bundle = Bundle::open(dir)?;
        let tokenizer = Tokenizer::load(&bundle)?;
        Ok(SubwordAnnotator::new(tokenizer, &bundle))
    }

    /// A tokenizer already built from `bundle`, which names the model in
    /// every layer's provenance: its id, its revision and the SHA-256 of
    /// its manifest.
    pub fn new(tokenizer: Tokenizer, bundle: &Bundle) -> SubwordAnnotator {
        SubwordAnnotator { tokenizer, model: crate::model_ref(bundle) }
    }

    pub fn tokenizer(&self) -> &Tokenizer {
        &self.tokenizer
    }

    /// The tokens every row of this model carries besides the text's own
    /// pieces: the template's special tokens and, for `prompt`, the
    /// prefix's tokens, counted on the prefix alone. A byte-level BPE or
    /// Unigram prefix can merge with the first piece of the text, so a
    /// row can come out a token shorter than the pieces plus this.
    pub fn overhead(&self, prompt: PromptRole) -> u32 {
        self.tokenizer.count("", prompt) as u32
    }

    /// The pieces of a view's text, with their spans in it.
    pub fn pieces(&self, view: &View) -> Result<Subwords, Error> {
        let text = view.text();
        let opts = Encode {
            add_special_tokens: false,
            truncation: Truncation::None,
            max_tokens: u32::MAX,
            prompt: PromptRole::None,
        };
        let (mut ids, mut spans) = (Vec::new(), Vec::new());
        self.tokenizer
            .encode_spans(text, opts, &mut ids, &mut spans)
            .map_err(|e| Error::Annotator { annotator: Self::ID, detail: e.to_string() })?;
        let mut columns = Spans::with_capacity(view.id(), spans.len());
        for &[start, end] in &spans {
            columns.push(Span::new(start, end));
        }
        Ok(Subwords { spans: columns, ids, model: self.model.clone(), token: None })
    }
}

impl Annotator for SubwordAnnotator {
    const ID: &'static str = "turbo:subwords";
    type Reads = (View, Option<Tokens>);
    type Writes = (Subwords,);

    fn annotate(&self, (view, tokens): Refs<'_, Self>) -> Result<(Subwords,), Error> {
        let mut sub = self.pieces(view)?;
        if let Some(t) = tokens {
            if t.spans.view() != view.id() {
                return Err(Error::Annotator {
                    annotator: Self::ID,
                    detail: format!("the tokens are in view {}, the text in view {}", t.spans.view().0, view.id().0),
                });
            }
            if !t.spans.is_ordered() {
                return Err(Error::Annotator {
                    annotator: Self::ID,
                    detail: "the tokens are not in text order, or overlap".into(),
                });
            }
            sub.token = align(&sub.spans, &t.spans);
        }
        Ok((sub,))
    }

    fn model(&self) -> Option<ModelRef> {
        Some(self.model.clone())
    }
}

/// The word token each piece belongs to, by one pass over both span
/// lists, which must be in text order with the tokens not overlapping:
///
/// - A piece inside a token belongs to it.
/// - A piece over two or more tokens (a byte-level piece across a word
///   tokenizer's split, such as "'s" or ".)") belongs to the first of
///   them.
/// - A piece in no token (whitespace or punctuation the word tokenizer
///   left out) belongs to the token before it, or to the first token
///   when no token comes before it.
///
/// So every piece has a token, and the pieces of one token are a run, as
/// `Subwords::of` needs. None when there are pieces but no tokens.
pub fn align(pieces: &Spans<Subwords>, tokens: &Spans<Tokens>) -> Option<Vec<Ix<Tokens>>> {
    if tokens.is_empty() {
        return if pieces.is_empty() { Some(Vec::new()) } else { None };
    }
    let (t_start, t_end) = tokens.columns();
    let (p_start, p_end) = pieces.columns();
    let mut out = Vec::with_capacity(p_start.len());
    // k: the first token that ends after the piece starts. Piece starts
    // never go down, so k only moves forward.
    let mut k = 0;
    for (&a, &b) in p_start.iter().zip(p_end) {
        while k < t_end.len() && t_end[k] <= a {
            k += 1;
        }
        // The piece overlaps token k when the token starts before the
        // piece ends; an empty piece, when the token starts at or before it.
        let overlaps = k < t_start.len() && (t_start[k] < b || (a == b && t_start[k] <= a));
        let i = if overlaps { k } else { k.saturating_sub(1) };
        out.push(Ix::new(i as u32));
    }
    Some(out)
}

/// Offers `SubwordAnnotator` to pipelines built from configuration, as
/// "subwords": the spec's location is the bundle directory.
pub struct SubwordsProvider;

impl Provider<dyn Step> for SubwordsProvider {
    fn name(&self) -> &'static str {
        "subwords"
    }

    fn supports(&self, spec: &Spec) -> bool {
        spec.location.is_some()
    }

    fn create(&self, spec: &Spec) -> Result<Box<dyn Step>, Error> {
        let dir = spec.location.as_deref().ok_or_else(|| Error::Pipeline("subwords: no bundle location".into()))?;
        let a =
            SubwordAnnotator::open(dir).map_err(|e| Error::Pipeline(format!("subwords: {}: {e}", dir.display())))?;
        Ok(Box::new(Bound::new(a)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use turbo_analysis::ViewId;

    fn spans<L>(list: &[(u32, u32)]) -> Spans<L> {
        let mut s = Spans::new(ViewId::SOURCE);
        for &(a, b) in list {
            s.push(Span::new(a, b));
        }
        s
    }

    fn indexes(v: Option<Vec<Ix<Tokens>>>) -> Vec<usize> {
        v.unwrap().into_iter().map(Ix::get).collect()
    }

    #[test]
    fn a_piece_inside_a_token_belongs_to_it() {
        // "playing cards": play ##ing | cards
        let t = spans(&[(0, 7), (8, 13)]);
        let p = spans(&[(0, 4), (4, 7), (8, 13)]);
        assert_eq!(indexes(align(&p, &t)), [0, 0, 1]);
    }

    #[test]
    fn a_piece_over_two_tokens_belongs_to_the_first() {
        // "it's done": the word tokenizer splits "it" "'s", a piece "it's".
        let t = spans(&[(0, 2), (2, 4), (5, 9)]);
        let p = spans(&[(0, 4), (5, 9)]);
        assert_eq!(indexes(align(&p, &t)), [0, 2]);
        // A piece from the middle of one token into the next.
        let p = spans(&[(0, 1), (1, 3), (3, 4), (5, 9)]);
        assert_eq!(indexes(align(&p, &t)), [0, 0, 1, 2]);
    }

    #[test]
    fn a_piece_in_no_token_belongs_to_the_token_before_it() {
        // "a -- b" with "--" left out by the word tokenizer, and a
        // whitespace piece after the last token.
        let t = spans(&[(0, 1), (5, 6)]);
        let p = spans(&[(0, 1), (1, 2), (2, 4), (5, 6), (6, 8)]);
        assert_eq!(indexes(align(&p, &t)), [0, 0, 0, 1, 1]);
    }

    #[test]
    fn a_piece_before_every_token_belongs_to_the_first() {
        let t = spans(&[(3, 6)]);
        let p = spans(&[(0, 2), (3, 6)]);
        assert_eq!(indexes(align(&p, &t)), [0, 0]);
    }

    #[test]
    fn pieces_sharing_a_span_share_a_token() {
        // Two byte-level pieces of one four-byte character.
        let t = spans(&[(0, 1), (2, 6)]);
        let p = spans(&[(0, 1), (2, 6), (2, 6)]);
        assert_eq!(indexes(align(&p, &t)), [0, 1, 1]);
    }

    #[test]
    fn an_empty_piece_inside_a_token_belongs_to_it() {
        let t = spans(&[(0, 4), (5, 9)]);
        let p = spans(&[(0, 4), (5, 5), (5, 9)]);
        assert_eq!(indexes(align(&p, &t)), [0, 1, 1]);
    }

    #[test]
    fn no_tokens_gives_no_column_unless_no_pieces() {
        let none: Spans<Tokens> = spans(&[]);
        assert!(align(&spans(&[(0, 1)]), &none).is_none());
        assert_eq!(align(&spans(&[]), &none).unwrap().len(), 0);
    }
}
