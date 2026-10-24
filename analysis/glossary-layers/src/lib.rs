//! The glossary step's layer and the glossary it points into. They live in
//! this small crate so a consumer of the matched terms (the chunker, an
//! indexer) does not depend on the matchers and readers that make them.
//!
//! A [`Glossary`] is a list of entries in registration order, each a
//! stable identifier and a surface form to find in text, OpenNLP's
//! `GlossaryEntry`. An identifier with several aliases is several entries
//! with the same identifier. [`GlossaryTerms`] is one item per term found
//! in a text: its span, the tokens it covers and the entry that matched.

pub mod export;

use std::fmt;
use std::sync::Arc;

use turbo_analysis::{IxRange, Key, KeyId, Layer, Overlap, Span, Spans, ViewId};
use turbo_analysis_layers::{Strs, Tokens};
use turbo_unicode::WhitespaceMode;

/// Whether `s` is empty or holds only Unicode `White_Space` characters:
/// OpenNLP's `StringUtil.isUnicodeBlank`, which the glossary classes use
/// for identifiers and terms.
pub fn is_blank(s: &str) -> bool {
    s.chars().all(|c| WhitespaceMode::Unicode.is_whitespace(c as u32))
}

/// `s` without leading and trailing Unicode `White_Space` characters:
/// OpenNLP's `StringUtil.trimUnicodeWhitespace`.
pub fn trim_blank(s: &str) -> &str {
    s.trim_matches(|c: char| WhitespaceMode::Unicode.is_whitespace(c as u32))
}

/// The position of an entry in its [`Glossary`], in registration order.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct EntryIx(pub u32);

impl EntryIx {
    pub const fn get(self) -> usize {
        self.0 as usize
    }
}

/// Why an entry was refused.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum EntryError {
    /// The identifier is empty or only whitespace.
    BlankId,
    /// The term is empty or only whitespace.
    BlankTerm,
}

impl fmt::Display for EntryError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            EntryError::BlankId => "id must not be blank",
            EntryError::BlankTerm => "term must not be blank",
        })
    }
}

impl std::error::Error for EntryError {}

/// One entry of a glossary, borrowed from it.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct GlossaryEntry<'g> {
    /// The stable identifier the term resolves to, such as a concept code.
    pub id: &'g str,
    /// The surface form to find, possibly several words.
    pub term: &'g str,
}

/// Glossary entries in registration order, stored by column: one buffer
/// of identifiers and one of terms. Neither an identifier nor a term is
/// blank; [`Glossary::push`] refuses one that is. Several entries may
/// share an identifier (aliases) or a term (the first registered wins
/// when both match the same text).
#[derive(Clone, Debug, Default)]
pub struct Glossary {
    id: Strs,
    term: Strs,
}

impl Glossary {
    pub fn new() -> Glossary {
        Glossary::default()
    }

    /// Adds an entry and returns its position. A blank identifier or term
    /// is refused, as OpenNLP's `GlossaryEntry` refuses it.
    pub fn push(&mut self, id: &str, term: &str) -> Result<EntryIx, EntryError> {
        if is_blank(id) {
            return Err(EntryError::BlankId);
        }
        if is_blank(term) {
            return Err(EntryError::BlankTerm);
        }
        let ix = EntryIx(self.id.len() as u32);
        self.id.push(id);
        self.term.push(term);
        Ok(ix)
    }

    pub fn len(&self) -> usize {
        self.id.len()
    }

    pub fn is_empty(&self) -> bool {
        self.id.is_empty()
    }

    /// Entry `i`. Panics when `i` is not an entry of this glossary.
    pub fn get(&self, i: EntryIx) -> GlossaryEntry<'_> {
        GlossaryEntry { id: self.id.get(i.get()), term: self.term.get(i.get()) }
    }

    /// Entry `i`, or `None` when `i` is not an entry of this glossary.
    pub fn try_get(&self, i: EntryIx) -> Option<GlossaryEntry<'_>> {
        (i.get() < self.len()).then(|| self.get(i))
    }

    pub fn id(&self, i: EntryIx) -> &str {
        self.id.get(i.get())
    }

    pub fn term(&self, i: EntryIx) -> &str {
        self.term.get(i.get())
    }

    /// Every entry with its position, in registration order.
    pub fn iter(&self) -> impl ExactSizeIterator<Item = (EntryIx, GlossaryEntry<'_>)> + '_ {
        (0..self.len() as u32).map(move |i| (EntryIx(i), self.get(EntryIx(i))))
    }
}

impl PartialEq for Glossary {
    /// The same entries in the same order.
    fn eq(&self, other: &Glossary) -> bool {
        self.len() == other.len() && self.id.iter().eq(other.id.iter()) && self.term.iter().eq(other.term.iter())
    }
}

impl Eq for Glossary {}

/// The glossary terms found in a text, one item per match: OpenNLP's
/// `GlossaryAnnotator.GLOSSARY` layer. Items are in text order and do not
/// overlap (the matchers resolve overlapping candidates before writing).
/// Each has the tokens it covers, in the view its span is in, and the
/// glossary entry that matched; `glossary` is the list the entries index,
/// shared with the step that wrote the layer rather than copied.
///
/// A match normally covers whole tokens. One found by its characters can
/// also start or end where a token ends and the next begins, so it may
/// cover no token at all (a term `-` between the tokens `a` and `b` of
/// "a-b" whose tokenizer drops the hyphen); its token range is then empty
/// and sits where the match is.
pub struct GlossaryTerms {
    pub spans: Spans<GlossaryTerms>,
    pub tokens: Vec<IxRange<Tokens>>,
    pub entry: Vec<EntryIx>,
    pub glossary: Arc<Glossary>,
}

impl GlossaryTerms {
    pub fn new(view: ViewId, glossary: Arc<Glossary>) -> GlossaryTerms {
        GlossaryTerms { spans: Spans::new(view), tokens: Vec::new(), entry: Vec::new(), glossary }
    }

    pub fn push(&mut self, span: Span, tokens: IxRange<Tokens>, entry: EntryIx) {
        self.spans.push(span);
        self.tokens.push(tokens);
        self.entry.push(entry);
    }

    /// The entry of item `i`.
    pub fn entry_of(&self, i: usize) -> GlossaryEntry<'_> {
        self.glossary.get(self.entry[i])
    }

    /// The entry of item `i`, or `None` when `i` is out of bounds or its
    /// entry index is outside `self.glossary`.
    pub fn try_entry_of(&self, i: usize) -> Option<GlossaryEntry<'_>> {
        self.entry.get(i).and_then(|&ix| self.glossary.try_get(ix))
    }
}

impl Layer for GlossaryTerms {
    const ID: &'static str = "turbo:glossary-terms";
    const OVERLAP: Overlap = Overlap::None;

    fn len(&self) -> usize {
        self.spans.len()
    }

    fn text_spans(&self) -> Option<turbo_analysis::TextSpans<'_>> {
        Some(self.spans.text_spans())
    }

    fn indexes(&self) -> Vec<(KeyId, usize)> {
        vec![(Key::<Tokens>::DEFAULT.id(), self.tokens.iter().map(|r| r.end.get()).max().unwrap_or(0))]
    }

    fn check(&self) -> Result<(), String> {
        let n = self.spans.len();
        if self.tokens.len() != n || self.entry.len() != n {
            return Err("tokens and an entry for every term".into());
        }
        if self.spans.iter().any(|(_, s)| s.is_empty()) {
            return Err("an empty term span".into());
        }
        self.spans.check(Self::OVERLAP).map_err(|_| "term spans not in text order or overlapping")?;
        debug_assert_eq!(self.glossary.id.len(), self.glossary.term.len());
        if self.entry.iter().any(|e| e.get() >= self.glossary.len()) {
            return Err(format!("an entry outside the glossary of {}", self.glossary.len()));
        }
        if !self.tokens.windows(2).all(|w| w[0].end <= w[1].start) {
            return Err("token ranges not in order or overlapping".into());
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn blank_is_unicode_white_space() {
        assert!(is_blank(""));
        assert!(is_blank(" \t\u{a0}\u{3000}\u{2028}"));
        assert!(!is_blank(" a "));
        // U+200B is not White_Space.
        assert!(!is_blank("\u{200b}"));
        assert_eq!(trim_blank("\u{a0} hot dog\u{3000}"), "hot dog");
    }

    #[test]
    fn entries_refuse_blank_ids_and_terms() {
        let mut g = Glossary::new();
        assert_eq!(g.push("c1", "hot dog"), Ok(EntryIx(0)));
        assert_eq!(g.push("c1", "frankfurter"), Ok(EntryIx(1)));
        assert_eq!(g.push(" ", "x"), Err(EntryError::BlankId));
        assert_eq!(g.push("c2", "\u{a0}"), Err(EntryError::BlankTerm));
        assert_eq!(g.len(), 2);
        assert_eq!(g.get(EntryIx(1)), GlossaryEntry { id: "c1", term: "frankfurter" });
        assert_eq!(g.try_get(EntryIx(1)), Some(GlossaryEntry { id: "c1", term: "frankfurter" }));
        assert_eq!(g.try_get(EntryIx(2)), None);
        let ids: Vec<&str> = g.iter().map(|(_, e)| e.id).collect();
        assert_eq!(ids, ["c1", "c1"]);
        let mut h = Glossary::new();
        h.push("c1", "hot dog").unwrap();
        assert_ne!(g, h);
        h.push("c1", "frankfurter").unwrap();
        assert_eq!(g, h);
    }

    #[test]
    fn check_wants_columns_entries_and_order() {
        let mut g = Glossary::new();
        g.push("a", "x").unwrap();
        let g = Arc::new(g);
        let mut t = GlossaryTerms::new(ViewId::SOURCE, g.clone());
        t.push(Span::new(0, 1), IxRange::new(0, 1), EntryIx(0));
        t.push(Span::new(2, 3), IxRange::new(1, 1), EntryIx(0));
        assert_eq!(t.check(), Ok(()));
        assert_eq!(t.entry_of(1).term, "x");
        assert_eq!(t.try_entry_of(1), Some(GlossaryEntry { id: "a", term: "x" }));
        assert_eq!(t.try_entry_of(2), None);
        assert_eq!(t.indexes()[0].1, 1);
        t.push(Span::new(4, 5), IxRange::new(0, 2), EntryIx(0));
        assert_eq!(t.check().unwrap_err(), "token ranges not in order or overlapping");
        let mut t = GlossaryTerms::new(ViewId::SOURCE, g.clone());
        t.push(Span::new(2, 2), IxRange::new(0, 1), EntryIx(0));
        assert_eq!(t.check().unwrap_err(), "an empty term span");
        let mut t = GlossaryTerms::new(ViewId::SOURCE, g.clone());
        t.push(Span::new(0, 1), IxRange::new(0, 1), EntryIx(1));
        assert_eq!(t.try_entry_of(0), None);
        assert!(t.check().unwrap_err().starts_with("an entry outside"));
        let mut t = GlossaryTerms::new(ViewId::SOURCE, g.clone());
        t.push(Span::new(0, 1), IxRange::new(0, 1), EntryIx(0));
        t.entry.pop();
        assert_eq!(t.check().unwrap_err(), "tokens and an entry for every term");
        assert_eq!(GlossaryTerms::ID, "turbo:glossary-terms");

        // Reversed, overlapping, or multi-fragment spans are rejected even when token ranges are non-decreasing.
        let mut reversed = GlossaryTerms::new(ViewId::SOURCE, g.clone());
        reversed.push(Span::new(4, 6), IxRange::new(1, 1), EntryIx(0));
        reversed.push(Span::new(0, 2), IxRange::new(1, 1), EntryIx(0));
        assert_eq!(reversed.check().unwrap_err(), "term spans not in text order or overlapping");

        let mut overlapping = GlossaryTerms::new(ViewId::SOURCE, g.clone());
        overlapping.push(Span::new(0, 4), IxRange::new(0, 1), EntryIx(0));
        overlapping.push(Span::new(2, 6), IxRange::new(1, 2), EntryIx(0));
        assert_eq!(overlapping.check().unwrap_err(), "term spans not in text order or overlapping");

        let mut fragmented = GlossaryTerms::new(ViewId::SOURCE, g);
        fragmented.spans.push_fragments(&[Span::new(0, 2), Span::new(4, 6)]);
        fragmented.tokens.push(IxRange::new(0, 2));
        fragmented.entry.push(EntryIx(0));
        assert_eq!(fragmented.check().unwrap_err(), "term spans not in text order or overlapping");
    }
}
