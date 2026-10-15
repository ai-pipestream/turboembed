//! The tokens a matcher runs over, with the layers its rules may read.

use std::fmt;

use turbo_analysis::{Span, View};
use turbo_analysis_layers::{Entities, Lemmas, Numbers, PosTags, Sentences, Strs, Tag, TagSet, Tokens};
use turbo_stem_layers::Stems;
use turbo_stopword_layers::Stopwords;

use crate::attr::{self, Attr, Flag};

/// A run of tokens over a text, and the values their layers give them.
/// Every layer added must have one item per token; text spans must lie in
/// the text and on character boundaries. The matchers only read it.
pub struct Sequence<'a> {
    pub(crate) text: &'a str,
    pub(crate) start: &'a [u32],
    pub(crate) end: &'a [u32],
    lemma: Option<&'a Strs>,
    stem: Option<&'a Strs>,
    pos: Option<(&'a [Tag], &'a TagSet)>,
    /// The entity kind of each token, `None` outside every entity.
    entity: Option<(Vec<Option<Tag>>, &'a TagSet)>,
    stop: Option<&'a [bool]>,
    sent_start: Option<Vec<bool>>,
    in_number: Option<Vec<bool>>,
    columns: Vec<(&'static str, &'a Strs)>,
}

/// Why a sequence could not be built.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum SequenceError {
    /// A layer had a different number of items than there are tokens.
    Length { layer: &'static str, items: usize, tokens: usize },
    /// A token span is outside the text or cuts a character.
    Span { token: usize, span: Span },
    /// Two layers that must share a view do not.
    View { layer: &'static str },
    /// Two columns were added under one name.
    Column(&'static str),
}

impl fmt::Display for SequenceError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            SequenceError::Length { layer, items, tokens } => {
                write!(f, "{layer} has {items} items for {tokens} tokens")
            }
            SequenceError::Span { token, span } => {
                write!(f, "token {token} spans {}..{}, outside the text or inside a character", span.start, span.end)
            }
            SequenceError::View { layer } => write!(f, "{layer} is in another view than the tokens"),
            SequenceError::Column(name) => write!(f, "two columns are named {name}"),
        }
    }
}

impl std::error::Error for SequenceError {}

impl<'a> Sequence<'a> {
    /// Tokens given by their spans in `text`.
    pub fn new(text: &'a str, start: &'a [u32], end: &'a [u32]) -> Result<Sequence<'a>, SequenceError> {
        if start.len() != end.len() {
            return Err(SequenceError::Length { layer: "token ends", items: end.len(), tokens: start.len() });
        }
        for (i, (&s, &e)) in start.iter().zip(end).enumerate() {
            let ok = s <= e
                && (e as usize) <= text.len()
                && text.is_char_boundary(s as usize)
                && text.is_char_boundary(e as usize);
            if !ok {
                return Err(SequenceError::Span { token: i, span: Span { start: s, end: e } });
            }
        }
        Ok(Sequence {
            text,
            start,
            end,
            lemma: None,
            stem: None,
            pos: None,
            entity: None,
            stop: None,
            sent_start: None,
            in_number: None,
            columns: Vec::new(),
        })
    }

    /// The `opennlp:tokens` of a view. When the tokens say which sentence
    /// each is in, `Flag::SentStart` comes from that.
    pub fn tokens(view: &'a View, tokens: &'a Tokens) -> Result<Sequence<'a>, SequenceError> {
        if tokens.spans.view() != view.id() {
            return Err(SequenceError::View { layer: "opennlp:tokens" });
        }
        let (start, end) = tokens.spans.columns();
        let mut seq = Sequence::new(view.text(), start, end)?;
        if let Some(sentence) = &tokens.sentence {
            seq.sent_start = Some((0..sentence.len()).map(|i| i == 0 || sentence[i] != sentence[i - 1]).collect());
        }
        Ok(seq)
    }

    pub fn len(&self) -> usize {
        self.start.len()
    }

    pub fn is_empty(&self) -> bool {
        self.start.is_empty()
    }

    /// The text of token `i`.
    pub fn text(&self, i: usize) -> &'a str {
        &self.text[self.start[i] as usize..self.end[i] as usize]
    }

    /// The span of tokens `[start, end)` in the text: from the start of
    /// the first to the end of the last.
    pub fn span(&self, start: usize, end: usize) -> Span {
        Span { start: self.start[start], end: self.end[end - 1] }
    }

    fn check_len(&self, layer: &'static str, items: usize) -> Result<(), SequenceError> {
        if items == self.len() { Ok(()) } else { Err(SequenceError::Length { layer, items, tokens: self.len() }) }
    }

    pub fn with_lemmas(mut self, lemmas: &'a Lemmas) -> Result<Self, SequenceError> {
        self.check_len("opennlp:lemmas", lemmas.lemma.len())?;
        self.lemma = Some(&lemmas.lemma);
        Ok(self)
    }

    pub fn with_stems(mut self, stems: &'a Stems) -> Result<Self, SequenceError> {
        self.check_len("turbo:stems", stems.stem.len())?;
        self.stem = Some(&stems.stem);
        Ok(self)
    }

    pub fn with_pos(mut self, pos: &'a PosTags) -> Result<Self, SequenceError> {
        self.check_len("opennlp:pos", pos.tags.len())?;
        self.pos = Some((&pos.tags, &pos.tagset));
        Ok(self)
    }

    /// The entity kind of each token: a token in two entities takes the
    /// kind of the one that comes first in the layer.
    pub fn with_entities(mut self, entities: &'a Entities) -> Result<Self, SequenceError> {
        let mut kind = vec![None; self.len()];
        for (r, &k) in entities.tokens.iter().zip(&entities.kind) {
            if r.end.get() > self.len() {
                return Err(SequenceError::Length {
                    layer: "opennlp:entities",
                    items: r.end.get(),
                    tokens: self.len(),
                });
            }
            for slot in &mut kind[r.start.get()..r.end.get()] {
                slot.get_or_insert(k);
            }
        }
        self.entity = Some((kind, &entities.kinds));
        Ok(self)
    }

    pub fn with_stopwords(mut self, stopwords: &'a Stopwords) -> Result<Self, SequenceError> {
        self.check_len("turbo:stopwords", stopwords.stop.len())?;
        self.stop = Some(&stopwords.stop);
        Ok(self)
    }

    /// Sentence starts from sentence spans, for tokens that do not say
    /// which sentence each is in: a token starts a sentence when it is the
    /// first token at or after a sentence's start. The sentences must be
    /// in the tokens' view.
    pub fn with_sentences(mut self, view: &View, sentences: &Sentences) -> Result<Self, SequenceError> {
        if sentences.spans.view() != view.id() || self.text.as_ptr() != view.text().as_ptr() {
            return Err(SequenceError::View { layer: "opennlp:sentences" });
        }
        let mut starts = vec![false; self.len()];
        let mut t = 0;
        for (_, s) in sentences.spans.iter() {
            while t < self.len() && self.start[t] < s.start {
                t += 1;
            }
            if t < self.len() && self.end[t] <= s.end {
                starts[t] = true;
            }
        }
        self.sent_start = Some(starts);
        Ok(self)
    }

    /// Number mentions: a token whose span lies inside one holds
    /// `Flag::LikeNum`. The mentions must be in the tokens' view.
    pub fn with_numbers(mut self, view: &View, numbers: &Numbers) -> Result<Self, SequenceError> {
        if numbers.spans.view() != view.id() || self.text.as_ptr() != view.text().as_ptr() {
            return Err(SequenceError::View { layer: "turbo:numbers" });
        }
        let mut mentions: Vec<Span> = numbers.spans.iter().map(|(_, s)| s).collect();
        mentions.sort_unstable();
        let mut inside = vec![false; self.len()];
        // Mentions by start: a token is inside one when the furthest end
        // among the mentions that start at or before it reaches its end.
        let mut m = 0;
        let mut reach = 0u32;
        for ((slot, &start), &end) in inside.iter_mut().zip(self.start).zip(self.end) {
            while m < mentions.len() && mentions[m].start <= start {
                reach = reach.max(mentions[m].end);
                m += 1;
            }
            *slot = m > 0 && end <= reach && start < end;
        }
        self.in_number = Some(inside);
        Ok(self)
    }

    /// A string column read by `Attr::Column(name)`.
    pub fn with_column(mut self, name: &'static str, values: &'a Strs) -> Result<Self, SequenceError> {
        self.check_len(name, values.len())?;
        if self.columns.iter().any(|&(n, _)| n == name) {
            return Err(SequenceError::Column(name));
        }
        self.columns.push((name, values));
        Ok(self)
    }

    /// Whether the sequence has what `attr` reads.
    pub fn has(&self, attr: Attr) -> bool {
        match attr {
            Attr::Text | Attr::Lower | Attr::Norm | Attr::Shape => true,
            Attr::Lemma => self.lemma.is_some(),
            Attr::Stem => self.stem.is_some(),
            Attr::Pos => self.pos.is_some(),
            Attr::Entity => self.entity.is_some(),
            Attr::Column(name) => self.columns.iter().any(|&(n, _)| n == name),
        }
    }

    /// Whether the sequence has what `flag` reads.
    pub fn has_flag(&self, flag: Flag) -> bool {
        match flag {
            Flag::Stop => self.stop.is_some(),
            Flag::SentStart => self.sent_start.is_some(),
            _ => true,
        }
    }

    /// Calls `f` with the value of `attr` for token `i`, computing it in
    /// `buf` when it is not stored. The attribute must be present (`has`).
    pub(crate) fn with_value<R>(&self, attr: Attr, i: usize, buf: &mut Buf, f: impl FnOnce(&str) -> R) -> R {
        match attr {
            Attr::Text => f(self.text(i)),
            Attr::Lower => {
                buf.out.clear();
                attr::lower(self.text(i), &mut buf.out);
                f(&buf.out)
            }
            Attr::Norm => {
                buf.out.clear();
                attr::norm(self.text(i), &mut buf.out, &mut buf.tmp);
                f(&buf.out)
            }
            Attr::Shape => {
                buf.out.clear();
                attr::shape(self.text(i), &mut buf.out);
                f(&buf.out)
            }
            Attr::Lemma => f(self.lemma.expect("lemmas present").get(i)),
            Attr::Stem => f(self.stem.expect("stems present").get(i)),
            Attr::Pos => {
                let (tags, set) = self.pos.expect("tags present");
                f(set.label(tags[i]))
            }
            Attr::Entity => {
                let (kinds, set) = self.entity.as_ref().expect("entities present");
                f(kinds[i].map_or("", |k| set.label(k)))
            }
            Attr::Column(name) => {
                let col = self.columns.iter().find(|&&(n, _)| n == name).expect("column present").1;
                f(col.get(i))
            }
        }
    }

    /// The flags of every token, a bit per `Flag`.
    pub(crate) fn flags(&self) -> Vec<u32> {
        self.flags_in(0..self.len())
    }

    /// The flags of the tokens in `range`.
    pub(crate) fn flags_in(&self, range: std::ops::Range<usize>) -> Vec<u32> {
        range
            .map(|i| {
                let text = self.text(i);
                let mut bits = attr::lexical_flags(text);
                if self.in_number.as_ref().is_some_and(|n| n[i]) {
                    bits |= Flag::LikeNum.bit();
                }
                if self.stop.is_some_and(|s| s[i]) {
                    bits |= Flag::Stop.bit();
                }
                if self.sent_start.as_ref().is_some_and(|s| s[i]) {
                    bits |= Flag::SentStart.bit();
                }
                if self.text[self.end[i] as usize..].chars().next().is_some_and(turbo_unicode::props::white_space) {
                    bits |= Flag::SpaceAfter.bit();
                }
                bits
            })
            .collect()
    }
}

/// Scratch strings for values computed from the text.
#[derive(Default)]
pub(crate) struct Buf {
    out: String,
    tmp: String,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn spans_must_lie_on_character_boundaries() {
        let text = "naïve x";
        assert!(Sequence::new(text, &[0, 7], &[6, 8]).is_ok());
        assert_eq!(
            Sequence::new(text, &[0], &[3]).err(),
            Some(SequenceError::Span { token: 0, span: Span { start: 0, end: 3 } })
        );
        assert!(Sequence::new(text, &[0], &[9]).is_err());
        assert!(Sequence::new(text, &[0, 1], &[1]).is_err());
    }

    #[test]
    fn computed_values() {
        let text = "The ﬁnal Straße";
        let seq = Sequence::new(text, &[0, 4, 11], &[3, 10, 18]).unwrap();
        let mut buf = Buf::default();
        let v = |a, i, buf: &mut Buf| seq.with_value(a, i, buf, |s| s.to_string());
        assert_eq!(v(Attr::Text, 1, &mut buf), "ﬁnal");
        assert_eq!(v(Attr::Lower, 0, &mut buf), "the");
        assert_eq!(v(Attr::Norm, 1, &mut buf), "final");
        assert_eq!(v(Attr::Norm, 2, &mut buf), "strasse");
        assert_eq!(v(Attr::Shape, 2, &mut buf), "Xxxxx");
        let flags = seq.flags();
        assert!(flags[0] & Flag::SpaceAfter.bit() != 0);
        assert!(flags[2] & Flag::SpaceAfter.bit() == 0);
        assert!(flags[0] & Flag::Title.bit() != 0);
    }
}
