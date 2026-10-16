//! The tokens a matcher reads and the layers aligned to them, and the
//! columns of value ids made from them once per call.

use std::ops::Range;

use turbo_analysis::{Ix, IxRange, Span, ViewId};
use turbo_analysis_layers::{Entities, Numbers, PosTags, Sentences, Strs, Tokens};

use crate::attr::{Attr, Flag, Iob, text_flags};
use crate::error::MatchError;
use crate::vocab::Vocab;

/// One document's tokens, the text they are in, and whichever token
/// layers the patterns read. The tokens' spans index `text`, the text of
/// the view they were made in. Every layer given is checked to be one
/// item per token (or, for entities, to index inside the tokens).
#[derive(Clone, Copy)]
pub struct Input<'a> {
    view: ViewId,
    text: &'a str,
    starts: &'a [u32],
    ends: &'a [u32],
    sentence: Option<&'a [Ix<Sentences>]>,
    pos: Option<&'a PosTags>,
    lemmas: Option<&'a Strs>,
    norms: Option<&'a Strs>,
    stems: Option<&'a Strs>,
    entities: Option<&'a Entities>,
    stop: Option<&'a [bool]>,
    numbers: Option<&'a Numbers>,
}

impl<'a> Input<'a> {
    /// The tokens of `text`, the text of the view they are in. Every span
    /// must lie inside it on character boundaries.
    pub fn new(text: &'a str, tokens: &'a Tokens) -> Result<Input<'a>, MatchError> {
        let (starts, ends) = tokens.spans.columns();
        if let Some(sentence) = &tokens.sentence
            && sentence.len() != starts.len()
        {
            return Err(MatchError::Misaligned { layer: "sentence column", len: sentence.len(), tokens: starts.len() });
        }
        for (index, (&start, &end)) in starts.iter().zip(ends).enumerate() {
            if start > end
                || end as usize > text.len()
                || !text.is_char_boundary(start as usize)
                || !text.is_char_boundary(end as usize)
            {
                return Err(MatchError::Token { index, start, end });
            }
        }
        Ok(Input {
            view: tokens.spans.view(),
            text,
            starts,
            ends,
            sentence: tokens.sentence.as_deref(),
            pos: None,
            lemmas: None,
            norms: None,
            stems: None,
            entities: None,
            stop: None,
            numbers: None,
        })
    }

    /// The view the input's tokens belong to.
    pub fn view(&self) -> ViewId {
        self.view
    }

    pub(crate) fn numbers(&self) -> Option<&'a Numbers> {
        self.numbers
    }

    fn aligned(&self, layer: &'static str, len: usize) -> Result<(), MatchError> {
        if len == self.len() { Ok(()) } else { Err(MatchError::Misaligned { layer, len, tokens: self.len() }) }
    }

    /// Part-of-speech tags, one per token.
    pub fn with_pos(mut self, pos: &'a PosTags) -> Result<Self, MatchError> {
        self.aligned("opennlp:pos", pos.tags.len())?;
        self.pos = Some(pos);
        Ok(self)
    }

    /// Lemmas, one per token.
    pub fn with_lemmas(mut self, lemmas: &'a Strs) -> Result<Self, MatchError> {
        self.aligned("opennlp:lemmas", lemmas.len())?;
        self.lemmas = Some(lemmas);
        Ok(self)
    }

    /// A normalized form per token, for [`Attr::Norm`]: any string column
    /// aligned to the tokens.
    pub fn with_norms(mut self, norms: &'a Strs) -> Result<Self, MatchError> {
        self.aligned("norms", norms.len())?;
        self.norms = Some(norms);
        Ok(self)
    }

    /// Stems, one per token.
    pub fn with_stems(mut self, stems: &'a Strs) -> Result<Self, MatchError> {
        self.aligned("turbo:stems", stems.len())?;
        self.stems = Some(stems);
        Ok(self)
    }

    /// Entities over these tokens.
    pub fn with_entities(mut self, entities: &'a Entities) -> Result<Self, MatchError> {
        if entities.spans.view() != self.view {
            return Err(MatchError::ViewMismatch {
                layer: "opennlp:entities",
                expected: self.view,
                actual: entities.spans.view(),
            });
        }
        let past = entities.tokens.iter().map(|r| r.end.get()).max().unwrap_or(0);
        if past > self.len() || entities.tokens.iter().any(|r| r.start > r.end) {
            return Err(MatchError::Misaligned { layer: "opennlp:entities", len: past, tokens: self.len() });
        }
        if entities.spans.len() != entities.tokens.len() || entities.kind.len() != entities.tokens.len() {
            return Err(MatchError::Misaligned {
                layer: "opennlp:entities",
                len: entities.spans.len().min(entities.kind.len()),
                tokens: entities.tokens.len(),
            });
        }
        self.entities = Some(entities);
        Ok(self)
    }

    /// Stopword flags, one per token.
    pub fn with_stopwords(mut self, stop: &'a [bool]) -> Result<Self, MatchError> {
        self.aligned("turbo:stopwords", stop.len())?;
        self.stop = Some(stop);
        Ok(self)
    }

    /// Number mentions in the same text as the tokens.
    pub fn with_numbers(mut self, numbers: &'a Numbers) -> Self {
        self.numbers = Some(numbers);
        self
    }

    /// How many tokens.
    pub fn len(&self) -> usize {
        self.starts.len()
    }

    pub fn is_empty(&self) -> bool {
        self.starts.is_empty()
    }

    /// Every token, as a range.
    pub fn all(&self) -> IxRange<Tokens> {
        IxRange::new(0, self.len() as u32)
    }

    /// The text of token `i`.
    pub fn token(&self, i: usize) -> &'a str {
        &self.text[self.starts[i] as usize..self.ends[i] as usize]
    }

    /// The bytes a token range covers in the text: from its tokens'
    /// first start to their last end (the first token's start and the
    /// last token's end, for tokens in text order). An empty range is an
    /// empty span at the next token's start (or the text's end).
    pub fn span(&self, r: IxRange<Tokens>) -> Span {
        let (a, b) = (r.start.get(), r.end.get());
        if a >= b {
            let at = self.starts.get(a).copied().unwrap_or(self.text.len() as u32);
            return Span::new(at, at);
        }
        let start = self.starts[a..b].iter().copied().min().unwrap_or(0);
        let end = self.ends[a..b].iter().copied().max().unwrap_or(0);
        Span::new(start, end)
    }

    /// `r` as a range of indexes, checked against the tokens.
    pub(crate) fn range(&self, r: IxRange<Tokens>) -> Result<Range<usize>, MatchError> {
        let (start, end) = (r.start.get(), r.end.get());
        if start > end || end > self.len() {
            return Err(MatchError::Range { start, end, tokens: self.len() });
        }
        Ok(start..end)
    }

    fn strings(&self, attr: Attr) -> Result<&'a Strs, MatchError> {
        let (column, layer) = match attr {
            Attr::Norm => (self.norms, "norms"),
            Attr::Lemma => (self.lemmas, "opennlp:lemmas"),
            Attr::Stem => (self.stems, "turbo:stems"),
            _ => unreachable!("a string column"),
        };
        column.ok_or(MatchError::Missing(layer))
    }

    /// The first entity covering each token of `range`, as an index into
    /// the entities, or `u32::MAX` where none does.
    fn entity_of(&self, range: &Range<usize>) -> Result<Vec<u32>, MatchError> {
        let entities = self.entities.ok_or(MatchError::Missing("opennlp:entities"))?;
        if entities.spans.view() != self.view {
            return Err(MatchError::ViewMismatch {
                layer: "opennlp:entities",
                expected: self.view,
                actual: entities.spans.view(),
            });
        }
        let mut of = vec![u32::MAX; range.len()];
        for (e, r) in entities.tokens.iter().enumerate() {
            let (a, b) = (r.start.get().max(range.start), r.end.get().min(range.end));
            let e_ix = Ix::new(e as u32);
            let disc = entities.spans.is_discontinuous(e_ix);
            for t in a..b {
                let slot = &mut of[t - range.start];
                if *slot == u32::MAX {
                    if disc {
                        let tok_span = Span::new(self.starts[t], self.ends[t]);
                        if !entities.spans.fragments(e_ix).any(|frag| frag.overlaps(tok_span)) {
                            continue;
                        }
                    }
                    *slot = e as u32;
                }
            }
        }
        Ok(of)
    }
}

/// What a set of patterns reads, so a call builds only those columns.
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq)]
pub(crate) struct Needs {
    /// One bit per [`Attr`], by its discriminant.
    pub(crate) attrs: u16,
    /// One bit per [`Flag`].
    pub(crate) flags: u32,
    pub(crate) iob: bool,
    pub(crate) length: bool,
}

impl Needs {
    pub(crate) fn attr(&mut self, a: Attr) {
        self.attrs |= 1 << a as u16;
    }

    pub(crate) fn has(&self, a: Attr) -> bool {
        self.attrs & (1 << a as u16) != 0
    }
}

/// The values the patterns test, for the tokens of one range: per string
/// attribute, a vocabulary id per token; the flags as a bit set per token;
/// the entity position and the length in characters when they are read.
/// Index `i` is token `range.start + i`.
#[derive(Default)]
pub(crate) struct Columns {
    pub(crate) values: [Vec<u32>; 10],
    pub(crate) flags: Vec<u32>,
    pub(crate) iob: Vec<Iob>,
    pub(crate) chars: Vec<u32>,
}

impl Columns {
    pub(crate) fn build(
        input: &Input<'_>,
        range: &Range<usize>,
        needs: Needs,
        vocab: &Vocab,
    ) -> Result<Columns, MatchError> {
        if let Some(numbers) = input.numbers
            && numbers.spans.view() != input.view
        {
            return Err(MatchError::ViewMismatch {
                layer: "turbo:numbers",
                expected: input.view,
                actual: numbers.spans.view(),
            });
        }
        let mut c = Columns::default();
        let mut scratch = String::new();
        let entity_of = if needs.has(Attr::Entity) || needs.iob { Some(input.entity_of(range)?) } else { None };
        for attr in Attr::ALL {
            if !needs.has(attr) {
                continue;
            }
            let column = match attr {
                Attr::Text | Attr::Lower | Attr::Prefix | Attr::Suffix | Attr::Shape => {
                    range.clone().map(|i| vocab.get(attr.of_text(input.token(i), &mut scratch))).collect()
                }
                Attr::Norm | Attr::Lemma | Attr::Stem => {
                    let strs = input.strings(attr)?;
                    range.clone().map(|i| vocab.get(strs.get(i))).collect()
                }
                Attr::Pos => {
                    let pos = input.pos.ok_or(MatchError::Missing("opennlp:pos"))?;
                    let ids: Vec<u32> = (0..pos.tagset.len())
                        .map(|t| vocab.get(pos.tagset.label(turbo_analysis::Tag(t as u16))))
                        .collect();
                    pos.tags[range.clone()].iter().map(|t| ids[t.0 as usize]).collect()
                }
                Attr::Entity => {
                    let entities = input.entities.expect("checked by entity_of");
                    let ids: Vec<u32> = (0..entities.kinds.len())
                        .map(|t| vocab.get(entities.kinds.label(turbo_analysis::Tag(t as u16))))
                        .collect();
                    let of = entity_of.as_deref().expect("built above");
                    // A token no entity covers has the empty kind, as
                    // spaCy's `ENT_TYPE` is "" there.
                    let outside = vocab.get("");
                    of.iter()
                        .map(|&e| if e == u32::MAX { outside } else { ids[entities.kind[e as usize].0 as usize] })
                        .collect()
                }
            };
            c.values[attr as usize] = column;
        }
        if needs.iob {
            let entities = input.entities.expect("checked by entity_of");
            let of = entity_of.as_deref().expect("built above");
            c.iob = of
                .iter()
                .zip(range.clone())
                .map(|(&e, i)| match e {
                    u32::MAX => Iob::Outside,
                    e if entities.tokens[e as usize].start.get() == i => Iob::Begin,
                    _ => Iob::Inside,
                })
                .collect();
        }
        if needs.flags != 0 {
            c.flags = flag_column(input, range, needs.flags)?;
        }
        if needs.length {
            c.chars = range.clone().map(|i| input.token(i).chars().count() as u32).collect();
        }
        Ok(c)
    }
}

/// The flags of each token of `range`, as bits, for the flags in `wanted`.
fn flag_column(input: &Input<'_>, range: &Range<usize>, wanted: u32) -> Result<Vec<u32>, MatchError> {
    let text_bits = wanted & !(Flag::SpaceAfter.bit() | Flag::SentStart.bit() | Flag::Stop.bit() | Flag::Number.bit());
    let mut bits: Vec<u32> =
        range.clone().map(|i| if text_bits == 0 { 0 } else { text_flags(input.token(i)) & text_bits }).collect();
    if wanted & Flag::SpaceAfter.bit() != 0 {
        for (b, i) in bits.iter_mut().zip(range.clone()) {
            if input.text[input.ends[i] as usize..].chars().next().is_some_and(char::is_whitespace) {
                *b |= Flag::SpaceAfter.bit();
            }
        }
    }
    if wanted & Flag::SentStart.bit() != 0 {
        let sentence = input.sentence.ok_or(MatchError::Missing("opennlp:sentences"))?;
        for (b, i) in bits.iter_mut().zip(range.clone()) {
            if i == 0 || sentence[i] != sentence[i - 1] {
                *b |= Flag::SentStart.bit();
            }
        }
    }
    if wanted & Flag::Stop.bit() != 0 {
        let stop = input.stop.ok_or(MatchError::Missing("turbo:stopwords"))?;
        for (b, i) in bits.iter_mut().zip(range.clone()) {
            if stop[i] {
                *b |= Flag::Stop.bit();
            }
        }
    }
    if wanted & Flag::Number.bit() != 0 {
        let numbers = input.numbers.ok_or(MatchError::Missing("turbo:numbers"))?;
        if numbers.spans.view() != input.view {
            return Err(MatchError::ViewMismatch {
                layer: "turbo:numbers",
                expected: input.view,
                actual: numbers.spans.view(),
            });
        }
        let (n_starts, n_ends) = numbers.spans.columns();
        for (b, i) in bits.iter_mut().zip(range.clone()) {
            let token = Span::new(input.starts[i], input.ends[i]);
            // Number mentions are in text order and do not overlap, so
            // the first one ending after the token's start is the only
            // one it can overlap first.
            let at = n_ends.partition_point(|&e| e <= token.start);
            if at < n_starts.len() && Span::new(n_starts[at], n_ends[at]).overlaps(token) {
                *b |= Flag::Number.bit();
            }
        }
    }
    Ok(bits)
}
