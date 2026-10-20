//! The coreference layers: [`Mentions`] (`opennlp:mentions`), the typed,
//! columnar mention layer every chain and antecedent link indexes, and
//! [`Speakers`] (`opennlp:speakers`), the optional speaker turns the
//! speaker sieve reads.
//!
//! Chains are `Groups<Mentions>` (`turbo:groups<opennlp:mentions>`) and
//! resolved antecedent links are `Edges<Mentions, Mentions>`
//! (`turbo:edges<opennlp:mentions, opennlp:mentions>`), both from
//! `turbo-analysis`; [`codec`] registers the wire codecs for all four.

use std::sync::Arc;

use turbo_analysis::{Document, Ix, IxRange, Key, KeyId, Layer, Overlap, Span, Spans, Tag, TagSet, TextSpans};
use turbo_analysis_layers::{Entities, Sentences, Tokens};

mod agreement;
pub mod codec;

pub use agreement::{Agreement, Animacy, Gender, MentionKind, Number, Origin, Person};

/// Coreference mentions (`opennlp:mentions`), one row per mention.
///
/// A mention is a span of one text view, discontinuous when the corpus
/// splits it (CorefUD's split and coordinated mentions), with its
/// sentence, its bounding token range, its head token inside that range,
/// the entity it was lifted from when there is one, and its form and
/// agreement. Spans may nest and cross (`Overlap::Any`): a possessor
/// lies inside its noun phrase, an entity inside a wider phrase.
///
/// [`Layer::check`] holds every column to the span count, every token
/// range to at least one token, every head inside its token range, and
/// each origin to the forms it can take; [`Layer::indexes`] gives
/// `Document::add` the bounds of every index into [`Sentences`],
/// [`Tokens`] and [`Entities`], so a mention cannot point past the
/// instances its step read.
#[derive(Clone, Debug)]
pub struct Mentions {
    pub spans: Spans<Mentions>,
    pub sentence: Vec<Ix<Sentences>>,
    /// The bounding token range of each mention; for a discontinuous
    /// mention, from its first fragment's first token to its last
    /// fragment's last token, gap tokens included.
    pub tokens: Vec<IxRange<Tokens>>,
    pub head: Vec<Ix<Tokens>>,
    /// The entity each mention was lifted from, when the layer was made
    /// with an entity layer at hand; `None` for a layer made without one
    /// (a gold layer read from a corpus).
    pub entity: Option<Vec<Option<Ix<Entities>>>>,
    pub kind: Vec<MentionKind>,
    pub origin: Vec<Origin>,
    pub agreement: Vec<Agreement>,
}

impl Mentions {
    /// Creates an empty `Mentions` layer in `view`.
    pub fn new(view: turbo_analysis::ViewId) -> Self {
        Self {
            spans: Spans::new(view),
            sentence: Vec::new(),
            tokens: Vec::new(),
            head: Vec::new(),
            entity: None,
            kind: Vec::new(),
            origin: Vec::new(),
            agreement: Vec::new(),
        }
    }

    /// Pushes one mention with default `Agreement::unknown()`.
    pub fn push(
        &mut self,
        span: Span,
        sentence: Ix<Sentences>,
        tokens: IxRange<Tokens>,
        head: Ix<Tokens>,
        kind: MentionKind,
    ) -> Ix<Mentions> {
        let origin = match kind {
            MentionKind::Proper => Origin::Entity,
            MentionKind::Nominal => Origin::NounPhrase,
            MentionKind::Pronominal | MentionKind::Demonstrative => Origin::Pronoun,
        };
        self.sentence.push(sentence);
        self.tokens.push(tokens);
        self.head.push(head);
        self.kind.push(kind);
        self.origin.push(origin);
        self.agreement.push(Agreement::new(Number::Unknown, Gender::Unknown, Animacy::Unknown, Person::Unknown));
        self.spans.push(span)
    }

    /// The span of mention `i` in source bytes, through `doc`'s views.
    pub fn source_span(&self, doc: &Document, i: Ix<Mentions>) -> Span {
        doc.to_source_bytes(self.spans.view(), self.spans.get(i))
    }

    /// The fragments of mention `i` in source bytes.
    pub fn source_fragments(&self, doc: &Document, i: Ix<Mentions>) -> Vec<Span> {
        self.spans.fragments(i).map(|f| doc.to_source_bytes(self.spans.view(), f)).collect()
    }

    /// The span of mention `i`'s head token.
    pub fn head_span(&self, i: Ix<Mentions>, tokens: &Tokens) -> Span {
        tokens.spans.get(self.head[i.get()])
    }

    /// The entity mention `i` was lifted from, if any.
    pub fn entity_of(&self, i: Ix<Mentions>) -> Option<Ix<Entities>> {
        self.entity.as_ref().and_then(|e| e[i.get()])
    }
}

impl Layer for Mentions {
    const ID: &'static str = "opennlp:mentions";
    const OVERLAP: Overlap = Overlap::Any;

    fn len(&self) -> usize {
        self.spans.len()
    }

    fn text_spans(&self) -> Option<TextSpans<'_>> {
        Some(self.spans.text_spans())
    }

    fn indexes(&self) -> Vec<(KeyId, usize)> {
        let tokens =
            self.tokens.iter().map(|r| r.end.get()).chain(self.head.iter().map(|h| h.get() + 1)).max().unwrap_or(0);
        let sentences = self.sentence.iter().map(|s| s.get() + 1).max().unwrap_or(0);
        let mut out = vec![(Key::<Tokens>::DEFAULT.id(), tokens), (Key::<Sentences>::DEFAULT.id(), sentences)];
        if let Some(entity) = &self.entity {
            let bound = entity.iter().flatten().map(|e| e.get() + 1).max().unwrap_or(0);
            out.push((Key::<Entities>::DEFAULT.id(), bound));
        }
        out
    }

    fn check(&self) -> Result<(), String> {
        let n = self.spans.len();
        if self.sentence.len() != n
            || self.tokens.len() != n
            || self.head.len() != n
            || self.kind.len() != n
            || self.origin.len() != n
            || self.agreement.len() != n
        {
            return Err("a sentence, tokens, a head, a kind, an origin and an agreement for every mention".into());
        }
        if self.entity.as_ref().is_some_and(|e| e.len() != n) {
            return Err("an entity slot for every mention".into());
        }
        for i in 0..n {
            let range = self.tokens[i];
            if range.start >= range.end {
                return Err(format!("mention {i} has no token"));
            }
            let head = self.head[i];
            if head < range.start || head >= range.end {
                return Err(format!(
                    "mention {i} has head token {} outside its tokens {}..{}",
                    head.get(),
                    range.start.get(),
                    range.end.get()
                ));
            }
            check_form(i, self.origin[i], self.kind[i])?;
            let entity = self.entity.as_ref().and_then(|e| e[i]);
            if self.origin[i] == Origin::Entity && self.entity.is_some() && entity.is_none() {
                return Err(format!("entity mention {i} names no entity"));
            }
            if entity.is_some() && self.origin[i] != Origin::Entity {
                return Err(format!("mention {i} names an entity but was not lifted from one"));
            }
        }
        self.spans.check(Self::OVERLAP)
    }
}

/// The forms each origin can take: an entity is a proper name, a noun
/// phrase is proper or nominal, a pronoun is pronominal or demonstrative;
/// a gold mention may be any.
fn check_form(i: usize, origin: Origin, kind: MentionKind) -> Result<(), String> {
    let fits = match origin {
        Origin::Entity => kind == MentionKind::Proper,
        Origin::NounPhrase => matches!(kind, MentionKind::Proper | MentionKind::Nominal),
        Origin::Pronoun => matches!(kind, MentionKind::Pronominal | MentionKind::Demonstrative),
        Origin::Gold => true,
    };
    if fits { Ok(()) } else { Err(format!("mention {i} is a {} mention of kind {kind:?}", origin.as_str())) }
}

/// Speaker turns (`opennlp:speakers`, `CorefAnnotator.SPEAKERS`): each
/// span covers the text one speaker says, labelled with the speaker. The
/// spans are in text order and do not overlap.
#[derive(Debug)]
pub struct Speakers {
    pub spans: Spans<Speakers>,
    pub speaker: Vec<Tag>,
    pub speakers: Arc<TagSet>,
}

impl Layer for Speakers {
    const ID: &'static str = "opennlp:speakers";

    fn len(&self) -> usize {
        self.spans.len()
    }

    fn text_spans(&self) -> Option<TextSpans<'_>> {
        Some(self.spans.text_spans())
    }

    fn check(&self) -> Result<(), String> {
        if self.speaker.len() != self.spans.len() {
            return Err("a speaker for every turn".into());
        }
        if self.speaker.iter().any(|t| t.0 as usize >= self.speakers.len()) {
            return Err(format!("a speaker outside {}", self.speakers.name));
        }
        if self.spans.iter().any(|(_, s)| s.is_empty()) {
            return Err("a speaker turn that covers no text".into());
        }
        self.spans.check(Self::OVERLAP)
    }
}

#[cfg(test)]
mod tests;
