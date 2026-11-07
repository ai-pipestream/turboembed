//! Personal and sensitive identifiers written in a text view: e-mail
//! addresses, phone numbers, payment and bank account numbers, network
//! addresses, credentials, wallet addresses, national and device
//! identifiers.
//!
//! The layer holds what OpenNLP's PII extractors find (`PiiPacks` in the
//! `opennlp.tools.pii` package): one item per mention with its kind, the
//! normalized form OpenNLP writes, and how strongly the text supports the
//! kind. Mentions of different kinds may overlap, as in the Java, so the
//! layer allows any overlap.

mod confidence;
mod custom;
mod kind;

pub use confidence::PiiConfidence;
pub use custom::{CustomKindId, CustomKinds, KindError};
pub use kind::{CUSTOM_CODE_BASE, KindRow, PiiKind, PiiKinds, mention_order};

use turbo_analysis::{Ix, Layer, Overlap, Span, Spans, ViewId};

use crate::Strs;

/// Every PII mention in a view, one item per mention, ordered by start,
/// then longer first, then by kind priority ([`mention_order`]). Equal
/// mentions (span, kind and normalized form) appear once; mentions of
/// different kinds, or of one kind with different normalized forms, may
/// share a span or overlap.
pub struct Pii {
    pub spans: Spans<Pii>,
    /// The table every [`PiiKind::Custom`] of [`Pii::kind`] indexes,
    /// exported with the layer.
    pub custom_kinds: CustomKinds,
    pub kind: Vec<PiiKind>,
    pub confidence: Vec<PiiConfidence>,
    /// The value as OpenNLP normalizes it: digits without separators, a
    /// lowercased domain, an RFC 5952 IPv6 address, an EIP-55 address.
    pub normalized: Strs,
}

/// One mention, as handed to [`Pii::push`].
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct PiiMention<'a> {
    pub span: Span,
    pub kind: PiiKind,
    pub confidence: PiiConfidence,
    pub normalized: &'a str,
}

impl Pii {
    /// An empty layer over `view`.
    pub fn new(view: ViewId) -> Pii {
        Pii::with_custom_kinds(view, CustomKinds::new())
    }

    /// An empty layer over `view` whose custom kinds index `custom_kinds`.
    pub fn with_custom_kinds(view: ViewId, custom_kinds: CustomKinds) -> Pii {
        Pii {
            spans: Spans::new(view),
            custom_kinds,
            kind: Vec::new(),
            confidence: Vec::new(),
            normalized: Strs::default(),
        }
    }

    /// The name of `kind` in this layer: the built-in name or the entry of
    /// [`Pii::custom_kinds`].
    pub fn kind_name(&self, kind: PiiKind) -> Option<&str> {
        self.custom_kinds.name(kind)
    }

    /// Appends one mention to every column. [`Layer::check`] refuses a
    /// custom kind outside [`Pii::custom_kinds`].
    pub fn push(&mut self, m: PiiMention<'_>) -> Ix<Pii> {
        self.kind.push(m.kind);
        self.confidence.push(m.confidence);
        self.normalized.push(m.normalized);
        self.spans.push(m.span)
    }

    /// Mention `i`, column by column.
    pub fn get(&self, i: Ix<Pii>) -> PiiMention<'_> {
        PiiMention {
            span: self.spans.get(i),
            kind: self.kind[i.get()],
            confidence: self.confidence[i.get()],
            normalized: self.normalized.get(i.get()),
        }
    }

    /// Every mention, in layer order.
    pub fn mentions(&self) -> impl Iterator<Item = PiiMention<'_>> {
        self.spans.iter().map(|(i, _)| self.get(i))
    }
}

impl Layer for Pii {
    const ID: &'static str = "turbo:pii";
    const OVERLAP: Overlap = Overlap::Any;
    positional!(Pii);

    fn content_hash(&self) -> Option<u64> {
        let mut h = turbo_analysis::Fnv1a::new();
        h.usize(self.kind.len()).u64(self.normalized.content_hash());
        h.usize(self.custom_kinds.len());
        for name in self.custom_kinds.names() {
            h.str(name);
        }
        for (&k, &c) in self.kind.iter().zip(&self.confidence) {
            h.u32(k.code()).u8(c.id());
        }
        Some(h.finish())
    }

    fn check(&self) -> Result<(), String> {
        let n = self.spans.len();
        if self.kind.len() != n || self.confidence.len() != n || self.normalized.len() != n {
            return Err("every column as long as the spans".into());
        }
        self.spans.check(Self::OVERLAP)?;
        let mut previous: Option<PiiMention<'_>> = None;
        for (i, m) in self.mentions().enumerate() {
            if m.span.is_empty() {
                return Err(format!("pii mention {i} is empty at {}", m.span.start));
            }
            if let Err(e) = self.custom_kinds.check(m.kind) {
                return Err(format!("pii mention {i}: {e}"));
            }
            if m.normalized.is_empty() {
                return Err(format!("pii mention {i} has no normalized form"));
            }
            let key = |x: PiiMention<'_>| mention_order(x.span.start, x.span.end, x.kind);
            if let Some(p) = previous
                && key(p) > key(m)
            {
                return Err(format!("pii mention {i} is out of order: by start, longer first, then by kind priority"));
            }
            // Mentions with one span and kind are adjacent, so a duplicate
            // is among the run of those just before.
            let duplicate =
                (0..i).rev().map(|j| self.get(Ix::new(j as u32))).take_while(|p| p.span == m.span && p.kind == m.kind);
            if duplicate.into_iter().any(|p| p.normalized == m.normalized) {
                return Err(format!("pii mention {i} repeats an earlier one"));
            }
            previous = Some(m);
        }
        Ok(())
    }
}

#[cfg(test)]
#[path = "pii_tests.rs"]
mod tests;
