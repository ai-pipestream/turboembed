//! What a PII rewrite changed (`turbo:pii-edits`): one item per
//! replacement, over the derived view the rewrite wrote.
//!
//! Each item holds its span in the derived view, its source span in that
//! view's parent, the mentions of the `turbo:pii` layer it covers (any
//! subset, so overlapping and non-contiguous members are listed one by
//! one), the rule of the policy that chose its strategy, and the
//! strategy's kind. No item holds the original value, and the layer holds
//! no text of its own: the replacement is in the derived view.
//!
//! The derived view's alignment is the one authoritative chain. The
//! document refuses an edits layer whose spans disagree with it, or whose
//! members are not exactly the mentions under each source span.

use std::any::TypeId;

use turbo_analysis::{Document, Ix, Key, KeyId, Layer, Span, Spans, View, ViewId};

use crate::{Pii, PiiKind};

/// How a replacement was written. The ids are stable wire numbers; 0 is
/// never a kind.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
#[repr(u8)]
pub enum PiiStrategyKind {
    Mask = 1,
    Pseudonym = 2,
    Hmac = 3,
    Format = 4,
    Static = 5,
    Sha256 = 6,
    Map = 7,
    Truncate = 8,
    Abbreviate = 9,
}

impl PiiStrategyKind {
    /// Every kind, in id order.
    pub const ALL: [PiiStrategyKind; 9] = [
        PiiStrategyKind::Mask,
        PiiStrategyKind::Pseudonym,
        PiiStrategyKind::Hmac,
        PiiStrategyKind::Format,
        PiiStrategyKind::Static,
        PiiStrategyKind::Sha256,
        PiiStrategyKind::Map,
        PiiStrategyKind::Truncate,
        PiiStrategyKind::Abbreviate,
    ];

    /// The stable id.
    pub const fn id(self) -> u8 {
        self as u8
    }

    /// The kind with stable id `id`; `None` for 0 and unknown ids.
    pub fn from_id(id: u8) -> Option<PiiStrategyKind> {
        PiiStrategyKind::ALL.get(usize::from(id).checked_sub(1)?).copied()
    }

    /// The lowercase name the wire and docs use.
    pub const fn name(self) -> &'static str {
        match self {
            PiiStrategyKind::Mask => "mask",
            PiiStrategyKind::Pseudonym => "pseudonym",
            PiiStrategyKind::Hmac => "hmac",
            PiiStrategyKind::Format => "format",
            PiiStrategyKind::Static => "static",
            PiiStrategyKind::Sha256 => "sha256",
            PiiStrategyKind::Map => "map",
            PiiStrategyKind::Truncate => "truncate",
            PiiStrategyKind::Abbreviate => "abbreviate",
        }
    }
}

/// Which rule of a policy chose an edit's strategy: the policy's default,
/// or its override for one kind.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum PiiRule {
    /// The default strategy, for a kind the policy does not override, and
    /// for every edit of a single-strategy rewrite.
    Default,
    /// The override for this kind, named in the custom-kind table of the
    /// pii layer the edits cover.
    Kind(PiiKind),
}

impl PiiRule {
    /// The rule with wire code `code`; a custom kind's index is checked
    /// against its table by the document.
    pub fn from_code(code: u32) -> Option<PiiRule> {
        if code == 0 {
            return Some(PiiRule::Default);
        }
        if code < crate::CUSTOM_CODE_BASE {
            return PiiKind::from_id(u16::try_from(code).ok()?).map(PiiRule::Kind);
        }
        let index = u16::try_from(code - crate::CUSTOM_CODE_BASE).ok()?;
        Some(PiiRule::Kind(PiiKind::Custom(crate::CustomKindId::new(index))))
    }

    /// The wire code: 0 for [`PiiRule::Default`], else the kind's
    /// [`PiiKind::code`], which is never 0.
    pub fn code(self) -> u32 {
        match self {
            PiiRule::Default => 0,
            PiiRule::Kind(k) => k.code(),
        }
    }
}

/// One edit, as handed to [`PiiEdits::push`].
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PiiEdit<'a> {
    /// The replacement's bytes in the derived view; empty where the
    /// rewrite deleted the text.
    pub span: Span,
    /// The replaced bytes in the derived view's parent.
    pub source: Span,
    /// The covered mentions of the pii layer, ascending.
    pub members: &'a [Ix<Pii>],
    pub rule: PiiRule,
    pub strategy: PiiStrategyKind,
}

/// Every replacement one rewrite made, in text order.
pub struct PiiEdits {
    pub spans: Spans<PiiEdits>,
    /// Each edit's source span in the parent of [`PiiEdits::spans`]'s view.
    pub source: Vec<Span>,
    /// One past each edit's last entry of [`PiiEdits::members`].
    pub member_end: Vec<u32>,
    /// The covered mentions, edit by edit, ascending within an edit.
    pub members: Vec<Ix<Pii>>,
    pub rule: Vec<PiiRule>,
    pub strategy: Vec<PiiStrategyKind>,
    /// The revision of the policy that made every edit, as its caller
    /// numbers it; 0 when the caller gave none.
    pub policy_revision: u64,
}

impl PiiEdits {
    /// An empty layer over `view`, the derived view the edits wrote, for a
    /// policy at `policy_revision`.
    pub fn new(view: ViewId, policy_revision: u64) -> PiiEdits {
        PiiEdits {
            spans: Spans::new(view),
            source: Vec::new(),
            member_end: Vec::new(),
            members: Vec::new(),
            rule: Vec::new(),
            strategy: Vec::new(),
            policy_revision,
        }
    }

    /// Appends one edit to every column.
    pub fn push(&mut self, e: PiiEdit<'_>) -> Ix<PiiEdits> {
        self.source.push(e.source);
        self.members.extend_from_slice(e.members);
        self.member_end.push(u32::try_from(self.members.len()).expect("fewer than 2^32 members"));
        self.rule.push(e.rule);
        self.strategy.push(e.strategy);
        self.spans.push(e.span)
    }

    /// The covered mentions of edit `i`.
    pub fn members_of(&self, i: Ix<PiiEdits>) -> &[Ix<Pii>] {
        let end = self.member_end[i.get()] as usize;
        let start = i.get().checked_sub(1).map_or(0, |p| self.member_end[p] as usize);
        &self.members[start..end]
    }

    /// Edit `i`, column by column.
    pub fn get(&self, i: Ix<PiiEdits>) -> PiiEdit<'_> {
        PiiEdit {
            span: self.spans.get(i),
            source: self.source[i.get()],
            members: self.members_of(i),
            rule: self.rule[i.get()],
            strategy: self.strategy[i.get()],
        }
    }

    /// Every edit, in text order.
    pub fn edits(&self) -> impl Iterator<Item = PiiEdit<'_>> {
        self.spans.iter().map(|(i, _)| self.get(i))
    }

    /// This layer moved onto `view`, every span kept.
    fn moved_to(&self, view: ViewId) -> Spans<PiiEdits> {
        let mut spans = Spans::with_capacity(view, self.spans.len());
        for (_, s) in self.spans.iter() {
            spans.push(s);
        }
        spans
    }
}

/// The view with id `id` among `doc`'s view layers.
fn view_by_id(doc: &Document, id: ViewId) -> Option<&View> {
    let ty = TypeId::of::<View>();
    doc.layers().filter(|k| k.ty == ty).filter_map(|k| doc.get_key(&Key::<View>::named(k.name))).find(|v| v.id() == id)
}

impl Layer for PiiEdits {
    const ID: &'static str = "turbo:pii-edits";
    positional!(PiiEdits);

    fn indexes(&self) -> Vec<(KeyId, usize)> {
        match self.members.iter().map(|m| m.get() + 1).max() {
            Some(bound) => vec![(Key::<Pii>::DEFAULT.id(), bound)],
            None => Vec::new(),
        }
    }

    fn bind_pending_view(&mut self, view: ViewId) {
        if self.spans.view() == ViewId::PENDING {
            self.spans = self.moved_to(view);
        }
    }

    fn content_hash(&self) -> Option<u64> {
        let mut h = turbo_analysis::Fnv1a::new();
        h.usize(self.source.len()).u64(self.policy_revision);
        for (i, e) in self.edits().enumerate() {
            h.usize(i).u32(e.source.start).u32(e.source.end).u32(e.rule.code()).u8(e.strategy.id());
            h.usize(e.members.len());
            for m in e.members {
                h.usize(m.get());
            }
        }
        Some(h.finish())
    }

    fn check(&self) -> Result<(), String> {
        let n = self.spans.len();
        if self.source.len() != n || self.member_end.len() != n || self.rule.len() != n || self.strategy.len() != n {
            return Err("every column as long as the spans".into());
        }
        if self.member_end.last().map_or(0, |&e| e as usize) != self.members.len() {
            return Err("member ends must end at the member count".into());
        }
        if !self.spans.is_ordered() {
            return Err("edits in text order, not overlapping".into());
        }
        let mut last_end = 0u32;
        let mut start = 0usize;
        for (i, &end) in self.member_end.iter().enumerate() {
            let end = end as usize;
            if end <= start || end > self.members.len() {
                return Err(format!("pii edit {i} covers no mention"));
            }
            if self.members[start..end].windows(2).any(|w| w[0] >= w[1]) {
                return Err(format!("pii edit {i} lists its mentions out of order or twice"));
            }
            let s = self.source[i];
            if s.is_empty() || s.start > s.end {
                return Err(format!("pii edit {i} has an empty source span"));
            }
            if s.start < last_end {
                return Err(format!("pii edit {i} overlaps the edit before it in the parent"));
            }
            last_end = s.end;
            start = end;
        }
        Ok(())
    }

    /// The cross-check against the authoritative chain: the view is a
    /// derived one, every edit's span is its source span mapped to the
    /// view and back, the pii layer is over the parent, and each edit's
    /// members are exactly the mentions overlapping its source span.
    fn check_document(&self, doc: &Document, on: &[KeyId]) -> Result<(), String> {
        let view = view_by_id(doc, self.spans.view()).ok_or("pii edits must be over a view of the document")?;
        let parent = view.parent().ok_or("pii edits must be over a derived view, not the source")?;
        let alignment = view.alignment();
        for (i, e) in self.edits().enumerate() {
            if alignment.span_to_child(e.source) != e.span || alignment.span_to_parent(e.span) != e.source {
                return Err(format!("pii edit {i} disagrees with the view's alignment"));
            }
        }
        let pii_key = on.iter().find(|k| k.ty == TypeId::of::<Pii>());
        let pii = match pii_key {
            Some(k) => doc.get_key(&Key::<Pii>::named(k.name)),
            None if self.members.is_empty() => return Ok(()),
            None => doc.get::<Pii>(),
        };
        let pii = pii.ok_or("pii edits need the pii layer they cover")?;
        if pii.spans.view() != parent {
            return Err("pii edits must cover a pii layer over the view's parent".into());
        }
        // Source spans are ordered and disjoint, so the edits a mention
        // overlaps are one run, found by binary search; mentions come in
        // index order, so each edit's list comes out ascending.
        let mut under: Vec<Vec<Ix<Pii>>> = vec![Vec::new(); self.source.len()];
        for (ix, m) in pii.spans.iter() {
            let first = self.source.partition_point(|s| s.end <= m.start);
            for (k, s) in self.source.iter().enumerate().skip(first) {
                if s.start >= m.end {
                    break;
                }
                under[k].push(ix);
            }
        }
        for (i, e) in self.edits().enumerate() {
            if let PiiRule::Kind(k) = e.rule
                && pii.custom_kinds.check(k).is_err()
            {
                return Err(format!("pii edit {i} names a rule kind outside the pii layer's table"));
            }
            if under[i] != e.members {
                return Err(format!("pii edit {i} does not list exactly the mentions under its source span"));
            }
        }
        Ok(())
    }
}

#[cfg(test)]
#[path = "pii_edits_tests.rs"]
mod tests;
