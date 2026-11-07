//! The view edit chain on the wire (issue #1907): every view's native
//! `Alignment` to its parent as runs of equal-width edits in canonical
//! UTF-8 bytes (`ViewChain`, `EditRun`), the encoding from a `Document`
//! and the decoding back to the native edit list byte for byte, and a
//! consumer, [`WireChain`], that maps spans between views from the wire
//! alone with the stepwise rules of `Document::map_span`.

use std::any::TypeId;

use turbo_analysis::{Alignment, Document, Edit, Key, Span, View, ViewId};

use super::bad;
use crate::api::Failure;
use crate::proto::result::{EditRun, InlineMapping, MappingPayload, ViewChain, mapping_payload};
use crate::results::{MAX_MAPPING_EDITS, over};

/// The version of the chain encoding and mapping rules `InlineMapping`
/// carries.
pub const MAPPING_SCHEMA_VERSION: u32 = 1;

/// `edits` as runs: consecutive edits of one parent width and one child
/// width, each starting where the one before it ends on both sides,
/// become one run, so a masked view's one edit per character costs one
/// run per stretch of equal-width characters. [`decode_runs`] gives the
/// list back exactly.
pub fn encode_runs(edits: &[Edit]) -> Vec<EditRun> {
    let mut runs: Vec<EditRun> = Vec::new();
    for e in edits {
        let (pw, cw) = (e.parent.end - e.parent.start, e.child.end - e.child.start);
        if let Some(run) = runs.last_mut()
            && run.parent_width == pw
            && run.child_width == cw
            && u64::from(run.parent_start) + u64::from(run.count) * u64::from(pw) == u64::from(e.parent.start)
            && u64::from(run.child_start) + u64::from(run.count) * u64::from(cw) == u64::from(e.child.start)
        {
            run.count += 1;
            continue;
        }
        runs.push(EditRun {
            parent_start: e.parent.start,
            parent_width: pw,
            child_start: e.child.start,
            child_width: cw,
            count: 1,
        });
    }
    runs
}

/// The native edit list `runs` encode, with every offset computed in
/// checked arithmetic: a run whose edits leave `u32` is refused, and so
/// is one that changes nothing or has no edits.
pub fn decode_runs(what: &str, runs: &[EditRun]) -> Result<Vec<Edit>, Failure> {
    let mut edits = Vec::new();
    for (i, run) in runs.iter().enumerate() {
        if run.count == 0 {
            return Err(bad(format!("{what} run {i} has no edits")));
        }
        if run.parent_width == 0 && run.child_width == 0 {
            return Err(bad(format!("{what} run {i} changes nothing")));
        }
        let total = edits.len() as u64 + run.count as u64;
        if total > MAX_MAPPING_EDITS {
            return Err(over("max_mapping_edits", "typed document mapping edits", total));
        }
        let (mut ps, mut cs) = (run.parent_start, run.child_start);
        for _ in 0..run.count {
            let pe = ps.checked_add(run.parent_width).ok_or_else(|| bad(format!("{what} run {i} leaves u32")))?;
            let ce = cs.checked_add(run.child_width).ok_or_else(|| bad(format!("{what} run {i} leaves u32")))?;
            edits.push(Edit { parent: Span::new(ps, pe), child: Span::new(cs, ce) });
            (ps, cs) = (pe, ce);
        }
    }
    Ok(edits)
}

/// Every view of `doc` that is there, in `ViewId` order, as the wire
/// chain: a document whose views hold more than `max_mapping_edits`
/// native edits in all is refused, never cut down.
pub fn inline_mapping(doc: &Document) -> Result<InlineMapping, Failure> {
    let mut views: Vec<ViewChain> = Vec::new();
    let mut total = 0u64;
    for key in doc.layers() {
        if key.ty != TypeId::of::<View>() {
            continue;
        }
        let Some(view) = doc.get_key(&Key::<View>::named(key.name)) else {
            continue;
        };
        let edits = view.alignment().edits();
        total += edits.len() as u64;
        if total > MAX_MAPPING_EDITS {
            return Err(over("max_mapping_edits", "typed document mapping edits", total));
        }
        views.push(ViewChain {
            view: u32::from(view.id().0),
            instance: key.name.to_owned(),
            parent: view.parent().map(|p| u32::from(p.0)),
            text_digest: view.text_digest().as_bytes().to_vec(),
            text_len: view.text().len() as u32,
            edit_count: edits.len() as u32,
            edits: encode_runs(edits),
        });
    }
    views.sort_by_key(|v| v.view);
    Ok(InlineMapping { mapping_schema_version: MAPPING_SCHEMA_VERSION, views })
}

/// One view rebuilt from the wire.
#[derive(Clone, Debug)]
pub struct ChainView {
    pub id: ViewId,
    pub instance: String,
    pub parent: Option<ViewId>,
    pub text_digest: [u8; 32],
    pub text_len: u32,
    pub alignment: Alignment,
}

/// The view tree rebuilt from an `InlineMapping`, mapping spans between
/// any two of its views exactly as `Document::map_span` does: up from
/// `from` to the lowest common ancestor one `span_to_parent` at a time,
/// then down to `to` one `span_to_child` at a time. A composed net
/// alignment is never used.
#[derive(Clone, Debug)]
pub struct WireChain {
    views: Vec<ChainView>,
}

impl WireChain {
    /// Rebuilds the chain from `mapping`, refusing an absent or reserved
    /// payload, an unknown schema version, a view out of order or with a
    /// parent that is not before it, a source view that is named or a
    /// derived view that is not or shares its name, or edits that are not
    /// an alignment.
    pub fn from_payload(mapping: Option<&MappingPayload>) -> Result<WireChain, Failure> {
        let Some(mapping_payload::Payload::Inline(inline)) = mapping.and_then(|m| m.payload.as_ref()) else {
            return Err(bad("the typed document carries no inline mapping"));
        };
        WireChain::from_inline(inline)
    }

    pub fn from_inline(inline: &InlineMapping) -> Result<WireChain, Failure> {
        if inline.mapping_schema_version != MAPPING_SCHEMA_VERSION {
            return Err(bad(format!(
                "mapping schema version {} is not the served {MAPPING_SCHEMA_VERSION}",
                inline.mapping_schema_version
            )));
        }
        let mut views: Vec<ChainView> = Vec::with_capacity(inline.views.len());
        let mut total = 0u64;
        for (i, v) in inline.views.iter().enumerate() {
            let what = format!("mapping view {}", v.view);
            if v.view as usize != i {
                return Err(bad(format!("{what} is listed at position {i}; views are listed in id order from 0")));
            }
            let id = ViewId(u16::try_from(v.view).map_err(|_| bad(format!("{what} exceeds u16::MAX")))?);
            let parent = match v.parent {
                None if i == 0 => None,
                None => return Err(bad(format!("{what} has no parent and is not the source view"))),
                Some(_) if i == 0 => return Err(bad("the source view has a parent")),
                Some(p) if (p as usize) < i => Some(ViewId(p as u16)),
                Some(p) => return Err(bad(format!("{what} names parent {p}, which is not before it"))),
            };
            let text_digest: [u8; 32] = v
                .text_digest
                .as_slice()
                .try_into()
                .map_err(|_| bad(format!("{what} text_digest is {} bytes, not 32", v.text_digest.len())))?;
            let edits = decode_runs(&what, &v.edits)?;
            total += edits.len() as u64;
            if total > MAX_MAPPING_EDITS {
                return Err(over("max_mapping_edits", "typed document mapping edits", total));
            }
            if edits.len() as u64 != u64::from(v.edit_count) {
                return Err(bad(format!("{what} says {} edits and its runs hold {}", v.edit_count, edits.len())));
            }
            let (parent_len, child_len) = (parent.map(|p| views[p.0 as usize].text_len), v.text_len);
            if let Some(last) = edits.last() {
                if last.child.end > child_len {
                    return Err(bad(format!("{what} edits end at {}, past its {child_len} bytes", last.child.end)));
                }
                if let Some(pl) = parent_len
                    && last.parent.end > pl
                {
                    return Err(bad(format!("{what} edits end at {} in its parent, past {pl} bytes", last.parent.end)));
                }
            }
            if parent.is_none() && !edits.is_empty() {
                return Err(bad("the source view has edits"));
            }
            let alignment = Alignment::from_edits(edits).map_err(|e| bad(format!("{what}: {e}")))?;
            if let Some(pl) = parent_len
                && !lengths_agree(&alignment, pl, child_len)
            {
                return Err(bad(format!("{what} edits do not turn {pl} parent bytes into {child_len} bytes")));
            }
            if i == 0 && !v.instance.is_empty() {
                return Err(bad(format!(
                    "the source view is instance {:?}; it is the default, unnamed instance",
                    v.instance
                )));
            }
            if i > 0 && v.instance.is_empty() {
                return Err(bad(format!("{what} has no instance name")));
            }
            if views.iter().any(|seen| seen.instance == v.instance) {
                return Err(bad(format!("duplicate mapping view instance {:?}", v.instance)));
            }
            views.push(ChainView {
                id,
                instance: v.instance.clone(),
                parent,
                text_digest,
                text_len: v.text_len,
                alignment,
            });
        }
        if views.is_empty() {
            return Err(bad("the mapping lists no views"));
        }
        Ok(WireChain { views })
    }

    pub fn views(&self) -> &[ChainView] {
        &self.views
    }

    pub fn view(&self, id: ViewId) -> Result<&ChainView, Failure> {
        self.views
            .get(id.0 as usize)
            .ok_or_else(|| bad(format!("view {} is outside the {} views", id.0, self.views.len())))
    }

    /// The view named `instance`, if the chain has it.
    pub fn by_instance(&self, instance: &str) -> Option<&ChainView> {
        self.views.iter().find(|v| v.instance == instance)
    }

    fn path(&self, from: ViewId, to: ViewId) -> Result<(Vec<&ChainView>, Vec<&ChainView>), Failure> {
        let mut up: Vec<&ChainView> = vec![self.view(from)?];
        while let Some(p) = up.last().expect("one view").parent {
            up.push(self.view(p)?);
        }
        let mut down: Vec<&ChainView> = Vec::new();
        let mut cur = self.view(to)?;
        loop {
            if let Some(pos) = up.iter().position(|v| v.id == cur.id) {
                up.truncate(pos);
                down.reverse();
                return Ok((up, down));
            }
            let Some(p) = cur.parent else {
                return Err(bad(format!("views {} and {} share no ancestor", from.0, to.0)));
            };
            down.push(cur);
            cur = self.view(p)?;
        }
    }

    /// Maps byte span `s` of view `from` into view `to`, stepwise along
    /// the lowest-common-ancestor path. Only the bounds of `s` are
    /// checked: the chain holds byte lengths, not every view's text, so
    /// a span that splits a UTF-8 scalar, which `Document::map_span`
    /// refuses, is mapped here and refused only where the text is.
    pub fn map_span(&self, from: ViewId, to: ViewId, mut s: Span) -> Result<Span, Failure> {
        if s.start > s.end || s.end > self.view(from)?.text_len {
            return Err(bad(format!("span [{}, {}) is outside view {}", s.start, s.end, from.0)));
        }
        let (up, down) = self.path(from, to)?;
        for v in up {
            s = v.alignment.span_to_parent(s);
        }
        for v in down {
            s = v.alignment.span_to_child(s);
        }
        Ok(s)
    }
}

/// Whether `alignment` turns `parent_len` bytes into `child_len` bytes:
/// the unchanged stretches are equal on both sides, so the lengths differ
/// by exactly what the edits add and remove.
fn lengths_agree(alignment: &Alignment, parent_len: u32, child_len: u32) -> bool {
    let (mut added, mut removed) = (0u64, 0u64);
    for e in alignment.edits() {
        added += u64::from(e.child.end - e.child.start);
        removed += u64::from(e.parent.end - e.parent.start);
    }
    u64::from(parent_len) + added == u64::from(child_len) + removed
}
