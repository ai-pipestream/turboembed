//! Borrowed text inspection over a composed alignment. Runs partition both
//! byte domains; an insertion or deletion has an empty span on one side.
//! Constructing the column table owns offsets and step metadata, never a
//! string per run. Iteration and text slices allocate nothing.

mod projection;
pub use projection::XrayProjection;

use crate::{Alignment, Document, Error, Key, Span, View, ViewId};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum XrayRunKind {
    Equal = 0,
    Replaced = 1,
    Inserted = 2,
    Deleted = 3,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct XrayRun {
    pub parent: Span,
    pub child: Span,
    pub kind: XrayRunKind,
}

/// The actual stored provenance of a view on the path, including the
/// origin. On a sibling path, entries ascend to the common ancestor and
/// then descend. `parent` always names the view's original parent.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ViewStepSummary {
    pub view: ViewId,
    pub parent: Option<ViewId>,
    pub instance: String,
    pub annotator: &'static str,
    pub version: &'static str,
    pub config_hash: u64,
    pub edit_count: u32,
    pub parent_bytes: u32,
    pub child_bytes: u32,
}

#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct NormalizationXray {
    parent_start: Vec<u32>,
    parent_end: Vec<u32>,
    child_start: Vec<u32>,
    child_end: Vec<u32>,
    kind: Vec<XrayRunKind>,
    steps: Vec<ViewStepSummary>,
    parent_bytes: u32,
    child_bytes: u32,
}

impl NormalizationXray {
    /// Inspect any views in one document, including reverse and sibling
    /// paths. The stored alignment supplies the edits; unchanged gaps are
    /// checked against the text rather than inferred from equal lengths.
    pub fn between(doc: &Document, from: ViewId, to: ViewId) -> Result<Self, Error> {
        let alignment = doc.view_alignment(from, to)?;
        let mut out = Self::build(doc.text(from).as_bytes(), doc.text(to).as_bytes(), &alignment)?;
        out.steps = summaries(doc, from, to)?;
        Ok(out)
    }

    /// Include the decoder's original-byte alignment below view zero.
    /// Parent offsets may address a non-UTF-8 encoding; use
    /// `parent_bytes_slice` for that domain. Step zero keeps the source
    /// view's recorded provenance, with the decoder's lengths/edit count.
    pub fn from_source_bytes(doc: &Document, to: ViewId) -> Result<Self, Error> {
        let aligned = doc.view_alignment(ViewId::SOURCE, to)?;
        let alignment = match doc.source_bytes() {
            Some(bytes) => bytes.alignment().compose(&aligned),
            None => aligned,
        };
        let mut out = Self::build(doc.bytes(), doc.text(to).as_bytes(), &alignment)?;
        out.steps = summaries(doc, ViewId::SOURCE, to)?;
        if let Some(bytes) = doc.source_bytes() {
            out.steps[0].edit_count = length(bytes.alignment().edits().len())?;
            out.steps[0].parent_bytes = out.parent_bytes;
        }
        Ok(out)
    }

    fn build(parent: &[u8], child: &[u8], alignment: &Alignment) -> Result<Self, Error> {
        let mut out =
            Self { parent_bytes: length(parent.len())?, child_bytes: length(child.len())?, ..Self::default() };
        let (mut p, mut c) = (0, 0);
        for e in alignment.edits() {
            if e.parent.start < p
                || e.child.start < c
                || e.parent.end < e.parent.start
                || e.child.end < e.child.start
                || e.parent.end > out.parent_bytes
                || e.child.end > out.child_bytes
            {
                return Err(invalid("alignment edit outside its ordered byte domains"));
            }
            out.gap(parent, child, Span::new(p, e.parent.start), Span::new(c, e.child.start))?;
            let kind = match (e.parent.is_empty(), e.child.is_empty()) {
                (true, true) => None,
                (true, false) => Some(XrayRunKind::Inserted),
                (false, true) => Some(XrayRunKind::Deleted),
                (false, false) => Some(XrayRunKind::Replaced),
            };
            if let Some(kind) = kind {
                out.push(e.parent, e.child, kind);
            }
            (p, c) = (e.parent.end, e.child.end);
        }
        out.gap(parent, child, Span::new(p, out.parent_bytes), Span::new(c, out.child_bytes))?;
        Ok(out)
    }

    fn gap(&mut self, p: &[u8], c: &[u8], parent: Span, child: Span) -> Result<(), Error> {
        if p[parent.range()] != c[child.range()] {
            return Err(invalid("unchanged alignment gap has different bytes"));
        }
        if !parent.is_empty() {
            self.push(parent, child, XrayRunKind::Equal);
        }
        Ok(())
    }

    fn push(&mut self, p: Span, c: Span, kind: XrayRunKind) {
        self.parent_start.push(p.start);
        self.parent_end.push(p.end);
        self.child_start.push(c.start);
        self.child_end.push(c.end);
        self.kind.push(kind);
    }

    pub fn len(&self) -> usize {
        self.kind.len()
    }
    pub fn is_empty(&self) -> bool {
        self.kind.is_empty()
    }
    pub fn parent_start(&self) -> &[u32] {
        &self.parent_start
    }
    pub fn parent_end(&self) -> &[u32] {
        &self.parent_end
    }
    pub fn child_start(&self) -> &[u32] {
        &self.child_start
    }
    pub fn child_end(&self) -> &[u32] {
        &self.child_end
    }
    pub fn kinds(&self) -> &[XrayRunKind] {
        &self.kind
    }
    pub fn steps(&self) -> &[ViewStepSummary] {
        &self.steps
    }

    pub fn run(&self, i: usize) -> Option<XrayRun> {
        Some(XrayRun {
            parent: Span::new(*self.parent_start.get(i)?, self.parent_end[i]),
            child: Span::new(self.child_start[i], self.child_end[i]),
            kind: self.kind[i],
        })
    }

    pub fn iter(&self) -> impl ExactSizeIterator<Item = XrayRun> + '_ {
        (0..self.len()).map(|i| self.run(i).expect("parallel private columns"))
    }

    /// Borrow the run's parent text, or None for an invalid index or a
    /// parent byte range that is not a UTF-8 character range in `text`.
    pub fn parent_slice<'a>(&self, i: usize, text: &'a str) -> Option<&'a str> {
        text.get(self.run(i)?.parent.range())
    }

    pub fn child_slice<'a>(&self, i: usize, text: &'a str) -> Option<&'a str> {
        text.get(self.run(i)?.child.range())
    }

    pub fn parent_bytes_slice<'a>(&self, i: usize, bytes: &'a [u8]) -> Option<&'a [u8]> {
        bytes.get(self.run(i)?.parent.range())
    }
}

fn invalid(detail: impl Into<String>) -> Error {
    Error::Invalid { layer: Key::<View>::DEFAULT.id(), detail: detail.into() }
}

fn length(n: usize) -> Result<u32, Error> {
    u32::try_from(n).map_err(|_| invalid("xray byte domain exceeds u32 offsets"))
}

fn summaries(doc: &Document, from: ViewId, to: ViewId) -> Result<Vec<ViewStepSummary>, Error> {
    let lca = doc.lca_view(from, to)?;
    let mut path = vec![from];
    let mut at = from;
    while at != lca {
        at = doc.views[at.0 as usize].parent.expect("ancestor path");
        path.push(at);
    }
    let mut down = Vec::new();
    at = to;
    while at != lca {
        down.push(at);
        at = doc.views[at.0 as usize].parent.expect("ancestor path");
    }
    path.extend(down.into_iter().rev());
    // One directory pass, rather than searching every layer per step.
    let mut entries = vec![None; doc.views.len()];
    for e in doc.entries() {
        if let Some(view) = e.value.as_ref().and_then(|v| v.downcast_ref::<View>()) {
            entries[view.id().0 as usize] = Some(e);
        }
    }
    path.into_iter()
        .map(|id| {
            let v = &doc.views[id.0 as usize];
            let e = entries[id.0 as usize].ok_or_else(|| invalid("view provenance is missing"))?;
            Ok(ViewStepSummary {
                view: id,
                parent: v.parent,
                instance: e.key.name.into(),
                annotator: e.provenance.annotator,
                version: e.provenance.version,
                config_hash: e.provenance.config,
                edit_count: length(v.alignment.edits().len())?,
                parent_bytes: length(v.parent.map_or(v.text.len(), |p| doc.text(p).len()))?,
                child_bytes: length(v.text.len())?,
            })
        })
        .collect()
}
