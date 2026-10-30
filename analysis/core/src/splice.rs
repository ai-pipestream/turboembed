//! Carrying a layer's items from one document to another: the typed
//! contract a layer implements ([`Splice`]), the type-erased form the
//! analyzer drives ([`LayerSplicer`]), and the item ranges a partition
//! layer cuts a document into. `Analyzer::run_spliced` uses them to keep
//! the outputs of partition-local steps on the sentences an edit did not
//! touch.

use std::any::Any;
use std::marker::PhantomData;
use std::ops::Range;

use crate::annotate::made_on_own;
use crate::doc::Document;
use crate::error::Error;
use crate::layer::{Anchor, Key, KeyId, Layer, Provenance};
use crate::span::{Ix, IxRange, Span, Spans, TextSpans};
use crate::text::ViewId;

/// A layer whose items can be copied, a run at a time, into a layer of
/// the same type on another document. Spans go through
/// [`SpliceCx::span`], and every index into another layer (or into the
/// layer itself) through [`SpliceCx::ix`] or [`SpliceCx::ix_range`], so
/// the copy lands where the destination's items are. Payload columns are
/// copied as they are. A layer that implements this also overrides
/// [`Layer::splicer`] with `Some(Splicer::<Self>::boxed())`.
pub trait Splice: Layer {
    /// An empty layer shaped like `like`: its tag sets, the optional
    /// columns it has, and spans in `view`.
    fn empty_like(like: &Self, view: ViewId) -> Self;

    /// Appends items `range` of `from`. `Err` when the two cannot be one
    /// layer (another tag set, a column one has and the other lacks) or
    /// when an index leaves its unit; the step is then re-run whole.
    fn append(&mut self, from: &Self, range: Range<usize>, cx: &mut SpliceCx<'_>) -> Result<(), String>;
}

/// Where the items of one indexed layer that belong to the unit being
/// copied are in the source layer (`from`), and where the unit's first
/// item lands in the destination layer (`to`).
#[derive(Clone, Debug)]
pub struct Shift {
    pub key: KeyId,
    pub from: Range<usize>,
    pub to: usize,
}

/// Where a copied item's spans and indexes go.
pub struct SpliceCx<'a> {
    span: &'a mut dyn FnMut(Span) -> Span,
    shifts: &'a [Shift],
    /// The spliced layer's own key, for a layer that indexes itself.
    own: KeyId,
}

impl<'a> SpliceCx<'a> {
    pub fn new(span: &'a mut dyn FnMut(Span) -> Span, shifts: &'a [Shift], own: KeyId) -> SpliceCx<'a> {
        SpliceCx { span, shifts, own }
    }

    /// A span of the source item in the destination's text.
    pub fn span(&mut self, s: Span) -> Span {
        (self.span)(s)
    }

    /// Appends items `range` of `from` to `out`, every span and fragment
    /// mapped.
    pub fn spans<L>(&mut self, out: &mut Spans<L>, from: &Spans<L>, range: Range<usize>) -> Result<(), String> {
        for i in range {
            let ix = Ix::new(i as u32);
            if from.is_discontinuous(ix) {
                let fragments: Vec<Span> = from.fragments(ix).map(|s| (self.span)(s)).collect();
                out.try_push_fragments(&fragments).map_err(|e| e.to_string())?;
            } else {
                out.push((self.span)(from.get(ix)));
            }
        }
        Ok(())
    }

    fn shift(&self, of: KeyId) -> Result<&Shift, String> {
        let found = if of.name.is_empty() {
            self.shifts.iter().find(|s| s.key.ty == of.ty)
        } else {
            self.shifts.iter().find(|s| s.key == of)
        };
        found.ok_or_else(|| format!("{} indexes {of}, which is not part of the splice", self.own))
    }

    /// The destination index of item `i` of layer `of` (its default key
    /// names the instance the source layer was made on). `Err` when the
    /// index leaves the unit being copied: the layer is not as local as
    /// its step declared.
    pub fn index(&self, of: KeyId, i: usize) -> Result<usize, String> {
        let s = self.shift(of)?;
        if !s.from.contains(&i) {
            return Err(format!("{} indexes item {i} of {}, outside its unit {:?}", self.own, s.key, s.from));
        }
        Ok(i - s.from.start + s.to)
    }

    /// [`Self::index`] for a typed index into the default instance of `K`.
    pub fn ix<K: Layer>(&self, i: Ix<K>) -> Result<Ix<K>, String> {
        self.index(Key::<K>::DEFAULT.id(), i.get()).map(|n| Ix::new(n as u32))
    }

    /// [`Self::index`] for a range of items of `K`: its end may be one
    /// past the unit's last item.
    pub fn ix_range<K: Layer>(&self, r: IxRange<K>) -> Result<IxRange<K>, String> {
        let s = self.shift(Key::<K>::DEFAULT.id())?;
        let (a, b) = (r.start.get(), r.end.get());
        if a < s.from.start || b > s.from.end {
            return Err(format!("{} indexes items {a}..{b} of {}, outside its unit {:?}", self.own, s.key, s.from));
        }
        Ok(IxRange::new((a - s.from.start + s.to) as u32, (b - s.from.start + s.to) as u32))
    }
}

/// [`Splice`] with the layer type erased, so the analyzer can carry any
/// layer its steps write without naming the type.
pub trait LayerSplicer: Send + Sync {
    fn anchor(&self) -> Anchor;
    /// The spans of a layer of this type, when it is anchored to text.
    fn text_spans<'a>(&self, layer: &'a (dyn Any + Send + Sync)) -> Option<TextSpans<'a>>;
    /// An empty layer shaped like `like` ([`Splice::empty_like`]); None
    /// when `like` is another type.
    fn start(&self, like: &(dyn Any + Send + Sync), view: ViewId) -> Option<Box<dyn Any + Send + Sync>>;
    /// [`Splice::append`] on erased layers.
    fn append(
        &self,
        out: &mut (dyn Any + Send + Sync),
        from: &(dyn Any + Send + Sync),
        range: Range<usize>,
        cx: &mut SpliceCx<'_>,
    ) -> Result<(), String>;
    /// Adds the assembled layer to `doc` as `key`, through the checked
    /// `Document::add`, with its step's provenance; `earlier` are the
    /// outputs the step writes before this one.
    fn add(
        &self,
        out: Box<dyn Any + Send + Sync>,
        doc: &mut Document,
        key: KeyId,
        earlier: &[KeyId],
        provenance: &Provenance,
    ) -> Result<(), Error>;
}

/// The [`LayerSplicer`] of a layer type that implements [`Splice`].
pub struct Splicer<L>(PhantomData<fn() -> L>);

impl<L: Splice> Splicer<L> {
    pub fn boxed() -> Box<dyn LayerSplicer> {
        Box::new(Splicer::<L>(PhantomData))
    }
}

fn not_this_type<L: Layer>(key: KeyId) -> Error {
    Error::Invalid { layer: key, detail: format!("the spliced layer is not a {}", L::NAME) }
}

impl<L: Splice> LayerSplicer for Splicer<L> {
    fn anchor(&self) -> Anchor {
        L::anchor()
    }

    fn text_spans<'a>(&self, layer: &'a (dyn Any + Send + Sync)) -> Option<TextSpans<'a>> {
        layer.downcast_ref::<L>()?.text_spans()
    }

    fn start(&self, like: &(dyn Any + Send + Sync), view: ViewId) -> Option<Box<dyn Any + Send + Sync>> {
        let like = like.downcast_ref::<L>()?;
        Some(Box::new(L::empty_like(like, view)))
    }

    fn append(
        &self,
        out: &mut (dyn Any + Send + Sync),
        from: &(dyn Any + Send + Sync),
        range: Range<usize>,
        cx: &mut SpliceCx<'_>,
    ) -> Result<(), String> {
        let out = out.downcast_mut::<L>().ok_or_else(|| format!("the spliced layer is not a {}", L::NAME))?;
        let from = from.downcast_ref::<L>().ok_or_else(|| format!("the source layer is not a {}", L::NAME))?;
        out.append(from, range, cx)
    }

    fn add(
        &self,
        out: Box<dyn Any + Send + Sync>,
        doc: &mut Document,
        key: KeyId,
        earlier: &[KeyId],
        provenance: &Provenance,
    ) -> Result<(), Error> {
        let layer = out.downcast::<L>().map_err(|_| not_this_type::<L>(key))?;
        let p = made_on_own(&*layer, earlier, provenance);
        doc.add(&Key::<L>::named(key.name), *layer, p)
    }
}

/// The partitions a layer cuts its view into: its top-level items, in
/// text order. For a flat layer (`Overlap::None`) every item; for a
/// nesting one (`Structure`) the items inside no other. None when an
/// item straddles another or the items are out of order.
pub fn partitions(spans: &TextSpans<'_>) -> Option<Vec<Span>> {
    let mut out: Vec<Span> = Vec::new();
    for (&start, &end) in spans.start.iter().zip(spans.end) {
        let s = Span::new(start, end);
        if let Some(&top) = out.last() {
            if start < top.start {
                return None;
            }
            if start < top.end || (s.is_empty() && start == top.end) {
                if top.contains(s) {
                    continue;
                }
                return None;
            }
        }
        out.push(s);
    }
    Some(out)
}

/// The items of a text-anchored layer inside each partition, in
/// partition order: `out[i]` are the items whose envelope lies inside
/// `partitions[i]`. None when an item lies in no partition, straddles
/// two, or the items are not in partition order, so the layer cannot be
/// spliced by these partitions.
pub fn partition_ranges(spans: &TextSpans<'_>, partitions: &[Span]) -> Option<Vec<Range<usize>>> {
    let n = spans.start.len();
    let mut at = 0;
    let mut out = Vec::with_capacity(partitions.len());
    for p in partitions {
        let a = at;
        while at < n && spans.start[at] >= p.start && spans.end[at] <= p.end {
            at += 1;
        }
        out.push(a..at);
    }
    (at == n).then_some(out)
}
