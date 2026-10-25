use super::*;

/// Writes a span layer's columns: its view's instance name, then per
/// item its fragments (one for a continuous span), so discontinuous
/// items round-trip exactly.
pub fn put_spans<L>(w: &mut Writer, spans: &Spans<L>, cx: &ExportCx) -> Result<(), ExportError> {
    w.str(cx.view_instance(spans.view())?)?;
    w.usize(spans.len())?;
    for (i, _) in spans.iter() {
        let fragments: Vec<Span> = spans.fragments(i).collect();
        w.usize(fragments.len())?;
        for fragment in fragments {
            w.span(fragment);
        }
    }
    Ok(())
}

/// Reads what [`put_spans`] wrote.
pub fn get_spans<L>(r: &mut Reader<'_>, cx: &mut ImportCx) -> Result<Spans<L>, ExportError> {
    let instance = r.string()?;
    let mut spans = Spans::new(cx.view(&instance)?);
    for _ in 0..r.usize()? {
        let mut fragments = Vec::new();
        for _ in 0..r.usize()? {
            fragments.push(r.span()?);
        }
        match &fragments[..] {
            [] => return Err(malformed("a span item with no fragments")),
            [one] => {
                spans.push(*one);
            }
            many => {
                spans
                    .try_push_fragments(many)
                    .map_err(|e| ExportError::InvalidLayer { layer: cx.layer.clone(), detail: e.to_string() })?;
            }
        }
    }
    Ok(spans)
}

/// Slices a [`put_spans`] payload to its view's kept range (see
/// [`SliceCx::view_range`]), for span-only codecs to implement
/// [`LayerCodec::slice_body`] with: the same format back out, with
/// clipped and rebased fragments, plus the old-to-new item remap.
/// A payload with trailing fields after the span column is refused
/// by the caller's own length checks, not read here.
/// Reads the raw fragment table of a [`put_spans`] column at the
/// reader's position, without resolving its view: the view
/// instance and every item's fragments. For payload surgery
/// (slicing) that must not allocate a document context.
pub fn get_spans_raw(r: &mut Reader<'_>) -> Result<(String, Vec<Vec<Span>>), ExportError> {
    let instance = r.str()?.to_owned();
    let mut items: Vec<Vec<Span>> = Vec::new();
    for _ in 0..r.usize()? {
        let mut fragments = Vec::new();
        for _ in 0..r.usize()? {
            fragments.push(r.span()?);
        }
        if fragments.is_empty() {
            return Err(malformed("a span item with no fragments"));
        }
        items.push(fragments);
    }
    Ok((instance, items))
}

/// Writes a raw fragment table in the [`put_spans`] format.
pub fn put_spans_raw(w: &mut Writer, instance: &str, items: &[Vec<Span>]) -> Result<(), ExportError> {
    w.str(instance)?;
    w.usize(items.len())?;
    for fragments in items {
        w.usize(fragments.len())?;
        for fragment in fragments {
            w.span(*fragment);
        }
    }
    Ok(())
}

pub fn slice_spans_payload(data: &[u8], cx: &SliceCx) -> Result<(Vec<u8>, Vec<Option<u32>>), ExportError> {
    let mut r = Reader::new(data);
    let (instance, items) = get_spans_raw(&mut r)?;
    let range = cx.view_range(&instance)?;
    if !r.is_done() {
        return Err(malformed("trailing bytes in a spans payload being sliced"));
    }
    let (clipped, remap) = crate::span::slice::clip_items(&items, range);
    let mut w = Writer::new();
    put_spans_raw(&mut w, &instance, &clipped)?;
    Ok((w.finish(), remap))
}

/// Writes a tag set: its name and its labels in tag order.
pub fn put_tagset(w: &mut Writer, kinds: &TagSet) -> Result<(), ExportError> {
    w.str(&kinds.name)?;
    w.usize(kinds.len())?;
    for i in 0..kinds.len() {
        w.str(kinds.label(Tag(u16::try_from(i).map_err(|_| malformed("a tag set longer than a u16 can index"))?)))?;
    }
    Ok(())
}

/// Reads what [`put_tagset`] wrote. A tag is a `u16`, so a set of more
/// than 65,536 labels cannot be indexed and is refused instead of
/// wrapping an index onto another label.
pub fn get_tagset(r: &mut Reader<'_>) -> Result<Arc<TagSet>, ExportError> {
    let name = r.string()?;
    let count = r.usize()?;
    if count > usize::from(u16::MAX) + 1 {
        return Err(malformed("a tag set longer than a u16 can index"));
    }
    let mut labels = Vec::new();
    for _ in 0..count {
        labels.push(r.string()?);
    }
    let labels: Vec<&str> = labels.iter().map(String::as_str).collect();
    Ok(TagSet::new(&name, &labels))
}

pub(super) fn put_layer_ref(w: &mut Writer, layer: &LayerRef) -> Result<(), ExportError> {
    w.str(&layer.layer)?;
    w.str(&layer.instance)
}

pub(super) fn get_layer_ref(r: &mut Reader<'_>) -> Result<LayerRef, ExportError> {
    Ok(LayerRef { layer: r.string()?, instance: r.string()? })
}
