use super::*;

/// An index column into another layer: the resolved target instance,
/// then the indexes. Import validates every index against the target
/// instance restored before this layer.
pub(super) fn put_ix_col<Target: Layer>(
    w: &mut Writer,
    cx: &ExportCx,
    key: &Key<Target>,
    values: &[Ix<Target>],
) -> Result<(), ExportError> {
    put_ref(w, &cx.resolve(key))?;
    w.usize(values.len())?;
    for v in values {
        w.u32(v.get() as u32);
    }
    Ok(())
}

pub(super) fn get_ix_col<Target: Layer>(r: &mut Reader<'_>, cx: &mut ImportCx) -> Result<Vec<Ix<Target>>, ExportError> {
    let target = get_ref(r)?;
    cx.expect_target::<Target>(&target)?;
    let mut values = Vec::new();
    for _ in 0..r.usize()? {
        let index = r.u32()?;
        cx.require_node(&NodeRef { layer: target.clone(), index })?;
        values.push(Ix::new(index));
    }
    Ok(values)
}

pub(super) fn put_opt_ix_col<Target: Layer>(
    w: &mut Writer,
    cx: &ExportCx,
    key: &Key<Target>,
    values: &Option<Vec<Ix<Target>>>,
) -> Result<(), ExportError> {
    match values {
        None => w.u8(0),
        Some(v) => {
            w.u8(1);
            put_ix_col(w, cx, key, v)?;
        }
    }
    Ok(())
}

pub(super) fn get_opt_ix_col<Target: Layer>(
    r: &mut Reader<'_>,
    cx: &mut ImportCx,
) -> Result<Option<Vec<Ix<Target>>>, ExportError> {
    Ok(match r.u8()? {
        0 => None,
        1 => Some(get_ix_col(r, cx)?),
        tag => return Err(malformed(format!("an optional column tag of {tag}"))),
    })
}

/// A column of ranges into another layer: the resolved target
/// instance, then start/end pairs. Import requires each range to end
/// at or before the target's item count.
pub(super) fn put_ranges<Target: Layer>(
    w: &mut Writer,
    cx: &ExportCx,
    key: &Key<Target>,
    values: &[IxRange<Target>],
) -> Result<(), ExportError> {
    put_ref(w, &cx.resolve(key))?;
    w.usize(values.len())?;
    for v in values {
        w.u32(v.start.get() as u32);
        w.u32(v.end.get() as u32);
    }
    Ok(())
}

pub(super) fn checked_range<Target: Layer>(
    cx: &ImportCx,
    target: &LayerRef,
    len: usize,
    start: u32,
    end: u32,
) -> Result<IxRange<Target>, ExportError> {
    if start > end {
        return Err(malformed(format!("a range that runs backwards: {start}..{end}")));
    }
    if end == u32::MAX {
        return Err(malformed("a range endpoint uses the reserved u32::MAX index"));
    }
    if end as usize > len {
        return Err(ExportError::DanglingEndpoint {
            layer: cx.layer().to_owned(),
            endpoint: NodeRef { layer: target.clone(), index: end },
        });
    }
    Ok(IxRange::new(start, end))
}

pub(super) fn get_ranges<Target: Layer>(
    r: &mut Reader<'_>,
    cx: &mut ImportCx,
) -> Result<Vec<IxRange<Target>>, ExportError> {
    let target = get_ref(r)?;
    cx.expect_target::<Target>(&target)?;
    let len = cx.layer_len(&target)?;
    let mut values = Vec::new();
    for _ in 0..r.usize()? {
        let (start, end) = (r.u32()?, r.u32()?);
        values.push(checked_range(cx, &target, len, start, end)?);
    }
    Ok(values)
}

/// A parent column into the layer itself. Import requires every
/// parent to sit strictly before its item, which makes cycles
/// unrepresentable.
pub(super) fn put_parents<L>(w: &mut Writer, values: &[Option<Ix<L>>]) -> Result<(), ExportError> {
    w.usize(values.len())?;
    for v in values {
        match v {
            None => w.u8(0),
            Some(p) => {
                w.u8(1);
                w.u32(p.get() as u32);
            }
        }
    }
    Ok(())
}

pub(super) fn get_parents<L>(r: &mut Reader<'_>) -> Result<Vec<Option<Ix<L>>>, ExportError> {
    let mut values = Vec::new();
    for i in 0..r.usize()? {
        values.push(match r.u8()? {
            0 => None,
            1 => {
                let p = r.u32()?;
                if p as usize >= i {
                    return Err(malformed(format!("item {i} has a parent at or after it")));
                }
                Some(Ix::new(p))
            }
            tag => return Err(malformed(format!("an optional index tag of {tag}"))),
        });
    }
    Ok(values)
}

/// Ranges into another layer plus the target instance to restore as a
/// key, for layers that carry their target key in the layer itself.
pub(super) fn put_keyed_ranges<Target: Layer>(
    w: &mut Writer,
    cx: &ExportCx,
    key: &Key<Target>,
    values: &[IxRange<Target>],
) -> Result<(), ExportError> {
    put_ranges(w, cx, key, values)
}

pub(super) fn get_keyed_ranges<Target: Layer>(
    r: &mut Reader<'_>,
    cx: &mut ImportCx,
) -> Result<(Key<Target>, Vec<IxRange<Target>>), ExportError> {
    let target = get_ref(r)?;
    want_target::<Target>(&target, cx)?;
    let len = cx.layer_len(&target)?;
    let mut values = Vec::new();
    for _ in 0..r.usize()? {
        let (start, end) = (r.u32()?, r.u32()?);
        values.push(checked_range(cx, &target, len, start, end)?);
    }
    Ok((cx.key::<Target>(&target.instance.clone()), values))
}

// Validate raw endpoints before constructing Ix, whose reserved index panics.
pub(super) fn get_checked_range<Target: Layer>(
    r: &mut Reader<'_>,
    cx: &ImportCx,
    target: &LayerRef,
) -> Result<IxRange<Target>, ExportError> {
    let len = cx.layer_len(target)?;
    let (start, end) = (r.u32()?, r.u32()?);
    checked_range(cx, target, len, start, end)
}
