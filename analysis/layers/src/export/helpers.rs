use super::*;

pub(super) fn malformed(detail: impl Into<String>) -> ExportError {
    ExportError::Malformed(detail.into())
}

pub(super) fn want_len(got: usize, want: usize, what: &str) -> Result<(), ExportError> {
    if got != want {
        return Err(malformed(format!("{got} {what} for {want} items")));
    }
    Ok(())
}

pub(super) fn done(r: &Reader<'_>, what: &str) -> Result<(), ExportError> {
    if r.is_done() { Ok(()) } else { Err(malformed(format!("trailing bytes in a {what} payload"))) }
}

pub(super) fn put_strs(w: &mut Writer, strs: &Strs) -> Result<(), ExportError> {
    w.usize(strs.len())?;
    for s in strs.iter() {
        w.str(s)?;
    }
    Ok(())
}

pub(super) fn get_strs(r: &mut Reader<'_>) -> Result<Strs, ExportError> {
    let mut strs = Strs::default();
    for _ in 0..r.usize()? {
        strs.push(&r.string()?);
    }
    Ok(strs)
}

pub(super) fn put_opt_strs(w: &mut Writer, strs: &Option<Strs>) -> Result<(), ExportError> {
    match strs {
        None => w.u8(0),
        Some(s) => {
            w.u8(1);
            put_strs(w, s)?;
        }
    }
    Ok(())
}

pub(super) fn get_opt_strs(r: &mut Reader<'_>) -> Result<Option<Strs>, ExportError> {
    Ok(match r.u8()? {
        0 => None,
        1 => Some(get_strs(r)?),
        tag => return Err(malformed(format!("an optional column tag of {tag}"))),
    })
}

pub(super) fn put_tags(w: &mut Writer, tags: &[Tag]) -> Result<(), ExportError> {
    w.usize(tags.len())?;
    for t in tags {
        w.u16(t.0);
    }
    Ok(())
}

pub(super) fn get_tags(r: &mut Reader<'_>, tagset: &TagSet) -> Result<Vec<Tag>, ExportError> {
    let mut tags = Vec::new();
    for _ in 0..r.usize()? {
        let tag = Tag(r.u16()?);
        if tag.0 as usize >= tagset.len() {
            return Err(malformed(format!("a tag outside {}", tagset.name)));
        }
        tags.push(tag);
    }
    Ok(tags)
}

pub(super) fn put_f64s(w: &mut Writer, values: &[f64]) -> Result<(), ExportError> {
    w.usize(values.len())?;
    for v in values {
        w.f64(*v);
    }
    Ok(())
}

pub(super) fn get_f64s(r: &mut Reader<'_>) -> Result<Vec<f64>, ExportError> {
    let mut values = Vec::new();
    for _ in 0..r.usize()? {
        values.push(r.f64()?);
    }
    Ok(values)
}

pub(super) fn put_opt_f64s(w: &mut Writer, values: &Option<Vec<f64>>) -> Result<(), ExportError> {
    match values {
        None => w.u8(0),
        Some(v) => {
            w.u8(1);
            put_f64s(w, v)?;
        }
    }
    Ok(())
}

pub(super) fn get_opt_f64s(r: &mut Reader<'_>) -> Result<Option<Vec<f64>>, ExportError> {
    Ok(match r.u8()? {
        0 => None,
        1 => Some(get_f64s(r)?),
        tag => return Err(malformed(format!("an optional column tag of {tag}"))),
    })
}

pub(super) fn put_ref(w: &mut Writer, layer: &LayerRef) -> Result<(), ExportError> {
    w.str(&layer.layer)?;
    w.str(&layer.instance)
}

pub(super) fn get_ref(r: &mut Reader<'_>) -> Result<LayerRef, ExportError> {
    Ok(LayerRef { layer: r.string()?, instance: r.string()? })
}

pub(super) fn want_target<Target: Layer>(named: &LayerRef, cx: &ImportCx) -> Result<(), ExportError> {
    cx.want_keyed_target::<Target>(named)
}

pub(super) fn put_model(w: &mut Writer, model: &ModelRef) -> Result<(), ExportError> {
    w.str(&model.id)?;
    w.str(&model.revision)?;
    w.str(&model.sha256)
}

pub(super) fn get_model(r: &mut Reader<'_>) -> Result<ModelRef, ExportError> {
    Ok(ModelRef { id: r.string()?, revision: r.string()?, sha256: r.string()? })
}

pub(super) fn put_bools(w: &mut Writer, values: &[bool]) -> Result<(), ExportError> {
    w.usize(values.len())?;
    for v in values {
        w.u8(u8::from(*v));
    }
    Ok(())
}

pub(super) fn get_bools(r: &mut Reader<'_>) -> Result<Vec<bool>, ExportError> {
    let mut values = Vec::new();
    for _ in 0..r.usize()? {
        values.push(match r.u8()? {
            0 => false,
            1 => true,
            tag => return Err(malformed(format!("a boolean tag of {tag}"))),
        });
    }
    Ok(values)
}

pub(super) fn put_u32s(w: &mut Writer, values: &[u32]) -> Result<(), ExportError> {
    w.usize(values.len())?;
    for v in values {
        w.u32(*v);
    }
    Ok(())
}

pub(super) fn get_u32s(r: &mut Reader<'_>) -> Result<Vec<u32>, ExportError> {
    let mut values = Vec::new();
    for _ in 0..r.usize()? {
        values.push(r.u32()?);
    }
    Ok(values)
}

pub(super) fn put_i32s(w: &mut Writer, values: &[i32]) -> Result<(), ExportError> {
    w.usize(values.len())?;
    for v in values {
        w.u32(*v as u32);
    }
    Ok(())
}

pub(super) fn get_i32s(r: &mut Reader<'_>) -> Result<Vec<i32>, ExportError> {
    let mut values = Vec::new();
    for _ in 0..r.usize()? {
        values.push(r.u32()? as i32);
    }
    Ok(values)
}

pub(super) fn put_i64s(w: &mut Writer, values: &[i64]) -> Result<(), ExportError> {
    w.usize(values.len())?;
    for v in values {
        w.u64(*v as u64);
    }
    Ok(())
}

pub(super) fn get_i64s(r: &mut Reader<'_>) -> Result<Vec<i64>, ExportError> {
    let mut values = Vec::new();
    for _ in 0..r.usize()? {
        values.push(r.u64()? as i64);
    }
    Ok(values)
}

pub(super) fn put_opt_i64s(w: &mut Writer, values: &[Option<i64>]) -> Result<(), ExportError> {
    w.usize(values.len())?;
    for v in values {
        match v {
            None => w.u8(0),
            Some(v) => {
                w.u8(1);
                w.u64(*v as u64);
            }
        }
    }
    Ok(())
}

pub(super) fn get_opt_i64s(r: &mut Reader<'_>) -> Result<Vec<Option<i64>>, ExportError> {
    let mut values = Vec::new();
    for _ in 0..r.usize()? {
        values.push(match r.u8()? {
            0 => None,
            1 => Some(r.u64()? as i64),
            tag => return Err(malformed(format!("an optional number tag of {tag}"))),
        });
    }
    Ok(values)
}

/// A fieldless enum as its position in `table`, the declared order.
pub(super) fn put_variant<E: Copy + PartialEq>(w: &mut Writer, table: &[E], value: E) {
    w.u8(table.iter().position(|v| *v == value).expect("a variant in the table") as u8);
}

pub(super) fn get_variant<E: Copy>(r: &mut Reader<'_>, table: &[E], what: &str) -> Result<E, ExportError> {
    table.get(r.u8()? as usize).copied().ok_or_else(|| malformed(format!("a {what} outside the table")))
}

pub(super) fn put_variants<E: Copy + PartialEq>(w: &mut Writer, table: &[E], values: &[E]) -> Result<(), ExportError> {
    w.usize(values.len())?;
    for v in values {
        put_variant(w, table, *v);
    }
    Ok(())
}

pub(super) fn get_variants<E: Copy>(r: &mut Reader<'_>, table: &[E], what: &str) -> Result<Vec<E>, ExportError> {
    let mut values = Vec::new();
    for _ in 0..r.usize()? {
        values.push(get_variant(r, table, what)?);
    }
    Ok(values)
}
