use super::*;
use crate::alternatives::{Alternatives, ModeSet, NBest};

fn want_len(got: usize, want: usize, what: &str) -> Result<(), ExportError> {
    if got != want { Err(malformed(format!("{got} {what} for {want} items"))) } else { Ok(()) }
}

fn done(r: &Reader<'_>, what: &str) -> Result<(), ExportError> {
    if r.is_done() { Ok(()) } else { Err(malformed(format!("trailing bytes in a {what} payload"))) }
}

fn checked<L: Layer>(layer: L, cx: &ImportCx) -> Result<L, ExportError> {
    layer.check().map_err(|detail| ExportError::InvalidLayer { layer: cx.layer().to_owned(), detail })?;
    Ok(layer)
}

fn put_u32s(w: &mut Writer, values: &[u32]) -> Result<(), ExportError> {
    w.usize(values.len())?;
    for &v in values {
        w.u32(v);
    }
    Ok(())
}

fn get_u32s(r: &mut Reader<'_>) -> Result<Vec<u32>, ExportError> {
    let mut values = Vec::new();
    for _ in 0..r.usize()? {
        values.push(r.u32()?);
    }
    Ok(values)
}

fn put_opt_f64s(w: &mut Writer, values: &Option<Vec<f64>>) -> Result<(), ExportError> {
    match values {
        None => w.u8(0),
        Some(v) => {
            w.u8(1);
            w.usize(v.len())?;
            for &x in v {
                w.f64(x);
            }
        }
    }
    Ok(())
}

fn get_opt_f64s(r: &mut Reader<'_>) -> Result<Option<Vec<f64>>, ExportError> {
    Ok(match r.u8()? {
        0 => None,
        1 => {
            let mut values = Vec::new();
            for _ in 0..r.usize()? {
                values.push(r.f64()?);
            }
            Some(values)
        }
        tag => return Err(malformed(format!("an optional column tag of {tag}"))),
    })
}

/// The item count of the already restored `Target` instance an
/// aligned layer on this record is aligned to: the instance the
/// record's provenance resolves `Target` to, else the default.
fn aligned_len<Target: Layer>(cx: &ImportCx) -> Result<usize, ExportError> {
    let name = Target::NAME.to_string();
    let instance = cx.resolved_instance(&name).to_owned();
    cx.layer_len(&LayerRef { layer: name, instance })
}

impl<L: Layer> LayerCodec for Alternatives<L> {
    fn export(&self, cx: &ExportCx) -> Result<Vec<u8>, ExportError> {
        let mut w = Writer::new();
        put_spans(&mut w, &self.spans, cx)?;
        put_u32s(&mut w, &self.position)?;
        put_u32s(&mut w, &self.position_length)?;
        w.u32(self.num_positions);
        w.usize(self.mode.len())?;
        for m in &self.mode {
            w.u8(m.0);
        }
        put_opt_f64s(&mut w, &self.score)?;
        match &self.form {
            None => w.u8(0),
            Some(forms) => {
                w.u8(1);
                w.usize(forms.len())?;
                for f in forms {
                    match f {
                        None => w.u8(0),
                        Some(f) => {
                            w.u8(1);
                            w.str(f)?;
                        }
                    }
                }
            }
        }
        Ok(w.finish())
    }

    fn import(data: &[u8], cx: &mut ImportCx, doc: &Document) -> Result<Self, ExportError> {
        let mut r = Reader::new(data);
        let spans: Spans<Alternatives<L>> = get_spans(&mut r, cx)?;
        let position = get_u32s(&mut r)?;
        let position_length = get_u32s(&mut r)?;
        let num_positions = r.u32()?;
        let mut mode = Vec::new();
        for _ in 0..r.usize()? {
            mode.push(ModeSet(r.u8()?));
        }
        let score = get_opt_f64s(&mut r)?;
        let form = match r.u8()? {
            0 => None,
            1 => {
                let mut forms = Vec::new();
                for _ in 0..r.usize()? {
                    forms.push(match r.u8()? {
                        0 => None,
                        1 => Some(r.string()?.into_boxed_str()),
                        tag => return Err(malformed(format!("an optional form tag of {tag}"))),
                    });
                }
                Some(forms)
            }
            tag => return Err(malformed(format!("an optional column tag of {tag}"))),
        };
        let n = spans.len();
        want_len(position.len(), n, "positions")?;
        want_len(position_length.len(), n, "position lengths")?;
        want_len(mode.len(), n, "mode sets")?;
        if let Some(s) = &score {
            want_len(s.len(), n, "scores")?;
        }
        if let Some(f) = &form {
            want_len(f.len(), n, "forms")?;
        }
        done(&r, "alternatives")?;

        let text = doc.text(spans.view());
        for (i, _) in spans.iter() {
            for frag in spans.fragments(i) {
                let (s, e) = (frag.start as usize, frag.end as usize);
                if s > e || e > text.len() || !text.is_char_boundary(s) || !text.is_char_boundary(e) {
                    return Err(ExportError::InvalidLayer {
                        layer: cx.layer().to_owned(),
                        detail: format!(
                            "span {}..{} is not inside view text of length {} on UTF-8 boundaries",
                            frag.start,
                            frag.end,
                            text.len()
                        ),
                    });
                }
            }
        }

        let mut layer = Alternatives::new(spans.view());
        layer.spans = spans;
        layer.position = position;
        layer.position_length = position_length;
        layer.num_positions = num_positions;
        layer.mode = mode;
        layer.score = score;
        layer.form = form;
        // The lattice check settles the rest: ordered positions, a
        // complete path through every alternative, exact byte
        // tiling and finite scores.
        checked(layer, cx)
    }
}

impl<L: Layer> LayerCodec for NBest<L> {
    fn export(&self, _cx: &ExportCx) -> Result<Vec<u8>, ExportError> {
        let mut w = Writer::new();
        put_tagset(&mut w, &self.tagset)?;
        w.usize(self.items)?;
        w.usize(self.score.len())?;
        for &s in &self.score {
            w.f64(s);
        }
        w.usize(self.tags.len())?;
        for t in &self.tags {
            w.u16(t.0);
        }
        put_opt_f64s(&mut w, &self.prob)?;
        Ok(w.finish())
    }

    fn import(data: &[u8], cx: &mut ImportCx, _doc: &Document) -> Result<Self, ExportError> {
        let mut r = Reader::new(data);
        let tagset = get_tagset(&mut r)?;
        let items = r.usize()?;
        let mut score = Vec::new();
        for _ in 0..r.usize()? {
            score.push(r.f64()?);
        }
        let mut tags = Vec::new();
        for _ in 0..r.usize()? {
            tags.push(Tag(r.u16()?));
        }
        let prob = get_opt_f64s(&mut r)?;
        done(&r, "nbest")?;
        let want = aligned_len::<L>(cx)?;
        if items != want {
            return Err(malformed(format!("nbest sequences of {items} items for an aligned layer of {want}")));
        }
        let mut layer = NBest::new(tagset, items);
        layer.score = score;
        layer.tags = tags;
        layer.prob = prob;
        // The layer check settles the table shape (with the
        // multiplication checked), NaN scores, best-first order,
        // tags inside the tag set and probabilities in [0, 1].
        checked(layer, cx)
    }
}
