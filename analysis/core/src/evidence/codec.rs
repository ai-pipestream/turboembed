//! `Evidence<L>` across the document graph boundary. Spans are source
//! bytes (view 0), so they cross as plain offsets; support names its
//! layer instances as `LayerRef`s. An import is refused unless the
//! records verify against the document restored so far: the models
//! named ran there, the spans lie in its source, and the support lies in
//! its layers.

use std::marker::PhantomData;
use std::sync::Arc;

use crate::doc::Document;
use crate::export::{ExportCx, ExportError, ImportCx, LayerCodec, LayerRef, Reader, Writer};
use crate::layer::{Layer, ModelRef};
use crate::span::Span;

use super::{Evidence, Producer, Support};

fn get_flag(r: &mut Reader<'_>, what: &str) -> Result<bool, ExportError> {
    match r.u8()? {
        0 => Ok(false),
        1 => Ok(true),
        b => Err(ExportError::Malformed(format!("a {what} flag of {b}"))),
    }
}

fn get_scores(r: &mut Reader<'_>) -> Result<Vec<(u32, f64)>, ExportError> {
    let mut v = Vec::new();
    for _ in 0..r.usize()? {
        v.push((r.u32()?, r.f64()?));
    }
    Ok(v)
}

fn put_u32s(w: &mut Writer, v: &[u32]) -> Result<(), ExportError> {
    w.usize(v.len())?;
    v.iter().for_each(|&x| w.u32(x));
    Ok(())
}

fn get_u32s(r: &mut Reader<'_>) -> Result<Vec<u32>, ExportError> {
    let n = r.usize()?;
    let mut v = Vec::new();
    for _ in 0..n {
        v.push(r.u32()?);
    }
    Ok(v)
}

impl<L: Layer> LayerCodec for Evidence<L> {
    fn export(&self, _cx: &ExportCx) -> Result<Vec<u8>, ExportError> {
        let mut w = Writer::new();
        w.usize(self.producers.len())?;
        for p in self.producers.iter() {
            w.str(&p.annotator)?;
            match &p.model {
                None => w.u8(0),
                Some(m) => {
                    w.u8(1);
                    w.str(&m.id)?;
                    w.str(&m.revision)?;
                    w.str(&m.sha256)?;
                }
            }
        }
        put_u32s(&mut w, &self.producer)?;
        put_u32s(&mut w, &self.span_row)?;
        w.usize(self.spans.len())?;
        self.spans.iter().for_each(|&s| w.span(s));
        for column in [&self.raw, &self.calibrated] {
            w.usize(column.len())?;
            for &(i, x) in column {
                w.u32(i);
                w.f64(x);
            }
        }
        w.usize(self.abstention.len())?;
        for (i, a) in &self.abstention {
            w.u32(*i);
            w.str(a)?;
        }
        w.usize(self.layers.len())?;
        for l in self.layers.iter() {
            w.str(&l.layer)?;
            w.str(&l.instance)?;
        }
        put_u32s(&mut w, &self.support_row)?;
        w.usize(self.support.len())?;
        for s in &self.support {
            w.u32(s.layer);
            w.u32(s.start);
            w.u32(s.end);
        }
        Ok(w.finish())
    }

    fn import(data: &[u8], cx: &mut ImportCx, doc: &Document) -> Result<Self, ExportError> {
        let invalid = |detail: String| ExportError::InvalidLayer { layer: cx.layer().to_owned(), detail };
        let mut r = Reader::new(data);
        let mut producers = Vec::new();
        for _ in 0..r.usize()? {
            let annotator = r.string()?;
            let model = if get_flag(&mut r, "model")? {
                Some(ModelRef { id: r.string()?, revision: r.string()?, sha256: r.string()? })
            } else {
                None
            };
            producers.push(Producer { annotator, model });
        }
        let producer = get_u32s(&mut r)?;
        let span_row = get_u32s(&mut r)?;
        let mut spans = Vec::new();
        for _ in 0..r.usize()? {
            let s: Span = r.span()?;
            spans.push(s);
        }
        let raw = get_scores(&mut r)?;
        let calibrated = get_scores(&mut r)?;
        let mut abstention = Vec::new();
        for _ in 0..r.usize()? {
            abstention.push((r.u32()?, r.string()?));
        }
        let mut layers = Vec::new();
        for _ in 0..r.usize()? {
            layers.push(LayerRef { layer: r.string()?, instance: r.string()? });
        }
        let support_row = get_u32s(&mut r)?;
        let mut support = Vec::new();
        for _ in 0..r.usize()? {
            support.push(Support { layer: r.u32()?, start: r.u32()?, end: r.u32()? });
        }
        if !r.is_done() {
            return Err(ExportError::Malformed("trailing bytes in an evidence payload".into()));
        }
        // Each support layer must be one the graph restores before this.
        for l in &layers {
            cx.layer_len(l)?;
        }
        let ev = Evidence {
            producers: Arc::new(producers),
            producer,
            span_row,
            spans,
            raw,
            calibrated,
            abstention,
            layers: Arc::new(layers),
            support_row,
            support,
            _of: PhantomData,
            built: false,
        };
        ev.verify(doc).map_err(|e| invalid(e.to_string()))?;
        Ok(ev)
    }
}
