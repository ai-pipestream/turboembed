//! `DocumentGraph` export and import for the relations layer.
//! Subject and object indexes are refused past the end of the
//! restored `Entities` instance this record's provenance resolves
//! to, as every other codec's indexes are; the
//! `Edges<Entities, Entities>` companion layer uses the core
//! graph codec and is only registered here.

use turbo_analysis::export::{
    Codecs, ExportCx, ExportError, ImportCx, LayerCodec, LayerRef, Reader, Writer, get_spans, get_tagset, put_spans,
    put_tagset,
};
use turbo_analysis::{Document, Edges, Ix, Layer, Tag};
use turbo_analysis_layers::Entities;

use crate::layer::Relations;

fn malformed(detail: impl Into<String>) -> ExportError {
    ExportError::Malformed(detail.into())
}

fn checked<L: Layer>(layer: L, cx: &ImportCx) -> Result<L, ExportError> {
    layer.check().map_err(|detail| ExportError::InvalidLayer { layer: cx.layer().to_owned(), detail })?;
    Ok(layer)
}

fn want_len(got: usize, want: usize, what: &str) -> Result<(), ExportError> {
    if got != want {
        return Err(malformed(format!("{got} {what} for {want} relations")));
    }
    Ok(())
}

fn done(r: &Reader<'_>, what: &str) -> Result<(), ExportError> {
    if r.is_done() { Ok(()) } else { Err(malformed(format!("trailing bytes in a {what} payload"))) }
}

/// The restored `Entities` instance this record binds to: the one
/// its provenance resolves `Entities` to, else the default.
fn entities_len(cx: &ImportCx) -> Result<usize, ExportError> {
    let name = Entities::NAME.to_string();
    let instance = cx.resolved_instance(&name).to_owned();
    cx.layer_len(&LayerRef { layer: name, instance })
}

impl LayerCodec for Relations {
    fn export(&self, cx: &ExportCx) -> Result<Vec<u8>, ExportError> {
        let mut w = Writer::new();
        put_spans(&mut w, &self.spans, cx)?;
        put_tagset(&mut w, &self.kinds)?;
        w.usize(self.subject.len())?;
        for (i, _) in self.spans.iter() {
            w.u32(self.subject[i.get()].get() as u32);
            w.u32(self.object[i.get()].get() as u32);
            w.u16(self.kind[i.get()].0);
            w.f64(self.score[i.get()]);
        }
        Ok(w.finish())
    }

    fn import(data: &[u8], cx: &mut ImportCx, _doc: &Document) -> Result<Self, ExportError> {
        let mut r = Reader::new(data);
        let spans = get_spans(&mut r, cx)?;
        let kinds = get_tagset(&mut r)?;
        let entities = entities_len(cx)?;
        let rows = r.usize()?;
        want_len(rows, spans.len(), "relation rows")?;
        let (mut subject, mut object, mut kind, mut score) = (Vec::new(), Vec::new(), Vec::new(), Vec::new());
        for _ in 0..rows {
            let (s, o, k, sc) = (r.u32()? as usize, r.u32()? as usize, r.u16()?, r.f64()?);
            if s >= entities || o >= entities {
                return Err(malformed(format!("a relation endpoint past {entities} entities")));
            }
            if k as usize >= kinds.len() {
                return Err(malformed(format!("a kind outside {}", kinds.name)));
            }
            subject.push(Ix::new(s as u32));
            object.push(Ix::new(o as u32));
            kind.push(Tag(k));
            score.push(sc);
        }
        done(&r, "relations")?;
        checked(Relations { spans, subject, object, kind, kinds, score }, cx)
    }
}

/// Registers this crate's codecs: the relations layer and its
/// companion entity-to-entity edges (the core graph codec).
pub fn register_codecs(codecs: &mut Codecs) {
    codecs.register::<Relations>().register::<Edges<Entities, Entities>>();
}

/// The standard codecs plus this crate's, for callers that build
/// no registry of their own.
pub fn codecs() -> Codecs {
    let mut codecs = Codecs::new();
    turbo_analysis_layers::export::register_codecs(&mut codecs);
    register_codecs(&mut codecs);
    codecs
}
