//! The document graph codecs for this crate's layers, so word types
//! and dictionary readings cross the export boundary with the rest
//! of the graph.

use turbo_analysis::export::{Codecs, ExportCx, ExportError, ImportCx, LayerCodec, LayerRef, Reader, Writer};
use turbo_analysis::{Document, Layer};
use turbo_analysis_layers::{Strs, Tokens};

use crate::{Readings, WordType, WordTypes};

fn malformed(detail: impl Into<String>) -> ExportError {
    ExportError::Malformed(detail.into())
}

/// The item count of the already restored `Target` instance an
/// aligned layer on this record is aligned to: the instance the
/// record's provenance resolves `Target` to, else the default.
fn aligned_len<Target: Layer>(cx: &ImportCx) -> Result<usize, ExportError> {
    let name = Target::NAME.to_string();
    let instance = cx.resolved_instance(&name).to_owned();
    cx.layer_len(&LayerRef { layer: name, instance })
}

fn checked<L: Layer>(layer: L, cx: &ImportCx) -> Result<L, ExportError> {
    layer.check().map_err(|detail| ExportError::InvalidLayer { layer: cx.layer().to_owned(), detail })?;
    Ok(layer)
}

fn put_opt_strs(w: &mut Writer, strs: &Option<Strs>) -> Result<(), ExportError> {
    match strs {
        None => w.u8(0),
        Some(s) => {
            w.u8(1);
            w.usize(s.len())?;
            for v in s.iter() {
                w.str(v)?;
            }
        }
    }
    Ok(())
}

fn get_opt_strs(r: &mut Reader<'_>) -> Result<Option<Strs>, ExportError> {
    Ok(match r.u8()? {
        0 => None,
        1 => {
            let mut strs = Strs::default();
            for _ in 0..r.usize()? {
                strs.push(&r.string()?);
            }
            Some(strs)
        }
        tag => return Err(malformed(format!("an optional column tag of {tag}"))),
    })
}

impl LayerCodec for WordTypes {
    fn export(&self, _cx: &ExportCx) -> Result<Vec<u8>, ExportError> {
        let mut w = Writer::new();
        w.usize(self.kind.len())?;
        for &kind in &self.kind {
            w.u8(kind as u8);
        }
        Ok(w.finish())
    }

    fn import(data: &[u8], cx: &mut ImportCx, _doc: &Document) -> Result<Self, ExportError> {
        let mut r = Reader::new(data);
        let mut kind = Vec::new();
        for _ in 0..r.usize()? {
            let at = r.u8()? as usize;
            kind.push(WordType::ALL.get(at).copied().ok_or_else(|| malformed(format!("a word type of {at}")))?);
        }
        if !r.is_done() {
            return Err(malformed("trailing bytes in a word types payload"));
        }
        let want = aligned_len::<Tokens>(cx)?;
        if kind.len() != want {
            return Err(malformed(format!("{} word types for {want} tokens", kind.len())));
        }
        checked(WordTypes { kind }, cx)
    }
}

impl LayerCodec for Readings {
    fn export(&self, _cx: &ExportCx) -> Result<Vec<u8>, ExportError> {
        let mut w = Writer::new();
        put_opt_strs(&mut w, &self.reading)?;
        put_opt_strs(&mut w, &self.pronunciation)?;
        put_opt_strs(&mut w, &self.lemma)?;
        Ok(w.finish())
    }

    fn import(data: &[u8], cx: &mut ImportCx, _doc: &Document) -> Result<Self, ExportError> {
        let mut r = Reader::new(data);
        let layer = Readings {
            reading: get_opt_strs(&mut r)?,
            pronunciation: get_opt_strs(&mut r)?,
            lemma: get_opt_strs(&mut r)?,
        };
        if !r.is_done() {
            return Err(malformed("trailing bytes in a readings payload"));
        }
        // The layer's own check wants at least one column, all of one
        // length; that length must be the aligned token count.
        let layer = checked(layer, cx)?;
        let want = aligned_len::<Tokens>(cx)?;
        if layer.len() != want {
            return Err(malformed(format!("{} readings for {want} tokens", layer.len())));
        }
        Ok(layer)
    }
}

/// Registers this crate's layer codecs on `codecs`.
pub fn register_codecs(codecs: &mut Codecs) {
    codecs.register::<WordTypes>().register::<Readings>();
}

/// A registry holding this crate's codecs and nothing else.
pub fn codecs() -> Codecs {
    let mut codecs = Codecs::new();
    register_codecs(&mut codecs);
    codecs
}
