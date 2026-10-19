//! The document graph codecs for this crate's layers, and the
//! registration of every layer type its annotators write.

use turbo_analysis::export::{Codecs, ExportCx, ExportError, ImportCx, LayerCodec, Reader, Writer};
use turbo_analysis::{Document, Edges, Groups, Layer};
use turbo_analysis_layers::{Boilerplate, Chunks, Sentences, Structure};

use crate::hash::LANES;
use crate::sketch::Sketches;

fn malformed(detail: impl Into<String>) -> ExportError {
    ExportError::Malformed(detail.into())
}

/// A sketch payload: the shingle width, the item count, then per item
/// its word count, exact hash (high half first), shape, SimHash and
/// `LANES` MinHash values.
impl<L: Layer> LayerCodec for Sketches<L> {
    fn export(&self, _cx: &ExportCx) -> Result<Vec<u8>, ExportError> {
        let mut w = Writer::new();
        w.u8(self.shingle);
        w.usize(self.len())?;
        for i in 0..self.len() {
            w.u32(self.words[i]);
            w.u64((self.exact[i] >> 64) as u64);
            w.u64(self.exact[i] as u64);
            w.u64(self.shape[i]);
            w.u64(self.simhash[i]);
            for &v in &self.minhash[i * LANES..(i + 1) * LANES] {
                w.u32(v);
            }
        }
        Ok(w.finish())
    }

    fn import(data: &[u8], _cx: &mut ImportCx, _doc: &Document) -> Result<Self, ExportError> {
        let mut r = Reader::new(data);
        let shingle = r.u8()?;
        if shingle == 0 {
            return Err(malformed("a sketch shingle of no words"));
        }
        let mut s = Sketches::empty(shingle);
        for _ in 0..r.usize()? {
            s.words.push(r.u32()?);
            let (hi, lo) = (r.u64()?, r.u64()?);
            s.exact.push((u128::from(hi) << 64) | u128::from(lo));
            s.shape.push(r.u64()?);
            s.simhash.push(r.u64()?);
            for _ in 0..LANES {
                s.minhash.push(r.u32()?);
            }
        }
        if !r.is_done() {
            return Err(malformed("trailing bytes in a sketches payload"));
        }
        s.check().map_err(malformed)?;
        Ok(s)
    }
}

/// Registers the codecs for what this crate's annotators write, at the
/// levels it supports: the sketches, the duplicate groups and links and
/// the boilerplate flags of structure blocks, sentences and chunks.
pub fn register_codecs(codecs: &mut Codecs) {
    codecs
        .register::<Sketches<Structure>>()
        .register::<Sketches<Sentences>>()
        .register::<Sketches<Chunks>>()
        .register::<Groups<Structure>>()
        .register::<Groups<Sentences>>()
        .register::<Groups<Chunks>>()
        .register::<Edges<Structure, Structure>>()
        .register::<Edges<Sentences, Sentences>>()
        .register::<Edges<Chunks, Chunks>>()
        .register::<Boilerplate<Structure>>()
        .register::<Boilerplate<Sentences>>()
        .register::<Boilerplate<Chunks>>();
}

/// A registry holding this crate's codecs and nothing else.
pub fn codecs() -> Codecs {
    let mut codecs = Codecs::new();
    register_codecs(&mut codecs);
    codecs
}
