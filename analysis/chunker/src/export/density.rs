use super::*;
use crate::terms::DensityTermStats;

pub(super) fn register(codecs: &mut Codecs) {
    codecs
        .register::<DensityTermStats<Chunks>>()
        .register::<DensityTermStats<Sentences>>()
        .register::<DensityTermStats<Structure>>()
        .register::<DensityTermStats<Levels>>()
        .register::<DensityTermStats<View>>();
}

impl<L: Layer> LayerCodec for DensityTermStats<L> {
    fn export(&self, _cx: &ExportCx) -> Result<Vec<u8>, ExportError> {
        let mut w = Writer::new();
        put_u32s(&mut w, &self.row)?;
        w.usize(self.term.len())?;
        for t in &self.term {
            w.u32(t.get() as u32);
        }
        for column in [&self.tf, &self.length] {
            w.usize(column.len())?;
            for &v in column {
                w.f64(v);
            }
        }
        put_u32s(&mut w, &self.df)?;
        Ok(w.finish())
    }
    fn import(data: &[u8], cx: &mut ImportCx, _doc: &Document) -> Result<Self, ExportError> {
        let mut r = Reader::new(data);
        let row = get_u32s(&mut r)?;
        let vocab = resolved::<Vocabulary>(cx);
        let mut term = Vec::new();
        for _ in 0..r.usize()? {
            term.push(get_ix::<Vocabulary>(&mut r, cx, &vocab)?);
        }
        let mut column = || -> Result<Vec<f64>, ExportError> {
            let mut out = Vec::new();
            for _ in 0..r.usize()? {
                out.push(r.f64()?);
            }
            Ok(out)
        };
        let tf = column()?;
        let length = column()?;
        let df = get_u32s(&mut r)?;
        done(&r, "density term stats")?;
        if length.len() != aligned_len::<L>(cx)? || df.len() != cx.layer_len(&vocab)? {
            return Err(malformed("density counts must match aligned units and vocabulary"));
        }
        checked(DensityTermStats { row, term, tf, length, df, _of: PhantomData }, cx)
    }
}
