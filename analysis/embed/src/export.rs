//! Document graph codecs for the layers this crate writes: the question
//! and ranked answers of extractive QA, and the shortlist and reranked
//! order of reranking. Import refuses a payload that does not hold
//! together: trailing bytes, a view or source instance that is not
//! restored, an item past the end of its source layer, a non-finite or
//! unsorted score, or anything `Layer::check` refuses.

use crate::{Answers, Question, Reranked, Shortlist};
use turbo_analysis::export::{
    Codecs, ExportCx, ExportError, ImportCx, LayerCodec, LayerRef, NodeRef, Reader, Writer, get_spans, put_spans,
};
use turbo_analysis::{Document, Ix, Key, Layer, ModelRef};
use turbo_analysis_layers::Levels;

fn malformed(detail: impl Into<String>) -> ExportError {
    ExportError::Malformed(detail.into())
}

fn invalid(cx: &ImportCx, detail: String) -> ExportError {
    ExportError::InvalidLayer { layer: cx.layer().to_owned(), detail }
}

fn done(r: &Reader<'_>, what: &str) -> Result<(), ExportError> {
    if r.is_done() { Ok(()) } else { Err(malformed(format!("trailing bytes in a {what} payload"))) }
}

fn checked<L: Layer>(layer: L, cx: &ImportCx) -> Result<L, ExportError> {
    layer.check().map_err(|e| invalid(cx, e))?;
    Ok(layer)
}

/// The source key and items of a shortlist or reranked layer.
fn put_items<L: Layer>(w: &mut Writer, cx: &ExportCx, key: &Key<L>, items: &[Ix<L>]) -> Result<(), ExportError> {
    let target = cx.resolve(key);
    w.str(&target.layer)?;
    w.str(&target.instance)?;
    w.usize(items.len())?;
    for i in items {
        w.u32(u32::try_from(i.get()).map_err(|_| malformed("an item index past u32"))?);
    }
    Ok(())
}

/// Reads what [`put_items`] wrote, holding every item to the restored
/// source instance's length.
fn get_items<L: Layer>(r: &mut Reader<'_>, cx: &mut ImportCx) -> Result<(Key<L>, Vec<Ix<L>>), ExportError> {
    let target = LayerRef { layer: r.string()?, instance: r.string()? };
    cx.want_keyed_target::<L>(&target)?;
    let len = cx.layer_len(&target)?;
    let mut items = Vec::new();
    for _ in 0..r.usize()? {
        let index = r.u32()?;
        if index as usize >= len || index == u32::MAX {
            return Err(ExportError::DanglingEndpoint {
                layer: cx.layer().to_owned(),
                endpoint: NodeRef { layer: target.clone(), index },
            });
        }
        items.push(Ix::new(index));
    }
    Ok((cx.key::<L>(&target.instance), items))
}

impl LayerCodec for Question {
    fn export(&self, _: &ExportCx) -> Result<Vec<u8>, ExportError> {
        let mut w = Writer::new();
        w.str(&self.0)?;
        Ok(w.finish())
    }
    fn import(data: &[u8], cx: &mut ImportCx, _: &Document) -> Result<Self, ExportError> {
        let mut r = Reader::new(data);
        let question = Question(r.string()?);
        done(&r, "question")?;
        checked(question, cx)
    }
}

impl LayerCodec for Answers {
    fn export(&self, cx: &ExportCx) -> Result<Vec<u8>, ExportError> {
        self.check().map_err(ExportError::Malformed)?;
        let mut w = Writer::new();
        put_spans(&mut w, &self.spans, cx)?;
        for (&s, &p) in self.scores.iter().zip(&self.probabilities) {
            w.f64(s);
            w.f64(p);
        }
        w.f64(self.null_score);
        w.f64(self.null_probability);
        Ok(w.finish())
    }
    fn import(data: &[u8], cx: &mut ImportCx, _: &Document) -> Result<Self, ExportError> {
        let mut r = Reader::new(data);
        let spans = get_spans(&mut r, cx)?;
        let (mut scores, mut probabilities) = (Vec::new(), Vec::new());
        for _ in 0..spans.len() {
            scores.push(r.f64()?);
            probabilities.push(r.f64()?);
        }
        let (null_score, null_probability) = (r.f64()?, r.f64()?);
        done(&r, "answers")?;
        checked(Answers { spans, scores, probabilities, null_score, null_probability }, cx)
    }
}

impl<L: Layer> LayerCodec for Shortlist<L> {
    fn export(&self, cx: &ExportCx) -> Result<Vec<u8>, ExportError> {
        self.check().map_err(ExportError::Malformed)?;
        let mut w = Writer::new();
        w.str(&self.query)?;
        put_items(&mut w, cx, &self.source, &self.items)?;
        Ok(w.finish())
    }
    fn import(data: &[u8], cx: &mut ImportCx, _: &Document) -> Result<Self, ExportError> {
        let mut r = Reader::new(data);
        let query = r.string()?;
        let (source, items) = get_items(&mut r, cx)?;
        done(&r, "shortlist")?;
        checked(Shortlist { query, source, items }, cx)
    }
}

impl<L: Layer> LayerCodec for Reranked<L> {
    fn export(&self, cx: &ExportCx) -> Result<Vec<u8>, ExportError> {
        self.check().map_err(ExportError::Malformed)?;
        let mut w = Writer::new();
        put_items(&mut w, cx, &self.source, &self.items)?;
        for s in &self.scores {
            w.u32(s.to_bits());
        }
        w.str(&self.model.id)?;
        w.str(&self.model.revision)?;
        w.str(&self.model.sha256)?;
        Ok(w.finish())
    }
    fn import(data: &[u8], cx: &mut ImportCx, _: &Document) -> Result<Self, ExportError> {
        let mut r = Reader::new(data);
        let (source, items) = get_items(&mut r, cx)?;
        let scores = (0..items.len()).map(|_| r.u32().map(f32::from_bits)).collect::<Result<Vec<_>, _>>()?;
        let model = ModelRef { id: r.string()?, revision: r.string()?, sha256: r.string()? };
        done(&r, "reranked")?;
        checked(Reranked { source, items, scores, model }, cx)
    }
}

/// Registers the codecs for this crate's writes on `codecs`: the QA
/// question and answers, and the shortlist and reranked order over
/// [`Levels`], the units [`RerankAnnotator`](crate::RerankAnnotator)
/// reads. A shortlist over another layer registers its own
/// `Shortlist<L>` and `Reranked<L>`.
pub fn register_codecs(codecs: &mut Codecs) {
    codecs.register::<Question>().register::<Answers>().register::<Shortlist<Levels>>().register::<Reranked<Levels>>();
}

/// A registry holding this crate's codecs and nothing else.
pub fn codecs() -> Codecs {
    let mut codecs = Codecs::new();
    register_codecs(&mut codecs);
    codecs
}
