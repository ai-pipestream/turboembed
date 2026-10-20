//! The `DocumentGraph` codec for the keywords layer: spans, the
//! algorithm tag set, and one (phrase, score, algorithm, term count)
//! row per item, so a populated document crosses the export boundary
//! with its keyphrases and comes back byte-identical. Import refuses a
//! payload that does not hold together: a truncated or trailing body, a
//! row count that disagrees with the spans, a tag outside the tag set,
//! a score that is not a salience in `[0, 1]`, an empty phrase, a term
//! count of zero, or a span that does not slice the view text.

use turbo_analysis::export::{
    Codecs, ExportCx, ExportError, ImportCx, LayerCodec, Reader, Writer, get_spans, get_tagset, put_spans, put_tagset,
};
use turbo_analysis::{Document, Layer};

use crate::Keywords;

fn malformed(detail: impl Into<String>) -> ExportError {
    ExportError::Malformed(detail.into())
}

fn invalid(cx: &ImportCx, detail: String) -> ExportError {
    ExportError::InvalidLayer { layer: cx.layer().to_owned(), detail }
}

impl LayerCodec for Keywords {
    fn export(&self, cx: &ExportCx) -> Result<Vec<u8>, ExportError> {
        let mut w = Writer::new();
        put_spans(&mut w, &self.spans, cx)?;
        put_tagset(&mut w, &self.algorithms)?;
        w.usize(self.spans.len())?;
        for i in 0..self.spans.len() {
            w.str(self.phrase.get(i))?;
            w.f64(self.score[i]);
            w.u16(self.algorithm[i].0);
            w.u16(self.terms[i]);
        }
        Ok(w.finish())
    }

    fn import(data: &[u8], cx: &mut ImportCx, doc: &Document) -> Result<Self, ExportError> {
        let mut r = Reader::new(data);
        let spans = get_spans(&mut r, cx)?;
        let algorithms = get_tagset(&mut r)?;
        let rows = r.usize()?;
        if rows != spans.len() {
            return Err(malformed(format!("{} keyword rows for {} spans", rows, spans.len())));
        }
        let mut layer = Keywords {
            spans,
            phrase: Default::default(),
            score: Vec::new(),
            algorithm: Vec::new(),
            algorithms,
            terms: Vec::new(),
        };
        for _ in 0..rows {
            let phrase = r.string()?;
            let score = r.f64()?;
            let algorithm = r.u16()?;
            let terms = r.u16()?;
            if phrase.is_empty() {
                return Err(invalid(cx, "an empty keyphrase".into()));
            }
            if !score.is_finite() || !(0.0..=1.0).contains(&score) {
                return Err(invalid(cx, format!("a keyword score outside [0, 1]: {score}")));
            }
            if algorithm as usize >= layer.algorithms.len() {
                return Err(malformed(format!("an algorithm outside {}", layer.algorithms.name)));
            }
            if terms == 0 {
                return Err(invalid(cx, "a keyphrase with a term count of zero".into()));
            }
            layer.phrase.push(&phrase);
            layer.score.push(score);
            layer.algorithm.push(turbo_analysis::Tag(algorithm));
            layer.terms.push(terms);
        }
        if !r.is_done() {
            return Err(malformed("trailing bytes in a keywords payload"));
        }

        let text = doc.text(spans_view(&layer));
        for (_, span) in layer.spans.iter() {
            let (s, e) = (span.start as usize, span.end as usize);
            if s > e || e > text.len() || !text.is_char_boundary(s) || !text.is_char_boundary(e) {
                return Err(invalid(
                    cx,
                    format!("span {s}..{e} is not inside view text of length {} on UTF-8 boundaries", text.len()),
                ));
            }
        }
        layer.check().map_err(|detail| invalid(cx, detail))?;
        Ok(layer)
    }
}

fn spans_view(layer: &Keywords) -> turbo_analysis::ViewId {
    layer.spans.view()
}

/// Registers this crate's layer codec on `codecs`.
pub fn register_codecs(codecs: &mut Codecs) {
    codecs.register::<Keywords>();
}

/// The standard codecs plus this crate's, for callers that build no
/// registry of their own.
pub fn codecs() -> Codecs {
    let mut codecs = Codecs::new();
    turbo_analysis_layers::export::register_codecs(&mut codecs);
    register_codecs(&mut codecs);
    codecs
}
