//! The document graph codec for this crate's layer, so matched
//! glossary terms and the glossary they point into cross the export
//! boundary with the rest of the graph.

use std::sync::Arc;

use turbo_analysis::export::{
    Codecs, ExportCx, ExportError, ImportCx, LayerCodec, LayerRef, NodeRef, Reader, Writer, get_spans, put_spans,
};
use turbo_analysis::{Document, IxRange, Key, Layer};
use turbo_analysis_layers::Tokens;

use crate::{EntryIx, Glossary, GlossaryTerms};

fn malformed(detail: impl Into<String>) -> ExportError {
    ExportError::Malformed(detail.into())
}

fn invalid(cx: &ImportCx, detail: String) -> ExportError {
    ExportError::InvalidLayer { layer: cx.layer().to_owned(), detail }
}

impl LayerCodec for GlossaryTerms {
    fn export(&self, cx: &ExportCx) -> Result<Vec<u8>, ExportError> {
        let mut w = Writer::new();
        put_spans(&mut w, &self.spans, cx)?;
        // The token ranges index through the default key, as the
        // layer's own `indexes` declares, so they cross under the
        // instance that key resolves to on this document.
        let target = cx.resolve(&Key::<Tokens>::DEFAULT);
        w.str(&target.layer)?;
        w.str(&target.instance)?;
        w.usize(self.tokens.len())?;
        for range in &self.tokens {
            w.u32(range.start.get() as u32);
            w.u32(range.end.get() as u32);
        }
        w.usize(self.glossary.len())?;
        for (_, entry) in self.glossary.iter() {
            w.str(entry.id)?;
            w.str(entry.term)?;
        }
        w.usize(self.entry.len())?;
        for entry in &self.entry {
            w.u32(entry.0);
        }
        Ok(w.finish())
    }

    fn import(data: &[u8], cx: &mut ImportCx, doc: &Document) -> Result<Self, ExportError> {
        let mut r = Reader::new(data);
        let spans = get_spans(&mut r, cx)?;

        let target = LayerRef { layer: r.string()?, instance: r.string()? };
        // No key is stored for the tokens, so the document binds the
        // ranges through the default key and its provenance. The
        // payload must name exactly that instance, the empty name
        // included: an empty target whose provenance resolves to a
        // named tokens instance is refused here rather than being
        // checked against the default and rebound to the named one.
        cx.expect_target::<Tokens>(&target)?;
        let tokens_len = cx.layer_len(&target)?;
        let mut tokens = Vec::new();
        for _ in 0..r.usize()? {
            let (start, end) = (r.u32()?, r.u32()?);
            if start > end {
                return Err(malformed(format!("a token range that runs backwards: {start}..{end}")));
            }
            if end as usize > tokens_len {
                return Err(ExportError::DanglingEndpoint {
                    layer: cx.layer().to_owned(),
                    endpoint: NodeRef { layer: target.clone(), index: end },
                });
            }
            tokens.push(IxRange::new(start, end));
        }

        let entries = r.usize()?;
        if entries > u32::MAX as usize {
            return Err(malformed("a glossary longer than an entry index can name"));
        }
        let mut glossary = Glossary::new();
        for _ in 0..entries {
            let (id, term) = (r.string()?, r.string()?);
            glossary.push(&id, &term).map_err(|e| invalid(cx, e.to_string()))?;
        }

        let mut entry = Vec::new();
        for _ in 0..r.usize()? {
            let at = r.u32()?;
            if at as usize >= glossary.len() {
                return Err(malformed(format!("an entry {at} outside the glossary of {}", glossary.len())));
            }
            entry.push(EntryIx(at));
        }

        let n = spans.len();
        if tokens.len() != n || entry.len() != n {
            return Err(malformed(format!("{} token ranges and {} entries for {n} terms", tokens.len(), entry.len())));
        }
        if !r.is_done() {
            return Err(malformed("trailing bytes in a glossary terms payload"));
        }

        let text = doc.text(spans.view());
        for (i, _) in spans.iter() {
            for frag in spans.fragments(i) {
                let (s, e) = (frag.start as usize, frag.end as usize);
                if s > e || e > text.len() || !text.is_char_boundary(s) || !text.is_char_boundary(e) {
                    return Err(invalid(
                        cx,
                        format!(
                            "span {}..{} is not inside view text of length {} on UTF-8 boundaries",
                            frag.start,
                            frag.end,
                            text.len()
                        ),
                    ));
                }
            }
        }

        let layer = GlossaryTerms { spans, tokens, entry, glossary: Arc::new(glossary) };
        layer.check().map_err(|detail| invalid(cx, detail))?;
        Ok(layer)
    }
}

/// Registers this crate's layer codec on `codecs`.
pub fn register_codecs(codecs: &mut Codecs) {
    codecs.register::<GlossaryTerms>();
}

/// A registry holding this crate's codec and nothing else.
pub fn codecs() -> Codecs {
    let mut codecs = Codecs::new();
    register_codecs(&mut codecs);
    codecs
}
