//! The whole document as one chunk.

use turbo_analysis::Document;
use turbo_analysis_layers::Sentences;

use super::{Chunker, chunk_span, need};
use crate::Result;
use crate::analysis::sentence_run;
use crate::embedder::Embedder;
use crate::record::Setting;
use crate::spans::Span;

/// One chunk per document, from its first sentence's start to its last
/// sentence's end: no chunking at all, the unit a document-level baseline
/// (BM25 over whole documents, as BEIR's) retrieves. A document with no
/// sentence has no chunk.
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq)]
pub struct WholeDocument;

impl Chunker for WholeDocument {
    fn name(&self) -> &'static str {
        "document"
    }

    fn config(&self) -> Vec<Setting> {
        Vec::new()
    }

    fn chunk(&self, doc: &Document, _: &mut dyn Embedder) -> Result<Vec<Span>> {
        let s = need::<Sentences>(doc, self.name())?;
        let n = s.spans.len();
        Ok(if n == 0 { Vec::new() } else { chunk_span(sentence_run(s, 0, n)).into_iter().collect() })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::chunkers::tests::{analyzed, texts};
    use crate::embedder::stub::Stub;

    #[test]
    fn a_document_is_one_chunk_from_its_first_sentence_to_its_last() {
        let text = "\n s one\ns two\n\ns three\n\n";
        let mut e = Stub::new(&["s"]);
        let doc = analyzed(text, &e, None);
        let spans = WholeDocument.chunk(&doc, &mut e).unwrap();
        assert_eq!(texts(text, &spans), ["s one\ns two\n\ns three"]);
        let empty = analyzed("\n\n", &e, None);
        assert!(WholeDocument.chunk(&empty, &mut e).unwrap().is_empty());
    }
}
