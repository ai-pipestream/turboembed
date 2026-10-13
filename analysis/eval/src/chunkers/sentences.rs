//! Windows of a fixed number of sentences.

use turbo_analysis::Document;
use turbo_analysis_layers::Sentences;

use super::{Chunker, chunk_span, need, setting};
use crate::Result;
use crate::analysis::sentence_run;
use crate::embedder::Embedder;
use crate::record::Setting;
use crate::spans::Span;

/// Consecutive windows of `sentences` sentences, without overlap; the last
/// may hold fewer. A window spans from its first sentence's start to its
/// last sentence's end, with what lies between them.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct SentenceWindows {
    sentences: u32,
}

impl SentenceWindows {
    pub fn new(sentences: u32) -> Result<SentenceWindows> {
        if sentences == 0 {
            return Err("a window of 0 sentences".into());
        }
        Ok(SentenceWindows { sentences })
    }
}

impl Chunker for SentenceWindows {
    fn name(&self) -> &'static str {
        "sentence-windows"
    }

    fn config(&self) -> Vec<Setting> {
        vec![setting("sentences", self.sentences)]
    }

    fn chunk(&self, doc: &Document, _: &mut dyn Embedder) -> Result<Vec<Span>> {
        let s = need::<Sentences>(doc, self.name())?;
        let n = s.spans.len();
        let size = self.sentences as usize;
        Ok((0..n).step_by(size).filter_map(|i| chunk_span(sentence_run(s, i, (i + size).min(n)))).collect())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::chunkers::tests::{analyzed, texts};
    use crate::embedder::stub::Stub;

    #[test]
    fn windows_by_hand() {
        // Five sentences, a line each: windows of two are 1-2, 3-4 and 5.
        let text = "s one\ns two\ns three\n\ns four\ns five\n";
        let mut e = Stub::new(&["s"]);
        let doc = analyzed(text, &e, None);
        let spans = SentenceWindows::new(2).unwrap().chunk(&doc, &mut e).unwrap();
        assert_eq!(texts(text, &spans), ["s one\ns two", "s three\n\ns four", "s five"]);
        let spans = SentenceWindows::new(5).unwrap().chunk(&doc, &mut e).unwrap();
        assert_eq!(texts(text, &spans), ["s one\ns two\ns three\n\ns four\ns five"]);
        assert!(SentenceWindows::new(0).is_err());
    }
}
