//! The merge-walk both extractors run: `Sentences` and `Tokens` are in
//! document order, so one cursor assigns each token its sentence index.
//! A token outside every sentence fails the document.

use turbo_analysis::Error;
use turbo_analysis_layers::{Sentences, Tokens};

/// The sentence index of every token, in token order.
pub(crate) fn sentence_of(sentences: &Sentences, tokens: &Tokens, who: &'static str) -> Result<Vec<u32>, Error> {
    let (tok_start, _) = tokens.spans.columns();
    let (sent_start, sent_end) = sentences.spans.columns();
    let mut out = vec![0u32; tok_start.len()];
    let mut s = 0usize;
    for (i, &start) in tok_start.iter().enumerate() {
        while s < sent_start.len() && start >= sent_end[s] {
            s += 1;
        }
        if s >= sent_start.len() || start < sent_start[s] {
            return Err(Error::Annotator { annotator: who, detail: format!("token {i} is outside every sentence") });
        }
        out[i] = s as u32;
    }
    Ok(out)
}
