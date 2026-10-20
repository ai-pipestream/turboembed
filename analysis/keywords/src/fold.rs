//! The merge-walk both extractors run: `Sentences` and `Tokens` are in
//! document order, so one cursor assigns each token its sentence index.
//! The assignment is strict: the layers must be spans of one view, a
//! token must lie wholly inside its sentence (not only start there),
//! and the tokens' own sentence column, when present, must agree with
//! what the spans say.

use turbo_analysis::{Error, Key};
use turbo_analysis_layers::{Sentences, Tokens};

fn invalid(detail: String) -> Error {
    Error::Invalid { layer: Key::<Tokens>::DEFAULT.id(), detail }
}

/// The sentence index of every token, in token order.
pub(crate) fn sentence_of(sentences: &Sentences, tokens: &Tokens, who: &'static str) -> Result<Vec<u32>, Error> {
    if sentences.spans.view() != tokens.spans.view() {
        return Err(invalid(format!("{who}: sentences and tokens are in different views")));
    }
    let (tok_start, tok_end) = tokens.spans.columns();
    let (sent_start, sent_end) = sentences.spans.columns();
    let mut out = vec![0u32; tok_start.len()];
    let mut s = 0usize;
    for (i, &start) in tok_start.iter().enumerate() {
        while s < sent_start.len() && start >= sent_end[s] {
            s += 1;
        }
        // The whole token is inside the sentence: a token that only
        // starts there and runs past the end crosses the boundary and
        // is refused.
        if s >= sent_start.len() || start < sent_start[s] || tok_end[i] > sent_end[s] {
            return Err(invalid(format!(
                "{who}: token {i} at {start}..{} is not wholly inside one sentence",
                tok_end[i]
            )));
        }
        out[i] = s as u32;
    }
    if let Some(claimed) = &tokens.sentence
        && claimed.len() == out.len()
    {
        for (i, (&claimed, &found)) in claimed.iter().zip(&out).enumerate() {
            if claimed.get() != found as usize {
                return Err(invalid(format!(
                    "{who}: token {i} claims sentence {} but its span is in sentence {found}",
                    claimed.get()
                )));
            }
        }
    }
    Ok(out)
}
