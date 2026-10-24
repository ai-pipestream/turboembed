//! Borrow token text once and retain sentence lengths for neural batches.
use turbo_analysis::{Error, View};
use turbo_analysis_layers::Tokens;

pub(super) fn words<'a>(
    caller: &'static str,
    view: &'a View,
    tokens: &Tokens,
    out: &mut Vec<&'a str>,
) -> Result<Vec<usize>, Error> {
    let fail = |detail| Error::Annotator { annotator: caller, detail };
    if tokens.spans.view() != view.id() {
        return Err(fail(format!(
            "the tokens are in view {}, not in view {} that is tagged",
            tokens.spans.view().0,
            view.id().0
        )));
    }
    let (starts, ends) = tokens.spans.columns();
    let n = starts.len();
    if tokens.sentence.as_ref().is_some_and(|s| s.len() != n) {
        return Err(fail("token sentence column length differs from token count".into()));
    }
    crate::tagger::slice_sentence_words(caller, view.text(), starts, ends, 0, n, out)?;
    let mut lens = Vec::new();
    let mut from = 0;
    while from < n {
        let len = tokens.sentence.as_ref().map_or(n - from, |s| s[from..].partition_point(|&i| i == s[from]));
        lens.push(len);
        from += len;
    }
    Ok(lens)
}
