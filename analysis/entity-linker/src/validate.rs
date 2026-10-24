//! Input checks for [`crate::EntityLinker`]: every span the linker slices
//! and every token index it follows is verified up front, so a direct
//! `annotate` call with a malformed layer returns [`Error::Annotator`]
//! instead of panicking partway through the document.

use turbo_analysis::{Annotator, Error, Layer, Span, View};
use turbo_analysis_layers::{Entities, Sentences, Tokens};

use crate::EntityLinker;

fn fail(detail: String) -> Error {
    Error::Annotator { annotator: EntityLinker::ID, detail }
}

/// Refuses a span that is inverted, runs past `text`, or starts or ends
/// inside a UTF-8 sequence.
fn check_span(text: &str, what: &str, i: usize, span: Span) -> Result<(), Error> {
    let (s, e) = (span.start as usize, span.end as usize);
    if text.get(s..e).is_none() {
        return Err(fail(format!(
            "{what} {i} span {s}..{e} is not a valid UTF-8 range in a view of length {}",
            text.len()
        )));
    }
    Ok(())
}

/// Validates the layers the linker reads. `sentences` and `tokens` are
/// only checked when they are in the entities' view, because the linker
/// ignores them otherwise.
pub(crate) fn check_inputs(
    view: &View,
    entities: &Entities,
    sentences: Option<&Sentences>,
    tokens: Option<&Tokens>,
) -> Result<(), Error> {
    if entities.spans.view() != view.id() {
        return Err(fail(format!(
            "Entities view {:?} does not match input View {:?}",
            entities.spans.view(),
            view.id()
        )));
    }
    let text = view.text();
    entities.check().map_err(|e| fail(format!("invalid Entities layer: {e}")))?;
    for (ix, span) in entities.spans.iter() {
        check_span(text, "entity", ix.get(), span)?;
    }
    if let Some(sents) = sentences.filter(|s| s.spans.view() == view.id()) {
        sents.check().map_err(|e| fail(format!("invalid Sentences layer: {e}")))?;
        for (ix, span) in sents.spans.iter() {
            check_span(text, "sentence", ix.get(), span)?;
        }
    }
    if let Some(toks) = tokens.filter(|t| t.spans.view() == view.id()) {
        toks.check().map_err(|e| fail(format!("invalid Tokens layer: {e}")))?;
        for (ix, span) in toks.spans.iter() {
            check_span(text, "token", ix.get(), span)?;
        }
        for (i, range) in entities.tokens.iter().enumerate() {
            if range.end.get() > toks.len() {
                return Err(fail(format!(
                    "entity {i} token range {}..{} runs past {} tokens",
                    range.start.get(),
                    range.end.get(),
                    toks.len()
                )));
            }
        }
    }
    Ok(())
}
