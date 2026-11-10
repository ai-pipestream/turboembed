//! What the precision stage holds its inputs and its own layer to, and how
//! it trims a candidate.

use std::any::TypeId;

use turbo_analysis::{Document, Error, Ix, IxRange, Key, KeyId, Layer, Span, View};
use turbo_analysis_layers::{Entities, Pii, Tokens};

use crate::Decisions;
use crate::precision::PrecisionCandidates;

/// Every input sound on its own, all of them over `view`, and each entity
/// the span of its tokens.
pub(crate) fn validate_inputs(
    view: &View,
    tokens: &Tokens,
    entities: &Entities,
    decisions: &Decisions,
    pii: Option<&Pii>,
) -> Result<(), Error> {
    let fail = |detail: String| Error::Invalid { layer: Key::<PrecisionCandidates>::DEFAULT.id(), detail };
    tokens.check().map_err(fail)?;
    entities.check().map_err(fail)?;
    decisions.check().map_err(fail)?;
    if entities.spans.view() != view.id() || tokens.spans.view() != view.id() {
        return Err(fail("precision inputs must share one view".into()));
    }
    if let Some(pii) = pii {
        pii.check().map_err(fail)?;
        // A negative rule compares byte offsets, which mean nothing
        // across views.
        if pii.spans.view() != view.id() {
            return Err(fail("the PII layer must be over the view of the precision inputs".into()));
        }
    }
    if entities.spans.len() != decisions.logits.len() {
        return Err(fail("precision decisions must align to entities".into()));
    }
    for (_, span) in tokens.spans.iter() {
        if view.text().get(span.range()).is_none() {
            return Err(fail(format!("token span {}..{} is outside its view", span.start, span.end)));
        }
    }
    let (starts, ends) = tokens.spans.columns();
    for (i, range) in entities.tokens.iter().enumerate() {
        let (start, end) = (range.start.get(), range.end.get());
        if start >= end || end > tokens.spans.len() {
            return Err(fail(format!("entity {i} has an empty or out-of-bounds token range")));
        }
        let expected = Span::new(starts[start], ends[end - 1]);
        let span = entities.spans.get(Ix::new(i as u32));
        if span != expected || view.text().get(span.range()).is_none() {
            return Err(fail(format!("entity {i} does not match its token span")));
        }
    }
    Ok(())
}

/// Both span columns of `layer` against the tokens it was made on: each
/// row's original and final span is exactly the span of its token range,
/// in the tokens' view. The document checked the tokens against the text,
/// so a span that equals a token range is in bounds and on character
/// boundaries.
pub(crate) fn check_in_document(layer: &PrecisionCandidates, doc: &Document, on: &[KeyId]) -> Result<(), String> {
    let key = on.iter().find(|k| k.ty == TypeId::of::<Tokens>()).ok_or("precision candidates need their tokens")?;
    let tokens = doc.require(&Key::<Tokens>::named(key.name)).map_err(|e| e.to_string())?;
    if tokens.spans.view() != layer.spans.view() {
        return Err("precision candidates and their tokens are in different views".into());
    }
    let (starts, ends) = tokens.spans.columns();
    let of = |range: IxRange<Tokens>| {
        let (start, end) = (range.start.get(), range.end.get());
        (start < end && end <= starts.len()).then(|| Span::new(starts[start], ends[end - 1]))
    };
    for i in 0..layer.spans.len() {
        let row = layer.get(Ix::new(i as u32));
        if of(row.original_tokens) != Some(row.original_span) {
            return Err(format!("precision candidate {i}: the original span is not the span of its tokens"));
        }
        if of(row.tokens) != Some(row.span) {
            return Err(format!("precision candidate {i}: the span is not the span of its tokens"));
        }
    }
    Ok(())
}

/// A candidate after trimming.
pub(crate) struct Trimmed {
    pub span: Span,
    pub tokens: IxRange<Tokens>,
    /// Trimming for articles left an article and nothing else.
    pub bare_article: bool,
}

/// Drops the edge tokens of `range` that have no letter or digit and, with
/// `articles`, a leading article, again and again until neither applies,
/// so punctuation an article stood before goes too. At least one token is
/// kept.
pub(crate) fn trim(view: &View, tokens: &Tokens, range: IxRange<Tokens>, articles: bool) -> Trimmed {
    let (starts, ends) = tokens.spans.columns();
    let word = |i: usize| &view.text()[starts[i] as usize..ends[i] as usize];
    let empty = |i: usize| !word(i).chars().any(char::is_alphanumeric);
    let article = |i: usize| matches!(word(i).to_ascii_lowercase().as_str(), "a" | "an" | "the");
    let (mut start, mut end) = (range.start.get(), range.end.get());
    loop {
        while end - start > 1 && empty(start) {
            start += 1;
        }
        while end - start > 1 && empty(end - 1) {
            end -= 1;
        }
        if !(articles && end - start > 1 && article(start)) {
            break;
        }
        start += 1;
    }
    Trimmed {
        span: Span::new(starts[start], ends[end - 1]),
        tokens: IxRange::new(start as u32, end as u32),
        bare_article: articles && end - start == 1 && article(start),
    }
}
