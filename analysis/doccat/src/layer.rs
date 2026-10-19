//! The layers [`crate::SpanCategorizer`] categorizes item by item. The
//! layer it writes, `Categories<L>` (`turbo:categories<L>`), is the shared
//! [`turbo_analysis_layers::Categories`].

use turbo_analysis::{Error, Ix, Span, Spans};
use turbo_analysis_layers::{CategoryScope, Chunks, Sentences, Structure, Tokens};

pub(crate) fn validate_spans<T>(
    annotator: &'static str,
    label: &'static str,
    spans: &Spans<T>,
    text: &str,
) -> Result<(), Error> {
    let (starts, ends) = spans.columns();
    if starts.len() != ends.len() {
        return Err(Error::Annotator {
            annotator,
            detail: format!(
                "{label} span columns have mismatched lengths ({} starts vs {} ends)",
                starts.len(),
                ends.len()
            ),
        });
    }
    for (idx, (&start, &end)) in starts.iter().zip(ends.iter()).enumerate() {
        let s = start as usize;
        let e = end as usize;
        if s > e || e > text.len() || !text.is_char_boundary(s) || !text.is_char_boundary(e) {
            return Err(Error::Annotator {
                annotator,
                detail: format!(
                    "{label} {idx} span {start}..{end} is not a valid UTF-8 boundary in view text of length {}",
                    text.len()
                ),
            });
        }
    }
    Ok(())
}

pub(crate) fn extract_tokens<'a>(
    annotator: &'static str,
    tokens: &Tokens,
    text: &'a str,
) -> Result<Vec<&'a str>, Error> {
    validate_spans(annotator, "token", &tokens.spans, text)?;
    let (starts, ends) = tokens.spans.columns();
    Ok(starts.iter().zip(ends.iter()).map(|(&s, &e)| &text[s as usize..e as usize]).collect())
}

/// A layer of spans (`Sentences`, `Structure`, `Chunks`) that
/// [`crate::SpanCategorizer`] categorizes span by span, writing
/// `Categories<Self>`.
pub trait SpanCategorizable: CategoryScope {
    /// The spans of this layer.
    fn layer_spans(&self) -> &Spans<Self>;

    /// The `i`-th span of this layer.
    fn span_at(&self, i: usize) -> Span {
        self.layer_spans().get(Ix::new(i as u32))
    }
}

impl SpanCategorizable for Sentences {
    fn layer_spans(&self) -> &Spans<Self> {
        &self.spans
    }
}

impl SpanCategorizable for Structure {
    fn layer_spans(&self) -> &Spans<Self> {
        &self.spans
    }
}

impl SpanCategorizable for Chunks {
    fn layer_spans(&self) -> &Spans<Self> {
        &self.spans
    }
}
