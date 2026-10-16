//! The layers [`crate::SpanCategorizer`] categorizes item by item. The
//! layer it writes, `Categories<L>` (`turbo:categories<L>`), is the shared
//! [`turbo_analysis_layers::Categories`].

use turbo_analysis::{Ix, Span, Spans};
use turbo_analysis_layers::{CategoryScope, Chunks, Sentences, Structure};

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
