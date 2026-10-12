//! The shared part of the normalizers that replace stretches of text with
//! placeholders (URLs, e-mail addresses, hashtags, numbers): one scan
//! function per normalizer writes the view and reports each replaced
//! stretch to a sink. The `Normalizer` impl passes a sink that drops the
//! reports; the `Annotator` impl passes one that fills a `Placeholders`
//! layer. Both run the same scan once.

use turbo_analysis::{Span, View, ViewBuilder};
use turbo_normalize_layers::{PlaceholderKind, Placeholders};

/// Receives each replaced stretch: its span in the text the scan read and
/// what it was.
pub(crate) type Sink<'s> = dyn FnMut(Span, PlaceholderKind) + 's;

/// A sink for scans whose placeholders nobody reads.
pub(crate) fn discard(_: Span, _: PlaceholderKind) {}

/// Byte offsets `[start, end)` as a span. Texts are limited to `u32::MAX`
/// bytes throughout the analysis crates.
pub(crate) fn span(start: usize, end: usize) -> Span {
    Span::new(start as u32, end as u32)
}

/// Runs `scan` over the text of `src` and returns the view it wrote and
/// the placeholders it reported, with spans in `src`.
pub(crate) fn annotate(
    src: &View,
    scan: impl FnOnce(&str, &mut ViewBuilder<'_>, &mut Sink<'_>),
) -> (View, Placeholders) {
    let text = src.text();
    let mut out = ViewBuilder::new(text);
    let mut found = Placeholders::new(src.id());
    scan(text, &mut out, &mut |s, kind| {
        found.spans.push(s);
        found.kind.push(kind);
    });
    let (normalized, alignment) = out.finish();
    (View::derived(src, normalized, alignment), found)
}
