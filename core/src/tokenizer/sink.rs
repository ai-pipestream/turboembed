//! Token sinks with optional spans and source-span helpers.

use super::*;

impl Sink for Vec<i32> {
    const SPANS: bool = false;

    #[inline]
    fn len(&self) -> usize {
        Vec::len(self)
    }

    #[inline]
    fn push(&mut self, id: i32, _: impl FnOnce() -> Span) {
        Vec::push(self, id);
    }

    #[inline]
    fn widen_last(&mut self, _: impl FnOnce() -> Span) {}

    fn truncate(&mut self, len: usize) {
        Vec::truncate(self, len);
    }

    fn drain(&mut self, r: Range<usize>) {
        Vec::drain(self, r);
    }

    fn reverse_from(&mut self, at: usize) {
        self[at..].reverse();
    }

    fn retain_from(&mut self, at: usize, keep: impl Fn(i32) -> bool) {
        let mut kept = at;
        for i in at..Vec::len(self) {
            if keep(self[i]) {
                self[kept] = self[i];
                kept += 1;
            }
        }
        Vec::truncate(self, kept);
    }

    fn split_off(&mut self, at: usize) -> (Vec<i32>, Vec<Span>) {
        (Vec::split_off(self, at), Vec::new())
    }

    fn extend_with(&mut self, ids: &[i32], _: &[Span]) {
        self.extend_from_slice(ids);
    }
}

/// A row's ids and spans, one span per id.
pub(crate) struct Spanned<'a> {
    pub(super) ids: &'a mut Vec<i32>,
    pub(super) spans: &'a mut Vec<Span>,
}

impl Sink for Spanned<'_> {
    const SPANS: bool = true;

    fn len(&self) -> usize {
        self.ids.len()
    }

    fn push(&mut self, id: i32, span: impl FnOnce() -> Span) {
        self.ids.push(id);
        self.spans.push(span());
    }

    fn widen_last(&mut self, span: impl FnOnce() -> Span) {
        let s = span();
        if let Some(last) = self.spans.last_mut() {
            *last = [last[0].min(s[0]), last[1].max(s[1])];
        }
    }

    fn truncate(&mut self, len: usize) {
        self.ids.truncate(len);
        self.spans.truncate(len);
    }

    fn drain(&mut self, r: Range<usize>) {
        self.ids.drain(r.clone());
        self.spans.drain(r);
    }

    fn reverse_from(&mut self, at: usize) {
        self.ids[at..].reverse();
        self.spans[at..].reverse();
    }

    fn retain_from(&mut self, at: usize, keep: impl Fn(i32) -> bool) {
        let mut kept = at;
        for i in at..self.ids.len() {
            if keep(self.ids[i]) {
                self.ids[kept] = self.ids[i];
                self.spans[kept] = self.spans[i];
                kept += 1;
            }
        }
        self.truncate(kept);
    }

    fn split_off(&mut self, at: usize) -> (Vec<i32>, Vec<Span>) {
        (self.ids.split_off(at), self.spans.split_off(at))
    }

    fn extend_with(&mut self, ids: &[i32], spans: &[Span]) {
        self.ids.extend_from_slice(ids);
        self.spans.extend_from_slice(spans);
    }
}

/// The span of a token made of normalized bytes whose sources are
/// `sources`: from the first source byte to the last, whitespace at
/// either end of `text` left out unless that is all there is.
pub(crate) fn covering(text: &str, sources: &[Span]) -> Span {
    let Some(&first) = sources.first() else { return NOWHERE };
    let mut s = first;
    for a in &sources[1..] {
        s = [s[0].min(a[0]), s[1].max(a[1])];
    }
    trim_span(text, s)
}

/// `s` with whitespace at either end of `text` left out, unless that is
/// all there is.
pub(crate) fn trim_span(text: &str, s: Span) -> Span {
    let piece = &text[s[0] as usize..s[1] as usize];
    let inner = piece.trim();
    if inner.is_empty() {
        return s;
    }
    let start = s[0] + (inner.as_ptr() as usize - piece.as_ptr() as usize) as u32;
    [start, start + inner.len() as u32]
}
