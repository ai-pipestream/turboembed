//! Item filtering for the shared layers: the `Filterable` impls a
//! [`FilterAnnotator`] (`turbo_analysis::FilterAnnotator`, porting
//! OpenNLP's `FilterAnnotator`) needs. Every impl preserves what the
//! layer promises: spans and discontinuous fragments (via
//! [`Spans::subset`]) and the columns aligned or indexed with the items.

use turbo_analysis::{Filterable, Ix, SpanItem};

use crate::{Sentences, Tokens};

impl Filterable for Tokens {
    type Item = SpanItem;

    fn item(&self, i: usize) -> SpanItem {
        SpanItem { view: self.spans.view(), span: self.spans.get(Ix::new(i as u32)) }
    }

    fn filter(&self, keep: &mut dyn FnMut(&SpanItem) -> bool) -> Tokens {
        let kept: Vec<usize> = (0..self.spans.len()).filter(|&i| keep(&self.item(i))).collect();
        Tokens {
            spans: self.spans.subset(&kept),
            sentence: self.sentence.as_ref().map(|s| kept.iter().map(|&i| s[i]).collect()),
        }
    }
}

impl Filterable for Sentences {
    type Item = SpanItem;

    fn item(&self, i: usize) -> SpanItem {
        SpanItem { view: self.spans.view(), span: self.spans.get(Ix::new(i as u32)) }
    }

    fn filter(&self, keep: &mut dyn FnMut(&SpanItem) -> bool) -> Sentences {
        let kept: Vec<usize> = (0..self.spans.len()).filter(|&i| keep(&self.item(i))).collect();
        Sentences { spans: self.spans.subset(&kept) }
    }
}
