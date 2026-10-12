//! `WindowFeatureGenerator`.

use super::{BoxedGenerator, FeatureGenerator};
use crate::features::Features;

/// `WindowFeatureGenerator`: the features of the inner generator for the
/// token itself, then for each of the `prev` tokens before it (prefixed
/// `p1`, `p2`, ...) and the `next` tokens after it (`n1`, `n2`, ...),
/// nearest first. Tokens beyond the sentence's ends are skipped.
pub struct Window {
    inner: BoxedGenerator,
    prev: usize,
    next: usize,
    scratch: Features,
}

impl Window {
    pub fn new(inner: BoxedGenerator, prev: usize, next: usize) -> Window {
        Window { inner, prev, next, scratch: Features::new() }
    }
}

impl FeatureGenerator for Window {
    fn create_features(&mut self, out: &mut Features, tokens: &[&str], index: usize, previous: &[&str]) {
        self.inner.create_features(out, tokens, index, previous);
        for i in 1..=self.prev {
            if let Some(at) = index.checked_sub(i) {
                self.scratch.clear();
                self.inner.create_features(&mut self.scratch, tokens, at, previous);
                out.extend_numbered("p", i, &self.scratch);
            }
        }
        for i in 1..=self.next {
            if index + i < tokens.len() {
                self.scratch.clear();
                self.inner.create_features(&mut self.scratch, tokens, index + i, previous);
                out.extend_numbered("n", i, &self.scratch);
            }
        }
    }

    fn update_adaptive_data(&mut self, tokens: &[&str], outcomes: &[&str]) {
        self.inner.update_adaptive_data(tokens, outcomes);
    }

    fn clear_adaptive_data(&mut self) {
        self.inner.clear_adaptive_data();
    }

    fn start_sequence(&mut self) {
        self.inner.start_sequence();
    }

    fn set_additional_context(&mut self, context: &[Vec<String>]) {
        self.inner.set_additional_context(context);
    }
}
