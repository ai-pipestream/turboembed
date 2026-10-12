//! `CachedFeatureGenerator`.

use super::{BoxedGenerator, FeatureGenerator};
use crate::features::Features;

/// How many tokens' features Java's cache holds (`new Cache<>(100)`).
const CAPACITY: usize = 100;

/// `CachedFeatureGenerator`: remembers the features of each token of the
/// current sentence and hands them out again without asking the inner
/// generator.
///
/// The cache is keyed by the token's index alone, as Java's is. The
/// outcomes decided before the token are not part of the key, so an inner
/// generator that reads them (such as [`PreviousTags`](super::PreviousTags)
/// in the default part-of-speech descriptor, which sits under
/// `cache="true"`) gives every path of the beam the features computed for
/// the first path that reached the token. The models were trained and are
/// run that way, so this is reproduced, not fixed.
///
/// Java's cache is a `LinkedHashMap` in insertion order that drops its
/// oldest entry beyond 100; so is this one. The JVM property
/// `opennlp.featuregen.cache.disabled` is not read.
pub struct Cached {
    inner: BoxedGenerator,
    /// The cached features by token index.
    slots: Vec<Option<Features>>,
    /// The cached indexes, oldest first.
    order: std::collections::VecDeque<usize>,
}

impl Cached {
    pub fn new(inner: BoxedGenerator) -> Cached {
        Cached { inner, slots: Vec::new(), order: std::collections::VecDeque::new() }
    }

    fn clear_cache(&mut self) {
        for &i in &self.order {
            self.slots[i] = None;
        }
        self.order.clear();
    }
}

impl FeatureGenerator for Cached {
    fn create_features(&mut self, out: &mut Features, tokens: &[&str], index: usize, previous: &[&str]) {
        if let Some(Some(cached)) = self.slots.get(index) {
            out.extend_from(cached);
            return;
        }
        let mut features = Features::new();
        self.inner.create_features(&mut features, tokens, index, previous);
        out.extend_from(&features);
        if self.slots.len() <= index {
            self.slots.resize_with(index + 1, || None);
        }
        self.slots[index] = Some(features);
        self.order.push_back(index);
        if self.order.len() > CAPACITY {
            let oldest = self.order.pop_front().expect("the cache is not empty");
            self.slots[oldest] = None;
        }
    }

    fn update_adaptive_data(&mut self, tokens: &[&str], outcomes: &[&str]) {
        self.inner.update_adaptive_data(tokens, outcomes);
    }

    fn clear_adaptive_data(&mut self) {
        self.inner.clear_adaptive_data();
    }

    fn start_sequence(&mut self) {
        self.clear_cache();
        self.inner.start_sequence();
    }

    fn set_additional_context(&mut self, context: &[Vec<String>]) {
        self.inner.set_additional_context(context);
    }
}
