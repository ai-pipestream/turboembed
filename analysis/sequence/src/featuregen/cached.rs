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
    /// The cached features by token index, paired with the generation in
    /// which the slot was populated (`0` when empty or evicted). Keeping
    /// the `Features` buffers across sentences avoids reallocating them on
    /// every token.
    slots: Vec<(u64, Features)>,
    /// The cached indexes, oldest first.
    order: std::collections::VecDeque<usize>,
    /// The current sentence generation (always >= 1).
    generation: u64,
}

impl Cached {
    pub fn new(inner: BoxedGenerator) -> Cached {
        Cached { inner, slots: Vec::new(), order: std::collections::VecDeque::new(), generation: 1 }
    }

    fn clear_cache(&mut self) {
        self.generation = self.generation.wrapping_add(1);
        if self.generation == 0 {
            for (at, _) in &mut self.slots {
                *at = 0;
            }
            self.generation = 1;
        }
        self.order.clear();
    }
}

impl FeatureGenerator for Cached {
    fn create_features(&mut self, out: &mut Features, tokens: &[&str], index: usize, previous: &[&str]) {
        if let Some((at, cached)) = self.slots.get(index)
            && *at == self.generation
        {
            out.extend_from(cached);
            return;
        }
        if self.slots.len() <= index {
            self.slots.resize_with(index + 1, || (0, Features::new()));
        }
        let (at, features) = &mut self.slots[index];
        features.clear();
        self.inner.create_features(features, tokens, index, previous);
        out.extend_from(features);
        *at = self.generation;
        self.order.push_back(index);
        if self.order.len() > CAPACITY {
            let oldest = self.order.pop_front().expect("the cache is not empty");
            self.slots[oldest].0 = 0;
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

#[cfg(test)]
mod tests {
    use super::super::TokenGenerator;
    use super::*;

    #[test]
    fn cached_reuses_slots_across_sequences_and_wraps_generation_cleanly() {
        let mut cached = Cached::new(Box::new(TokenGenerator::new(true)));
        cached.generation = u64::MAX;
        let s1 = ["Alpha", "Beta"];
        let mut out = Features::new();
        cached.create_features(&mut out, &s1, 0, &[]);
        assert_eq!(out.get(0), b"w=alpha");

        // `start_sequence` wraps `u64::MAX` -> 0 -> resets slot generations and sets `generation = 1`.
        cached.start_sequence();
        assert_eq!(cached.generation, 1);
        out.clear();
        let s2 = ["Gamma", "Delta"];
        cached.create_features(&mut out, &s2, 0, &[]);
        assert_eq!(out.get(0), b"w=gamma");
    }

    #[test]
    fn a_slot_of_the_last_sentence_is_not_served_in_the_next() {
        let mut cached = Cached::new(Box::new(TokenGenerator::new(true)));
        let mut out = Features::new();
        cached.create_features(&mut out, &["Alpha"], 0, &[]);
        assert_eq!(out.get(0), b"w=alpha");
        cached.start_sequence();
        out.clear();
        cached.create_features(&mut out, &["Gamma"], 0, &[]);
        assert_eq!(out.get(0), b"w=gamma");
        // Within a sentence the slot is served again, as Java's cache does,
        // whatever the tokens now say.
        out.clear();
        cached.create_features(&mut out, &["Delta"], 0, &[]);
        assert_eq!(out.get(0), b"w=gamma");
    }

    #[test]
    fn the_oldest_slot_beyond_the_capacity_is_evicted() {
        let mut cached = Cached::new(Box::new(TokenGenerator::new(true)));
        let tokens: Vec<String> = (0..=CAPACITY).map(|i| format!("T{i}")).collect();
        let tokens: Vec<&str> = tokens.iter().map(String::as_str).collect();
        let mut out = Features::new();
        for i in 0..=CAPACITY {
            out.clear();
            cached.create_features(&mut out, &tokens, i, &[]);
        }
        // Index 0 was the oldest of CAPACITY + 1 entries, so it is computed
        // again; index 1 is still held.
        assert_eq!(cached.slots[0].0, 0);
        assert_eq!(cached.slots[1].0, cached.generation);
        let changed: Vec<&str> = tokens.iter().map(|_| "x").collect();
        out.clear();
        cached.create_features(&mut out, &changed, 1, &[]);
        assert_eq!(out.get(0), b"w=t1");
        out.clear();
        cached.create_features(&mut out, &changed, 0, &[]);
        assert_eq!(out.get(0), b"w=x");
    }
}
