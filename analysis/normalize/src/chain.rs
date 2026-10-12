//! A chain of normalizers that runs as one.

use turbo_analysis::ViewBuilder;

use crate::{DynNormalizer, Normalizer, compose, hash_config};

/// Several normalizers run in order, with one alignment from the last
/// text back to the first. The intermediate texts exist only while the
/// chain runs; a pipeline sees one view.
///
/// OpenNLP's `TextNormalizer.builder()...buildAligned()` is the same
/// shape. Its default chain (`TextNormalizer.defaultChain()`) is invisible
/// characters, NFC, whitespace, quotes, dashes, case fold, accent fold.
#[derive(Default)]
pub struct Chain {
    steps: Vec<Box<dyn DynNormalizer>>,
}

impl Chain {
    pub fn new() -> Chain {
        Chain::default()
    }

    /// Adds a step after the ones already there.
    pub fn then<N: Normalizer>(mut self, n: N) -> Chain {
        self.steps.push(Box::new(n));
        self
    }

    pub fn then_dyn(mut self, n: Box<dyn DynNormalizer>) -> Chain {
        self.steps.push(n);
        self
    }

    pub fn len(&self) -> usize {
        self.steps.len()
    }

    pub fn is_empty(&self) -> bool {
        self.steps.is_empty()
    }

    /// The ids of the steps, in order.
    pub fn ids(&self) -> impl Iterator<Item = &'static str> + '_ {
        self.steps.iter().map(|s| s.id())
    }
}

impl Normalizer for Chain {
    const ID: &'static str = "turbo:normalize:chain";

    fn normalize(&self, text: &str, out: &mut ViewBuilder<'_>) {
        let Some((first, rest)) = self.steps.split_first() else { return };
        let mut b = ViewBuilder::new(text);
        first.normalize_dyn(text, &mut b);
        let (mut current, mut alignment) = b.finish();
        for step in rest {
            let mut b = ViewBuilder::new(&current);
            step.normalize_dyn(&current, &mut b);
            let (next, next_alignment) = b.finish();
            // Fold the new stage into the alignment back to `text`, so the
            // chain carries one text and one alignment at a time.
            let mut folded = ViewBuilder::new(text);
            compose(text, &alignment, &next, &next_alignment, &mut folded);
            (current, alignment) = folded.finish();
            debug_assert_eq!(current, next);
        }
        // Replay the composed alignment into the caller's builder.
        for e in alignment.edits() {
            out.keep(e.parent.start as usize);
            out.replace(e.parent.end as usize, &current[e.child.range()]);
        }
    }

    fn config_hash(&self) -> u64 {
        let parts: Vec<(&'static str, &'static str, u64)> =
            self.steps.iter().map(|s| (s.id(), s.version(), s.config_hash_dyn())).collect();
        hash_config(&parts)
    }
}
