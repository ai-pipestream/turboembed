//! What a run's own costs say about a stretch cut either way: the sum of
//! each chunk's cost as the optimizer computes it (`chunk_cost`), from
//! the shared cut the stretch follows to the end of the stretch.

use turbo_analysis::Ix;
use turbo_analysis_layers::Chunks;

use super::Pricing;
use crate::chunk_cost;

/// One run's choice over a stretch: its chunks, the row of its chunk at
/// the shared cut the stretch follows, and the rows of the cuts it made
/// in the stretch.
pub(super) type Choice<'c> = (&'c Chunks, usize, &'c [usize]);

/// How much more `p` charges for `other`'s choice than for `own`'s. Both
/// choices share the cut before the stretch and the one that ends it
/// (`before`, or the end of the parent), so whatever lies outside the
/// stretch is the same in both and drops out: the difference is how much
/// more the run's whole cut would cost had it cut the stretch as `other`
/// did.
pub(super) fn margin(p: Option<Pricing<'_>>, before: Option<u32>, own: Choice<'_>, other: Choice<'_>) -> Option<i128> {
    let p = p?;
    Some(price(&p, before, other)? - price(&p, before, own)?)
}

/// What `p` charges for a choice's chunks: each closed by the next cut,
/// the last by `before` (free at the end of the parent, as the end of the
/// optimizer's run is), sized by the layer's own token count, which
/// holds the overhead and does not depend on the backend: the subwords
/// come from the tokenizer, not the model. The layer records no chunk
/// term, so none is added.
fn price(p: &Pricing<'_>, before: Option<u32>, (chunks, first, cuts): Choice<'_>) -> Option<i128> {
    let start = |row: usize| chunks.spans.get(Ix::new(row as u32)).start;
    let rows = std::iter::once(first).chain(cuts.iter().copied());
    let mut closing = cuts.iter().map(|&r| Some(start(r))).chain(std::iter::once(before));
    let mut total = 0i128;
    for row in rows {
        let closes = match closing.next().flatten() {
            Some(at) => boundary(p, at)?,
            None => 0,
        };
        let size = u64::from(chunks.model_tokens[row]);
        total = total.saturating_add(chunk_cost(closes, size, p.budget, 0));
    }
    Some(total)
}

/// The cost of a boundary at this offset: the cost after the sentence
/// that ends before the one starting here, as the chunker reads it.
/// `None` where no sentence but the first starts.
pub(super) fn boundary(p: &Pricing<'_>, offset: u32) -> Option<i32> {
    let (starts, _) = p.sentences.spans.columns();
    let i = starts.partition_point(|&s| s < offset);
    (i > 0 && starts.get(i) == Some(&offset)).then(|| p.boundaries.cost[i - 1])
}
