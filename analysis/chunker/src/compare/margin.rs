//! What a run's own costs say about a stretch cut either way: the run's
//! boundary costs at the cuts, plus each chunk's size cost, as the
//! optimizer adds them up.

use turbo_analysis::Ix;
use turbo_analysis_layers::Chunks;

use super::Costs;

/// One run's choice over a stretch: its chunks, the row of its chunk at
/// the shared cut the stretch follows, and the rows of the cuts it made
/// in the stretch.
pub(super) type Choice<'c> = (&'c Chunks, usize, &'c [usize]);

/// How much more `costs` charge for `other`'s choice than for `own`'s.
/// Both choices share the cut before the stretch and the one after it, so
/// whatever lies outside the stretch, and the boundary that closes it,
/// is the same in both and drops out.
pub(super) fn margin(costs: Option<Costs<'_>>, own: Choice<'_>, other: Choice<'_>) -> Option<f64> {
    let c = costs?;
    Some(price(&c, other)? - price(&c, own)?)
}

/// What `c` charge for a choice: the boundary at each of its cuts, and the
/// size of each of its chunks from the shared cut on. Sizes are the
/// layer's own token counts, which do not depend on the backend: the
/// subwords come from the tokenizer, not the model. Summed in f64.
fn price(c: &Costs<'_>, (chunks, first, cuts): Choice<'_>) -> Option<f64> {
    let mut total = size(c, chunks.model_tokens[first]);
    for &row in cuts {
        let start = chunks.spans.get(Ix::new(row as u32)).start;
        total += f64::from(boundary(c, start)?) + size(c, chunks.model_tokens[row]);
    }
    Some(total)
}

/// The cost of a boundary at this offset: the cost after the sentence
/// that ends before the one starting here. `None` where no sentence but
/// the first starts.
pub(super) fn boundary(c: &Costs<'_>, offset: u32) -> Option<f32> {
    let (starts, _) = c.sentences.spans.columns();
    let i = starts.partition_point(|&s| s < offset);
    (i > 0 && starts.get(i) == Some(&offset)).then(|| c.boundaries.cost[i - 1])
}

/// A chunk's size cost, in f32 as `segment` computes it: its squared
/// distance from the target, relative to the target, weighted. The
/// layer's token count already holds the overhead.
fn size(c: &Costs<'_>, tokens: u32) -> f64 {
    let b = c.budget;
    let d = (tokens as f32 - b.target as f32) / b.target as f32;
    f64::from(b.size_weight * d * d)
}
