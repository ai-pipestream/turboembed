//! Pairwise cosines for one level of a `ProjectChunks` projection (#2248):
//! the strict upper triangle over the level's items that carry a vector of
//! non-zero length, computed from the stored vectors in `f64`.

use turbo_analysis::math::{dot_f64, sum_sq_f64};
use turbo_analysis::{Document, Ix, Layer};
use turbo_analysis_layers::{Chunks, Sentences, Vectors};

use crate::api::{Failure, TURBO_E_INVALID_ARGUMENT};
use crate::proto::result::{PairwiseCosinesProto, PairwiseCosinesSpec, ProjectedPointProto};
use crate::results::over;

/// Items compared when the request leaves `max_items` at 0.
pub const DEFAULT_PAIRWISE_ITEMS: u32 = 256;

/// The most items `max_items` may name: 130,816 cosines.
pub const MAX_PAIRWISE_ITEMS: u32 = 512;

/// The cosines for `spec.level` over `points`, the projection just built
/// from `doc`, read from the same vectors layer the projection used.
pub(crate) fn pairwise(
    doc: &Document,
    spec: &PairwiseCosinesSpec,
    points: &[ProjectedPointProto],
) -> Result<PairwiseCosinesProto, Failure> {
    let cap = if spec.max_items == 0 { DEFAULT_PAIRWISE_ITEMS } else { spec.max_items };
    if cap > MAX_PAIRWISE_ITEMS {
        return Err(over("max_pairwise_items", "pairwise.max_items", u64::from(cap)));
    }
    if points.is_empty() {
        return Err(Failure::new(
            TURBO_E_INVALID_ARGUMENT,
            0,
            "pairwise: the document graph carries no chunk or sentence vectors to compare",
        ));
    }
    let level: Vec<u32> = points.iter().filter(|p| p.level == spec.level && !p.empty).map(|p| p.item).collect();
    let (items, cosines) = if let Some(vectors) = doc.get::<Vectors<Chunks>>() {
        compare(vectors, &level, cap)?
    } else if let Some(vectors) = doc.get::<Vectors<Sentences>>() {
        compare(vectors, &level, cap)?
    } else {
        (Vec::new(), Vec::new())
    };
    Ok(PairwiseCosinesProto { level: spec.level, items, cosines })
}

/// The rows of `candidates` with a non-zero norm, and the strict upper
/// triangle of their cosines, row-major. Refuses more than `cap` rows.
fn compare<L: Layer>(vectors: &Vectors<L>, candidates: &[u32], cap: u32) -> Result<(Vec<u32>, Vec<f32>), Failure> {
    let mut items = Vec::with_capacity(candidates.len());
    let mut norms = Vec::with_capacity(candidates.len());
    for &item in candidates {
        let norm = sum_sq_f64(vectors.row(Ix::new(item))).sqrt();
        if norm > 0.0 {
            items.push(item);
            norms.push(norm);
        }
    }
    if items.len() as u64 > u64::from(cap) {
        return Err(over(&format!("pairwise.max_items ({cap})"), "pairwise items", items.len() as u64));
    }
    let n = items.len();
    let mut cosines = Vec::with_capacity(n * n.saturating_sub(1) / 2);
    for a in 0..n {
        let row = vectors.row(Ix::new(items[a]));
        for b in a + 1..n {
            let cosine = dot_f64(row, vectors.row(Ix::new(items[b]))) / (norms[a] * norms[b]);
            cosines.push(cosine.clamp(-1.0, 1.0) as f32);
        }
    }
    Ok((items, cosines))
}
