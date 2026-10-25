//! The `ProjectChunks` vector projection (issue #1220): request caps and
//! wire spans over the library's deterministic PCA,
//! [`turbo_chunker::VectorProjection`].

use turbo_analysis::{Ix, Layer, Span};
use turbo_analysis_layers::Vectors;
use turbo_chunker::{ProjectionDims, VectorProjection};

use crate::api::{Failure, TURBO_E_INVALID_ARGUMENT};
use crate::offsets::Offsetter;
use crate::proto::result::{ProjectedPointProto, VectorProjectionProto};
use crate::results::{model_to_proto, over};

/// Maximum number of vector rows accepted by `compute_pca`.
pub const MAX_PROJECT_ROWS: usize = 512;

/// Maximum embedding dimension accepted by `compute_pca`.
pub const MAX_PROJECT_DIMS: usize = 2_048;

fn bad(message: impl Into<String>) -> Failure {
    Failure::new(TURBO_E_INVALID_ARGUMENT, 0, message)
}

/// Validates that every value in every present row of `vectors` is finite.
pub(crate) fn validate_finite_vectors<L: Layer>(vectors: &Vectors<L>, n: usize) -> Result<(), Failure> {
    for row_idx in 0..n {
        let ix = Ix::new(row_idx as u32);
        if vectors.is_empty(ix) {
            continue;
        }
        for (coord_idx, &val) in vectors.row(ix).iter().enumerate() {
            if !val.is_finite() {
                return Err(bad(format!("vectors row {row_idx} coordinate {coord_idx} is non-finite ({val})")));
            }
        }
    }
    Ok(())
}

/// Projects `vectors` (aligned with `spans` and `levels`) into `pca_dims` (2
/// or 3) principal components, returning [`VectorProjectionProto`]. The
/// projection is the library's [`VectorProjection`]; this adds the request
/// caps and the wire spans.
pub(crate) fn compute_pca<L: Layer>(
    vectors: &Vectors<L>,
    spans: &[Span],
    levels: &[u8],
    text: &str,
    offsetter: &Offsetter,
    pca_dims: u32,
) -> Result<VectorProjectionProto, Failure> {
    let n = spans.len();
    let d = vectors.dim as usize;
    if n > MAX_PROJECT_ROWS {
        return Err(over("max_project_rows", "projected vector rows", n as u64));
    }
    if d > MAX_PROJECT_DIMS {
        return Err(over("max_project_dims", "projected vector dimensions", d as u64));
    }
    validate_finite_vectors(vectors, n)?;
    let k = pca_dims as usize;
    let model = Some(model_to_proto(&vectors.model));
    if n == 0 || d == 0 {
        return Ok(VectorProjectionProto { model, pca_dims, explained_variance: vec![0.0; k], points: Vec::new() });
    }
    let dims = if pca_dims == 3 { ProjectionDims::D3 } else { ProjectionDims::D2 };
    let projection = VectorProjection::compute(vectors, dims).map_err(|e| bad(format!("vector projection: {e}")))?;
    let points = (0..n)
        .map(|i| {
            let ix = Ix::new(i as u32);
            let (start, end) = offsetter.to_wire(text, spans[i]);
            let row = projection.row(ix);
            ProjectedPointProto {
                item: i as u32,
                start,
                end,
                level: u32::from(levels.get(i).copied().unwrap_or(0)),
                x: row[0],
                y: row[1],
                z: row.get(2).copied().unwrap_or(0.0),
                empty: vectors.is_empty(ix),
                adjacent_cosine: projection.adjacent_cosine[i],
            }
        })
        .collect();
    Ok(VectorProjectionProto {
        model,
        pca_dims,
        explained_variance: projection.explained_variance[..k].to_vec(),
        points,
    })
}

#[cfg(test)]
mod tests;
