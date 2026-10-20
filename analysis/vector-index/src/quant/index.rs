use turbovec::TurboQuantIndex;

use super::{CALIBRATION_ROWS, TurboQuant};
use crate::rows::{self, Block};
use crate::{Hit, IndexError, Metric, Space, Which};

/// An index whose rows are stored as TurboQuant codes. It answers the
/// same queries as [`crate::VectorIndex`], checks the query's [`Space`]
/// the same way, and keeps the same metadata; its scores are estimates of
/// the exact index's, and rows whose estimates tie come in the order the
/// turbovec kernel returns them. Immutable: queries take `&self`, so one
/// index serves many threads.
pub struct QuantizedIndex<M> {
    space: Space,
    metric: Metric,
    config: TurboQuant,
    codes: TurboQuantIndex,
    meta: Vec<M>,
}

fn refused(detail: impl std::fmt::Display) -> IndexError {
    IndexError::Quantized(detail.to_string())
}

impl<M> QuantizedIndex<M> {
    /// Codes `rows` (`bpr` blocks a row, each prepared and checked by the
    /// builder) under `config`.
    pub(crate) fn from_rows(
        space: Space,
        metric: Metric,
        bpr: usize,
        rows: &[Block],
        meta: Vec<M>,
        config: TurboQuant,
    ) -> Result<QuantizedIndex<M>, IndexError> {
        // The fields are public, so a width can reach here without `new`.
        TurboQuant::new(config.bits, config.calibrate)?;
        let dim = space.dim as usize;
        let mut codes = TurboQuantIndex::new(dim, usize::from(config.bits)).map_err(refused)?;
        let flat: Vec<f32> = rows.chunks_exact(bpr).flat_map(|r| rows::flat(r)[..dim].iter().copied()).collect();
        let n = meta.len();
        if config.calibrate && n > 0 {
            let stride = n.div_ceil(CALIBRATION_ROWS);
            let sample: Vec<f32> = flat.chunks_exact(dim).step_by(stride).flatten().copied().collect();
            codes.calibrate(&sample).map_err(refused)?;
        }
        codes.add_2d(&flat, dim).map_err(refused)?;
        codes.prepare();
        Ok(QuantizedIndex { space, metric, config, codes, meta })
    }

    pub fn space(&self) -> &Space {
        &self.space
    }

    pub fn metric(&self) -> Metric {
        self.metric
    }

    pub fn config(&self) -> TurboQuant {
        self.config
    }

    pub fn len(&self) -> usize {
        self.meta.len()
    }

    pub fn is_empty(&self) -> bool {
        self.meta.is_empty()
    }

    /// The metadata of `row`, as it was added.
    pub fn meta(&self, row: u32) -> &M {
        &self.meta[row as usize]
    }

    /// The index in turbovec's `.tv` format, which pins the codes: the
    /// same rows and settings give the same bytes.
    pub fn to_bytes(&self) -> Vec<u8> {
        self.codes.to_bytes()
    }

    /// The estimated score of `query`, a vector of `space`, against `row`:
    /// the value a [`Hit`] for that row carries. Panics if `row` is not
    /// below [`QuantizedIndex::len`]. Each call is a masked search over
    /// the whole index, so it costs a scan, not one row's work.
    pub fn score(&self, space: &Space, query: &[f32], row: u32) -> Result<f32, IndexError> {
        assert!((row as usize) < self.len(), "row {row} of {}", self.len());
        let q = self.prepare(space, query, 1)?;
        let mut mask = vec![false; self.len()];
        mask[row as usize] = true;
        let found = self.codes.try_search_with_mask(&q, 1, Some(&mask)).map_err(refused)?;
        Ok(self.finish(found.scores[0]))
    }

    /// The `k` rows whose estimated scores against `query`, a vector of
    /// `space`, are highest, best first. Fewer than `k` when the index
    /// holds fewer rows. From 32,768 rows turbovec searches 2- and 4-bit
    /// indexes in stages (a sign-bit shortlist, then full scores for the
    /// shortlist only), so a row whose estimate would rank in the top `k`
    /// can be missed; the scores of the rows returned are still exact
    /// estimates.
    pub fn top_k(&self, space: &Space, query: &[f32], k: usize) -> Result<Vec<Hit>, IndexError> {
        if query.len() != self.space.dim as usize {
            return Err(IndexError::Dimension { index: self.space.dim, other: query.len() });
        }
        Ok(self.top_k_batch(space, query, k)?.pop().unwrap_or_default())
    }

    /// [`QuantizedIndex::top_k`] for each query of `queries`, `space.dim`
    /// values a query one after another. Any query refused refuses the
    /// batch.
    pub fn top_k_batch(&self, space: &Space, queries: &[f32], k: usize) -> Result<Vec<Vec<Hit>>, IndexError> {
        let dim = self.space.dim as usize;
        if !queries.len().is_multiple_of(dim) || queries.is_empty() {
            return Err(IndexError::Dimension { index: self.space.dim, other: queries.len() });
        }
        let nq = queries.len() / dim;
        let qs = self.prepare(space, queries, nq)?;
        if k == 0 || self.is_empty() {
            return Ok(vec![Vec::new(); nq]);
        }
        let found = self.codes.try_search_with_mask(&qs, k, None).map_err(refused)?;
        Ok((0..nq)
            .map(|q| {
                let at = q * found.k..(q + 1) * found.k;
                found.indices[at.clone()]
                    .iter()
                    .zip(&found.scores[at])
                    .map(|(&row, &s)| Hit { row: row as u32, score: self.finish(s) })
                    .collect()
            })
            .collect())
    }

    #[inline]
    fn finish(&self, s: f32) -> f32 {
        match self.metric {
            Metric::Dot => s,
            Metric::Cosine => s.clamp(-1.0, 1.0),
        }
    }

    /// Checks that `space` is the index's and returns the `nq` queries of
    /// `values` as the index scores them, one after another.
    fn prepare(&self, space: &Space, values: &[f32], nq: usize) -> Result<Vec<f32>, IndexError> {
        if space.model != self.space.model {
            return Err(IndexError::Model {
                index: Box::new(self.space.model.clone()),
                other: Box::new(space.model.clone()),
            });
        }
        if space.dim != self.space.dim {
            return Err(IndexError::Dimension { index: self.space.dim, other: space.dim as usize });
        }
        if space.normalization != self.space.normalization {
            return Err(IndexError::Normalization { index: self.space.normalization, other: space.normalization });
        }
        let dim = self.space.dim as usize;
        if values.len() != nq * dim {
            return Err(IndexError::Dimension { index: self.space.dim, other: values.len() });
        }
        let mut block = vec![Block::default(); rows::blocks(self.space.dim)];
        let mut out = Vec::with_capacity(nq * dim);
        for (i, v) in values.chunks_exact(dim).enumerate() {
            rows::prepare(v, &self.space, self.metric, Which::Query(i), &mut block)?;
            out.extend_from_slice(&rows::flat(&block)[..dim]);
        }
        Ok(out)
    }
}
