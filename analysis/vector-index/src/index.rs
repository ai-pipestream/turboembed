use crate::kernel::{self, Path};
use crate::rows::{self, Block};
use crate::topk::TopK;
use crate::{Hit, IndexError, Metric, Space, Which};

/// The bytes of rows a scan scores at a time, so a batch of queries reads
/// them from the core's cache rather than from memory.
const TILE_BYTES: usize = 256 * 1024;

/// The queries of a batch scored against one tile at a time.
const QUERY_GROUP: usize = 16;

/// An exact index over rows of one [`Space`]. Immutable: queries take
/// `&self`, so one index serves many threads.
pub struct VectorIndex<M> {
    space: Space,
    metric: Metric,
    bpr: usize,
    rows: Vec<Block>,
    meta: Vec<M>,
    path: Path,
}

impl<M> VectorIndex<M> {
    pub(crate) fn from_parts(
        space: Space,
        metric: Metric,
        bpr: usize,
        rows: Vec<Block>,
        meta: Vec<M>,
        path: Path,
    ) -> VectorIndex<M> {
        VectorIndex { space, metric, bpr, rows, meta, path }
    }

    pub fn space(&self) -> &Space {
        &self.space
    }

    pub fn metric(&self) -> Metric {
        self.metric
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

    /// `row` as the index scores it: as added for [`Metric::Dot`], divided
    /// by its length for [`Metric::Cosine`].
    pub fn row(&self, row: u32) -> &[f32] {
        let at = row as usize * self.bpr;
        &rows::flat(&self.rows[at..at + self.bpr])[..self.space.dim as usize]
    }

    /// The score of `query`, a vector of `space`, against `row`: the value
    /// a [`Hit`] for that row carries. Panics if `row` is not below
    /// [`VectorIndex::len`], as [`VectorIndex::meta`] and
    /// [`VectorIndex::row`] do.
    pub fn score(&self, space: &Space, query: &[f32], row: u32) -> Result<f32, IndexError> {
        let q = self.prepare(space, query, 1)?;
        let at = row as usize * self.bpr;
        let mut out = [0.0f32];
        kernel::scores(self.path, &q, &self.rows[at..at + self.bpr], self.bpr, &mut out);
        Ok(self.finish(out[0]))
    }

    /// The `k` rows that score highest against `query`, a vector of
    /// `space`, best first; rows that tie are in insertion order. Fewer
    /// than `k` when the index holds fewer rows.
    pub fn top_k(&self, space: &Space, query: &[f32], k: usize) -> Result<Vec<Hit>, IndexError> {
        Ok(self.top_k_batch(space, query, k)?.pop().unwrap_or_default())
    }

    /// [`VectorIndex::top_k`] for each query of `queries`, `space.dim`
    /// values a query one after another. The rows are read once for every
    /// sixteen queries rather than once a query. Any query refused refuses
    /// the batch.
    pub fn top_k_batch(&self, space: &Space, queries: &[f32], k: usize) -> Result<Vec<Vec<Hit>>, IndexError> {
        let dim = self.space.dim as usize;
        if !queries.len().is_multiple_of(dim) || queries.is_empty() {
            return Err(IndexError::Dimension { index: self.space.dim, other: queries.len() });
        }
        let nq = queries.len() / dim;
        let qs = self.prepare(space, queries, nq)?;
        let mut best: Vec<TopK> = (0..nq).map(|_| TopK::new(k.min(self.len()))).collect();
        if k == 0 || self.is_empty() {
            return Ok(best.into_iter().map(TopK::into_sorted).collect());
        }
        let tile = (TILE_BYTES / (self.bpr * size_of::<Block>())).clamp(16, 4096);
        let mut scores = vec![0.0f32; QUERY_GROUP.min(nq) * tile];
        for (t, rows) in self.rows.chunks(tile * self.bpr).enumerate() {
            let n = rows.len() / self.bpr;
            let first = (t * tile) as u32;
            for (g, group) in qs.chunks(QUERY_GROUP * self.bpr).enumerate() {
                let gq = group.len() / self.bpr;
                let out = &mut scores[..gq * n];
                kernel::scores(self.path, group, rows, self.bpr, out);
                for (line, top) in out.chunks_exact(n).zip(&mut best[g * QUERY_GROUP..]) {
                    for (r, &s) in line.iter().enumerate() {
                        top.offer(first + r as u32, self.finish(s));
                    }
                }
            }
        }
        Ok(best.into_iter().map(TopK::into_sorted).collect())
    }

    /// Checks that `space` is the index's and writes the `nq` queries of
    /// `values` as the kernels read them.
    fn prepare(&self, space: &Space, values: &[f32], nq: usize) -> Result<Vec<Block>, IndexError> {
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
        let mut out = vec![Block::default(); nq * self.bpr];
        for (i, (v, o)) in values.chunks_exact(dim).zip(out.chunks_exact_mut(self.bpr)).enumerate() {
            rows::prepare(v, &self.space, self.metric, Which::Query(i), o)?;
        }
        Ok(out)
    }

    #[inline]
    fn finish(&self, s: f32) -> f32 {
        match self.metric {
            Metric::Dot => s,
            Metric::Cosine => s.clamp(-1.0, 1.0),
        }
    }
}

#[cfg(test)]
impl<M> VectorIndex<M> {
    pub(crate) fn with_path(mut self, path: Path) -> VectorIndex<M> {
        self.path = path;
        self
    }
}
