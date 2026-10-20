use super::encode::Encoder;
use super::hadamard::Rotation;
use super::levels::Grid;
use super::scan::Tables;
use super::{Layout, TurboQuant, sketch_seed};
use crate::rows::{self, Block};
use crate::topk::TopK;
use crate::{Hit, IndexError, Metric, Space, Which};

/// An index whose rows are stored as TurboQuant codes. It answers the
/// same queries as [`crate::VectorIndex`], checks the query's [`Space`]
/// the same way, and keeps the same metadata; its scores are estimates.
/// Immutable: queries take `&self`, so one index serves many threads.
pub struct QuantizedIndex<M> {
    space: Space,
    metric: Metric,
    config: TurboQuant,
    layout: Layout,
    grid: Option<Grid>,
    rotation: Rotation,
    sketch: Rotation,
    codes: Vec<u8>,
    factors: Vec<[f32; 2]>,
    meta: Vec<M>,
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
    ) -> QuantizedIndex<M> {
        let layout = Layout::new(space.dim, config, metric == Metric::Cosine);
        let grid = (layout.grid_bits > 0).then(|| Grid::new(layout.grid_bits as u8));
        let rotation = Rotation::new(layout.padded, config.seed);
        let sketch = Rotation::new(layout.padded, sketch_seed(config.seed));
        let n = meta.len();
        let mut codes = vec![0u8; n * layout.row_bytes];
        let mut factors = Vec::with_capacity(n);
        let dim = space.dim as usize;
        let mut encoder = Encoder::new(&layout, grid.as_ref(), &rotation, &sketch);
        for (row, out) in rows.chunks_exact(bpr).zip(codes.chunks_exact_mut(layout.row_bytes)) {
            factors.push(encoder.encode(&rows::flat(row)[..dim], out));
        }
        QuantizedIndex { space, metric, config, layout, grid, rotation, sketch, codes, factors, meta }
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

    /// The bytes each row stores: its packed codes and its two `f32`
    /// factors.
    pub fn bytes_per_row(&self) -> usize {
        self.layout.row_bytes + size_of::<[f32; 2]>()
    }

    /// The packed codes of `row`: the grid codes, `bits` (or `bits - 1`)
    /// a padded coordinate, little-endian, then for
    /// [`super::Estimator::InnerProduct`] one sign bit a padded
    /// coordinate.
    pub fn codes(&self, row: u32) -> &[u8] {
        let at = row as usize * self.layout.row_bytes;
        &self.codes[at..at + self.layout.row_bytes]
    }

    /// The estimated score of `query`, a vector of `space`, against `row`:
    /// the value a [`Hit`] for that row carries. Panics if `row` is not
    /// below [`QuantizedIndex::len`].
    pub fn score(&self, space: &Space, query: &[f32], row: u32) -> Result<f32, IndexError> {
        let tables = self.tables(space, query, 1)?.pop().expect("one query");
        Ok(self.row_score(&tables, row as usize))
    }

    /// The `k` rows whose estimated scores against `query`, a vector of
    /// `space`, are highest, best first; rows that tie are in insertion
    /// order. Fewer than `k` when the index holds fewer rows.
    pub fn top_k(&self, space: &Space, query: &[f32], k: usize) -> Result<Vec<Hit>, IndexError> {
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
        let all = self.tables(space, queries, queries.len() / dim)?;
        Ok(all
            .iter()
            .map(|tables| {
                let mut top = TopK::new(k.min(self.len()));
                if k > 0 {
                    for row in 0..self.len() {
                        top.offer(row as u32, self.row_score(tables, row));
                    }
                }
                top.into_sorted()
            })
            .collect())
    }

    #[inline]
    fn row_score(&self, tables: &Tables, row: usize) -> f32 {
        let at = row * self.layout.row_bytes;
        let s = tables.score(&self.layout, &self.codes[at..at + self.layout.row_bytes], self.factors[row]);
        match self.metric {
            Metric::Dot => s,
            Metric::Cosine => s.clamp(-1.0, 1.0),
        }
    }

    /// Checks that `space` is the index's and builds the tables of the
    /// `nq` queries of `values`.
    fn tables(&self, space: &Space, values: &[f32], nq: usize) -> Result<Vec<Tables>, IndexError> {
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
        let mut out = Vec::with_capacity(nq);
        for (i, v) in values.chunks_exact(dim).enumerate() {
            rows::prepare(v, &self.space, self.metric, Which::Query(i), &mut block)?;
            let q = &rows::flat(&block)[..dim];
            out.push(Tables::new(&self.layout, self.grid.as_ref(), &self.rotation, &self.sketch, q));
        }
        Ok(out)
    }
}
