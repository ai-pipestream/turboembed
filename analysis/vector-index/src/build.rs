use turbo_analysis::{Ix, Layer};
use turbo_analysis_layers::Vectors;

use crate::rows::{self, Block};
use crate::{IndexError, Metric, Space, VectorIndex, Which, kernel};

/// Collects rows and their metadata for a [`VectorIndex`]. A row the
/// builder refuses leaves it as it was.
pub struct IndexBuilder<M> {
    space: Space,
    metric: Metric,
    bpr: usize,
    rows: Vec<Block>,
    meta: Vec<M>,
}

impl<M> IndexBuilder<M> {
    /// A builder for rows of `space`, compared by `metric`.
    pub fn new(space: Space, metric: Metric) -> Result<IndexBuilder<M>, IndexError> {
        if space.dim == 0 {
            return Err(IndexError::ZeroDimension);
        }
        let bpr = rows::blocks(space.dim);
        Ok(IndexBuilder { space, metric, bpr, rows: Vec::new(), meta: Vec::new() })
    }

    /// Room for `n` more rows.
    pub fn reserve(&mut self, n: usize) {
        self.rows.reserve(n * self.bpr);
        self.meta.reserve(n);
    }

    pub fn len(&self) -> usize {
        self.meta.len()
    }

    pub fn is_empty(&self) -> bool {
        self.meta.is_empty()
    }

    /// Adds `vector` with `meta` as the next row, and returns the row's
    /// number. The values are copied.
    pub fn push(&mut self, vector: &[f32], meta: M) -> Result<u32, IndexError> {
        if vector.len() != self.space.dim as usize {
            return Err(IndexError::Dimension { index: self.space.dim, other: vector.len() });
        }
        let row = self.len();
        let id = u32::try_from(row).map_err(|_| IndexError::Full)?;
        let at = self.rows.len();
        self.rows.resize(at + self.bpr, Block::default());
        if let Err(e) = rows::prepare(vector, &self.space, self.metric, Which::Row(row), &mut self.rows[at..]) {
            self.rows.truncate(at);
            return Err(e);
        }
        self.meta.push(meta);
        Ok(id)
    }

    /// Adds every item of `vectors` that has a vector, in order, with the
    /// metadata `meta` gives for it, and returns how many were added. The
    /// items the layer lists in `empty` have no vector and are skipped; a
    /// zero row the layer does not list is refused. The layer must be of
    /// the builder's model and dimension; if any row is refused, none is
    /// added.
    pub fn push_layer<L: Layer>(
        &mut self,
        vectors: &Vectors<L>,
        mut meta: impl FnMut(Ix<L>) -> M,
    ) -> Result<usize, IndexError> {
        if vectors.model != self.space.model {
            return Err(IndexError::Model {
                index: Box::new(self.space.model.clone()),
                other: Box::new(vectors.model.clone()),
            });
        }
        if vectors.dim != self.space.dim {
            return Err(IndexError::Dimension { index: self.space.dim, other: vectors.dim as usize });
        }
        vectors.check().map_err(IndexError::Layer)?;
        let (rows_before, meta_before) = (self.rows.len(), self.meta.len());
        let mut empty = vectors.empty.iter().peekable();
        for i in 0..vectors.len() {
            let ix = Ix::<L>::new(i as u32);
            if empty.next_if_eq(&&ix).is_some() {
                continue;
            }
            if let Err(e) = self.push(vectors.row(ix), meta(ix)) {
                self.rows.truncate(rows_before);
                self.meta.truncate(meta_before);
                return Err(e);
            }
        }
        Ok(self.meta.len() - meta_before)
    }

    /// The index of the rows added, in the order they were added.
    pub fn build(self) -> VectorIndex<M> {
        VectorIndex::from_parts(self.space, self.metric, self.bpr, self.rows, self.meta, kernel::detect())
    }

    /// The index of the rows added, in the order they were added, stored
    /// as TurboQuant codes under `config`. Refused when the dimension is
    /// not a positive multiple of 8 or is past 16384.
    #[cfg(feature = "turboquant")]
    pub fn build_quantized(self, config: crate::TurboQuant) -> Result<crate::QuantizedIndex<M>, IndexError> {
        crate::QuantizedIndex::from_rows(self.space, self.metric, self.bpr, &self.rows, self.meta, config)
    }
}
