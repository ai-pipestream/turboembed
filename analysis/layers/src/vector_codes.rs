use std::marker::PhantomData;

use turbo_analysis::{Anchor, Ix, Layer, ModelRef};

/// How a `VectorCodes` layer codes each value of a vector.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CodeKind {
    /// One signed byte per value: `clamp(floor((x - min) / step), 0, 255)
    /// - 128` with `step = (max - min) / 255` (1 when that is 0), from a
    /// per-dimension `[min, max]` the caller calibrated.
    Int8,
    /// One bit per value, `x > 0`, eight to a byte, the first value in the
    /// most significant bit; a last byte the width does not fill is
    /// padded with 0 bits.
    UBinary,
}

impl CodeKind {
    /// The bytes one row of `dim` values takes.
    pub fn row_bytes(self, dim: u32) -> usize {
        match self {
            CodeKind::Int8 => dim as usize,
            CodeKind::UBinary => dim.div_ceil(8) as usize,
        }
    }
}

/// The codes of a `Vectors<L>` layer's rows, one row per item of `L`, in
/// the layout turbo's core returns for `turbo_embed_options` fields 7 and
/// 8 and sentence-transformers' `quantize_embeddings` makes. `dim` is the
/// vectors' width; each row takes `kind.row_bytes(dim)` bytes, int8 codes
/// as two's-complement bytes.
///
/// `empty` is the vectors' own list: those items have no vector, and
/// their codes are the codes of a zero row, not to be compared.
pub struct VectorCodes<L> {
    pub kind: CodeKind,
    pub dim: u32,
    pub data: Vec<u8>,
    pub empty: Vec<Ix<L>>,
    pub model: ModelRef,
    pub _of: PhantomData<fn() -> L>,
}

impl<L: Layer> VectorCodes<L> {
    pub fn new(kind: CodeKind, dim: u32, data: Vec<u8>, empty: Vec<Ix<L>>, model: ModelRef) -> VectorCodes<L> {
        VectorCodes { kind, dim, data, empty, model, _of: PhantomData }
    }

    pub fn row(&self, i: Ix<L>) -> &[u8] {
        let b = self.kind.row_bytes(self.dim);
        &self.data[i.get() * b..(i.get() + 1) * b]
    }

    /// Whether item `i` has no vector. A binary search over `empty`.
    pub fn is_empty(&self, i: Ix<L>) -> bool {
        self.empty.binary_search(&i).is_ok()
    }
}

impl<L: Layer> Layer for VectorCodes<L> {
    const ID: &'static str = "turbo:vector-codes";
    const OF: Option<&'static turbo_analysis::LayerName> = Some(L::NAME);
    fn anchor() -> Anchor {
        Anchor::aligned::<L>()
    }
    fn len(&self) -> usize {
        if self.dim == 0 { 0 } else { self.data.len() / self.kind.row_bytes(self.dim) }
    }
    fn check(&self) -> Result<(), String> {
        if self.dim == 0 || !self.data.len().is_multiple_of(self.kind.row_bytes(self.dim)) {
            return Err("a whole number of rows".into());
        }
        if !self.empty.windows(2).all(|w| w[0] < w[1]) {
            return Err("the items with no vector in order, each once".into());
        }
        if self.empty.last().is_some_and(|i| i.get() >= self.len()) {
            return Err("the items with no vector among the rows".into());
        }
        Ok(())
    }
}
