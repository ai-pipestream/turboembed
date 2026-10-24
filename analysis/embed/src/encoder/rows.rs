use super::*;
/// Rows of token ids for one run, laid out `stride` apart and padded
/// with the fill id under mask 0; the buffers are kept between runs.
pub struct Rows {
    ids: Vec<i32>,
    mask: Vec<i32>,
    types: Vec<i32>,
    stride: u32,
    fill: i32,
    batch: u32,
    seq: u32,
}

impl Rows {
    /// Rows of at most `stride` tokens, padded with `fill`.
    /// Panics if `stride` is zero.
    pub fn new(stride: u32, fill: i32) -> Rows {
        assert!(stride > 0, "row stride must be positive");
        Rows { ids: Vec::new(), mask: Vec::new(), types: Vec::new(), stride, fill, batch: 0, seq: 0 }
    }

    pub fn clear(&mut self) {
        self.ids.clear();
        self.mask.clear();
        self.types.clear();
        self.batch = 0;
        self.seq = 0;
    }

    pub fn len(&self) -> u32 {
        self.batch
    }

    pub fn is_empty(&self) -> bool {
        self.batch == 0
    }

    /// The longest row: the batch's seq.
    pub fn seq(&self) -> u32 {
        self.seq
    }

    /// Adds a row made of `parts` one after another, and returns its
    /// index. Panics when they hold more than `stride` tokens.
    pub fn push(&mut self, parts: &[&[i32]]) -> u32 {
        let stride = self.stride as usize;
        let len = parts.iter().try_fold(0usize, |n, p| n.checked_add(p.len())).expect("row length overflow");
        assert!(len <= stride, "a row of {len} tokens is over the stride {stride}");
        let at = self.ids.len();
        self.ids.resize(at + stride, self.fill);
        self.mask.resize(at + stride, 0);
        if !self.types.is_empty() {
            self.types.resize(at + stride, 0);
        }
        let mut p = at;
        for part in parts {
            self.ids[p..p + part.len()].copy_from_slice(part);
            p += part.len();
        }
        self.mask[at..p].fill(1);
        self.seq = self.seq.max(len as u32);
        self.batch = self.batch.checked_add(1).expect("row count overflow");
        self.batch - 1
    }

    /// Adds a pre-encoded row of `ids` and matching segment `types`, and
    /// returns its index. Panics when `ids.len() != types.len()` or when
    /// `ids.len()` exceeds `stride`.
    pub fn push_with_types(&mut self, ids: &[i32], types: &[i32]) -> u32 {
        assert_eq!(ids.len(), types.len(), "row ids and types must have equal length");
        let stride = self.stride as usize;
        let len = ids.len();
        assert!(len <= stride, "a row of {len} tokens is over the stride {stride}");
        let at = self.ids.len();
        if self.types.is_empty() && at > 0 {
            self.types.resize(at, 0);
        }
        self.ids.resize(at + stride, self.fill);
        self.mask.resize(at + stride, 0);
        self.types.resize(at + stride, 0);
        self.ids[at..at + len].copy_from_slice(ids);
        self.mask[at..at + len].fill(1);
        self.types[at..at + len].copy_from_slice(types);
        self.seq = self.seq.max(len as u32);
        self.batch = self.batch.checked_add(1).expect("row count overflow");
        self.batch - 1
    }

    /// Row `r`'s ids and mask, through its stride.
    pub fn row(&self, r: u32) -> (&[i32], &[i32]) {
        let at = r as usize * self.stride as usize;
        (&self.ids[at..at + self.stride as usize], &self.mask[at..at + self.stride as usize])
    }

    /// Builds a `turbo_token_batch` descriptor borrowing these buffers.
    pub fn batch(&self) -> turbo_token_batch {
        turbo_token_batch {
            struct_size: size_of::<turbo_token_batch>() as u32,
            batch: self.batch,
            seq: self.seq,
            row_stride: self.stride,
            ids: self.ids.as_ptr(),
            mask: self.mask.as_ptr(),
            types: if self.types.is_empty() { ptr::null() } else { self.types.as_ptr() },
        }
    }
}
