//! A range write with taps (turbo_backend.h, `taps`): each range pooled
//! from each of the last T layers in the one pass, the run's output
//! [T, ranges, output_dim], block k the states of layer layers - T + k,
//! the last block the final layer's, as the plain range write pools it.

use super::session::Session;
use crate::levelzero::gpu::{Arg, Queue, Res};

impl Session {
    /// Whether layer `l`'s states go to x in F32 where the run keeps them
    /// in F16 otherwise (FASTEST): the last layer's, for the pooling, and
    /// each layer a tap reads.
    pub(super) fn states_in_f32(&self, l: u32) -> bool {
        l + self.taps >= self.model().desc.layers
    }

    /// Whether the run pools a tap from layer `l` before the next layer
    /// runs: a tapped layer that is not the last, which pool_out pools.
    pub(super) fn tapped(&self, l: u32) -> bool {
        let layers = self.model().desc.layers;
        self.taps > 1 && l + 1 < layers && l + self.taps >= layers
    }

    /// Where block `k` of the output starts.
    pub(super) fn block(&self, k: u32) -> u64 {
        self.output.ptr as u64 + k as u64 * self.ranges as u64 * self.output_dim as u64 * 4
    }

    /// The ranges pooled from layer `l`'s F32 states at `x` into their
    /// block, unnormalized: pool_out normalizes every block at once.
    pub(super) fn tap(&self, q: &mut Queue, l: u32, x: u64) -> Res<()> {
        let k = l + self.taps - self.model().desc.layers;
        self.pool_ranges_into(q, x, self.output_dim, self.block(k))
    }

    /// The mean over each range of the states at `x`, `dim` values of
    /// each, into `to`.
    pub(super) fn pool_ranges_into(&self, q: &mut Queue, x: u64, dim: u32, to: u64) -> Res<()> {
        use Arg::*;
        let h = self.model().desc.hidden;
        let args = [
            Ptr(x),
            Ptr(self.packed_mask),
            Ptr(self.rows),
            Ptr(self.ranges_at() as u64),
            I32(h as i32),
            I32(dim as i32),
            Ptr(to),
        ];
        self.kernels.pool_ranges.launch(self.ctx(), q, "range pooling", &args, [self.ranges, dim.div_ceil(16), 1])
    }
}
