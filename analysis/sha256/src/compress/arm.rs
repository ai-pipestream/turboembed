//! SHA-256 blocks on the AArch64 SHA-2 crypto extension. Each
//! `sha256h`/`sha256h2` pair runs four rounds.

use core::arch::aarch64::{
    uint32x4_t, vaddq_u32, vld1q_u8, vld1q_u32, vreinterpretq_u32_u8, vrev32q_u8, vsha256h2q_u32, vsha256hq_u32,
    vsha256su0q_u32, vsha256su1q_u32, vst1q_u32,
};

use super::K;

/// Whether this CPU has every feature [`blocks`] enables.
pub(super) fn available() -> bool {
    std::arch::is_aarch64_feature_detected!("sha2")
}

/// Runs the compression function over each of `blocks`.
///
/// # Safety
///
/// The CPU must have the `neon` and `sha2` features, as [`available`]
/// checks.
#[target_feature(enable = "neon,sha2")]
pub(super) unsafe fn blocks(state: &mut [u32; 8], blocks: &[[u8; 64]]) {
    // SAFETY: `state` is eight words, two four-word loads.
    let (mut abcd, mut efgh) = unsafe { (vld1q_u32(state.as_ptr()), vld1q_u32(state.as_ptr().add(4))) };
    for block in blocks {
        let (abcd0, efgh0) = (abcd, efgh);
        // SAFETY: `block` is 64 bytes, four 16-byte loads; big-endian
        // words, so the bytes of each 32-bit lane are reversed.
        let mut w: [uint32x4_t; 4] =
            unsafe { [0, 1, 2, 3].map(|i| vreinterpretq_u32_u8(vrev32q_u8(vld1q_u8(block.as_ptr().add(16 * i))))) };
        for i in 0..16 {
            // SAFETY: `i < 16`, so the four constants read are inside `K`.
            let t = vaddq_u32(w[0], unsafe { vld1q_u32(K.as_ptr().add(4 * i)) });
            let next = if i < 12 { vsha256su1q_u32(vsha256su0q_u32(w[0], w[1]), w[2], w[3]) } else { w[0] };
            let abcd_in = abcd;
            abcd = vsha256hq_u32(abcd, efgh, t);
            efgh = vsha256h2q_u32(efgh, abcd_in, t);
            w = [w[1], w[2], w[3], next];
        }
        abcd = vaddq_u32(abcd, abcd0);
        efgh = vaddq_u32(efgh, efgh0);
    }
    // SAFETY: the same eight words as the loads.
    unsafe {
        vst1q_u32(state.as_mut_ptr(), abcd);
        vst1q_u32(state.as_mut_ptr().add(4), efgh);
    }
}
