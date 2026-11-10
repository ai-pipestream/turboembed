//! SHA-256 blocks on the x86 SHA extensions. The state is kept as the
//! `ABEF` and `CDGH` halves `sha256rnds2` works on, and each
//! `sha256rnds2` runs two rounds.

use core::arch::x86_64::{
    __m128i, _mm_add_epi32, _mm_alignr_epi8, _mm_blend_epi16, _mm_loadu_si128, _mm_set_epi64x, _mm_sha256msg1_epu32,
    _mm_sha256msg2_epu32, _mm_sha256rnds2_epu32, _mm_shuffle_epi8, _mm_shuffle_epi32, _mm_storeu_si128,
};

use super::K;

/// Whether this CPU has every feature [`blocks`] enables.
pub(super) fn available() -> bool {
    std::arch::is_x86_feature_detected!("sha")
        && std::arch::is_x86_feature_detected!("sse4.1")
        && std::arch::is_x86_feature_detected!("ssse3")
}

/// The next four message words, `W[t..t + 4]`, from the sixteen before them.
#[inline]
#[target_feature(enable = "sha,sse2,ssse3,sse4.1")]
fn schedule(w0: __m128i, w1: __m128i, w2: __m128i, w3: __m128i) -> __m128i {
    let t = _mm_add_epi32(_mm_sha256msg1_epu32(w0, w1), _mm_alignr_epi8(w3, w2, 4));
    _mm_sha256msg2_epu32(t, w3)
}

/// Rounds `4 * i` to `4 * i + 3` over the message words `w`.
#[inline]
#[target_feature(enable = "sha,sse2,ssse3,sse4.1")]
fn rounds4(abef: &mut __m128i, cdgh: &mut __m128i, w: __m128i, i: usize) {
    // SAFETY: `i < 16`, so the four constants read are inside `K`.
    let k = unsafe { _mm_loadu_si128(K.as_ptr().add(4 * i).cast()) };
    let t = _mm_add_epi32(w, k);
    *cdgh = _mm_sha256rnds2_epu32(*cdgh, *abef, t);
    *abef = _mm_sha256rnds2_epu32(*abef, *cdgh, _mm_shuffle_epi32(t, 0x0E));
}

/// Runs the compression function over each of `blocks`.
///
/// # Safety
///
/// The CPU must have the `sha`, `sse2`, `ssse3` and `sse4.1` features,
/// as [`available`] checks.
#[target_feature(enable = "sha,sse2,ssse3,sse4.1")]
pub(super) unsafe fn blocks(state: &mut [u32; 8], blocks: &[[u8; 64]]) {
    // Big-endian words: reverse the bytes of each 32-bit lane.
    let swap = _mm_set_epi64x(0x0c0d_0e0f_0809_0a0b, 0x0405_0607_0001_0203);
    let s = state.as_mut_ptr().cast::<__m128i>();
    // SAFETY: `state` is 32 bytes, two unaligned 16-byte loads.
    let (dcba, hgfe) = unsafe { (_mm_loadu_si128(s), _mm_loadu_si128(s.add(1))) };
    let cdab = _mm_shuffle_epi32(dcba, 0xB1);
    let efgh = _mm_shuffle_epi32(hgfe, 0x1B);
    let mut abef = _mm_alignr_epi8(cdab, efgh, 8);
    let mut cdgh = _mm_blend_epi16(efgh, cdab, 0xF0);
    for block in blocks {
        let (abef0, cdgh0) = (abef, cdgh);
        let b = block.as_ptr().cast::<__m128i>();
        // SAFETY: `block` is 64 bytes, four unaligned 16-byte loads.
        let mut w = unsafe { [0, 1, 2, 3].map(|i| _mm_shuffle_epi8(_mm_loadu_si128(b.add(i)), swap)) };
        for (i, wi) in w.into_iter().enumerate() {
            rounds4(&mut abef, &mut cdgh, wi, i);
        }
        for i in 4..16 {
            let next = schedule(w[0], w[1], w[2], w[3]);
            w = [w[1], w[2], w[3], next];
            rounds4(&mut abef, &mut cdgh, next, i);
        }
        abef = _mm_add_epi32(abef, abef0);
        cdgh = _mm_add_epi32(cdgh, cdgh0);
    }
    let feba = _mm_shuffle_epi32(abef, 0x1B);
    let dchg = _mm_shuffle_epi32(cdgh, 0xB1);
    let dcba = _mm_blend_epi16(feba, dchg, 0xF0);
    let hgef = _mm_alignr_epi8(dchg, feba, 8);
    // SAFETY: the same 32 bytes as the loads.
    unsafe {
        _mm_storeu_si128(s, dcba);
        _mm_storeu_si128(s.add(1), hgef);
    }
}
