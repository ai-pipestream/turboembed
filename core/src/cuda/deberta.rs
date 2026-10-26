//! The tests' hooks into the cuda backend's DeBERTa-v2 attention
//! (cuda/deberta.h). Built only with `internals`.

unsafe extern "C" {
    fn turbo_cuda_use_general_attention(general: i32);
    fn turbo_cuda_last_deberta_attention() -> i32;
    fn turbo_cuda_deberta_tiled_rows(
        buckets: u32,
        maximum: u32,
        seq_cap: i32,
        head_dim: i32,
        reach: *mut i32,
        room: *mut i32,
    ) -> i32;
}

/// DeBERTa-v2 sessions made from now on run the general attention kernel
/// when `on`, else the tiled one for their head width.
pub fn use_general_attention(on: bool) {
    unsafe { turbo_cuda_use_general_attention(i32::from(on)) };
}

/// The attention kernel the last DeBERTa-v2 session made picked, "tiled" or
/// "general"; None before any.
pub fn last_attention() -> Option<&'static str> {
    match unsafe { turbo_cuda_last_deberta_attention() } {
        2 => Some("tiled"),
        1 => Some("general"),
        _ => None,
    }
}

/// The relative rows a DeBERTa-v2 model's tiled attention reaches at most
/// in sessions of max_seq tokens, and the rows it holds (None: no tiled
/// kernel at the width); the tiled kernel runs only where they fit.
pub fn tiled_rows(
    position_buckets: u32,
    max_relative_positions: u32,
    max_seq: u32,
    head_dim: u32,
) -> Option<(u32, u32)> {
    let (mut reach, mut room) = (0, 0);
    let has = unsafe {
        turbo_cuda_deberta_tiled_rows(
            position_buckets,
            max_relative_positions,
            max_seq as i32,
            head_dim as i32,
            &mut reach,
            &mut room,
        )
    };
    (has != 0).then_some((reach as u32, room as u32))
}
