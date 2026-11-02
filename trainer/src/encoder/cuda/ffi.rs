//! The C API of trainer/cuda/train.h, as the library built by build.rs
//! exports it: every function returns 0 or a CUDA error number.

#![allow(clippy::too_many_arguments)]

use std::ffi::{c_char, c_void};

unsafe extern "C" {
    pub fn tt_device_count(n: *mut i32) -> i32;
    pub fn tt_set_device(device: i32) -> i32;
    pub fn tt_error_string(error: i32) -> *const c_char;

    pub fn tt_malloc(bytes: usize, p: *mut *mut c_void) -> i32;
    pub fn tt_free(p: *mut c_void) -> i32;
    pub fn tt_upload(dst: *mut c_void, src: *const c_void, bytes: usize) -> i32;
    pub fn tt_download(dst: *mut c_void, src: *const c_void, bytes: usize) -> i32;
    pub fn tt_copy(dst: *mut c_void, src: *const c_void, bytes: usize) -> i32;
    pub fn tt_zero(dst: *mut c_void, bytes: usize) -> i32;
    pub fn tt_sync() -> i32;

    pub fn tt_gemm(
        a: *const f32,
        b: *const f32,
        c: *mut f32,
        m: i32,
        n: i32,
        k: i32,
        ta: i32,
        tb: i32,
        alpha: f32,
        beta: f32,
    ) -> i32;
    pub fn tt_gather(table: *const f32, ids: *const i32, n: i32, h: i32, out: *mut f32) -> i32;
    pub fn tt_scatter_add(rows: *const f32, ids: *const i32, n: i32, h: i32, table: *mut f32) -> i32;
    pub fn tt_layernorm(
        x: *const f32,
        w: *const f32,
        rows: i32,
        width: i32,
        eps: f64,
        y: *mut f32,
        mean: *mut f64,
        rstd: *mut f64,
    ) -> i32;
    pub fn tt_layernorm_back(
        x: *const f32,
        w: *const f32,
        mean: *const f64,
        rstd: *const f64,
        dy: *const f32,
        rows: i32,
        width: i32,
        dx: *mut f32,
        dw: *mut f32,
    ) -> i32;
    pub fn tt_rope(
        x: *mut f32,
        rows: i32,
        ld: i32,
        cols: i32,
        pairs: i32,
        cs: *const f32,
        sn: *const f32,
        back: i32,
    ) -> i32;
    pub fn tt_attention_probs(
        qkv: *const f32,
        n: i32,
        hidden: i32,
        heads: i32,
        window: i32,
        round: i32,
        p: *mut f32,
    ) -> i32;
    pub fn tt_attention_ctx(
        qkv: *const f32,
        p: *const f32,
        n: i32,
        hidden: i32,
        heads: i32,
        window: i32,
        ctx: *mut f32,
    ) -> i32;
    pub fn tt_attention_ds(
        qkv: *const f32,
        p: *const f32,
        dctx: *const f32,
        n: i32,
        hidden: i32,
        heads: i32,
        window: i32,
        ds: *mut f32,
    ) -> i32;
    pub fn tt_attention_dq(
        qkv: *const f32,
        ds: *const f32,
        n: i32,
        hidden: i32,
        heads: i32,
        window: i32,
        dqkv: *mut f32,
    ) -> i32;
    pub fn tt_attention_dkv(
        qkv: *const f32,
        p: *const f32,
        ds: *const f32,
        dctx: *const f32,
        n: i32,
        hidden: i32,
        heads: i32,
        window: i32,
        dqkv: *mut f32,
    ) -> i32;
    pub fn tt_geglu(pre: *const f32, rows: i32, inter: i32, act: *mut f32) -> i32;
    pub fn tt_geglu_back(pre: *const f32, dact: *const f32, rows: i32, inter: i32, dpre: *mut f32) -> i32;
    pub fn tt_dropout(x: *mut f32, count: usize, p: f64, key: u64) -> i32;
    pub fn tt_axpy(y: *mut f32, x: *const f32, count: usize, alpha: f32) -> i32;
    pub fn tt_round_bf16(x: *mut f32, count: usize) -> i32;
}
