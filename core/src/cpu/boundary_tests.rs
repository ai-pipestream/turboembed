//! The backend at its edges, through the function table the core calls.

use std::ffi::c_void;

use crate::backend::{TURBO_FAMILY_STATIC, turbo_backend_embed_rows, turbo_backend_model, turbo_backend_run};
use crate::backend::{turbo_backend_session_desc, turbo_backend_tensor};
use crate::status::{INVALID_ARGUMENT, UNSUPPORTED_OPTION};
use crate::{
    TURBO_DTYPE_F32, TURBO_NORMALIZE_NONE, TURBO_PLACE_HOST, TURBO_POOLING_MEAN, TURBO_PRECISION_EXACT,
    TURBO_PRECISION_FASTEST, TURBO_TASK_EMBED, turbo_buffer_desc, turbo_error,
};

use super::BACKEND;

fn error() -> turbo_error {
    let mut e: turbo_error = unsafe { std::mem::zeroed() };
    e.struct_size = size_of::<turbo_error>() as u32;
    e
}

fn message(e: &turbo_error) -> String {
    crate::backend::cstr(&e.message)
}

const VOCAB: usize = 4;
const DIM: usize = 8;

/// A static model over `table`, [VOCAB, DIM] F32, with no weights or
/// mapping, and a session of `precision` on it: the session, or the
/// refusal's status and message. The model is released before return.
fn session(table: &[f32], precision: u32) -> Result<Vec<f32>, (i32, u32, String)> {
    let none = turbo_backend_tensor {
        name: std::ptr::null(),
        data: std::ptr::null(),
        shape: [0; 2],
        ndim: 0,
        dtype: 0,
        bytes: 0,
    };
    let tensors = [
        turbo_backend_tensor {
            data: table.as_ptr() as *const c_void,
            shape: [VOCAB as u64, DIM as u64],
            ndim: 2,
            dtype: TURBO_DTYPE_F32,
            bytes: (table.len() * 4) as u64,
            ..none
        },
        none,
        none,
    ];
    let desc = turbo_backend_model {
        struct_size: size_of::<turbo_backend_model>() as u32,
        family: TURBO_FAMILY_STATIC,
        dtype: TURBO_DTYPE_F32,
        layers: 0,
        hidden: DIM as u32,
        heads: 0,
        intermediate: 0,
        vocab_size: VOCAB as u32,
        max_positions: 0,
        token_types: 1,
        layer_norm_eps: 0.0,
        tensor_count: 3,
        position_offset: 0,
        tensors: tensors.as_ptr(),
        format: crate::backend::TURBO_FORMAT_SAFETENSORS,
        graph_input: crate::backend::TURBO_INPUT_TOKEN_IDS,
        graph_output: crate::backend::TURBO_OUTPUT_HIDDEN_STATES,
        compute_dtype: 0,
        fixed_seq: 0,
        fixed_batch: 0,
        artifact: std::ptr::null(),
        artifact_bytes: 0,
        artifact2: std::ptr::null(),
        artifact2_bytes: 0,
        head: std::ptr::null(),
        head_out: 0,
        lookahead: 0,
        position_buckets: 0,
        max_relative_positions: 0,
    };
    let mut err = error();
    let mut model = std::ptr::null_mut();
    assert_eq!(unsafe { (BACKEND.model_load.unwrap())(std::ptr::null_mut(), &desc, &mut model, &mut err) }, 0);
    let sd = turbo_backend_session_desc {
        struct_size: size_of::<turbo_backend_session_desc>() as u32,
        task: TURBO_TASK_EMBED,
        max_batch: VOCAB as u32,
        max_seq: 1,
        precision,
        cpu_threads: 1,
        max_ranges: VOCAB as u32,
        reserved: 0,
    };
    let (mut dtype, mut s) = (0u32, std::ptr::null_mut());
    let open = BACKEND.session_open.unwrap();
    let rc = unsafe { open(model, &sd, std::ptr::null_mut(), &mut dtype, &mut s, &mut err) };
    let got = if rc != 0 {
        Err((rc, err.field, message(&err)))
    } else {
        // Each token alone in its own row.
        let (ids, mask): (Vec<i32>, Vec<i32>) = ((0..VOCAB as i32).collect(), vec![1; VOCAB]);
        let rows = turbo_backend_embed_rows {
            struct_size: size_of::<turbo_backend_embed_rows>() as u32,
            batch: VOCAB as u32,
            seq: 1,
            row_stride: 1,
            ids: ids.as_ptr(),
            mask: mask.as_ptr(),
            types: std::ptr::null(),
            pooling: TURBO_POOLING_MEAN,
            normalize: TURBO_NORMALIZE_NONE,
            output_dim: DIM as u32,
            head: 0,
        };
        assert_eq!(unsafe { (BACKEND.embed_write.unwrap())(s, &rows, &mut err) }, 0);
        let mut run: turbo_backend_run = unsafe { std::mem::zeroed() };
        assert_eq!(unsafe { (BACKEND.session_run.unwrap())(s, &mut run, &mut err) }, 0);
        assert_eq!(run.placement, TURBO_PLACE_HOST);
        let v = unsafe { std::slice::from_raw_parts(run.host as *const f32, VOCAB * DIM) }.to_vec();
        unsafe { (BACKEND.session_release.unwrap())(s) };
        Ok(v)
    };
    unsafe { (BACKEND.model_release.unwrap())(model) };
    got
}

/// A table whose row 1 holds a NaN and row 2 an infinity, the others
/// finite. The stored table's vectors carry them: row 1's NaN in its
/// column, row 2's infinity in its. The I8 table cannot hold either,
/// so a FASTEST session on it is refused, naming the first such row,
/// rather than summing rows that were quantized to something else.
#[test]
fn a_fastest_session_is_rejected_for_a_table_with_values_that_are_not_finite() {
    let mut table: Vec<f32> = (0..VOCAB * DIM).map(|i| (i as f32 - 13.0) / 7.0).collect();
    table[DIM + 3] = f32::NAN;
    table[2 * DIM + 5] = f32::INFINITY;
    let exact = session(&table, TURBO_PRECISION_EXACT).expect("EXACT sums the stored table");
    assert!(exact[DIM + 3].is_nan() && exact[2 * DIM + 5] == f32::INFINITY);
    assert!(exact[..DIM].iter().chain(&exact[3 * DIM..]).all(|v| v.is_finite()));
    match session(&table, TURBO_PRECISION_FASTEST) {
        Err((code, field, m)) => {
            assert_eq!((code, field), (UNSUPPORTED_OPTION, 3), "{m}");
            assert!(m.contains("row 1") && m.contains("NaN"), "{m}");
        }
        Ok(v) => panic!(
            "FASTEST opened on a table it cannot hold, and gave row 1 {:?} and row 2 {:?}",
            &v[DIM..2 * DIM],
            &v[2 * DIM..3 * DIM]
        ),
    }
    // A finite table still opens at FASTEST.
    table[DIM + 3] = 0.5;
    table[2 * DIM + 5] = -0.5;
    assert!(session(&table, TURBO_PRECISION_FASTEST).is_ok());
}

/// A buffer of no bytes is refused: the allocator is not asked for a
/// zero-size layout, which it may not be given.
#[test]
fn a_buffer_of_no_bytes_is_rejected() {
    let mut desc: turbo_buffer_desc = unsafe { std::mem::zeroed() };
    desc.placement = TURBO_PLACE_HOST;
    desc.bytes = 0;
    let mut err = error();
    let (mut out, mut host) = (std::ptr::null_mut(), std::ptr::null_mut());
    let rc = unsafe { (BACKEND.buffer_alloc.unwrap())(std::ptr::null_mut(), &desc, &mut out, &mut host, &mut err) };
    assert_eq!(rc, INVALID_ARGUMENT, "{}", message(&err));
    assert!(out.is_null() && host.is_null());
    // One byte is allocated, on the 64-byte boundary.
    desc.bytes = 1;
    let rc = unsafe { (BACKEND.buffer_alloc.unwrap())(std::ptr::null_mut(), &desc, &mut out, &mut host, &mut err) };
    assert_eq!(rc, 0, "{}", message(&err));
    assert_eq!(host as usize % 64, 0);
    unsafe { (BACKEND.buffer_release.unwrap())(out) };
}
