//! The levelzero backend's table called directly, as a caller that skips
//! the core's checks would: a tensor table that is not the model's layers,
//! and rows that do not fit the session, are refused before the backend
//! copies or writes anything.
//!
//! Built with the `levelzero` feature only. A test that needs a device
//! says it was skipped, and passes, when the backend lists none; with
//! TURBO_TEST_REQUIRE_LEVELZERO=1 it fails instead.

#![cfg(feature = "levelzero")]

use std::ffi::c_void;
use std::ptr;

use turbo::backend::{
    TURBO_BERT_EMBEDDING_TENSORS, TURBO_BERT_LAYER_TENSORS, TURBO_FAMILY_BERT, TURBO_FORMAT_SAFETENSORS, turbo_backend,
    turbo_backend_embed_rows, turbo_backend_model, turbo_backend_run, turbo_backend_tensor,
};
use turbo::status::{CAPACITY, INVALID_ARGUMENT, INVALID_SHAPE};
use turbo::*;

const LAYERS: u32 = 1;
const HIDDEN: u32 = 32;
const HEADS: u32 = 2;
const INTERMEDIATE: u32 = 64;
const VOCAB: u32 = 16;
const POSITIONS: u32 = 16;
const TYPES: u32 = 2;

fn backend() -> &'static turbo_backend {
    turbo::backend::linked().iter().find(|b| b.name() == "levelzero").expect("linked")
}

fn error() -> turbo_error {
    let mut e: turbo_error = unsafe { std::mem::zeroed() };
    e.struct_size = size_of::<turbo_error>() as u32;
    e
}

fn message(e: &turbo_error) -> String {
    let bytes: Vec<u8> = e.message.iter().take_while(|&&c| c != 0).map(|&c| c as u8).collect();
    String::from_utf8_lossy(&bytes).into_owned()
}

/// A context on the backend's first device, or None after saying the test
/// is skipped.
fn context(test: &str) -> Option<*mut c_void> {
    let b = backend();
    let (mut n, mut err) = (0u32, error());
    assert_eq!(unsafe { (b.device_count)(&mut n, &mut err) }, 0, "{}", message(&err));
    if n == 0 {
        let required = std::env::var("TURBO_TEST_REQUIRE_LEVELZERO").is_ok_and(|v| v == "1");
        assert!(!required, "{test}: TURBO_TEST_REQUIRE_LEVELZERO=1 and the levelzero backend lists no device");
        println!("{test}: skipped: the levelzero backend lists no device");
        return None;
    }
    let mut ctx = ptr::null_mut();
    let rc = unsafe { (b.context_create.unwrap())(0, None, ptr::null_mut(), &mut ctx, &mut err) };
    assert_eq!(rc, 0, "{}", message(&err));
    Some(ctx)
}

/// The tensors of a BERT of LAYERS layers, F32, every value 0.01 and each
/// LayerNorm's weight 1, in the order turbo_backend.h gives them.
fn weights() -> Vec<Vec<f32>> {
    let (h, i) = (HIDDEN as usize, INTERMEDIATE as usize);
    let mut t =
        vec![vec![0.01; VOCAB as usize * h], vec![0.01; POSITIONS as usize * h], vec![0.01; TYPES as usize * h]];
    t.extend([vec![1.0; h], vec![0.0; h]]);
    for _ in 0..LAYERS {
        for _ in 0..3 {
            t.extend([vec![0.01; h * h], vec![0.0; h]]);
        }
        t.extend([vec![0.01; h * h], vec![0.0; h], vec![1.0; h], vec![0.0; h]]);
        t.extend([vec![0.01; i * h], vec![0.0; i], vec![0.01; h * i], vec![0.0; h], vec![1.0; h], vec![0.0; h]]);
    }
    assert_eq!(t.len() as u32, TURBO_BERT_EMBEDDING_TENSORS + LAYERS * TURBO_BERT_LAYER_TENSORS);
    t
}

fn tensors(w: &[Vec<f32>]) -> Vec<turbo_backend_tensor> {
    w.iter()
        .map(|v| turbo_backend_tensor {
            name: ptr::null(),
            data: v.as_ptr().cast(),
            shape: [v.len() as u64, 0],
            ndim: 1,
            dtype: TURBO_DTYPE_F32,
            bytes: v.len() as u64 * 4,
        })
        .collect()
}

fn desc(t: &[turbo_backend_tensor]) -> turbo_backend_model {
    let mut d: turbo_backend_model = unsafe { std::mem::zeroed() };
    d.struct_size = size_of::<turbo_backend_model>() as u32;
    d.family = TURBO_FAMILY_BERT;
    d.dtype = TURBO_DTYPE_F32;
    (d.layers, d.hidden, d.heads, d.intermediate) = (LAYERS, HIDDEN, HEADS, INTERMEDIATE);
    (d.vocab_size, d.max_positions, d.token_types) = (VOCAB, POSITIONS, TYPES);
    d.layer_norm_eps = 1e-12;
    d.tensor_count = t.len() as u32;
    d.tensors = t.as_ptr();
    d.format = TURBO_FORMAT_SAFETENSORS;
    d
}

#[test]
fn a_tensor_table_short_of_the_layers_is_refused_at_load() {
    let Some(ctx) = context("a_tensor_table_short_of_the_layers_is_refused_at_load") else { return };
    let b = backend();
    let w = weights();
    let t = tensors(&w);
    // The embeddings alone, for a model of one layer.
    let d = desc(&t[..TURBO_BERT_EMBEDDING_TENSORS as usize]);
    let (mut model, mut err) = (ptr::null_mut(), error());
    let rc = unsafe { (b.model_load.unwrap())(ctx, &d, &mut model, &mut err) };
    if rc == 0 {
        unsafe { (b.model_release.unwrap())(model) };
    }
    unsafe { (b.context_release.unwrap())(ctx) };
    assert_eq!(rc, INVALID_ARGUMENT, "{}", message(&err));
    assert!(message(&err).contains("tensor_count 5: a BERT of 1 layers is 21 tensors"), "{}", message(&err));
}

/// `rows` rows of `seq` tokens at `stride`, every token live.
fn rows(batch: u32, seq: u32, stride: u32, ids: &[i32]) -> turbo_backend_embed_rows {
    assert!(ids.len() >= ((batch - 1) * stride + seq) as usize);
    turbo_backend_embed_rows {
        struct_size: size_of::<turbo_backend_embed_rows>() as u32,
        batch,
        seq,
        row_stride: stride,
        ids: ids.as_ptr(),
        mask: ids.as_ptr(),
        types: ptr::null(),
        pooling: TURBO_POOLING_MEAN,
        normalize: TURBO_NORMALIZE_NONE,
        output_dim: HIDDEN,
        head: 0,
    }
}

#[test]
fn rows_past_the_session_are_refused_before_the_staging_is_written() {
    let Some(ctx) = context("rows_past_the_session_are_refused_before_the_staging_is_written") else { return };
    let b = backend();
    let w = weights();
    let t = tensors(&w);
    let d = desc(&t);
    let (mut model, mut err) = (ptr::null_mut(), error());
    assert_eq!(unsafe { (b.model_load.unwrap())(ctx, &d, &mut model, &mut err) }, 0, "{}", message(&err));
    let (mut session, mut dtype) = (ptr::null_mut(), 0u32);
    let open = b.session_create.unwrap();
    let rc = unsafe { open(model, TURBO_TASK_EMBED, 2, 4, TURBO_PRECISION_EXACT, &mut dtype, &mut session, &mut err) };
    assert_eq!(rc, 0, "{}", message(&err));
    // Ids of 1, which are also a live mask.
    let ones = [1i32; 16];
    let write = |r: &turbo_backend_embed_rows| {
        let mut err = error();
        let rc = unsafe { (b.embed_write.unwrap())(session, r, &mut err) };
        (rc, message(&err))
    };
    let refused = [
        (write(&rows(3, 4, 4, &ones)), CAPACITY, "3 rows of 4 tokens: the levelzero session holds 2 rows of 4"),
        (write(&rows(1, 5, 5, &ones)), CAPACITY, "1 rows of 5 tokens"),
        (write(&rows(2, 4, 3, &ones)), INVALID_SHAPE, "row_stride 3 is under seq 4"),
    ];
    // A write that fits still runs after the refusals.
    let fits = write(&rows(2, 4, 4, &ones));
    let mut run: turbo_backend_run = unsafe { std::mem::zeroed() };
    run.struct_size = size_of::<turbo_backend_run>() as u32;
    let ran = unsafe { (b.session_run.unwrap())(session, &mut run, &mut err) };
    unsafe {
        (b.session_release.unwrap())(session);
        (b.model_release.unwrap())(model);
        (b.context_release.unwrap())(ctx);
    }
    for ((rc, text), code, says) in refused {
        assert_eq!(rc, code, "{text}");
        assert!(text.contains(says), "{text}");
    }
    assert_eq!(fits.0, 0, "{}", fits.1);
    assert_eq!(ran, 0, "{}", message(&err));
}
