//! What the backend refuses at its boundary: a word table whose rank or
//! size the core did not check, and rows that do not fit the session.

use std::ffi::CString;
use std::sync::Mutex;

use crate::backend::{
    TURBO_BERT_EMBEDDING_TENSORS, TURBO_INPUT_EMBEDDINGS, turbo_backend_embed_rows, turbo_backend_tensor,
};
use crate::npu::ze;
use crate::status::{CAPACITY, INVALID_SHAPE, RUNTIME};
use crate::{TURBO_DTYPE_F32, TURBO_POOLING_MEAN};

use super::args::*;
use super::buffers::*;
use super::model::*;
use super::session::*;
use super::write::*;

/// The word table's message for `tensor`, under a manifest of
/// `vocab` rows of `hidden`.
fn word_table_refusal(tensor: turbo_backend_tensor, vocab: u32, hidden: u32) -> (i32, String) {
    let tensors = [tensor; TURBO_BERT_EMBEDDING_TENSORS as usize];
    let mut desc = super::tests::manifest(TURBO_INPUT_EMBEDDINGS, hidden, 1, 4, 1);
    desc.vocab_size = vocab;
    desc.dtype = TURBO_DTYPE_F32;
    desc.tensor_count = TURBO_BERT_EMBEDDING_TENSORS;
    desc.tensors = tensors.as_ptr();
    match word_table(&desc) {
        Ok(_) => panic!("the word table was taken"),
        Err(e) => (e.code, e.message),
    }
}

#[test]
fn a_word_table_of_a_rank_past_two_is_refused_with_its_shape() {
    let data = [0f32; 4];
    let name = CString::new("embeddings.word_embeddings.weight").unwrap();
    let tensor = turbo_backend_tensor {
        name: name.as_ptr(),
        data: data.as_ptr().cast(),
        shape: [2, 2],
        ndim: 3,
        dtype: TURBO_DTYPE_F32,
        bytes: 16,
    };
    let (code, message) = word_table_refusal(tensor, 2, 2);
    assert_eq!(code, RUNTIME, "{message}");
    assert!(message.contains("is [2, 2] of rank 3"), "{message}");
}

#[test]
fn a_word_table_whose_bytes_pass_u64_is_refused() {
    // u32::MAX rows of u32::MAX: the elements fit u64, four bytes each
    // do not. The data is never read.
    let data = [0f32; 1];
    let name = CString::new("embeddings.word_embeddings.weight").unwrap();
    let tensor = turbo_backend_tensor {
        name: name.as_ptr(),
        data: data.as_ptr().cast(),
        shape: [u32::MAX as u64, u32::MAX as u64],
        ndim: 2,
        dtype: TURBO_DTYPE_F32,
        bytes: 4,
    };
    let (code, message) = word_table_refusal(tensor, u32::MAX, u32::MAX);
    assert_eq!(code, RUNTIME, "{message}");
    assert!(message.contains("is 4 bytes") && message.contains("73786976260478468100 bytes"), "{message}");
}

/// A buffer of host memory the test owns: Drop calls no driver.
fn host(v: &mut [i64]) -> Buffer {
    Buffer { free: None, ptr: v.as_mut_ptr().cast(), bytes: (v.len() * 8) as u64 }
}

/// A token-id model of frames of 2 rows of 4 tokens, hidden 2, with no
/// context: keep_rows reads the model's inputs alone.
fn token_model() -> Model {
    let arg =
        |index, name: &str| Arg { index, name: name.into(), precision: ze::GRAPH_ARGUMENT_PRECISION_INT64, elem: 8 };
    Model {
        ctx: std::ptr::null(),
        graph: std::ptr::null_mut(),
        run: Mutex::new(()),
        inputs: GraphInputs::Tokens { ids: arg(0, "input_ids"), mask: arg(1, "attention_mask"), types: None },
        output: Arg {
            index: 2,
            name: "last_hidden_state".into(),
            precision: ze::GRAPH_ARGUMENT_PRECISION_FP32,
            elem: 4,
        },
        frame_batch: 2,
        seq: 4,
        hidden: 2,
        compute_dtype: TURBO_DTYPE_F32,
        graph_format: "NATIVE",
    }
}

/// Rows of a session made for `max_batch` rows of `max_seq`.
fn rows_for(max_batch: usize, max_seq: usize) -> Rows {
    Rows {
        ids: vec![0; max_batch * max_seq],
        mask: vec![0; max_batch * max_seq],
        types: vec![0; max_batch * max_seq],
        acc: vec![0.0; 2],
        token: vec![0.0; 2],
        gathered_rows: Vec::new(),
        gathered_bias: Vec::new(),
        ranges: Vec::new(),
        state: State {
            batch: 0,
            seq: 0,
            pooling: 0,
            normalize: 0,
            output_dim: 0,
            vectors: Vectors::Rows,
            written: false,
        },
    }
}

/// keep_rows of `batch` rows of `seq` at `stride`, every token live, on
/// a session made for 2 rows of 4.
fn keep(batch: u32, seq: u32, stride: u32) -> Result<(), (i32, String)> {
    let model = token_model();
    let mut memory = [[0i64; 8]; 4];
    let [ids, mask, out, result] = &mut memory;
    let session = Session {
        model: &model,
        max_seq: 4,
        max_ranges: 2,
        frames: Frames::Tokens { ids: host(ids), mask: host(mask), types: None },
        out_buf: host(out),
        result: Box::new(host(result)),
        rows: Mutex::new(rows_for(2, 4)),
    };
    let n = (batch as usize).saturating_sub(1) * stride as usize + seq as usize;
    let (ids, live) = (vec![1i32; n], vec![1i32; n]);
    let r = turbo_backend_embed_rows {
        struct_size: size_of::<turbo_backend_embed_rows>() as u32,
        batch,
        seq,
        row_stride: stride,
        ids: ids.as_ptr(),
        mask: live.as_ptr(),
        types: std::ptr::null(),
        pooling: TURBO_POOLING_MEAN,
        normalize: 0,
        output_dim: 2,
        reserved: 0,
    };
    let mut rows = rows_for(2, 4);
    unsafe { keep_rows(&session, &r, &mut rows) }.map_err(|e| (e.code, e.message))?;
    assert!(rows.state.written);
    Ok(())
}

#[test]
fn rows_past_the_session_are_refused_before_a_row_is_copied() {
    keep(2, 4, 4).unwrap();
    let (code, message) = keep(3, 4, 4).unwrap_err();
    assert_eq!(code, CAPACITY, "{message}");
    assert!(message.contains("3 rows of 4 tokens; the session holds 2 rows of 4"), "{message}");
    let (code, message) = keep(1, 5, 5).unwrap_err();
    assert_eq!(code, CAPACITY, "{message}");
    assert!(message.contains("1 rows of 5 tokens"), "{message}");
    let (code, message) = keep(2, 4, 3).unwrap_err();
    assert_eq!(code, INVALID_SHAPE, "{message}");
    assert!(message.contains("row_stride 3 is under seq 4"), "{message}");
}
