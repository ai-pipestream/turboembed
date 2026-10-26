use std::ffi::CString;

use crate::backend::{
    TURBO_BERT_EMBEDDING_TENSORS, TURBO_INPUT_EMBEDDINGS, TURBO_INPUT_TOKEN_IDS, turbo_backend_model,
    turbo_backend_tensor,
};
use crate::npu::ze;
use crate::status::UNSUPPORTED_OPTION;
use crate::{TURBO_DTYPE_F16, TURBO_DTYPE_F32, TURBO_STAGE_DEVICE, TURBO_STAGE_HOST};

use super::accept::*;
use super::args::*;
use super::buffers::*;
use super::host::*;
use super::run::*;
use super::session::*;
use super::*;

#[test]
fn halves_read_back_as_the_f32_they_name() {
    assert_eq!(half_to_f32(0x0000), 0.0);
    assert_eq!(half_to_f32(0x8000), -0.0);
    assert_eq!(half_to_f32(0x3c00), 1.0);
    assert_eq!(half_to_f32(0xbc00), -1.0);
    assert_eq!(half_to_f32(0x4000), 2.0);
    assert_eq!(half_to_f32(0x3555), 0.333_251_95);
    assert_eq!(half_to_f32(0x7bff), 65504.0, "the largest finite half");
    assert_eq!(half_to_f32(0x0001), 5.960_464_5e-8, "the smallest subnormal");
    assert_eq!(half_to_f32(0x03ff), 6.097_555e-5, "the largest subnormal");
    assert_eq!(half_to_f32(0x7c00), f32::INFINITY);
    assert_eq!(half_to_f32(0xfc00), f32::NEG_INFINITY);
    assert!(half_to_f32(0x7e00).is_nan());
}

fn ok_dtype(manifest: u32, precision: u32) -> u32 {
    match session_dtype(manifest, precision) {
        Ok(d) => d,
        Err(e) => panic!("{}", e.message),
    }
}

#[test]
fn widen_f16_is_half_to_f32_for_every_half() {
    let all: Vec<u16> = (0..=u16::MAX).collect();
    // An odd length runs the tail after the eight-wide steps.
    let src = &all[..all.len() - 3];
    let mut wide = vec![0.0f32; src.len()];
    widen_f16(src, &mut wide);
    for (&h, &w) in src.iter().zip(&wide) {
        let one = half_to_f32(h);
        if one.is_nan() {
            assert!(w.is_nan(), "{h:#06x}");
        } else {
            assert_eq!(w.to_bits(), one.to_bits(), "{h:#06x}");
        }
    }
}

pub(super) fn err_msg(r: Res<()>) -> String {
    match r {
        Ok(()) => panic!("expected a refusal"),
        Err(e) => e.message,
    }
}

pub(super) fn ok_msg<T>(r: Result<T, Fail>) -> T {
    match r {
        Ok(v) => v,
        Err(e) => panic!("{}", e.message),
    }
}

pub(super) fn err_any<T>(r: Result<T, Fail>) -> String {
    match r {
        Ok(_) => panic!("expected a refusal"),
        Err(e) => e.message,
    }
}

#[test]
fn the_session_dtype_is_the_compiled_precision() {
    assert_eq!(ok_dtype(0, ze::GRAPH_ARGUMENT_PRECISION_FP16), TURBO_DTYPE_F16);
    assert_eq!(ok_dtype(0, ze::GRAPH_ARGUMENT_PRECISION_FP32), TURBO_DTYPE_F32);
    assert_eq!(ok_dtype(TURBO_DTYPE_F16, ze::GRAPH_ARGUMENT_PRECISION_FP16), TURBO_DTYPE_F16);
    assert_eq!(ok_dtype(TURBO_DTYPE_F32, ze::GRAPH_ARGUMENT_PRECISION_FP32), TURBO_DTYPE_F32);
    let e = match session_dtype(TURBO_DTYPE_F32, ze::GRAPH_ARGUMENT_PRECISION_FP16) {
        Ok(d) => panic!("kept manifest dtype {d}"),
        Err(e) => e.message,
    };
    assert!(e.contains("DTYPE_F16") && e.contains("DTYPE_F32"), "{e}");
    let e = match session_dtype(TURBO_DTYPE_F16, ze::GRAPH_ARGUMENT_PRECISION_FP32) {
        Ok(d) => panic!("kept manifest dtype {d}"),
        Err(e) => e.message,
    };
    assert!(e.contains("computes in DTYPE_F32"), "{e}");
    let e = match session_dtype(TURBO_DTYPE_F16, ze::GRAPH_ARGUMENT_PRECISION_BF16) {
        Ok(_) => panic!("BF16 became a session dtype"),
        Err(e) => e.message,
    };
    assert!(e.contains("FP16 and FP32"), "{e}");
}

#[test]
fn the_device_layout_is_the_packed_one_the_flags_named() {
    assert!(expect_packed_layout("input_ids", 2, ze::GRAPH_ARGUMENT_LAYOUT_NC).is_ok());
    assert!(expect_packed_layout("last_hidden_state", 3, ze::GRAPH_ARGUMENT_LAYOUT_CHW).is_ok());
    let e = err_msg(expect_packed_layout("input_ids", 2, 0xC8));
    assert!(e.contains("BLOCKED") && e.contains("NC"), "{e}");
    let e = err_msg(expect_packed_layout("attention_mask", 2, 0));
    assert!(e.contains("ANY"), "{e}");
    assert!(expect_packed_layout("input_ids", 4, ze::GRAPH_ARGUMENT_LAYOUT_NC).is_err());
    assert!(expect_packed_layout("attn_bias", 4, ze::GRAPH_ARGUMENT_LAYOUT_NCHW).is_ok());
    let e = err_msg(expect_packed_layout("attn_bias", 4, ze::GRAPH_ARGUMENT_LAYOUT_NC));
    assert!(e.contains("NCHW"), "{e}");
}

pub(super) fn manifest(graph_input: u32, hidden: u32, heads: u32, seq: u32, batch: u32) -> turbo_backend_model {
    turbo_backend_model {
        struct_size: 0,
        family: 0,
        dtype: 0,
        layers: 0,
        hidden,
        heads,
        intermediate: 0,
        vocab_size: 0,
        max_positions: 0,
        token_types: 0,
        layer_norm_eps: 0.0,
        tensor_count: 0,
        position_offset: 0,
        tensors: std::ptr::null(),
        format: 0,
        graph_input,
        graph_output: 0,
        compute_dtype: 0,
        fixed_seq: seq,
        fixed_batch: batch,
        artifact: std::ptr::null(),
        artifact_bytes: 0,
        artifact2: std::ptr::null(),
        artifact2_bytes: 0,
        head: std::ptr::null(),
        head_out: 0,
        lookahead: 0,
        position_buckets: 0,
        max_relative_positions: 0,
    }
}

fn compiled(name: &str, precision: u32, dims: &[u32], layout: u32) -> Compiled {
    let mut d = [0u32; 5];
    d[..dims.len()].copy_from_slice(dims);
    let elem = if precision == ze::GRAPH_ARGUMENT_PRECISION_FP16 { 2 } else { 4 };
    Compiled { arg: Arg { index: 0, name: name.into(), precision, elem }, dims: d, rank: dims.len() as u32, layout }
}

#[test]
fn the_host_gathers_word_rows_and_a_mask_bias() {
    // Four word rows of hidden 2. Written length 3 of a frame of 4:
    // the last position repeats the first id, and keys the mask drops
    // are MASKED for every head and every query.
    let table = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    let mut rows = vec![0.0; 8];
    let mut bias = vec![0.0; 2 * 16];
    ok_msg(gather_row(
        &Gather {
            table: &table,
            vocab: 4,
            hidden: 2,
            heads: 2,
            model_seq: 4,
            written: 3,
            ids: &[2, 0, 1],
            mask: &[1, 0, 1],
        },
        &mut rows,
        &mut bias,
    ));
    assert_eq!(rows, [5.0, 6.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    for head in 0..2 {
        for q in 0..4 {
            let base = (head * 4 + q) * 4;
            assert_eq!(&bias[base..base + 4], &[0.0, MASKED, 0.0, MASKED], "head {head} query {q}");
        }
    }
}

#[test]
fn a_token_id_outside_the_table_is_refused() {
    let table = [1.0, 2.0];
    let mut rows = vec![0.0; 2];
    let mut bias = vec![0.0; 1];
    let e = err_any(gather_row(
        &Gather { table: &table, vocab: 1, hidden: 2, heads: 1, model_seq: 1, written: 1, ids: &[3], mask: &[1] },
        &mut rows,
        &mut bias,
    ));
    assert!(e.contains("token id 3") && e.contains("1 rows"), "{e}");
}

#[test]
fn halves_round_trip_the_values_the_gather_writes() {
    for v in [0.0, -0.0, 1.0, -1.0, 2.0, MASKED] {
        assert_eq!(half_to_f32(f32_to_half(v)).to_bits(), v.to_bits());
    }
    for bits in 0u16..=0xffff {
        let f = half_to_f32(bits);
        if f.is_nan() {
            assert!(half_to_f32(f32_to_half(f)).is_nan());
        } else {
            assert_eq!(f32_to_half(f), bits, "half {bits:#06x}");
        }
    }
}

#[test]
fn an_embeddings_graph_is_word_rows_and_an_attention_bias() {
    let desc = manifest(TURBO_INPUT_EMBEDDINGS, 2, 2, 4, 1);
    let inputs = vec![
        compiled("word_rows", ze::GRAPH_ARGUMENT_PRECISION_FP32, &[1, 4, 2], ze::GRAPH_ARGUMENT_LAYOUT_CHW),
        compiled("attn_bias", ze::GRAPH_ARGUMENT_PRECISION_FP16, &[1, 2, 4, 4], ze::GRAPH_ARGUMENT_LAYOUT_NCHW),
    ];
    let output =
        compiled("last_hidden_state", ze::GRAPH_ARGUMENT_PRECISION_FP16, &[1, 4, 2], ze::GRAPH_ARGUMENT_LAYOUT_CHW);
    let (rows, bias, out, batch, seq, hidden) = ok_msg(accept_embeddings(inputs, output, &desc));
    assert_eq!(
        (rows.name.as_str(), bias.name.as_str(), out.name.as_str(), batch, seq, hidden),
        ("word_rows", "attn_bias", "last_hidden_state", 1, 4, 2)
    );

    let inputs = vec![
        compiled("input_ids", ze::GRAPH_ARGUMENT_PRECISION_FP32, &[1, 4, 2], ze::GRAPH_ARGUMENT_LAYOUT_CHW),
        compiled("attn_bias", ze::GRAPH_ARGUMENT_PRECISION_FP32, &[1, 2, 4, 4], ze::GRAPH_ARGUMENT_LAYOUT_NCHW),
    ];
    let e = err_any(accept_embeddings(
        inputs,
        compiled("h", ze::GRAPH_ARGUMENT_PRECISION_FP32, &[1, 4, 2], ze::GRAPH_ARGUMENT_LAYOUT_CHW),
        &desc,
    ));
    assert!(e.contains("input_ids") && e.contains("word_rows"), "{e}");

    let inputs = vec![
        compiled("word_rows", ze::GRAPH_ARGUMENT_PRECISION_FP32, &[1, 4, 2], ze::GRAPH_ARGUMENT_LAYOUT_CHW),
        compiled("attn_bias", ze::GRAPH_ARGUMENT_PRECISION_FP32, &[1, 3, 4, 4], ze::GRAPH_ARGUMENT_LAYOUT_NCHW),
    ];
    let e = err_any(accept_embeddings(
        inputs,
        compiled("h", ze::GRAPH_ARGUMENT_PRECISION_FP32, &[1, 4, 2], ze::GRAPH_ARGUMENT_LAYOUT_CHW),
        &desc,
    ));
    assert!(e.contains("[1, 3, 4, 4]") && e.contains("[1, 2, 4, 4]"), "{e}");

    let inputs = vec![
        compiled("word_rows", ze::GRAPH_ARGUMENT_PRECISION_FP32, &[1, 4, 2], ze::GRAPH_ARGUMENT_LAYOUT_NC),
        compiled("attn_bias", ze::GRAPH_ARGUMENT_PRECISION_FP32, &[1, 2, 4, 4], ze::GRAPH_ARGUMENT_LAYOUT_NCHW),
    ];
    let e = err_any(accept_embeddings(
        inputs,
        compiled("h", ze::GRAPH_ARGUMENT_PRECISION_FP32, &[1, 4, 2], ze::GRAPH_ARGUMENT_LAYOUT_CHW),
        &desc,
    ));
    assert!(e.contains("CHW"), "{e}");
}

#[test]
fn a_token_id_graph_still_refuses_an_embeddings_input_name() {
    let desc = manifest(TURBO_INPUT_TOKEN_IDS, 2, 2, 4, 1);
    let inputs = vec![
        compiled("input_ids", ze::GRAPH_ARGUMENT_PRECISION_INT64, &[1, 4], ze::GRAPH_ARGUMENT_LAYOUT_NC),
        compiled("attention_mask", ze::GRAPH_ARGUMENT_PRECISION_INT32, &[1, 4], ze::GRAPH_ARGUMENT_LAYOUT_NC),
    ];
    let (got, out, batch, seq, hidden) = ok_msg(accept_tokens(
        inputs,
        compiled("last_hidden_state", ze::GRAPH_ARGUMENT_PRECISION_FP16, &[1, 4, 2], ze::GRAPH_ARGUMENT_LAYOUT_CHW),
        &desc,
    ));
    assert!(matches!(got, GraphInputs::Tokens { types: None, .. }));
    assert_eq!((batch, seq, hidden, out.name.as_str()), (1, 4, 2, "last_hidden_state"));

    let inputs = vec![
        compiled("word_rows", ze::GRAPH_ARGUMENT_PRECISION_INT64, &[1, 4], ze::GRAPH_ARGUMENT_LAYOUT_NC),
        compiled("attention_mask", ze::GRAPH_ARGUMENT_PRECISION_INT64, &[1, 4], ze::GRAPH_ARGUMENT_LAYOUT_NC),
    ];
    let e = err_any(accept_tokens(
        inputs,
        compiled("h", ze::GRAPH_ARGUMENT_PRECISION_FP32, &[1, 4, 2], ze::GRAPH_ARGUMENT_LAYOUT_CHW),
        &desc,
    ));
    assert!(e.contains("word_rows"), "{e}");
}

#[test]
fn a_nonzero_token_type_on_an_embeddings_graph_names_the_row() {
    let e = type_error(1, 4, 2, "an INPUT_EMBEDDINGS graph computes token type 0 only");
    assert_eq!(e.code, UNSUPPORTED_OPTION);
    assert!(
        e.message.contains("token type 2") && e.message.contains("row 1") && e.message.contains("position 4"),
        "{}",
        e.message
    );
    assert!(
        type_refusal(&GraphInputs::Embeddings {
            rows: Arg { index: 0, name: String::new(), precision: 0, elem: 4 },
            bias: Arg { index: 1, name: String::new(), precision: 0, elem: 4 },
            heads: 1,
            table: std::ptr::null(),
            vocab: 1,
        })
        .unwrap()
        .contains("INPUT_EMBEDDINGS")
    );
}

#[test]
fn the_word_table_is_the_f32_host_tensor() {
    let data = [1.0f32, 2.0, 3.0, 4.0];
    let name = CString::new("embeddings.word_embeddings.weight").unwrap();
    let tensor = turbo_backend_tensor {
        name: name.as_ptr(),
        data: data.as_ptr() as *const _,
        shape: [2, 2],
        ndim: 2,
        dtype: TURBO_DTYPE_F32,
        bytes: 16,
    };
    let tensors = [tensor, tensor, tensor, tensor, tensor];
    let mut desc = manifest(TURBO_INPUT_EMBEDDINGS, 2, 1, 4, 1);
    desc.vocab_size = 2;
    desc.dtype = TURBO_DTYPE_F32;
    desc.tensor_count = TURBO_BERT_EMBEDDING_TENSORS;
    desc.tensors = tensors.as_ptr();
    let (p, vocab) = ok_msg(word_table(&desc));
    assert_eq!(vocab, 2);
    assert_eq!(unsafe { std::slice::from_raw_parts(p, 4) }, &data);

    desc.dtype = TURBO_DTYPE_F16;
    let e = err_any(word_table(&desc));
    assert!(e.contains("DTYPE_F16"), "{e}");
    desc.tensor_count = 1;
    desc.dtype = TURBO_DTYPE_F32;
    let e = err_any(word_table(&desc));
    assert!(e.contains("5 host embedding tensors") && e.contains("1 were handed"), "{e}");
    desc.tensor_count = 0;
    let e = err_any(word_table(&desc));
    assert!(e.contains("none was handed"), "{e}");
}

#[test]
fn the_mask_and_the_output_share_the_ids_frame() {
    assert!(expect_frame("attention_mask", &[1, 128], 1, 128).is_ok());
    assert!(expect_frame("token_type_ids", &[1, 128, 0, 0, 0], 1, 128).is_ok());
    assert!(expect_frame("last_hidden_state", &[1, 128, 384], 1, 128).is_ok());
    let e = err_msg(expect_frame("attention_mask", &[1, 64], 1, 128));
    assert!(e.contains("[1, 64, ...]"), "{e}");
    let e = err_msg(expect_frame("last_hidden_state", &[2, 128, 384], 1, 128));
    assert!(e.contains("the token frame is [1, 128]"), "{e}");
}

#[test]
fn two_rows_of_one_frame_are_gathered_into_the_host_buffers() {
    // Frame of three. Two rows are written, at a stride longer than
    // the compiled seq, with ids past the written length that must
    // not be read. The third row stays zero word rows and an all
    // MASKED bias, and both land in the host buffers.
    let table = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    let hidden = 2usize;
    let model_seq = 4usize;
    let heads = 2usize;
    let stride = 8usize;
    let written = 3usize;
    let frame = 3usize;
    let mut ids = vec![99i32; frame * stride];
    let mut mask = vec![1i32; frame * stride];
    ids[..3].copy_from_slice(&[2, 0, 1]);
    mask[..3].copy_from_slice(&[1, 0, 1]);
    ids[stride..stride + 3].copy_from_slice(&[1, 3, 0]);
    mask[stride..stride + 3].copy_from_slice(&[1, 1, 0]);
    let row_elems = model_seq * hidden;
    let bias_elems = heads * model_seq * model_seq;
    let mut rows = Rows {
        ids,
        mask,
        types: vec![0; frame * stride],
        acc: vec![0.0; hidden],
        token: vec![0.0; hidden],
        gathered_rows: vec![7.0; frame * row_elems],
        gathered_bias: vec![7.0; frame * bias_elems],
        ranges: Vec::new(),
        state: State {
            batch: 2,
            seq: written as u32,
            pooling: 0,
            normalize: 0,
            output_dim: hidden as u32,
            vectors: Vectors::Rows,
            written: true,
        },
    };
    let mut host_rows = vec![9.0f32; frame * row_elems];
    let mut host_bias = vec![9.0f32; frame * bias_elems];
    let rows_buf = Buffer { free: None, ptr: host_rows.as_mut_ptr().cast(), bytes: (host_rows.len() * 4) as u64 };
    let bias_buf = Buffer { free: None, ptr: host_bias.as_mut_ptr().cast(), bytes: (host_bias.len() * 4) as u64 };
    let rows_arg = Arg { index: 0, name: "word_rows".into(), precision: ze::GRAPH_ARGUMENT_PRECISION_FP32, elem: 4 };
    let bias_arg = Arg { index: 1, name: "attn_bias".into(), precision: ze::GRAPH_ARGUMENT_PRECISION_FP32, elem: 4 };
    ok_msg(write_embedding_frame(
        &EmbedFrame {
            rows_arg: &rows_arg,
            bias_arg: &bias_arg,
            rows_buf: &rows_buf,
            bias_buf: &bias_buf,
            table: table.as_ptr(),
            vocab: 4,
            heads: heads as u32,
            hidden,
            model_seq,
            written,
        },
        2,
        0,
        stride,
        &mut rows,
    ));
    assert_eq!(&host_rows[0..8], &[5.0, 6.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0], "row 0");
    assert_eq!(&host_rows[8..16], &[3.0, 4.0, 7.0, 8.0, 1.0, 2.0, 3.0, 4.0], "row 1");
    assert!(host_rows[16..].iter().all(|&v| v == 0.0), "the unused row is zero");
    let plane = model_seq * model_seq;
    for head in 0..heads {
        for q in 0..model_seq {
            let base = head * plane + q * model_seq;
            assert_eq!(&host_bias[base..base + 4], &[0.0, MASKED, 0.0, MASKED], "row 0 head {head} query {q}");
            let base = bias_elems + base;
            assert_eq!(&host_bias[base..base + 4], &[0.0, 0.0, MASKED, MASKED], "row 1 head {head} query {q}");
        }
    }
    assert!(host_bias[2 * bias_elems..].iter().all(|&v| v == MASKED), "the unused bias is dropped");
}

#[test]
fn an_embeddings_gather_reports_lookup_on_the_host() {
    let arg = |name: &str| Arg { index: 0, name: name.into(), precision: ze::GRAPH_ARGUMENT_PRECISION_FP32, elem: 4 };
    let embeddings = GraphInputs::Embeddings {
        rows: arg("word_rows"),
        bias: arg("attn_bias"),
        heads: 1,
        table: std::ptr::null(),
        vocab: 1,
    };
    assert_eq!(lookup_stage(&embeddings), TURBO_STAGE_HOST);
    let tokens = GraphInputs::Tokens { ids: arg("input_ids"), mask: arg("attention_mask"), types: None };
    assert_eq!(lookup_stage(&tokens), TURBO_STAGE_DEVICE);
}
