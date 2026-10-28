//! A DeBERTa-v2 model through the function table the core calls, against
//! transformers' last hidden states for tests/fixtures/deberta.

use std::ffi::c_void;

use crate::backend::{
    TURBO_FAMILY_DEBERTA_V2, TURBO_FORMAT_SAFETENSORS, TURBO_INPUT_TOKEN_IDS, TURBO_OUTPUT_HIDDEN_STATES,
    turbo_backend_embed_rows, turbo_backend_model, turbo_backend_run, turbo_backend_session_desc, turbo_backend_tensor,
};
use crate::safetensors::File;
use crate::session::turbo_token_range;
use crate::{
    TURBO_DTYPE_F32, TURBO_NORMALIZE_L2, TURBO_NORMALIZE_NONE, TURBO_POOLING_CLS, TURBO_POOLING_MEAN,
    TURBO_PRECISION_EXACT, TURBO_TASK_EMBED, turbo_error,
};

use super::BACKEND;

const WEIGHTS: &[u8] = include_bytes!("../../tests/fixtures/deberta/weights.safetensors");
const REFERENCE: &[u8] = include_bytes!("../../tests/fixtures/deberta/reference.json");
const HIDDEN: usize = 32;
const LAYERS: usize = 2;

#[derive(serde::Deserialize)]
struct Case {
    ids: Vec<i32>,
    mask: Vec<i32>,
    /// The embeddings' states, then each layer's, [tokens, HIDDEN] each.
    layers: Vec<Vec<f32>>,
}

/// The fixture's tensors as F32, in TURBO_DEBERTA_* order and then each
/// layer's in TURBO_BERT_* order.
fn table() -> Vec<Vec<f32>> {
    let file = File::parse("weights.safetensors", WEIGHTS).unwrap();
    let mut names: Vec<String> = [
        "embeddings.word_embeddings.weight",
        "embeddings.LayerNorm.weight",
        "embeddings.LayerNorm.bias",
        "encoder.rel_embeddings.weight",
        "encoder.LayerNorm.weight",
        "encoder.LayerNorm.bias",
    ]
    .map(String::from)
    .to_vec();
    for layer in 0..LAYERS {
        for p in ["attention.self.query_proj", "attention.self.key_proj", "attention.self.value_proj"]
            .into_iter()
            .chain(["attention.output.dense", "attention.output.LayerNorm"])
            .chain(["intermediate.dense", "output.dense", "output.LayerNorm"])
        {
            for w in ["weight", "bias"] {
                names.push(format!("encoder.layer.{layer}.{p}.{w}"));
            }
        }
    }
    names.iter().map(|n| file.tensor(n).unwrap_or_else(|| panic!("{n}")).f32s()).collect()
}

fn error() -> turbo_error {
    let mut e: turbo_error = unsafe { std::mem::zeroed() };
    e.struct_size = size_of::<turbo_error>() as u32;
    e
}

/// Run `rows` (each padded to the longest) through a model loaded from
/// the fixture: one vector per row, or per range when `ranges` has any.
fn run(cases: &[Case], pooling: u32, normalize: u32, ranges: &[turbo_token_range]) -> Vec<f32> {
    let data = table();
    let tensors: Vec<_> = data
        .iter()
        .map(|t| turbo_backend_tensor {
            name: std::ptr::null(),
            data: t.as_ptr() as *const c_void,
            shape: [t.len() as u64, 0],
            ndim: 1,
            dtype: TURBO_DTYPE_F32,
            bytes: (t.len() * 4) as u64,
        })
        .collect();
    let mut desc: turbo_backend_model = unsafe { std::mem::zeroed() };
    desc.struct_size = size_of::<turbo_backend_model>() as u32;
    desc.family = TURBO_FAMILY_DEBERTA_V2;
    desc.dtype = TURBO_DTYPE_F32;
    desc.layers = LAYERS as u32;
    desc.hidden = HIDDEN as u32;
    desc.heads = 2;
    desc.intermediate = 48;
    desc.vocab_size = 16;
    desc.max_positions = 32;
    desc.layer_norm_eps = 1e-7;
    desc.tensor_count = tensors.len() as u32;
    desc.tensors = tensors.as_ptr();
    desc.format = TURBO_FORMAT_SAFETENSORS;
    desc.graph_input = TURBO_INPUT_TOKEN_IDS;
    desc.graph_output = TURBO_OUTPUT_HIDDEN_STATES;
    desc.position_buckets = 8;
    desc.max_relative_positions = 32;
    let mut err = error();
    let mut model = std::ptr::null_mut();
    let rc = unsafe { (BACKEND.model_load.unwrap())(std::ptr::null_mut(), &desc, &mut model, &mut err) };
    assert_eq!(rc, 0, "{}", crate::backend::cstr(&err.message));
    let seq = cases.iter().map(|c| c.ids.len()).max().unwrap();
    let (mut ids, mut mask) = (vec![0; cases.len() * seq], vec![0; cases.len() * seq]);
    for (r, c) in cases.iter().enumerate() {
        ids[r * seq..][..c.ids.len()].copy_from_slice(&c.ids);
        mask[r * seq..][..c.mask.len()].copy_from_slice(&c.mask);
    }
    let sd = turbo_backend_session_desc {
        struct_size: size_of::<turbo_backend_session_desc>() as u32,
        task: TURBO_TASK_EMBED,
        max_batch: cases.len() as u32,
        max_seq: seq as u32,
        precision: TURBO_PRECISION_EXACT,
        cpu_threads: 2,
        max_ranges: ranges.len().max(1) as u32,
        reserved: 0,
    };
    let (mut dtype, mut s) = (0u32, std::ptr::null_mut());
    let rc = unsafe { (BACKEND.session_open.unwrap())(model, &sd, std::ptr::null_mut(), &mut dtype, &mut s, &mut err) };
    assert_eq!(rc, 0, "{}", crate::backend::cstr(&err.message));
    let rows = turbo_backend_embed_rows {
        struct_size: size_of::<turbo_backend_embed_rows>() as u32,
        batch: cases.len() as u32,
        seq: seq as u32,
        row_stride: seq as u32,
        ids: ids.as_ptr(),
        mask: mask.as_ptr(),
        types: std::ptr::null(),
        pooling,
        normalize,
        output_dim: HIDDEN as u32,
        head: 0,
    };
    let rc = if ranges.is_empty() {
        unsafe { (BACKEND.embed_write.unwrap())(s, &rows, &mut err) }
    } else {
        let write = BACKEND.embed_write_ranges.unwrap();
        unsafe { write(s, &rows, ranges.as_ptr(), ranges.len() as u32, &mut err) }
    };
    assert_eq!(rc, 0);
    let mut out: turbo_backend_run = unsafe { std::mem::zeroed() };
    assert_eq!(unsafe { (BACKEND.session_run.unwrap())(s, &mut out, &mut err) }, 0);
    let n = ranges.len().max(if ranges.is_empty() { cases.len() } else { 0 });
    let v = unsafe { std::slice::from_raw_parts(out.host as *const f32, n * HIDDEN) }.to_vec();
    unsafe { (BACKEND.session_release.unwrap())(s) };
    unsafe { (BACKEND.model_release.unwrap())(model) };
    v
}

fn cases() -> Vec<Case> {
    serde_json::from_slice(REFERENCE).unwrap()
}

fn largest(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len());
    a.iter().zip(b).map(|(x, y)| (x - y).abs()).fold(0.0, f32::max)
}

/// Every live token alone in a range gives transformers' last hidden
/// state for it, with the rows of other lengths in the same batch.
#[test]
fn one_token_ranges_are_the_reference_last_hidden_states() {
    let cases = cases();
    let mut ranges = Vec::new();
    let mut expected = Vec::new();
    for (r, c) in cases.iter().enumerate() {
        for p in (0..c.ids.len()).filter(|&p| c.mask[p] != 0) {
            ranges.push(turbo_token_range { row: r as u32, start: p as u32, end: p as u32 + 1 });
            expected.extend_from_slice(&c.layers[LAYERS][p * HIDDEN..][..HIDDEN]);
        }
    }
    let got = run(&cases, TURBO_POOLING_MEAN, TURBO_NORMALIZE_NONE, &ranges);
    let error = largest(&got, &expected);
    assert!(error < 2e-5, "largest difference {error}");
}

/// CLS pooling is the first token's state, and MEAN with L2 the mean of
/// the live tokens' states, normalized.
#[test]
fn pooled_rows_are_the_reference_pooled_states() {
    let cases: Vec<_> = cases().into_iter().filter(|c| c.mask[0] != 0).collect();
    let cls: Vec<f32> = cases.iter().flat_map(|c| c.layers[LAYERS][..HIDDEN].to_vec()).collect();
    let error = largest(&run(&cases, TURBO_POOLING_CLS, TURBO_NORMALIZE_NONE, &[]), &cls);
    assert!(error < 2e-5, "CLS: largest difference {error}");
    let mut mean = Vec::new();
    for c in &cases {
        let mut v = vec![0f64; HIDDEN];
        let live: Vec<_> = (0..c.ids.len()).filter(|&p| c.mask[p] != 0).collect();
        for &p in &live {
            for (d, &x) in v.iter_mut().zip(&c.layers[LAYERS][p * HIDDEN..][..HIDDEN]) {
                *d += x as f64 / live.len() as f64;
            }
        }
        let norm = v.iter().map(|x| x * x).sum::<f64>().sqrt();
        mean.extend(v.iter().map(|x| (x / norm) as f32));
    }
    let error = largest(&run(&cases, TURBO_POOLING_MEAN, TURBO_NORMALIZE_L2, &[]), &mean);
    assert!(error < 2e-5, "MEAN, L2: largest difference {error}");
}
