//! Linear token-projection heads (`manifest.projection` and
//! `turbo::head::ProjectionHead`, `#709`): manifest and loader rules 2 and 8,
//! bias-free (ColBERT-style) and biased linear projection, optional per-row
//! L2 normalization, F16/BF16 weight widening, and parity against the
//! reference model's unpooled hidden states and projected vectors.

mod common;

use std::fs;
use std::path::PathBuf;

use common::*;
use serde_json::{Value, json};
use turbo::bundle::Bundle;
use turbo::head::ProjectionHead;
use turbo::safetensors::{Dtype, File};
use turbo::status::*;
use turbo::*;

fn tiny_with_projection(name: &str, extra: Vec<Tensor>, edit: impl FnOnce(&mut Value)) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("turbo-proj-{}-{name}", std::process::id()));
    let _ = fs::remove_dir_all(&dir);
    for f in ["tokenizer.json", "reference/reference.safetensors"] {
        fs::create_dir_all(dir.join(f).parent().unwrap()).unwrap();
        fs::copy(tiny_bundle().join(f), dir.join(f)).unwrap();
    }
    let src_bytes = fs::read(tiny_bundle().join("weights/model.safetensors")).unwrap();
    let n = u64::from_le_bytes(src_bytes[..8].try_into().unwrap()) as usize;
    let raw_header: serde_json::Map<String, Value> = serde_json::from_slice(&src_bytes[8..8 + n]).unwrap();
    let data = &src_bytes[8 + n..];
    let mut tensors = Vec::new();
    for (k, v) in raw_header {
        if k == "__metadata__" {
            continue;
        }
        let shape: Vec<u64> = v["shape"].as_array().unwrap().iter().map(|d| d.as_u64().unwrap()).collect();
        let offs = v["data_offsets"].as_array().unwrap();
        let (a, b) = (offs[0].as_u64().unwrap() as usize, offs[1].as_u64().unwrap() as usize);
        tensors.push(Tensor { name: k, dtype: "F32", shape, data: data[a..b].to_vec() });
    }
    tensors.extend(extra);
    let weights_bytes = safetensors_file(&tensors);
    fs::create_dir_all(dir.join("weights")).unwrap();
    fs::write(dir.join("weights/model.safetensors"), &weights_bytes).unwrap();

    let mut m: Value = serde_json::from_slice(&fs::read(tiny_bundle().join("manifest.json")).unwrap()).unwrap();
    edit(&mut m);
    let files = m["files"].as_array_mut().unwrap();
    let w_entry = files.iter_mut().find(|f| f["path"] == "weights/model.safetensors").unwrap();
    w_entry["size"] = json!(weights_bytes.len());
    w_entry["sha256"] = json!(sha256_hex(&weights_bytes));
    fs::write(dir.join("manifest.json"), serde_json::to_vec_pretty(&m).unwrap()).unwrap();
    dir
}

fn linspace_vals(start: f32, end: f32, count: usize) -> Vec<f32> {
    (0..count)
        .map(|i| {
            let t = i as f32 / (count - 1) as f32;
            start + (end - start) * t
        })
        .collect()
}

fn slice_f32(vals: &[f32]) -> Vec<u8> {
    vals.iter().flat_map(|v| v.to_le_bytes()).collect()
}

#[test]
fn unpooled_hidden_states_and_colbert_projection_match_reference() {
    let base_bundle = Bundle::open(&tiny_bundle()).unwrap();
    assert!(ProjectionHead::load_optional(&base_bundle).unwrap().is_none());

    let proj_dim = 16usize;
    let hidden = 32usize;
    let weight_vals = linspace_vals(-0.3, 0.4, proj_dim * hidden);
    let bias_vals = linspace_vals(-0.05, 0.1, proj_dim);

    // 1. Bias-free ColBERT-style linear projection (`projection_weight` only).
    let dir_nobias = tiny_with_projection(
        "colbert-nobias",
        vec![Tensor {
            name: "linear.weight".into(),
            dtype: "F32",
            shape: vec![proj_dim as u64, hidden as u64],
            data: slice_f32(&weight_vals),
        }],
        |m| {
            m["projection"] = json!({ "dim": proj_dim });
            let tn = m["artifacts"][0]["tensor_names"].as_object_mut().unwrap();
            tn.insert("projection_weight".into(), json!("linear.weight"));
        },
    );

    // 2. Biased linear projection (`projection_weight` + `projection_bias`).
    let dir_bias = tiny_with_projection(
        "colbert-bias",
        vec![
            Tensor {
                name: "linear.weight".into(),
                dtype: "F32",
                shape: vec![proj_dim as u64, hidden as u64],
                data: slice_f32(&weight_vals),
            },
            Tensor {
                name: "linear.bias".into(),
                dtype: "F32",
                shape: vec![proj_dim as u64],
                data: slice_f32(&bias_vals),
            },
        ],
        |m| {
            m["projection"] = json!({ "dim": proj_dim });
            let tn = m["artifacts"][0]["tensor_names"].as_object_mut().unwrap();
            tn.insert("projection_weight".into(), json!("linear.weight"));
            tn.insert("projection_bias".into(), json!("linear.bias"));
        },
    );

    let b_nobias = Bundle::open(&dir_nobias).unwrap();
    let head_nobias = ProjectionHead::load_optional(&b_nobias).unwrap().expect("projection present");
    assert!(!head_nobias.has_bias);
    assert_eq!((head_nobias.hidden, head_nobias.dim), (hidden, proj_dim));

    let b_bias = Bundle::open(&dir_bias).unwrap();
    let head_bias = ProjectionHead::load(&b_bias).unwrap();
    assert!(head_bias.has_bias);
    assert_eq!((head_bias.hidden, head_bias.dim), (hidden, proj_dim));

    let l = Loaded::load(&dir_nobias).unwrap();
    let mut d = session_desc(0, 0, TURBO_PRECISION_MODEL);
    d.max_ranges = 32;
    let s = Session::create(l.m, Some(&d)).unwrap();

    let ref_bytes = fs::read(dir_nobias.join("reference/reference.safetensors")).unwrap();
    let ref_st = File::parse("reference.safetensors", &ref_bytes).unwrap();
    let ref_ids = ref_st.get("ids", Dtype::I32, 2).unwrap().i32s();
    let ref_lens = ref_st.get("lengths", Dtype::I32, 1).unwrap().i32s();
    let ref_embs = ref_st.get("embeddings", Dtype::F32, 2).unwrap().f32s();
    let width = ref_st.get("ids", Dtype::I32, 2).unwrap().shape[1] as usize;

    let row1 = ref_ids[width..width + ref_lens[1] as usize].to_vec();
    let row2 = ref_ids[2 * width..2 * width + ref_lens[2] as usize].to_vec();
    let tokens = Tokens::new(&[row1.clone(), row2.clone()], 0);
    let mut ranges = Vec::new();
    for pos in 0..row1.len() as u32 {
        ranges.push(range(0, pos, pos + 1));
    }
    for pos in 0..row2.len() as u32 {
        ranges.push(range(1, pos, pos + 1));
    }
    let mut opts = embed_options();
    opts.normalize = TURBO_NORMALIZE_NONE;
    s.write_ranges(&tokens.batch(), &ranges, Some(&opts)).unwrap();
    let states: Vec<f32> = s.run().unwrap().rows().into_iter().flatten().collect();

    // Verify that the mean of the unpooled token hidden states for each row
    // (after L2 normalization) matches the upstream PyTorch reference vector
    // in `reference.safetensors` within the bundle's F32 tolerance.
    let plain = PlainBert::new(&dir_nobias);
    for (r_idx, (row_ids, case_idx)) in [(&row1, 1usize), (&row2, 2usize)].into_iter().enumerate() {
        let offset = if r_idx == 0 { 0 } else { row1.len() };
        let mut mean = vec![0.0f64; hidden];
        for pos in 0..row_ids.len() {
            let tok_state = &states[(offset + pos) * hidden..(offset + pos + 1) * hidden];
            let mask = vec![1i32; row_ids.len()];
            let types = vec![0i32; row_ids.len()];
            let want_tok = plain.embed_range(row_ids, &mask, &types, pos, pos + 1, hidden, false);
            for (g, w) in tok_state.iter().zip(&want_tok) {
                assert!((*g as f64 - w).abs() < 1e-5, "row {r_idx} pos {pos}: {g} vs {w}");
            }
            for (m, &v) in mean.iter_mut().zip(tok_state) {
                *m += v as f64 / row_ids.len() as f64;
            }
        }
        let norm: f64 = mean.iter().map(|x| x * x).sum::<f64>().sqrt();
        let ref_vec = &ref_embs[case_idx * hidden..(case_idx + 1) * hidden];
        for (m, &w) in mean.iter().zip(ref_vec) {
            let m_l2 = (*m / norm) as f32;
            assert!((m_l2 - w).abs() < 1e-5, "case {case_idx} pooled vs reference: {m_l2} vs {w}");
        }
    }

    // Verify unnormalized and L2-normalized projection parity against f64 reference arithmetic.
    let total_tokens = row1.len() + row2.len();
    for (head, b_slice) in [(&head_nobias, None), (&head_bias, Some(bias_vals.as_slice()))] {
        let raw_proj = head.project(&states, false).unwrap();
        let l2_proj = head.project(&states, true).unwrap();
        assert_eq!(raw_proj.len(), total_tokens * proj_dim);
        assert_eq!(l2_proj.len(), total_tokens * proj_dim);

        for t in 0..total_tokens {
            let x = &states[t * hidden..(t + 1) * hidden];
            let got_raw = &raw_proj[t * proj_dim..(t + 1) * proj_dim];
            let got_l2 = &l2_proj[t * proj_dim..(t + 1) * proj_dim];
            let mut want = vec![0.0f64; proj_dim];
            for o in 0..proj_dim {
                let mut sum = b_slice.map_or(0.0f64, |b| b[o] as f64);
                let w_row = &weight_vals[o * hidden..(o + 1) * hidden];
                for i in 0..hidden {
                    sum += (w_row[i] as f64) * (x[i] as f64);
                }
                want[o] = sum;
                assert!((got_raw[o] as f64 - sum).abs() < 1e-5, "token {t} dim {o}: {} vs {sum}", got_raw[o]);
            }
            let norm = want.iter().map(|v| v * v).sum::<f64>().sqrt();
            let got_norm = got_l2.iter().map(|&v| (v as f64) * (v as f64)).sum::<f64>().sqrt();
            assert!((got_norm - 1.0).abs() < 1e-5, "token {t} L2 norm {got_norm}");
            for o in 0..proj_dim {
                let want_l2 = want[o] / norm;
                assert!((got_l2[o] as f64 - want_l2).abs() < 5e-5, "token {t} L2 dim {o}: {} vs {want_l2}", got_l2[o]);
            }
        }
    }

    let _ = fs::remove_dir_all(&dir_nobias);
    let _ = fs::remove_dir_all(&dir_bias);
}

#[test]
fn f16_and_bf16_projection_weights_widen_and_validate() {
    for (dtype, half_one, half_neg_half) in [("F16", 0x3c00u16, 0xb800u16), ("BF16", 0x3f80u16, 0xbf00u16)] {
        let mut f = Fixture::new(&format!("proj-{dtype}"), {
            let mut m = model_manifest();
            m["projection"] = json!({ "dim": 2 });
            let tn = m["artifacts"][0]["tensor_names"].as_object_mut().unwrap();
            tn.insert("projection_weight".into(), json!("linear.weight"));
            tn.insert("projection_bias".into(), json!("linear.bias"));
            m
        });
        let mut tensors: Vec<Tensor> = tiny_weights(0)
            .into_iter()
            .map(|t| {
                let n: usize = t.shape.iter().map(|&d| d as usize).product();
                Tensor { name: t.name, dtype, shape: t.shape, data: half_one.to_le_bytes().repeat(n) }
            })
            .collect();
        tensors.push(Tensor {
            name: "linear.weight".into(),
            dtype,
            shape: vec![2, 8],
            data: half_one.to_le_bytes().repeat(16),
        });
        tensors.push(Tensor {
            name: "linear.bias".into(),
            dtype,
            shape: vec![2],
            data: [half_one.to_le_bytes(), half_neg_half.to_le_bytes()].concat(),
        });
        f.weights("weights/model.safetensors", &tensors);
        let dir = f.write();
        let b = Bundle::open(dir).unwrap();
        let head = ProjectionHead::load(&b).unwrap();
        assert_eq!(head.linear.weight, vec![1.0f32; 16]);
        assert_eq!(head.linear.bias, vec![1.0f32, -0.5f32]);
        let out = head.project(&[0.25f32; 8], false).unwrap();
        assert_eq!(out, vec![3.0f32, 1.5f32]);
    }
}

#[test]
fn manifest_and_loader_reject_malformed_projection_configs_and_tensors() {
    // 1. Zero projection.dim.
    {
        let mut f = Fixture::model("proj-zero-dim");
        f.manifest["projection"] = json!({ "dim": 0 });
        let e = f.load().err().unwrap();
        assert!(e.is(BUNDLE_INVALID, "projection.dim: must be over 0"), "{e:?}");
    }

    // 2. Safetensors artifact missing projection_weight.
    {
        let mut f = Fixture::model("proj-no-weight-role");
        f.manifest["projection"] = json!({ "dim": 4 });
        let e = f.load().err().unwrap();
        assert!(e.is(BUNDLE_INVALID, "no projection_weight, which a projection needs"), "{e:?}");
    }

    // 3. Projection tensor role without manifest.projection.
    {
        let mut f = Fixture::model("role-without-projection");
        let tn = f.manifest["artifacts"][0]["tensor_names"].as_object_mut().unwrap();
        tn.insert("projection_weight".into(), json!("linear.weight"));
        let e = f.load().err().unwrap();
        assert!(e.is(BUNDLE_INVALID, "a projection role, and the bundle has no projection"), "{e:?}");
    }

    // 4. Rule 8: wrong projection_weight shape caught by turbo_model_load and ProjectionHead::load.
    {
        let mut f = Fixture::model("proj-wrong-shape");
        f.manifest["projection"] = json!({ "dim": 4 });
        let tn = f.manifest["artifacts"][0]["tensor_names"].as_object_mut().unwrap();
        tn.insert("projection_weight".into(), json!("linear.weight"));
        let mut tensors = tiny_weights(0);
        tensors.push(Tensor {
            name: "linear.weight".into(),
            dtype: "F32",
            shape: vec![3, 8], // 3 rows instead of 4
            data: vec![0u8; 3 * 8 * 4],
        });
        f.weights("weights/model.safetensors", &tensors);
        let e = f.load().err().unwrap();
        assert!(
            e.is(BUNDLE_INVALID, "linear.weight (projection_weight) has shape [3, 8]; the architecture implies [4, 8]"),
            "{e:?}"
        );
        let b = Bundle::open(&f.dir).unwrap();
        let pe = ProjectionHead::load(&b).unwrap_err();
        assert_eq!(pe.code, BUNDLE_INVALID);
        assert!(pe.message.contains("has shape [3, 8]; the architecture implies [4, 8]"), "{pe:?}");
    }

    // 5. Direct ProjectionHead validation and boundary checks.
    assert_eq!(ProjectionHead::new(0, 4, vec![], None).unwrap_err().code, INVALID_ARGUMENT);
    assert_eq!(ProjectionHead::new(4, 2, vec![1.0; 7], None).unwrap_err().code, INVALID_SHAPE);
    let head = ProjectionHead::new(2, 2, vec![1.0, 0.0, 0.0, 1.0], None).unwrap();
    assert_eq!(head.project(&[1.0, 2.0, 3.0], false).unwrap_err().code, INVALID_SHAPE);
    assert_eq!(head.project(&[f32::NAN, 1.0], false).unwrap_err().code, INVALID_ARGUMENT);
    let mut bad_out = [0.0f32; 1];
    assert_eq!(head.project_into(&[1.0, 2.0], false, &mut bad_out).unwrap_err().code, INVALID_SHAPE);

    // 6. Numerical edge cases: tiny finite rows normalize cleanly without
    // reciprocal overflow, and finite inputs that overflow f32 either return
    // a typed error (unnormalized) or normalize in f64 when representable (L2).
    let id1 = ProjectionHead::new(1, 1, vec![1.0], None).unwrap();
    assert_eq!(id1.project(&[1e-40], true).unwrap(), vec![1.0]);
    let tiny_2d = head.project(&[3e-40, 4e-40], true).unwrap();
    assert!((tiny_2d[0] - 0.6).abs() < 1e-6 && (tiny_2d[1] - 0.8).abs() < 1e-6, "{tiny_2d:?}");

    let huge = ProjectionHead::new(1, 1, vec![f32::MAX], None).unwrap();
    assert_eq!(huge.project(&[2.0], false).unwrap_err().code, INVALID_ARGUMENT);
    assert_eq!(huge.project(&[2.0], true).unwrap(), vec![1.0]);
}
