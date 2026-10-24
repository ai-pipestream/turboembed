//! Loader rule 8 for the MLM head behind `embed.sparse`: turbo_model_load
//! checks the head's tensors with the encoder's (#1059), so a sparse
//! bundle whose head is missing, misshapen or stored in another dtype is
//! refused at load, not when the sparse output is first asked for.

mod common;

use common::*;
use serde_json::json;
use turbo::status::*;

fn f32_bytes(n: usize, v: f32) -> Vec<u8> {
    (0..n).flat_map(|_| v.to_le_bytes()).collect()
}

/// The MLM head tensors of the small BERT (hidden 8), the decoder weight
/// only when `tied` is false.
fn head(tied: bool) -> Vec<Tensor> {
    let (h, v) = (8u64, 30522u64);
    let mut out = vec![
        Tensor { name: "mlm.transform.weight".into(), dtype: "F32", shape: vec![h, h], data: f32_bytes(64, 0.01) },
        Tensor { name: "mlm.transform.bias".into(), dtype: "F32", shape: vec![h], data: f32_bytes(8, 0.0) },
        Tensor { name: "mlm.ln.weight".into(), dtype: "F32", shape: vec![h], data: f32_bytes(8, 1.0) },
        Tensor { name: "mlm.ln.bias".into(), dtype: "F32", shape: vec![h], data: f32_bytes(8, 0.0) },
        Tensor { name: "mlm.decoder.bias".into(), dtype: "F32", shape: vec![v], data: f32_bytes(v as usize, 0.0) },
    ];
    if !tied {
        out.push(Tensor {
            name: "mlm.decoder.weight".into(),
            dtype: "F32",
            shape: vec![v, h],
            data: f32_bytes((v * h) as usize, 0.02),
        });
    }
    out
}

/// The small BERT with `embed.sparse` declared and the MLM roles named,
/// its weights file holding the encoder and `tensors`.
fn sparse_model(name: &str, tied: bool, tensors: Vec<Tensor>) -> Fixture {
    let mut f = Fixture::model(name);
    f.manifest["embed"]["sparse"] = json!({ "pooling": "SPARSE_POOLING_MAXLOG" });
    let tn = f.manifest["artifacts"][0]["tensor_names"].as_object_mut().unwrap();
    for (role, t) in [
        ("mlm_transform_weight", "mlm.transform.weight"),
        ("mlm_transform_bias", "mlm.transform.bias"),
        ("mlm_transform_ln_weight", "mlm.ln.weight"),
        ("mlm_transform_ln_bias", "mlm.ln.bias"),
        ("mlm_decoder_bias", "mlm.decoder.bias"),
    ] {
        tn.insert(role.into(), json!(t));
    }
    if !tied {
        tn.insert("mlm_decoder_weight".into(), json!("mlm.decoder.weight"));
    }
    let mut all = tiny_weights(0);
    all.extend(tensors);
    f.weights("weights/model.safetensors", &all);
    f
}

#[test]
fn a_complete_head_loads_tied_and_untied() {
    for tied in [true, false] {
        let f = sparse_model(&format!("mlm-ok-{tied}"), tied, head(tied));
        if let Err(e) = f.load() {
            panic!("tied {tied}: {e:?}");
        }
    }
}

#[test]
fn a_missing_head_tensor_is_refused_at_load() {
    for tied in [true, false] {
        let mut t = head(tied);
        t.retain(|t| t.name != "mlm.transform.bias");
        let e = sparse_model(&format!("mlm-missing-{tied}"), tied, t).load().err().expect("loaded");
        assert!(e.is(BUNDLE_INVALID, "mlm.transform.bias"), "tied {tied}: {e:?}");
    }
}

#[test]
fn a_misshapen_head_tensor_is_refused_at_load() {
    let mut t = head(false);
    let d = t.iter_mut().find(|t| t.name == "mlm.decoder.weight").unwrap();
    d.shape = vec![30522, 4];
    d.data.truncate(30522 * 4 * 4);
    let e = sparse_model("mlm-shape", false, t).load().err().expect("loaded");
    assert!(e.is(BUNDLE_INVALID, "mlm.decoder.weight"), "{e:?}");
    assert!(e.message.contains("[30522, 8]"), "{e:?}");
}

#[test]
fn a_head_in_another_dtype_is_refused_at_load() {
    // The encoder is F32; an untied F16 decoder weight would mix dtypes in
    // one artifact, which rule 8 forbids.
    let mut t = head(false);
    let d = t.iter_mut().find(|t| t.name == "mlm.decoder.weight").unwrap();
    d.dtype = "F16";
    d.data.truncate(30522 * 8 * 2);
    let e = sparse_model("mlm-dtype", false, t).load().err().expect("loaded");
    assert!(e.is(BUNDLE_INVALID, "mlm.decoder.weight"), "{e:?}");
    assert!(e.message.contains("F16"), "{e:?}");
}
