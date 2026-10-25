//! Classification heads and label maps (`manifest.classifier` and
//! `turbo::head::Head`): manifest and loader rules 2 and 8, host-side
//! pooler (`dense` + `tanh`) and linear classifier (`dense`), numerically
//! stable softmax, and end-to-end parity against PyTorch
//! `BertForTokenClassification` and `BertForSequenceClassification`
//! source checkpoint logits on the small BERT bundle.

mod common;

use std::fs;
use std::path::PathBuf;

use common::*;
use serde_json::{Value, json};
use turbo::bundle::Bundle;
use turbo::head::{Head, Linear, softmax};
use turbo::manifest::ClassifierKind;
use turbo::safetensors::{Dtype, File};
use turbo::status::*;
use turbo::*;

/// Copy `testdata/tiny-bert-bundle` into a temporary directory, append
/// `extra` tensors to `weights/model.safetensors`, apply `edit` to
/// `manifest.json`, and update `files[]` with the new weights size and hash.
fn tiny_with_head(name: &str, extra: Vec<Tensor>, edit: impl FnOnce(&mut Value)) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("turbo-head-{}-{name}", std::process::id()));
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

fn linspace_f32(start: f32, end: f32, count: usize) -> Vec<u8> {
    (0..count)
        .flat_map(|i| {
            let t = i as f32 / (count - 1) as f32;
            (start + (end - start) * t).to_le_bytes()
        })
        .collect()
}

fn slice_f32(vals: &[f32]) -> Vec<u8> {
    vals.iter().flat_map(|v| v.to_le_bytes()).collect()
}

#[test]
fn token_classifier_matches_pytorch_bert_for_token_classification_logits_and_softmax() {
    // Same weights as BertForTokenClassification in turbo-reference:latest:
    // weight = torch.linspace(-0.25, 0.35, 3 * 32).reshape(3, 32)
    // bias = [0.1, -0.2, 0.05]
    let extra = vec![
        Tensor {
            name: "classifier.weight".into(),
            dtype: "F32",
            shape: vec![3, 32],
            data: linspace_f32(-0.25, 0.35, 96),
        },
        Tensor { name: "classifier.bias".into(), dtype: "F32", shape: vec![3], data: slice_f32(&[0.1, -0.2, 0.05]) },
    ];
    let dir = tiny_with_head("token-cls", extra, |m| {
        m["classifier"] = json!({
            "kind": "CLASSIFIER_TOKEN",
            "labels": ["O", "B-PER", "I-PER"]
        });
        let tn = m["artifacts"][0]["tensor_names"].as_object_mut().unwrap();
        tn.insert("classifier_weight".into(), json!("classifier.weight"));
        tn.insert("classifier_bias".into(), json!("classifier.bias"));
    });

    let b = Bundle::open(&dir).unwrap();
    let head = Head::load(&b).unwrap();
    assert_eq!(head.kind, ClassifierKind::Token);
    assert_eq!(head.hidden, 32);
    assert_eq!(head.labels, ["O", "B-PER", "I-PER"]);
    assert_eq!(head.num_labels(), 3);
    assert_eq!(head.label(1), Some("B-PER"));
    assert_eq!(head.label(3), None);
    assert!(head.pooler.is_none());

    let l = Loaded::load(&dir).unwrap();
    let mut d = session_desc(0, 0, TURBO_PRECISION_MODEL);
    d.max_ranges = 32;
    let s = Session::create(l.m, Some(&d)).unwrap();

    let ref_bytes = fs::read(dir.join("reference/reference.safetensors")).unwrap();
    let ref_st = File::parse("reference.safetensors", &ref_bytes).unwrap();
    let ref_ids = ref_st.get("ids", Dtype::I32, 2).unwrap().i32s();
    let ref_lens = ref_st.get("lengths", Dtype::I32, 1).unwrap().i32s();
    let width = ref_st.get("ids", Dtype::I32, 2).unwrap().shape[1] as usize;

    // Cases 1 and 2 from reference.safetensors ("The quick brown fox..." and "Café naïve RÉSUMÉ").
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

    let logits = head.logits(&states).unwrap();
    let probs = head.probabilities(&states).unwrap();

    // Exact PyTorch BertForTokenClassification outputs from turbo-reference:latest.
    let want1: &[[f32; 3]] = &[
        [0.27554902, -0.024450973, 0.22554915],
        [0.24154612, -0.058453873, 0.19154625],
        [0.10706566, -0.19293433, 0.057065655],
        [-0.04633256, -0.34633282, -0.0963331],
        [0.40385517, 0.103855, 0.353855],
        [0.33581725, 0.03581728, 0.28581733],
        [-0.08852353, -0.38852352, -0.13852341],
        [0.022376351, -0.27762386, -0.027624052],
        [-0.17707065, -0.47707057, -0.22707053],
        [0.4012613, 0.10126124, 0.35126144],
        [0.360499, 0.060498938, 0.31049895],
        [0.35369653, 0.0536965, 0.30369663],
    ];
    let want2: &[[f32; 3]] = &[
        [0.2779843, -0.022015706, 0.22798418],
        [0.071114056, -0.22888577, 0.021114353],
        [0.40183434, 0.10183419, 0.35183427],
        [0.12726459, -0.1727354, 0.07726483],
        [0.23899096, -0.061009154, 0.18899088],
    ];
    let want: Vec<f32> = want1.iter().chain(want2.iter()).flatten().copied().collect();
    assert_eq!(logits.len(), want.len());
    for (i, (&g, &w)) in logits.iter().zip(&want).enumerate() {
        assert!((g - w).abs() < 1e-5, "token logit {i}: got {g}, want {w}");
    }
    for row_probs in probs.as_chunks::<3>().0 {
        let sum: f64 = row_probs.iter().sum();
        assert!((sum - 1.0).abs() < 1e-12, "row probs sum {sum}");
        assert!(row_probs[0] > row_probs[2] && row_probs[2] > row_probs[1]);
    }

    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn sequence_classifier_with_pooler_matches_pytorch_bert_for_sequence_classification() {
    // testdata/tiny-bert-bundle/weights/model.safetensors already carries
    // pooler.dense.weight [32, 32] and pooler.dense.bias [32].
    let extra = vec![
        Tensor {
            name: "classifier.weight".into(),
            dtype: "F32",
            shape: vec![2, 32],
            data: linspace_f32(0.3, -0.2, 64),
        },
        Tensor { name: "classifier.bias".into(), dtype: "F32", shape: vec![2], data: slice_f32(&[-0.05, 0.15]) },
    ];
    let dir = tiny_with_head("seq-cls", extra, |m| {
        m["classifier"] = json!({
            "kind": "CLASSIFIER_SEQUENCE",
            "labels": ["NEG", "POS"]
        });
        let tn = m["artifacts"][0]["tensor_names"].as_object_mut().unwrap();
        tn.insert("pooler_weight".into(), json!("pooler.dense.weight"));
        tn.insert("pooler_bias".into(), json!("pooler.dense.bias"));
        tn.insert("classifier_weight".into(), json!("classifier.weight"));
        tn.insert("classifier_bias".into(), json!("classifier.bias"));
    });

    let b = Bundle::open(&dir).unwrap();
    let head = Head::load(&b).unwrap();
    assert_eq!(head.kind, ClassifierKind::Sequence);
    assert_eq!(head.labels, ["NEG", "POS"]);
    assert!(head.pooler.is_some());

    let l = Loaded::load(&dir).unwrap();
    let s = Session::create(l.m, None).unwrap();

    let ref_bytes = fs::read(dir.join("reference/reference.safetensors")).unwrap();
    let ref_st = File::parse("reference.safetensors", &ref_bytes).unwrap();
    let ref_ids = ref_st.get("ids", Dtype::I32, 2).unwrap().i32s();
    let ref_lens = ref_st.get("lengths", Dtype::I32, 1).unwrap().i32s();
    let width = ref_st.get("ids", Dtype::I32, 2).unwrap().shape[1] as usize;

    let row1 = ref_ids[width..width + ref_lens[1] as usize].to_vec();
    let row2 = ref_ids[2 * width..2 * width + ref_lens[2] as usize].to_vec();
    let tokens = Tokens::new(&[row1, row2], 0);
    let mut opts = embed_options();
    opts.pooling = TURBO_POOLING_CLS;
    opts.normalize = TURBO_NORMALIZE_NONE;
    s.write_tokens(&tokens.batch(), Some(&opts)).unwrap();
    let cls_states: Vec<f32> = s.run().unwrap().rows().into_iter().flatten().collect();

    let logits = head.logits(&cls_states).unwrap();
    let probs = head.probabilities(&cls_states).unwrap();

    // Exact PyTorch BertForSequenceClassification outputs from turbo-reference:latest.
    let want = [0.16030255f32, 0.06783607, 0.15911731, 0.06804016];
    assert_eq!(logits.len(), want.len());
    for (i, (&g, &w)) in logits.iter().zip(&want).enumerate() {
        assert!((g - w).abs() < 1e-5, "sequence logit {i}: got {g}, want {w}");
    }
    for row in probs.as_chunks::<2>().0 {
        let sum: f64 = row.iter().sum();
        assert!((sum - 1.0).abs() < 1e-12);
        assert!(row[0] > row[1]);
    }

    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn f16_and_bf16_head_weights_are_widened_to_f32() {
    for (dtype, half_one, half_neg_half) in [("F16", 0x3c00u16, 0xb800u16), ("BF16", 0x3f80u16, 0xbf00u16)] {
        let mut f = Fixture::new(&format!("head-{dtype}"), {
            let mut m = model_manifest();
            m["classifier"] = json!({
                "kind": "CLASSIFIER_SEQUENCE",
                "labels": ["A", "B"]
            });
            let tn = m["artifacts"][0]["tensor_names"].as_object_mut().unwrap();
            tn.insert("classifier_weight".into(), json!("classifier.weight"));
            tn.insert("classifier_bias".into(), json!("classifier.bias"));
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
            name: "classifier.weight".into(),
            dtype,
            shape: vec![2, 8],
            data: half_one.to_le_bytes().repeat(16),
        });
        tensors.push(Tensor {
            name: "classifier.bias".into(),
            dtype,
            shape: vec![2],
            data: [half_one.to_le_bytes(), half_neg_half.to_le_bytes()].concat(),
        });
        f.weights("weights/model.safetensors", &tensors);
        let dir = f.write();
        let b = Bundle::open(dir).unwrap();
        let head = Head::load(&b).unwrap();
        assert_eq!(head.classifier.weight(), vec![1.0f32; 16]);
        assert_eq!(head.classifier.bias(), vec![1.0f32, -0.5f32]);
        let logits = head.logits(&[0.25f32; 8]).unwrap();
        assert_eq!(logits, vec![3.0f32, 1.5f32]);
    }
}

#[test]
fn manifest_and_loader_reject_malformed_classifier_configs_and_tensors() {
    let valid_classifier = || {
        json!({
            "kind": "CLASSIFIER_TOKEN",
            "labels": ["O", "B-PER"]
        })
    };

    // 1. Empty labels list.
    {
        let mut f = Fixture::model("cls-empty-labels");
        f.manifest["classifier"] = json!({ "kind": "CLASSIFIER_TOKEN", "labels": [] });
        let e = f.load().err().unwrap();
        assert!(e.is(BUNDLE_INVALID, "classifier.labels: empty"), "{e:?}");
    }

    // 2. Empty label string.
    {
        let mut f = Fixture::model("cls-blank-label");
        f.manifest["classifier"] = json!({ "kind": "CLASSIFIER_TOKEN", "labels": ["O", ""] });
        let e = f.load().err().unwrap();
        assert!(e.is(BUNDLE_INVALID, "classifier.labels[1]: required"), "{e:?}");
    }

    // 3. Safetensors artifact missing classifier_weight.
    {
        let mut f = Fixture::model("cls-no-weight-role");
        f.manifest["classifier"] = valid_classifier();
        let e = f.load().err().unwrap();
        assert!(e.is(BUNDLE_INVALID, "no classifier_weight, which a classifier needs"), "{e:?}");
    }

    // 4. pooler_weight without pooler_bias.
    {
        let mut f = Fixture::model("cls-half-pooler");
        f.manifest["classifier"] = json!({ "kind": "CLASSIFIER_SEQUENCE", "labels": ["NEG", "POS"] });
        let tn = f.manifest["artifacts"][0]["tensor_names"].as_object_mut().unwrap();
        tn.insert("classifier_weight".into(), json!("classifier.weight"));
        tn.insert("classifier_bias".into(), json!("classifier.bias"));
        tn.insert("pooler_weight".into(), json!("pooler.dense.weight"));
        let e = f.load().err().unwrap();
        assert!(e.is(BUNDLE_INVALID, "pooler_weight and pooler_bias come together, or neither"), "{e:?}");
    }

    // 5. CLASSIFIER_TOKEN with pooler tensors.
    {
        let mut f = Fixture::model("cls-token-with-pooler");
        f.manifest["classifier"] = valid_classifier();
        let tn = f.manifest["artifacts"][0]["tensor_names"].as_object_mut().unwrap();
        tn.insert("classifier_weight".into(), json!("classifier.weight"));
        tn.insert("classifier_bias".into(), json!("classifier.bias"));
        tn.insert("pooler_weight".into(), json!("pooler.dense.weight"));
        tn.insert("pooler_bias".into(), json!("pooler.dense.bias"));
        let e = f.load().err().unwrap();
        assert!(e.is(BUNDLE_INVALID, "CLASSIFIER_TOKEN has no pooler"), "{e:?}");
    }

    // 6. Classifier tensor role without manifest.classifier.
    {
        let mut f = Fixture::model("role-without-classifier");
        let tn = f.manifest["artifacts"][0]["tensor_names"].as_object_mut().unwrap();
        tn.insert("classifier_weight".into(), json!("classifier.weight"));
        let e = f.load().err().unwrap();
        assert!(e.is(BUNDLE_INVALID, "a classifier role, and the bundle has no classifier"), "{e:?}");
    }

    // 7. Rule 8: wrong classifier_weight shape in safetensors file caught by turbo_model_load and Head::load.
    {
        let mut f = Fixture::model("cls-wrong-shape");
        f.manifest["classifier"] = valid_classifier();
        let tn = f.manifest["artifacts"][0]["tensor_names"].as_object_mut().unwrap();
        tn.insert("classifier_weight".into(), json!("classifier.weight"));
        tn.insert("classifier_bias".into(), json!("classifier.bias"));
        let mut tensors = tiny_weights(0);
        tensors.push(Tensor {
            name: "classifier.weight".into(),
            dtype: "F32",
            shape: vec![3, 8], // 3 rows instead of 2 labels
            data: vec![0u8; 3 * 8 * 4],
        });
        tensors.push(Tensor { name: "classifier.bias".into(), dtype: "F32", shape: vec![2], data: vec![0u8; 2 * 4] });
        f.weights("weights/model.safetensors", &tensors);
        let e = f.load().err().unwrap();
        assert!(
            e.is(
                BUNDLE_INVALID,
                "classifier.weight (classifier_weight) has shape [3, 8]; the architecture implies [2, 8]"
            ),
            "{e:?}"
        );
        let b = Bundle::open(&f.dir).unwrap();
        let he = Head::load(&b).unwrap_err();
        assert_eq!(he.code, BUNDLE_INVALID);
        assert!(he.message.contains("has shape [3, 8]; the architecture implies [2, 8]"), "{he:?}");
    }
}

#[test]
fn softmax_and_linear_validate_shapes_and_numerical_stability() {
    let mut probs = [0.0f64; 3];
    softmax(&[1000.0, 1000.0, 999.0], &mut probs).unwrap();
    let sum: f64 = probs.iter().sum();
    assert!((sum - 1.0).abs() < 1e-12);
    assert!((probs[0] - probs[1]).abs() < 1e-12);
    assert!(probs[0] > probs[2]);

    assert_eq!(softmax(&[], &mut []).unwrap_err().code, INVALID_SHAPE);
    assert_eq!(softmax(&[1.0, 2.0], &mut [0.0; 3]).unwrap_err().code, INVALID_SHAPE);
    assert_eq!(softmax(&[f32::NAN, 1.0], &mut [0.0; 2]).unwrap_err().code, INVALID_ARGUMENT);
    assert_eq!(softmax(&[f32::INFINITY, 1.0], &mut [0.0; 2]).unwrap_err().code, INVALID_ARGUMENT);

    assert_eq!(Linear::new(0, 2, vec![], vec![0.0, 0.0]).unwrap_err().code, INVALID_ARGUMENT);
    assert_eq!(Linear::new(2, 2, vec![1.0; 3], vec![0.0, 0.0]).unwrap_err().code, INVALID_SHAPE);
    assert_eq!(Linear::new(2, 2, vec![f32::NAN; 4], vec![0.0, 0.0]).unwrap_err().code, INVALID_ARGUMENT);
}
