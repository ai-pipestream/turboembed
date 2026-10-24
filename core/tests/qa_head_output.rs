mod common;
#[path = "qa/common.rs"]
mod qa_common;
use common::{Tensor, bert_weights, safetensors_file};
use std::{fs, path::Path};
use turbo::qa::ExtractiveQa;
#[test]
fn cpu_hidden_two_still_applies_the_span_head() {
    let mut f = qa_common::Fixture::new(Path::new(env!("CARGO_MANIFEST_DIR")).parent().unwrap());
    f.manifest["architecture"]["hidden"] = 2.into();
    f.manifest["architecture"]["heads"] = 1.into();
    f.manifest["architecture"]["intermediate"] = 4.into();
    let layers = f.manifest["architecture"]["layers"].as_u64().unwrap() as usize;
    let tensors = bert_weights(layers, 2, 4, 0, 1.0);
    fs::write(f.dir.join("weights/model.safetensors"), safetensors_file(&tensors)).unwrap();
    f.refresh("weights/model.safetensors");
    let tensors = vec![
        Tensor { name: "qa_outputs.weight".into(), dtype: "F32", shape: vec![2, 2], data: vec![0; 16] },
        Tensor {
            name: "qa_outputs.bias".into(),
            dtype: "F32",
            shape: vec![2],
            data: [7f32, 11f32].into_iter().flat_map(f32::to_le_bytes).collect(),
        },
    ];
    fs::write(f.dir.join("weights/head.safetensors"), safetensors_file(&tensors)).unwrap();
    f.refresh("weights/head.safetensors");
    let mut qa = ExtractiveQa::open(&f.dir, 1).unwrap();
    let ws = qa.logits("question", "alpha beta").unwrap();
    assert!(!ws.is_empty());
    for w in ws {
        assert!(w.start.len() >= 3);
        assert_eq!(w.start, vec![7.; w.start.len()]);
        assert_eq!(w.end, vec![11.; w.end.len()]);
    }
}
