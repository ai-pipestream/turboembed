//! RoBERTa: the positions counted from the offset.

use crate::common::*;
use crate::support::*;
use serde_json::json;
use turbo::*;

/// The small BERT's weights with the position table's first two rows
/// set to 3.0 and -3.0 by turns (a constant row would vanish in the
/// embeddings' LayerNorm): a row read from the table's start comes out
/// unlike one read from row 2.
fn weights_with_marked_first_positions() -> Vec<Tensor> {
    let mut tensors = tiny_weights(0);
    let t = tensors.iter_mut().find(|t| t.name == "embeddings.position_embeddings.weight").unwrap();
    for i in 0..16 {
        let v = if i % 2 == 0 { 3.0f32 } else { -3.0 };
        t.data[4 * i..4 * i + 4].copy_from_slice(&v.to_le_bytes());
    }
    tensors
}

/// The small BERT as FAMILY_ROBERTA: its rows read the position table
/// from row 2, as the f64 encoder does with the offset, and not from row
/// 0, which the same weights as FAMILY_BERT read.
#[test]
fn a_roberta_model_reads_positions_from_its_offset() {
    let mut f = Fixture::new("roberta", {
        let mut m = model_manifest();
        m["architecture"]["family"] = json!("FAMILY_ROBERTA");
        m["architecture"]["position_offset"] = json!(2);
        m
    });
    f.weights("weights/model.safetensors", &weights_with_marked_first_positions());
    let l = f.load().unwrap();
    let s = session(&l);
    let tok = Tok::create(&f.dir).unwrap();
    let plain = PlainBert::new(&f.dir);
    let got = s.embed(&TEXTS, None).unwrap();
    let mut b = Fixture::new("roberta-as-bert", model_manifest());
    b.weights("weights/model.safetensors", &weights_with_marked_first_positions());
    let bert = session(&b.load().unwrap()).embed(&TEXTS, None).unwrap();
    for (i, t) in TEXTS.iter().enumerate() {
        let ids = tok.row(t, None).unwrap();
        let ones = vec![1; ids.len()];
        let want = plain.embed(&ids, &ones, &vec![0; ids.len()], TURBO_POOLING_MEAN, 8, true);
        for (a, w) in got[i].iter().zip(&want) {
            assert!((*a as f64 - w).abs() < 1e-5, "{a} vs {w}");
        }
        // The reading from row 0 is another vector, by far more than the
        // f64 encoder is matched at.
        assert!(max_abs_diff(&got[i], &bert[i]) > 1e-4, "{t:?}: the same vector from row 0 of the table");
    }
    // The bundle's max_seq, which the manifest keeps inside the table
    // less the offset, is a session's to take in full.
    Session::create(l.m, Some(&session_desc(1, MAX_SEQ as u32, 0))).unwrap();
}
