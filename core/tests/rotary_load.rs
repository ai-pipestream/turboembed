//! A ModernBERT or nomic-bert bundle through the loader (#1461): its
//! tensors found in the family's order and checked against the
//! architecture, its rotary fields and cos/sin table in the description a
//! backend is handed, and only the CPU running it.

#![cfg(feature = "cpu")]

mod common;
#[path = "common/rotary.rs"]
mod rotary;

use rotary::{HEADS, HIDDEN, LAYERS};
use turbo::backend::*;
use turbo::bundle::Bundle;
use turbo::model::{Weights, choose};
use turbo::status::{BUNDLE_INVALID, UNSUPPORTED};

fn weights(f: &common::Fixture) -> Weights {
    let bundle = Bundle::open(f.write()).unwrap();
    let index = choose(&bundle.manifest, "cpu", format_bit(TURBO_FORMAT_SAFETENSORS), "").unwrap();
    Weights::load(&bundle, index).unwrap()
}

#[test]
fn a_modernbert_bundle_loads_its_tensors_in_the_family_order() {
    let f = rotary::modernbert("rotary-modernbert");
    let w = weights(&f);
    let t = w.tensors();
    let (emb, per) = (TURBO_MODERNBERT_EMBEDDING_TENSORS as usize, TURBO_MODERNBERT_LAYER_TENSORS as usize);
    assert_eq!(t.len(), emb + LAYERS * per);
    // Layer 0 has no attention norm: its entry is NULL, every other is there.
    let at = emb + TURBO_MODERNBERT_ATTN_LN_WEIGHT as usize;
    assert!(t[at].data.is_null() && t[at].bytes == 0);
    assert!(t.iter().enumerate().all(|(i, x)| i == at || !x.data.is_null()));
    assert_eq!(t[emb + per + TURBO_MODERNBERT_FFN_IN_WEIGHT as usize].shape, [2 * rotary::INTERMEDIATE, HIDDEN]);
    assert_eq!(t[emb + TURBO_MODERNBERT_QKV_WEIGHT as usize].shape, [3 * HIDDEN, HIDDEN]);

    let d = w.desc(&t);
    assert_eq!(d.family, TURBO_FAMILY_MODERNBERT);
    assert_eq!((d.mlp, d.activation, d.norm), (TURBO_MLP_GATED, TURBO_ACTIVATION_GELU_ERF, TURBO_NORM_LAYER_NO_BIAS));
    assert_eq!((d.attention_window, d.global_every), (4, 3));
    assert_eq!((d.rope_global_theta, d.rope_local_theta), (160000.0, 10000.0));
    let pairs = (HIDDEN / HEADS / 2) as usize;
    assert_eq!(d.rope_pairs as usize, pairs);
    // The table the description points at is the core's for both bases.
    let positions = d.max_positions as usize;
    let table = unsafe { std::slice::from_raw_parts(d.rope_table, 4 * positions * pairs) };
    assert_eq!(table, turbo::rope::table(160000.0, 10000.0, positions, pairs).unwrap());
}

#[test]
fn a_nomic_bundle_loads_with_one_rotary_base() {
    let f = rotary::nomic("rotary-nomic");
    let w = weights(&f);
    let t = w.tensors();
    assert_eq!(t.len(), (TURBO_NOMIC_EMBEDDING_TENSORS + LAYERS as u32 * TURBO_NOMIC_LAYER_TENSORS) as usize);
    assert!(t.iter().all(|x| !x.data.is_null()));
    // The gate is fc12 and the value fc11, as upstream pairs them.
    let gate = unsafe {
        std::ffi::CStr::from_ptr(t[(TURBO_NOMIC_EMBEDDING_TENSORS + TURBO_NOMIC_FFN_GATE_WEIGHT) as usize].name)
    };
    assert_eq!(gate.to_str().unwrap(), "encoder.layers.0.mlp.fc12.weight");
    let d = w.desc(&t);
    assert_eq!(d.family, TURBO_FAMILY_NOMIC_BERT);
    assert_eq!((d.mlp, d.activation, d.norm), (TURBO_MLP_GATED, TURBO_ACTIVATION_SILU, TURBO_NORM_LAYER));
    assert_eq!((d.attention_window, d.global_every), (0, 0));
    assert_eq!((d.rope_global_theta, d.rope_local_theta), (1000.0, 1000.0));
}

/// The families' descriptions carry nothing of theirs for any other.
#[test]
fn a_bert_bundle_describes_no_rotary_fields() {
    let f = common::Fixture::model("rotary-bert");
    let w = weights(&f);
    let d = w.desc(&w.tensors());
    assert!(d.rope_table.is_null());
    assert_eq!([d.rope_pairs, d.mlp, d.activation, d.norm, d.attention_window, d.global_every], [0; 6]);
}

#[test]
fn a_modernbert_file_with_a_first_attention_norm_is_refused() {
    let mut f = rotary::modernbert("rotary-first-norm");
    f.weights("weights/model.safetensors", &rotary::modernbert_weights(true));
    let e = f.load().err().unwrap();
    assert!(e.is(BUNDLE_INVALID, "first layer has no attention norm"), "{e:?}");
}

/// The CPU, CUDA, Level Zero and Metal run both families (#1461); every
/// other backend refuses them before model_load.
#[test]
fn only_the_cpu_cuda_levelzero_and_metal_run_a_rotary_family() {
    for f in [rotary::modernbert("rotary-cpu-mb"), rotary::nomic("rotary-cpu-nb")] {
        assert!(f.load().is_ok());
    }
    for (family, name) in
        [(TURBO_FAMILY_MODERNBERT, "FAMILY_MODERNBERT"), (TURBO_FAMILY_NOMIC_BERT, "FAMILY_NOMIC_BERT")]
    {
        assert!(turbo::model::runs_family("cpu", family).is_ok());
        assert!(turbo::model::runs_family("cuda", family).is_ok());
        assert!(turbo::model::runs_family("levelzero", family).is_ok());
        assert!(turbo::model::runs_family("metal", family).is_ok());
        for b in ["npu", "hailo"] {
            let e = turbo::model::runs_family(b, family).err().unwrap();
            assert_eq!(e.code, UNSUPPORTED);
            assert!(e.message.contains(&format!("{name}: the {b} backend has no encoder for it yet")), "{}", e.message);
        }
    }
}

/// A row padded on the left, on a sliding-window layer: its first pads
/// see no live key in their window. Their context must be 0, not NaN, or
/// the NaN reaches the row's live tokens a layer later (found by the
/// Level Zero leg on rows of 17, 5, 40, 9 and 1 tokens padded to 40).
#[test]
fn a_left_padded_modernbert_row_stays_finite() {
    let f = rotary::modernbert("rotary-left-pad");
    let loaded = f.load().unwrap();
    let session =
        common::Session::create(loaded.m, Some(&common::session_desc(8, 64, turbo::TURBO_PRECISION_MODEL))).unwrap();
    let lengths = [17usize, 5, 40, 9, 1];
    let rows: Vec<Vec<i32>> = lengths.iter().map(|&n| (0..n as i32).map(|k| 1000 + 37 * k).collect()).collect();
    let mut t = common::Tokens::new(&rows, 0);
    // Row 3 moves to the right end of its 40 columns.
    let (seq, n) = (t.seq as usize, lengths[3]);
    let at = 3 * seq;
    t.ids[at..at + seq].fill(0);
    t.mask[at..at + seq].fill(0);
    t.ids[at + seq - n..at + seq].copy_from_slice(&rows[3]);
    t.mask[at + seq - n..at + seq].fill(1);
    session.write_tokens(&t.batch(), None).unwrap();
    let out = session.run().unwrap().rows();
    for (r, v) in out.iter().enumerate() {
        assert!(v.iter().all(|x| x.is_finite()), "row {r}: {v:?}");
    }
}
