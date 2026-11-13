use std::path::Path;

use super::*;

#[test]
fn weights_load_rejects_out_of_bounds_artifact_index() {
    let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("../testdata/tiny-bert-bundle");
    let bundle = Bundle::open(&dir).expect("tiny-bert-bundle opens");
    let err = match Weights::load(&bundle, bundle.manifest.artifacts.len()) {
        Ok(_) => panic!("out-of-bounds artifact index must fail"),
        Err(e) => e,
    };
    assert_eq!(err.code, INVALID_ARGUMENT);
}

#[test]
fn only_the_cpu_cuda_levelzero_and_metal_run_a_banded_model() {
    for b in ["cpu", "cuda", "levelzero", "metal"] {
        assert!(runs_band(b, 16).is_ok(), "{b}");
    }
    for b in ["npu", "hailo"] {
        assert!(runs_band(b, 0).is_ok());
        let e = runs_band(b, 16).err().unwrap();
        assert_eq!(e.code, UNSUPPORTED);
        assert!(e.message.contains(&format!("the {b} backend has no limited-lookahead attention")), "{}", e.message);
    }
}

#[test]
fn only_the_cpu_cuda_levelzero_and_metal_run_a_rotary_model() {
    use crate::backend::{TURBO_FAMILY_MODERNBERT, TURBO_FAMILY_NOMIC_BERT};
    for family in [TURBO_FAMILY_MODERNBERT, TURBO_FAMILY_NOMIC_BERT] {
        for b in ["cpu", "cuda", "levelzero", "metal"] {
            assert!(runs_family(b, family).is_ok());
        }
        for b in ["npu", "hailo"] {
            let e = runs_family(b, family).err().unwrap();
            assert_eq!(e.code, UNSUPPORTED);
            assert!(
                e.message
                    .contains(&format!("the {b} backend has no encoder for it yet; use cpu, cuda, levelzero or metal")),
                "{}",
                e.message
            );
        }
    }
}

#[test]
fn only_the_cpu_cuda_levelzero_and_metal_run_a_deberta_model() {
    for b in ["cpu", "cuda", "levelzero", "metal"] {
        assert!(runs_family(b, TURBO_FAMILY_DEBERTA_V2).is_ok());
    }
    for b in ["npu", "hailo"] {
        assert!(runs_family(b, crate::backend::TURBO_FAMILY_BERT).is_ok());
        let e = runs_family(b, TURBO_FAMILY_DEBERTA_V2).err().unwrap();
        assert_eq!(e.code, UNSUPPORTED);
        assert!(e.message.contains(&format!("the {b} backend has no DeBERTa-v2 encoder")), "{}", e.message);
    }
}

/// Rule 6 skips an artifact whose backend has no attention for the band
/// or no encoder for the family, with the reason runs_band or runs_family
/// gives, rather than choosing it for a load that refuses it after hashing.
#[test]
fn choose_skips_an_artifact_whose_backend_cannot_run_the_band_or_the_family() {
    use crate::manifest::{Family, PositionEmbedding};
    let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("../testdata/tiny-bert-bundle");
    let mut bundle = Bundle::open(&dir).expect("tiny-bert-bundle opens");
    let safetensors = format_bit(crate::backend::TURBO_FORMAT_SAFETENSORS);
    let m = &mut bundle.manifest;
    assert_eq!(m.artifacts[0].backends, ["cuda", "metal", "levelzero", "cpu"]);
    for b in ["cpu", "cuda", "metal", "levelzero"] {
        assert_eq!(choose(m, b, safetensors, "").unwrap(), 0, "{b}");
    }

    m.architecture.as_mut().unwrap().lookahead = Some(4);
    m.artifacts[0].backends.push("npu".into());
    for b in ["cpu", "cuda", "levelzero", "metal"] {
        assert_eq!(choose(m, b, safetensors, "").unwrap(), 0, "{b}");
    }
    let e = choose(m, "npu", safetensors, "").unwrap_err();
    assert_eq!(e.code, BUNDLE_NO_ARTIFACT);
    let why = "weights-f32: architecture.lookahead 4: the npu backend has no limited-lookahead attention";
    assert!(e.message.contains(why), "{}", e.message);

    let a = m.architecture.as_mut().unwrap();
    a.lookahead = None;
    (a.family, a.position_embedding) = (Family::DebertaV2, PositionEmbedding::Relative);
    for b in ["cpu", "cuda", "metal", "levelzero"] {
        assert_eq!(choose(m, b, safetensors, "").unwrap(), 0, "{b}");
    }
    let e = choose(m, "npu", safetensors, "").unwrap_err();
    assert_eq!(e.code, BUNDLE_NO_ARTIFACT);
    let why = "weights-f32: FAMILY_DEBERTA_V2: the npu backend has no DeBERTa-v2 encoder yet";
    assert!(e.message.contains(why), "{}", e.message);
    // A reason that comes first is the one given: npu loads no raw weights.
    let e = choose(m, "npu", 0, "").unwrap_err();
    assert!(
        e.message.contains("weights-f32: FORMAT_SAFETENSORS is not a format the npu backend loads"),
        "{}",
        e.message
    );
}

/// A backend keys on the digest the manifest recorded, so it must be
/// the bytes sha256sum prints, and never a partial parse.
#[test]
fn a_manifest_digest_is_its_bytes_or_all_zero() {
    let hex = turbo_sha256::sha256(b"weights").to_hex();
    assert_eq!(digest(&hex), turbo_sha256::sha256(b"weights").0);
    for bad in ["", &hex[..62], &hex.to_uppercase(), &format!("{}zz", &hex[..62])] {
        assert_eq!(digest(bad), [0; 32], "{bad:?}");
    }
}
