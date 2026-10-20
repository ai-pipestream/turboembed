use std::path::PathBuf;
use turbo::deberta::{Encoder, package::PackageManifest};
use turbo_bundle::deberta::{convert, from_huggingface, write_package};
const WEIGHTS: &[u8] = include_bytes!("../../core/tests/fixtures/deberta/weights.safetensors");
fn config_json() -> serde_json::Value {
    serde_json::from_slice(include_bytes!("../../core/tests/fixtures/deberta/config.json")).unwrap()
}

struct Scratch(PathBuf);
impl Scratch {
    fn new() -> Self {
        let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../target/deberta-tests");
        std::fs::create_dir_all(&root).unwrap();
        let name = format!(
            "{}-{}",
            std::process::id(),
            std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_nanos()
        );
        Self(root.join(name))
    }
}
impl Drop for Scratch {
    fn drop(&mut self) {
        if self.0.exists() {
            std::fs::remove_dir_all(&self.0).unwrap();
        }
    }
}

#[test]
fn conversion_is_deterministic_and_package_is_verified() {
    let cfg = from_huggingface(&config_json()).unwrap();
    let a = convert(&cfg, WEIGHTS, "").unwrap();
    assert_eq!(a, convert(&cfg, WEIGHTS, "").unwrap());
    let expected = Encoder::from_safetensors(cfg.clone(), WEIGHTS).unwrap().encode(&[0, 1, 2], &[true; 3]).unwrap();
    let path = Scratch::new();
    let source = ("fastino/gliner2-base-v1", "f9634218e53580c56edf0de97ca1a7d3f1c2354e", "Apache-2.0");
    write_package(cfg.clone(), WEIGHTS, "", source, &path.0).unwrap();
    let (meta, loaded) = PackageManifest::load(&path.0).unwrap();
    assert_eq!(meta.source_revision, source.1);
    assert_eq!(loaded.encode(&[0, 1, 2], &[true; 3]).unwrap().data(), expected.data());
    assert!(write_package(cfg, WEIGHTS, "", source, &path.0).is_err());
    let mut broken = a;
    let n = broken.len();
    broken[n - 1] ^= 1;
    std::fs::write(path.0.join("weights.safetensors"), broken).unwrap();
    assert!(PackageManifest::load(&path.0).err().unwrap().message.contains("SHA-256"));
}

#[test]
fn unsupported_hf_profiles_and_invalid_metadata_are_errors() {
    for (field, value) in [
        ("model_type", serde_json::json!("bert")),
        ("conv_kernel_size", serde_json::json!(3)),
        ("share_att_key", serde_json::json!(false)),
        ("position_biased_input", serde_json::json!(true)),
        ("pos_att_type", serde_json::json!(["c2p"])),
        ("type_vocab_size", serde_json::json!(2)),
        ("hidden_act", serde_json::json!("relu")),
        ("embedding_size", serde_json::json!(16)),
        ("norm_rel_ebd", serde_json::json!("none")),
        ("z_steps", serde_json::json!(2)),
        ("num_attention_heads", serde_json::json!(0)),
        ("max_relative_positions", serde_json::json!(0)),
    ] {
        let mut v = config_json();
        v[field] = value;
        assert!(from_huggingface(&v).is_err(), "accepted {field}");
    }
    let cfg = from_huggingface(&config_json()).unwrap();
    assert!(convert(&cfg, WEIGHTS, "wrong.").is_err());
    assert!(convert(&cfg, &WEIGHTS[..100], "").is_err());
    let path = Scratch::new();
    assert!(write_package(cfg, WEIGHTS, "", ("repo", "main", "Apache-2.0"), &path.0).is_err());
    assert!(!path.0.exists());
}
