use std::fs;
use std::path::{Path, PathBuf};
use std::sync::OnceLock;

use turbo::deberta::package::{PackageManifest, digest};

pub fn package() -> PathBuf {
    static PACKAGE: OnceLock<PathBuf> = OnceLock::new();
    PACKAGE
        .get_or_init(|| {
            let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("..");
            let dir = std::env::temp_dir().join(format!("turbo-kserve-gliner2-package-{}", std::process::id()));
            fs::create_dir_all(dir.join("encoder")).unwrap();
            let config = turbo::deberta::Config {
                hidden: 32,
                intermediate: 48,
                heads: 2,
                layers: 2,
                vocabulary: 800,
                max_sequence: 128,
                position_buckets: 8,
                max_relative_positions: 32,
                layer_norm_eps: 1e-7,
            };
            let raw = fs::read(root.join("analysis/relation/tests/fixtures/glirel/reduced.safetensors")).unwrap();
            let weights = turbo_bundle::deberta::convert(&config, &raw, "token_rep_layer.bert_layer.model.").unwrap();
            let heads = fs::read(root.join("analysis/relation/tests/fixtures/gliner2/reduced.safetensors")).unwrap();
            let spm = fs::read(root.join("testdata/sentencepiece/unigram-identity.model")).unwrap();
            let encoder = PackageManifest {
                version: 1,
                config,
                source_repository: "fixture/gliner2".into(),
                source_revision: "2".repeat(40),
                source_license: "Apache-2.0".into(),
                weights_sha256: digest(&weights),
            };
            fs::write(dir.join("encoder/deberta.json"), serde_json::to_vec(&encoder).unwrap()).unwrap();
            fs::write(dir.join("encoder/weights.safetensors"), weights).unwrap();
            fs::write(dir.join("model.safetensors"), heads).unwrap();
            fs::write(dir.join("spm.model"), spm).unwrap();
            let config = serde_json::json!({
                "counting_layer": "count_lstm_v2",
                "token_pooling": "first",
                "max_width": 8,
            });
            fs::write(dir.join("config.json"), serde_json::to_vec(&config).unwrap()).unwrap();
            let added = serde_json::json!({
                "[SEP_STRUCT]": 12,
                "[SEP_TEXT]": 13,
                "[P]": 14,
                "[R]": 15,
            });
            fs::write(dir.join("added_tokens.json"), serde_json::to_vec(&added).unwrap()).unwrap();
            dir
        })
        .clone()
}
