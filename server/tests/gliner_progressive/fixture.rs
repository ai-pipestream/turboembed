use std::fs;
use std::path::{Path, PathBuf};
use std::sync::OnceLock;

use turbo::deberta::package::{PackageManifest, digest};

fn tokenizer(root: &Path) -> Vec<u8> {
    fn varint(bytes: &[u8], pos: &mut usize) -> usize {
        let mut out = 0;
        for shift in (0..64).step_by(7) {
            let b = bytes[*pos];
            *pos += 1;
            out |= usize::from(b & 127) << shift;
            if b < 128 {
                return out;
            }
        }
        panic!("fixture varint");
    }
    let mut out = Vec::new();
    for (name, kind) in [("<unk>", 2), ("<s>", 3), ("</s>", 3), ("▁", 1)] {
        let mut piece = vec![10, name.len() as u8];
        piece.extend_from_slice(name.as_bytes());
        piece.extend_from_slice(&[24, kind]);
        out.extend_from_slice(&[10, piece.len() as u8]);
        out.extend(piece);
    }
    let bytes = fs::read(root.join("testdata/sentencepiece/t5.model")).unwrap();
    let mut pos = 0;
    while pos < bytes.len() {
        let start = pos;
        let key = varint(&bytes, &mut pos);
        assert_eq!(key & 7, 2);
        let len = varint(&bytes, &mut pos);
        pos += len;
        if key >> 3 == 3 {
            out.extend_from_slice(&bytes[start..pos]);
            return out;
        }
    }
    panic!("fixture normalizer missing");
}

pub fn package() -> PathBuf {
    static PACKAGE: OnceLock<PathBuf> = OnceLock::new();
    PACKAGE
        .get_or_init(|| {
            let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("..");
            let dir = std::env::temp_dir().join(format!("turbo-kserve-gliner-package-{}", std::process::id()));
            fs::create_dir_all(dir.join("encoder")).unwrap();
            let weights = fs::read(root.join("core/tests/fixtures/deberta/weights.safetensors")).unwrap();
            let heads = fs::read(root.join("analysis/gliner/tests/fixtures/heads.safetensors")).unwrap();
            let tokenizer = tokenizer(&root);
            let config = turbo::deberta::Config {
                hidden: 32,
                intermediate: 48,
                heads: 2,
                layers: 2,
                vocabulary: 16,
                max_sequence: 32,
                position_buckets: 8,
                max_relative_positions: 32,
                layer_norm_eps: 1e-7,
            };
            let encoder = PackageManifest {
                version: 1,
                config,
                source_repository: "fixture/gliner".into(),
                source_revision: "1".repeat(40),
                source_license: "Apache-2.0".into(),
                weights_sha256: digest(&weights),
            };
            let encoder_bytes = serde_json::to_vec(&encoder).unwrap();
            let gliner = turbo_gliner::package::Manifest {
                version: 1,
                config: turbo_gliner::Config {
                    hidden: 16,
                    projection_factor: 4,
                    prompt_factor: 4,
                    out_project_mlp: true,
                    token_level: false,
                    prompt_projection: true,
                    max_width: 3,
                    entity_token: 12,
                    separator_token: 13,
                    cls_token: 1,
                    eos_token: 2,
                },
                encoder_sha256: digest(&encoder_bytes),
                heads_sha256: digest(&heads),
                tokenizer_sha256: digest(&tokenizer),
            };
            fs::write(dir.join("encoder/deberta.json"), encoder_bytes).unwrap();
            fs::write(dir.join("encoder/weights.safetensors"), weights).unwrap();
            fs::write(dir.join("heads.safetensors"), heads).unwrap();
            fs::write(dir.join("spm.model"), tokenizer).unwrap();
            fs::write(dir.join("gliner.json"), serde_json::to_vec(&gliner).unwrap()).unwrap();
            dir
        })
        .clone()
}
