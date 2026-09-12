//! Generate the mock-backend reference embeddings (goldens) under
//! `testdata/reference_embeddings/`.
//!
//! Run from the repo root whenever the golden schema or prompt set changes:
//!
//! ```bash
//! cargo run -p inferstream-server --example gen_reference_embeddings
//! ```
//!
//! The mock backend is fully deterministic, so regenerated files are
//! byte-identical unless its algorithm changes — which is exactly what the
//! golden tests are meant to catch. GPU goldens (ORT CUDA on krick,
//! OpenVINO GenAI on krick-1) are regenerated separately; see
//! `testdata/reference_embeddings/README.md`.

use std::fmt::Write as _;
use std::path::PathBuf;

use inferstream_backend_mock::MockBackend;

/// The fixed prompt set. Names double as file names.
fn prompts() -> Vec<(&'static str, String)> {
    vec![
        ("short", "hello world".to_string()),
        (
            "medium",
            "The quick brown fox jumps over the lazy dog while the \
             inference server streams embeddings back to its clients."
                .to_string(),
        ),
        ("empty", String::new()),
        (
            "unicode",
            "καλημέρα κόσμε — 你好世界 — здравствуй мир ✓🦀".to_string(),
        ),
        // Long enough that real models with max_seq_len 256/512 truncate it;
        // the mock embeds the full text (it has no sequence limit), which is
        // fine — goldens are compared per (model, text, params) tuple.
        (
            "long_truncation",
            "token ".repeat(600).trim_end().to_string(),
        ),
    ]
}

fn main() {
    let backend = MockBackend::default();
    let out_dir =
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../testdata/reference_embeddings");
    std::fs::create_dir_all(&out_dir).expect("create testdata/reference_embeddings");

    for (name, text) in prompts() {
        let vector = backend.embed(text.as_bytes());
        let l2 = vector.iter().map(|v| v * v).sum::<f32>().sqrt();

        // Hand-rolled JSON keeps the example dependency-free; the vectors are
        // small (dim 8) so full vectors are stored. For large GPU vectors the
        // schema also allows hash + head/tail + l2 (see README).
        let mut values = String::new();
        for (i, v) in vector.iter().enumerate() {
            if i > 0 {
                values.push_str(", ");
            }
            write!(values, "{v:?}").unwrap();
        }
        let json = format!(
            "{{\n  \"model\": \"mock-embed\",\n  \"backend\": \"mock\",\n  \
             \"text\": {},\n  \"pooling\": null,\n  \"normalize\": false,\n  \
             \"dim\": {},\n  \"l2\": {:?},\n  \"vector\": [{}]\n}}\n",
            escape_json(&text),
            vector.len(),
            l2,
            values
        );
        let path = out_dir.join(format!("mock_{name}.json"));
        std::fs::write(&path, json).expect("write golden");
        println!("wrote {}", path.display());
    }
}

fn escape_json(s: &str) -> String {
    let mut out = String::with_capacity(s.len() + 2);
    out.push('"');
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if (c as u32) < 0x20 => {
                write!(out, "\\u{:04x}", c as u32).unwrap();
            }
            c => out.push(c),
        }
    }
    out.push('"');
    out
}
