//! Generate the OVMS reference embeddings (goldens) under
//! `testdata/reference_embeddings/` by calling the `inferstream.v1.Embed`
//! RPC through a running inferstream-intel façade routing to OVMS.
//!
//! Run on the OVMS host (krick-1) with the façade up on :8461:
//!
//! ```bash
//! ./target/release/inferstream-intel --config config/intel.toml &
//! INFERSTREAM_API_KEY=change-me \
//!   cargo run -p inferstream-arch-intel --example gen_ovms_goldens
//! ```
//!
//! Optional args: `-- <endpoint>` (default `http://127.0.0.1:8461`).
//!
//! Embeddings come from the OVMS DAG pipelines (`minilm_pipeline`,
//! `mpnet_pipeline`) executing on the Intel GPU, so regenerated files can
//! differ in the low-order bits between driver/OVMS versions — the golden
//! test compares cosine similarity, not exact values. Regenerate only after
//! an intentional pipeline/model change, and review the l2/dim diff.

use std::fmt::Write as _;
use std::path::PathBuf;

use inferstream_protocol::extension::inferstream_service_client::InferstreamServiceClient;
use inferstream_protocol::extension::EmbedRequest;
use tonic::metadata::MetadataValue;
use tonic::Request;

/// The fixed prompt set — mirrors `gen_reference_embeddings` (mock goldens)
/// so the same inputs cover both backends. Names double as file suffixes.
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
        // Long enough that the pipeline tokenizers (max_seq_len 256/384)
        // truncate it, pinning the truncation behavior into the golden.
        (
            "long_truncation",
            "token ".repeat(600).trim_end().to_string(),
        ),
    ]
}

/// (OVMS pipeline name, golden file prefix).
const MODELS: &[(&str, &str)] = &[
    ("minilm_pipeline", "ovms_minilm"),
    ("mpnet_pipeline", "ovms_mpnet"),
];

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let endpoint = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "http://127.0.0.1:8461".to_string());
    let api_key = std::env::var("INFERSTREAM_API_KEY").ok();

    let out_dir =
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../testdata/reference_embeddings");
    std::fs::create_dir_all(&out_dir)?;

    let mut client = InferstreamServiceClient::connect(endpoint.clone()).await?;
    println!("connected to {endpoint}");

    for (model, prefix) in MODELS {
        for (name, text) in prompts() {
            let mut request = Request::new(EmbedRequest {
                model_name: (*model).to_string(),
                texts: vec![text.clone()],
                ..Default::default()
            });
            if let Some(key) = &api_key {
                request.metadata_mut().insert(
                    "authorization",
                    MetadataValue::try_from(format!("Bearer {key}"))?,
                );
            }
            let response = client.embed(request).await?.into_inner();
            let vector = &response.embeddings[0].values;
            assert_eq!(vector.len(), response.dim as usize, "{model}/{name}: dim");
            let l2 = vector.iter().map(|v| v * v).sum::<f32>().sqrt();

            // Hand-rolled JSON keeps the example dependency-free; full
            // vectors at 384/768 dims are tens of KB, which the schema
            // explicitly allows (see testdata README).
            let mut values = String::new();
            for (i, v) in vector.iter().enumerate() {
                if i > 0 {
                    values.push_str(", ");
                }
                write!(values, "{v:?}")?;
            }
            let json = format!(
                "{{\n  \"model\": \"{model}\",\n  \"backend\": \"ovms\",\n  \
                 \"text\": {},\n  \"pooling\": null,\n  \"normalize\": false,\n  \
                 \"dim\": {},\n  \"l2\": {:?},\n  \"vector\": [{}]\n}}\n",
                escape_json(&text),
                vector.len(),
                l2,
                values
            );
            let path = out_dir.join(format!("{prefix}_{name}.json"));
            std::fs::write(&path, json)?;
            println!("wrote {} (dim {}, l2 {l2:.6})", path.display(), vector.len());
        }
    }
    Ok(())
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
