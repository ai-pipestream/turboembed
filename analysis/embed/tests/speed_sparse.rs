//! How fast the sparse step encodes the reference texts on the CPU:
//! tokenizing, every window through the encoder, the MLM head and the
//! SPLADE pooling. Ignored, since it measures rather than checks; run it
//! with
//!
//! ```
//! cargo test --release -p turbo-analysis-embed --test speed_sparse -- --ignored --nocapture
//! ```
//!
//! The bundle is the small BERT of testdata/tiny-bert-bundle with a
//! hand-built MLM head appended (its two layers of width 32 say little
//! about a real SPLADE model; TURBO_SPARSE_BUNDLE names another bundle
//! to measure); TURBO_SPEED_RUNS names the timed runs (default 5). The
//! texts are the SaT speed test's (analysis/sat/tests/data/texts.jsonl).

use std::fs;
use std::path::{Path, PathBuf};
use std::time::Instant;

use serde_json::{Value, json};
use turbo_analysis::{Annotator, Document, Ix};
use turbo_analysis_embed::{SparseAnnotator, SparseOptions, SubwordAnnotator};

fn bert() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../testdata/tiny-bert-bundle")
}

/// The small BERT with an MLM head appended, the decoder tied: the
/// sparse bundle the step opens.
fn bundle() -> PathBuf {
    if let Some(p) = std::env::var_os("TURBO_SPARSE_BUNDLE") {
        return PathBuf::from(p);
    }
    let dir = std::env::temp_dir().join(format!("turbo-embed-sparse-speed-{}", std::process::id()));
    let _ = fs::remove_dir_all(&dir);
    for f in ["tokenizer.json", "reference/reference.safetensors"] {
        fs::create_dir_all(dir.join(f).parent().unwrap()).unwrap();
        fs::copy(bert().join(f), dir.join(f)).unwrap();
    }
    let src = fs::read(bert().join("weights/model.safetensors")).unwrap();
    let n = u64::from_le_bytes(src[..8].try_into().unwrap()) as usize;
    let mut header: serde_json::Map<String, Value> = serde_json::from_slice(&src[8..8 + n]).unwrap();
    let mut data = src[8 + n..].to_vec();
    let lin = |start: f32, end: f32, count: usize| {
        (0..count)
            .flat_map(|i| {
                let t = i as f32 / (count - 1) as f32;
                (start + (end - start) * t).to_le_bytes()
            })
            .collect::<Vec<u8>>()
    };
    for (name, shape, bytes) in [
        ("mlm.transform.weight", vec![32, 32], lin(-0.2, 0.3, 1024)),
        ("mlm.transform.bias", vec![32], lin(-0.1, 0.1, 32)),
        ("mlm.transform.LayerNorm.weight", vec![32], lin(0.8, 1.2, 32)),
        ("mlm.transform.LayerNorm.bias", vec![32], lin(-0.05, 0.05, 32)),
        ("mlm.decoder.bias", vec![30522], lin(-0.06, 0.06, 30522)),
    ] {
        let at = data.len();
        data.extend(bytes);
        header.insert(name.into(), json!({ "dtype": "F32", "shape": shape, "data_offsets": [at, data.len()] }));
    }
    let mut h = serde_json::to_vec(&header).unwrap();
    while !h.len().is_multiple_of(8) {
        h.push(b' ');
    }
    let mut weights = (h.len() as u64).to_le_bytes().to_vec();
    weights.extend(h);
    weights.extend(data);
    fs::create_dir_all(dir.join("weights")).unwrap();
    fs::write(dir.join("weights/model.safetensors"), &weights).unwrap();
    let mut m: Value = serde_json::from_slice(&fs::read(bert().join("manifest.json")).unwrap()).unwrap();
    m["embed"]["sparse"] = json!({ "pooling": "SPARSE_POOLING_MAXLOG" });
    let tn = m["artifacts"][0]["tensor_names"].as_object_mut().unwrap();
    tn.insert("mlm_transform_weight".into(), json!("mlm.transform.weight"));
    tn.insert("mlm_transform_bias".into(), json!("mlm.transform.bias"));
    tn.insert("mlm_transform_ln_weight".into(), json!("mlm.transform.LayerNorm.weight"));
    tn.insert("mlm_transform_ln_bias".into(), json!("mlm.transform.LayerNorm.bias"));
    tn.insert("mlm_decoder_bias".into(), json!("mlm.decoder.bias"));
    let w = m["files"].as_array_mut().unwrap().iter_mut().find(|f| f["path"] == "weights/model.safetensors").unwrap();
    w["size"] = json!(weights.len());
    w["sha256"] = json!(turbo::bundle::sha256_hex(&weights));
    fs::write(dir.join("manifest.json"), serde_json::to_vec_pretty(&m).unwrap()).unwrap();
    dir
}

#[test]
#[ignore]
fn sparse_speed_on_the_cpu() {
    let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("../sat/tests/data/texts.jsonl");
    let texts: Vec<String> =
        std::fs::read_to_string(path).unwrap().lines().map(|l| serde_json::from_str(l).unwrap()).collect();
    // Documents of a few kilobytes, as the SaT speed test makes them.
    let docs: Vec<String> = (0..4).flat_map(|_| texts.chunks(10).map(|c| c.join("\n"))).collect();
    let bytes: usize = docs.iter().map(|d| d.len()).sum();
    let runs: u32 = std::env::var("TURBO_SPEED_RUNS").ok().and_then(|r| r.parse().ok()).unwrap_or(5);
    let a = SparseAnnotator::open(&bundle(), &SparseOptions::default()).unwrap();
    let sub = SubwordAnnotator::open(&bundle()).unwrap();
    let mut built: Vec<(Document, _)> = Vec::with_capacity(docs.len());
    for t in &docs {
        let d = Document::new(t.as_str());
        let s = {
            let v = d.get::<turbo_analysis::View>().unwrap();
            sub.pieces(v).unwrap()
        };
        built.push((d, s));
    }
    let docs = built;
    let pieces: usize = docs.iter().map(|(_, s)| s.ids.len()).sum();

    // One warm pass, then the timed ones.
    let encode = || {
        docs.iter()
            .map(|(d, s)| {
                let v = d.get::<turbo_analysis::View>().unwrap();
                a.annotate((v, s)).unwrap().0
            })
            .collect::<Vec<_>>()
    };
    let terms: usize = encode().iter().map(|t| t.entries(Ix::new(0)).0.len()).sum();
    let start = Instant::now();
    for _ in 0..runs {
        std::hint::black_box(encode());
    }
    let secs = start.elapsed().as_secs_f64() / f64::from(runs);
    eprintln!(
        "{}: {} documents, {bytes} bytes, {pieces} pieces, {terms} terms: {:.1} ms a pass, {:.1} texts/s, {:.0} pieces/s, {:.0} terms/s",
        bundle().display(),
        docs.len(),
        secs * 1e3,
        docs.len() as f64 / secs,
        pieces as f64 / secs,
        terms as f64 / secs,
    );
}
