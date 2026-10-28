//! A bundle whose manifest declares fewer encoder layers than its weights
//! hold is cut there (`architecture.layers`, docs/bundle.md): it runs
//! exactly its first layers, then the model's final norm, whatever the
//! weights hold past them. Checked on the three-layer ModernBERT and the
//! two-layer BERT in testdata, cut at every layer, against the same cut
//! with the later layers' tensors removed from the weights.

use std::fs;
use std::path::{Path, PathBuf};

use serde_json::{Map, Value, json};
use turbo::bundle::sha256_hex;
use turbo::turbo_token_range;
use turbo_analysis_embed::encoder::{Encoder, EncoderOptions, Rows};

fn testdata(name: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../testdata").join(name)
}

/// Every token's final state for one row.
fn states(dir: &Path) -> Vec<f32> {
    let mut enc = Encoder::open(dir, &EncoderOptions::default()).unwrap();
    let mut rows = Rows::new(16, 0);
    let ids = [1, 100, 200, 300, 400, 500, 600, 700, 800, 2];
    rows.push(&[&ids]);
    let ranges: Vec<turbo_token_range> =
        (0..ids.len() as u32).map(|i| turbo_token_range { row: 0, start: i, end: i + 1 }).collect();
    let mut out = Vec::new();
    enc.run_ranges(&rows, &ranges, &mut out).unwrap();
    out
}

/// The bundle `src` copied to `name`, cut to `layers`, and with `strip`
/// the tensors of the layers past the cut removed from its weights.
fn cut(src: &Path, name: &str, layers: u32, strip: bool) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("turbo-layer-cut-{}-{name}", std::process::id()));
    let _ = fs::remove_dir_all(&dir);
    let mut m: Value = serde_json::from_slice(&fs::read(src.join("manifest.json")).unwrap()).unwrap();
    for f in m["files"].as_array().unwrap() {
        let path = f["path"].as_str().unwrap();
        fs::create_dir_all(dir.join(path).parent().unwrap()).unwrap();
        fs::copy(src.join(path), dir.join(path)).unwrap();
    }
    assert!(m["architecture"]["layers"].as_u64().unwrap() > layers as u64);
    m["architecture"]["layers"] = json!(layers);
    if strip {
        let weights = dir.join("weights/model.safetensors");
        let kept = without_layers_from(&fs::read(&weights).unwrap(), layers);
        fs::write(&weights, &kept).unwrap();
        let f = m["files"].as_array_mut().unwrap().iter_mut().find(|f| f["path"] == "weights/model.safetensors");
        let f = f.unwrap();
        f["size"] = json!(kept.len());
        f["sha256"] = json!(sha256_hex(&kept));
    }
    fs::write(dir.join("manifest.json"), serde_json::to_vec_pretty(&m).unwrap()).unwrap();
    dir
}

/// A safetensors file without the tensors of `layers.{k}.` for k at or
/// past `layers`.
fn without_layers_from(src: &[u8], layers: u32) -> Vec<u8> {
    let n = u64::from_le_bytes(src[..8].try_into().unwrap()) as usize;
    let header: Map<String, Value> = serde_json::from_slice(&src[8..8 + n]).unwrap();
    let data = &src[8 + n..];
    let past = |name: &str| {
        name.split('.')
            .collect::<Vec<_>>()
            .windows(2)
            .any(|w| (w[0] == "layers" || w[0] == "layer") && w[1].parse::<u32>().is_ok_and(|k| k >= layers))
    };
    let (mut out_header, mut out_data, mut dropped) = (Map::new(), Vec::new(), 0);
    for (name, v) in header {
        if name == "__metadata__" {
            out_header.insert(name, v);
            continue;
        }
        if past(&name) {
            dropped += 1;
            continue;
        }
        let o = &v["data_offsets"];
        let (a, b) = (o[0].as_u64().unwrap() as usize, o[1].as_u64().unwrap() as usize);
        let start = out_data.len();
        out_data.extend_from_slice(&data[a..b]);
        let mut v = v.clone();
        v["data_offsets"] = json!([start, out_data.len()]);
        out_header.insert(name, v);
    }
    assert!(dropped > 0, "no tensor of a layer past {layers}");
    let mut head = serde_json::to_vec(&out_header).unwrap();
    head.resize(head.len().next_multiple_of(8), b' ');
    let mut out = (head.len() as u64).to_le_bytes().to_vec();
    out.extend(head);
    out.extend(out_data);
    out
}

fn max_diff(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len());
    a.iter().zip(b).map(|(x, y)| (x - y).abs()).fold(0f32, f32::max)
}

#[test]
fn a_cut_bundle_runs_exactly_its_first_layers() {
    for (bundle, total) in [("tiny-modernbert-bundle", 3), ("tiny-bert-bundle", 2)] {
        let src = testdata(bundle);
        let full = states(&src);
        let mut seen = vec![full];
        for layers in 1..total {
            let kept = states(&cut(&src, &format!("{bundle}-{layers}-kept"), layers, false));
            let stripped = states(&cut(&src, &format!("{bundle}-{layers}-stripped"), layers, true));
            assert_eq!(kept, stripped, "{bundle} at {layers}: a tensor past the cut was read");
            assert!(kept.iter().all(|v| v.is_finite()), "{bundle} at {layers}");
            for (k, other) in seen.iter().enumerate() {
                assert!(max_diff(&kept, other) > 1e-3, "{bundle} at {layers} gives the states of cut {k}");
            }
            seen.push(kept);
        }
    }
}
