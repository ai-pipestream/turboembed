//! The bundle form of a BiLSTM(-CRF) part-of-speech model
//! (`turbo-bundle postag-convert` and `postag-verify`).
//!
//! `ONLP-BLPT-1` to `-5` stays the interchange format. A bundle holds the
//! same model for the device backends: every tensor as F32 safetensors
//! and as an F16 copy for `FASTEST`, LSTM gate rows in `(i, f, g, o)`
//! order, the vocabularies and tags as JSON, and a manifest that holds the
//! SHA-256 of the source file and of every file written.

use crate::Result;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use std::{collections::BTreeMap, fs, path::Path};
use turbo::deberta::package::digest;
use turbo::safetensors::{Dtype, File};
use turbo_postag::neural::BilstmPosModel;
use turbo_postag::neural::bilstm_model::Vocabularies;

/// The manifest's file name in a bundle directory.
pub const MANIFEST: &str = "postag.json";
/// What the manifest's `format` names.
pub const FORMAT: &str = "turbo-postag-bilstm";
const WEIGHTS_F32: &str = "weights.safetensors";
const WEIGHTS_F16: &str = "weights-f16.safetensors";
const VOCAB: &str = "vocab.json";

/// `postag.json`.
#[derive(Debug, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct Manifest {
    pub format: String,
    pub version: u32,
    /// SHA-256 of the `ONLP-BLPT` file the bundle was converted from.
    pub source_sha256: String,
    pub source_bytes: u64,
    pub license: String,
    pub gate_order: String,
    pub crf: bool,
    pub stacked: bool,
    pub adapter: bool,
    /// SHA-256 of each file of the bundle, by name.
    pub files: BTreeMap<String, String>,
}

fn safetensors(tensors: &[(String, Vec<usize>, Vec<u8>)], dtype: &str) -> Vec<u8> {
    let mut header = BTreeMap::<String, Value>::new();
    let mut data = Vec::new();
    for (name, shape, bytes) in tensors {
        let start = data.len();
        data.extend_from_slice(bytes);
        header.insert(name.clone(), json!({"dtype": dtype, "shape": shape, "data_offsets": [start, data.len()]}));
    }
    let mut h = serde_json::to_vec(&header).expect("a JSON map of strings and numbers");
    h.resize(h.len().next_multiple_of(8), b' ');
    let mut out = (h.len() as u64).to_le_bytes().to_vec();
    out.extend(h);
    out.extend(data);
    out
}

fn vocab_json(v: &Vocabularies) -> Vec<u8> {
    serde_json::to_vec_pretty(&json!({
        "words": v.words,
        "chars": v.chars,
        "tags": v.tags,
        "pretrained": v.pretrained,
        "max_word_length": v.max_word_length,
    }))
    .expect("JSON of strings and numbers")
}

/// Converts the `ONLP-BLPT` file `bin` into a new bundle directory `out`.
pub fn convert(bin: &[u8], license: &str, out: &Path) -> Result<()> {
    let model = BilstmPosModel::load(bin).map_err(|e| format!("postag-convert: {e}"))?;
    let mut named = model.named_tensors();
    named.sort_by(|a, b| a.name.cmp(&b.name));
    let mut f32s = Vec::new();
    let mut f16s = Vec::new();
    for t in &named {
        if let Some(bad) = t.values.iter().find(|x| !(**x as f32).is_finite()) {
            return Err(format!("postag-convert: {} holds {bad}, not a finite F32", t.name));
        }
        let a: Vec<u8> = t.values.iter().flat_map(|&x| (x as f32).to_le_bytes()).collect();
        let h: Vec<u8> = t.values.iter().flat_map(|&x| half::f16::from_f64(x).to_le_bytes()).collect();
        f32s.push((t.name.clone(), t.shape.clone(), a));
        f16s.push((t.name.clone(), t.shape.clone(), h));
    }
    let files = [
        (WEIGHTS_F32, safetensors(&f32s, "F32")),
        (WEIGHTS_F16, safetensors(&f16s, "F16")),
        (VOCAB, vocab_json(&model.vocabularies())),
    ];
    let manifest = Manifest {
        format: FORMAT.into(),
        version: 1,
        source_sha256: digest(bin),
        source_bytes: bin.len() as u64,
        license: license.into(),
        gate_order: "i,f,g,o".into(),
        crf: model.is_crf(),
        stacked: model.is_stacked(),
        adapter: model.has_adapter(),
        files: files.iter().map(|(n, b)| ((*n).to_owned(), digest(b))).collect(),
    };
    fs::create_dir(out).map_err(|e| format!("{}: {e} (postag-convert writes a new directory)", out.display()))?;
    for (name, bytes) in &files {
        fs::write(out.join(name), bytes).map_err(|e| format!("{name}: {e}"))?;
    }
    let m = serde_json::to_vec_pretty(&manifest).expect("a manifest of strings and numbers");
    fs::write(out.join(MANIFEST), m).map_err(|e| format!("{MANIFEST}: {e}"))
}

fn read(dir: &Path, name: &str) -> Result<Vec<u8>> {
    fs::read(dir.join(name)).map_err(|e| format!("{}: {e}", dir.join(name).display()))
}

/// Reads the manifest and checks every file against its digest, and,
/// given the source file, that the bundle was converted from it.
pub fn verify(dir: &Path, source: Option<&[u8]>) -> Result<Manifest> {
    let m: Manifest = serde_json::from_slice(&read(dir, MANIFEST)?).map_err(|e| format!("{MANIFEST}: {e}"))?;
    if m.format != FORMAT || m.version != 1 || m.gate_order != "i,f,g,o" {
        return Err(format!("{MANIFEST}: not a {FORMAT} version 1 bundle with (i, f, g, o) gates"));
    }
    let want: Vec<&str> = vec![VOCAB, WEIGHTS_F16, WEIGHTS_F32];
    if m.files.keys().map(String::as_str).collect::<Vec<_>>() != want {
        return Err(format!("{MANIFEST}: files names {:?}, not {want:?}", m.files.keys()));
    }
    for (name, sha) in &m.files {
        let got = digest(&read(dir, name)?);
        if &got != sha {
            return Err(format!("{name}: SHA-256 {got}, the manifest says {sha}"));
        }
    }
    if let Some(src) = source {
        let got = digest(src);
        if got != m.source_sha256 || src.len() as u64 != m.source_bytes {
            return Err(format!(
                "the source file's SHA-256 is {got}; the bundle was converted from {}",
                m.source_sha256
            ));
        }
    }
    Ok(m)
}

/// The model a verified bundle holds, from its F32 weights, or from the
/// F16 copy when `f16`.
pub fn load(dir: &Path, f16: bool) -> Result<BilstmPosModel> {
    verify(dir, None)?;
    #[derive(Deserialize)]
    struct V {
        words: Vec<(String, usize)>,
        chars: Vec<(Vec<u8>, usize)>,
        tags: Vec<String>,
        pretrained: Vec<(String, usize)>,
        max_word_length: usize,
    }
    let v: V = serde_json::from_slice(&read(dir, VOCAB)?).map_err(|e| format!("{VOCAB}: {e}"))?;
    let name = if f16 { WEIGHTS_F16 } else { WEIGHTS_F32 };
    let bytes = read(dir, name)?;
    let file = File::parse(name, &bytes).map_err(|e| e.message)?;
    let dtype = if f16 { Dtype::F16 } else { Dtype::F32 };
    let mut bad = None;
    let vocab = Vocabularies {
        words: v.words,
        chars: v.chars,
        tags: v.tags,
        pretrained: v.pretrained,
        max_word_length: v.max_word_length,
    };
    let model = BilstmPosModel::from_named(vocab, |key| {
        let t = file.tensor(key)?;
        if t.dtype != dtype {
            bad = Some(format!("{name}: {key} is {}, not {}", t.dtype_name, dtype.name()));
            return None;
        }
        let values = if f16 {
            t.data.as_chunks::<2>().0.iter().map(|b| half::f16::from_le_bytes(*b).to_f64()).collect()
        } else {
            t.f32s().into_iter().map(f64::from).collect()
        };
        Some((t.shape.iter().map(|&d| d as usize).collect(), values))
    });
    if let Some(b) = bad {
        return Err(b);
    }
    model.map_err(|e| format!("{}: {e}", dir.display()))
}
