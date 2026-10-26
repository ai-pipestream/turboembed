//! A bundle directory built on disk for each test from the upstream
//! all-MiniLM-L6-v2 tokenizer.json in testdata/, and the C calls the tests
//! make on it.
//!
//! The reference file's ids come from upstream `tokenizers`, the library
//! sentence-transformers tokenizes with, on the same tokenizer.json. Its
//! embeddings are zeros: nothing up to turbo_tokenizer_create reads them.
//! Vectors are checked against the sealed bundle in
//! testdata/tiny-bert-bundle, whose reference is upstream's output
//! (tests/conformance.rs), and against a plain f64 encoder written below.

#![allow(dead_code)]

use std::fs;
use std::path::{Path, PathBuf};
use std::ptr;

use serde_json::{Value, json};
use turbo::*;

pub mod conformance;

pub const MAX_SEQ: usize = 256;

pub fn testdata() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../testdata")
}

pub fn upstream_tokenizer_json() -> PathBuf {
    testdata().join("all-minilm-l6-v2/tokenizer.json")
}

/// The upstream tokenizer as sentence-transformers runs it for this model:
/// the file's own padding and truncation off, cut on the right at max_seq.
pub fn upstream() -> tokenizers::Tokenizer {
    upstream_at(&upstream_tokenizer_json())
}

/// As `upstream`, on the tokenizer file at `path`.
pub fn upstream_at(path: &Path) -> tokenizers::Tokenizer {
    let mut t = tokenizers::Tokenizer::from_file(path).expect("upstream tokenizer.json");
    t.with_padding(None);
    t.with_truncation(Some(tokenizers::TruncationParams {
        max_length: MAX_SEQ,
        strategy: tokenizers::TruncationStrategy::LongestFirst,
        direction: tokenizers::TruncationDirection::Right,
        stride: 0,
    }))
    .unwrap();
    t
}

pub fn upstream_ids(t: &tokenizers::Tokenizer, text: &str) -> Vec<i32> {
    t.encode(text, true).expect("upstream encode").get_ids().iter().map(|&i| i as i32).collect()
}

/// About 120 tokens.
pub const PARAGRAPH: &str = "Embedding models turn a passage of text into a fixed-length vector so that \
passages with similar meaning land near each other. A search system embeds every document once, stores \
the vectors in an index, and embeds each query as it arrives; the nearest documents by cosine similarity \
are the candidates it returns. The model reads at most a fixed number of tokens, so long documents are \
split into chunks first, and each chunk gets its own vector. Pooling averages the token states into one \
vector, and normalization scales it to unit length.";

pub fn long_text() -> String {
    [PARAGRAPH, PARAGRAPH, PARAGRAPH].join(" ")
}

/// The cases of docs/bundle.md's example, with its two placeholders filled.
pub fn cases() -> Value {
    json!([
        { "text": "", "prompt_role": "PROMPT_NONE" },
        { "text": "The quick brown fox jumps over the lazy dog.", "prompt_role": "PROMPT_NONE" },
        { "text": "Café naïve RÉSUMÉ", "prompt_role": "PROMPT_NONE" },
        { "text": "东京是日本的首都。", "prompt_role": "PROMPT_NONE" },
        { "text": "emoji 🙂 and tabs\tand\nnewlines", "prompt_role": "PROMPT_NONE" },
        { "text": "how do I reset a password", "prompt_role": "PROMPT_QUERY" },
        { "text": "To reset a password, open Settings and choose Security.", "prompt_role": "PROMPT_DOCUMENT" },
        { "text": PARAGRAPH, "prompt_role": "PROMPT_NONE" },
        { "text": long_text(), "prompt_role": "PROMPT_NONE" }
    ])
}

/// The manifest of docs/bundle.md cut to what these files are: the
/// tokenizer, the reference, and a weights artifact that is listed but
/// absent (nothing up to turbo_tokenizer_create opens it).
pub fn manifest() -> Value {
    json!({
      "bundle_version": 1,
      "model": {
        "id": "sentence-transformers/all-MiniLM-L6-v2",
        "revision": "3",
        "source": {
          "repository": "https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2",
          "commit": "c9745ed1d9f207416be6d2e6f8de32d1f16199bf"
        },
        "license": "Apache-2.0"
      },
      "task": "TASK_EMBED",
      "embed": {
        "dim": 384,
        "pooling": "POOLING_MEAN",
        "normalize": "NORMALIZE_L2",
        "max_seq": MAX_SEQ,
        "max_batch": 64,
        "prefix_query": "",
        "prefix_document": "",
        "output_dims": []
      },
      "tokenizer": {
        "file": "tokenizer.json",
        "normalizer": {
          "clean_text": true,
          "lowercase": true,
          "strip_accents": true,
          "split_cjk": true,
          "unicode_form": "UNICODE_NONE"
        },
        "wordpiece": { "continuing_prefix": "##", "max_chars_per_word": 100 },
        "special_tokens": [
          { "role": "SPECIAL_PAD",  "content": "[PAD]",  "id": 0 },
          { "role": "SPECIAL_UNK",  "content": "[UNK]",  "id": 100 },
          { "role": "SPECIAL_BOS",  "content": "[CLS]",  "id": 101 },
          { "role": "SPECIAL_EOS",  "content": "[SEP]",  "id": 102 },
          { "role": "SPECIAL_MASK", "content": "[MASK]", "id": 103 }
        ],
        "template": ["[CLS]", "$TEXT", "[SEP]"],
        "truncation": "TRUNCATE_RIGHT"
      },
      "architecture": {
        "family": "FAMILY_BERT",
        "layers": 6,
        "hidden": 384,
        "heads": 12,
        "intermediate": 1536,
        "activation": "ACTIVATION_GELU_ERF",
        "layer_norm_eps": 1e-12,
        "position_embedding": "POSITION_ABSOLUTE",
        "max_positions": 512,
        "token_types": 2,
        "vocab_size": 30522
      },
      "artifacts": [
        {
          "name": "weights-f32",
          "format": "FORMAT_SAFETENSORS",
          "files": ["weights/model.safetensors"],
          "backends": ["cuda", "metal"],
          "graph_input": "INPUT_TOKEN_IDS",
          "graph_output": "OUTPUT_HIDDEN_STATES",
          "tensor_names": {
            "word_embeddings": "embeddings.word_embeddings.weight",
            "position_embeddings": "embeddings.position_embeddings.weight",
            "token_type_embeddings": "embeddings.token_type_embeddings.weight",
            "embeddings_ln_weight": "embeddings.LayerNorm.weight",
            "embeddings_ln_bias": "embeddings.LayerNorm.bias",
            "q_weight": "encoder.layer.{layer}.attention.self.query.weight",
            "q_bias": "encoder.layer.{layer}.attention.self.query.bias",
            "k_weight": "encoder.layer.{layer}.attention.self.key.weight",
            "k_bias": "encoder.layer.{layer}.attention.self.key.bias",
            "v_weight": "encoder.layer.{layer}.attention.self.value.weight",
            "v_bias": "encoder.layer.{layer}.attention.self.value.bias",
            "attn_out_weight": "encoder.layer.{layer}.attention.output.dense.weight",
            "attn_out_bias": "encoder.layer.{layer}.attention.output.dense.bias",
            "attn_ln_weight": "encoder.layer.{layer}.attention.output.LayerNorm.weight",
            "attn_ln_bias": "encoder.layer.{layer}.attention.output.LayerNorm.bias",
            "ffn_in_weight": "encoder.layer.{layer}.intermediate.dense.weight",
            "ffn_in_bias": "encoder.layer.{layer}.intermediate.dense.bias",
            "ffn_out_weight": "encoder.layer.{layer}.output.dense.weight",
            "ffn_out_bias": "encoder.layer.{layer}.output.dense.bias",
            "ffn_ln_weight": "encoder.layer.{layer}.output.LayerNorm.weight",
            "ffn_ln_bias": "encoder.layer.{layer}.output.LayerNorm.bias"
          }
        }
      ],
      "reference": {
        "file": "reference/reference.safetensors",
        "cases": cases(),
        "produced_by": {
          "tool": "tokenizers",
          "tool_version": "0.22",
          "container": "none: built by the test from the upstream tokenizer.json",
          "args": [],
          "reproducible": true
        }
      },
      "files": [
        // Listed so the artifact is well formed; absent on disk.
        { "path": "weights/model.safetensors", "size": 90868376, "sha256": "0".repeat(64) }
      ]
    })
}

pub fn sha256_hex(bytes: &[u8]) -> String {
    turbo::bundle::sha256_hex(bytes)
}

/// A safetensors file with ids [n, L], lengths [n] and embeddings [n, dim].
pub fn reference_file(ids: &[Vec<i32>], width: usize, pad: i32, dim: usize) -> Vec<u8> {
    let n = ids.len();
    let mut id_bytes = Vec::new();
    for row in ids {
        for p in 0..width {
            id_bytes.extend_from_slice(&row.get(p).copied().unwrap_or(pad).to_le_bytes());
        }
    }
    let len_bytes: Vec<u8> = ids.iter().flat_map(|r| (r.len() as i32).to_le_bytes()).collect();
    let emb_bytes = vec![0u8; n * dim * 4];
    let a = id_bytes.len();
    let b = a + len_bytes.len();
    let c = b + emb_bytes.len();
    let header = json!({
        "ids": { "dtype": "I32", "shape": [n, width], "data_offsets": [0, a] },
        "lengths": { "dtype": "I32", "shape": [n], "data_offsets": [a, b] },
        "embeddings": { "dtype": "F32", "shape": [n, dim], "data_offsets": [b, c] }
    });
    let mut h = serde_json::to_vec(&header).unwrap();
    while !h.len().is_multiple_of(8) {
        h.push(b' ');
    }
    let mut out = (h.len() as u64).to_le_bytes().to_vec();
    out.extend(h);
    out.extend(id_bytes);
    out.extend(len_bytes);
    out.extend(emb_bytes);
    out
}

/// Upstream's ids for each case of `m`, with its prefixes applied and cut
/// on the side its tokenizer.truncation names.
pub fn reference_ids(m: &Value) -> Vec<Vec<i32>> {
    reference_ids_at(m, &upstream_tokenizer_json())
}

/// As `reference_ids`, with the tokenizer file at `path`.
pub fn reference_ids_at(m: &Value, path: &Path) -> Vec<Vec<i32>> {
    let mut up = upstream_at(path);
    let mut t = up.get_truncation().unwrap().clone();
    // Cut where the manifest's max_seq says, MAX_SEQ in the standard one.
    t.max_length = m["embed"]["max_seq"].as_u64().map_or(MAX_SEQ, |n| n as usize);
    if m["tokenizer"]["truncation"] == "TRUNCATE_LEFT" {
        t.direction = tokenizers::TruncationDirection::Left;
    }
    up.with_truncation(Some(t)).unwrap();
    let prefix = |role: &str| match role {
        "PROMPT_QUERY" => m["embed"]["prefix_query"].as_str().unwrap_or_default().to_owned(),
        "PROMPT_DOCUMENT" => m["embed"]["prefix_document"].as_str().unwrap_or_default().to_owned(),
        _ => String::new(),
    };
    m["reference"]["cases"]
        .as_array()
        .unwrap()
        .iter()
        .map(|c| {
            upstream_ids(&up, &format!("{}{}", prefix(c["prompt_role"].as_str().unwrap()), c["text"].as_str().unwrap()))
        })
        .collect()
}

pub struct Fixture {
    pub dir: PathBuf,
    pub manifest: Value,
}

impl Fixture {
    /// A fresh directory for `name`, with the tokenizer and a reference
    /// file made from `manifest`'s cases.
    pub fn new(name: &str, manifest: Value) -> Fixture {
        Fixture::with_tokenizer(name, manifest, &upstream_tokenizer_json())
    }

    /// As `new`, with the tokenizer file at `tokenizer_json`, whose ids
    /// the reference file carries, padded with the manifest's pad id.
    pub fn with_tokenizer(name: &str, manifest: Value, tokenizer_json: &Path) -> Fixture {
        let dir = std::env::temp_dir().join(format!("turbo-test-{}-{name}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(dir.join("reference")).unwrap();
        fs::copy(tokenizer_json, dir.join("tokenizer.json")).unwrap();
        let ids = reference_ids_at(&manifest, tokenizer_json);
        let width = ids.iter().map(Vec::len).max().unwrap_or(1).max(1);
        let dim = manifest["embed"]["dim"].as_u64().unwrap() as usize;
        let pad = manifest["tokenizer"]["special_tokens"]
            .as_array()
            .unwrap()
            .iter()
            .find(|s| s["role"] == "SPECIAL_PAD")
            .map_or(-1, |s| s["id"].as_i64().unwrap() as i32);
        fs::write(dir.join("reference/reference.safetensors"), reference_file(&ids, width, pad, dim)).unwrap();
        let mut f = Fixture { dir, manifest };
        f.list("tokenizer.json");
        f.list("reference/reference.safetensors");
        f
    }

    pub fn standard(name: &str) -> Fixture {
        Fixture::new(name, manifest())
    }

    /// The small BERT of `model_manifest`, with its weights on disk.
    pub fn model(name: &str) -> Fixture {
        let mut f = Fixture::new(name, model_manifest());
        f.weights("weights/model.safetensors", &tiny_weights(0));
        f
    }

    /// Write `tensors` as a safetensors file at `path` and list it.
    pub fn weights(&mut self, path: &str, tensors: &[Tensor]) {
        let p = self.dir.join(path);
        fs::create_dir_all(p.parent().unwrap()).unwrap();
        fs::write(&p, safetensors_file(tensors)).unwrap();
        self.list(path);
    }

    /// The hash files[] gives `path`.
    pub fn sha256(&self, path: &str) -> String {
        let files = self.manifest["files"].as_array().unwrap();
        files.iter().find(|f| f["path"] == path).unwrap()["sha256"].as_str().unwrap().to_owned()
    }

    pub fn load(&self) -> Result<Loaded, Failure> {
        self.write();
        Loaded::load(&self.dir)
    }

    /// As load, on the device `device` picks.
    pub fn load_on(&self, device: impl Fn(*mut turbo_runtime) -> u32) -> Result<Loaded, Failure> {
        self.write();
        Loaded::load_on(&self.dir, device)
    }

    /// Put `path`'s size and hash in files[], replacing any entry for it.
    pub fn list(&mut self, path: &str) {
        let bytes = fs::read(self.dir.join(path)).unwrap();
        let files = self.manifest["files"].as_array_mut().unwrap();
        files.retain(|f| f["path"] != path);
        files.push(json!({ "path": path, "size": bytes.len(), "sha256": sha256_hex(&bytes) }));
    }

    pub fn write(&self) -> &Path {
        fs::write(self.dir.join("manifest.json"), serde_json::to_vec_pretty(&self.manifest).unwrap()).unwrap();
        &self.dir
    }

    pub fn open(&self) -> Result<Tok, Failure> {
        self.write();
        Tok::create(&self.dir)
    }
}

impl Drop for Fixture {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.dir);
    }
}

#[derive(Debug)]
pub struct Failure {
    pub code: i32,
    pub field: u32,
    pub message: String,
}

impl Failure {
    pub fn is(&self, code: i32, containing: &str) -> bool {
        self.code == code && self.message.contains(containing)
    }
}

pub fn new_error() -> turbo_error {
    turbo_error { struct_size: size_of::<turbo_error>() as u32, code: -1, field: 0, message: [0; 496] }
}

pub fn failure(code: i32, e: &turbo_error) -> Failure {
    let bytes: Vec<u8> =
        e.message.iter().take_while(|&&c| c != 0).map(|&c| u8::from_ne_bytes(c.to_ne_bytes())).collect();
    assert_eq!(e.code, code, "turbo_error.code and the returned status differ");
    Failure { code, field: e.field, message: String::from_utf8(bytes).expect("message is UTF-8") }
}

pub fn text(s: &str) -> turbo_text {
    turbo_text { ptr: s.as_ptr() as *const _, len: s.len() as u64 }
}

/// A tokenizer made through the C interface.
#[derive(Debug)]
pub struct Tok {
    pub rt: *mut turbo_runtime,
    pub t: *mut turbo_tokenizer,
}

impl Tok {
    pub fn create(dir: &Path) -> Result<Tok, Failure> {
        let mut err = new_error();
        let mut rt = ptr::null_mut();
        let rc = unsafe { turbo_runtime_create(ptr::null(), &mut rt, &mut err) };
        assert_eq!(rc, 0, "turbo_runtime_create: {:?}", failure(rc, &err));
        let path = dir.to_str().unwrap();
        let mut t = ptr::null_mut();
        let rc = unsafe { turbo_tokenizer_create(rt, text(path), &mut t, &mut err) };
        if rc != 0 {
            unsafe { turbo_runtime_release(rt) };
            return Err(failure(rc, &err));
        }
        Ok(Tok { rt, t })
    }

    pub fn info(&self) -> turbo_tokenizer_info {
        let mut info: turbo_tokenizer_info = unsafe { std::mem::zeroed() };
        info.struct_size = size_of::<turbo_tokenizer_info>() as u32;
        let mut err = new_error();
        let rc = unsafe { turbo_tokenizer_get_info(self.t, &mut info, &mut err) };
        assert_eq!(rc, 0, "{:?}", failure(rc, &err));
        info
    }

    /// Encoded rows cut to their lengths, with the full padded rows and masks.
    pub fn encode(&self, texts: &[&str], opts: Option<&turbo_encode_options>, stride: u32) -> Result<Encoded, Failure> {
        let views: Vec<turbo_text> = texts.iter().map(|s| text(s)).collect();
        let n = texts.len() * stride as usize;
        let mut out = Encoded {
            ids: vec![-7; n],
            mask: vec![-7; n],
            types: vec![-7; n],
            lengths: vec![0; texts.len()],
            stride: stride as usize,
        };
        let mut err = new_error();
        let rc = unsafe {
            turbo_tokenizer_encode(
                self.t,
                views.as_ptr(),
                views.len() as u32,
                opts.map_or(ptr::null(), |o| o as *const _),
                out.ids.as_mut_ptr(),
                out.mask.as_mut_ptr(),
                out.types.as_mut_ptr(),
                stride,
                out.lengths.as_mut_ptr(),
                &mut err,
            )
        };
        if rc != 0 {
            return Err(failure(rc, &err));
        }
        Ok(out)
    }

    pub fn row(&self, text: &str, opts: Option<&turbo_encode_options>) -> Result<Vec<i32>, Failure> {
        // A row no text can overrun: a token is at least a byte, and the
        // prefix a role adds is short.
        Ok(self.encode(&[text], opts, (text.len() + 1024) as u32)?.row(0))
    }
}

impl Drop for Tok {
    fn drop(&mut self) {
        unsafe {
            turbo_tokenizer_release(self.t);
            turbo_runtime_release(self.rt);
        }
    }
}

#[derive(Debug)]
pub struct Encoded {
    pub ids: Vec<i32>,
    pub mask: Vec<i32>,
    pub types: Vec<i32>,
    pub lengths: Vec<u32>,
    pub stride: usize,
}

impl Encoded {
    pub fn row(&self, i: usize) -> Vec<i32> {
        self.ids[i * self.stride..i * self.stride + self.lengths[i] as usize].to_vec()
    }
}

include!("model.rs");
include!("session.rs");
include!("plain_bert.rs");
