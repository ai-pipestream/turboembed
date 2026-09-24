//! A bundle directory built on disk for each test from the upstream
//! all-MiniLM-L6-v2 tokenizer.json in testdata/, and the C calls the tests
//! make on it.
//!
//! The reference file's ids come from upstream `tokenizers`, the library
//! sentence-transformers tokenizes with, on the same tokenizer.json. Its
//! embeddings are zeros: nothing up to turbo_tokenizer_create reads them,
//! and no model output is available to this build. A bundle with real
//! reference vectors is tested through TURBO_TEST_BUNDLE (tests/bundle.rs).

#![allow(dead_code)]

use std::fs;
use std::path::{Path, PathBuf};
use std::ptr;

use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use turbo::*;

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
    let mut t = tokenizers::Tokenizer::from_file(upstream_tokenizer_json()).expect("upstream tokenizer.json");
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
    Sha256::digest(bytes).iter().map(|b| format!("{b:02x}")).collect()
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
    let mut up = upstream();
    if m["tokenizer"]["truncation"] == "TRUNCATE_LEFT" {
        let mut t = up.get_truncation().unwrap().clone();
        t.direction = tokenizers::TruncationDirection::Left;
        up.with_truncation(Some(t)).unwrap();
    }
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
        let dir = std::env::temp_dir().join(format!("turbo-test-{}-{name}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(dir.join("reference")).unwrap();
        fs::copy(upstream_tokenizer_json(), dir.join("tokenizer.json")).unwrap();
        let ids = reference_ids(&manifest);
        let width = ids.iter().map(Vec::len).max().unwrap_or(1).max(1);
        let dim = manifest["embed"]["dim"].as_u64().unwrap() as usize;
        fs::write(dir.join("reference/reference.safetensors"), reference_file(&ids, width, 0, dim)).unwrap();
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
    let bytes: Vec<u8> = e.message.iter().take_while(|&&c| c != 0).map(|&c| c as u8).collect();
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
        Ok(self.encode(&[text], opts, 1024)?.row(0))
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

pub fn options(omit_special_tokens: u32, truncate: u32, max_tokens: u32, prompt_role: u32) -> turbo_encode_options {
    turbo_encode_options {
        struct_size: size_of::<turbo_encode_options>() as u32,
        omit_special_tokens,
        truncate,
        max_tokens,
        prompt_role,
    }
}

/// The texts in testdata/tokenizer-texts.jsonl.
pub fn parity_texts() -> Vec<String> {
    fs::read_to_string(testdata().join("tokenizer-texts.jsonl"))
        .unwrap()
        .lines()
        .map(|l| serde_json::from_str::<Value>(l).unwrap()["text"].as_str().unwrap().to_owned())
        .collect()
}

// ---- A small BERT --------------------------------------------------------

/// Two layers, hidden 8, two heads, intermediate 16, over the upstream
/// vocabulary: the shapes of a BERT encoder at a size a test writes in
/// a moment.
pub fn tiny_architecture() -> Value {
    json!({
        "family": "FAMILY_BERT",
        "layers": 2,
        "hidden": 8,
        "heads": 2,
        "intermediate": 16,
        "activation": "ACTIVATION_GELU_ERF",
        "layer_norm_eps": 1e-12,
        "position_embedding": "POSITION_ABSOLUTE",
        "max_positions": 512,
        "token_types": 2,
        "vocab_size": 30522
    })
}

/// The standard manifest for the small BERT, loadable on the CPU: its
/// architecture, a dim of its hidden width, and cpu among the weights
/// artifact's backends, as the MiniLM recipe has it.
pub fn model_manifest() -> Value {
    let mut m = manifest();
    m["architecture"] = tiny_architecture();
    m["embed"]["dim"] = json!(8);
    m["artifacts"][0]["backends"] = json!(["cuda", "metal", "cpu"]);
    m["files"] = json!([]);
    m
}

pub struct Tensor {
    pub name: String,
    pub dtype: &'static str,
    pub shape: Vec<u64>,
    pub data: Vec<u8>,
}

/// Every tensor of the small BERT under the upstream names, F32, with
/// values that depend on `seed` and the element's place and nothing else.
pub fn tiny_weights(seed: u32) -> Vec<Tensor> {
    let (h, i, v, p, t) = (8u64, 16u64, 30522u64, 512u64, 2u64);
    let mut shapes: Vec<(String, Vec<u64>)> = vec![
        ("embeddings.word_embeddings.weight".into(), vec![v, h]),
        ("embeddings.position_embeddings.weight".into(), vec![p, h]),
        ("embeddings.token_type_embeddings.weight".into(), vec![t, h]),
        ("embeddings.LayerNorm.weight".into(), vec![h]),
        ("embeddings.LayerNorm.bias".into(), vec![h]),
    ];
    for l in 0..2 {
        let at = |s: &str| format!("encoder.layer.{l}.{s}");
        for (s, shape) in [
            ("attention.self.query.weight", vec![h, h]),
            ("attention.self.query.bias", vec![h]),
            ("attention.self.key.weight", vec![h, h]),
            ("attention.self.key.bias", vec![h]),
            ("attention.self.value.weight", vec![h, h]),
            ("attention.self.value.bias", vec![h]),
            ("attention.output.dense.weight", vec![h, h]),
            ("attention.output.dense.bias", vec![h]),
            ("attention.output.LayerNorm.weight", vec![h]),
            ("attention.output.LayerNorm.bias", vec![h]),
            ("intermediate.dense.weight", vec![i, h]),
            ("intermediate.dense.bias", vec![i]),
            ("output.dense.weight", vec![h, i]),
            ("output.dense.bias", vec![h]),
            ("output.LayerNorm.weight", vec![h]),
            ("output.LayerNorm.bias", vec![h]),
        ] {
            shapes.push((at(s), shape));
        }
    }
    // A tensor upstream carries that a BERT encoder does not use.
    shapes.push(("pooler.dense.bias".into(), vec![h]));
    let mut k = 0u64;
    shapes
        .into_iter()
        .map(|(name, shape)| {
            let n: u64 = shape.iter().product();
            let data = (0..n)
                .flat_map(|_| {
                    k += 1;
                    let x = ((k * 7919 + seed as u64 * 104729) % 2001) as f32 / 1000.0 - 1.0;
                    x.to_le_bytes()
                })
                .collect();
            Tensor { name, dtype: "F32", shape, data }
        })
        .collect()
}

/// A safetensors file of `tensors`, in the order given.
pub fn safetensors_file(tensors: &[Tensor]) -> Vec<u8> {
    let mut header = serde_json::Map::new();
    header.insert("__metadata__".into(), json!({ "format": "pt" }));
    let mut at = 0usize;
    for t in tensors {
        let end = at + t.data.len();
        header.insert(t.name.clone(), json!({ "dtype": t.dtype, "shape": t.shape, "data_offsets": [at, end] }));
        at = end;
    }
    let mut h = serde_json::to_vec(&header).unwrap();
    while !h.len().is_multiple_of(8) {
        h.push(b' ');
    }
    let mut out = (h.len() as u64).to_le_bytes().to_vec();
    out.extend(h);
    for t in tensors {
        out.extend(&t.data);
    }
    out
}

/// A model loaded through the C interface on the CPU, with the runtime and
/// context it was loaded with.
pub struct Loaded {
    pub rt: *mut turbo_runtime,
    pub ctx: *mut turbo_context,
    pub m: *mut turbo_model,
}

impl Loaded {
    pub fn load(dir: &Path) -> Result<Loaded, Failure> {
        let mut err = new_error();
        let mut rt = ptr::null_mut();
        assert_eq!(unsafe { turbo_runtime_create(ptr::null(), &mut rt, &mut err) }, 0);
        let mut ctx = ptr::null_mut();
        let rc = unsafe { turbo_context_create(rt, cpu(rt), &mut ctx, &mut err) };
        assert_eq!(rc, 0, "{:?}", failure(rc, &err));
        let mut m = ptr::null_mut();
        let path = dir.to_str().unwrap();
        let rc = unsafe { turbo_model_load(ctx, text(path), &mut m, &mut err) };
        let loaded = Loaded { rt, ctx, m };
        if rc != 0 {
            assert!(m.is_null(), "a failed load wrote out");
            return Err(failure(rc, &err));
        }
        Ok(loaded)
    }

    /// The handles, which the caller now releases.
    pub fn into_raw(self) -> (*mut turbo_runtime, *mut turbo_context, *mut turbo_model) {
        let l = std::mem::ManuallyDrop::new(self);
        (l.rt, l.ctx, l.m)
    }

    /// A model handle alone, released on drop.
    pub fn model(m: *mut turbo_model) -> Loaded {
        Loaded { rt: ptr::null_mut(), ctx: ptr::null_mut(), m }
    }

    pub fn info(&self) -> turbo_model_info {
        let mut info: turbo_model_info = unsafe { std::mem::zeroed() };
        info.struct_size = size_of::<turbo_model_info>() as u32;
        let mut err = new_error();
        let rc = unsafe { turbo_model_get_info(self.m, &mut info, &mut err) };
        assert_eq!(rc, 0, "{:?}", failure(rc, &err));
        info
    }
}

impl Drop for Loaded {
    fn drop(&mut self) {
        unsafe {
            turbo_model_release(self.m);
            turbo_context_release(self.ctx);
            turbo_runtime_release(self.rt);
        }
    }
}

/// The index of the device the cpu backend lists.
pub fn cpu(rt: *mut turbo_runtime) -> u32 {
    let mut n = 0;
    assert_eq!(unsafe { turbo_runtime_device_count(rt, &mut n, ptr::null_mut()) }, 0);
    (0..n)
        .find(|&i| {
            let mut info: turbo_device_info = unsafe { std::mem::zeroed() };
            info.struct_size = size_of::<turbo_device_info>() as u32;
            assert_eq!(unsafe { turbo_runtime_device_info(rt, i, &mut info, ptr::null_mut()) }, 0);
            info.kind == TURBO_DEVICE_CPU
        })
        .expect("the cpu backend lists the host")
}

/// A fixed-size C string field as a String.
pub fn field(b: &[std::ffi::c_char]) -> String {
    let bytes: Vec<u8> = b.iter().take_while(|&&c| c != 0).map(|&c| c as u8).collect();
    String::from_utf8(bytes).unwrap()
}
