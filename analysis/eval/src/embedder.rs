//! The embedder a pipeline runs: one vector per text, and the model's
//! own subword pieces for the chunkers that count tokens.
//!
//! [`Embedder`] is what the pipeline and the chunkers see. The real one,
//! [`TurboEmbedder`], tokenizes with the bundle's tokenizer and runs the
//! rows through a turbo session (`turbo-analysis-embed`'s `Encoder`); the
//! tests run a stub behind the same trait.

use std::ffi::c_char;
use std::path::Path;

use turbo::bundle::Bundle;
use turbo::manifest::{PromptRole, Truncation};
use turbo::record::{BundleId, Device, Machine};
use turbo::tokenizer::{Encode, Tokenizer};
use turbo::{TURBO_DEVICE_CPU, TURBO_DEVICE_GPU, TURBO_DEVICE_IGPU, TURBO_DEVICE_NPU, TURBO_POOLING_MEAN};
use turbo_analysis::{ModelRef, Span, Spans, ViewId};
use turbo_analysis_embed::{Encoder, EncoderOptions, Precision, Rows};
use turbo_analysis_layers::Subwords;

use crate::Result;

/// Which side of retrieval a text is on: the bundle's prompt prefix for
/// it, when it has one, goes in front of the text.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum Role {
    Query,
    Document,
}

impl Role {
    fn prompt(self) -> PromptRole {
        match self {
            Role::Query => PromptRole::Query,
            Role::Document => PromptRole::Document,
        }
    }
}

/// Where the embedder runs and which model it is, as a record names them.
#[derive(Clone, Debug, PartialEq)]
pub struct Identity {
    pub machine: Machine,
    pub device: Device,
    pub bundle: BundleId,
}

impl Identity {
    /// The model as an analysis layer names it: id, revision and the
    /// SHA-256 of the manifest.
    pub fn model_ref(&self) -> ModelRef {
        ModelRef {
            id: self.bundle.model_id.clone(),
            revision: self.bundle.revision.clone(),
            sha256: self.bundle.manifest_sha256.clone(),
        }
    }
}

pub trait Embedder {
    fn identity(&self) -> &Identity;

    /// Values per vector.
    fn dim(&self) -> usize;

    /// The most tokens a row holds, the template's special tokens and the
    /// prompt prefix included. A longer text is cut as the bundle cuts it.
    fn max_tokens(&self) -> u32;

    /// Tokens a row of `role` carries besides the text's own pieces.
    fn overhead(&self, role: Role) -> u32;

    /// The text's own pieces with their byte spans, in the source view,
    /// with no special tokens and no prefix, never cut.
    fn subwords(&self, text: &str) -> Result<Subwords>;

    /// The tokens a row of `text` would hold before any cut: its pieces
    /// and the overhead.
    fn count(&self, text: &str, role: Role) -> u64;

    /// One unit length vector per text into `out`, `[texts.len(), dim]`
    /// values; a text with no direction gets zeros. Returns the tokens
    /// the model was given, counted on the rows as they were run.
    fn embed(&mut self, texts: &[&str], role: Role, out: &mut Vec<f32>) -> Result<u64>;
}

/// An embedder that counts what it is given, so every stage's tokens add
/// up in one place.
pub(crate) struct Counted<'a> {
    pub inner: &'a mut dyn Embedder,
    pub tokens: u64,
}

impl Embedder for Counted<'_> {
    fn identity(&self) -> &Identity {
        self.inner.identity()
    }
    fn dim(&self) -> usize {
        self.inner.dim()
    }
    fn max_tokens(&self) -> u32 {
        self.inner.max_tokens()
    }
    fn overhead(&self, role: Role) -> u32 {
        self.inner.overhead(role)
    }
    fn subwords(&self, text: &str) -> Result<Subwords> {
        self.inner.subwords(text)
    }
    fn count(&self, text: &str, role: Role) -> u64 {
        self.inner.count(text, role)
    }
    fn embed(&mut self, texts: &[&str], role: Role, out: &mut Vec<f32>) -> Result<u64> {
        let n = self.inner.embed(texts, role, out)?;
        self.tokens += n;
        Ok(n)
    }
}

/// Scales `v` to unit length, summing in f64; a zero vector stays zero.
pub fn unit(v: &mut [f32]) {
    let norm = v.iter().map(|&x| f64::from(x) * f64::from(x)).sum::<f64>().sqrt();
    if norm > 0.0 {
        let inv = (1.0 / norm) as f32;
        v.iter_mut().for_each(|x| *x *= inv);
    }
}

/// A bundle run through a turbo session, mean pooled, as the reference
/// embedder (all-MiniLM-L6-v2, the model turbo-bench measures) is.
pub struct TurboEmbedder {
    encoder: Encoder,
    tokenizer: Tokenizer,
    identity: Identity,
    rows: Rows,
    /// One row's ids per text, kept between calls.
    ids: Vec<Vec<i32>>,
    vectors: Vec<f32>,
}

/// A fixed-size C string field as a String.
fn field(b: &[c_char]) -> String {
    let bytes: Vec<u8> = b.iter().take_while(|&&c| c != 0).map(|&c| u8::from_ne_bytes(c.to_ne_bytes())).collect();
    String::from_utf8_lossy(&bytes).into_owned()
}

fn kind_name(kind: u32) -> &'static str {
    match kind {
        TURBO_DEVICE_CPU => "DEVICE_CPU",
        TURBO_DEVICE_GPU => "DEVICE_GPU",
        TURBO_DEVICE_IGPU => "DEVICE_IGPU",
        TURBO_DEVICE_NPU => "DEVICE_NPU",
        _ => "",
    }
}

impl TurboEmbedder {
    /// The bundle at `dir` on `device`. A bundle that does not mean pool
    /// is refused: the session's row write here pools by mean.
    pub fn open(dir: &Path, device: turbo_analysis_embed::Device, precision: Precision) -> Result<TurboEmbedder> {
        let at = |e: turbo::status::Error| format!("{}: {e}", dir.display());
        let bundle = Bundle::open(dir).map_err(at)?;
        let tokenizer = Tokenizer::load(&bundle).map_err(at)?;
        let encoder =
            Encoder::open(dir, &EncoderOptions { device, precision, ..EncoderOptions::default() }).map_err(at)?;
        if encoder.info().pooling != TURBO_POOLING_MEAN {
            return Err(format!(
                "{}: the bundle pools with TURBO_POOLING {}; the harness embeds by mean pooling only",
                dir.display(),
                encoder.info().pooling
            ));
        }
        let m = encoder.model_info();
        let d = encoder.device_info();
        let host_cpu = encoder.host_cpu().map_err(at)?.map(|c| field(&c.name)).unwrap_or_default();
        let identity = Identity {
            machine: Machine { arch: field(&d.arch), host_cpu, os: std::env::consts::OS.into() },
            device: Device {
                backend: field(&d.backend),
                kind: kind_name(d.kind).into(),
                name: field(&d.name),
                vendor: field(&d.vendor),
                driver_version: field(&d.driver_version),
                runtime_version: field(&d.runtime_version),
                memory_total: d.memory_total,
            },
            bundle: BundleId {
                model_id: field(&m.model_id),
                revision: field(&m.revision),
                manifest_sha256: field(&m.manifest_sha256),
                artifact_sha256: field(&m.artifact_sha256),
                tokenizer_sha256: field(&m.tokenizer_sha256),
            },
        };
        let rows = Rows::new(encoder.info().max_seq, tokenizer.fill_id());
        Ok(TurboEmbedder { encoder, tokenizer, identity, rows, ids: Vec::new(), vectors: Vec::new() })
    }

    fn encode(&self, role: Role) -> Encode {
        Encode {
            add_special_tokens: true,
            truncation: self.tokenizer.truncation(),
            max_tokens: self.encoder.info().max_seq,
            prompt: role.prompt(),
        }
    }
}

impl Embedder for TurboEmbedder {
    fn identity(&self) -> &Identity {
        &self.identity
    }

    fn dim(&self) -> usize {
        self.encoder.info().dim as usize
    }

    fn max_tokens(&self) -> u32 {
        self.encoder.info().max_seq
    }

    fn overhead(&self, role: Role) -> u32 {
        self.tokenizer.count("", role.prompt()) as u32
    }

    fn subwords(&self, text: &str) -> Result<Subwords> {
        let opts = Encode {
            add_special_tokens: false,
            truncation: Truncation::None,
            max_tokens: u32::MAX,
            prompt: PromptRole::None,
        };
        let (mut ids, mut spans) = (Vec::new(), Vec::new());
        self.tokenizer.encode_spans(text, opts, &mut ids, &mut spans).map_err(|e| e.to_string())?;
        let mut columns = Spans::with_capacity(ViewId::SOURCE, spans.len());
        for &[start, end] in &spans {
            columns.push(Span::new(start, end));
        }
        Ok(Subwords { spans: columns, ids, model: self.identity.model_ref(), token: None })
    }

    fn count(&self, text: &str, role: Role) -> u64 {
        self.tokenizer.count(text, role.prompt()) as u64
    }

    fn embed(&mut self, texts: &[&str], role: Role, out: &mut Vec<f32>) -> Result<u64> {
        let dim = self.dim();
        out.clear();
        out.resize(texts.len() * dim, 0.0);
        self.ids.resize_with(texts.len(), Vec::new);
        let mut tokens = 0u64;
        for (i, t) in texts.iter().enumerate() {
            let opts = self.encode(role);
            let row = &mut self.ids[i];
            row.clear();
            self.tokenizer.encode_into(t, opts, row).map_err(|e| format!("text {i}: {e}"))?;
            tokens += row.len() as u64;
        }
        // Rows of like length run together, so little of a run is padding.
        let mut order: Vec<usize> = (0..texts.len()).collect();
        order.sort_by_key(|&i| self.ids[i].len());
        let batch = self.encoder.info().max_batch.max(1) as usize;
        for group in order.chunks(batch) {
            self.rows.clear();
            for &i in group {
                self.rows.push(&[&self.ids[i]]);
            }
            self.encoder.run_rows(&self.rows, true, &mut self.vectors).map_err(|e| e.to_string())?;
            for (k, &i) in group.iter().enumerate() {
                out[i * dim..(i + 1) * dim].copy_from_slice(&self.vectors[k * dim..(k + 1) * dim]);
            }
        }
        for v in out.chunks_exact_mut(dim) {
            unit(v);
        }
        Ok(tokens)
    }
}

#[cfg(test)]
pub(crate) mod stub {
    //! A deterministic embedder for tests: each word is a piece, and a
    //! text's vector counts the words of a fixed vocabulary in it.

    use super::*;

    pub(crate) struct Stub {
        pub vocab: Vec<&'static str>,
        pub identity: Identity,
        /// Rows hold at most this many tokens, the two specials included.
        pub max_tokens: u32,
    }

    pub(crate) fn identity() -> Identity {
        Identity {
            machine: Machine { arch: "x86_64".into(), host_cpu: "Example CPU".into(), os: "linux".into() },
            device: Device {
                backend: "cpu".into(),
                kind: "DEVICE_CPU".into(),
                name: "Example CPU".into(),
                vendor: "Example".into(),
                driver_version: String::new(),
                runtime_version: String::new(),
                memory_total: 1 << 34,
            },
            bundle: BundleId {
                model_id: "stub".into(),
                revision: "1".into(),
                manifest_sha256: "a".repeat(64),
                artifact_sha256: "b".repeat(64),
                tokenizer_sha256: "c".repeat(64),
            },
        }
    }

    impl Stub {
        pub(crate) fn new(vocab: &[&'static str]) -> Stub {
            Stub { vocab: vocab.to_vec(), identity: identity(), max_tokens: 64 }
        }

        /// The byte spans of the words: runs of characters that are not
        /// whitespace.
        fn words(text: &str) -> Vec<(u32, u32)> {
            let mut out = Vec::new();
            let mut start = None;
            for (i, c) in text.char_indices() {
                match (c.is_whitespace(), start) {
                    (false, None) => start = Some(i),
                    (true, Some(s)) => {
                        out.push((s as u32, i as u32));
                        start = None;
                    }
                    _ => {}
                }
            }
            if let Some(s) = start {
                out.push((s as u32, text.len() as u32));
            }
            out
        }
    }

    impl Embedder for Stub {
        fn identity(&self) -> &Identity {
            &self.identity
        }
        fn dim(&self) -> usize {
            self.vocab.len()
        }
        fn max_tokens(&self) -> u32 {
            self.max_tokens
        }
        fn overhead(&self, _: Role) -> u32 {
            2
        }
        fn subwords(&self, text: &str) -> Result<Subwords> {
            let words = Stub::words(text);
            let mut spans = Spans::new(ViewId::SOURCE);
            for &(s, e) in &words {
                spans.push(Span::new(s, e));
            }
            Ok(Subwords { spans, ids: vec![0; words.len()], model: self.identity.model_ref(), token: None })
        }
        fn count(&self, text: &str, role: Role) -> u64 {
            Stub::words(text).len() as u64 + u64::from(self.overhead(role))
        }
        fn embed(&mut self, texts: &[&str], role: Role, out: &mut Vec<f32>) -> Result<u64> {
            let dim = self.dim();
            out.clear();
            out.resize(texts.len() * dim, 0.0);
            let mut tokens = 0;
            for (i, t) in texts.iter().enumerate() {
                let keep = self.max_tokens as usize - 2;
                let words = Stub::words(t);
                tokens += words.len().min(keep) as u64 + u64::from(self.overhead(role));
                for &(s, e) in words.iter().take(keep) {
                    let w = t[s as usize..e as usize].trim_matches(|c: char| !c.is_alphanumeric()).to_lowercase();
                    if let Some(k) = self.vocab.iter().position(|v| *v == w) {
                        out[i * dim + k] += 1.0;
                    }
                }
                unit(&mut out[i * dim..(i + 1) * dim]);
            }
            Ok(tokens)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The small BERT the analysis tests run (2 layers, 32 dimensions,
    /// rows of 64), in the tree, so this runs offline. It is not the
    /// reference model: it checks the session path, not retrieval quality.
    fn tiny_bert() -> TurboEmbedder {
        let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../testdata/tiny-bert-bundle");
        TurboEmbedder::open(&dir, turbo_analysis_embed::Device::Cpu, Precision::Model).unwrap()
    }

    #[test]
    fn the_session_embeds_one_unit_vector_per_text() {
        let mut e = tiny_bert();
        assert_eq!((e.dim(), e.max_tokens()), (32, 64));
        let id = e.identity().clone();
        assert_eq!(id.bundle.model_id, "sentence-transformers/all-MiniLM-L6-v2");
        assert_eq!(id.device.backend, "cpu");
        assert_eq!(id.device.kind, "DEVICE_CPU");
        assert_eq!(id.bundle.manifest_sha256.len(), 64);

        // [CLS] and [SEP] around the text; the query prefix "query: " adds
        // its own pieces in front.
        assert_eq!(e.overhead(Role::Document), 2);
        assert!(e.overhead(Role::Query) > 2);
        let texts = ["the cat sat", "a much longer text about rust and its compiler", "the cat sat"];
        let mut out = Vec::new();
        let tokens = e.embed(&texts, Role::Document, &mut out).unwrap();
        let counted: u64 = texts.iter().map(|t| e.count(t, Role::Document)).sum();
        assert_eq!(tokens, counted);
        assert_eq!(out.len(), 3 * 32);
        for v in out.chunks_exact(32) {
            let norm: f64 = v.iter().map(|&x| f64::from(x) * f64::from(x)).sum();
            assert!((norm - 1.0).abs() < 1e-5, "{norm}");
        }
        // The same text gives the same vector, whatever it is batched with.
        assert_eq!(out[..32], out[64..]);
        let mut alone = Vec::new();
        e.embed(&texts[..1], Role::Document, &mut alone).unwrap();
        let diff = alone.iter().zip(&out[..32]).map(|(a, b)| (a - b).abs()).fold(0f32, f32::max);
        assert!(diff < 1e-5, "{diff}");

        // A text over the row is cut as the bundle cuts it (on the right),
        // so the model is given 64 tokens.
        let long = "word ".repeat(200);
        assert!(e.count(&long, Role::Document) > 64);
        assert_eq!(e.embed(&[long.as_str()], Role::Document, &mut out).unwrap(), 64);

        // The pieces cover the words, without the specials.
        let s = e.subwords("the cat sat").unwrap();
        assert_eq!(s.spans.len(), 3);
        assert_eq!(s.ids.len(), 3);
    }
}
