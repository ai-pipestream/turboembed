//! Static token-embedding provider.
//!
//! A model2vec-style model is a table of one vector per vocabulary token. An
//! embedding is the mean of the rows for a text's tokens, optionally
//! L2-normalized. There is no attention and no context: two texts with the
//! same multiset of tokens embed identically. The provider says so through
//! its capability notes and reports every stage as host work.
//!
//! Bundle contract: a `static` artifact holding little-endian f32 values in
//! row-major `[vocab_size, dim]` order (`turbo-bundle import --static-from`
//! produces it from safetensors), a `tokenizer.json`, `contract.pooling =
//! "mean"`, `contract.normalize` set, `contract.dim`, and `contract.vocab_size`
//! equal to the table's rows. Tokens are encoded without special tokens, as
//! model2vec does.
//!
//! This provider offers one capability cell, `EMBED x TEXT x CPU`, and is
//! reached only by explicit device selection (AUTO never picks a CPU). Its
//! tokenization goes through the Hugging Face tokenizers crate, which
//! allocates per call; `provider_allocs` is therefore reported as unknown
//! rather than claimed to be zero.

#![deny(missing_docs)]

use std::sync::Arc;

use turbo_core::abi;
use turbo_core::buffer::{BufferDesc, HostBuffer, NativeHandle, ProviderBuffer};
use turbo_core::bundle::Bundle;
use turbo_core::error::{Error, Result};
use turbo_core::provider::{
    Capability, ContextDesc, DeviceInfo, EmbedOptions, ModelDesc, ModelInfo, Output, Provider, ProviderContext,
    ProviderModel, ProviderResult, ProviderSession, RunOptions, SessionDesc, SessionStats, TokenBatch,
};
use turbo_core::tokenizer::{EncodeOptions, Tokenizer};
use turbo_core::types::{
    CapStatus, DType, DeviceKind, HandleKind, Modality, ModelKind, Normalize, Placement, Pooling, Stage,
    StagePlacement, StagePlacements, Task,
};

/// Provider id.
pub const STATIC_PROVIDER_ID: &str = "static";
/// Artifact format name.
pub const STATIC_ARTIFACT: &str = "static";
/// Capability bits honored.
pub const STATIC_CAPS: u64 = abi::TURBO_CAP_HOST_PTR_IMPORT
    | abi::TURBO_CAP_DYNAMIC_SHAPE
    | abi::TURBO_CAP_WEIGHT_SHARING
    | abi::TURBO_CAP_DETERMINISTIC
    | abi::TURBO_CAP_OPT_TRUNCATE
    | abi::TURBO_CAP_OPT_MAX_TOKENS
    | abi::TURBO_CAP_OPT_PROMPT_ROLE
    | abi::TURBO_CAP_OPT_NORMALIZE
    | abi::TURBO_CAP_OPT_OUTPUT_DIM;

/// The provider.
#[derive(Debug, Default)]
pub struct StaticProvider;

impl StaticProvider {
    /// Construct.
    pub fn new() -> Self {
        Self
    }
}

impl Provider for StaticProvider {
    fn id(&self) -> &str {
        STATIC_PROVIDER_ID
    }

    fn version(&self) -> &str {
        env!("CARGO_PKG_VERSION")
    }

    fn devices(&self) -> Result<Vec<DeviceInfo>> {
        Ok(vec![DeviceInfo {
            kind: DeviceKind::Cpu,
            ordinal: 0,
            vendor_id: 0,
            caps: STATIC_CAPS,
            memory_total: 0,
            memory_free: 0,
            name: "Static embedding table (host CPU)".into(),
            vendor: "Pipestream".into(),
            provider_id: STATIC_PROVIDER_ID.into(),
            provider_version: env!("CARGO_PKG_VERSION").into(),
            runtime_version: format!("tokenizers {}", tokenizers_version()),
            driver_version: String::new(),
        }])
    }

    fn capability(&self, ordinal: u32, task: Task, modality: Modality) -> Capability {
        if ordinal != 0 || task != Task::Embed || modality != Modality::Text {
            return Capability::unsupported();
        }
        Capability {
            status: CapStatus::Experimental,
            dtype: Some(DType::F32),
            reference_dtype: Some(DType::F32),
            cosine_floor: 0.0,
            max_abs_error: 0.0,
            deterministic: true,
            notes: "static table: mean of token vectors, no context; precision receipt vs model2vec pending".into(),
        }
    }

    fn can_run(&self, ordinal: u32, bundle: &Bundle, task: Task, modality: Modality) -> Result<()> {
        if !self.capability(ordinal, task, modality).is_offered() {
            return Err(Error::unsupported_task(format!(
                "static provider offers EMBED x TEXT only, not {task:?} x {modality:?}"
            )));
        }
        if bundle.artifact(STATIC_ARTIFACT).is_none() {
            return Err(Error::bundle_no_artifact(format!(
                "bundle `{}` has no `static` artifact",
                bundle.manifest().model_id
            )));
        }
        if bundle.kind() != ModelKind::Embedding {
            return Err(Error::unsupported_task(format!(
                "static provider serves embedding bundles, not {:?}",
                bundle.kind()
            )));
        }
        if bundle.pooling()? != Some(Pooling::Mean) {
            return Err(Error::unsupported(format!(
                "static tables are mean-pooled; bundle `{}` declares {:?}",
                bundle.manifest().model_id,
                bundle.contract().pooling
            )));
        }
        Ok(())
    }

    fn create_context(&self, ordinal: u32, desc: &ContextDesc) -> Result<Arc<dyn ProviderContext>> {
        if ordinal != 0 {
            return Err(Error::device_not_found(format!("static provider has no device ordinal {ordinal}")));
        }
        desc.options.reject_unknown(&[], "static context")?;
        Ok(Arc::new(StaticContext))
    }
}

fn tokenizers_version() -> &'static str {
    // The tokenizers crate does not expose its version at runtime; record the
    // workspace pin so device info names the implementation.
    "0.23"
}

struct StaticContext;

impl ProviderContext for StaticContext {
    fn ordinal(&self) -> u32 {
        0
    }

    fn alloc(&self, desc: &BufferDesc) -> Result<Arc<dyn ProviderBuffer>> {
        if desc.placement != Placement::Host {
            return Err(Error::unsupported_placement(format!(
                "static provider allocates TURBO_PLACE_HOST only, not {:?}",
                desc.placement
            )));
        }
        Ok(HostBuffer::new(desc.clone())?)
    }

    fn import(&self, _desc: &BufferDesc, handle: &NativeHandle) -> Result<Arc<dyn ProviderBuffer>> {
        if handle.kind != HandleKind::HostPtr {
            return Err(Error::unsupported("static provider imports TURBO_HANDLE_HOST_PTR only"));
        }
        Err(Error::not_implemented("static provider buffer import"))
    }

    fn load_model(&self, bundle: Arc<Bundle>, desc: &ModelDesc) -> Result<Arc<dyn ProviderModel>> {
        desc.options.reject_unknown(&[], "static model")?;
        let c = bundle.contract();
        if c.dim == 0 || c.vocab_size == 0 {
            return Err(Error::bundle_invalid("static bundle must declare contract.dim and contract.vocab_size"));
        }
        let path = bundle.artifact_path(STATIC_ARTIFACT)?;
        let bytes = std::fs::read(&path)?;
        let expected = (c.vocab_size as usize)
            .checked_mul(c.dim as usize)
            .and_then(|n| n.checked_mul(4))
            .ok_or_else(|| Error::bundle_invalid("vocab_size * dim overflows"))?;
        if bytes.len() != expected {
            return Err(Error::bundle_integrity(format!(
                "static table `{}` is {} bytes; contract vocab_size {} x dim {} needs {expected}",
                path.display(),
                bytes.len(),
                c.vocab_size,
                c.dim
            )));
        }
        let mut table = Vec::with_capacity(bytes.len() / 4);
        for chunk in bytes.chunks_exact(4) {
            let v = f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
            if !v.is_finite() {
                return Err(Error::bundle_integrity(format!(
                    "static table `{}` contains a non-finite value",
                    path.display()
                )));
            }
            table.push(v);
        }
        let tokenizer = Tokenizer::from_bundle(&bundle)?;
        if tokenizer.info().vocab_size != c.vocab_size {
            return Err(Error::bundle_invalid(format!(
                "tokenizer vocabulary is {} but the static table has {} rows",
                tokenizer.info().vocab_size,
                c.vocab_size
            )));
        }
        let max_batch = if bundle.manifest().limits.max_batch == 0 { 64 } else { bundle.manifest().limits.max_batch };
        let host = StagePlacement::Host;
        let info = ModelInfo {
            task: Task::Embed,
            kind: ModelKind::Embedding,
            modality: Modality::Text,
            dim: c.dim,
            labels: Vec::new(),
            pooling: Some(Pooling::Mean),
            normalize: bundle.normalize()?,
            aggregation: None,
            max_seq: if c.max_seq == 0 { 512 } else { c.max_seq },
            max_batch,
            dtype_used: Some(DType::F32),
            stages: StagePlacements::NONE
                .with(Stage::Tokenize, host)
                .with(Stage::Encode, host)
                .with(Stage::Pool, host)
                .with(Stage::Normalize, host),
            inputs: Vec::new(),
            outputs: Vec::new(),
            vocab_size: c.vocab_size,
            model_id: bundle.manifest().model_id.clone(),
            revision: bundle.manifest().revision.clone(),
            tokenizer_sha256: bundle.tokenizer_sha256().to_string(),
            provider_id: STATIC_PROVIDER_ID.into(),
            prefix_query: c.prompts.query.clone(),
            prefix_document: c.prompts.document.clone(),
        };
        Ok(Arc::new(StaticModel { info, table: Arc::from(table), tokenizer, truncate_dims: c.truncate_dims.clone() }))
    }
}

struct StaticModel {
    info: ModelInfo,
    table: Arc<[f32]>,
    tokenizer: Arc<Tokenizer>,
    truncate_dims: Vec<u32>,
}

impl ProviderModel for StaticModel {
    fn info(&self) -> &ModelInfo {
        &self.info
    }

    fn create_session(&self, desc: &SessionDesc) -> Result<Box<dyn ProviderSession>> {
        desc.options.reject_unknown(&[], "static session")?;
        let cap = desc.max_batch as usize * desc.max_seq as usize;
        Ok(Box::new(StaticSession {
            table: self.table.clone(),
            tokenizer: self.tokenizer.clone(),
            info: self.info.clone(),
            max_seq: desc.max_seq,
            ids: vec![0; cap],
            mask: vec![0; cap],
            lengths: vec![0; desc.max_batch as usize],
            n_rows: 0,
            out: HostBuffer::packed(DType::F32, &[desc.max_batch as u64, self.info.dim as u64])?,
            full: vec![0.0; self.info.dim as usize],
            opts: EmbedOptions::default(),
            runs: 0,
            name: Arc::from("embeddings"),
            _truncate_dims: self.truncate_dims.clone(),
        }))
    }
}

struct StaticSession {
    table: Arc<[f32]>,
    tokenizer: Arc<Tokenizer>,
    info: ModelInfo,
    max_seq: u32,
    ids: Vec<i32>,
    mask: Vec<i32>,
    lengths: Vec<u32>,
    n_rows: u32,
    out: Arc<HostBuffer>,
    full: Vec<f32>,
    opts: EmbedOptions,
    runs: u64,
    name: Arc<str>,
    _truncate_dims: Vec<u32>,
}

impl ProviderSession for StaticSession {
    fn write_text(&mut self, texts: &[&str], opts: &EmbedOptions) -> Result<()> {
        self.opts = *opts;
        self.n_rows = texts.len() as u32;
        let stride = self.max_seq as usize;
        let enc = EncodeOptions {
            add_special_tokens: false,
            truncate: opts.truncate,
            max_tokens: if opts.max_tokens == 0 { self.max_seq } else { opts.max_tokens },
            pad_to: 0,
            prompt_role: opts.prompt_role,
        };
        self.tokenizer.encode_into(
            texts,
            &enc,
            turbo_core::tokenizer::EncodeTarget {
                ids: &mut self.ids,
                mask: &mut self.mask,
                types: None,
                row_stride: stride,
                lengths: &mut self.lengths,
            },
        )
    }

    fn write_tokens(&mut self, batch: &TokenBatch<'_>) -> Result<()> {
        self.opts = EmbedOptions::default();
        self.n_rows = batch.batch;
        let stride = self.max_seq as usize;
        for r in 0..batch.batch as usize {
            let ids = batch.ids_row(r);
            let mask = batch.mask_row(r);
            let base = r * stride;
            self.ids[base..base + ids.len()].copy_from_slice(ids);
            self.mask[base..base + mask.len()].copy_from_slice(mask);
            for c in ids.len()..stride {
                self.mask[base + c] = 0;
            }
            self.lengths[r] = mask.iter().filter(|&&m| m == 1).count() as u32;
        }
        Ok(())
    }

    fn run(&mut self, opts: &RunOptions) -> Result<ProviderResult> {
        opts.params.reject_unknown(&[], "static run")?;
        let dim = self.info.dim as usize;
        let dim_out = if self.opts.output_dim == 0 { dim } else { self.opts.output_dim as usize };
        let normalize = match self.opts.normalize {
            Normalize::Model => self.info.normalize == Some(Normalize::L2),
            Normalize::L2 => true,
            Normalize::None => false,
        };
        let n = self.n_rows as usize;
        let stride = self.max_seq as usize;
        // SAFETY: the core holds the session lock with no result lease outstanding.
        let out = unsafe { self.out.as_f32_mut()? };
        for r in 0..n {
            let full = &mut self.full;
            for v in full.iter_mut() {
                *v = 0.0;
            }
            let base = r * stride;
            let mut count = 0usize;
            for c in 0..stride {
                if self.mask[base + c] == 0 {
                    continue;
                }
                let id = self.ids[base + c];
                if id < 0 || id as usize >= self.info.vocab_size as usize {
                    return Err(Error::invalid_argument(format!(
                        "token id {id} is outside the table of {} rows",
                        self.info.vocab_size
                    )));
                }
                let row = &self.table[id as usize * dim..(id as usize + 1) * dim];
                for (f, t) in full.iter_mut().zip(row) {
                    *f += *t;
                }
                count += 1;
            }
            if count > 0 {
                for v in full.iter_mut() {
                    *v /= count as f32;
                }
            }
            let dst = &mut out[r * dim_out..(r + 1) * dim_out];
            dst.copy_from_slice(&full[..dim_out]);
            if normalize {
                let norm = dst.iter().map(|x| x * x).sum::<f32>().sqrt();
                if norm > 1e-12 {
                    for x in dst.iter_mut() {
                        *x /= norm;
                    }
                }
            }
        }
        self.runs += 1;
        Ok(ProviderResult {
            outputs: vec![Output {
                name: self.name.clone(),
                buffer: self.out.clone(),
                shape: vec![n as u64, dim_out as u64],
            }],
            spans: Vec::new(),
        })
    }

    fn stats(&self) -> Result<SessionStats> {
        Ok(SessionStats {
            runs: self.runs,
            host_allocs: None,
            h2d_bytes: 0,
            d2h_bytes: 0,
            input_bytes: (self.ids.len() * 8) as u64,
            output_bytes: self.out.desc().bytes,
            // The Hugging Face tokenizer allocates per encode; not counted.
            provider_allocs: None,
        })
    }
}

turbo_core::export_provider!(c"static", c"2.0.0-alpha.0", || Arc::new(StaticProvider::new()));

/// Write a synthetic static bundle for tests: a deterministic table over the
/// MiniLM tokenizer vocabulary with `dim` columns.
pub fn write_test_bundle(dir: &std::path::Path, tokenizer_json: &std::path::Path, dim: u32) -> Result<()> {
    use turbo_core::bundle::{sha256_file, MANIFEST_NAME};
    let tok_dst = dir.join("tokenizer.json");
    std::fs::copy(tokenizer_json, &tok_dst)?;
    let vocab = 30522u32;
    let mut bytes = Vec::with_capacity(vocab as usize * dim as usize * 4);
    for id in 0..vocab {
        for k in 0..dim {
            // Deterministic pseudo-random in [-1, 1).
            let h = (id as u64)
                .wrapping_mul(0x9E37_79B9_7F4A_7C15)
                .wrapping_add((k as u64).wrapping_mul(0xD1B5_4A32_D192_ED03));
            let h = h ^ (h >> 29);
            let v = ((h >> 11) as f64 / (1u64 << 53) as f64 * 2.0 - 1.0) as f32;
            bytes.extend_from_slice(&v.to_le_bytes());
        }
    }
    let table = dir.join("static.f32");
    std::fs::write(&table, &bytes)?;
    let manifest = serde_json_manifest(&sha256_file(&tok_dst)?, &sha256_file(&table)?, vocab, dim);
    std::fs::write(dir.join(MANIFEST_NAME), manifest)?;
    Bundle::open(dir)?;
    Ok(())
}

fn serde_json_manifest(tok_sha: &str, table_sha: &str, vocab: u32, dim: u32) -> String {
    format!(
        r#"{{
  "bundle_version": 2,
  "model_id": "turbo/static-test",
  "revision": "test",
  "license": "Apache-2.0",
  "task": "embed",
  "kind": "embedding",
  "modality": "text",
  "family": "static",
  "tokenizer": {{ "kind": "wordpiece", "files": {{ "tokenizer.json": {{ "path": "tokenizer.json", "sha256": "{tok_sha}" }} }} }},
  "contract": {{ "pooling": "mean", "normalize": "l2", "max_seq": 64, "dim": {dim}, "vocab_size": {vocab},
                 "truncate_dims": [4], "prompts": {{ "query": "query: ", "document": "passage: " }} }},
  "artifacts": {{ "static": {{ "path": "static.f32", "sha256": "{table_sha}" }} }},
  "limits": {{ "max_batch": 8 }}
}}"#
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::{Path, PathBuf};
    use turbo_core::{Context, DeviceSelector, RuntimeDesc, SelectPolicy};

    fn tokenizer_json() -> PathBuf {
        Path::new(env!("CARGO_MANIFEST_DIR")).join("../../testdata/bundles/minilm-tokenizer/tokenizer.json")
    }

    fn setup(dim: u32) -> (tempfile::TempDir, Arc<Context>) {
        let tmp = tempfile::tempdir().unwrap();
        write_test_bundle(tmp.path(), &tokenizer_json(), dim).unwrap();
        let rt = turbo_core::Runtime::new(RuntimeDesc::default(), vec![Arc::new(StaticProvider::new())]).unwrap();
        assert_eq!(rt.select(&DeviceSelector::default()).unwrap_err().code(), abi::TURBO_E_DEVICE_NOT_FOUND);
        let idx = rt
            .select(&DeviceSelector {
                policy: SelectPolicy::Explicit,
                provider_id: "static".into(),
                ..Default::default()
            })
            .unwrap();
        let ctx = Context::create(rt, idx, &ContextDesc::default()).unwrap();
        (tmp, ctx)
    }

    fn embed(ctx: &Arc<Context>, dir: &Path, texts: &[&str], opts: &EmbedOptions) -> Vec<Vec<f32>> {
        let model = ctx.load_model(dir, &ModelDesc::default()).unwrap();
        let session = model.create_session(&SessionDesc::default()).unwrap();
        session.write_text(texts, opts).unwrap();
        let r = session.run(&RunOptions::default()).unwrap();
        let out = r.output(0).unwrap();
        let mut bytes = vec![0u8; out.logical_bytes().unwrap() as usize];
        r.read(0, &mut bytes).unwrap();
        let cols = out.shape[1] as usize;
        bytes
            .chunks(4)
            .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
            .collect::<Vec<_>>()
            .chunks(cols)
            .map(|c| c.to_vec())
            .collect()
    }

    #[test]
    fn mean_of_rows_and_normalization() {
        let (tmp, ctx) = setup(8);
        let v = embed(&ctx, tmp.path(), &["hello world", "world hello", "hello", ""], &EmbedOptions::default());
        assert_eq!(v[0], v[1], "bag of tokens: order does not matter");
        assert_ne!(v[0], v[2]);
        let norm: f32 = v[0].iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 1e-5);
        assert!(v[3].iter().all(|&x| x == 0.0), "empty text is the zero vector");
        let again = embed(&ctx, tmp.path(), &["hello world"], &EmbedOptions::default());
        assert_eq!(again[0], v[0], "a table lookup is deterministic across model loads");
        let raw = embed(
            &ctx,
            tmp.path(),
            &["hello world"],
            &EmbedOptions { normalize: Normalize::None, ..Default::default() },
        );
        let raw_norm: f32 = raw[0].iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((raw_norm - 1.0).abs() > 1e-3);
    }

    #[test]
    fn options_are_honored() {
        let (tmp, ctx) = setup(8);
        let plain = embed(&ctx, tmp.path(), &["hello"], &EmbedOptions::default());
        let q = embed(
            &ctx,
            tmp.path(),
            &["hello"],
            &EmbedOptions { prompt_role: turbo_core::PromptRole::Query, ..Default::default() },
        );
        assert_ne!(plain[0], q[0]);
        // A narrowed vector is the leading components of the full one,
        // renormalized, not an arbitrary four numbers of unit length.
        let full = embed(&ctx, tmp.path(), &["hello world"], &EmbedOptions::default());
        let small = embed(&ctx, tmp.path(), &["hello world"], &EmbedOptions { output_dim: 4, ..Default::default() });
        assert_eq!(small[0].len(), 4);
        assert!((small[0].iter().map(|x| x * x).sum::<f32>().sqrt() - 1.0).abs() < 1e-5);
        let head_norm: f32 = full[0][..4].iter().map(|x| x * x).sum::<f32>().sqrt();
        for (i, (got, head)) in small[0].iter().zip(&full[0][..4]).enumerate() {
            let want = head / head_norm;
            assert!(
                (got - want).abs() < 1e-5,
                "component {i} of the narrowed vector is {got}, not the renormalized {want} of the full one"
            );
        }
        let model = ctx.load_model(tmp.path(), &ModelDesc::default()).unwrap();
        let session = model.create_session(&SessionDesc::default()).unwrap();
        let long = "word ".repeat(200);
        let e = session
            .write_text(&[&long], &EmbedOptions { truncate: turbo_core::Truncate::None, ..Default::default() })
            .unwrap_err();
        assert_eq!(e.code(), abi::TURBO_E_CAPACITY);
        let e = session.write_text(&["x"], &EmbedOptions { pooling: Pooling::Cls, ..Default::default() }).unwrap_err();
        assert_eq!(e.code(), abi::TURBO_E_UNSUPPORTED_OPTION);
        assert_eq!(e.field(), EmbedOptions::FIELD_POOLING);
    }

    #[test]
    fn rejects_bundles_that_do_not_fit() {
        let (tmp, ctx) = setup(8);
        let bad = tmp.path().join("bad");
        std::fs::create_dir(&bad).unwrap();
        write_test_bundle(&bad, &tokenizer_json(), 8).unwrap();
        // Tamper with the table size after hashing would fail integrity; instead
        // change the contract so table and contract disagree.
        let manifest = std::fs::read_to_string(bad.join("bundle.json")).unwrap().replace("\"dim\": 8", "\"dim\": 16");
        std::fs::write(bad.join("bundle.json"), manifest).unwrap();
        let e = ctx.load_model(&bad, &ModelDesc::default()).unwrap_err();
        assert_eq!(e.code(), abi::TURBO_E_BUNDLE_INTEGRITY);
    }
}
