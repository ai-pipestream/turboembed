//! Import a model directory into a Turbo bundle.
//!
//! The contract is derived from the model's own files, never from its name:
//! `modules.json` and `1_Pooling/config.json` (sentence-transformers v6 and
//! legacy schemas), `config_sentence_transformers.json` (prompts),
//! `sentence_bert_config.json` (sequence limit), the Hugging Face
//! `config.json` (hidden size, labels, model type, vocabulary), and
//! `tokenizer_config.json` (chat template). Ambiguous or unsupported inputs
//! fail with a message that names the file and field; nothing is guessed.
//!
//! Artifacts are copied into the bundle and hashed. The output directory is
//! staged as a sibling and renamed into place after `bundle.json` is written,
//! so an interrupted import never leaves a loadable-looking bundle.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use serde_json::Value;
use turbo_core::bundle::{
    kind_from_name, sha256_file, task_from_name, Bundle, Contract, FileEntry, Limits, Manifest, Prompts, TokenizerSpec,
    BUNDLE_VERSION, MANIFEST_NAME,
};
use turbo_core::types::ModelKind;

use crate::safetensors::SafeTensors;

/// Import parameters.
#[derive(Debug, Default)]
pub struct ImportRequest {
    /// Source model directory (sentence-transformers or Hugging Face layout).
    pub source: PathBuf,
    /// Output bundle directory; must not exist.
    pub output: PathBuf,
    /// Model identifier; defaults to `config_sentence_transformers.json` / directory name.
    pub model_id: Option<String>,
    /// Revision to record.
    pub revision: Option<String>,
    /// SPDX license; required.
    pub license: Option<String>,
    /// Task name override.
    pub task: Option<String>,
    /// Kind name override.
    pub kind: Option<String>,
    /// Artifacts as `format=path`.
    pub artifacts: Vec<(String, PathBuf)>,
    /// Build a `static` artifact from a safetensors tensor: `path[:tensor]`.
    pub static_from: Option<(PathBuf, Option<String>)>,
    /// Matryoshka dimensions.
    pub truncate_dims: Vec<u32>,
    /// Maximum batch to record.
    pub max_batch: u32,
    /// Fixed-shape artifact.
    pub fixed_shape: bool,
    /// Override the sequence limit.
    pub max_seq: Option<u32>,
}

/// Outcome of an import.
#[derive(Debug)]
pub struct ImportReport {
    /// Where the bundle was written.
    pub output: PathBuf,
    /// The manifest.
    pub manifest: Manifest,
    /// Human-readable notes about derivation decisions.
    pub notes: Vec<String>,
}

fn read_json(path: &Path) -> Result<Option<Value>, String> {
    if !path.is_file() {
        return Ok(None);
    }
    let text = std::fs::read_to_string(path).map_err(|e| format!("read {}: {e}", path.display()))?;
    serde_json::from_str(&text).map(Some).map_err(|e| format!("parse {}: {e}", path.display()))
}

/// Derive pooling and normalization from sentence-transformers files.
fn derive_pooling(source: &Path, notes: &mut Vec<String>) -> Result<(Option<String>, Option<String>), String> {
    let modules = read_json(&source.join("modules.json"))?;
    let Some(modules) = modules else {
        notes.push("no modules.json; pooling and normalization not derived".into());
        return Ok((None, None));
    };
    let list = modules.as_array().ok_or("modules.json is not a list")?;
    let mut pooling_dir: Option<String> = None;
    let mut normalize = false;
    for m in list {
        let ty = m.get("type").and_then(Value::as_str).unwrap_or("");
        let path = m.get("path").and_then(Value::as_str).unwrap_or("");
        if ty.ends_with("Pooling") {
            pooling_dir = Some(path.to_string());
        } else if ty.ends_with("Normalize") {
            normalize = true;
        }
    }
    let Some(dir) = pooling_dir else {
        return Err("modules.json lists no Pooling module".into());
    };
    let cfg_path = source.join(&dir).join("config.json");
    let cfg = read_json(&cfg_path)?.ok_or_else(|| format!("{} is missing", cfg_path.display()))?;
    let pooling = if let Some(mode) = cfg.get("pooling_mode") {
        // v6 schema.
        let mode = mode.as_str().ok_or("pooling_mode is not a string (tuples are not supported)")?;
        notes.push(format!("pooling from {} (v6 schema): {mode}", cfg_path.display()));
        map_pooling(mode)?
    } else {
        // Legacy schema: six booleans, documented precedence, mean by default.
        let flag = |k: &str| cfg.get(k).and_then(Value::as_bool).unwrap_or(false);
        let set: Vec<&str> = [
            ("pooling_mode_cls_token", "cls"),
            ("pooling_mode_mean_tokens", "mean"),
            ("pooling_mode_max_tokens", "max"),
            ("pooling_mode_mean_sqrt_len_tokens", "mean_sqrt_len_tokens"),
            ("pooling_mode_weightedmean_tokens", "weightedmean"),
            ("pooling_mode_lasttoken", "lasttoken"),
        ]
        .iter()
        .filter(|(k, _)| flag(k))
        .map(|(_, v)| *v)
        .collect();
        let mode = match set.len() {
            0 => {
                notes.push(format!(
                    "no pooling flag set in {}; sentence-transformers defaults to mean",
                    cfg_path.display()
                ));
                "mean"
            }
            1 => set[0],
            _ => {
                return Err(format!(
                    "{} sets several pooling modes ({}); concatenated pooling is not supported",
                    cfg_path.display(),
                    set.join(", ")
                ))
            }
        };
        notes.push(format!("pooling from {} (legacy schema): {mode}", cfg_path.display()));
        map_pooling(mode)?
    };
    notes.push(format!("normalization: {}", if normalize { "l2 (Normalize module present)" } else { "none" }));
    Ok((Some(pooling), Some(if normalize { "l2".into() } else { "none".into() })))
}

fn map_pooling(mode: &str) -> Result<String, String> {
    match mode {
        "cls" => Ok("cls".into()),
        "mean" => Ok("mean".into()),
        "lasttoken" | "last" => Ok("last".into()),
        other => Err(format!("pooling mode `{other}` is not supported by the contract (mean, cls, last)")),
    }
}

fn hash_entry(path: &Path, rel: &str, provenance: Option<String>) -> Result<FileEntry, String> {
    let sha256 = sha256_file(path).map_err(|e| format!("hash {}: {e}", path.display()))?;
    Ok(FileEntry { path: rel.to_string(), sha256, provenance })
}

fn copy_into(staging: &Path, src: &Path, name: &str) -> Result<PathBuf, String> {
    let dst = staging.join(name);
    if dst.exists() {
        return Err(format!("staging already has `{name}`; artifact file names must be unique"));
    }
    std::fs::copy(src, &dst).map_err(|e| format!("copy {} to {}: {e}", src.display(), dst.display()))?;
    Ok(dst)
}

/// Run an import.
pub fn import(req: &ImportRequest) -> Result<ImportReport, String> {
    if !req.source.is_dir() {
        return Err(format!("source `{}` is not a directory", req.source.display()));
    }
    if req.output.exists() {
        return Err(format!("output `{}` already exists; refusing to overwrite", req.output.display()));
    }
    let license = req.license.clone().ok_or("--license is required (SPDX identifier of the weights)")?;
    let mut notes = Vec::new();

    let hf_config = read_json(&req.source.join("config.json"))?;
    let st_config = read_json(&req.source.join("config_sentence_transformers.json"))?;
    let bert_config = read_json(&req.source.join("sentence_bert_config.json"))?;
    let tok_config = read_json(&req.source.join("tokenizer_config.json"))?;

    let model_id = req
        .model_id
        .clone()
        .or_else(|| hf_config.as_ref().and_then(|c| c.get("_name_or_path")).and_then(Value::as_str).map(str::to_string))
        .or_else(|| req.source.file_name().map(|n| n.to_string_lossy().into_owned()))
        .ok_or("cannot determine model_id; pass --model-id")?;

    let (mut pooling, mut normalize) = derive_pooling(&req.source, &mut notes)?;
    if req.static_from.is_some() && pooling.is_none() {
        // Static embedding tables (model2vec) are mean-pooled by definition;
        // normalization follows the model's config.json.
        let norm = hf_config.as_ref().and_then(|c| c.get("normalize")).and_then(Value::as_bool);
        let Some(norm) = norm else {
            return Err(
                "static model: config.json must declare `normalize` (true/false) so the contract is explicit".into()
            );
        };
        pooling = Some("mean".into());
        normalize = Some(if norm { "l2".into() } else { "none".into() });
        notes.push(format!("static table: pooling mean, normalize {}", if norm { "l2" } else { "none" }));
    }

    let mut prompts = Prompts::default();
    if let Some(st) = &st_config {
        if let Some(p) = st.get("prompts").and_then(Value::as_object) {
            for (k, v) in p {
                let text = v.as_str().ok_or_else(|| format!("prompts.{k} is not a string"))?;
                match k.as_str() {
                    "query" => prompts.query = text.to_string(),
                    "document" | "passage" => prompts.document = text.to_string(),
                    other => notes.push(format!("ignored prompt `{other}` (only query/document are in the contract)")),
                }
            }
        }
        if let Some(d) = st.get("default_prompt_name").and_then(Value::as_str) {
            notes.push(format!("default_prompt_name `{d}` is informational; callers choose the role per call"));
        }
    }
    let similarity_fn =
        st_config.as_ref().and_then(|c| c.get("similarity_fn_name")).and_then(Value::as_str).map(str::to_string);

    let max_seq = req
        .max_seq
        .or_else(|| {
            bert_config.as_ref().and_then(|c| c.get("max_seq_length")).and_then(Value::as_u64).map(|v| v as u32)
        })
        .or_else(|| {
            hf_config.as_ref().and_then(|c| c.get("max_position_embeddings")).and_then(Value::as_u64).map(|v| {
                notes.push(format!("max_seq from config.json max_position_embeddings = {v}"));
                v as u32
            })
        });
    let hidden = hf_config.as_ref().and_then(|c| c.get("hidden_size")).and_then(Value::as_u64).map(|v| v as u32);
    let vocab_size =
        hf_config.as_ref().and_then(|c| c.get("vocab_size")).and_then(Value::as_u64).map(|v| v as u32).unwrap_or(0);
    let family = hf_config.as_ref().and_then(|c| c.get("model_type")).and_then(Value::as_str).unwrap_or("").to_string();
    let labels: Vec<String> = hf_config
        .as_ref()
        .and_then(|c| c.get("id2label"))
        .and_then(Value::as_object)
        .map(|m| {
            let mut v: Vec<(u32, String)> =
                m.iter().filter_map(|(k, v)| Some((k.parse().ok()?, v.as_str()?.to_string()))).collect();
            v.sort();
            v.into_iter().map(|(_, l)| l).collect()
        })
        .unwrap_or_default();

    // Kind and task.
    let kind_name = match &req.kind {
        Some(k) => k.clone(),
        None => {
            let arch = hf_config
                .as_ref()
                .and_then(|c| c.get("architectures"))
                .and_then(Value::as_array)
                .and_then(|a| a.first())
                .and_then(Value::as_str)
                .unwrap_or("");
            if req.static_from.is_some() || pooling.is_some() {
                "embedding".into()
            } else if arch.ends_with("ForSequenceClassification") {
                if labels.len() == 1 {
                    "reranker".into()
                } else {
                    "classifier".into()
                }
            } else if arch.ends_with("ForTokenClassification") {
                "token_classifier".into()
            } else if arch.ends_with("ForCausalLM") {
                "generative".into()
            } else {
                return Err(format!(
                    "cannot infer model kind from architectures {arch:?}; pass --kind (embedding, reranker, classifier, token_classifier, generative, generic)"
                ));
            }
        }
    };
    let kind = kind_from_name(&kind_name).map_err(|e| e.to_string())?;
    let task_name = match &req.task {
        Some(t) => t.clone(),
        None => match kind {
            ModelKind::Embedding => "embed",
            ModelKind::Reranker => "rerank",
            ModelKind::Classifier => "classify",
            ModelKind::TokenClassifier => "token_classify",
            ModelKind::Generative => "generate",
            ModelKind::Generic => "run",
        }
        .to_string(),
    };
    task_from_name(&task_name).map_err(|e| e.to_string())?;
    notes.push(format!("kind {kind_name}, task {task_name}"));

    // Stage.
    let parent = req.output.parent().ok_or("output has no parent directory")?;
    std::fs::create_dir_all(parent).map_err(|e| format!("create {}: {e}", parent.display()))?;
    let staging = parent.join(format!(".{}.staging", req.output.file_name().unwrap_or_default().to_string_lossy()));
    if staging.exists() {
        std::fs::remove_dir_all(&staging).map_err(|e| format!("remove stale staging {}: {e}", staging.display()))?;
    }
    std::fs::create_dir(&staging).map_err(|e| format!("create staging {}: {e}", staging.display()))?;
    let result = stage(
        req,
        &staging,
        StageInputs {
            model_id,
            license,
            task_name,
            kind_name,
            kind,
            family,
            pooling,
            normalize,
            prompts,
            similarity_fn,
            max_seq,
            hidden,
            vocab_size,
            labels,
            chat_template: tok_config
                .as_ref()
                .and_then(|c| c.get("chat_template"))
                .and_then(Value::as_str)
                .map(str::to_string),
        },
        &mut notes,
    );
    match result {
        Ok(manifest) => {
            std::fs::rename(&staging, &req.output)
                .map_err(|e| format!("rename staging into {}: {e}", req.output.display()))?;
            Bundle::open(&req.output).map_err(|e| format!("verification of the written bundle failed: {e}"))?;
            Ok(ImportReport { output: req.output.clone(), manifest, notes })
        }
        Err(e) => {
            let _ = std::fs::remove_dir_all(&staging);
            Err(e)
        }
    }
}

struct StageInputs {
    model_id: String,
    license: String,
    task_name: String,
    kind_name: String,
    kind: ModelKind,
    family: String,
    pooling: Option<String>,
    normalize: Option<String>,
    prompts: Prompts,
    similarity_fn: Option<String>,
    max_seq: Option<u32>,
    hidden: Option<u32>,
    vocab_size: u32,
    labels: Vec<String>,
    chat_template: Option<String>,
}

fn stage(req: &ImportRequest, staging: &Path, inp: StageInputs, notes: &mut Vec<String>) -> Result<Manifest, String> {
    // Tokenizer: tokenizer.json as shipped, or one built from vocab.txt
    // (BERT WordPiece) with the casing from tokenizer_config.json.
    let mut tokenizer = None;
    let tok_src = req.source.join("tokenizer.json");
    let vocab_src = req.source.join("vocab.txt");
    if tok_src.is_file() {
        let dst = copy_into(staging, &tok_src, "tokenizer.json")?;
        let mut files = BTreeMap::new();
        files.insert("tokenizer.json".to_string(), hash_entry(&dst, "tokenizer.json", None)?);
        let kind = detect_tokenizer_kind(&tok_src)?;
        tokenizer = Some(TokenizerSpec { kind, files, chat_template: inp.chat_template.clone() });
    } else if vocab_src.is_file() {
        let tok_config = read_json(&req.source.join("tokenizer_config.json"))?;
        let lowercase = tok_config
            .as_ref()
            .and_then(|c| c.get("do_lower_case"))
            .and_then(Value::as_bool)
            .ok_or("vocab.txt without tokenizer.json needs tokenizer_config.json with do_lower_case")?;
        let json = tokenizer_json_from_vocab(&vocab_src, lowercase)?;
        let dst = staging.join("tokenizer.json");
        std::fs::write(&dst, json).map_err(|e| format!("write {}: {e}", dst.display()))?;
        let mut files = BTreeMap::new();
        files.insert(
            "tokenizer.json".to_string(),
            hash_entry(&dst, "tokenizer.json", Some(format!("built from vocab.txt, do_lower_case = {lowercase}")))?,
        );
        notes.push(format!("tokenizer.json built from vocab.txt (lowercase {lowercase})"));
        tokenizer = Some(TokenizerSpec { kind: "wordpiece".into(), files, chat_template: inp.chat_template.clone() });
    } else {
        notes.push("no tokenizer.json or vocab.txt in the source; the bundle has no tokenizer".into());
    }

    // Artifacts.
    let mut artifacts = BTreeMap::new();
    let mut dim = inp.hidden.unwrap_or(0);
    let mut vocab_size = inp.vocab_size;
    for (format, path) in &req.artifacts {
        if !path.is_file() {
            return Err(format!("artifact `{format}` path `{}` is not a file", path.display()));
        }
        let name = path.file_name().ok_or("artifact path has no file name")?.to_string_lossy().into_owned();
        let dst = copy_into(staging, path, &name)?;
        artifacts.insert(format.clone(), hash_entry(&dst, &name, Some(format!("copied from {}", path.display())))?);
        // OpenVINO IR needs the .bin next to the .xml.
        if format == "openvino_ir" {
            let bin = path.with_extension("bin");
            if !bin.is_file() {
                return Err(format!("openvino_ir artifact `{}` has no sibling .bin", path.display()));
            }
            let bin_name = bin.file_name().unwrap().to_string_lossy().into_owned();
            let bdst = copy_into(staging, &bin, &bin_name)?;
            artifacts.insert(
                "openvino_ir_weights".into(),
                hash_entry(&bdst, &bin_name, Some(format!("copied from {}", bin.display())))?,
            );
        }
    }
    if let Some((path, tensor)) = &req.static_from {
        let st = SafeTensors::open(path)?;
        let tensor_name = match tensor {
            Some(t) => t.clone(),
            None => {
                let names = st.names();
                if names.len() == 1 {
                    names[0].to_string()
                } else if names.contains(&"embeddings") {
                    "embeddings".to_string()
                } else {
                    return Err(format!(
                        "safetensors has several tensors ({}); name one with path:tensor",
                        names.join(", ")
                    ));
                }
            }
        };
        let (values, shape) = st.table_f32(&tensor_name)?;
        let dst = staging.join("static.f32");
        let mut bytes = Vec::with_capacity(values.len() * 4);
        for v in &values {
            bytes.extend_from_slice(&v.to_le_bytes());
        }
        std::fs::write(&dst, &bytes).map_err(|e| format!("write {}: {e}", dst.display()))?;
        artifacts.insert(
            "static".into(),
            hash_entry(
                &dst,
                "static.f32",
                Some(format!(
                    "tensor `{tensor_name}` of {} as little-endian f32 [{}, {}]",
                    path.display(),
                    shape[0],
                    shape[1]
                )),
            )?,
        );
        if dim != 0 && dim != shape[1] as u32 {
            notes.push(format!("static table dim {} overrides config.json hidden_size {dim}", shape[1]));
        }
        dim = shape[1] as u32;
        if vocab_size != 0 && vocab_size != shape[0] as u32 {
            return Err(format!("static table has {} rows but the tokenizer vocabulary is {vocab_size}", shape[0]));
        }
        vocab_size = shape[0] as u32;
    }
    if artifacts.is_empty() && tokenizer.is_none() {
        return Err("no artifacts and no tokenizer; pass --artifact format=path or --static-from".into());
    }

    let max_seq = inp.max_seq.unwrap_or(0);
    let contract = Contract {
        pooling: inp.pooling.clone(),
        normalize: inp.normalize.clone(),
        max_seq,
        dim,
        truncate_dims: req.truncate_dims.clone(),
        prompts: inp.prompts.clone(),
        similarity_fn: inp.similarity_fn.clone(),
        dtype: Some("f32".into()),
        vocab_size,
        labels: inp.labels.clone(),
        activation: match inp.kind {
            ModelKind::Classifier => Some("softmax".into()),
            ModelKind::Reranker => Some("sigmoid".into()),
            _ => None,
        },
        aggregation: if inp.kind == ModelKind::TokenClassifier { Some("simple".into()) } else { None },
        tagging: if inp.kind == ModelKind::TokenClassifier { Some("BIO".into()) } else { None },
    };
    let manifest = Manifest {
        bundle_version: BUNDLE_VERSION,
        model_id: inp.model_id,
        revision: req.revision.clone().unwrap_or_default(),
        license: inp.license,
        task: inp.task_name,
        kind: inp.kind_name,
        modality: "text".into(),
        family: inp.family,
        tokenizer,
        contract,
        artifacts,
        limits: Limits { max_batch: req.max_batch, fixed_shape: req.fixed_shape },
    };
    let text = serde_json::to_string_pretty(&manifest).map_err(|e| format!("serialize manifest: {e}"))?;
    std::fs::write(staging.join(MANIFEST_NAME), text).map_err(|e| format!("write manifest: {e}"))?;
    // Validate the staged bundle with the same code that loads it.
    Bundle::open(staging).map_err(|e| format!("staged bundle does not validate: {e}"))?;
    Ok(manifest)
}

/// The tokenizer.json that `BertWordPieceTokenizer(vocab.txt).save()` writes:
/// BertNormalizer, BertPreTokenizer, WordPiece model, BertProcessing.
fn tokenizer_json_from_vocab(vocab_path: &Path, lowercase: bool) -> Result<String, String> {
    let text = std::fs::read_to_string(vocab_path).map_err(|e| format!("read {}: {e}", vocab_path.display()))?;
    let mut vocab = serde_json::Map::new();
    let mut ids: BTreeMap<&str, u32> = BTreeMap::new();
    for (i, line) in text.lines().enumerate() {
        let tok = line.trim_end_matches(['\r', '\n']);
        if tok.is_empty() {
            return Err(format!("{}: line {} is empty", vocab_path.display(), i + 1));
        }
        if vocab.insert(tok.to_string(), Value::from(i as u32)).is_some() {
            return Err(format!("{}: duplicate token `{tok}` at line {}", vocab_path.display(), i + 1));
        }
        ids.insert(tok, i as u32);
    }
    let special = |name: &str| -> Result<u32, String> {
        ids.get(name).copied().ok_or_else(|| format!("{}: special token {name} is missing", vocab_path.display()))
    };
    let (pad, unk, cls, sep, mask) =
        (special("[PAD]")?, special("[UNK]")?, special("[CLS]")?, special("[SEP]")?, special("[MASK]")?);
    let added = |name: &str, id: u32| serde_json::json!({"id": id, "content": name, "single_word": false, "lstrip": false, "rstrip": false, "normalized": false, "special": true});
    let mut added_tokens: Vec<(u32, Value)> = vec![
        (pad, added("[PAD]", pad)),
        (unk, added("[UNK]", unk)),
        (cls, added("[CLS]", cls)),
        (sep, added("[SEP]", sep)),
        (mask, added("[MASK]", mask)),
    ];
    added_tokens.sort_by_key(|(id, _)| *id);
    let doc = serde_json::json!({
        "version": "1.0",
        "truncation": null,
        "padding": null,
        "added_tokens": added_tokens.into_iter().map(|(_, v)| v).collect::<Vec<_>>(),
        "normalizer": {"type": "BertNormalizer", "clean_text": true, "handle_chinese_chars": true, "strip_accents": null, "lowercase": lowercase},
        "pre_tokenizer": {"type": "BertPreTokenizer"},
        "post_processor": {"type": "BertProcessing", "sep": ["[SEP]", sep], "cls": ["[CLS]", cls]},
        "decoder": {"type": "WordPiece", "prefix": "##", "cleanup": true},
        "model": {"type": "WordPiece", "unk_token": "[UNK]", "continuing_subword_prefix": "##", "max_input_chars_per_word": 100, "vocab": vocab}
    });
    serde_json::to_string(&doc).map_err(|e| format!("serialize tokenizer.json: {e}"))
}

fn detect_tokenizer_kind(path: &Path) -> Result<String, String> {
    let v = read_json(path)?.ok_or("tokenizer.json missing")?;
    let model_type = v.get("model").and_then(|m| m.get("type")).and_then(Value::as_str).unwrap_or("");
    Ok(match model_type {
        "WordPiece" => "wordpiece",
        "BPE" => "bpe",
        "Unigram" => "unigram",
        "WordLevel" => "wordlevel",
        other => return Err(format!("tokenizer.json model type `{other}` is not recognized")),
    }
    .to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fixture_tokenizer() -> PathBuf {
        Path::new(env!("CARGO_MANIFEST_DIR")).join("../../testdata/bundles/minilm-tokenizer/tokenizer.json")
    }

    fn write_st_source(dir: &Path, legacy: bool) {
        std::fs::create_dir_all(dir.join("1_Pooling")).unwrap();
        std::fs::copy(fixture_tokenizer(), dir.join("tokenizer.json")).unwrap();
        std::fs::write(
            dir.join("modules.json"),
            serde_json::json!([
                {"idx": 0, "name": "0", "path": "", "type": "sentence_transformers.models.Transformer"},
                {"idx": 1, "name": "1", "path": "1_Pooling", "type": "sentence_transformers.models.Pooling"},
                {"idx": 2, "name": "2", "path": "2_Normalize", "type": "sentence_transformers.models.Normalize"}
            ])
            .to_string(),
        )
        .unwrap();
        let pooling = if legacy {
            serde_json::json!({"word_embedding_dimension": 384, "pooling_mode_cls_token": false, "pooling_mode_mean_tokens": true, "pooling_mode_max_tokens": false})
        } else {
            serde_json::json!({"embedding_dimension": 384, "pooling_mode": "mean", "include_prompt": true})
        };
        std::fs::write(dir.join("1_Pooling/config.json"), pooling.to_string()).unwrap();
        std::fs::write(dir.join("sentence_bert_config.json"), r#"{"max_seq_length": 256, "do_lower_case": false}"#)
            .unwrap();
        std::fs::write(
            dir.join("config_sentence_transformers.json"),
            serde_json::json!({"prompts": {"query": "query: ", "document": "passage: "}, "similarity_fn_name": "cosine"}).to_string(),
        )
        .unwrap();
        std::fs::write(
            dir.join("config.json"),
            serde_json::json!({"model_type": "bert", "hidden_size": 384, "vocab_size": 30522, "max_position_embeddings": 512, "architectures": ["BertModel"]}).to_string(),
        )
        .unwrap();
        std::fs::write(dir.join("model.onnx"), b"not really onnx").unwrap();
    }

    #[test]
    fn imports_sentence_transformers_layout_both_schemas() {
        for legacy in [false, true] {
            let tmp = tempfile::tempdir().unwrap();
            let src = tmp.path().join("src");
            write_st_source(&src, legacy);
            let out = tmp.path().join("bundle");
            let report = import(&ImportRequest {
                source: src.clone(),
                output: out.clone(),
                license: Some("Apache-2.0".into()),
                artifacts: vec![("onnx".into(), src.join("model.onnx"))],
                truncate_dims: vec![256, 128],
                max_batch: 16,
                ..Default::default()
            })
            .unwrap();
            let m = &report.manifest;
            assert_eq!(m.kind, "embedding");
            assert_eq!(m.contract.pooling.as_deref(), Some("mean"));
            assert_eq!(m.contract.normalize.as_deref(), Some("l2"));
            assert_eq!(m.contract.max_seq, 256);
            assert_eq!(m.contract.dim, 384);
            assert_eq!(m.contract.prompts.query, "query: ");
            assert_eq!(m.contract.truncate_dims, vec![256, 128]);
            assert_eq!(m.tokenizer.as_ref().unwrap().kind, "wordpiece");
            assert!(m.artifacts.contains_key("onnx"));
            let b = Bundle::open(&out).unwrap();
            assert_eq!(b.manifest().model_id, "src");
            assert!(!tmp.path().join(".bundle.staging").exists());
        }
    }

    #[test]
    fn refuses_existing_output_and_missing_license() {
        let tmp = tempfile::tempdir().unwrap();
        let src = tmp.path().join("src");
        write_st_source(&src, false);
        let out = tmp.path().join("bundle");
        let e = import(&ImportRequest { source: src.clone(), output: out.clone(), ..Default::default() }).unwrap_err();
        assert!(e.contains("--license"));
        std::fs::create_dir(&out).unwrap();
        let e = import(&ImportRequest { source: src, output: out, license: Some("MIT".into()), ..Default::default() })
            .unwrap_err();
        assert!(e.contains("already exists"));
    }

    #[test]
    fn ambiguous_legacy_pooling_is_an_error() {
        let tmp = tempfile::tempdir().unwrap();
        let src = tmp.path().join("src");
        write_st_source(&src, true);
        std::fs::write(
            src.join("1_Pooling/config.json"),
            r#"{"word_embedding_dimension": 384, "pooling_mode_cls_token": true, "pooling_mode_mean_tokens": true}"#,
        )
        .unwrap();
        let e = import(&ImportRequest {
            source: src.clone(),
            output: tmp.path().join("bundle"),
            license: Some("Apache-2.0".into()),
            artifacts: vec![("onnx".into(), src.join("model.onnx"))],
            ..Default::default()
        })
        .unwrap_err();
        assert!(e.contains("several pooling modes"), "{e}");
    }

    #[test]
    fn static_table_from_safetensors() {
        let tmp = tempfile::tempdir().unwrap();
        let src = tmp.path().join("src");
        std::fs::create_dir_all(&src).unwrap();
        std::fs::copy(fixture_tokenizer(), src.join("tokenizer.json")).unwrap();
        std::fs::write(src.join("config.json"), r#"{"model_type": "model2vec", "normalize": true}"#).unwrap();
        let header =
            serde_json::json!({"embeddings": {"dtype": "F16", "shape": [30522, 4], "data_offsets": [0, 30522 * 8]}})
                .to_string();
        let mut bytes = Vec::new();
        bytes.extend_from_slice(&(header.len() as u64).to_le_bytes());
        bytes.extend_from_slice(header.as_bytes());
        bytes.resize(bytes.len() + 30522 * 8, 0);
        std::fs::write(src.join("model.safetensors"), bytes).unwrap();
        let out = tmp.path().join("bundle");
        let report = import(&ImportRequest {
            source: src.clone(),
            output: out.clone(),
            license: Some("MIT".into()),
            static_from: Some((src.join("model.safetensors"), None)),
            max_seq: Some(512),
            ..Default::default()
        })
        .unwrap();
        assert_eq!(report.manifest.kind, "embedding");
        assert_eq!(report.manifest.contract.dim, 4);
        assert_eq!(report.manifest.contract.vocab_size, 30522);
        assert_eq!(std::fs::metadata(out.join("static.f32")).unwrap().len(), 30522 * 4 * 4);
    }
}
