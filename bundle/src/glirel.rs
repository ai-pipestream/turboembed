//! A GLiREL checkpoint (`pytorch_model.bin`, `torch.save`'s zip form) as a
//! package the native scorer reads (`turbo_relation::glirel`):
//!
//! - `encoder/`: the DeBERTa-v3 encoder as a native package, its
//!   configuration the base model's `config.json` with the vocabulary the
//!   checkpoint resized it to;
//! - `heads.safetensors`: every other tensor, under its checkpoint name;
//! - `glirel.json`: the source, the checkpoint's SHA-256, the special token
//!   ids, and the digests of the heads and of the `spm.model` it was
//!   written with.
//!
//! GLiREL's tokenizer is the base model's with `[FLERT]` added as a special
//! token and then `[REL]`, `[E]` and `[/E]` (`[SEP]` is already there):
//! the SentencePiece pieces, `[MASK]` after them, then those four. The
//! checkpoint's vocabulary must be exactly that long.

use std::fs;
use std::path::Path;

use serde_json::json;
use turbo::deberta::package::digest;
use turbo::sentencepiece::SentencePiece;

use crate::Result;

const ENCODER: &str = "token_rep_layer.bert_layer.model.";

/// The longest text GLiREL reads, in words (its `max_len`).
pub const MAX_WORDS: usize = 512;

/// Write the package for `checkpoint` into the new directory `out`.
/// `source` is the checkpoint's repository, revision and licence.
pub fn write_package(
    checkpoint: &[u8],
    base_config: &[u8],
    spm: &[u8],
    source: (&str, &str, &str),
    out: &Path,
) -> Result<()> {
    let tensors = crate::torch::read(checkpoint)?;
    let rows = tensors
        .get(&format!("{ENCODER}embeddings.word_embeddings.weight"))
        .map(|t| t.shape[0])
        .ok_or("glirel: no encoder word embeddings")?;
    let sp = SentencePiece::parse(spm).map_err(|e| e.message)?;
    let id = |p: &str| sp.id(p).ok_or_else(|| format!("glirel: spm.model has no {p}"));
    let (cls, sep) = (id("[CLS]")?, id("[SEP]")?);
    // [MASK], [FLERT], [REL], [E], [/E] after the pieces.
    if rows != sp.len() + 5 {
        return Err(format!("glirel: {rows} vocabulary rows, not spm.model's {} and five added tokens", sp.len()));
    }
    let rel = (sp.len() + 2) as u32;
    let mut config: serde_json::Value = serde_json::from_slice(base_config).map_err(|e| format!("base config: {e}"))?;
    config["vocab_size"] = json!(rows);
    let config = crate::deberta::from_huggingface(&config)?;
    let encoder = crate::torch::to_safetensors(&tensors, |k| k.strip_prefix(ENCODER).map(str::to_owned));
    let heads = crate::torch::to_safetensors(&tensors, |k| (!k.starts_with(ENCODER)).then(|| k.to_owned()));
    fs::create_dir(out).map_err(|e| format!("glirel output directory must be new: {e}"))?;
    crate::deberta::write_package(config, &encoder, "", source, &out.join("encoder"))?;
    let meta = json!({
        "version": 1,
        "repository": source.0,
        "revision": source.1,
        "license": source.2,
        "sha256": digest(checkpoint),
        "heads_sha256": digest(&heads),
        "spm_sha256": digest(spm),
        "cls": cls,
        "sep": sep,
        "rel": rel,
        "max_words": MAX_WORDS,
    });
    fs::write(out.join("heads.safetensors"), heads).map_err(|e| e.to_string())?;
    let meta = serde_json::to_vec_pretty(&meta).map_err(|e| e.to_string())?;
    fs::write(out.join("glirel.json"), meta).map_err(|e| e.to_string())
}
