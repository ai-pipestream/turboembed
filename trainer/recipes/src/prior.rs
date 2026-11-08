//! A head an earlier run wrote, read back to score beside a new one
//! (`--against`): its spec, its weights, a name for it, and the files it
//! was read from with their digests. The new run checks the spec against
//! its own and builds the head from the weights.

use std::path::Path;

use turbo::head::TokenTagger;
use turbo::manifest::{HeadKind, HeadSpec};
use turbo::safetensors;

use crate::corpus::FileRecord;
use crate::receipt::EncoderRecord;
use crate::{Error, Result};

/// The files of a run directory a prior head is read from.
pub const FILES: [&str; 3] = ["head.json", "head.safetensors", "receipt.json"];

pub struct Prior {
    /// The run directory's name.
    pub name: String,
    pub spec: HeadSpec,
    pub weights: Vec<u8>,
    pub files: Vec<FileRecord>,
}

/// The head written to `dir` (`head.json`, `head.safetensors` and
/// `receipt.json`, as the trainers write them), which must have read the
/// encoder `encoder` names: the same model, revision, manifest digest,
/// width, layers and hidden size (backend, device and precision may
/// differ). Only the receipt's `encoder` is read, so a receipt from an
/// earlier trainer that wrote fields since retired still names it.
///
/// # Errors
///
/// A file missing or malformed, no encoder in the receipt, or another
/// encoder.
pub fn read(dir: &Path, encoder: &EncoderRecord) -> Result<Prior> {
    let at = |name: &str| dir.join(name).display().to_string();
    let read = |name: &str| std::fs::read(dir.join(name)).map_err(|e| Error(format!("{}: {e}", at(name))));
    let (head_json, weights, receipt_json) = (read(FILES[0])?, read(FILES[1])?, read(FILES[2])?);
    let spec: HeadSpec = serde_json::from_slice(&head_json).map_err(|e| Error(format!("{}: {e}", at(FILES[0]))))?;
    let receipt: serde_json::Value =
        serde_json::from_slice(&receipt_json).map_err(|e| Error(format!("{}: {e}", at(FILES[2]))))?;
    let theirs: Option<EncoderRecord> = receipt
        .get("encoder")
        .map(|e| serde_json::from_value(e.clone()))
        .transpose()
        .map_err(|e| Error(format!("{}: encoder: {e}", at(FILES[2]))))?;
    let same = |e: &EncoderRecord| {
        (&e.model, &e.revision, &e.manifest_sha256, e.hidden, e.layers, e.width)
            == (
                &encoder.model,
                &encoder.revision,
                &encoder.manifest_sha256,
                encoder.hidden,
                encoder.layers,
                encoder.width,
            )
    };
    if !theirs.as_ref().is_some_and(same) {
        return Err(refused(dir, "the encoder"));
    }
    let record = |path: &str, bytes: &[u8]| FileRecord {
        path: path.into(),
        size: bytes.len() as u64,
        sha256: turbo::bundle::sha256_hex(bytes),
    };
    let files = vec![record(FILES[0], &head_json), record(FILES[1], &weights), record(FILES[2], &receipt_json)];
    let name = dir.file_name().map_or("head".into(), |n| n.to_string_lossy().into_owned());
    Ok(Prior { name, spec, weights, files })
}

/// The refusal of a prior head whose `what` differs from the new run's.
pub fn refused(dir: &Path, what: &str) -> Error {
    Error(format!("{}: {what} differs from this run's", dir.display()))
}

impl Prior {
    /// The head `build` makes from the weights' tensors under the spec's
    /// tensor prefix, by suffix.
    ///
    /// # Errors
    ///
    /// Weights that do not parse, or what `build` refuses.
    pub fn head<T>(
        &self,
        build: impl FnOnce(&dyn Fn(&str) -> Option<Vec<f32>>) -> turbo::status::Result<T>,
    ) -> Result<T> {
        let file = safetensors::File::parse(FILES[1], &self.weights).map_err(|e| Error(e.message))?;
        let get = |s: &str| file.tensor(&format!("{}.{s}", self.spec.tensor_prefix)).map(|t| t.f32s());
        build(&get).map_err(|e| Error(format!("{}: {}", self.name, e.message)))
    }
}

/// The tagger an earlier `train` wrote to `dir`, to score beside a run's
/// `head`: its name with its decode, the tagger, and the files read. It
/// must read the features `head` reads: the same labels, word pooling and
/// language table, on the same encoder.
///
/// # Errors
///
/// As [`read`], or a head that reads other features.
pub fn tagger(dir: &Path, head: &HeadSpec, encoder: &EncoderRecord) -> Result<(String, TokenTagger, Vec<FileRecord>)> {
    let p = read(dir, encoder)?;
    if p.spec.kind != HeadKind::TokenTagger || p.spec.labels != head.labels {
        return Err(refused(dir, "the head's kind or labels"));
    }
    if p.spec.word_pooling != head.word_pooling {
        return Err(refused(dir, "word pooling (--pooling)"));
    }
    if p.spec.language.is_some() != head.language.is_some() {
        return Err(refused(dir, "the language table"));
    }
    let tagger = p.head(|get| TokenTagger::from_tensors(&p.spec, encoder.hidden, get))?;
    Ok((format!("{} ({:?})", p.name, p.spec.decode), tagger, p.files))
}
