//! Server-side local tokenizer for the `inferstream.v1` Tokenize/Detokenize
//! RPCs.
//!
//! When a model's config carries `tokenizer_dir` (a `tokenizer.json` file or
//! a directory containing one), the server loads a HuggingFace fast tokenizer
//! at startup and answers Tokenize/Detokenize locally — even for backends
//! that tokenize elsewhere (e.g. OVMS pipelines tokenize server-side on the
//! Model Server). Backends without a configured local tokenizer may still
//! implement [`inferstream_backend::Backend::tokenize`] themselves (the mock
//! does, for CI).

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Mutex;

use inferstream_backend::{BackendError, TokenizeOptions};
use inferstream_protocol::extension::{Encoding, Offset};
use tokenizers::Tokenizer;

use crate::config::Config;
use crate::ServerError;

/// A loaded `tokenizer.json` serving Tokenize/Detokenize for one model.
///
/// The inner tokenizer is mutated per call to apply request-scoped
/// truncation/padding, hence the mutex; these RPCs are conveniences, not the
/// inference hot path.
pub struct LocalTokenizer {
    inner: Mutex<Tokenizer>,
}

impl LocalTokenizer {
    /// Load from a `tokenizer.json` file or a directory containing one.
    pub fn from_path(path: impl AsRef<Path>) -> Result<Self, BackendError> {
        let path = path.as_ref();
        let file: PathBuf = if path.is_dir() {
            path.join("tokenizer.json")
        } else {
            path.to_path_buf()
        };
        let tokenizer = Tokenizer::from_file(&file).map_err(|e| {
            BackendError::Unavailable(format!(
                "failed to load tokenizer from {}: {e}",
                file.display()
            ))
        })?;
        Ok(Self {
            inner: Mutex::new(tokenizer),
        })
    }

    pub fn tokenize(
        &self,
        texts: &[String],
        options: &TokenizeOptions,
    ) -> Result<Vec<Encoding>, BackendError> {
        use tokenizers::{PaddingParams, PaddingStrategy, TruncationParams};

        let mut tokenizer = self.inner.lock().expect("tokenizer lock poisoned");
        let truncation = options.truncate_to.map(|len| TruncationParams {
            max_length: len,
            ..Default::default()
        });
        tokenizer
            .with_truncation(truncation)
            .map_err(|e| BackendError::InvalidRequest(format!("invalid truncation: {e}")))?;
        if options.pad_to_longest {
            tokenizer.with_padding(Some(PaddingParams {
                strategy: PaddingStrategy::BatchLongest,
                ..Default::default()
            }));
        } else {
            tokenizer.with_padding(None);
        }

        let encodings = tokenizer
            .encode_batch(texts.to_vec(), options.add_special_tokens)
            .map_err(|e| BackendError::InvalidRequest(format!("tokenization failed: {e}")))?;

        Ok(encodings
            .into_iter()
            .map(|encoding| {
                let offsets = if options.with_offsets {
                    encoding
                        .get_offsets()
                        .iter()
                        .map(|&(start, end)| Offset {
                            start: start as u32,
                            end: end as u32,
                        })
                        .collect()
                } else {
                    Vec::new()
                };
                Encoding {
                    input_ids: encoding.get_ids().to_vec(),
                    attention_mask: encoding.get_attention_mask().to_vec(),
                    tokens: encoding.get_tokens().to_vec(),
                    offsets,
                }
            })
            .collect())
    }

    pub fn detokenize(
        &self,
        sequences: &[Vec<u32>],
        skip_special_tokens: bool,
    ) -> Result<Vec<String>, BackendError> {
        let tokenizer = self.inner.lock().expect("tokenizer lock poisoned");
        sequences
            .iter()
            .map(|ids| {
                tokenizer
                    .decode(ids, skip_special_tokens)
                    .map_err(|e| BackendError::InvalidRequest(format!("decode failed: {e}")))
            })
            .collect()
    }
}

/// Model name → local tokenizer, built once at startup.
pub type TokenizerMap = HashMap<String, LocalTokenizer>;

/// Load a [`LocalTokenizer`] for every model whose config sets
/// `tokenizer_dir`. A configured-but-unloadable tokenizer fails startup —
/// same fail-fast policy as engine construction.
pub fn build_tokenizers(config: &Config) -> Result<TokenizerMap, ServerError> {
    let mut map = TokenizerMap::new();
    for model in &config.models {
        if let Some(dir) = &model.tokenizer_dir {
            let tokenizer =
                LocalTokenizer::from_path(dir).map_err(|e| ServerError::InvalidModelConfig {
                    model: model.name.clone(),
                    backend: model.backend.as_str(),
                    message: e.to_string(),
                })?;
            map.insert(model.name.clone(), tokenizer);
        }
    }
    Ok(map)
}
