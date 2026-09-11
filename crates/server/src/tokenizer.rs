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

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a tiny WordLevel tokenizer ([CLS] $A [SEP], whitespace split)
    /// and save it as `tokenizer.json` in a fresh temp directory.
    fn write_test_tokenizer() -> PathBuf {
        use tokenizers::models::wordlevel::WordLevel;
        use tokenizers::pre_tokenizers::whitespace::Whitespace;
        use tokenizers::processors::template::TemplateProcessing;

        let vocab: Vec<(String, u32)> = [
            ("[PAD]", 0u32),
            ("[CLS]", 1),
            ("[SEP]", 2),
            ("[UNK]", 3),
            ("hello", 4),
            ("world", 5),
            ("rust", 6),
            ("grpc", 7),
            ("inference", 8),
        ]
        .into_iter()
        .map(|(k, v)| (k.to_string(), v))
        .collect();
        let model = WordLevel::builder()
            .vocab(vocab.into_iter().collect())
            .unk_token("[UNK]".to_string())
            .build()
            .expect("wordlevel builds");
        let mut tokenizer = Tokenizer::new(model);
        // Register the frame tokens as specials so skip_special_tokens
        // decoding drops them (matching real HF tokenizer.json files).
        let _ = tokenizer.add_special_tokens([
            tokenizers::AddedToken::from("[PAD]", true),
            tokenizers::AddedToken::from("[CLS]", true),
            tokenizers::AddedToken::from("[SEP]", true),
            tokenizers::AddedToken::from("[UNK]", true),
        ]);
        tokenizer.with_pre_tokenizer(Some(Whitespace));
        let post = TemplateProcessing::builder()
            .try_single("[CLS] $A [SEP]")
            .expect("template parses")
            .special_tokens(vec![("[CLS]", 1), ("[SEP]", 2)])
            .build()
            .expect("post processor builds");
        tokenizer.with_post_processor(Some(post));

        let dir = std::env::temp_dir().join(format!(
            "inferstream-tok-test-{}-{:?}",
            std::process::id(),
            std::thread::current().id()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        tokenizer
            .save(dir.join("tokenizer.json"), false)
            .expect("tokenizer saves");
        dir
    }

    #[test]
    fn loads_from_directory_and_file_path() {
        let dir = write_test_tokenizer();
        assert!(LocalTokenizer::from_path(&dir).is_ok());
        assert!(LocalTokenizer::from_path(dir.join("tokenizer.json")).is_ok());
        assert!(LocalTokenizer::from_path(dir.join("missing.json")).is_err());
    }

    #[test]
    fn tokenize_round_trips_through_detokenize() {
        let tokenizer = LocalTokenizer::from_path(write_test_tokenizer()).unwrap();
        let texts = vec!["hello world".to_string(), "rust grpc inference".to_string()];
        let encodings = tokenizer
            .tokenize(&texts, &TokenizeOptions::default())
            .unwrap();
        assert_eq!(encodings[0].input_ids, vec![1, 4, 5, 2]);
        assert_eq!(
            encodings[0].tokens,
            vec!["[CLS]", "hello", "world", "[SEP]"]
        );
        assert!(encodings[0].attention_mask.iter().all(|&m| m == 1));

        let sequences: Vec<Vec<u32>> = encodings.iter().map(|e| e.input_ids.clone()).collect();
        let decoded = tokenizer.detokenize(&sequences, true).unwrap();
        assert_eq!(decoded, texts);
        // With specials kept, the frame tokens come back too.
        let raw = tokenizer.detokenize(&sequences[..1], false).unwrap();
        assert!(raw[0].contains("[CLS]") && raw[0].contains("[SEP]"));
    }

    #[test]
    fn unknown_words_map_to_unk_and_still_decode() {
        let tokenizer = LocalTokenizer::from_path(write_test_tokenizer()).unwrap();
        let encodings = tokenizer
            .tokenize(&["hello quixotic".to_string()], &TokenizeOptions::default())
            .unwrap();
        assert_eq!(encodings[0].input_ids, vec![1, 4, 3, 2]);
        assert_eq!(encodings[0].tokens[2], "[UNK]");
    }

    #[test]
    fn no_special_tokens_drops_the_frame() {
        let tokenizer = LocalTokenizer::from_path(write_test_tokenizer()).unwrap();
        let encodings = tokenizer
            .tokenize(
                &["hello world".to_string()],
                &TokenizeOptions {
                    add_special_tokens: false,
                    ..Default::default()
                },
            )
            .unwrap();
        assert_eq!(encodings[0].input_ids, vec![4, 5]);
    }

    #[test]
    fn truncation_caps_sequence_length() {
        let tokenizer = LocalTokenizer::from_path(write_test_tokenizer()).unwrap();
        let encodings = tokenizer
            .tokenize(
                &["hello world rust grpc inference".to_string()],
                &TokenizeOptions {
                    truncate_to: Some(4),
                    ..Default::default()
                },
            )
            .unwrap();
        assert_eq!(encodings[0].input_ids.len(), 4);
        // A later call without truncation is unaffected (per-request config).
        let full = tokenizer
            .tokenize(
                &["hello world rust grpc inference".to_string()],
                &TokenizeOptions::default(),
            )
            .unwrap();
        assert_eq!(full[0].input_ids.len(), 7);
    }

    #[test]
    fn batch_padding_to_longest_with_mask() {
        let tokenizer = LocalTokenizer::from_path(write_test_tokenizer()).unwrap();
        let encodings = tokenizer
            .tokenize(
                &["hello".to_string(), "hello world rust".to_string()],
                &TokenizeOptions {
                    pad_to_longest: true,
                    ..Default::default()
                },
            )
            .unwrap();
        let longest = encodings[1].input_ids.len();
        assert_eq!(encodings[0].input_ids.len(), longest);
        assert!(
            encodings[0].input_ids.ends_with(&[0, 0]),
            "padded with [PAD]"
        );
        assert!(encodings[0].attention_mask.ends_with(&[0, 0]));
        assert_eq!(
            encodings[0].attention_mask.iter().sum::<u32>(),
            3,
            "[CLS] hello [SEP]"
        );
    }

    #[test]
    fn offsets_slice_the_original_text() {
        let tokenizer = LocalTokenizer::from_path(write_test_tokenizer()).unwrap();
        let text = "hello world".to_string();
        let encodings = tokenizer
            .tokenize(
                std::slice::from_ref(&text),
                &TokenizeOptions {
                    with_offsets: true,
                    ..Default::default()
                },
            )
            .unwrap();
        let offsets = &encodings[0].offsets;
        assert_eq!(offsets.len(), encodings[0].input_ids.len());
        assert_eq!(
            &text[offsets[1].start as usize..offsets[1].end as usize],
            "hello"
        );
        assert_eq!(
            &text[offsets[2].start as usize..offsets[2].end as usize],
            "world"
        );
        // Without the flag, offsets stay empty (cheap default).
        let plain = tokenizer
            .tokenize(std::slice::from_ref(&text), &TokenizeOptions::default())
            .unwrap();
        assert!(plain[0].offsets.is_empty());
    }

    #[test]
    fn build_tokenizers_loads_configured_models_only() {
        let dir = write_test_tokenizer();
        let config = Config::from_toml(&format!(
            r#"
            [[models]]
            name = "with-tok"
            backend = "mock"
            tokenizer_dir = "{}"

            [[models]]
            name = "without-tok"
            backend = "mock"
            "#,
            dir.display()
        ))
        .unwrap();
        let map = build_tokenizers(&config).unwrap();
        assert!(map.contains_key("with-tok"));
        assert!(!map.contains_key("without-tok"));
    }

    #[test]
    fn build_tokenizers_fails_fast_on_bad_path() {
        let config = Config::from_toml(
            r#"
            [[models]]
            name = "broken"
            backend = "mock"
            tokenizer_dir = "/definitely/not/a/real/path"
            "#,
        )
        .unwrap();
        assert!(matches!(
            build_tokenizers(&config),
            Err(ServerError::InvalidModelConfig { .. })
        ));
    }
}
