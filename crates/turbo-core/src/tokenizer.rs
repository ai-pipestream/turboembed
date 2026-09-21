//! Tokenizers loaded from bundles.
//!
//! The general path is the Hugging Face `tokenizers` crate (Apache-2.0),
//! which reads `tokenizer.json` for WordPiece, BPE, and Unigram models and
//! reproduces the reference token ids exactly. Providers with a native fast
//! path (the C++ WordPiece write-through) verify their ids against this
//! implementation in the conformance suite.
//!
//! Encoding writes directly into caller-owned `[rows, row_stride]` arrays so
//! a session's input buffers can be filled without an intermediate copy.
//! Padding, truncation, special tokens, and prompt prefixes follow the
//! caller's options, never a hidden default: a text that exceeds the budget
//! with truncation `NONE` is an error.

use std::path::Path;
use std::sync::Arc;

use tokenizers::{
    EncodeInput, InputSequence, Tokenizer as HfTokenizer, TruncationDirection, TruncationParams, TruncationStrategy,
};

use crate::bundle::Bundle;
use crate::error::{Error, Result};
use crate::types::{PromptRole, Truncate};

/// Encode options.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct EncodeOptions {
    /// Add the model's special tokens (`[CLS]`/`[SEP]`, BOS/EOS).
    pub add_special_tokens: bool,
    /// Truncation policy. `Model` means the bundle's default: `Right`.
    pub truncate: Truncate,
    /// Token budget including specials; 0 = the bundle's `max_seq`.
    pub max_tokens: u32,
    /// Pad every row to this many tokens; 0 = pad to `row_stride` on write-through, none on `encode`.
    pub pad_to: u32,
    /// Prompt prefix role.
    pub prompt_role: PromptRole,
}

impl Default for EncodeOptions {
    fn default() -> Self {
        Self {
            add_special_tokens: true,
            truncate: Truncate::Model,
            max_tokens: 0,
            pad_to: 0,
            prompt_role: PromptRole::None,
        }
    }
}

/// Static facts about a tokenizer.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TokenizerInfo {
    /// Vocabulary size including added tokens.
    pub vocab_size: u32,
    /// Bundle `max_seq`.
    pub max_seq: u32,
    /// Pad token id, if the tokenizer defines one.
    pub pad_id: Option<i32>,
    /// Beginning-of-sequence / `[CLS]` id, if defined.
    pub bos_id: Option<i32>,
    /// End-of-sequence / `[SEP]` id, if defined.
    pub eos_id: Option<i32>,
    /// Unknown token id, if defined.
    pub unk_id: Option<i32>,
    /// Number of special tokens added to a single sequence.
    pub specials_per_sequence: u32,
    /// Tokenizer kind from the bundle (`wordpiece`, `bpe`, `unigram`, ...).
    pub kind: String,
    /// Hex SHA-256 of the tokenizer file.
    pub sha256: String,
}

/// Caller-owned destination arrays for [`Tokenizer::encode_into`]: row-major
/// `[rows, row_stride]` ids and mask, optional type ids, and per-row lengths.
pub struct EncodeTarget<'a> {
    /// Token ids.
    pub ids: &'a mut [i32],
    /// Attention mask.
    pub mask: &'a mut [i32],
    /// Token type ids, or `None`.
    pub types: Option<&'a mut [i32]>,
    /// Elements between row starts.
    pub row_stride: usize,
    /// Live token count per row.
    pub lengths: &'a mut [u32],
}

/// One encoded text.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Encoding {
    /// Token ids.
    pub ids: Vec<i32>,
    /// Token type ids.
    pub type_ids: Vec<i32>,
    /// Attention mask (all ones; padding is applied at write-through).
    pub mask: Vec<i32>,
    /// Byte offsets `(start, end)` into the prefixed input per token; specials are `(0, 0)`.
    pub offsets: Vec<(u32, u32)>,
    /// True if the input was truncated.
    pub truncated: bool,
}

/// A tokenizer. Thread-safe; encode calls do not mutate it.
pub struct Tokenizer {
    inner: HfTokenizer,
    info: TokenizerInfo,
    prefix_query: String,
    prefix_document: String,
}

impl std::fmt::Debug for Tokenizer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Tokenizer").field("info", &self.info).finish()
    }
}

impl Tokenizer {
    /// Load the tokenizer a bundle declares (`tokenizer.files["tokenizer.json"]`).
    pub fn from_bundle(bundle: &Bundle) -> Result<Arc<Self>> {
        let spec = bundle.manifest().tokenizer.as_ref().ok_or_else(|| {
            Error::bundle_invalid(format!("bundle `{}` declares no tokenizer", bundle.manifest().model_id))
        })?;
        let entry = spec.files.get("tokenizer.json").ok_or_else(|| {
            Error::bundle_invalid(format!(
                "bundle `{}` tokenizer kind `{}` has no `tokenizer.json` file entry; only Hugging Face tokenizer.json is supported by the core tokenizer",
                bundle.manifest().model_id, spec.kind
            ))
        })?;
        let path = bundle.resolve(&entry.path)?;
        let c = bundle.contract();
        let max_seq = if c.max_seq == 0 { 512 } else { c.max_seq };
        Self::from_file(&path, &spec.kind, &entry.sha256, max_seq, &c.prompts.query, &c.prompts.document)
    }

    /// Load from a `tokenizer.json` path with explicit contract values.
    pub fn from_file(
        path: &Path,
        kind: &str,
        sha256: &str,
        max_seq: u32,
        prefix_query: &str,
        prefix_document: &str,
    ) -> Result<Arc<Self>> {
        if max_seq == 0 {
            return Err(Error::invalid_argument("max_seq must be non-zero"));
        }
        let mut inner = HfTokenizer::from_file(path)
            .map_err(|e| Error::bundle_invalid(format!("tokenizer `{}`: {e}", path.display())))?;
        // The library controls truncation and padding per call; disable any
        // defaults baked into the file so behavior never depends on it.
        inner.with_truncation(None).map_err(|e| Error::internal(format!("tokenizer truncation reset: {e}")))?;
        inner.with_padding(None);
        let vocab_size = inner.get_vocab_size(true) as u32;
        let id_of = |tok: &str| inner.token_to_id(tok).map(|i| i as i32);
        let pad_id = ["[PAD]", "<pad>", "<|endoftext|>"].iter().find_map(|t| id_of(t));
        let bos_id = ["[CLS]", "<s>", "<bos>", "<|im_start|>"].iter().find_map(|t| id_of(t));
        let eos_id = ["[SEP]", "</s>", "<eos>", "<|im_end|>", "<|endoftext|>"].iter().find_map(|t| id_of(t));
        let unk_id = ["[UNK]", "<unk>"].iter().find_map(|t| id_of(t));
        let specials_per_sequence = inner
            .encode(EncodeInput::Single(InputSequence::Raw("".into())), true)
            .map_err(|e| Error::internal(format!("tokenizer probe: {e}")))?
            .get_ids()
            .len() as u32;
        let info = TokenizerInfo {
            vocab_size,
            max_seq,
            pad_id,
            bos_id,
            eos_id,
            unk_id,
            specials_per_sequence,
            kind: kind.to_string(),
            sha256: sha256.to_string(),
        };
        Ok(Arc::new(Self {
            inner,
            info,
            prefix_query: prefix_query.to_string(),
            prefix_document: prefix_document.to_string(),
        }))
    }

    /// Static facts.
    pub fn info(&self) -> &TokenizerInfo {
        &self.info
    }

    fn prefix(&self, role: PromptRole) -> &str {
        match role {
            PromptRole::None => "",
            PromptRole::Query => &self.prefix_query,
            PromptRole::Document => &self.prefix_document,
        }
    }

    fn budget(&self, opts: &EncodeOptions) -> Result<usize> {
        let b = if opts.max_tokens == 0 { self.info.max_seq } else { opts.max_tokens };
        if b > self.info.max_seq {
            return Err(Error::capacity(format!(
                "max_tokens {b} exceeds the tokenizer's max_seq {}",
                self.info.max_seq
            )));
        }
        if opts.add_special_tokens && b < self.info.specials_per_sequence {
            return Err(Error::capacity(format!(
                "max_tokens {b} is below the {} special tokens a sequence needs",
                self.info.specials_per_sequence
            )));
        }
        Ok(b as usize)
    }

    /// Encode one text. Applies the prefix for `opts.prompt_role`, special
    /// tokens, and truncation. Over-budget input with truncation `None` is
    /// `TURBO_E_CAPACITY`.
    pub fn encode(&self, text: &str, opts: &EncodeOptions) -> Result<Encoding> {
        let budget = self.budget(opts)?;
        let prefix = self.prefix(opts.prompt_role);
        let owned;
        let input: &str = if prefix.is_empty() {
            text
        } else {
            owned = format!("{prefix}{text}");
            &owned
        };
        let mut tk = self.inner.clone_for_call();
        let direction = match opts.truncate {
            Truncate::Left => TruncationDirection::Left,
            _ => TruncationDirection::Right,
        };
        if opts.truncate != Truncate::None {
            tk.with_truncation(Some(TruncationParams {
                max_length: budget,
                strategy: TruncationStrategy::LongestFirst,
                stride: 0,
                direction,
            }))
            .map_err(|e| Error::internal(format!("tokenizer truncation: {e}")))?;
        }
        let enc = tk
            .encode(EncodeInput::Single(InputSequence::Raw(input.into())), opts.add_special_tokens)
            .map_err(|e| Error::runtime(format!("tokenizer encode: {e}")))?;
        let n = enc.get_ids().len();
        if n > budget {
            return Err(Error::capacity(format!(
                "input tokenizes to {n} tokens but the budget is {budget} and truncation is NONE"
            )));
        }
        let mut out = Encoding {
            ids: Vec::with_capacity(n),
            type_ids: Vec::with_capacity(n),
            mask: vec![1; n],
            offsets: Vec::with_capacity(n),
            truncated: !enc.get_overflowing().is_empty(),
        };
        for (i, &id) in enc.get_ids().iter().enumerate() {
            if id > i32::MAX as u32 {
                return Err(Error::internal(format!("token id {id} exceeds i32")));
            }
            out.ids.push(id as i32);
            out.type_ids.push(enc.get_type_ids()[i] as i32);
            let (s, e) = enc.get_offsets()[i];
            out.offsets.push((s as u32, e as u32));
        }
        Ok(out)
    }

    /// Encode `texts` directly into caller-owned `[texts.len(), row_stride]`
    /// row-major arrays, padding each row to `pad_to` (or `row_stride` when
    /// `pad_to` is 0) with `pad_id` and mask 0. Returns each row's live length.
    /// `types` may be `None`. The arrays must hold at least
    /// `(texts.len() - 1) * row_stride + row_stride` elements.
    pub fn encode_into(&self, texts: &[&str], opts: &EncodeOptions, target: EncodeTarget<'_>) -> Result<()> {
        let EncodeTarget { ids, mask, mut types, row_stride, lengths } = target;
        let budget = self.budget(opts)?;
        if row_stride < budget {
            return Err(Error::capacity(format!("row_stride {row_stride} is smaller than the token budget {budget}")));
        }
        let pad_to = if opts.pad_to == 0 { row_stride } else { opts.pad_to as usize };
        if pad_to > row_stride {
            return Err(Error::capacity(format!("pad_to {pad_to} exceeds row_stride {row_stride}")));
        }
        let need = texts.len() * row_stride;
        if ids.len() < need || mask.len() < need || types.as_ref().is_some_and(|t| t.len() < need) {
            return Err(Error::capacity(format!(
                "output arrays need {need} elements for {} rows of stride {row_stride}",
                texts.len()
            )));
        }
        if lengths.len() < texts.len() {
            return Err(Error::capacity("lengths array is shorter than the number of texts"));
        }
        let pad = self.info.pad_id.unwrap_or(0);
        for (r, text) in texts.iter().enumerate() {
            let enc = self.encode(text, opts)?;
            let n = enc.ids.len();
            let base = r * row_stride;
            ids[base..base + n].copy_from_slice(&enc.ids);
            mask[base..base + n].copy_from_slice(&enc.mask);
            if let Some(t) = types.as_deref_mut() {
                t[base..base + n].copy_from_slice(&enc.type_ids);
            }
            let fill_to = n.max(pad_to);
            for c in n..fill_to {
                ids[base + c] = pad;
                mask[base + c] = 0;
            }
            if let Some(t) = types.as_deref_mut() {
                for c in n..fill_to {
                    t[base + c] = 0;
                }
            }
            lengths[r] = n as u32;
        }
        Ok(())
    }

    /// Decode ids to text.
    pub fn decode(&self, ids: &[i32], skip_special_tokens: bool) -> Result<String> {
        let mut u: Vec<u32> = Vec::with_capacity(ids.len());
        for &id in ids {
            if id < 0 || id as u32 >= self.info.vocab_size {
                return Err(Error::invalid_argument(format!("token id {id} is outside 0..{}", self.info.vocab_size)));
            }
            u.push(id as u32);
        }
        self.inner.decode(&u, skip_special_tokens).map_err(|e| Error::runtime(format!("tokenizer decode: {e}")))
    }

    /// Number of tokens `text` produces (no truncation, no prefix).
    pub fn count(&self, text: &str, add_special_tokens: bool) -> Result<u32> {
        let enc = self
            .inner
            .encode(EncodeInput::Single(InputSequence::Raw(text.into())), add_special_tokens)
            .map_err(|e| Error::runtime(format!("tokenizer encode: {e}")))?;
        Ok(enc.get_ids().len() as u32)
    }
}

/// Cheap per-call clone so truncation can be set without a lock; the
/// heavyweight model is shared behind `Arc`s inside `tokenizers`.
trait CloneForCall {
    fn clone_for_call(&self) -> HfTokenizer;
}

impl CloneForCall for HfTokenizer {
    fn clone_for_call(&self) -> HfTokenizer {
        self.clone()
    }
}

impl crate::chunker::TokenCounter for Tokenizer {
    fn count_tokens(&self, text: &str) -> usize {
        // Chunk budgets exclude specials (they are `reserved_tokens`).
        self.count(text, false).map(|n| n as usize).unwrap_or(usize::MAX)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn minilm() -> Arc<Tokenizer> {
        let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../testdata/bundles/minilm-tokenizer/tokenizer.json");
        Tokenizer::from_file(&root, "wordpiece", "", 128, "query: ", "passage: ").unwrap()
    }

    #[test]
    fn minilm_reference_ids() {
        let t = minilm();
        let e = t.encode("hello world", &EncodeOptions::default()).unwrap();
        assert_eq!(e.ids, vec![101, 7592, 2088, 102]);
        assert_eq!(e.mask, vec![1, 1, 1, 1]);
        assert_eq!(e.type_ids, vec![0, 0, 0, 0]);
        assert_eq!(e.offsets[1], (0, 5));
        assert_eq!(t.info().bos_id, Some(101));
        assert_eq!(t.info().eos_id, Some(102));
        assert_eq!(t.info().pad_id, Some(0));
        assert_eq!(t.info().specials_per_sequence, 2);
        assert_eq!(t.decode(&[7592, 2088], true).unwrap(), "hello world");
    }

    #[test]
    fn prefix_and_truncation() {
        let t = minilm();
        let q = t.encode("hello", &EncodeOptions { prompt_role: PromptRole::Query, ..Default::default() }).unwrap();
        assert_eq!(&q.ids[..3], &[101, 23032, 1024]); // query :
        let long = "word ".repeat(300);
        let none = t.encode(&long, &EncodeOptions { truncate: Truncate::None, ..Default::default() }).unwrap_err();
        assert_eq!(none.code(), turbo_abi::TURBO_E_CAPACITY);
        let right = t.encode(&long, &EncodeOptions { max_tokens: 8, ..Default::default() }).unwrap();
        assert_eq!(right.ids.len(), 8);
        assert!(right.truncated);
        assert_eq!(*right.ids.last().unwrap(), 102);
        let left = t
            .encode(
                "a b c d e f g h i j",
                &EncodeOptions { max_tokens: 4, truncate: Truncate::Left, ..Default::default() },
            )
            .unwrap();
        assert_eq!(left.ids.len(), 4);
        assert_eq!(t.decode(&left.ids, true).unwrap(), "i j");
    }

    #[test]
    fn write_through_pads_rows() {
        let t = minilm();
        let mut ids = vec![-1i32; 2 * 8];
        let mut mask = vec![-1i32; 2 * 8];
        let mut lengths = [0u32; 2];
        t.encode_into(
            &["hello world", "hi"],
            &EncodeOptions { max_tokens: 8, ..Default::default() },
            EncodeTarget { ids: &mut ids, mask: &mut mask, types: None, row_stride: 8, lengths: &mut lengths },
        )
        .unwrap();
        assert_eq!(lengths, [4, 3]);
        assert_eq!(&ids[..8], &[101, 7592, 2088, 102, 0, 0, 0, 0]);
        assert_eq!(&mask[..8], &[1, 1, 1, 1, 0, 0, 0, 0]);
        assert_eq!(&ids[8..11], &[101, 7632, 102]);
        assert_eq!(&mask[8..], &[1, 1, 1, 0, 0, 0, 0, 0]);
        let short = t.encode_into(
            &["x"],
            &EncodeOptions { max_tokens: 8, ..Default::default() },
            EncodeTarget { ids: &mut ids[..4], mask: &mut mask, types: None, row_stride: 8, lengths: &mut lengths },
        );
        assert_eq!(short.unwrap_err().code(), turbo_abi::TURBO_E_CAPACITY);
    }

    #[test]
    fn decode_rejects_out_of_vocab() {
        let t = minilm();
        assert_eq!(t.decode(&[999_999], true).unwrap_err().code(), turbo_abi::TURBO_E_INVALID_ARGUMENT);
    }
}
