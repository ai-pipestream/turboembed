//! Tokenizers loaded from bundles.
//!
//! The default path is native: the WordPiece tokenizer in
//! [`crate::wordpiece`] for BERT-family `tokenizer.json` files and the
//! byte-level BPE tokenizer in [`crate::bpe`] for the GPT-2 family (Qwen,
//! GPT-2, RoBERTa, Llama 3), with no dependency beyond `serde_json`; both
//! reproduce the Hugging Face `tokenizers` crate's ids and byte offsets.
//! With the `hf-tokenizers` feature the Hugging Face crate (Apache-2.0)
//! serves the files the native path does not (Unigram, sentencepiece BPE
//! with byte fallback); without it, such a file is `TURBO_E_UNSUPPORTED`
//! naming what it declares. Providers with a native fast path (the C++ WordPiece
//! write-through) verify their ids against this implementation in the
//! conformance suite.
//!
//! Encoding writes directly into caller-owned `[rows, row_stride]` arrays so
//! a session's input buffers can be filled without an intermediate copy.
//! Padding, truncation, special tokens, and prompt prefixes follow the
//! caller's options, never a hidden default: a text that exceeds the budget
//! with truncation `NONE` is an error.

use std::path::Path;
use std::sync::Arc;

use crate::bpe::Bpe;
use crate::bundle::Bundle;
use crate::error::{Error, Result};
use crate::types::{PromptRole, Truncate};
use crate::wordpiece::WordPiece;

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

/// The implementation behind a tokenizer.
enum Backend {
    Native(WordPiece),
    /// Boxed: the BPE state is several times the WordPiece state.
    Bpe(Box<Bpe>),
    #[cfg(feature = "hf-tokenizers")]
    Hf(tokenizers::Tokenizer),
}

/// A tokenizer. Thread-safe; encode calls do not mutate it.
pub struct Tokenizer {
    backend: Backend,
    info: TokenizerInfo,
    prefix_query: String,
    prefix_document: String,
}

impl std::fmt::Debug for Tokenizer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Tokenizer").field("info", &self.info).finish()
    }
}

/// The token strings a tokenizer file may use for its specials, in the
/// order they are looked up.
const PAD_TOKENS: &[&str] = &["[PAD]", "<pad>", "<|endoftext|>"];
const BOS_TOKENS: &[&str] = &["[CLS]", "<s>", "<bos>", "<|im_start|>"];
const EOS_TOKENS: &[&str] = &["[SEP]", "</s>", "<eos>", "<|im_end|>", "<|endoftext|>"];
const UNK_TOKENS: &[&str] = &["[UNK]", "<unk>"];

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
        let what = format!("tokenizer `{}`", path.display());
        let text = std::fs::read_to_string(path).map_err(|e| Error::bundle_invalid(format!("{what}: {e}")))?;
        // WordPiece, then byte-level BPE; a file neither serves goes to the
        // Hugging Face crate when it is built in, else its reason is the error.
        let backend = match WordPiece::from_json(&text, &what) {
            Ok(wp) => Backend::Native(wp),
            Err(e) if e.code() == turbo_abi::TURBO_E_UNSUPPORTED => match Bpe::from_json(&text, &what) {
                Ok(bpe) => Backend::Bpe(Box::new(bpe)),
                Err(e) if e.code() == turbo_abi::TURBO_E_UNSUPPORTED => Self::other_backend(&text, &what, e)?,
                Err(e) => return Err(e),
            },
            Err(e) => return Err(e),
        };
        let info = Self::info_of(&backend, kind, sha256, max_seq)?;
        Ok(Arc::new(Self {
            backend,
            info,
            prefix_query: prefix_query.to_string(),
            prefix_document: prefix_document.to_string(),
        }))
    }

    #[cfg(feature = "hf-tokenizers")]
    fn other_backend(text: &str, what: &str, _native: Error) -> Result<Backend> {
        let mut inner = tokenizers::Tokenizer::from_bytes(text.as_bytes())
            .map_err(|e| Error::bundle_invalid(format!("{what}: {e}")))?;
        // The library controls truncation and padding per call; disable any
        // defaults baked into the file so behavior never depends on it.
        inner.with_truncation(None).map_err(|e| Error::internal(format!("tokenizer truncation reset: {e}")))?;
        inner.with_padding(None);
        Ok(Backend::Hf(inner))
    }

    #[cfg(not(feature = "hf-tokenizers"))]
    fn other_backend(_text: &str, _what: &str, native: Error) -> Result<Backend> {
        Err(native)
    }

    fn info_of(backend: &Backend, kind: &str, sha256: &str, max_seq: u32) -> Result<TokenizerInfo> {
        let (vocab_size, id_of, specials): (u32, Box<dyn Fn(&str) -> Option<i32> + '_>, u32) = match backend {
            Backend::Native(wp) => (wp.vocab_size(), Box::new(|t| wp.token_to_id(t)), wp.specials_per_sequence()),
            Backend::Bpe(bpe) => (bpe.vocab_size(), Box::new(|t| bpe.token_to_id(t)), bpe.specials_per_sequence()),
            #[cfg(feature = "hf-tokenizers")]
            Backend::Hf(hf) => {
                let n = hf
                    .encode(tokenizers::EncodeInput::Single(tokenizers::InputSequence::Raw("".into())), true)
                    .map_err(|e| Error::internal(format!("tokenizer probe: {e}")))?
                    .get_ids()
                    .len() as u32;
                (hf.get_vocab_size(true) as u32, Box::new(|t| hf.token_to_id(t).map(|i| i as i32)), n)
            }
        };
        Ok(TokenizerInfo {
            vocab_size,
            max_seq,
            pad_id: PAD_TOKENS.iter().find_map(|t| id_of(t)),
            bos_id: BOS_TOKENS.iter().find_map(|t| id_of(t)),
            eos_id: EOS_TOKENS.iter().find_map(|t| id_of(t)),
            unk_id: UNK_TOKENS.iter().find_map(|t| id_of(t)),
            specials_per_sequence: specials,
            kind: kind.to_string(),
            sha256: sha256.to_string(),
        })
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

    /// The full, untruncated encoding of `input` (prefix already applied).
    fn encode_full(&self, input: &str, add_special_tokens: bool) -> Result<Encoding> {
        match &self.backend {
            Backend::Native(wp) => {
                let content = wp.tokenize(input);
                let (ids, type_ids, offsets) = wp.apply_template(&content, add_special_tokens);
                let n = ids.len();
                Ok(Encoding { ids, type_ids, mask: vec![1; n], offsets, truncated: false })
            }
            Backend::Bpe(bpe) => {
                let content = bpe.tokenize(input);
                let (ids, type_ids, offsets) = bpe.apply_template(&content, add_special_tokens);
                let n = ids.len();
                Ok(Encoding { ids, type_ids, mask: vec![1; n], offsets, truncated: false })
            }
            #[cfg(feature = "hf-tokenizers")]
            Backend::Hf(hf) => {
                let enc = hf
                    .encode(
                        tokenizers::EncodeInput::Single(tokenizers::InputSequence::Raw(input.into())),
                        add_special_tokens,
                    )
                    .map_err(|e| Error::runtime(format!("tokenizer encode: {e}")))?;
                let n = enc.get_ids().len();
                let mut out = Encoding {
                    ids: Vec::with_capacity(n),
                    type_ids: Vec::with_capacity(n),
                    mask: vec![1; n],
                    offsets: Vec::with_capacity(n),
                    truncated: false,
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
        }
    }

    /// Whether `id` at position `i` of a full encoding is one of the
    /// template's special tokens (kept whole by truncation).
    fn is_template_special(&self, enc: &Encoding, i: usize) -> bool {
        match &self.backend {
            Backend::Native(wp) => enc.offsets[i] == (0, 0) && wp.is_special(enc.ids[i]),
            Backend::Bpe(bpe) => enc.offsets[i] == (0, 0) && bpe.is_special(enc.ids[i]),
            #[cfg(feature = "hf-tokenizers")]
            Backend::Hf(_) => {
                enc.offsets[i] == (0, 0)
                    && (Some(enc.ids[i]) == self.info.bos_id || Some(enc.ids[i]) == self.info.eos_id)
            }
        }
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
        let mut enc = self.encode_full(input, opts.add_special_tokens)?;
        let n = enc.ids.len();
        if n <= budget {
            return Ok(enc);
        }
        if opts.truncate == Truncate::None {
            return Err(Error::capacity(format!(
                "input tokenizes to {n} tokens but the budget is {budget} and truncation is NONE"
            )));
        }
        // Truncation keeps the template's specials and cuts content from
        // the end (Right, the model default) or the start (Left): the same
        // rule as the Hugging Face crate's LongestFirst on one sequence.
        let leading = (0..n).take_while(|&i| self.is_template_special(&enc, i)).count();
        let trailing = (0..n).rev().take_while(|&i| self.is_template_special(&enc, i)).count();
        let content = n - leading - trailing;
        let keep = budget.saturating_sub(leading + trailing);
        let cut = content - keep;
        let (drop_from, drop_to) = match opts.truncate {
            Truncate::Left => (leading, leading + cut),
            _ => (leading + keep, leading + content),
        };
        enc.ids.drain(drop_from..drop_to);
        enc.type_ids.drain(drop_from..drop_to);
        enc.mask.drain(drop_from..drop_to);
        enc.offsets.drain(drop_from..drop_to);
        enc.truncated = true;
        Ok(enc)
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
        for &id in ids {
            if id < 0 || id as u32 >= self.info.vocab_size {
                return Err(Error::invalid_argument(format!("token id {id} is outside 0..{}", self.info.vocab_size)));
            }
        }
        match &self.backend {
            Backend::Native(wp) => wp.decode(ids, skip_special_tokens),
            Backend::Bpe(bpe) => bpe.decode(ids, skip_special_tokens),
            #[cfg(feature = "hf-tokenizers")]
            Backend::Hf(hf) => {
                let u: Vec<u32> = ids.iter().map(|&i| i as u32).collect();
                hf.decode(&u, skip_special_tokens).map_err(|e| Error::runtime(format!("tokenizer decode: {e}")))
            }
        }
    }

    /// Number of tokens `text` produces (no truncation, no prefix).
    pub fn count(&self, text: &str, add_special_tokens: bool) -> Result<u32> {
        Ok(self.encode_full(text, add_special_tokens)?.ids.len() as u32)
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

    fn minilm_path() -> std::path::PathBuf {
        Path::new(env!("CARGO_MANIFEST_DIR")).join("../../testdata/bundles/minilm-tokenizer/tokenizer.json")
    }

    fn minilm() -> Arc<Tokenizer> {
        Tokenizer::from_file(&minilm_path(), "wordpiece", "", 128, "query: ", "passage: ").unwrap()
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

    #[test]
    fn normalization_follows_the_bert_rules() {
        let t = minilm();
        let opts = EncodeOptions { add_special_tokens: false, ..Default::default() };
        // Accents are stripped and case folded: "Café" is "cafe".
        let e = t.encode("Café", &opts).unwrap();
        assert_eq!(t.decode(&e.ids, true).unwrap(), "cafe");
        assert_eq!(e.offsets, vec![(0, 5)], "the token spans the accented source bytes");
        // Punctuation is its own token and offsets point at it.
        let e = t.encode("hi, there!", &opts).unwrap();
        assert_eq!(t.decode(&e.ids, true).unwrap(), "hi, there!");
        assert_eq!(e.offsets, vec![(0, 2), (2, 3), (4, 9), (9, 10)]);
        // A subword split reports the offsets of each piece.
        let e = t.encode("turboembedding", &opts).unwrap();
        assert!(e.ids.len() > 1, "{:?}", e.ids);
        assert_eq!(e.offsets.first().unwrap().0, 0);
        assert_eq!(e.offsets.last().unwrap().1, 14);
        assert_eq!(t.decode(&e.ids, true).unwrap(), "turboembedding");
        // CJK characters are one token each; a control character vanishes.
        let e = t.encode("你好\u{0}x", &opts).unwrap();
        assert_eq!(e.ids.len(), 3);
        assert_eq!(e.offsets[2], (7, 8));
        // Special tokens in the text are matched verbatim.
        let e = t.encode("[SEP] a", &opts).unwrap();
        assert_eq!(e.ids[0], 102);
        // An unknown script maps to [UNK] with the word's offsets.
        let e = t.encode("\u{0e01}\u{0e02}", &opts).unwrap();
        assert_eq!(e.ids, vec![100]);
    }

    /// With the Hugging Face crate available, the native tokenizer must
    /// agree with it on ids, type ids and offsets over the STS corpus, the
    /// reference texts and a set of adversarial strings.
    #[cfg(feature = "hf-tokenizers")]
    #[test]
    fn native_matches_hugging_face() {
        let native = minilm();
        let mut hf = tokenizers::Tokenizer::from_file(minilm_path()).unwrap();
        hf.with_truncation(None).unwrap();
        hf.with_padding(None);
        let corpus_path = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../testdata/corpus/sts-pairs.jsonl");
        let mut texts: Vec<String> = Vec::new();
        for line in std::fs::read_to_string(&corpus_path).unwrap().lines().filter(|l| !l.trim().is_empty()) {
            let v: serde_json::Value = serde_json::from_str(line).unwrap();
            texts.push(v["text_a"].as_str().unwrap().to_string());
            texts.push(v["text_b"].as_str().unwrap().to_string());
        }
        texts.extend(
            [
                "",
                " ",
                "Hello, World!",
                "naïve café résumé Ångström",
                "ΑΒΓ αβγ Straße İstanbul",
                "日本語のテキスト and 中文",
                "e-mail: a.b@c.com (see http://x.y/z?q=1&r=2)",
                "tabs\tand\nnewlines\r\n  and   runs   of   spaces",
                "emoji 😀 and \u{200b} zero width and \u{0301} lone mark",
                "[CLS] embedded [SEP] specials [MASK] [UNK] [PAD]",
                "超长的中文句子用来检查每个字符都是独立的词元",
                "a".repeat(150).as_str(),
                "ﬁ ligature ㎡ squared Ⅻ roman ①②③",
                "'Tis n't 's 're 've 'm quotes' and \"double\"",
            ]
            .iter()
            .map(|s| s.to_string()),
        );
        let opts = EncodeOptions { truncate: Truncate::None, max_tokens: 128, ..Default::default() };
        let mut compared = 0;
        for text in &texts {
            let expect = hf
                .encode(tokenizers::EncodeInput::Single(tokenizers::InputSequence::Raw(text.as_str().into())), true)
                .unwrap();
            if expect.get_ids().len() > 128 {
                continue;
            }
            let got = native.encode(text, &opts).unwrap();
            let want_ids: Vec<i32> = expect.get_ids().iter().map(|&i| i as i32).collect();
            assert_eq!(got.ids, want_ids, "ids differ for {text:?}");
            let want_types: Vec<i32> = expect.get_type_ids().iter().map(|&i| i as i32).collect();
            assert_eq!(got.type_ids, want_types, "type ids differ for {text:?}");
            let want_offsets: Vec<(u32, u32)> =
                expect.get_offsets().iter().map(|&(s, e)| (s as u32, e as u32)).collect();
            assert_eq!(got.offsets, want_offsets, "offsets differ for {text:?}");
            let want_text = hf.decode(expect.get_ids(), true).unwrap();
            assert_eq!(native.decode(&got.ids, true).unwrap(), want_text, "decode differs for {text:?}");
            compared += 1;
        }
        assert!(compared > 150, "compared only {compared} texts");
    }

    /// The byte-level BPE file the BPE tests use: `TURBO_BPE_TOKENIZER`, or
    /// Qwen3-Embedding-0.6B's `tokenizer.json` under `~/opt/models`. The
    /// file is 11 MB, so it is not a fixture in the tree; a machine without
    /// it skips these tests and says so.
    pub(super) fn bpe_path() -> Option<std::path::PathBuf> {
        let path = match std::env::var_os("TURBO_BPE_TOKENIZER") {
            Some(p) => std::path::PathBuf::from(p),
            None => {
                let home = std::env::var_os("HOME")?;
                Path::new(&home).join("opt/models/qwen3-embed-0.6b/tokenizer.json")
            }
        };
        if path.is_file() {
            Some(path)
        } else {
            eprintln!("skipped: no byte-level BPE tokenizer at {} (set TURBO_BPE_TOKENIZER)", path.display());
            None
        }
    }

    #[cfg(feature = "hf-tokenizers")]
    fn bpe_texts() -> Vec<String> {
        let mut texts: Vec<String> = Vec::new();
        for name in ["sts-pairs.jsonl", "multilingual.jsonl"] {
            let corpus_path = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../testdata/corpus").join(name);
            for line in std::fs::read_to_string(&corpus_path).unwrap().lines().filter(|l| !l.trim().is_empty()) {
                let v: serde_json::Value = serde_json::from_str(line).unwrap();
                for key in ["text", "text_a", "text_b"] {
                    if let Some(t) = v[key].as_str() {
                        texts.push(t.to_string());
                    }
                }
            }
        }
        texts
    }

    #[test]
    fn qwen_reference_ids() {
        let Some(path) = bpe_path() else { return };
        let t = Tokenizer::from_file(&path, "bpe", "", 512, "", "").unwrap();
        let opts = EncodeOptions { truncate: Truncate::None, max_tokens: 512, ..Default::default() };
        // Qwen3's ids for "hello world" and the template's <|endoftext|>.
        let e = t.encode("hello world", &opts).unwrap();
        assert_eq!(e.ids, vec![14990, 1879, 151643]);
        assert_eq!(e.offsets, vec![(0, 5), (5, 11), (0, 0)]);
        assert_eq!(e.type_ids, vec![0, 0, 0]);
        assert_eq!(t.info().specials_per_sequence, 1);
        assert_eq!(t.info().pad_id, Some(151643));
        assert_eq!(t.info().vocab_size, 151669);
        assert_eq!(t.decode(&e.ids, true).unwrap(), "hello world");
        // Digits are one token each; a special token in the text is itself.
        let e = t.encode("2026 <|im_end|>", &opts).unwrap();
        assert_eq!(e.ids, vec![17, 15, 17, 21, 220, 151645, 151643]);
        // NFC: a decomposed e + acute is the composed character's bytes.
        let composed = t.encode("caf\u{e9}", &opts).unwrap();
        let decomposed = t.encode("cafe\u{301}", &opts).unwrap();
        assert_eq!(composed.ids, decomposed.ids);
        assert_eq!(decomposed.offsets.last().unwrap(), &(0, 0));
        // The composed character keeps the base letter's bytes; the mark's
        // two bytes belong to no token (the Hugging Face alignment rule).
        assert_eq!(decomposed.offsets[decomposed.offsets.len() - 2].1, 4);
    }

    /// The native BPE must agree with the Hugging Face crate on ids, type
    /// ids, offsets and decoded text over both corpora.
    #[cfg(feature = "hf-tokenizers")]
    #[test]
    fn native_bpe_matches_hugging_face() {
        let Some(path) = bpe_path() else { return };
        let native = Tokenizer::from_file(&path, "bpe", "", 4096, "", "").unwrap();
        let mut hf = tokenizers::Tokenizer::from_file(&path).unwrap();
        hf.with_truncation(None).unwrap();
        hf.with_padding(None);
        let opts = EncodeOptions { truncate: Truncate::None, max_tokens: 4096, ..Default::default() };
        let mut compared = 0;
        for text in &bpe_texts() {
            let expect = hf
                .encode(tokenizers::EncodeInput::Single(tokenizers::InputSequence::Raw(text.as_str().into())), true)
                .unwrap();
            let got = native.encode(text, &opts).unwrap();
            let want_ids: Vec<i32> = expect.get_ids().iter().map(|&i| i as i32).collect();
            assert_eq!(got.ids, want_ids, "ids differ for {text:?}");
            let want_types: Vec<i32> = expect.get_type_ids().iter().map(|&i| i as i32).collect();
            assert_eq!(got.type_ids, want_types, "type ids differ for {text:?}");
            let want_offsets: Vec<(u32, u32)> =
                expect.get_offsets().iter().map(|&(s, e)| (s as u32, e as u32)).collect();
            assert_eq!(got.offsets, want_offsets, "offsets differ for {text:?}");
            let want_text = hf.decode(expect.get_ids(), true).unwrap();
            assert_eq!(native.decode(&got.ids, true).unwrap(), want_text, "decode differs for {text:?}");
            compared += 1;
        }
        assert!(compared > 200, "compared only {compared} texts");
    }
}

#[cfg(all(test, feature = "hf-tokenizers"))]
mod speed {
    //! `cargo test -p turbo-core --features hf-tokenizers --release speed -- --ignored --nocapture`
    //! prints the throughput of the native tokenizer and the Hugging Face
    //! crate on the same texts, one thread each.
    use super::*;
    use std::time::Instant;

    #[test]
    #[ignore]
    fn native_versus_hugging_face_throughput() {
        let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../testdata/bundles/minilm-tokenizer/tokenizer.json");
        throughput("wordpiece", &path);
        if let Some(path) = super::tests::bpe_path() {
            throughput("byte-level BPE", &path);
        }
    }

    fn throughput(label: &str, path: &Path) {
        eprintln!("{label}: {}", path.display());
        let native = Tokenizer::from_file(path, "tokenizer", "", 4096, "", "").unwrap();
        let mut hf = tokenizers::Tokenizer::from_file(path).unwrap();
        hf.with_truncation(None).unwrap();
        hf.with_padding(None);
        let corpus = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../testdata/corpus/sts-pairs.jsonl");
        let mut texts: Vec<String> = Vec::new();
        for line in std::fs::read_to_string(&corpus).unwrap().lines().filter(|l| !l.trim().is_empty()) {
            let v: serde_json::Value = serde_json::from_str(line).unwrap();
            texts.push(v["text_a"].as_str().unwrap().to_string());
            texts.push(v["text_b"].as_str().unwrap().to_string());
        }
        // A long paragraph as well, so per-call overhead is not the whole story.
        let long = texts.iter().take(40).cloned().collect::<Vec<_>>().join(" ");
        let opts = EncodeOptions { truncate: Truncate::None, ..Default::default() };
        for (label, set) in [("short (STS sentences)", texts.clone()), ("long (about 1000 tokens)", vec![long])] {
            let rounds = if set.len() > 1 { 200 } else { 2000 };
            let mut tokens = 0u64;
            let t0 = Instant::now();
            for _ in 0..rounds {
                for t in &set {
                    tokens += native.encode(t, &opts).unwrap().ids.len() as u64;
                }
            }
            let native_s = t0.elapsed().as_secs_f64();
            let t1 = Instant::now();
            for _ in 0..rounds {
                for t in &set {
                    hf.encode(tokenizers::EncodeInput::Single(tokenizers::InputSequence::Raw(t.as_str().into())), true)
                        .unwrap();
                }
            }
            let hf_s = t1.elapsed().as_secs_f64();
            eprintln!(
                "{label}: native {:.0} tokens/s ({:.1} us/text), hugging face {:.0} tokens/s ({:.1} us/text), ratio {:.2}x",
                tokens as f64 / native_s,
                native_s * 1e6 / (rounds * set.len()) as f64,
                tokens as f64 / hf_s,
                hf_s * 1e6 / (rounds * set.len()) as f64,
                hf_s / native_s
            );
        }
    }
}
