//! Native WordPiece bindings and the row encoder shared by every text task.
//!
//! The C encoder (`native/wordpiece`) writes ids straight into caller rows
//! and does not allocate after load. This module wraps it and ports the
//! OpenVINO provider's row layout so both providers produce byte-identical
//! token rows for the same input: `[CLS] prefix text [SEP] [PAD...]`.

use std::ffi::{c_char, c_void, CString};
use std::path::Path;

use turbo_core::error::{Error, Result};
use turbo_core::types::Truncate;

const WORDPIECE_OK: i32 = 0;
const WORDPIECE_ERR_INVALID_ARGUMENT: i32 = 1;
const WORDPIECE_ERR_NOT_FOUND: i32 = 2;

/// Pair truncation: drop from the longer sequence first.
pub const TRUNC_LONGEST_FIRST: u32 = 0;
/// Pair truncation: keep the query whole and truncate the document.
pub const TRUNC_QUERY_PRIORITY: u32 = 1;
/// Pair truncation: fail when the pair exceeds the budget.
pub const TRUNC_ERROR: u32 = 2;

#[repr(C)]
struct wordpiece_vocab {
    _private: [u8; 0],
}

extern "C" {
    fn wordpiece_vocab_load(path: *const c_char, out: *mut *mut wordpiece_vocab) -> i32;
    fn wordpiece_vocab_destroy(v: *mut wordpiece_vocab);
    fn wordpiece_cls_id(v: *const wordpiece_vocab) -> i32;
    fn wordpiece_sep_id(v: *const wordpiece_vocab) -> i32;
    fn wordpiece_pad_id(v: *const wordpiece_vocab) -> i32;
    fn wordpiece_tokenize(
        v: *const wordpiece_vocab,
        utf8: *const c_char,
        utf8_len: usize,
        ids: *mut c_void,
        ids_cap: usize,
        elem_width: u32,
        n_out: *mut usize,
    ) -> i32;
    #[allow(clippy::too_many_arguments)]
    fn wordpiece_pack_pair(
        v: *const wordpiece_vocab,
        query: *const c_char,
        query_len: usize,
        doc: *const c_char,
        doc_len: usize,
        ids: *mut c_void,
        mask: *mut c_void,
        types: *mut c_void,
        pos: *mut c_void,
        seq: u32,
        stride: u32,
        elem_width: u32,
        truncation: u32,
        max_length: u32,
    ) -> i32;
}

/// A loaded vocabulary. Immutable after load, so it is shared freely.
pub struct Vocab {
    raw: *mut wordpiece_vocab,
    cls: i32,
    sep: i32,
    pad: i32,
}

// SAFETY: the C encoder never mutates the vocabulary after load.
unsafe impl Send for Vocab {}
unsafe impl Sync for Vocab {}

impl Vocab {
    /// Load `tokenizer.json` (or `vocab.txt`). The loader accepts uncased
    /// and cased BERT configurations only and rejects anything else.
    pub fn load(path: &Path) -> Result<Self> {
        let c = CString::new(path.as_os_str().as_encoded_bytes())
            .map_err(|_| Error::invalid_argument("tokenizer path contains a NUL byte"))?;
        let mut raw: *mut wordpiece_vocab = std::ptr::null_mut();
        // SAFETY: valid C string and out-pointer.
        let rc = unsafe { wordpiece_vocab_load(c.as_ptr(), &mut raw) };
        match rc {
            WORDPIECE_OK if !raw.is_null() => {}
            WORDPIECE_ERR_NOT_FOUND => {
                return Err(Error::bundle_not_found(format!("tokenizer file `{}` not found", path.display())))
            }
            WORDPIECE_ERR_INVALID_ARGUMENT => {
                return Err(Error::bundle_invalid(format!(
                    "`{}` is not a supported BERT WordPiece configuration (uncased or cased)",
                    path.display()
                )))
            }
            _ => return Err(Error::internal(format!("wordpiece loader failed with status {rc}"))),
        }
        // SAFETY: raw is a loaded vocabulary.
        let (cls, sep, pad) = unsafe { (wordpiece_cls_id(raw), wordpiece_sep_id(raw), wordpiece_pad_id(raw)) };
        if cls < 0 || sep < 0 || pad < 0 {
            // SAFETY: destroying what we just loaded.
            unsafe { wordpiece_vocab_destroy(raw) };
            return Err(Error::bundle_invalid("tokenizer defines no [CLS], [SEP], or [PAD] token"));
        }
        Ok(Self { raw, cls, sep, pad })
    }

    /// `[PAD]` id.
    pub fn pad_id(&self) -> i32 {
        self.pad
    }

    /// Count the WordPiece tokens of `text` without writing them.
    pub fn count(&self, text: &[u8]) -> Result<usize> {
        let mut n = 0usize;
        // SAFETY: NULL ids means count-only; the text pointer is valid for its length.
        let rc = unsafe {
            wordpiece_tokenize(self.raw, text.as_ptr().cast(), text.len(), std::ptr::null_mut(), 0, 4, &mut n)
        };
        if rc != WORDPIECE_OK {
            return Err(Error::invalid_argument("text is not valid UTF-8 or could not be tokenized"));
        }
        Ok(n)
    }

    /// Tokenize `text` into `out` (i32), writing at most `out.len()` ids.
    /// Returns the number written.
    pub fn tokenize_into(&self, text: &[u8], out: &mut [i32]) -> Result<usize> {
        if out.is_empty() {
            return Ok(0);
        }
        let mut n = 0usize;
        // SAFETY: `out` is a valid i32 buffer of `out.len()` elements.
        let rc = unsafe {
            wordpiece_tokenize(
                self.raw,
                text.as_ptr().cast(),
                text.len(),
                out.as_mut_ptr().cast(),
                out.len(),
                4,
                &mut n,
            )
        };
        if rc != WORDPIECE_OK {
            return Err(Error::internal("text tokenization failed after a successful count"));
        }
        Ok(n.min(out.len()))
    }

    /// Pack a cross-encoder pair into one row of `seq` i32 elements:
    /// `[CLS] query [SEP] doc [SEP] [PAD...]`, types 0 for the query and 1
    /// for the document. `truncation` is [`TRUNC_LONGEST_FIRST`] or
    /// [`TRUNC_ERROR`]; `budget` is the token limit including specials.
    #[allow(clippy::too_many_arguments)]
    pub fn pack_pair(
        &self,
        query: &[u8],
        doc: &[u8],
        ids: &mut [i32],
        mask: &mut [i32],
        types: &mut [i32],
        pos: &mut [i32],
        seq: u32,
        truncation: u32,
        budget: u32,
    ) -> Result<()> {
        let n = seq as usize;
        if ids.len() < n || mask.len() < n || types.len() < n || pos.len() < n {
            return Err(Error::internal("pair row buffers are shorter than the session sequence"));
        }
        // SAFETY: every row buffer holds at least `seq` i32 elements.
        let rc = unsafe {
            wordpiece_pack_pair(
                self.raw,
                query.as_ptr().cast(),
                query.len(),
                doc.as_ptr().cast(),
                doc.len(),
                ids.as_mut_ptr().cast(),
                mask.as_mut_ptr().cast(),
                types.as_mut_ptr().cast(),
                pos.as_mut_ptr().cast(),
                seq,
                seq,
                4,
                truncation,
                budget,
            )
        };
        match rc {
            WORDPIECE_OK => Ok(()),
            WORDPIECE_ERR_INVALID_ARGUMENT => Err(Error::capacity(format!(
                "query/document pair does not fit the budget of {budget} tokens with truncation NONE, or is not valid UTF-8"
            ))),
            _ => Err(Error::internal(format!("wordpiece pair packing failed with status {rc}"))),
        }
    }
}

impl Drop for Vocab {
    fn drop(&mut self) {
        // SAFETY: loaded by wordpiece_vocab_load; destroyed once.
        unsafe { wordpiece_vocab_destroy(self.raw) };
    }
}

/// One whitespace-delimited word (or one punctuation character) of an input
/// row and the token columns it occupies. Used for span aggregation.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct WordSpan {
    /// Byte start in the input text.
    pub start: u64,
    /// Byte end (exclusive).
    pub end: u64,
    /// First token column in the row.
    pub first_token: u32,
    /// Number of sub-tokens.
    pub n_tokens: u32,
}

/// Scratch the encoder needs between calls; sized once per session.
pub struct RowScratch {
    /// Left-truncation and prefix staging, `seq * 16` ids.
    pub ids: Vec<i32>,
}

impl RowScratch {
    /// Allocate for a session of `seq` columns.
    pub fn new(seq: u32) -> Self {
        Self { ids: vec![0; seq as usize * 16] }
    }
}

/// Encode `text` with an optional `prefix` into one row: `[CLS] prefix text
/// [SEP] [PAD...]`. `budget` counts specials. Returns the live token count.
/// When `words` is given it receives the word boundaries that survived
/// truncation, in row order.
#[allow(clippy::too_many_arguments)]
pub fn encode_row(
    vocab: &Vocab,
    row: u32,
    prefix: &str,
    text: &str,
    truncate: Truncate,
    budget: u32,
    seq: u32,
    scratch: &mut RowScratch,
    row_ids: &mut [i32],
    row_mask: &mut [i32],
    words: Option<&mut Vec<WordSpan>>,
) -> Result<u32> {
    let seq_len = seq as usize;
    if row_ids.len() < seq_len || row_mask.len() < seq_len {
        return Err(Error::internal("row buffers are shorter than the session sequence"));
    }
    if budget < 2 {
        return Err(Error::capacity("token budget must be at least 2 for [CLS] and [SEP]"));
    }
    let content_budget = (budget - 2) as usize;
    // Prefix tokens go first and are never truncated away on the right.
    let mut n_prefix = 0usize;
    if !prefix.is_empty() {
        n_prefix = vocab.tokenize_into(prefix.as_bytes(), &mut scratch.ids)?;
        if n_prefix > content_budget {
            return Err(Error::capacity(format!(
                "prompt prefix alone is {n_prefix} tokens but the content budget is {content_budget}"
            )));
        }
    }
    let n_text = vocab.count(text.as_bytes())?;
    let avail = content_budget - n_prefix;
    let (mut skip, mut take) = (0usize, n_text);
    if n_text > avail {
        match truncate {
            Truncate::None => {
                return Err(Error::capacity(format!(
                    "input row {row} tokenizes to {} tokens but the budget is {budget} and truncation is NONE",
                    n_text + n_prefix + 2
                )))
            }
            Truncate::Left => {
                skip = n_text - avail;
                take = avail;
            }
            Truncate::Right | Truncate::Model => take = avail,
        }
    }
    let mut col = 0usize;
    row_ids[col] = vocab.cls;
    col += 1;
    row_ids[col..col + n_prefix].copy_from_slice(&scratch.ids[..n_prefix]);
    col += n_prefix;
    if take > 0 {
        if skip == 0 {
            let n = vocab.tokenize_into(text.as_bytes(), &mut row_ids[col..col + take])?;
            col += n;
        } else {
            if n_text > scratch.ids.len() {
                return Err(Error::capacity(format!(
                    "left truncation needs {n_text} staging tokens but the session holds {}",
                    scratch.ids.len()
                )));
            }
            vocab.tokenize_into(text.as_bytes(), &mut scratch.ids)?;
            row_ids[col..col + take].copy_from_slice(&scratch.ids[skip..skip + take]);
            col += take;
        }
    }
    row_ids[col] = vocab.sep;
    col += 1;
    for m in row_mask[..col].iter_mut() {
        *m = 1;
    }
    for c in col..seq_len {
        row_ids[c] = vocab.pad;
        row_mask[c] = 0;
    }
    if let Some(words) = words {
        collect_words(vocab, text, n_prefix, skip, take, words)?;
    }
    Ok(col as u32)
}

/// Word boundaries for span aggregation: whitespace-delimited runs, with
/// each ASCII punctuation character as its own word, each tokenized alone so
/// sub-token counts line up with the row (WordPiece never crosses
/// whitespace).
fn collect_words(
    vocab: &Vocab,
    text: &str,
    n_prefix: usize,
    skip: usize,
    take: usize,
    words: &mut Vec<WordSpan>,
) -> Result<()> {
    words.clear();
    let bytes = text.as_bytes();
    let is_punct =
        |c: u8| (33..=47).contains(&c) || (58..=64).contains(&c) || (91..=96).contains(&c) || (123..=126).contains(&c);
    let mut tok = 1 + n_prefix as u32;
    let mut seen = 0usize;
    let mut i = 0usize;
    while i < bytes.len() {
        while i < bytes.len() && bytes[i] <= b' ' {
            i += 1;
        }
        if i >= bytes.len() {
            break;
        }
        let start = i;
        if is_punct(bytes[i]) {
            i += 1;
        } else {
            while i < bytes.len() && bytes[i] > b' ' && !is_punct(bytes[i]) {
                i += 1;
            }
        }
        let nw = vocab.count(&bytes[start..i])?;
        if seen + nw <= skip {
            seen += nw;
            continue;
        }
        if seen >= skip + take {
            break;
        }
        words.push(WordSpan { start: start as u64, end: i as u64, first_token: tok, n_tokens: nw as u32 });
        tok += nw as u32;
        seen += nw;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    fn vocab() -> Vocab {
        let p =
            PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../testdata/bundles/minilm-tokenizer/tokenizer.json");
        Vocab::load(&p).expect("load the MiniLM tokenizer")
    }

    #[test]
    fn encodes_cls_text_sep_pad() {
        let v = vocab();
        let mut ids = vec![0; 8];
        let mut mask = vec![0; 8];
        let mut scratch = RowScratch::new(8);
        let n = encode_row(&v, 0, "", "hello world", Truncate::Model, 8, 8, &mut scratch, &mut ids, &mut mask, None)
            .unwrap();
        assert_eq!(n, 4);
        assert_eq!(ids[0], 101);
        assert_eq!(ids[3], 102);
        assert_eq!(&mask[..8], &[1, 1, 1, 1, 0, 0, 0, 0]);
        assert!(ids[4..].iter().all(|&i| i == v.pad_id()));
    }

    #[test]
    fn truncation_policies() {
        let v = vocab();
        let mut ids = vec![0; 4];
        let mut mask = vec![0; 4];
        let mut scratch = RowScratch::new(4);
        let text = "one two three four";
        let e = encode_row(&v, 0, "", text, Truncate::None, 4, 4, &mut scratch, &mut ids, &mut mask, None).unwrap_err();
        assert_eq!(e.code(), turbo_core::abi::TURBO_E_CAPACITY);
        encode_row(&v, 0, "", text, Truncate::Right, 4, 4, &mut scratch, &mut ids, &mut mask, None).unwrap();
        let right = ids.clone();
        encode_row(&v, 0, "", text, Truncate::Left, 4, 4, &mut scratch, &mut ids, &mut mask, None).unwrap();
        assert_ne!(right[1..3], ids[1..3], "left truncation keeps the tail");
    }

    #[test]
    fn word_spans_split_punctuation() {
        let v = vocab();
        let mut ids = vec![0; 16];
        let mut mask = vec![0; 16];
        let mut scratch = RowScratch::new(16);
        let mut words = Vec::new();
        let text = "Berlin, Germany.";
        encode_row(&v, 0, "", text, Truncate::Model, 16, 16, &mut scratch, &mut ids, &mut mask, Some(&mut words))
            .unwrap();
        let texts: Vec<&str> = words.iter().map(|w| &text[w.start as usize..w.end as usize]).collect();
        assert_eq!(texts, vec!["Berlin", ",", "Germany", "."]);
        assert_eq!(words[0].first_token, 1);
        let total: u32 = words.iter().map(|w| w.n_tokens).sum();
        assert_eq!(total + 2, mask.iter().sum::<i32>() as u32);
    }

    #[test]
    fn pair_packing_types() {
        let v = vocab();
        let (mut ids, mut mask, mut types, mut pos) = (vec![0; 12], vec![0; 12], vec![0; 12], vec![0; 12]);
        v.pack_pair(
            b"what is it",
            b"it is a thing",
            &mut ids,
            &mut mask,
            &mut types,
            &mut pos,
            12,
            TRUNC_LONGEST_FIRST,
            12,
        )
        .unwrap();
        assert_eq!(ids[0], 101);
        assert_eq!(types[0], 0);
        let live = mask.iter().sum::<i32>() as usize;
        assert_eq!(ids[live - 1], 102);
        assert_eq!(types[live - 1], 1);
        let e = v
            .pack_pair(b"what is it", b"it is a thing", &mut ids, &mut mask, &mut types, &mut pos, 12, TRUNC_ERROR, 6)
            .unwrap_err();
        assert_eq!(e.code(), turbo_core::abi::TURBO_E_CAPACITY);
    }
}
