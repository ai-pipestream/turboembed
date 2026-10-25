use super::*;
#[derive(Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Tokenizer {
    pub file: String,
    /// BertNormalizer's switches, with `wordpiece`. A `unigram` tokenizer
    /// normalizes with the map its file carries, so it has none.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub normalizer: Option<Normalizer>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub wordpiece: Option<WordPiece>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub unigram: Option<Unigram>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub bpe: Option<Bpe>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub sentencepiece: Option<SentencePiece>,
    pub special_tokens: Vec<SpecialToken>,
    pub template: Vec<String>,
    pub truncation: Truncation,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Normalizer {
    pub clean_text: bool,
    pub lowercase: bool,
    pub strip_accents: bool,
    pub split_cjk: bool,
    pub unicode_form: UnicodeForm,
}

/// Only UNICODE_NONE is read in this cut; the other forms arrive with a
/// tokenizer that needs them.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum UnicodeForm {
    #[serde(rename = "UNICODE_NONE")]
    None,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct WordPiece {
    pub continuing_prefix: String,
    pub max_chars_per_word: u32,
}

/// SentencePiece's Unigram as upstream `tokenizers` runs it: the text is
/// normalized by the precompiled character map the tokenizer file
/// carries (when `precompiled_charsmap`), runs of spaces are collapsed to
/// one (when `collapse_spaces`), every space becomes `metaspace` and each
/// piece of text between special tokens gets one in front (when
/// `add_prefix_space`); the text is then cut at each `metaspace` into
/// words, and each word into the vocabulary's pieces of highest total
/// score, a character no piece covers being the SPECIAL_UNK token.
///
/// After the collapse of spaces, as BGE-M3's static models normalize:
/// each ASCII punctuation character gets a space on each side (when
/// `space_punctuation`), each run of whitespace becomes one space (when
/// `collapse_whitespace`) and the text is trimmed (when `strip`). With
/// `whole_text` the text is not cut at the metaspace: it is segmented at
/// once, as a Metaspace pre-tokenizer with `split` false leaves it.
/// With `whitespace_split` (XLM-RoBERTa's own file: a WhitespaceSplit
/// pre-tokenizer before the Metaspace) the normalized text is first cut
/// at every Unicode whitespace character, which is dropped, and each word
/// gets its own metaspace in front.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Unigram {
    pub precompiled_charsmap: bool,
    pub collapse_spaces: bool,
    pub metaspace: String,
    pub add_prefix_space: bool,
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub space_punctuation: bool,
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub collapse_whitespace: bool,
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub strip: bool,
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub whole_text: bool,
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub whitespace_split: bool,
}

/// Byte-level BPE as upstream `tokenizers` runs it for GPT-2, RoBERTa,
/// Llama-3 and Qwen2: the text, put in NFC when `nfc`, is cut into words
/// by `pattern` (with a space put in front of each piece of text between
/// special tokens that does not start with one, when `add_prefix_space`),
/// each word's bytes are written as GPT-2's printable characters, and the
/// tokenizer file's merges join them pair by pair, lowest rank first. With
/// `ignore_merges`, a word that is a piece of the vocabulary is that
/// piece.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Bpe {
    pub add_prefix_space: bool,
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub ignore_merges: bool,
    #[serde(default, skip_serializing_if = "BpePattern::is_gpt2")]
    pub pattern: BpePattern,
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub nfc: bool,
}

/// The pattern that cuts a byte-level BPE's text into words. GPT-2's is
/// the ByteLevel pre-tokenizer's own; Llama-3's and Qwen2's are a Split
/// before a ByteLevel that splits nothing, and differ only in digits:
/// Llama-3 takes runs of up to three, Qwen2 one at a time.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
pub enum BpePattern {
    #[default]
    #[serde(rename = "BPE_PATTERN_GPT2")]
    Gpt2,
    #[serde(rename = "BPE_PATTERN_LLAMA3")]
    Llama3,
    #[serde(rename = "BPE_PATTERN_QWEN2")]
    Qwen2,
}

impl BpePattern {
    fn is_gpt2(&self) -> bool {
        *self == BpePattern::Gpt2
    }
}

/// SentencePiece's own tokenizer, as Google's sentencepiece library runs
/// it: `tokenizer.file` is the model's `.model` file, which holds the
/// pieces, the normalizer and every switch (docs/tokenizer.md). The
/// manifest says which algorithm the model must be. Special tokens are
/// the model's pieces under their ids; none is matched in the text, as
/// the library matches none.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SentencePiece {
    pub algorithm: SentencePieceAlgorithm,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SentencePieceAlgorithm {
    #[serde(rename = "unigram")]
    Unigram,
    #[serde(rename = "bpe")]
    Bpe,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SpecialToken {
    pub role: SpecialRole,
    pub content: String,
    pub id: u32,
    /// The token takes the whitespace before it in the text, as upstream's
    /// `lstrip` added tokens do.
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub lstrip: bool,
    /// The token is matched in the normalized text, its content
    /// normalized too, as upstream matches a `normalized` added token,
    /// and only as a whole word when `single_word`; it takes the
    /// whitespace after it when `rstrip`. WordPiece, or BPE, which has no
    /// normalizer and matches it in the text as it is (no `single_word`
    /// or `rstrip`).
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub normalized: bool,
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub single_word: bool,
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub rstrip: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum SpecialRole {
    #[serde(rename = "SPECIAL_PAD")]
    Pad,
    #[serde(rename = "SPECIAL_UNK")]
    Unk,
    #[serde(rename = "SPECIAL_BOS")]
    Bos,
    #[serde(rename = "SPECIAL_EOS")]
    Eos,
    #[serde(rename = "SPECIAL_MASK")]
    Mask,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Truncation {
    #[serde(rename = "TRUNCATE_NONE")]
    None,
    #[serde(rename = "TRUNCATE_RIGHT")]
    Right,
    #[serde(rename = "TRUNCATE_LEFT")]
    Left,
}
