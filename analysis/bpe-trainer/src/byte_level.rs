//! Byte-level BPE as upstream `tokenizers` trains it for GPT-2 and
//! RoBERTa (`BpeTrainer` with the `ByteLevel` pre-tokenizer and its
//! alphabet), written as a `tokenizer.json` that the core's byte-level
//! BPE (`turbo::bpe`) and upstream both load.
//!
//! Each text is cut into words by GPT-2's pattern, the core's own
//! [`turbo::bpe::split`], with a space put in front first when
//! `add_prefix_space` is set and the text does not start with one. A
//! word's bytes are written as GPT-2's characters ([`byte_chars`]). The
//! vocabulary starts with the special tokens, then all 256 byte
//! characters in the order of their code points; each merge adds the two
//! pieces joined, unless that piece is already there. Training stops when
//! the vocabulary has `vocab_size` pieces, when no pair is left, or when
//! the most frequent pair occurs fewer than `min_frequency` times.
//!
//! Of pairs with the same count, the pair whose left piece has the
//! smaller id merges first, then the one whose right piece has: upstream's
//! order, which gives the same ids.
//!
//! Where this differs from upstream: there is no `limit_alphabet`,
//! `initial_alphabet`, `continuing_subword_prefix`, `end_of_word_suffix`
//! or `max_token_length`. The alphabet is always the 256 byte
//! characters, which the core's encoder needs, and it reads no prefix or
//! suffix. With the options it has, `tests/byte_level.rs` finds upstream's
//! file on every corpus and setting it tries.

use turbo::bpe::{byte_chars, split};

use crate::merger::{Merger, Order};
use crate::pieces::{FxMap, Pair, Pieces, SymbolId};
use crate::{Merge, TrainError, json_string};

/// A special token, added to the vocabulary before anything else, in
/// the order given.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SpecialToken {
    pub content: String,
    /// The token takes the whitespace before it (`lstrip`), as RoBERTa's
    /// `<mask>` does.
    pub lstrip: bool,
}

impl SpecialToken {
    pub fn new(content: &str) -> SpecialToken {
        SpecialToken { content: content.into(), lstrip: false }
    }
}

/// What the tokenizer file says to add around a text.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum PostProcessor {
    /// None: the bundle's template adds the special tokens.
    None,
    /// RoBERTa's: `cls` before and `sep` after, both special tokens.
    Roberta { cls: String, sep: String },
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ByteLevelConfig {
    /// The size of the vocabulary to stop at, special tokens and the 256
    /// byte characters included.
    pub vocab_size: usize,
    /// The fewest times a pair must occur to be merged; 0 and 1 both
    /// merge any pair that occurs.
    pub min_frequency: u64,
    pub special_tokens: Vec<SpecialToken>,
    /// Put a space in front of a text that does not start with one, as
    /// the core's `bpe.add_prefix_space` does when encoding.
    pub add_prefix_space: bool,
    pub post_processor: PostProcessor,
}

/// A trained byte-level BPE.
#[derive(Clone, Debug)]
pub struct ByteLevelBpe {
    config: ByteLevelConfig,
    vocab: Vec<String>,
    merges: Vec<Merge>,
}

/// Upstream's order: the smaller ids first.
pub(crate) struct ById;

impl Order for ById {
    type Key = Pair;

    fn key(&self, _: &Pieces, pair: Pair) -> Pair {
        pair
    }
}

/// The words of `texts`, in GPT-2's characters, each with how often it
/// occurs, in the order first seen.
pub fn words<'a>(texts: impl IntoIterator<Item = &'a str>, add_prefix_space: bool) -> Vec<(String, u64)> {
    let chars = byte_chars();
    let mut at: FxMap<String, usize> = FxMap::default();
    let mut out: Vec<(String, u64)> = Vec::new();
    let mut prefixed = String::new();
    for text in texts {
        let text = if add_prefix_space && !text.starts_with(' ') {
            prefixed.clear();
            prefixed.push(' ');
            prefixed.push_str(text);
            prefixed.as_str()
        } else {
            text
        };
        split(text, |a, b| {
            let word: String = text.as_bytes()[a..b].iter().map(|&b| chars[b as usize]).collect();
            match at.get(&word) {
                Some(&i) => out[i].1 += 1,
                None => {
                    at.insert(word.clone(), out.len());
                    out.push((word, 1));
                }
            }
        });
    }
    out
}

impl ByteLevelBpe {
    /// Trains on `texts`.
    pub fn train<'a>(
        texts: impl IntoIterator<Item = &'a str>,
        config: ByteLevelConfig,
    ) -> Result<ByteLevelBpe, TrainError> {
        let mut pieces = Pieces::default();
        for s in &config.special_tokens {
            if s.content.is_empty() {
                return Err(TrainError::EmptySpecialToken);
            }
            pieces.intern(&s.content);
        }
        let mut alphabet: Vec<char> = byte_chars().to_vec();
        alphabet.sort_unstable();
        for c in alphabet {
            pieces.intern(c.encode_utf8(&mut [0; 4]));
        }
        let ids = |pieces: &Pieces, word: &str| -> Vec<SymbolId> {
            word.chars().map(|c| pieces.get(c.encode_utf8(&mut [0; 4])).expect("every byte character")).collect()
        };
        let words: Vec<(Vec<SymbolId>, u64)> =
            words(texts, config.add_prefix_space).into_iter().map(|(w, n)| (ids(&pieces, &w), n)).collect();

        let mut m = Merger::new(ById, pieces, words);
        let mut merges = Vec::new();
        let least = config.min_frequency.max(1);
        while m.pieces().len() < config.vocab_size {
            let Some(c) = m.best() else { break };
            if c.count < least {
                break;
            }
            let p = m.pieces();
            merges.push(Merge {
                left: p.text(c.pair.left).to_owned(),
                right: p.text(c.pair.right).to_owned(),
                count: c.count,
            });
            m.merge(c.pair);
        }
        let vocab = m.into_pieces().iter().map(str::to_owned).collect();
        Ok(ByteLevelBpe { config, vocab, merges })
    }

    /// Every piece, by id.
    pub fn vocab(&self) -> &[String] {
        &self.vocab
    }

    /// The merges, in the order made, which is their rank.
    pub fn merges(&self) -> &[Merge] {
        &self.merges
    }

    pub fn config(&self) -> &ByteLevelConfig {
        &self.config
    }

    /// The `tokenizer.json`, written as upstream `tokenizers` writes it
    /// (`Tokenizer::save` without pretty printing): the special tokens
    /// as added tokens, no normalizer, the ByteLevel pre-tokenizer and
    /// decoder, and the BPE model with no unknown token.
    pub fn to_tokenizer_json(&self) -> String {
        let c = &self.config;
        let mut s = String::from(r#"{"version":"1.0","truncation":null,"padding":null,"added_tokens":["#);
        for (i, t) in c.special_tokens.iter().enumerate() {
            if i > 0 {
                s.push(',');
            }
            let id = self.vocab.iter().position(|v| *v == t.content).expect("a special token is a piece");
            s.push_str(&format!(
                r#"{{"id":{id},"content":{},"single_word":false,"lstrip":{},"rstrip":false,"normalized":false,"special":true}}"#,
                json_string(&t.content),
                t.lstrip
            ));
        }
        s.push_str(&format!(
            r#"],"normalizer":null,"pre_tokenizer":{{"type":"ByteLevel","add_prefix_space":{},"trim_offsets":true,"use_regex":true}},"post_processor":"#,
            c.add_prefix_space
        ));
        match &c.post_processor {
            PostProcessor::None => s.push_str("null"),
            PostProcessor::Roberta { cls, sep } => {
                let id = |t: &str| self.vocab.iter().position(|v| v == t);
                s.push_str(&format!(
                    r#"{{"type":"RobertaProcessing","sep":[{},{}],"cls":[{},{}],"trim_offsets":true,"add_prefix_space":{}}}"#,
                    json_string(sep),
                    id(sep).map_or("null".into(), |i| i.to_string()),
                    json_string(cls),
                    id(cls).map_or("null".into(), |i| i.to_string()),
                    c.add_prefix_space
                ));
            }
        }
        s.push_str(
            r#","decoder":{"type":"ByteLevel","add_prefix_space":true,"trim_offsets":true,"use_regex":true},"model":{"type":"BPE","dropout":null,"unk_token":null,"continuing_subword_prefix":null,"end_of_word_suffix":null,"fuse_unk":false,"byte_fallback":false,"ignore_merges":false,"vocab":{"#,
        );
        for (i, v) in self.vocab.iter().enumerate() {
            if i > 0 {
                s.push(',');
            }
            s.push_str(&json_string(v));
            s.push(':');
            s.push_str(&i.to_string());
        }
        s.push_str(r#"},"merges":["#);
        for (i, m) in self.merges.iter().enumerate() {
            if i > 0 {
                s.push(',');
            }
            s.push('[');
            s.push_str(&json_string(&m.left));
            s.push(',');
            s.push_str(&json_string(&m.right));
            s.push(']');
        }
        s.push_str("]}}");
        s
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn config(vocab_size: usize) -> ByteLevelConfig {
        ByteLevelConfig {
            vocab_size,
            min_frequency: 0,
            special_tokens: vec![SpecialToken::new("<unk>")],
            add_prefix_space: false,
            post_processor: PostProcessor::None,
        }
    }

    #[test]
    fn the_vocabulary_starts_with_the_specials_then_the_bytes_by_code_point() {
        let m = ByteLevelBpe::train(["ab ab"], config(257)).unwrap();
        assert_eq!(m.vocab().len(), 257);
        assert_eq!(m.vocab()[0], "<unk>");
        assert_eq!(m.vocab()[1], "!");
        // The bytes GPT-2 writes as themselves, then those from U+0100 on.
        assert_eq!(m.vocab()[256], "\u{143}");
        assert!(m.merges().is_empty());
    }

    #[test]
    fn a_space_goes_in_front_only_of_a_text_without_one() {
        assert_eq!(words(["a", " b"], true), [("\u{120}a".to_owned(), 1), ("\u{120}b".to_owned(), 1)]);
        assert_eq!(words(["a b"], false), [("a".to_owned(), 1), ("\u{120}b".to_owned(), 1)]);
    }

    #[test]
    fn training_stops_at_the_vocabulary_size_or_the_least_frequency() {
        let texts = ["hello hello hello world"];
        let m = ByteLevelBpe::train(texts, config(259)).unwrap();
        assert_eq!(m.vocab().len(), 259);
        assert_eq!(m.merges().len(), 2);
        let mut c = config(10_000);
        c.min_frequency = 3;
        let m = ByteLevelBpe::train(texts, c).unwrap();
        assert!(m.merges().iter().all(|m| m.count >= 3));
        // The four merges of hello, which occurs three times; the space
        // before it occurs twice.
        assert_eq!(m.merges().len(), 4);
    }

    #[test]
    fn an_empty_special_token_is_refused() {
        let mut c = config(300);
        c.special_tokens.push(SpecialToken::new(""));
        assert!(matches!(ByteLevelBpe::train(["x"], c), Err(TrainError::EmptySpecialToken)));
    }
}
