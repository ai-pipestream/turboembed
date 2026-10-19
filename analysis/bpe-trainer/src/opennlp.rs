//! OpenNLP's BPE (`BPETokenizerTrainer` and `BPETokenizer`, Sennrich et
//! al. 2016): the corpus is cut into words at whitespace by OpenNLP's
//! `WhitespaceTokenizer`, each word is its characters with `</w>` joined
//! to the last, and the most frequent pair of adjacent symbols, counted
//! over every word as often as the word occurs, is merged in every word,
//! `num_merges` times or until no pair is left.
//!
//! Of pairs with the same count, [`TieOrder`] says which merges:
//!
//! - [`TieOrder::Jvm`], the default: the JVM's. It takes the first in
//!   the iteration order of the `HashMap` it counted the pairs in, which
//!   follows Java hash codes and the history of its maps; the loop in
//!   `jvm.rs` works that order out and makes the JVM's merges. Each merge
//!   costs a pass over the distinct words, as on the JVM.
//! - [`TieOrder::MadeFirst`]: the pair whose left symbol was made first,
//!   then the one whose right symbol was. The symbols a word starts as
//!   count as made before any merge, in Java's order of their strings
//!   (`String.compareTo`); a merged symbol is made when it is first
//!   merged. It does not depend on the order of the texts, and keeps
//!   the counts as the merge loop of [`crate::merger`] does, visiting
//!   only the words that hold the pair merged.
//!
//! Either way a symbol is a character (a Unicode code point), where the
//! JVM's is a UTF-16 code unit: the JVM splits a character outside the
//! Basic Multilingual Plane into its two surrogates and may merge each
//! with its neighbours. Such a merge has a lone surrogate on one side,
//! which no UTF-8 text can hold, and which the JVM itself writes as `?`
//! when it saves the model. Both are tested in `tests/opennlp.rs`.
//!
//! [`OpenNlpTokenizer`] is `BPETokenizer`: each word's symbols are merged
//! by the rank of their pair, lowest first, and `</w>` is taken off the
//! end of each token.

use turbo_jvm::chars::compare as java_order;
use turbo_tokenize::{TokenizerOptions, WhitespaceMode, whitespace_spans};

use crate::byte_level::ById;
use crate::merger::Merger;
use crate::pieces::{FxMap, Pieces, SymbolId};
use crate::{Merge, TrainError};

mod jvm;

/// What OpenNLP joins to the last symbol of a word (`BPETokenizer.END_OF_WORD`).
pub const END_OF_WORD: &str = "</w>";

/// OpenNLP's `WhitespaceTokenizer.INSTANCE` in its default mode.
const WHITESPACE: TokenizerOptions = TokenizerOptions { whitespace: WhitespaceMode::Unicode, keep_newlines: false };

/// Which of the pairs with the highest count is merged.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum TieOrder {
    /// The first in the JVM's `HashMap` order: the JVM's merges.
    #[default]
    Jvm,
    /// The pair whose left symbol was made first, then the one whose
    /// right symbol was.
    MadeFirst,
}

/// A trained OpenNLP BPE: its merges in rank order.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct OpenNlpBpe {
    merges: Vec<Merge>,
    jvm_exact: bool,
}

/// The words of `texts` cut at whitespace, each with how often it
/// occurs, in the order first seen.
pub fn words<'a>(texts: impl IntoIterator<Item = &'a str>) -> Vec<(&'a str, u64)> {
    counted_words(texts).words
}

/// The words of a corpus, and whether a word occurs again after the last
/// new word was first seen.
struct CountedWords<'a> {
    words: Vec<(&'a str, u64)>,
    again: bool,
}

fn counted_words<'a>(texts: impl IntoIterator<Item = &'a str>) -> CountedWords<'a> {
    let mut at: FxMap<&'a str, usize> = FxMap::default();
    let mut out: Vec<(&'a str, u64)> = Vec::new();
    let mut again = false;
    let mut spans = Vec::new();
    for text in texts {
        spans.clear();
        whitespace_spans(text, WHITESPACE, &mut spans);
        for s in &spans {
            let word = &text[s.start as usize..s.end as usize];
            match at.get(word) {
                Some(&i) => {
                    out[i].1 += 1;
                    again = true;
                }
                None => {
                    at.insert(word, out.len());
                    out.push((word, 1));
                    again = false;
                }
            }
        }
    }
    CountedWords { words: out, again }
}

/// A word's first symbols: its characters, the last with `</w>`.
pub fn symbols(word: &str) -> Vec<String> {
    let mut out: Vec<String> = word.chars().map(String::from).collect();
    if let Some(last) = out.last_mut() {
        last.push_str(END_OF_WORD);
    }
    out
}

impl OpenNlpBpe {
    /// `BPETokenizerTrainer.train(corpus, numMerges, language)`: at most
    /// `num_merges` merges learned from `texts`, ties in the JVM's order.
    pub fn train<'a>(texts: impl IntoIterator<Item = &'a str>, num_merges: usize) -> Result<OpenNlpBpe, TrainError> {
        OpenNlpBpe::train_with(texts, num_merges, TieOrder::Jvm)
    }

    /// At most `num_merges` merges learned from `texts`, ties in `ties`.
    pub fn train_with<'a>(
        texts: impl IntoIterator<Item = &'a str>,
        num_merges: usize,
        ties: TieOrder,
    ) -> Result<OpenNlpBpe, TrainError> {
        if num_merges == 0 {
            return Err(TrainError::NoMerges);
        }
        match ties {
            TieOrder::Jvm => {
                let c = counted_words(texts);
                let (merges, jvm_exact) = jvm::learn_merges(&c.words, c.again, num_merges);
                Ok(OpenNlpBpe { merges, jvm_exact })
            }
            TieOrder::MadeFirst => Ok(OpenNlpBpe { merges: made_first(texts, num_merges), jvm_exact: false }),
        }
    }

    /// Whether the merges are certainly the JVM's for the same corpus:
    /// trained in [`TieOrder::Jvm`], on a corpus with no character
    /// outside the Basic Multilingual Plane, and with no tree bin of the
    /// JVM's maps (eight or more keys in one bin of a table of 64 or more)
    /// that held keys with equal hash codes, which the JVM orders by
    /// identity hash code and so differently from run to run. That order
    /// changes the merges only when two pairs in that bin tie for the
    /// highest count; false is not a difference found, only one not ruled
    /// out.
    pub fn is_jvm_exact(&self) -> bool {
        self.jvm_exact
    }

    /// The model of a list of merges in rank order, such as the JVM's.
    /// Their counts are not known, and are 0.
    pub fn from_merges(pairs: impl IntoIterator<Item = (String, String)>) -> OpenNlpBpe {
        let merges = pairs.into_iter().map(|(left, right)| Merge { left, right, count: 0 }).collect();
        OpenNlpBpe { merges, jvm_exact: false }
    }

    pub fn merges(&self) -> &[Merge] {
        &self.merges
    }

    /// The merges one per line, `left right` as `SymbolPair.toString`
    /// writes them. A symbol holds no whitespace, so the space between
    /// them is not ambiguous.
    pub fn to_merges_text(&self) -> String {
        let mut s = String::new();
        for m in &self.merges {
            s.push_str(&m.left);
            s.push(' ');
            s.push_str(&m.right);
            s.push('\n');
        }
        s
    }

    /// Reads what `to_merges_text` writes.
    pub fn parse_merges_text(text: &str) -> Result<OpenNlpBpe, TrainError> {
        let mut pairs = Vec::new();
        for (i, line) in text.lines().enumerate() {
            let mut parts = line.split(' ');
            match (parts.next(), parts.next(), parts.next()) {
                (Some(l), Some(r), None)
                    if !l.is_empty()
                        && !r.is_empty()
                        && !l.chars().any(char::is_whitespace)
                        && !r.chars().any(char::is_whitespace) =>
                {
                    pairs.push((l.to_owned(), r.to_owned()));
                }
                _ => return Err(TrainError::BadMergeLine { line: i + 1 }),
            }
        }
        Ok(OpenNlpBpe::from_merges(pairs))
    }

    /// The tokenizer the merges make.
    pub fn tokenizer(&self) -> OpenNlpTokenizer {
        let mut ranks = FxMap::default();
        for (rank, m) in self.merges.iter().enumerate() {
            // A pair listed twice has its later rank, as the JVM's
            // LinkedHashMap.put keeps the later value.
            ranks.insert((m.left.clone(), m.right.clone()), rank);
        }
        OpenNlpTokenizer { ranks }
    }
}

/// The merges with ties in [`TieOrder::MadeFirst`].
fn made_first<'a>(texts: impl IntoIterator<Item = &'a str>, num_merges: usize) -> Vec<Merge> {
    let words: Vec<(Vec<String>, u64)> = words(texts).into_iter().map(|(w, n)| (symbols(w), n)).collect();
    let mut first: Vec<&str> = words.iter().flat_map(|(s, _)| s.iter().map(String::as_str)).collect();
    first.sort_unstable_by(|a, b| java_order(a, b));
    first.dedup();
    let mut pieces = Pieces::default();
    for s in first {
        pieces.intern(s);
    }
    let words: Vec<(Vec<SymbolId>, u64)> =
        words.iter().map(|(s, n)| (s.iter().map(|s| pieces.get(s).expect("interned above")).collect(), *n)).collect();
    let mut m = Merger::new(ById, pieces, words);
    let mut merges = Vec::new();
    while merges.len() < num_merges {
        let Some(c) = m.best() else { break };
        let p = m.pieces();
        merges.push(Merge {
            left: p.text(c.pair.left).to_owned(),
            right: p.text(c.pair.right).to_owned(),
            count: c.count,
        });
        m.merge(c.pair);
    }
    merges
}

/// OpenNLP's `BPETokenizer`.
pub struct OpenNlpTokenizer {
    ranks: FxMap<(String, String), usize>,
}

impl OpenNlpTokenizer {
    /// `tokenize`: each word's tokens, `</w>` taken off.
    pub fn tokenize(&self, text: &str) -> Vec<String> {
        self.tokenize_pos(text).into_iter().map(|(a, b)| text[a as usize..b as usize].to_owned()).collect()
    }

    /// `tokenizePos`, in UTF-8 bytes: each token's span, from the start of
    /// its word on by the length of the tokens before it, `</w>` taken off
    /// the last token of each word.
    pub fn tokenize_pos(&self, text: &str) -> Vec<(u32, u32)> {
        let mut spans = Vec::new();
        whitespace_spans(text, WHITESPACE, &mut spans);
        let mut out = Vec::new();
        for s in spans {
            let word = &text[s.start as usize..s.end as usize];
            let mut at = s.start;
            let merged = self.apply(symbols(word));
            let last = merged.len().saturating_sub(1);
            for (i, token) in merged.into_iter().enumerate() {
                let clean = if i == last { token.strip_suffix(END_OF_WORD).unwrap_or(&token) } else { &token };
                let len = clean.len() as u32;
                out.push((at, at + len));
                at += len;
            }
        }
        out
    }

    /// `applyMerges`: while any pair of adjacent symbols is listed, the
    /// pair of lowest rank (the leftmost of it) is merged everywhere in
    /// the word, left to right.
    fn apply(&self, mut symbols: Vec<String>) -> Vec<String> {
        let mut key = (String::new(), String::new());
        while symbols.len() > 1 {
            let mut best: Option<(usize, usize)> = None;
            for i in 0..symbols.len() - 1 {
                key.0.clone_from(&symbols[i]);
                key.1.clone_from(&symbols[i + 1]);
                if let Some(&rank) = self.ranks.get(&key)
                    && best.is_none_or(|(r, _)| rank < r)
                {
                    best = Some((rank, i));
                }
            }
            let Some((_, at)) = best else { break };
            let (left, right) = (symbols[at].clone(), symbols[at + 1].clone());
            let mut next = Vec::with_capacity(symbols.len());
            let mut i = 0;
            while i < symbols.len() {
                if i + 1 < symbols.len() && symbols[i] == left && symbols[i + 1] == right {
                    next.push(format!("{left}{right}"));
                    i += 2;
                } else {
                    next.push(std::mem::take(&mut symbols[i]));
                    i += 1;
                }
            }
            symbols = next;
        }
        symbols
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn pairs(m: &OpenNlpBpe) -> Vec<(&str, &str)> {
        m.merges().iter().map(|m| (m.left.as_str(), m.right.as_str())).collect()
    }

    #[test]
    fn a_word_ends_with_the_marker() {
        assert_eq!(symbols("low"), ["l", "o", "w</w>"]);
        assert_eq!(symbols("é"), ["é</w>"]);
        assert!(symbols("").is_empty());
    }

    /// Sennrich et al.'s corpus: low 5, lower 2, newest 6, widest 3. es
    /// and st</w> both occur 9 times, and e was made before s; after lo
    /// (7), ne, ew and w est</w> occur 6 times, and e was made first.
    #[test]
    fn the_papers_corpus_merges_by_count_then_by_id() {
        let corpus = "low ".repeat(5) + &"lower ".repeat(2) + &"newest ".repeat(6) + &"widest ".repeat(3);
        let m = OpenNlpBpe::train_with([corpus.as_str()], 4, TieOrder::MadeFirst).unwrap();
        assert_eq!(pairs(&m), [("e", "s"), ("es", "t</w>"), ("l", "o"), ("e", "w")]);
        assert_eq!(m.merges().iter().map(|m| m.count).collect::<Vec<_>>(), [9, 9, 7, 6]);
    }

    #[test]
    fn zero_merges_is_refused_as_the_jvm_refuses_it() {
        assert!(matches!(OpenNlpBpe::train(["a b"], 0), Err(TrainError::NoMerges)));
    }

    #[test]
    fn training_stops_when_every_word_is_one_symbol() {
        let m = OpenNlpBpe::train(["ab ab a"], 100).unwrap();
        assert_eq!(pairs(&m), [("a", "b</w>")]);
    }

    #[test]
    fn the_tokenizer_takes_the_marker_off_and_spans_each_token() {
        let m = OpenNlpBpe::from_merges([("l".into(), "o".into()), ("lo".into(), "w</w>".into())]);
        let t = m.tokenizer();
        assert_eq!(t.tokenize(" low  lower\tl"), ["low", "lo", "w", "e", "r", "l"]);
        assert_eq!(t.tokenize_pos("ab low"), [(0, 1), (1, 2), (3, 6)]);

        // A non-final piece ending with literal "</w>" from the input word
        // must keep its bytes so subsequent spans stay aligned.
        let m_literal =
            OpenNlpBpe::from_merges([("<".into(), "/".into()), ("</".into(), "w".into()), ("</w".into(), ">".into())]);
        let t_literal = m_literal.tokenizer();
        assert_eq!(t_literal.tokenize("</w>a😀"), ["</w>", "a", "😀"]);
        assert_eq!(t_literal.tokenize_pos("</w>a😀"), [(0, 4), (4, 5), (5, 9)]);
    }

    #[test]
    fn merges_text_reads_back() {
        let m = OpenNlpBpe::train(["the cat sat on the mat"], 10).unwrap();
        let text = m.to_merges_text();
        let back = OpenNlpBpe::parse_merges_text(&text).unwrap();
        assert_eq!(pairs(&back), pairs(&m));
        assert!(matches!(OpenNlpBpe::parse_merges_text("a b c\n"), Err(TrainError::BadMergeLine { line: 1 })));
        assert!(matches!(OpenNlpBpe::parse_merges_text("a\tb c\n"), Err(TrainError::BadMergeLine { line: 1 })));
    }
}
