//! The basal chunking phase context generator and sequence validator
//! (`opennlp.tools.parser.ChunkContextGenerator` and `ParserChunkerSequenceValidator`).

use std::collections::{HashMap, HashSet, VecDeque};
use std::fmt;

use turbo_sequence::chunker::TokenTag;
use turbo_sequence::{ContextGenerator, Features, SequenceValidator};

use crate::bottom_up::dictionary_key;
use crate::parse::{CONT, OTHER, Parse, START};

use super::*;

/// Builds the Cons window `(c_2, c_1, c0, c1, c2)` and unigram/bigram/trigram dictionary flags.
pub struct ConsWindow {
    pub c_2: Cons,
    pub c_1: Cons,
    pub c0: Cons,
    pub c1: Cons,
    pub c2: Cons,
    pub u_2: bool,
    pub u_1: bool,
    pub u0: bool,
    pub u1: bool,
    pub u2: bool,
    pub b_2_1: bool,
    pub b_10: bool,
    pub b01: bool,
    pub b12: bool,
    pub t_2_10: bool,
    pub t_101: bool,
    pub t012: bool,
}

impl ConsWindow {
    /// Computes the constituent window and dictionary membership flags around `nodes`
    /// (`[c_2, c_1, c0, c1, c2]`). When `dict` is `None`, all unigram/bigram/trigram flags are
    /// `true`.
    pub fn build<'a>(nodes: [Option<&'a Parse>; 5], dict: Option<&HashSet<String>>) -> Self {
        let [cf_2, cf_1, cf0, cf1, cf2] = nodes;
        let head_word = |n: Option<&'a Parse>| -> &'a str { n.map_or(EOS, Parse::head_text) };
        let w_2 = head_word(cf_2);
        let w_1 = head_word(cf_1);
        let w0 = head_word(cf0);
        let w1 = head_word(cf1);
        let w2 = head_word(cf2);

        let (u_2, u_1, u0, u1, u2, b_2_1, b_10, b01, b12, t_2_10, t_101, t012) = match dict {
            Some(words) => (
                words.contains(&dictionary_key(&[w_2])),
                words.contains(&dictionary_key(&[w_1])),
                words.contains(&dictionary_key(&[w0])),
                words.contains(&dictionary_key(&[w1])),
                words.contains(&dictionary_key(&[w2])),
                words.contains(&dictionary_key(&[w_2, w_1])),
                words.contains(&dictionary_key(&[w_1, w0])),
                words.contains(&dictionary_key(&[w0, w1])),
                words.contains(&dictionary_key(&[w1, w2])),
                words.contains(&dictionary_key(&[w_2, w_1, w0])),
                words.contains(&dictionary_key(&[w_1, w0, w1])),
                words.contains(&dictionary_key(&[w0, w1, w2])),
            ),
            None => (true, true, true, true, true, true, true, true, true, true, true, true),
        };

        let c_2 = Cons::new(cons(cf_2, -2), cons_bo(cf_2, -2), -2, u_2);
        let c_1 = Cons::new(cons(cf_1, -1), cons_bo(cf_1, -1), -1, u_1);
        let c0 = Cons::new(cons(cf0, 0), cons_bo(cf0, 0), 0, u0);
        let c1 = Cons::new(cons(cf1, 1), cons_bo(cf1, 1), 1, u1);
        let c2 = Cons::new(cons(cf2, 2), cons_bo(cf2, 2), 2, u2);

        Self { c_2, c_1, c0, c1, c2, u_2, u_1, u0, u1, u2, b_2_1, b_10, b01, b12, t_2_10, t_101, t012 }
    }
}

/// Context generator for the parser's basal chunking phase (`opennlp.tools.parser.ChunkContextGenerator`).
///
/// Reproduces `ChunkContextGenerator`'s 19 features and its optional cache, keyed as OpenNLP keys
/// it by `i + t_2 + t1 + t0 + t1 + t2 + p_2 + p_1`.
#[derive(Clone, Debug, Default)]
pub struct ChunkContextGenerator {
    cache_size: usize,
    words_key: Vec<String>,
    tags_key: Vec<String>,
    cache: VecDeque<(String, Vec<String>)>,
}

impl ChunkContextGenerator {
    /// Creates an uncached [`ChunkContextGenerator`].
    pub fn new() -> Self {
        Self::with_cache(0)
    }

    /// Creates a [`ChunkContextGenerator`] with `cache_size` cached contexts.
    pub fn with_cache(cache_size: usize) -> Self {
        Self { cache_size, words_key: Vec::new(), tags_key: Vec::new(), cache: VecDeque::with_capacity(cache_size) }
    }

    /// Empties the cache, as a new `words` array does in OpenNLP.
    pub fn reset(&mut self) {
        self.cache.clear();
        self.words_key.clear();
        self.tags_key.clear();
    }

    /// Generates the 19 chunking features at token `i` (`ChunkContextGenerator.getContext`):
    /// `default`, the word and tag of each of the five positions with and without the word,
    /// then four pairs of positions -1 and 0, and four of 0 and 1. A position outside the
    /// sentence has `eos` for its word, tag and chunk tag.
    ///
    /// OpenNLP keeps the cache while `words` is the same array (`words == wordsKey`). A slice
    /// address can be reused by a later sentence, so the cache is kept here while `words` and
    /// `tags` hold the same strings, which gives the features a fresh call would.
    pub fn get_context(&mut self, i: usize, words: &[&str], tags: &[&str], preds: &[&str]) -> Vec<String> {
        self.context_strings(i, tags.len(), |j| (words[j], tags[j]), preds)
    }

    fn context_strings<'a>(
        &mut self,
        i: usize,
        len: usize,
        token: impl Fn(usize) -> (&'a str, &'a str),
        preds: &[&str],
    ) -> Vec<String> {
        let (t_2, p_2) = i.checked_sub(2).map_or((EOS, EOS), |j| (token(j).1, preds[j]));
        let p_1 = i.checked_sub(1).map_or(EOS, |j| preds[j]);
        let t0 = token(i).1;
        let t1 = if i + 1 < len { token(i + 1).1 } else { EOS };
        let t2 = if i + 2 < len { token(i + 2).1 } else { EOS };

        let cache_key = if self.cache_size > 0 {
            let same = self.words_key.len() == len
                && self.tags_key.len() == len
                && (0..len).all(|j| {
                    let (word, tag) = token(j);
                    self.words_key[j] == word && self.tags_key[j] == tag
                });
            if !same {
                self.cache.clear();
                self.words_key = (0..len).map(|j| token(j).0.to_owned()).collect();
                self.tags_key = (0..len).map(|j| token(j).1.to_owned()).collect();
            }
            // OpenNLP's key: `t1` twice and no `t_1`.
            let key = format!("{i}{t_2}{t1}{t0}{t1}{t2}{p_2}{p_1}");
            if let Some(pos) = self.cache.iter().position(|(k, _)| k == &key) {
                let entry = self.cache.remove(pos).expect("pos in bounds");
                let out = entry.1.clone();
                self.cache.push_front(entry);
                return out;
            }
            Some(key)
        } else {
            None
        };

        let mut features = Vec::with_capacity(19);
        Self::populate(i, len, token, preds, &mut features);

        if let Some(key) = cache_key {
            if self.cache.len() >= self.cache_size {
                self.cache.pop_back();
            }
            self.cache.push_front((key, features.clone()));
        }
        features
    }

    /// Replaces `out` with the uncached chunk features, reusing its byte storage.
    pub fn get_features(&self, i: usize, words: &[&str], tags: &[&str], preds: &[&str], out: &mut Features) {
        out.clear();
        Self::populate(i, tags.len(), |j| (words[j], tags[j]), preds, out);
    }

    fn populate<'a>(
        i: usize,
        len: usize,
        token: impl Fn(usize) -> (&'a str, &'a str),
        preds: &[&str],
        out: &mut impl FeatureSink,
    ) {
        assert!(i < len, "chunk token index out of bounds");
        let slots = [-2, -1, 0, 1, 2].map(|offset| {
            let at = i.checked_add_signed(offset);
            let (word, tag) = at.filter(|&j| j < len).map_or((EOS, EOS), &token);
            let prior = if offset < 0 { at.map_or(EOS, |j| preds[j]) } else { "" };
            (offset, word, tag, prior)
        });
        let lexical = |n: usize, backoff: bool| {
            let (offset, word, tag, prior) = slots[n];
            DisplayFn(move |f: &mut fmt::Formatter<'_>| {
                write!(f, "{offset}{}=", if backoff { "*" } else { "" })?;
                if !backoff {
                    write!(f, "{word}|")?;
                }
                f.write_str(tag)?;
                if offset < 0 {
                    write!(f, "|{prior}")?;
                }
                Ok(())
            })
        };
        let pieces =
            std::array::from_fn::<_, 5, _>(|n| [out.prepare(lexical(n, false)), out.prepare(lexical(n, true))]);
        let lexical = |n: usize, backoff: bool| &pieces[n][usize::from(backoff)];
        out.emit(format_args!("default"));
        for n in 0..5 {
            out.emit(format_args!("{}", lexical(n, false)));
            out.emit(format_args!("{}", lexical(n, true)));
        }
        for (a, b) in [(1, 2), (2, 3)] {
            for (ab, bb) in [(false, false), (true, false), (false, true), (true, true)] {
                out.emit(format_args!("{},{}", lexical(a, ab), lexical(b, bb)));
            }
        }
    }
}

impl<'a> ContextGenerator<TokenTag<'a>> for ChunkContextGenerator {
    fn start_sequence(&mut self) {
        self.reset();
    }

    fn context(&mut self, index: usize, sequence: &[TokenTag<'a>], prior: &[&str], out: &mut Features) {
        let token = |j: usize| (sequence[j].token, sequence[j].tag);
        if self.cache_size == 0 {
            Self::populate(index, sequence.len(), token, prior, out);
        } else {
            // Preserve the opt-in JVM cache's key and eviction behavior. The parser's
            // default uncached beam path writes directly from TokenTag without collections.
            for feature in self.context_strings(index, sequence.len(), token, prior) {
                out.push_str(&feature);
                out.finish();
            }
        }
    }
}

/// Sequence validator for the parser's basal chunker (`opennlp.tools.parser.ParserChunkerSequenceValidator`).
///
/// Outcomes use `S-<type>`, `C-<type>`, and `O`. A continuation outcome `C-<type>` is valid only
/// after `S-<type>` or `C-<type>`.
#[derive(Clone, Debug, Default)]
pub struct ParserChunkerSequenceValidator {
    continue_start_map: HashMap<String, String>,
}

impl ParserChunkerSequenceValidator {
    /// Creates a validator from the chunker model's outcome labels.
    pub fn new<S: AsRef<str>>(outcomes: &[S]) -> Self {
        let mut continue_start_map = HashMap::with_capacity(outcomes.len());
        for outcome in outcomes {
            let outcome = outcome.as_ref();
            if let Some(rest) = outcome.strip_prefix(CONT) {
                continue_start_map.insert(outcome.to_string(), format!("{START}{rest}"));
            }
        }
        Self { continue_start_map }
    }

    /// Validates whether `outcome` can occur at `i` after `outcomes`.
    pub fn valid_outcome(&self, outcomes: &[&str], outcome: &str) -> bool {
        if let Some(start_tag) = self.continue_start_map.get(outcome) {
            if outcomes.is_empty() {
                return false;
            }
            let last_tag = outcomes[outcomes.len() - 1];
            if last_tag == outcome || last_tag == start_tag {
                return true;
            }
            if last_tag == OTHER {
                return false;
            }
            return false;
        }
        true
    }
}

impl<'a> SequenceValidator<TokenTag<'a>> for ParserChunkerSequenceValidator {
    fn valid_sequence(&self, _index: usize, _sequence: &[TokenTag<'a>], prior: &[&str], outcome: &str) -> bool {
        self.valid_outcome(prior, outcome)
    }
}
