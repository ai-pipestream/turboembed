//! `DictionaryFeatureGenerator`.

use std::sync::Arc;

use turbo_maxent::Dictionary;

use super::FeatureGenerator;
use crate::features::Features;

/// `DictionaryFeatureGenerator`: for a token inside a dictionary entry
/// found in the sentence, `:w=dic` and `:w=dic=` with the token, each
/// after the prefix (empty when the descriptor makes it).
///
/// Entries are found as `DictionaryNameFinder` finds them: from each
/// token not inside an earlier match, the longest run of at most
/// `max_tokens` tokens that is an entry, matched as the dictionary's case
/// setting says. They are found once per sentence (Java's `InSpanGenerator`
/// keeps them for the current `String[]`).
pub struct InDictionary {
    prefix: String,
    dictionary: Arc<Dictionary>,
    /// The entries found in the current sentence, as token ranges, once
    /// found.
    names: Option<Vec<(usize, usize)>>,
    scratch: String,
}

impl InDictionary {
    pub fn new(prefix: &str, dictionary: Arc<Dictionary>) -> InDictionary {
        InDictionary { prefix: prefix.to_owned(), dictionary, names: None, scratch: String::new() }
    }

    /// `DictionaryNameFinder.find`.
    fn find(&mut self, tokens: &[&str]) -> Vec<(usize, usize)> {
        let max = self.dictionary.max_tokens();
        let mut names = Vec::new();
        let mut from = 0;
        while from < tokens.len() {
            let mut found = None;
            for to in from..tokens.len() {
                if to - from + 1 > max {
                    break;
                }
                if self.dictionary.contains_tokens(&tokens[from..=to], &mut self.scratch) {
                    found = Some((from, to + 1));
                }
            }
            match found {
                Some((start, end)) => {
                    names.push((start, end));
                    from = end;
                }
                None => from += 1,
            }
        }
        names
    }
}

impl FeatureGenerator for InDictionary {
    fn create_features(&mut self, out: &mut Features, tokens: &[&str], index: usize, _previous: &[&str]) {
        if self.names.is_none() {
            self.names = Some(self.find(tokens));
        }
        let names = self.names.as_deref().expect("found above");
        if names.iter().any(|&(start, end)| start <= index && index < end) {
            out.push_str(&self.prefix).push_str(":w=dic").finish();
            out.push_str(&self.prefix).push_str(":w=dic=").push_str(tokens[index]).finish();
        }
    }

    fn start_sequence(&mut self) {
        self.names = None;
    }
}
