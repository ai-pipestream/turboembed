//! Fuzzy candidate generation over the token matcher's analyzed phrases.

use std::collections::HashMap;

use turbo_analysis::{IxRange, Span};
use turbo_analysis_layers::{Strs, Tokens};
use turbo_matcher::{FuzzyMatchPolicy, Label, Lexicon, PhraseId, PhraseMatch, Scratch, double_metaphone, soundex};

#[derive(Clone, Copy, Debug, Hash, PartialEq, Eq, PartialOrd, Ord)]
enum PhoneticKey {
    Soundex([u8; 4]),
    Metaphone([u8; 8]),
}

pub(crate) struct TokenFuzzy {
    entries: Box<[Box<str>]>,
    phrase_ids: Box<[u32]>,
    lexicon: Lexicon,
    phonetic: HashMap<PhoneticKey, Vec<u32>>,
    policy: FuzzyMatchPolicy,
    max_tokens: usize,
}

impl TokenFuzzy {
    pub(crate) fn new(phrases: Vec<String>, policy: FuzzyMatchPolicy) -> Self {
        let mut order: Vec<usize> = (0..phrases.len()).collect();
        order.sort_by(|&a, &b| phrases[a].cmp(&phrases[b]));
        let entries: Box<[Box<str>]> = order.iter().map(|&i| phrases[i].clone().into_boxed_str()).collect();
        let phrase_ids: Box<[u32]> = order.iter().map(|&i| i as u32).collect();
        let refs: Vec<&str> = entries.iter().map(Box::as_ref).collect();
        let lexicon = Lexicon::new(&refs);
        let max_tokens = entries.iter().map(|p| p.bytes().filter(|&b| b == 0).count() + 1).max().unwrap_or(0);
        let mut phonetic: HashMap<PhoneticKey, Vec<u32>> = HashMap::new();
        if policy.phonetic_fallback() {
            let mut keys = Vec::new();
            for (index, entry) in entries.iter().enumerate() {
                phonetic_keys(entry, &mut keys);
                for &key in &keys {
                    phonetic.entry(key).or_default().push(index as u32);
                }
            }
        }
        Self { entries, phrase_ids, lexicon, phonetic, policy, max_tokens }
    }

    pub(crate) fn policy(&self) -> FuzzyMatchPolicy {
        self.policy
    }

    pub(crate) fn heap_bytes(&self) -> usize {
        self.entries.iter().map(|p| p.len() + std::mem::size_of::<Box<str>>()).sum::<usize>()
            + self.phrase_ids.len() * 4
            + self.lexicon.heap_bytes()
            + self.phonetic.capacity() * std::mem::size_of::<(PhoneticKey, Vec<u32>)>()
            + self.phonetic.values().map(|v| v.capacity() * 4).sum::<usize>()
    }

    pub(crate) fn append(&self, text: &str, tokens: &Tokens, values: Option<&Strs>, out: &mut Vec<PhraseMatch>) {
        let (starts, ends) = tokens.spans.columns();
        let mut query = String::new();
        let mut scratch = Scratch::default();
        let mut hits = Vec::new();
        let mut candidates = Vec::new();
        let mut keys = Vec::new();
        for start in 0..tokens.spans.len() {
            query.clear();
            for end in start + 1..=(start + self.max_tokens).min(tokens.spans.len()) {
                if end > start + 1 {
                    query.push('\0');
                }
                let index = end - 1;
                let span = Span::new(starts[index], ends[index]);
                query.push_str(values.map_or(&text[span.range()], |v| v.get(index)));
                candidates.clear();
                self.lexicon.search(&query, self.policy.max_edit_distance(), &mut scratch, &mut hits);
                candidates.extend(hits.iter().map(|hit| hit.index));
                if self.policy.phonetic_fallback() {
                    phonetic_keys(&query, &mut keys);
                    for key in &keys {
                        if let Some(indices) = self.phonetic.get(key) {
                            candidates.extend(indices);
                        }
                    }
                }
                candidates.sort_unstable();
                candidates.dedup();
                for &index in &candidates {
                    let candidate = &self.entries[index as usize];
                    if self.policy.score(&query, candidate).is_some() {
                        out.push(PhraseMatch {
                            tokens: IxRange::new(start as u32, end as u32),
                            label: Label(0),
                            phrase: PhraseId(self.phrase_ids[index as usize]),
                        });
                    }
                }
            }
        }
    }
}

fn phonetic_keys(text: &str, out: &mut Vec<PhoneticKey>) {
    out.clear();
    let soundex = soundex(text);
    if soundex[0] != b'0' {
        out.push(PhoneticKey::Soundex(soundex));
    }
    let (primary, alternate) = double_metaphone(text);
    for key in [primary, alternate] {
        if key.iter().any(|&b| b != 0) {
            out.push(PhoneticKey::Metaphone(key));
        }
    }
    out.sort_unstable();
    out.dedup();
}
