//! The finders as analysis steps: each reads a text view and its tokens
//! and writes `opennlp:entities`.
//!
//! The finders run once per sentence, on the tokens of that sentence,
//! when the tokens say which sentence each is in, and once on all the
//! tokens otherwise. An entity's text span runs from its first token's
//! start to its last token's end. An empty name (a user pattern can match
//! the empty string at a token boundary) is an empty span at the start of
//! the token it stands before, or at the end of the last token when it
//! stands after them all. Scores are 0, the probability OpenNLP's `Span`
//! carries when a finder gives none.

use std::hash::{Hash, Hasher};

use turbo_analysis::{Annotator, Error, IxRange, Refs, Span, Spans, View};
use turbo_analysis_layers::{Entities, Tokens};

use crate::dictionary::DictionaryNameFinder;
use crate::name::{Name, TokenNameFinder};
use crate::regex_finder::{Pattern, RegexNameFinder};

impl Annotator for DictionaryNameFinder {
    const ID: &'static str = "turbo-namefind:dictionary";
    type Reads = (View, Tokens);
    type Writes = (Entities,);

    fn annotate(&self, (view, tokens): Refs<'_, Self>) -> Result<(Entities,), Error> {
        find_entities(Self::ID, self, view, tokens).map(|e| (e,))
    }

    fn config_hash(&self) -> u64 {
        // The dictionary's entries in no particular order, so combined by
        // a sum, which does not depend on the order.
        let mut sum = 0u64;
        for entry in self.dictionary().entries() {
            let mut h = Fnv::default();
            entry.for_each(|t| {
                t.hash(&mut h);
            });
            sum = sum.wrapping_add(h.finish());
        }
        let mut h = Fnv::default();
        (sum, self.dictionary().is_case_sensitive(), self.kinds().label(turbo_analysis_layers::Tag(0))).hash(&mut h);
        h.finish()
    }
}

impl Annotator for RegexNameFinder {
    const ID: &'static str = "turbo-namefind:regex";
    type Reads = (View, Tokens);
    type Writes = (Entities,);

    fn annotate(&self, (view, tokens): Refs<'_, Self>) -> Result<(Entities,), Error> {
        find_entities(Self::ID, self, view, tokens).map(|e| (e,))
    }

    fn config_hash(&self) -> u64 {
        let mut h = Fnv::default();
        for (kind, patterns) in self.patterns() {
            kind.hash(&mut h);
            for p in patterns {
                match p {
                    Pattern::BuiltIn(b) => (0u8, b.java_name()).hash(&mut h),
                    Pattern::Java(r) => (1u8, r.as_str(), r.flags().case_insensitive, r.flags().dotall).hash(&mut h),
                }
            }
        }
        h.finish()
    }
}

/// Runs `finder` over each sentence's tokens and writes the names found.
fn find_entities(
    id: &'static str,
    finder: &dyn TokenNameFinder,
    view: &View,
    tokens: &Tokens,
) -> Result<Entities, Error> {
    let fail = |detail: String| Error::Annotator { annotator: id, detail };
    if tokens.spans.view() != view.id() {
        return Err(fail("the tokens are in another view".into()));
    }
    let text = view.text();
    let (starts, ends) = tokens.spans.columns();
    let mut entities = Entities {
        spans: Spans::new(view.id()),
        tokens: Vec::new(),
        kind: Vec::new(),
        kinds: finder.kinds().clone(),
        score: Vec::new(),
    };
    let mut words: Vec<&str> = Vec::new();
    let mut names: Vec<Name> = Vec::new();
    for range in sentences(tokens) {
        let (base, end) = (range.start.get(), range.end.get());
        words.clear();
        words.extend((base..end).map(|i| &text[starts[i] as usize..ends[i] as usize]));
        names.clear();
        finder.find(&words, &mut names).map_err(|e| fail(e.to_string()))?;
        for n in &names {
            let (a, b) = (base + n.start as usize, base + n.end as usize);
            let span = if a < b {
                Span::new(starts[a], ends[b - 1])
            } else if a < end {
                Span::new(starts[a], starts[a])
            } else {
                Span::new(ends[end - 1], ends[end - 1])
            };
            entities.spans.push(span);
            entities.tokens.push(IxRange::new(a as u32, b as u32));
            entities.kind.push(n.kind);
            entities.score.push(0.0);
        }
    }
    Ok(entities)
}

/// The token range of each sentence, or one range over all the tokens.
fn sentences(tokens: &Tokens) -> Vec<IxRange<Tokens>> {
    let n = tokens.spans.len();
    let Some(sentence) = tokens.sentence.as_deref() else {
        return if n == 0 { Vec::new() } else { vec![IxRange::new(0, n as u32)] };
    };
    let mut out = Vec::new();
    let mut start = 0;
    for i in 1..=n {
        if i == n || sentence[i] != sentence[start] {
            out.push(IxRange::new(start as u32, i as u32));
            start = i;
        }
    }
    out
}

/// 64-bit FNV-1a, for configuration hashes that stay the same across runs.
struct Fnv(u64);

impl Default for Fnv {
    fn default() -> Fnv {
        Fnv(0xcbf2_9ce4_8422_2325)
    }
}

impl Hasher for Fnv {
    fn finish(&self) -> u64 {
        self.0
    }

    fn write(&mut self, bytes: &[u8]) {
        for &b in bytes {
            self.0 = (self.0 ^ b as u64).wrapping_mul(0x0100_0000_01b3);
        }
    }
}
