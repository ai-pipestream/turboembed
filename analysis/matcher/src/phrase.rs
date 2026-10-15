//! The phrase matcher, spaCy's `PhraseMatcher`: a large list of
//! multi-token phrases matched over one token attribute, as a trie over
//! token ids.
//!
//! Each phrase token is turned into the attribute's value, then into an
//! id from the matcher's vocabulary, so a phrase is a sequence of ids and
//! the phrase list is a trie over them. The root's edges are an array
//! indexed by id, since every start tries one; deeper edges are
//! `(state, id)` pairs in one open addressing table. A document's tokens
//! are looked up once per call, then each start walks the trie as far as
//! the tokens follow it, so a call costs one lookup per token plus one
//! edge per token a walk takes.

use turbo_analysis::IxRange;
use turbo_analysis_layers::Tokens;

use crate::attr::{Field, text_flags};
use crate::error::{MatchError, PatternError};
use crate::hash::EdgeTable;
use crate::input::{Columns, Input, Needs};
use crate::vocab::{Label, Labels, NONE, Vocab};

/// A phrase's number in its matcher, in the order phrases were added; a
/// phrase added again under the same label keeps its first number.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct PhraseId(pub u32);

/// One match of a phrase: the tokens it covers, the label it was added
/// under, and which phrase it is.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct PhraseMatch {
    pub tokens: IxRange<Tokens>,
    pub label: Label,
    pub phrase: PhraseId,
}

/// A state no edge leads to: the root, and the end of a phrase list.
const NO: u32 = u32::MAX;

/// Phrases under labels, matched over one [`Field`] of the tokens.
///
/// Every occurrence of every phrase is reported, overlapping ones too,
/// in token order (by start, then end, then the order phrases were
/// added). A phrase added under two labels matches once for each; added
/// twice under one label, once.
pub struct PhraseMatcher {
    field: Field,
    vocab: Vocab,
    labels: Labels,
    /// The state the root's edge for each id leads to, or `NO`.
    root: Vec<u32>,
    edges: EdgeTable,
    /// Per state (state 0 is the root), the first phrase ending there.
    head: Vec<u32>,
    /// Per phrase, its label and the next phrase ending at its state.
    phrase_label: Vec<Label>,
    phrase_next: Vec<u32>,
    longest: usize,
}

impl PhraseMatcher {
    /// A matcher of phrases over `field`. Over [`Attr::Text`] a phrase
    /// matches tokens with the same text; over [`Attr::Lower`],
    /// [`Attr::Prefix`], [`Attr::Suffix`] and [`Attr::Shape`] the phrase's
    /// tokens are turned into that attribute first, as spaCy makes a
    /// pattern document's attributes; over [`Attr::Norm`], [`Attr::Lemma`],
    /// [`Attr::Stem`], [`Attr::Pos`] and [`Attr::Entity`] the phrase's
    /// tokens are the values themselves (lemmas, tags). Over a flag, each
    /// phrase token's flag is read from its text.
    ///
    /// [`Attr::Text`]: crate::Attr::Text
    /// [`Attr::Lower`]: crate::Attr::Lower
    /// [`Attr::Prefix`]: crate::Attr::Prefix
    /// [`Attr::Suffix`]: crate::Attr::Suffix
    /// [`Attr::Shape`]: crate::Attr::Shape
    /// [`Attr::Norm`]: crate::Attr::Norm
    /// [`Attr::Lemma`]: crate::Attr::Lemma
    /// [`Attr::Stem`]: crate::Attr::Stem
    /// [`Attr::Pos`]: crate::Attr::Pos
    /// [`Attr::Entity`]: crate::Attr::Entity
    pub fn new(field: Field) -> PhraseMatcher {
        PhraseMatcher {
            field,
            vocab: Vocab::default(),
            labels: Labels::default(),
            root: Vec::new(),
            edges: EdgeTable::default(),
            head: vec![NO],
            phrase_label: Vec::new(),
            phrase_next: Vec::new(),
            longest: 0,
        }
    }

    pub fn field(&self) -> Field {
        self.field
    }

    /// The id of one phrase token's value.
    fn value(&mut self, token: &str, scratch: &mut String) -> Result<u32, PatternError> {
        match self.field {
            Field::Attr(a) => Ok(self.vocab.intern(a.of_text(token, scratch))),
            Field::Flag(f) if f.text_decides() => Ok(u32::from(text_flags(token) & f.bit() != 0)),
            Field::Flag(_) => Err(PatternError::NotFromText(self.field)),
        }
    }

    /// Adds a phrase, given as its tokens, under `label`.
    pub fn add(&mut self, label: &str, tokens: &[&str]) -> Result<PhraseId, PatternError> {
        if tokens.is_empty() {
            return Err(PatternError::Empty);
        }
        let mut scratch = String::new();
        let mut ids = Vec::with_capacity(tokens.len());
        for t in tokens {
            ids.push(self.value(t, &mut scratch)?);
        }
        let label = self.labels.get_or_add(label);
        let mut state = 0u32;
        for (depth, &id) in ids.iter().enumerate() {
            let next = if depth == 0 {
                if self.root.len() <= id as usize {
                    self.root.resize(id as usize + 1, NO);
                }
                self.root[id as usize]
            } else {
                self.edges.get(state, id).unwrap_or(NO)
            };
            state = if next != NO {
                next
            } else {
                let made = self.head.len() as u32;
                self.head.push(NO);
                if depth == 0 {
                    self.root[id as usize] = made;
                } else {
                    self.edges.insert(state, id, made);
                }
                made
            };
        }
        self.longest = self.longest.max(ids.len());
        // The phrases ending here, in the order they were added.
        let mut last = NO;
        let mut p = self.head[state as usize];
        while p != NO {
            if self.phrase_label[p as usize] == label {
                return Ok(PhraseId(p));
            }
            last = p;
            p = self.phrase_next[p as usize];
        }
        let id = self.phrase_label.len() as u32;
        self.phrase_label.push(label);
        self.phrase_next.push(NO);
        if last == NO {
            self.head[state as usize] = id;
        } else {
            self.phrase_next[last as usize] = id;
        }
        Ok(PhraseId(id))
    }

    /// Removes a label and its phrases. Their trie states stay, unused.
    pub fn remove(&mut self, label: &str) -> Result<(), MatchError> {
        let gone = self.labels.remove(label)?;
        for state in 0..self.head.len() {
            let mut p = self.head[state];
            let mut prev = NO;
            while p != NO {
                let next = self.phrase_next[p as usize];
                if self.phrase_label[p as usize] == gone {
                    if prev == NO {
                        self.head[state] = next;
                    } else {
                        self.phrase_next[prev as usize] = next;
                    }
                } else {
                    prev = p;
                }
                p = next;
            }
        }
        Ok(())
    }

    pub fn label(&self, name: &str) -> Option<Label> {
        self.labels.get(name)
    }

    pub fn label_name(&self, label: Label) -> &str {
        self.labels.name(label)
    }

    pub fn contains(&self, name: &str) -> bool {
        self.labels.get(name).is_some()
    }

    /// How many labels, as spaCy's `len(phrase_matcher)`.
    pub fn len(&self) -> usize {
        self.labels.len()
    }

    pub fn is_empty(&self) -> bool {
        self.labels.len() == 0
    }

    /// The label a phrase was added under.
    pub fn phrase_label(&self, p: PhraseId) -> Label {
        self.phrase_label[p.0 as usize]
    }

    /// How many phrases were added, removed ones included.
    pub fn phrases(&self) -> usize {
        self.phrase_label.len()
    }

    /// How many trie states, the root included.
    pub fn states(&self) -> usize {
        self.head.len()
    }

    /// How many distinct token values the phrases hold.
    pub fn vocabulary(&self) -> usize {
        self.vocab.len()
    }

    /// The most tokens in one phrase.
    pub fn longest(&self) -> usize {
        self.longest
    }

    /// The bytes the matcher holds on the heap: the vocabulary, the root
    /// array, the edge table and the per-state and per-phrase columns.
    /// The allocator's own overhead is not counted.
    pub fn heap_bytes(&self) -> usize {
        self.vocab.heap_bytes()
            + self.root.capacity() * 4
            + self.edges.heap_bytes()
            + self.head.capacity() * 4
            + self.phrase_label.capacity() * 4
            + self.phrase_next.capacity() * 4
    }

    /// Every phrase occurrence in all of `input`'s tokens.
    pub fn find(&self, input: &Input<'_>) -> Result<Vec<PhraseMatch>, MatchError> {
        self.find_in(input, input.all())
    }

    /// Every phrase occurrence inside tokens `range`.
    pub fn find_in(&self, input: &Input<'_>, range: IxRange<Tokens>) -> Result<Vec<PhraseMatch>, MatchError> {
        let mut out = Vec::new();
        self.find_into(input, range, &mut out)?;
        Ok(out)
    }

    /// [`PhraseMatcher::find_in`], appending to `out`.
    pub fn find_into(
        &self,
        input: &Input<'_>,
        range: IxRange<Tokens>,
        out: &mut Vec<PhraseMatch>,
    ) -> Result<(), MatchError> {
        let range = input.range(range)?;
        if range.is_empty() || self.phrase_label.is_empty() {
            return Ok(());
        }
        let mut needs = Needs::default();
        let column = match self.field {
            Field::Attr(a) => {
                needs.attr(a);
                let mut cols = Columns::build(input, &range, needs, &self.vocab)?;
                std::mem::take(&mut cols.values[a as usize])
            }
            Field::Flag(f) => {
                needs.flags = f.bit();
                let cols = Columns::build(input, &range, needs, &self.vocab)?;
                cols.flags.iter().map(|&b| u32::from(b & f.bit() != 0)).collect()
            }
        };
        for start in 0..column.len() {
            let id = column[start];
            if id == NONE {
                continue;
            }
            let Some(&first) = self.root.get(id as usize) else { continue };
            let mut state = first;
            let mut end = start;
            while state != NO {
                end += 1;
                let mut p = self.head[state as usize];
                while p != NO {
                    let s = (range.start + start) as u32;
                    let e = (range.start + end) as u32;
                    out.push(PhraseMatch {
                        tokens: IxRange::new(s, e),
                        label: self.phrase_label[p as usize],
                        phrase: PhraseId(p),
                    });
                    p = self.phrase_next[p as usize];
                }
                if end == column.len() || column[end] == NONE {
                    break;
                }
                state = self.edges.get(state, column[end]).unwrap_or(NO);
            }
        }
        Ok(())
    }
}
