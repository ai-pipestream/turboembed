//! WordNet layers: the senses a lexicon gives each token
//! ([`WordNetSenses`]) and the terms it expands each token to
//! ([`LexicalExpansions`]). Both hold one row per (token, sense) or
//! (token, term), spanning the token, so a token with three senses has
//! three rows over the same text.
//!
//! The relation of an expansion is a code: [`SAME_SYNSET`] for a member
//! of one of the token's own synsets, and `1 + i` for the relation at
//! index `i` of `turbo_wordnet::WordNetRelation::ALL`, up to
//! [`WORDNET_RELATIONS`]. This crate does not depend on `turbo-wordnet`,
//! which owns the relation enum, so the codes are checked by range here
//! and mapped by that crate.

use std::sync::Arc;

use turbo_analysis::{Ix, Key, KeyId, Layer, Overlap, Span, Spans, ViewId};

use crate::{Strs, Tokens};

/// The relation code of a synonym: a member of one of the token's own
/// synsets, reached by no relation.
pub const SAME_SYNSET: u8 = 0;

/// How many typed relations there are, so the highest relation code.
pub const WORDNET_RELATIONS: u8 = 106;

/// The WordNet part of speech codes: noun, verb, adjective, adverb.
pub const WORDNET_POS: [u8; 4] = *b"nvar";

/// One row per sense of a token, in token order and then sense rank.
/// Row i's synset members are `synset_lemmas[synset_lemma_row[i]..
/// synset_lemma_row[i + 1]]`.
pub struct WordNetSenses {
    pub spans: Spans<WordNetSenses>,
    pub token: Vec<Ix<Tokens>>,
    pub synset_id: Strs,
    /// The lexicon's sense id, empty when it gives none.
    pub sense_id: Strs,
    /// One of [`WORDNET_POS`].
    pub pos: Vec<u8>,
    /// The lemma the token was looked up as.
    pub lemma: Strs,
    pub synset_lemmas: Strs,
    pub synset_lemma_row: Vec<u32>,
    pub gloss: Strs,
    /// 0 for the lexicon's first sense of the lemma.
    pub sense_rank: Vec<u32>,
    /// The lexicon the senses come from, such as `oewn:2024`.
    pub lexicon_id: Arc<str>,
}

/// The values of one [`WordNetSenses`] row.
pub struct SenseRow<'a> {
    pub token: Ix<Tokens>,
    pub synset_id: &'a str,
    pub sense_id: &'a str,
    pub pos: u8,
    pub lemma: &'a str,
    pub synset_lemmas: &'a [String],
    pub gloss: &'a str,
    pub sense_rank: u32,
}

impl WordNetSenses {
    pub fn new(view: ViewId, lexicon_id: Arc<str>) -> WordNetSenses {
        WordNetSenses {
            spans: Spans::new(view),
            token: Vec::new(),
            synset_id: Strs::default(),
            sense_id: Strs::default(),
            pos: Vec::new(),
            lemma: Strs::default(),
            synset_lemmas: Strs::default(),
            synset_lemma_row: vec![0],
            gloss: Strs::default(),
            sense_rank: Vec::new(),
            lexicon_id,
        }
    }

    pub fn push(&mut self, span: Span, row: SenseRow<'_>) -> Ix<WordNetSenses> {
        self.token.push(row.token);
        self.synset_id.push(row.synset_id);
        self.sense_id.push(row.sense_id);
        self.pos.push(row.pos);
        self.lemma.push(row.lemma);
        for l in row.synset_lemmas {
            self.synset_lemmas.push(l);
        }
        self.synset_lemma_row.push(self.synset_lemmas.len() as u32);
        self.gloss.push(row.gloss);
        self.sense_rank.push(row.sense_rank);
        self.spans.push(span)
    }

    /// The members of row i's synset.
    pub fn members(&self, i: usize) -> impl Iterator<Item = &str> {
        (self.synset_lemma_row[i] as usize..self.synset_lemma_row[i + 1] as usize).map(|k| self.synset_lemmas.get(k))
    }
}

impl Layer for WordNetSenses {
    const ID: &'static str = "turbo:wordnet-senses";
    const OVERLAP: Overlap = Overlap::Any;
    positional!(WordNetSenses);

    fn indexes(&self) -> Vec<(KeyId, usize)> {
        vec![(Key::<Tokens>::DEFAULT.id(), self.token.iter().map(|t| t.get() + 1).max().unwrap_or(0))]
    }

    fn check(&self) -> Result<(), String> {
        let n = self.spans.len();
        let lens = [
            self.token.len(),
            self.synset_id.len(),
            self.sense_id.len(),
            self.pos.len(),
            self.lemma.len(),
            self.gloss.len(),
            self.sense_rank.len(),
        ];
        if lens.iter().any(|&l| l != n) {
            return Err(format!("{n} senses and columns of {lens:?}"));
        }
        let row = &self.synset_lemma_row;
        if row.len() != n + 1 || row[0] != 0 || row[n] as usize != self.synset_lemmas.len() {
            return Err("one synset lemma row start per sense and one past the last lemma".into());
        }
        if !row.windows(2).all(|w| w[0] <= w[1]) {
            return Err("synset lemma rows in sense order".into());
        }
        if let Some(p) = self.pos.iter().find(|p| !WORDNET_POS.contains(p)) {
            return Err(format!("a part of speech code {p} that is none of n, v, a, r"));
        }
        if !self.token.windows(2).all(|w| w[0] <= w[1]) {
            return Err("senses not in token order".into());
        }
        self.spans.check(Self::OVERLAP)
    }
}

/// One row per expansion of a token, in token order and then the
/// expander's ranking.
pub struct LexicalExpansions {
    pub spans: Spans<LexicalExpansions>,
    pub token: Vec<Ix<Tokens>>,
    pub term: Strs,
    /// [`SAME_SYNSET`], or `1 + ` the index of the relation followed in
    /// `WordNetRelation::ALL`.
    pub relation: Vec<u8>,
    /// Relation steps from the token's sense: 0 for a synonym.
    pub depth: Vec<u8>,
    /// The rank of the token's sense the expansion came from.
    pub sense_rank: Vec<u32>,
    /// In (0, 1].
    pub weight: Vec<f64>,
    pub lexicon_id: Arc<str>,
    /// The synset the term is a member of.
    pub synset_id: Strs,
}

/// The values of one [`LexicalExpansions`] row.
pub struct ExpansionRow<'a> {
    pub token: Ix<Tokens>,
    pub term: &'a str,
    pub relation: u8,
    pub depth: u8,
    pub sense_rank: u32,
    pub weight: f64,
    pub synset_id: &'a str,
}

impl LexicalExpansions {
    pub fn new(view: ViewId, lexicon_id: Arc<str>) -> LexicalExpansions {
        LexicalExpansions {
            spans: Spans::new(view),
            token: Vec::new(),
            term: Strs::default(),
            relation: Vec::new(),
            depth: Vec::new(),
            sense_rank: Vec::new(),
            weight: Vec::new(),
            lexicon_id,
            synset_id: Strs::default(),
        }
    }

    pub fn push(&mut self, span: Span, row: ExpansionRow<'_>) -> Ix<LexicalExpansions> {
        self.token.push(row.token);
        self.term.push(row.term);
        self.relation.push(row.relation);
        self.depth.push(row.depth);
        self.sense_rank.push(row.sense_rank);
        self.weight.push(row.weight);
        self.synset_id.push(row.synset_id);
        self.spans.push(span)
    }
}

impl Layer for LexicalExpansions {
    const ID: &'static str = "turbo:lexical-expansions";
    const OVERLAP: Overlap = Overlap::Any;
    positional!(LexicalExpansions);

    fn indexes(&self) -> Vec<(KeyId, usize)> {
        vec![(Key::<Tokens>::DEFAULT.id(), self.token.iter().map(|t| t.get() + 1).max().unwrap_or(0))]
    }

    fn check(&self) -> Result<(), String> {
        let n = self.spans.len();
        let lens = [
            self.token.len(),
            self.term.len(),
            self.relation.len(),
            self.depth.len(),
            self.sense_rank.len(),
            self.weight.len(),
            self.synset_id.len(),
        ];
        if lens.iter().any(|&l| l != n) {
            return Err(format!("{n} expansions and columns of {lens:?}"));
        }
        if let Some(r) = self.relation.iter().find(|&&r| r > WORDNET_RELATIONS) {
            return Err(format!("a relation code {r} past the {WORDNET_RELATIONS} relations"));
        }
        if let Some(i) = (0..n).find(|&i| (self.relation[i] == SAME_SYNSET) != (self.depth[i] == 0)) {
            return Err(format!("expansion {i}: depth 0 if and only if the term is in the token's own synset"));
        }
        if let Some(w) = self.weight.iter().find(|w| !(w.is_finite() && **w > 0.0 && **w <= 1.0)) {
            return Err(format!("a weight outside (0, 1]: {w}"));
        }
        if !self.token.windows(2).all(|w| w[0] <= w[1]) {
            return Err("expansions not in token order".into());
        }
        self.spans.check(Self::OVERLAP)
    }
}

#[cfg(test)]
#[path = "wordnet_tests.rs"]
mod tests;
