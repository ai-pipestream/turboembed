//! The term layers: a document's vocabulary, the term of each token, and
//! term counts per unit of a level (a chunk, a structure block, the whole
//! view).

use std::marker::PhantomData;

use turbo_analysis::{Anchor, Ix, Key, KeyId, Layer, LayerName};
use turbo_analysis_layers::{Strs, Tokens};

/// The distinct terms of one document, in byte order, each once. A term's
/// id is its place in that order, so the ids depend only on which terms
/// the document holds, not on where they first appear. A term is either the
/// text of a token in the view the terms were read from (`TermIndexer`) or
/// the selected column of `turbo_analysis_layers::Terms`
/// (`NormalizedTermIndexer`), byte for byte as handed to the indexer.
pub struct Vocabulary {
    pub text: Strs,
}

impl Vocabulary {
    pub fn get(&self, t: Ix<Vocabulary>) -> &str {
        self.text.get(t.get())
    }

    /// A term's id: a binary search, as the terms are in byte order.
    pub fn lookup(&self, term: &str) -> Option<Ix<Vocabulary>> {
        let (mut lo, mut hi) = (0, self.text.len());
        while lo < hi {
            let mid = (lo + hi) / 2;
            match self.text.get(mid).cmp(term) {
                std::cmp::Ordering::Less => lo = mid + 1,
                std::cmp::Ordering::Greater => hi = mid,
                std::cmp::Ordering::Equal => return Some(Ix::new(mid as u32)),
            }
        }
        None
    }

    pub fn len(&self) -> usize {
        self.text.len()
    }

    pub fn is_empty(&self) -> bool {
        self.text.is_empty()
    }

    /// The layer's check: terms in byte order, each once.
    /// One pass, each term read once.
    pub fn validate(&self) -> Result<(), String> {
        let mut terms = self.text.iter();
        if let Some(mut prev) = terms.next() {
            for term in terms {
                if prev >= term {
                    return Err("terms in byte order, each once".into());
                }
                prev = term;
            }
        }
        Ok(())
    }
}

impl Layer for Vocabulary {
    const ID: &'static str = "turbo:vocabulary";
    fn anchor() -> Anchor {
        Anchor::Document
    }
    fn len(&self) -> usize {
        self.text.len()
    }
    fn check(&self) -> Result<(), String> {
        self.validate()
    }
}

/// The term of each token, or none for a token that is not a term (one
/// with no letter or digit, such as a period).
pub struct IndexedTerms {
    pub term: Vec<Option<Ix<Vocabulary>>>,
}

impl Layer for IndexedTerms {
    const ID: &'static str = "turbo:indexed-terms";
    fn anchor() -> Anchor {
        Anchor::aligned::<Tokens>()
    }
    fn len(&self) -> usize {
        self.term.len()
    }
    fn indexes(&self) -> Vec<(KeyId, usize)> {
        let bound = self.term.iter().flatten().map(|t| t.get() + 1).max().unwrap_or(0);
        vec![(Key::<Vocabulary>::DEFAULT.id(), bound)]
    }
}

/// Compatibility name for retrieval IDs. Linguistic text forms use
/// [`turbo_analysis_layers::Terms`] and the separate `turbo:terms` identity.
pub type Terms = IndexedTerms;

/// Term counts for each item of layer `L`, the units of one level: each
/// chunk of `Chunks`, each block of `Structure`, or the whole document as
/// the one item of a `View`. Stored as sparse rows by column: unit i's
/// terms are entries `row[i]..row[i + 1]`, in ascending term id, each with
/// its count in the unit.
///
/// `df` is how many units of this layer hold each term. That is a
/// frequency within this one document, over this level's units; it is not
/// a corpus's document frequency, and BM25's idf needs the corpus's (see
/// `Collection`).
pub struct TermStats<L> {
    /// Where each unit's entries start, and one more for where the last
    /// ends: one longer than the units.
    pub row: Vec<u32>,
    pub term: Vec<Ix<Vocabulary>>,
    /// How often the entry's term occurs in the unit: at least once.
    pub tf: Vec<u32>,
    /// Each unit's length: the number of term tokens in it, the sum of its
    /// counts. Tokens that are not terms do not count.
    pub length: Vec<u32>,
    /// For each term of the vocabulary, the number of units holding it.
    pub df: Vec<u32>,
    pub _of: PhantomData<fn() -> L>,
}

impl<L> TermStats<L> {
    /// Unit `i`'s terms, in ascending id, and how often each occurs.
    pub fn unit(&self, i: Ix<L>) -> (&[Ix<Vocabulary>], &[u32]) {
        let r = self.row[i.get()] as usize..self.row[i.get() + 1] as usize;
        (&self.term[r.clone()], &self.tf[r])
    }

    /// How often term `t` occurs in unit `i`: a binary search over the
    /// unit's terms.
    pub fn tf(&self, i: Ix<L>, t: Ix<Vocabulary>) -> u32 {
        let (terms, tf) = self.unit(i);
        terms.binary_search(&t).map_or(0, |k| tf[k])
    }

    pub fn units(&self) -> impl Iterator<Item = Ix<L>> + use<L> {
        (0..self.length.len() as u32).map(Ix::new)
    }

    /// The layer's check, for any level: one row start per unit and one
    /// past the last entry, terms ascending and each once per unit, no
    /// count of zero, each length the sum of its counts, every term inside
    /// the vocabulary's `df`, and each df the number of units holding its
    /// term, so never more than the units.
    pub fn validate(&self) -> Result<(), String> {
        let n = self.length.len();
        if self.row.len() != n + 1 || self.row[0] != 0 || self.row[n] as usize != self.term.len() {
            return Err("one row start per unit and one past the last entry".into());
        }
        if self.tf.len() != self.term.len() {
            return Err("a count for every entry".into());
        }
        if !self.row.windows(2).all(|w| w[0] <= w[1]) {
            return Err("rows in unit order".into());
        }
        // One pass over the entries: each unit's terms ascending, its
        // counts nonzero and summing to its length, and each term counted
        // toward its df.
        let mut df = vec![0u32; self.df.len()];
        for i in 0..n {
            let (mut prev, mut sum) = (None, 0u64);
            for (t, &c) in
                self.term[self.row[i] as usize..self.row[i + 1] as usize].iter().zip(&self.tf[self.row[i] as usize..])
            {
                let t = t.get();
                if prev.is_some_and(|p| p >= t) {
                    return Err(format!("unit {i}: terms in ascending id, each once"));
                }
                if c == 0 {
                    return Err(format!("unit {i}: a count of zero"));
                }
                let Some(d) = df.get_mut(t) else {
                    return Err(format!("unit {i}: a term past the vocabulary's {}", self.df.len()));
                };
                *d += 1;
                (prev, sum) = (Some(t), sum + u64::from(c));
            }
            if sum != u64::from(self.length[i]) {
                return Err(format!("unit {i}: a length that is not the sum of its counts"));
            }
        }
        if df != self.df {
            return Err("a df that is not the number of units holding the term".into());
        }
        Ok(())
    }

    /// `validate` for these counts and for `vocabulary`, and one df per
    /// term of `vocabulary`: what a caller handing both to a builder must
    /// hold, checked before anything is changed.
    pub fn validate_with(&self, vocabulary: &Vocabulary) -> Result<(), String> {
        self.validate()?;
        vocabulary.validate()?;
        if self.df.len() != vocabulary.len() {
            return Err(format!("{} df counts for a vocabulary of {}", self.df.len(), vocabulary.len()));
        }
        Ok(())
    }
}

impl<L: Layer> Layer for TermStats<L> {
    const ID: &'static str = "turbo:term-stats";
    const OF: Option<&'static LayerName> = Some(L::NAME);
    fn anchor() -> Anchor {
        Anchor::aligned::<L>()
    }
    fn len(&self) -> usize {
        self.length.len()
    }
    fn indexes(&self) -> Vec<(KeyId, usize)> {
        vec![(Key::<Vocabulary>::DEFAULT.id(), self.df.len())]
    }
    fn check(&self) -> Result<(), String> {
        self.validate()
    }
}
