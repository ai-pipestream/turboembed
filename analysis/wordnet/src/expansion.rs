//! Bounded query/index lexical expansion over a [`LexicalKnowledgeBase`].

use std::cmp::Ordering;
use std::collections::{HashMap, HashSet, VecDeque};
use std::sync::Arc;

use crate::error::WordNetError;
use crate::folding::fold;
use crate::lexicon::{LexicalKnowledgeBase, Synset};
use crate::morphy::{MorphyLemmatizer, UNKNOWN_LEMMA};
use crate::pos::WordNetPos;
use crate::relation::WordNetRelation;

/// Source relation kind of an [`Expansion`] candidate (`LexicalExpander.Kind`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum ExpansionKind {
    /// A member of one of the term's own synsets.
    Synonym,
    /// A lemma of an ancestor synset, `depth` steps up.
    Hypernym,
    /// A lemma of a direct child synset (`depth == 1`).
    Hyponym,
}

/// One weighted expansion of a term (`LexicalExpander.Expansion`).
#[derive(Debug, Clone, PartialEq)]
pub struct Expansion {
    term: String,
    kind: ExpansionKind,
    depth: usize,
    sense_rank: usize,
    weight: f64,
}

impl Expansion {
    /// Creates and validates a new [`Expansion`] entry.
    ///
    /// # Errors
    ///
    /// Returns [`WordNetError::InvalidInput`] if `term` is blank, `depth` does not match `kind`
    /// (`0` for `Synonym`, `>= 1` for `Hypernym`, `1` for `Hyponym`), or `weight` is not in `(0.0, 1.0]`.
    pub fn new(
        term: impl Into<String>,
        kind: ExpansionKind,
        depth: usize,
        sense_rank: usize,
        weight: f64,
    ) -> Result<Self, WordNetError> {
        let term = term.into();
        if term.trim().is_empty() {
            return Err(WordNetError::invalid("term must not be null or blank"));
        }
        match kind {
            ExpansionKind::Synonym if depth != 0 => {
                return Err(WordNetError::invalid(format!(
                    "depth must be 0 for a synonym: {depth}"
                )));
            }
            ExpansionKind::Hypernym if depth == 0 => {
                return Err(WordNetError::invalid(format!(
                    "depth must be positive for a hypernym: {depth}"
                )));
            }
            ExpansionKind::Hyponym if depth != 1 => {
                return Err(WordNetError::invalid(format!(
                    "depth must be 1 for a hyponym: {depth}"
                )));
            }
            _ => {}
        }
        if !(weight > 0.0 && weight <= 1.0) {
            return Err(WordNetError::invalid(format!(
                "weight must be in (0, 1]: {weight}"
            )));
        }
        Ok(Self {
            term,
            kind,
            depth,
            sense_rank,
            weight,
        })
    }

    /// Returns the expanded term in the lexicon's written form.
    #[inline]
    pub fn term(&self) -> &str {
        &self.term
    }

    /// Returns how the term relates to the input ([`ExpansionKind`]).
    #[inline]
    pub fn kind(&self) -> ExpansionKind {
        self.kind
    }

    /// Returns the relation distance (`0` for synonyms, `>= 1` for hypernyms, `1` for hyponyms).
    #[inline]
    pub fn depth(&self) -> usize {
        self.depth
    }

    /// Returns the 0-based rank of the input sense this expansion came from.
    #[inline]
    pub fn sense_rank(&self) -> usize {
        self.sense_rank
    }

    /// Returns the heuristic weight in `(0.0, 1.0]`.
    #[inline]
    pub fn weight(&self) -> f64 {
        self.weight
    }
}

type LemmatizeFn = Arc<dyn Fn(&str, &str) -> String + Send + Sync>;

/// Bounded lexical expander for query and index expansion (`opennlp.wordnet.LexicalExpander`).
#[derive(Clone)]
pub struct LexicalExpander<L: LexicalKnowledgeBase> {
    lexicon: L,
    lemmatizer: Option<LemmatizeFn>,
    max_senses: usize,
    hypernym_depth: usize,
    include_hyponyms: bool,
    max_expansions: usize,
    sense_decay: f64,
    depth_decay: f64,
}

/// Builder for [`LexicalExpander`].
pub struct LexicalExpanderBuilder<L: LexicalKnowledgeBase> {
    lexicon: L,
    lemmatizer: Option<LemmatizeFn>,
    max_senses: usize,
    hypernym_depth: usize,
    include_hyponyms: bool,
    max_expansions: usize,
    sense_decay: f64,
    depth_decay: f64,
}

impl<L: LexicalKnowledgeBase> LexicalExpander<L> {
    /// Starts a builder over `lexicon` with default settings (`max_senses = 3`,
    /// `hypernym_depth = 1`, `include_hyponyms = false`, `max_expansions = 20`,
    /// `sense_decay = 0.5`, `depth_decay = 0.5`).
    pub fn builder(lexicon: L) -> LexicalExpanderBuilder<L> {
        LexicalExpanderBuilder {
            lexicon,
            lemmatizer: None,
            max_senses: 3,
            hypernym_depth: 1,
            include_hyponyms: false,
            max_expansions: 20,
            sense_decay: 0.5,
            depth_decay: 0.5,
        }
    }

    /// Expands `term` for one part of speech.
    ///
    /// # Errors
    ///
    /// Returns [`WordNetError::InvalidInput`] if `term` is blank.
    pub fn expand(&self, term: &str, pos: WordNetPos) -> Result<Vec<Expansion>, WordNetError> {
        self.collect(term, &[pos])
    }

    /// Expands `term` across all parts of speech (`WordNetPos::ALL`).
    ///
    /// # Errors
    ///
    /// Returns [`WordNetError::InvalidInput`] if `term` is blank.
    pub fn expand_all_pos(&self, term: &str) -> Result<Vec<Expansion>, WordNetError> {
        self.collect(term, &WordNetPos::ALL)
    }

    fn collect(&self, term: &str, poses: &[WordNetPos]) -> Result<Vec<Expansion>, WordNetError> {
        if term.trim().is_empty() {
            return Err(WordNetError::invalid("term must not be null or blank"));
        }
        let mut best: HashMap<String, Expansion> = HashMap::new();
        let mut excluded: HashSet<String> = HashSet::new();
        excluded.insert(fold(term));

        for &pos in poses {
            let Some(subject) = self.resolve_subject(term, pos) else {
                continue;
            };
            let senses = self.lexicon.lookup(&subject, pos);
            let sense_count = senses.len().min(self.max_senses);
            for (rank, sense) in senses.into_iter().take(sense_count).enumerate() {
                let sense_weight = self.sense_decay.powi(rank as i32);
                self.expand_sense(sense, rank, sense_weight, &mut best, &excluded);
            }
        }

        let mut ordered: Vec<Expansion> = best.into_values().collect();
        ordered.sort_by(|a, b| {
            b.weight
                .partial_cmp(&a.weight)
                .unwrap_or(Ordering::Equal)
                .then_with(|| a.kind.cmp(&b.kind))
                .then_with(|| a.term.cmp(&b.term))
        });
        if ordered.len() > self.max_expansions {
            ordered.truncate(self.max_expansions);
        }
        Ok(ordered)
    }

    fn resolve_subject(&self, term: &str, pos: WordNetPos) -> Option<String> {
        if self.lexicon.contains(term, pos) {
            return Some(term.to_string());
        }
        let lemmatizer = self.lemmatizer.as_ref()?;
        let lemma = lemmatizer(term, pos.java_name());
        if lemma == UNKNOWN_LEMMA || !self.lexicon.contains(&lemma, pos) {
            return None;
        }
        Some(lemma)
    }

    fn expand_sense(
        &self,
        sense: &Synset,
        rank: usize,
        sense_weight: f64,
        best: &mut HashMap<String, Expansion>,
        excluded: &HashSet<String>,
    ) {
        if sense_weight == 0.0 {
            return;
        }
        for lemma in sense.lemmas() {
            if let Ok(exp) = Expansion::new(lemma.clone(), ExpansionKind::Synonym, 0, rank, sense_weight) {
                Self::offer(best, excluded, exp);
            }
        }

        if self.hypernym_depth > 0 {
            let mut visited: HashSet<String> = HashSet::new();
            visited.insert(sense.id().to_string());
            let mut frontier: VecDeque<String> = Self::hypernyms_of(sense).into();
            let mut next: VecDeque<String> = VecDeque::new();
            for depth in 1..=self.hypernym_depth {
                if frontier.is_empty() {
                    break;
                }
                let depth_weight = sense_weight * self.depth_decay.powi(depth as i32);
                if depth_weight == 0.0 {
                    break;
                }
                while let Some(id) = frontier.pop_front() {
                    if !visited.insert(id.clone()) {
                        continue;
                    }
                    let Some(ancestor) = self.lexicon.synset(&id) else {
                        continue;
                    };
                    for lemma in ancestor.lemmas() {
                        if let Ok(exp) = Expansion::new(
                            lemma.clone(),
                            ExpansionKind::Hypernym,
                            depth,
                            rank,
                            depth_weight,
                        ) {
                            Self::offer(best, excluded, exp);
                        }
                    }
                    next.extend(Self::hypernyms_of(ancestor));
                }
                frontier.append(&mut next);
            }
        }

        let hyponym_weight = sense_weight * self.depth_decay;
        if self.include_hyponyms && hyponym_weight > 0.0 {
            let children = sense
                .related(WordNetRelation::Hyponym)
                .iter()
                .chain(sense.related(WordNetRelation::InstanceHyponym).iter());
            for id in children {
                if let Some(child) = self.lexicon.synset(id) {
                    for lemma in child.lemmas() {
                        if let Ok(exp) = Expansion::new(
                            lemma.clone(),
                            ExpansionKind::Hyponym,
                            1,
                            rank,
                            hyponym_weight,
                        ) {
                            Self::offer(best, excluded, exp);
                        }
                    }
                }
            }
        }
    }

    fn hypernyms_of(synset: &Synset) -> Vec<String> {
        let direct = synset.related(WordNetRelation::Hypernym);
        let instance = synset.related(WordNetRelation::InstanceHypernym);
        let mut all = Vec::with_capacity(direct.len() + instance.len());
        all.extend(direct.iter().cloned());
        all.extend(instance.iter().cloned());
        all
    }

    fn offer(
        best: &mut HashMap<String, Expansion>,
        excluded: &HashSet<String>,
        candidate: Expansion,
    ) {
        let key = fold(candidate.term());
        if excluded.contains(&key) {
            return;
        }
        match best.get(&key) {
            None => {
                best.insert(key, candidate);
            }
            Some(current) if candidate.weight() > current.weight() => {
                best.insert(key, candidate);
            }
            _ => {}
        }
    }
}

impl<L: LexicalKnowledgeBase> LexicalExpanderBuilder<L> {
    /// Configures a [`MorphyLemmatizer`] used when the input term is not directly in the lexicon.
    pub fn lemmatizer<M>(mut self, lemmatizer: MorphyLemmatizer<M>) -> Self
    where
        M: LexicalKnowledgeBase + Send + Sync + 'static,
    {
        self.lemmatizer = Some(Arc::new(move |tok, tag| lemmatizer.lemmatize_one(tok, tag)));
        self
    }

    /// Configures a custom lemmatizer callback `(token, pos_tag) -> lemma` (or `"O"`).
    pub fn lemmatizer_fn<F>(mut self, f: F) -> Self
    where
        F: Fn(&str, &str) -> String + Send + Sync + 'static,
    {
        self.lemmatizer = Some(Arc::new(f));
        self
    }

    /// Configures how many senses are expanded per part of speech (`>= 1`, default `3`).
    pub fn max_senses(mut self, max_senses: usize) -> Result<Self, WordNetError> {
        if max_senses < 1 {
            return Err(WordNetError::invalid(format!(
                "maxSenses must be positive: {max_senses}"
            )));
        }
        self.max_senses = max_senses;
        Ok(self)
    }

    /// Configures how many hypernym steps are walked (`0` disables hypernym expansion, default `1`).
    pub fn hypernym_depth(mut self, hypernym_depth: usize) -> Self {
        self.hypernym_depth = hypernym_depth;
        self
    }

    /// Configures whether direct hyponyms are included (default `false`).
    pub fn include_hyponyms(mut self, include_hyponyms: bool) -> Self {
        self.include_hyponyms = include_hyponyms;
        self
    }

    /// Configures the maximum number of expansions returned after ranking (`>= 1`, default `20`).
    pub fn max_expansions(mut self, max_expansions: usize) -> Result<Self, WordNetError> {
        if max_expansions < 1 {
            return Err(WordNetError::invalid(format!(
                "maxExpansions must be positive: {max_expansions}"
            )));
        }
        self.max_expansions = max_expansions;
        Ok(self)
    }

    /// Configures the weight multiplier applied per sense rank step in `(0.0, 1.0]` (default `0.5`).
    pub fn sense_decay(mut self, sense_decay: f64) -> Result<Self, WordNetError> {
        if !(sense_decay > 0.0 && sense_decay <= 1.0) {
            return Err(WordNetError::invalid(format!(
                "senseDecay must be in (0, 1]: {sense_decay}"
            )));
        }
        self.sense_decay = sense_decay;
        Ok(self)
    }

    /// Configures the weight multiplier applied per hypernym or hyponym step in `(0.0, 1.0]` (default `0.5`).
    pub fn depth_decay(mut self, depth_decay: f64) -> Result<Self, WordNetError> {
        if !(depth_decay > 0.0 && depth_decay <= 1.0) {
            return Err(WordNetError::invalid(format!(
                "depthDecay must be in (0, 1]: {depth_decay}"
            )));
        }
        self.depth_decay = depth_decay;
        Ok(self)
    }

    /// Builds the configured [`LexicalExpander`].
    pub fn build(self) -> LexicalExpander<L> {
        LexicalExpander {
            lexicon: self.lexicon,
            lemmatizer: self.lemmatizer,
            max_senses: self.max_senses,
            hypernym_depth: self.hypernym_depth,
            include_hyponyms: self.include_hyponyms,
            max_expansions: self.max_expansions,
            sense_decay: self.sense_decay,
            depth_decay: self.depth_decay,
        }
    }
}
