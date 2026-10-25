//! [`WordNetAnnotator`]: the senses and expansions of each token, as the
//! `WordNetSenses` and `LexicalExpansions` layers.
//!
//! A token is looked up as its lemma from the `Lemmas` layer when the
//! document has one and it is not `O`; otherwise as what Morphy makes of
//! it when the annotator has Morphy; otherwise as written. A POS tag
//! picks the token's one WordNet part of speech, and a tag WordNet has no
//! part of speech for (a determiner, punctuation) leaves the token out.
//! Without POS tags every part of speech is tried, in noun, verb,
//! adjective, adverb order.

use std::sync::Arc;

use turbo_analysis::{Annotator, Error, Ix, Refs, View};
use turbo_analysis_layers::{
    ExpansionRow, Lemmas, LexicalExpansions, PosTags, SAME_SYNSET, SenseRow, Tokens, WordNetSenses,
};

use crate::expansion::{Expansion, ExpansionSource, LexicalExpander};
use crate::lexicon::LexicalKnowledgeBase;
use crate::morphy::{MorphyExceptions, MorphyLemmatizer, UNKNOWN_LEMMA};
use crate::pos::WordNetPos;
use crate::relation::WordNetRelation;
use crate::{WordNetError, fold};

/// How many senses and expansions the annotator writes per token.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct WordNetOptions {
    /// Senses per part of speech (at least 1).
    pub max_senses: usize,
    /// Hypernym steps an expansion walks (0 for none).
    pub hypernym_depth: usize,
    pub include_hyponyms: bool,
    /// Expansions per part of speech (at least 1).
    pub max_expansions: usize,
}

impl Default for WordNetOptions {
    /// `LexicalExpander`'s defaults: 3 senses, 1 hypernym step, no
    /// hyponyms, 20 expansions.
    fn default() -> Self {
        WordNetOptions { max_senses: 3, hypernym_depth: 1, include_hyponyms: false, max_expansions: 20 }
    }
}

/// Reads `(View, Tokens, Option<PosTags>, Option<Lemmas>)` and writes
/// `(WordNetSenses, LexicalExpansions)` from one lexicon.
pub struct WordNetAnnotator<L: LexicalKnowledgeBase + Send + Sync + 'static> {
    lexicon: Arc<L>,
    lexicon_id: Arc<str>,
    options: WordNetOptions,
    expander: LexicalExpander<Arc<L>>,
    morphy: Option<MorphyLemmatizer<Arc<L>>>,
    config: u64,
}

impl<L: LexicalKnowledgeBase + Send + Sync + 'static> WordNetAnnotator<L> {
    /// An annotator over `lexicon`, which the layers name `lexicon_id`.
    ///
    /// # Errors
    ///
    /// Returns [`WordNetError::InvalidInput`] when `max_senses` or
    /// `max_expansions` is 0, or `hypernym_depth` is past 255, the most
    /// steps the layer records.
    pub fn new(lexicon: Arc<L>, lexicon_id: &str, options: WordNetOptions) -> Result<Self, WordNetError> {
        if options.hypernym_depth > usize::from(u8::MAX) {
            return Err(WordNetError::invalid(format!("hypernym depth past 255: {}", options.hypernym_depth)));
        }
        if options.max_senses == 0 {
            return Err(WordNetError::invalid("max_senses must be positive"));
        }
        let expander = LexicalExpander::builder(Arc::clone(&lexicon))
            .max_senses(options.max_senses)?
            .hypernym_depth(options.hypernym_depth)
            .include_hyponyms(options.include_hyponyms)
            .max_expansions(options.max_expansions)?
            .build();
        let config = configuration_hash(lexicon_id, options, None);
        Ok(WordNetAnnotator { lexicon, lexicon_id: lexicon_id.into(), options, expander, morphy: None, config })
    }

    /// Lemmatizes tokens with Morphy and `exceptions` when the document
    /// has no lemma for them.
    pub fn with_morphy(mut self, exceptions: MorphyExceptions) -> Self {
        self.config = configuration_hash(&self.lexicon_id, self.options, Some(&exceptions));
        self.morphy = Some(MorphyLemmatizer::new(Arc::clone(&self.lexicon), exceptions));
        self
    }

    /// The lemma token `i` is looked up as for `pos`.
    fn lemma(&self, token: &str, lemmas: Option<&Lemmas>, i: usize, pos: WordNetPos) -> String {
        if let Some(l) = lemmas.map(|l| l.lemma.get(i))
            && !l.is_empty()
            && l != UNKNOWN_LEMMA
        {
            return l.to_owned();
        }
        if !self.lexicon.contains(token, pos)
            && let Some(morphy) = &self.morphy
        {
            let lemma = morphy.lemmatize_one(token, pos.java_name());
            if lemma != UNKNOWN_LEMMA {
                return lemma;
            }
        }
        token.to_owned()
    }
}

fn fail(detail: String) -> Error {
    Error::Annotator { annotator: "turbo-wordnet:senses", detail }
}

/// The layer code of an expansion's relation.
fn relation_code(source: &ExpansionSource) -> u8 {
    source.relation.map_or(SAME_SYNSET, WordNetRelation::layer_code)
}

// Resource contents are immutable; sort and hash them once during construction.
fn configuration_hash(lexicon_id: &str, options: WordNetOptions, exceptions: Option<&MorphyExceptions>) -> u64 {
    const FNV_OFFSET: u64 = 0xcbf2_9ce4_8422_2325;
    const FNV_PRIME: u64 = 0x0100_0000_01b3;
    let o = &options;
    let mut h = FNV_OFFSET;
    let fields = [o.max_senses as u64, o.hypernym_depth as u64, u64::from(o.include_hyponyms), o.max_expansions as u64];
    let mut mix = |bytes: &[u8]| {
        for &b in bytes {
            h = (h ^ u64::from(b)).wrapping_mul(FNV_PRIME);
        }
    };
    mix(lexicon_id.as_bytes());
    for field in fields {
        mix(&field.to_le_bytes());
    }
    if let Some(exceptions) = exceptions {
        mix(b"morphy-exceptions-v1");
        exceptions.visit_entries(|pos, form, bases| {
            mix(&[pos.code() as u8]);
            mix(&(form.len() as u64).to_le_bytes());
            mix(form.as_bytes());
            mix(&(bases.len() as u64).to_le_bytes());
            for base in bases {
                mix(&(base.len() as u64).to_le_bytes());
                mix(base.as_bytes());
            }
        });
    }
    h ^ u64::from(exceptions.is_some())
}

impl<L: LexicalKnowledgeBase + Send + Sync + 'static> Annotator for WordNetAnnotator<L> {
    const ID: &'static str = "turbo-wordnet:senses";
    type Reads = (View, Tokens, Option<PosTags>, Option<Lemmas>);
    type Writes = (WordNetSenses, LexicalExpansions);

    fn annotate(&self, (view, tokens, pos, lemmas): Refs<'_, Self>) -> Result<Self::Writes, Error> {
        if tokens.spans.view() != view.id() {
            return Err(fail("the tokens are in another view".into()));
        }
        let n = tokens.spans.len();
        if let Some(p) = pos.filter(|p| p.tags.len() != n) {
            return Err(fail(format!("pos has {} items, tokens has {n}", p.tags.len())));
        }
        if let Some(l) = lemmas.filter(|l| l.lemma.len() != n) {
            return Err(fail(format!("lemmas has {} items, tokens has {n}", l.lemma.len())));
        }
        let text = view.text();
        let mut senses = WordNetSenses::new(view.id(), Arc::clone(&self.lexicon_id));
        let mut expansions = LexicalExpansions::new(view.id(), Arc::clone(&self.lexicon_id));
        let mut found: Vec<(Expansion, ExpansionSource)> = Vec::new();
        for i in 0..n {
            let ix = Ix::new(i as u32);
            let span = tokens.spans.get(ix);
            let token = text
                .get(span.start as usize..span.end as usize)
                .ok_or_else(|| fail(format!("token {i} is not a slice of the view")))?;
            let one;
            let poses: &[WordNetPos] = match pos {
                Some(p) => match MorphyLemmatizer::<Arc<L>>::pos_from_tag(p.tagset.label(p.tags[i])) {
                    Some(wp) => {
                        one = [wp];
                        &one
                    }
                    None => continue,
                },
                None => &WordNetPos::ALL,
            };
            for &wp in poses {
                let lemma = self.lemma(token, lemmas, i, wp);
                let folded = fold(&lemma);
                for (rank, synset) in
                    self.lexicon.lookup(&folded, wp).into_iter().take(self.options.max_senses).enumerate()
                {
                    let row = SenseRow {
                        token: ix,
                        synset_id: synset.id(),
                        sense_id: "",
                        pos: wp.code() as u8,
                        lemma: &folded,
                        synset_lemmas: synset.lemmas(),
                        gloss: synset.gloss(),
                        sense_rank: rank as u32,
                    };
                    senses.push(span, row);
                }
                found.clear();
                found.extend(self.expander.expand_sourced(&folded, wp).map_err(|e| fail(e.to_string()))?);
                for (e, source) in &found {
                    let row = ExpansionRow {
                        token: ix,
                        term: e.term(),
                        relation: relation_code(source),
                        depth: e.depth() as u8,
                        sense_rank: e.sense_rank() as u32,
                        weight: e.weight(),
                        synset_id: &source.synset_id,
                    };
                    expansions.push(span, row);
                }
            }
        }
        Ok((senses, expansions))
    }

    fn config_hash(&self) -> u64 {
        self.config
    }
}
