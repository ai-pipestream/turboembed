//! The feature generation framework of `opennlp.tools.util.featuregen`:
//! the [`FeatureGenerator`] trait (`AdaptiveFeatureGenerator`), the
//! generators the English part-of-speech, chunker and name finder models
//! use, and [`descriptor`], which builds generators from the
//! `generator.featuregen` XML stored in a model.
//!
//! Each generator writes the same strings as its Java class, in the same
//! order, so a model sums the same parameters in the same order and gives
//! bit-identical probabilities. The Java class each one ports is named in
//! its documentation.
//!
//! | Java class | Here |
//! |---|---|
//! | `AggregatedFeatureGenerator` | [`Aggregated`] |
//! | `CachedFeatureGenerator` | [`Cached`] |
//! | `WindowFeatureGenerator` | [`Window`] |
//! | `TokenFeatureGenerator` | [`TokenGenerator`] |
//! | `TokenClassFeatureGenerator` | [`TokenClass`] |
//! | `TokenPatternFeatureGenerator` | [`TokenPattern`] |
//! | `PrefixFeatureGenerator`, `SuffixFeatureGenerator` | [`Prefix`], [`Suffix`] |
//! | `CharacterNgramFeatureGenerator` | [`CharacterNgram`] |
//! | `BigramNameFeatureGenerator`, `TrigramNameFeatureGenerator` | [`BigramName`], [`TrigramName`] |
//! | `SentenceFeatureGenerator` | [`SentencePosition`] |
//! | `OutcomePriorFeatureGenerator` (also made by `DefinitionFeatureGeneratorFactory`) | [`OutcomePrior`] |
//! | `DocumentBeginFeatureGenerator` | [`DocumentBegin`] |
//! | `AdditionalContextFeatureGenerator` | [`AdditionalContext`] |
//! | `PreviousMapFeatureGenerator`, `PreviousTwoMapFeatureGenerator` | [`PreviousMap`], [`PreviousTwoMap`] |
//! | `PosTaggerFeatureGenerator` | [`PreviousTags`] |
//! | `BrownTokenFeatureGenerator`, `BrownTokenClassFeatureGenerator`, `BrownBigramFeatureGenerator` | [`BrownToken`], [`BrownTokenClass`], [`BrownBigram`] |
//! | `WordClusterFeatureGenerator` | [`WordCluster`] |
//! | `DictionaryFeatureGenerator` (with `InSpanGenerator` and `DictionaryNameFinder`) | [`InDictionary`] |
//!
//! Not ported: `POSTaggerNameFeatureGenerator`, which runs a whole
//! part-of-speech tagger inside the name finder, and
//! `EmojiAnnotationFeatureGenerator`. A descriptor naming either is an
//! error ([`Error::Unsupported`](crate::Error::Unsupported)).
//!
//! # State
//!
//! Java keeps a generator's per-sentence state in thread-locals and
//! notices a new sentence when it is handed a different `String[]`
//! object. Here a generator is owned by one tagger at a time (`&mut self`)
//! and is told about a new sentence by
//! [`start_sequence`](FeatureGenerator::start_sequence). Code that calls
//! the generators directly calls it once per token array, before the first
//! [`create_features`](FeatureGenerator::create_features) for that array;
//! the beam search does it for its context generator.

mod cached;
mod cluster;
mod context;
pub mod descriptor;
mod dictionary;
mod history;
mod token;
mod token_pattern;
mod window;

pub use cached::Cached;
pub use cluster::{BrownBigram, BrownCluster, BrownToken, BrownTokenClass, WordCluster, WordClusterDictionary};
pub use context::{
    AdditionalContext, BigramName, DocumentBegin, OutcomePrior, PreviousTags, SentencePosition, TrigramName,
};
pub use dictionary::InDictionary;
pub use history::{PreviousMap, PreviousTwoMap};
pub use token::{CharacterNgram, Prefix, Suffix, TokenClass, TokenGenerator};
pub use token_pattern::TokenPattern;
pub use window::Window;

use crate::features::Features;

/// `AdaptiveFeatureGenerator`: writes the features of the token at
/// `index`.
pub trait FeatureGenerator: Send {
    /// Appends the features of `tokens[index]` to `out`. `previous` holds
    /// the outcomes decided so far (the tags of the tokens before
    /// `index`, during a beam search); generators that read it index it
    /// below `index`.
    fn create_features(&mut self, out: &mut Features, tokens: &[&str], index: usize, previous: &[&str]);

    /// Learns from a finished sentence and its outcomes, for the
    /// generators that remember earlier sentences of a document.
    fn update_adaptive_data(&mut self, _tokens: &[&str], _outcomes: &[&str]) {}

    /// Forgets what [`update_adaptive_data`](Self::update_adaptive_data)
    /// learned: the start of a new document.
    fn clear_adaptive_data(&mut self) {}

    /// A new token array follows. Java notices this by the identity of the
    /// `String[]` it is given; see the module documentation.
    fn start_sequence(&mut self) {}

    /// Sets the per-token additional context that [`AdditionalContext`]
    /// writes (`NameFinderME.find(tokens, additionalContext)`). Generators
    /// that hold others pass it on.
    fn set_additional_context(&mut self, _context: &[Vec<String>]) {}
}

/// A generator built at run time, as the descriptor makes them.
pub type BoxedGenerator = Box<dyn FeatureGenerator>;

/// `AggregatedFeatureGenerator`: the features of each generator in turn.
pub struct Aggregated {
    generators: Vec<BoxedGenerator>,
}

impl Aggregated {
    pub fn new(generators: Vec<BoxedGenerator>) -> Aggregated {
        Aggregated { generators }
    }

    pub fn generators(&self) -> &[BoxedGenerator] {
        &self.generators
    }

    /// Adds a generator after the others.
    pub fn push(&mut self, generator: BoxedGenerator) {
        self.generators.push(generator);
    }
}

impl FeatureGenerator for Aggregated {
    fn create_features(&mut self, out: &mut Features, tokens: &[&str], index: usize, previous: &[&str]) {
        for g in &mut self.generators {
            g.create_features(out, tokens, index, previous);
        }
    }

    fn update_adaptive_data(&mut self, tokens: &[&str], outcomes: &[&str]) {
        for g in &mut self.generators {
            g.update_adaptive_data(tokens, outcomes);
        }
    }

    fn clear_adaptive_data(&mut self) {
        for g in &mut self.generators {
            g.clear_adaptive_data();
        }
    }

    fn start_sequence(&mut self) {
        for g in &mut self.generators {
            g.start_sequence();
        }
    }

    fn set_additional_context(&mut self, context: &[Vec<String>]) {
        for g in &mut self.generators {
            g.set_additional_context(context);
        }
    }
}
