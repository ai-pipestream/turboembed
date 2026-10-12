//! The name finder's context generator and codec choice
//! (`opennlp.tools.namefind`).

use crate::beam::{ContextGenerator, SequenceValidator};
use crate::codec::{BilouCodec, BilouValidator, BioCodec, BioValidator, SequenceCodec, Span};
use crate::error::Result;
use crate::featuregen::{AdditionalContext, BoxedGenerator, Window, descriptor};
use crate::features::Features;
use crate::pattern::{PatternMode, token_feature};
use turbo_maxent::ModelFile;

/// The model entry of the classifier.
pub const MODEL_ENTRY_NAME: &str = "nameFinder.model";
/// `NameFinderME.DEFAULT_BEAM_SIZE`.
pub const DEFAULT_BEAM_SIZE: usize = 3;
/// The manifest property naming the sequence codec class.
pub const CODEC_PROPERTY: &str = "sequenceCodecImplName";

/// OpenNLP's default name finder descriptor
/// (`opennlp/tools/namefind/ner-default-features.xml`), which
/// `TokenNameFinderFactory` uses for every model that carries none, the
/// OpenNLP 1.5 models included.
pub const DEFAULT_DESCRIPTOR: &str = include_str!("descriptors/ner-default-features.xml");

/// `DefaultNameContextGenerator`: the features of each generator, in
/// order, then the previous outcome (`po=`), the previous outcome with the
/// token (`pow=`) and with the token's class (`powf=`), and the outcome
/// before that (`ppo=`). A missing outcome is `other`.
pub struct DefaultNameContext {
    generators: Vec<BoxedGenerator>,
    mode: PatternMode,
}

impl DefaultNameContext {
    pub fn new(generators: Vec<BoxedGenerator>, mode: PatternMode) -> DefaultNameContext {
        DefaultNameContext { generators, mode }
    }

    /// The context generator `NameFinderME` builds for a model: the
    /// generators of the model's descriptor (or of the default one), and
    /// after them an [`AdditionalContext`] in a window of eight tokens on
    /// each side.
    pub fn for_model(model: &ModelFile, mode: PatternMode) -> Result<DefaultNameContext> {
        let generator = match descriptor::from_model(model, mode)? {
            Some(g) => g,
            None => descriptor::build(DEFAULT_DESCRIPTOR.as_bytes(), Some(model), mode)?,
        };
        let additional = Box::new(Window::new(Box::new(AdditionalContext::new()), 8, 8));
        Ok(DefaultNameContext::new(vec![generator, additional], mode))
    }

    /// `addFeatureGenerator`: adds a generator after the others.
    pub fn add_feature_generator(&mut self, generator: BoxedGenerator) {
        self.generators.push(generator);
    }

    /// Passes a finished sentence's outcomes to every generator, as
    /// `NameFinderME.find` does after each sentence.
    ///
    /// # Panics
    ///
    /// When the lengths differ, as Java throws.
    pub fn update_adaptive_data(&mut self, tokens: &[&str], outcomes: &[&str]) {
        assert_eq!(tokens.len(), outcomes.len(), "The tokens and outcome arrays MUST have the same size!");
        for g in &mut self.generators {
            g.update_adaptive_data(tokens, outcomes);
        }
    }

    /// Forgets the document so far (`clearAdaptiveData`).
    pub fn clear_adaptive_data(&mut self) {
        for g in &mut self.generators {
            g.clear_adaptive_data();
        }
    }

    /// Sets the additional context of the next sentence
    /// (`NameFinderME.find(tokens, additionalContext)`); an empty slice
    /// for none.
    pub fn set_additional_context(&mut self, context: &[Vec<String>]) {
        for g in &mut self.generators {
            g.set_additional_context(context);
        }
    }
}

impl<'a> ContextGenerator<&'a str> for DefaultNameContext {
    fn start_sequence(&mut self) {
        for g in &mut self.generators {
            g.start_sequence();
        }
    }

    fn context(&mut self, index: usize, sequence: &[&'a str], prior: &[&str], out: &mut Features) {
        for g in &mut self.generators {
            g.create_features(out, sequence, index, prior);
        }
        let po = if index > 0 { prior[index - 1] } else { BioCodec::OTHER };
        let ppo = if index > 1 { prior[index - 2] } else { BioCodec::OTHER };
        let token = sequence[index];
        out.push_str("po=").push_str(po).finish();
        out.push_str("pow=").push_str(po).push_str(",").push_str(token).finish();
        out.push_str("powf=").push_str(po).push_str(",").push_str(token_feature(token, self.mode)).finish();
        out.push_str("ppo=").push_str(ppo).finish();
    }
}

/// The codec a name finder model uses.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum NameCodec {
    Bio,
    Bilou,
}

impl NameCodec {
    /// `TokenNameFinderFactory.createSequenceCodec`: the class the
    /// manifest names, BIO when it names none or a class that cannot be
    /// made.
    pub fn for_model(model: &ModelFile) -> NameCodec {
        match model.manifest().get(CODEC_PROPERTY) {
            Some("opennlp.tools.namefind.BilouCodec") => NameCodec::Bilou,
            _ => NameCodec::Bio,
        }
    }

    pub fn decode(self, outcomes: &[&str]) -> Vec<Span> {
        match self {
            NameCodec::Bio => BioCodec.decode(outcomes),
            NameCodec::Bilou => BilouCodec.decode(outcomes),
        }
    }

    pub fn encode(self, spans: &[Span], length: usize) -> Vec<String> {
        match self {
            NameCodec::Bio => BioCodec.encode(spans, length),
            NameCodec::Bilou => BilouCodec.encode(spans, length),
        }
    }
}

impl<T> SequenceValidator<T> for NameCodec {
    fn valid_sequence(&self, index: usize, sequence: &[T], prior: &[&str], outcome: &str) -> bool {
        match self {
            NameCodec::Bio => BioValidator.valid_sequence(index, sequence, prior, outcome),
            NameCodec::Bilou => BilouValidator.valid_sequence(index, sequence, prior, outcome),
        }
    }
}
