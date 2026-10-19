//! The name finder's context generator and codec choice
//! (`opennlp.tools.namefind`).

use crate::beam::{BEAM_SIZE_PROPERTY, BeamSearch, ContextGenerator, SequenceValidator, manifest_beam_size};
use crate::codec::{BilouCodec, BilouValidator, BioCodec, BioValidator, SequenceCodec, Span};
use crate::error::{Error, Result};
use crate::featuregen::{AdditionalContext, BoxedGenerator, Window, descriptor};
use crate::features::Features;
use crate::pattern::{PatternMode, token_feature};
use turbo_maxent::{MaxentModel, ModelFile};

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
            Some(class) if class == NameCodec::Bilou.class_name() => NameCodec::Bilou,
            _ => NameCodec::Bio,
        }
    }

    pub fn decode(self, outcomes: &[&str]) -> Vec<Span> {
        match self {
            NameCodec::Bio => BioCodec.decode(outcomes),
            NameCodec::Bilou => BilouCodec.decode(outcomes),
        }
    }

    /// `encode`: the outcomes of `length` tokens holding the spans.
    ///
    /// # Panics
    ///
    /// Where Java throws `ArrayIndexOutOfBoundsException`: with BIO, an
    /// empty span after the last token; with BILOU, an empty span at the
    /// first token, whose unit would go at index -1.
    pub fn encode(self, spans: &[Span], length: usize) -> Vec<String> {
        match self {
            NameCodec::Bio => BioCodec.encode(spans, length),
            NameCodec::Bilou => BilouCodec.encode(spans, length),
        }
    }

    /// `areOutcomesCompatible`: whether a model with these outcomes can
    /// be decoded with the codec, as `TokenNameFinderModel` checks a
    /// trained model.
    pub fn are_outcomes_compatible(self, outcomes: &[&str]) -> bool {
        match self {
            NameCodec::Bio => BioCodec.are_outcomes_compatible(outcomes),
            NameCodec::Bilou => BilouCodec.are_outcomes_compatible(outcomes),
        }
    }

    /// The codec's class, which a trained model's manifest names under
    /// [`CODEC_PROPERTY`].
    pub fn class_name(self) -> &'static str {
        match self {
            NameCodec::Bio => "opennlp.tools.namefind.BioCodec",
            NameCodec::Bilou => "opennlp.tools.namefind.BilouCodec",
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

/// A name `NameFinderME.find` returned: its span, typed as the model's
/// outcomes name it, and the arithmetic mean of the probabilities of the
/// outcomes it spans (`NameFinderME.probs(Span[])`), which is `NaN` for
/// an empty span (0 / 0 in Java).
#[derive(Debug, Clone, PartialEq)]
pub struct FoundName {
    pub span: Span,
    pub prob: f64,
}

/// `NameFinderME` over a model file: the beam search of the model's beam
/// size, the context generator `NameFinderME` builds for the model, and
/// the model's codec with its sequence validator. Sentences are given in
/// document order; the context generator's adaptive data carries over
/// from one to the next until [`NameFinder::clear_adaptive_data`].
pub struct NameFinder<'m> {
    search: BeamSearch<'m>,
    context: DefaultNameContext,
    codec: NameCodec,
    probs: Vec<f64>,
}

impl<'m> NameFinder<'m> {
    /// `new NameFinderME(new TokenNameFinderModel(file))`, with `model` the
    /// classifier of the file's `nameFinder.model` entry.
    pub fn new(file: &ModelFile, model: &'m MaxentModel, mode: PatternMode) -> Result<NameFinder<'m>> {
        let beam = manifest_beam_size(file, DEFAULT_BEAM_SIZE)?;
        if beam == 0 {
            return Err(Error::Manifest { property: BEAM_SIZE_PROPERTY.into(), value: "0".into() });
        }
        Ok(NameFinder {
            search: BeamSearch::new(beam, model)?,
            context: DefaultNameContext::for_model(file, mode)?,
            codec: NameCodec::for_model(file),
            probs: Vec::new(),
        })
    }

    pub fn codec(&self) -> NameCodec {
        self.codec
    }

    /// `find(tokens)`: the names of one sentence, with no additional
    /// context. The best outcome sequence is passed to the generators'
    /// adaptive data and decoded; when the search finds no sequence
    /// scoring above its minimum, Java returns no names and leaves the
    /// adaptive data as it was, and so does this.
    pub fn find(&mut self, tokens: &[&str]) -> Vec<FoundName> {
        self.find_with_context(tokens, &[])
    }

    /// `find(tokens, additionalContext)`: the names of one sentence with
    /// caller-supplied per-token additional context features (`ne=<val>`).
    pub fn find_with_context(&mut self, tokens: &[&str], additional_context: &[Vec<String>]) -> Vec<FoundName> {
        self.context.set_additional_context(additional_context);
        let Some(best) = self.search.best_sequence(tokens, &mut self.context, &self.codec) else {
            return Vec::new();
        };
        self.context.update_adaptive_data(tokens, best.outcomes());
        self.probs.clear();
        self.probs.extend_from_slice(best.probs());
        let probs = best.probs();
        self.codec
            .decode(best.outcomes())
            .into_iter()
            .map(|span| {
                let mut p = 0.0;
                for prob in &probs[span.start..span.end] {
                    p += prob;
                }
                let prob = p / span.len() as f64;
                FoundName { span, prob }
            })
            .collect()
    }

    /// `probs()`: the probability of each outcome of the best sequence of
    /// the last [`find`](NameFinder::find) that found one.
    pub fn probs(&self) -> &[f64] {
        &self.probs
    }

    /// `clearAdaptiveData`: the next sentence starts a document.
    pub fn clear_adaptive_data(&mut self) {
        self.context.clear_adaptive_data();
    }
}

#[cfg(test)]
mod tests {
    use turbo_train::{Event, Key, Manifest, ModelPackage, TrainingConfiguration, TrainingParameters};

    use super::*;

    fn test_model_and_file(beam_size: Option<&str>) -> (ModelFile, MaxentModel) {
        let events = [
            Event::from_strs("person-start", &["ne=hint", "w=Alice"]),
            Event::from_strs("person-start", &["ne=hint", "w=Bob"]),
            Event::from_strs("other", &["w=Alice"]),
            Event::from_strs("other", &["w=Bob"]),
            Event::from_strs("other", &["w= walked"]),
        ];
        let mut params = TrainingParameters::new();
        params.put(Key::ITERATIONS, 50);
        params.put(Key::CUTOFF, 0);
        let (trained, _) =
            turbo_train::gis::train_events(&events, &params, &mut TrainingConfiguration::default()).unwrap();
        let model_bytes = turbo_train::write_gis(&trained).unwrap();
        let maxent = MaxentModel::read(&model_bytes).unwrap();
        let mut manifest = Manifest::new("TokenNameFinder", "en", 1_700_000_000_000);
        if let Some(bs) = beam_size {
            manifest.set(BEAM_SIZE_PROPERTY, bs);
        }
        let mut pkg = ModelPackage::new(manifest);
        pkg.add_entry(MODEL_ENTRY_NAME, model_bytes);
        let file = ModelFile::open(pkg.to_bytes()).unwrap();
        (file, maxent)
    }

    #[test]
    fn each_codec_names_its_class_and_checks_outcomes_as_its_codec_does() {
        assert_eq!(NameCodec::Bio.class_name(), "opennlp.tools.namefind.BioCodec");
        assert_eq!(NameCodec::Bilou.class_name(), "opennlp.tools.namefind.BilouCodec");
        let bio = ["other", "person-start", "person-cont"];
        assert!(NameCodec::Bio.are_outcomes_compatible(&bio));
        assert!(!NameCodec::Bilou.are_outcomes_compatible(&bio));
        // No start outcome: a model trained on samples without names.
        assert!(!NameCodec::Bio.are_outcomes_compatible(&["other"]));
        assert!(NameCodec::Bilou.are_outcomes_compatible(&["other", "x-unit"]));
        assert_eq!(NameCodec::Bio.encode(&[Span::new(0, 1, Some("x"))], 2), ["x-start", "other"]);
    }

    #[test]
    fn name_finder_rejects_zero_beam_size_and_propagates_additional_context() {
        let (zero_file, maxent) = test_model_and_file(Some("0"));
        let err = NameFinder::new(&zero_file, &maxent, PatternMode::Utf16Units).err().unwrap();
        assert!(
            matches!(&err, Error::Manifest { property, value } if &**property == BEAM_SIZE_PROPERTY && value == "0"),
            "{err:?}"
        );

        let (file, maxent) = test_model_and_file(None);
        let mut finder = NameFinder::new(&file, &maxent, PatternMode::Utf16Units).unwrap();
        assert!(finder.find(&["Alice"]).is_empty());
        let found = finder.find_with_context(&["Alice"], &[vec!["hint".into()]]);
        assert_eq!(found.len(), 1);
        assert_eq!(found[0].span, Span::new(0, 1, Some("person")));
    }
}
