use crate::{
    ArcStandardState, DependencyContextGenerator, DependencyEventStream, DependencyGraph, DependencySample,
    ParserInput, Result, Transition, error,
};
use std::collections::HashSet;
use std::sync::Arc;
use turbo_analysis::ModelRef;
use turbo_analysis_layers::WordOrTokenAnchor;
use turbo_maxent::{MaxentModel, ModelFile};
use turbo_train::{Manifest, ModelPackage, TrainingConfiguration, TrainingParameters};

pub const COMPONENT_NAME: &str = "DependencyParserME";
pub const MODEL_ENTRY: &str = "depparse.model";
/// Validated outcome inventory, classifier, and content-addressed provenance.
pub struct DependencyModel {
    pub(crate) classifier: MaxentModel,
    pub(crate) transitions: Vec<Transition>,
    pub(crate) identity: ModelRef,
    bytes: Option<Vec<u8>>,
    label_encoding: crate::LabelEncoding,
}
impl DependencyModel {
    /// The caller supplies the identity of an externally constructed classifier.
    pub fn from_maxent(classifier: MaxentModel, identity: ModelRef) -> Result<Self> {
        let mut transitions = Vec::new();
        let mut seen = HashSet::new();
        for i in 0..classifier.num_outcomes() {
            let s = classifier.outcome_str(i).ok_or_else(|| error("non-Unicode transition outcome"))?;
            let t = Transition::decode(s)?;
            if !seen.insert(t.clone()) {
                return Err(error("duplicate transition outcome"));
            }
            transitions.push(t);
        }
        if !transitions.contains(&Transition::Shift)
            || !transitions.iter().any(|t| matches!(t, Transition::RightArc(_)))
        {
            return Err(error("model requires SHIFT and RIGHT_ARC outcomes"));
        }
        Ok(Self { classifier, transitions, identity, bytes: None, label_encoding: crate::LabelEncoding::Plain })
    }
    pub fn from_raw(bytes: &[u8]) -> Result<Self> {
        Self::from_maxent(
            MaxentModel::read(bytes).map_err(error)?,
            ModelRef {
                id: "opennlp:DependencyParserME:raw".into(),
                revision: "1".into(),
                sha256: turbo_maxent::sha256(bytes).to_hex(),
            },
        )
    }
    pub fn open(bytes: Vec<u8>) -> Result<Self> {
        let file = ModelFile::open(bytes.clone()).map_err(error)?;
        if file.component_name() != COMPONENT_NAME {
            return Err(error("wrong dependency model component"));
        }
        let mut out = Self::from_raw(&file.entry(MODEL_ENTRY).map_err(error)?)?;
        out.identity = ModelRef {
            id: format!("opennlp:DependencyParserME:{}", file.language()),
            revision: file.opennlp_version().into(),
            sha256: file.sha256().to_hex(),
        };
        out.label_encoding = file
            .manifest()
            .get("Turbo-Dependency-Label-Encoding")
            .map(crate::LabelEncoding::read)
            .transpose()?
            .unwrap_or_else(|| crate::LabelEncoding::historical(&bytes));
        out.bytes = Some(bytes);
        Ok(out)
    }
    pub fn label_encoding(&self) -> crate::LabelEncoding {
        self.label_encoding
    }
    pub fn bytes(&self) -> Option<&[u8]> {
        self.bytes.as_deref()
    }
    pub fn model_ref(&self) -> &ModelRef {
        &self.identity
    }
    pub fn train<A: WordOrTokenAnchor>(
        language: &str,
        samples: &[DependencySample<A>],
        params: &TrainingParameters,
    ) -> Result<Self> {
        let stream = DependencyEventStream::generate(samples)?;
        let (model, report) =
            turbo_train::trainer::train_events(&stream.events, params, &mut TrainingConfiguration::default())
                .map_err(error)?;
        let mut manifest = Manifest::new(COMPONENT_NAME, language, 0);
        manifest.set_training(&report);
        manifest.set("Turbo-Dependency-Label-Encoding", crate::LabelEncoding::EscapedV1.name());
        let mut package = ModelPackage::new(manifest);
        package.add_entry(MODEL_ENTRY, model.write().map_err(error)?);
        Self::open(package.to_bytes())
    }
}
/// A parser can be substituted in evaluation without erasing its anchor type.
pub trait DependencyParser<A: WordOrTokenAnchor> {
    fn parse(&self, tokens: &[&str], tags: &[&str]) -> Result<DependencyGraph<A>>;
}
#[derive(Clone)]
pub struct DependencyParserME {
    pub(crate) model: Arc<DependencyModel>,
}
impl DependencyParserME {
    pub fn new(model: DependencyModel) -> Self {
        Self { model: Arc::new(model) }
    }
    pub fn parse<A: WordOrTokenAnchor>(&self, tokens: &[&str], tags: &[&str]) -> Result<DependencyGraph<A>> {
        ParserInput::check(tokens, tags)?;
        let mut state = ArcStandardState::<A>::new(tokens.len())?;
        let mut scores = vec![0.0; self.model.transitions.len()];
        while !state.is_terminal() {
            let features = DependencyContextGenerator::context(&state, tokens, tags)?;
            self.model.classifier.eval(&features.iter().map(String::as_str).collect::<Vec<_>>(), &mut scores);
            if scores.iter().any(|p| !p.is_finite()) {
                return Err(error("non-finite model score"));
            }
            let mut best = None;
            let mut score = f64::NEG_INFINITY;
            for (i, t) in self.model.transitions.iter().enumerate() {
                if scores[i] > score && state.can_apply(t) {
                    best = Some(t);
                    score = scores[i];
                }
            }
            state.apply(best.ok_or_else(|| error("no applicable model outcome"))?)?;
        }
        state.to_graph()?.deprojectivize_with(self.model.label_encoding)
    }
}
impl<A: WordOrTokenAnchor> DependencyParser<A> for DependencyParserME {
    fn parse(&self, tokens: &[&str], tags: &[&str]) -> Result<DependencyGraph<A>> {
        self.parse(tokens, tags)
    }
}
