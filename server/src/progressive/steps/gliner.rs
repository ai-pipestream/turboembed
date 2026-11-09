use std::any::Any;
use std::sync::Arc;

use turbo_analysis::annotate::{AnchorUse, Step};
use turbo_analysis::{Annotator, Document, Error, Evidence, Execution, Key, KeyId, ModelRef, View};
use turbo_analysis_layers::{Entities, Sentences, Tokens};
use turbo_gliner::{Decisions, ExtractionSchema, GlinerAnnotator, Model};

use crate::api::{Failure, TURBO_E_BUNDLE_NOT_FOUND, TURBO_E_INVALID_ARGUMENT};
use crate::state::State;

pub(super) fn open(state: Option<&State>, name: &str) -> Result<Arc<Model>, Failure> {
    let name = name.trim();
    if name.is_empty() {
        return Err(Failure::new(
            TURBO_E_INVALID_ARGUMENT,
            0,
            "turbo:gliner requires model to name a converted GLiNER package",
        ));
    }
    let state = state.ok_or_else(|| {
        Failure::new(TURBO_E_BUNDLE_NOT_FOUND, 0, "turbo:gliner requires a loaded converted GLiNER package")
    })?;
    let (served, _) = state.peek(name)?;
    served.gliner.get().cloned().ok_or_else(|| {
        Failure::new(TURBO_E_BUNDLE_NOT_FOUND, 0, format!("model {name} is not a converted GLiNER package"))
    })
}

pub(super) struct GlinerStep {
    annotator: GlinerAnnotator,
}

impl GlinerStep {
    pub(super) fn new(model: Arc<Model>, schema: ExtractionSchema) -> Self {
        Self { annotator: GlinerAnnotator::new(model, schema) }
    }

    fn keys() -> [KeyId; 3] {
        [Key::<Entities>::DEFAULT.id(), Key::<Decisions>::DEFAULT.id(), Key::<Evidence<Entities>>::DEFAULT.id()]
    }
}

impl Step for GlinerStep {
    fn id(&self) -> &'static str {
        GlinerAnnotator::ID
    }

    fn version(&self) -> &'static str {
        GlinerAnnotator::VERSION
    }

    fn reads(&self) -> Vec<KeyId> {
        vec![Key::<View>::DEFAULT.id(), Key::<Tokens>::DEFAULT.id(), Key::<Sentences>::DEFAULT.id()]
    }

    fn required(&self) -> Vec<bool> {
        vec![true, true, false]
    }

    fn writes(&self) -> Vec<KeyId> {
        Self::keys().to_vec()
    }

    fn anchors(&self) -> Vec<AnchorUse> {
        let entities = Key::<Entities>::DEFAULT.id();
        vec![AnchorUse::None, AnchorUse::To(entities), AnchorUse::To(entities)]
    }

    fn model(&self) -> Option<ModelRef> {
        self.annotator.model()
    }

    fn execution(&self) -> Option<Execution> {
        self.annotator.execution()
    }

    fn config_hash(&self) -> u64 {
        self.annotator.config_hash()
    }

    fn compute(&self, docs: &[Document]) -> Box<dyn Any + Send> {
        let out: Vec<Option<Result<(Entities, Decisions), Error>>> = docs
            .iter()
            .map(|doc| {
                let result = (|| {
                    let view = doc.require(&Key::<View>::DEFAULT)?;
                    let tokens = doc.require(&Key::<Tokens>::DEFAULT)?;
                    let sentences = doc.get::<Sentences>();
                    self.annotator.annotate((view, tokens, sentences))
                })();
                Some(result)
            })
            .collect();
        Box::new(out)
    }

    fn store(&self, out: Box<dyn Any + Send>, docs: &mut [Document], status: &mut [Result<(), Error>]) {
        let out = *out.downcast::<Vec<Option<Result<(Entities, Decisions), Error>>>>().expect("step output");
        for (i, result) in out.into_iter().enumerate() {
            let Some(result) = result else { continue };
            let doc = &mut docs[i];
            let provenance = self.expected_provenance(doc);
            match result {
                Ok((entities, decisions)) => {
                    let stored = (|| {
                        doc.add(&Key::<Entities>::DEFAULT, entities, provenance.clone())?;
                        doc.add(&Key::<Decisions>::DEFAULT, decisions, provenance.clone())?;
                        let evidence = GlinerAnnotator::evidence(doc, Key::DEFAULT, Key::DEFAULT)?;
                        doc.add(&Key::<Evidence<Entities>>::DEFAULT, evidence, provenance.clone())
                    })();
                    if let Err(error) = stored
                        && status[i].is_ok()
                    {
                        status[i] = Err(error);
                    }
                }
                Err(Error::Skipped(why)) => {
                    for key in Self::keys() {
                        let _ = doc.skip(key, why.clone(), provenance.clone());
                    }
                }
                Err(error) => {
                    for key in Self::keys() {
                        if !doc.has(key) {
                            let _ = doc.fail(key, error.clone(), provenance.clone());
                        }
                    }
                    if status[i].is_ok() {
                        status[i] = Err(error);
                    }
                }
            }
        }
    }
}
