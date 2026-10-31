//! The shared encoder pass behind the word-level head steps
//! (`steps/heads.rs`): [`HeadGroup`] holds one request's heads and
//! encoder and caches the read of the document, [`GroupStep`] decodes one
//! layer from it, and [`SpanStep`] and [`RelationStep`] run the heads
//! that read on their own under their step ids.

use std::marker::PhantomData;
use std::path::Path;
use std::sync::{Arc, Mutex, PoisonError};

use turbo::bundle::Bundle;
use turbo::manifest::WordPooling;
use turbo::status::Error as CoreError;
use turbo_analysis::{Annotator, Edges, Error, Execution, Fnv1a, ModelRef, Refs, View};
use turbo_analysis_embed::EncoderOptions;
use turbo_analysis_layers::{Dependencies, Entities, Lemmas, MorphFeatures, PosTags, Sentences, Subwords, Tokens};
use turbo_relation::Relations;
use turbo_tagger::annotate::{HeadRunner, find_head};
use turbo_tagger::multi::{Head, HeadLayer, Input};
use turbo_tagger::{RelationTagger, SpanTagger, WordEncoder};

use super::HeadStep;
use crate::api::TURBO_E_INVALID_ARGUMENT;

/// What one read of a document's words gave, and what it was read from.
struct Cached {
    key: u64,
    seqs: Vec<Vec<usize>>,
    features: Arc<Vec<Vec<Vec<f32>>>>,
}

/// The word-level heads of one request and the encoder pass they share.
pub struct HeadGroup {
    words: Mutex<WordEncoder>,
    heads: Vec<(HeadStep, Head)>,
    poolings: Vec<WordPooling>,
    cache: Mutex<Option<Cached>>,
    model: ModelRef,
    execution: Execution,
    width: usize,
}

impl HeadGroup {
    /// The heads of the bundle at `dir` for `steps`, each a shared step
    /// the bundle provides, and the bundle's encoder.
    pub fn open(dir: &Path, steps: &[HeadStep], o: &EncoderOptions) -> Result<Arc<HeadGroup>, CoreError> {
        let bundle = Bundle::open(dir)?;
        let refuse = |m: String| CoreError::new(TURBO_E_INVALID_ARGUMENT, m);
        let mut heads = Vec::with_capacity(steps.len());
        for &s in steps {
            if !s.shared() {
                return Err(refuse(format!("{} reads on its own", s.name())));
            }
            let spec = s
                .head(&bundle)
                .ok_or_else(|| refuse(format!("{}: no one {:?} head writes {}", s.name(), s.kind(), s.layer())))?;
            // By name: another kind of head may write the same layer.
            let spec = find_head(&bundle, Some(&spec.name), s.layer(), s.kind())?;
            heads.push((s, s.open(&bundle, spec)?));
        }
        let words = WordEncoder::open(dir, o)?;
        for (_, h) in &heads {
            if let Some(&n) = h.hidden().iter().find(|&&n| n != words.hidden()) {
                return Err(refuse(format!(
                    "head {:?} reads {n} values a word, the encoder gives {}",
                    h.spec().name,
                    words.hidden()
                )));
            }
        }
        let mut poolings: Vec<WordPooling> = Vec::new();
        for p in heads.iter().flat_map(|(_, h)| h.poolings()) {
            if !poolings.contains(&p) {
                poolings.push(p);
            }
        }
        let m = &bundle.manifest.model;
        Ok(Arc::new(HeadGroup {
            model: ModelRef { id: m.id.clone(), revision: m.revision.clone(), sha256: bundle.manifest_sha256.clone() },
            execution: words.execution().clone(),
            width: words.width(),
            words: Mutex::new(words),
            heads,
            poolings,
            cache: Mutex::new(None),
        }))
    }

    /// Rows the encoder has read: one a window, however many steps
    /// decoded from it.
    pub fn rows(&self) -> u64 {
        self.words.lock().unwrap_or_else(PoisonError::into_inner).rows()
    }

    /// The steps' heads, in the order opened.
    pub fn steps(&self) -> Vec<HeadStep> {
        self.heads.iter().map(|(s, _)| *s).collect()
    }

    fn head(&self, step: HeadStep) -> Result<&Head, String> {
        self.heads
            .iter()
            .find(|(s, _)| *s == step)
            .map(|(_, h)| h)
            .ok_or_else(|| format!("{} was not opened with this group", step.name()))
    }

    /// The document's words as every head reads them, from the cached
    /// read when the document is the one read before.
    fn input<'a>(
        &self,
        view: &'a View,
        tokens: &'a Tokens,
        sentences: Option<&Sentences>,
        subwords: Option<&Subwords>,
    ) -> Result<Input<'a>, String> {
        let seqs = HeadRunner::sequences(view, tokens, sentences)?;
        let (starts, ends) = tokens.spans.columns();
        let mut h = Fnv1a::new();
        h.str(view.text()).u64(view.id().0 as u64);
        for &x in starts.iter().chain(ends) {
            h.u64(x as u64);
        }
        let key = h.finish();
        let mut cache = self.cache.lock().unwrap_or_else(PoisonError::into_inner);
        let features = match &*cache {
            Some(c) if c.key == key && c.seqs == seqs => Arc::clone(&c.features),
            _ => {
                let mut words = self.words.lock().unwrap_or_else(PoisonError::into_inner);
                let f = Arc::new(words.token_features_pooled(view, tokens, &seqs, subwords, &self.poolings)?);
                *cache = Some(Cached { key, seqs: seqs.clone(), features: Arc::clone(&f) });
                f
            }
        };
        Ok(Input::new(view, tokens, seqs, self.poolings.clone(), features))
    }

    /// Layer `L` of the step from the shared read of the document.
    fn write<L: HeadLayer>(
        &self,
        step: HeadStep,
        view: &View,
        tokens: &Tokens,
        sentences: Option<&Sentences>,
        subwords: Option<&Subwords>,
    ) -> Result<L, String> {
        let head = self.head(step)?;
        let input = self.input(view, tokens, sentences, subwords)?;
        L::write(head, &input)
    }
}

/// A layer a shared head step writes.
pub trait GroupLayer: HeadLayer {
    const STEP: HeadStep;
}

impl GroupLayer for PosTags {
    const STEP: HeadStep = HeadStep::Pos;
}

impl GroupLayer for MorphFeatures<Tokens> {
    const STEP: HeadStep = HeadStep::Morphology;
}

impl GroupLayer for Entities {
    const STEP: HeadStep = HeadStep::Entities;
}

impl GroupLayer for Dependencies<Tokens> {
    const STEP: HeadStep = HeadStep::Dependencies;
}

impl GroupLayer for Lemmas {
    const STEP: HeadStep = HeadStep::Lemmas;
}

/// A shared head step: layer `L` from its group's one encoder pass.
pub struct GroupStep<L: GroupLayer> {
    group: Arc<HeadGroup>,
    _layer: PhantomData<fn() -> L>,
}

impl<L: GroupLayer> GroupStep<L> {
    pub fn new(group: Arc<HeadGroup>) -> GroupStep<L> {
        GroupStep { group, _layer: PhantomData }
    }
}

impl<L: GroupLayer> Annotator for GroupStep<L> {
    const ID: &'static str = L::STEP.name();
    type Reads = (View, Tokens, Option<Subwords>, Option<Sentences>);
    type Writes = (L,);

    fn annotate(&self, (view, tokens, subwords, sentences): Refs<'_, Self>) -> Result<Self::Writes, Error> {
        self.group
            .write::<L>(L::STEP, view, tokens, sentences, subwords)
            .map(|l| (l,))
            .map_err(|detail| Error::Annotator { annotator: Self::ID, detail })
    }

    fn model(&self) -> Option<ModelRef> {
        Some(self.group.model.clone())
    }

    fn execution(&self) -> Option<Execution> {
        Some(self.group.execution.clone())
    }

    fn config_hash(&self) -> u64 {
        let name = self.group.head(L::STEP).map_or("", |h| h.spec().name.as_str());
        Fnv1a::new().str(name).u64(self.group.width as u64).finish()
    }
}

fn renamed(id: &'static str, e: Error) -> Error {
    match e {
        Error::Annotator { detail, .. } => Error::Annotator { annotator: id, detail },
        e => e,
    }
}

/// The span classifier head's entities, under the step's id.
pub struct SpanStep(pub SpanTagger);

impl Annotator for SpanStep {
    const ID: &'static str = HeadStep::Spans.name();
    type Reads = <SpanTagger as Annotator>::Reads;
    type Writes = (Entities,);

    fn annotate(&self, input: Refs<'_, Self>) -> Result<Self::Writes, Error> {
        self.0.annotate(input).map_err(|e| renamed(Self::ID, e))
    }

    fn model(&self) -> Option<ModelRef> {
        self.0.model()
    }

    fn execution(&self) -> Option<Execution> {
        self.0.execution()
    }

    fn config_hash(&self) -> u64 {
        self.0.config_hash()
    }
}

/// The relation classifier head's relations, under the step's id.
pub struct RelationStep(pub RelationTagger);

impl Annotator for RelationStep {
    const ID: &'static str = HeadStep::Relations.name();
    type Reads = <RelationTagger as Annotator>::Reads;
    type Writes = (Relations, Edges<Entities, Entities>);

    fn annotate(&self, input: Refs<'_, Self>) -> Result<Self::Writes, Error> {
        self.0.annotate(input).map_err(|e| renamed(Self::ID, e))
    }

    fn model(&self) -> Option<ModelRef> {
        self.0.model()
    }

    fn execution(&self) -> Option<Execution> {
        self.0.execution()
    }

    fn config_hash(&self) -> u64 {
        self.0.config_hash()
    }
}
