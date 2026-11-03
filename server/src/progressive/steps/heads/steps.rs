//! The head steps as annotators, each decoding its layer from the
//! request's [`HeadGroup`] read: [`GroupStep`] for the layers a tagger
//! crate head writes from shared features (part of speech, morphology,
//! BIO entities, dependencies, lemmas, syntactic words), and
//! [`SentenceStep`], [`SpanStep`], [`RelationStep`] and [`CorefStep`]
//! through the tagger crate's own decoders. Each layer is the one that
//! crate's annotator writes from the same features; only the step id and,
//! where docs/grpc-analysis.md says so, the output instance differ.

use std::marker::PhantomData;
use std::sync::Arc;

use turbo_analysis::analyzer::Builder;
use turbo_analysis::{Annotator, Bound, Edges, Error, Execution, Fnv1a, Groups, ModelRef, Refs, View};
use turbo_analysis_layers::{
    Dependencies, Entities, Lemmas, MorphFeatures, PosTags, Sentences, Subwords, Tokens, Words,
};
use turbo_relation::Relations;
use turbo_tagger::multi::{Head, HeadLayer};
use turbo_tagger::segment::decode_sentences;

use super::group::GroupHead;
use super::{COREF_INSTANCE, HeadGroup, HeadStep, Plan};

/// A layer a shared head step writes.
pub trait GroupLayer: HeadLayer {
    const STEP: HeadStep;
}

macro_rules! group_layer {
    ($($L:ty => $S:ident),+ $(,)?) => {
        $(impl GroupLayer for $L {
            const STEP: HeadStep = HeadStep::$S;
        })+
    };
}

group_layer!(
    PosTags => Pos,
    MorphFeatures<Tokens> => Morphology,
    Entities => Entities,
    Dependencies<Tokens> => Dependencies,
    Lemmas => Lemmas,
    Words => Words,
);

fn failed(id: &'static str) -> impl Fn(String) -> Error {
    move |detail| Error::Annotator { annotator: id, detail }
}

fn config(group: &HeadGroup, step: HeadStep) -> u64 {
    let name = group.head(step).map_or(String::new(), |h| h.spec_name());
    Fnv1a::new().str(&name).u64(group.width as u64).finish()
}

impl GroupHead {
    fn spec_name(&self) -> String {
        match self {
            GroupHead::Layer(h) => h.spec().name.clone(),
            GroupHead::Span(h) => h.head().spec().name.clone(),
            GroupHead::Relation(h) => h.head().spec().name.clone(),
            GroupHead::Coref(h) => h.head().spec().name.clone(),
        }
    }
}

macro_rules! provenance {
    ($step:expr) => {
        fn model(&self) -> Option<ModelRef> {
            Some(self.group.model.clone())
        }

        fn execution(&self) -> Option<Execution> {
            Some(self.group.execution.clone())
        }

        fn config_hash(&self) -> u64 {
            config(&self.group, $step)
        }
    };
}

/// A shared head step: layer `L` from its group's one encoder read.
pub struct GroupStep<L: GroupLayer> {
    group: Arc<HeadGroup>,
    _layer: PhantomData<fn() -> L>,
}

impl<L: GroupLayer> GroupStep<L> {
    pub fn new(group: Arc<HeadGroup>) -> GroupStep<L> {
        GroupStep { group, _layer: PhantomData }
    }
}

fn layer_head(group: &HeadGroup, step: HeadStep) -> Result<&Head, String> {
    match group.head(step)? {
        GroupHead::Layer(h) => Ok(h),
        _ => Err(format!("{} is not a layer head", step.name())),
    }
}

impl<L: GroupLayer> Annotator for GroupStep<L> {
    const ID: &'static str = L::STEP.name();
    type Reads = (View, Tokens, Option<Subwords>, Option<Sentences>);
    type Writes = (L,);

    fn annotate(&self, (view, tokens, subwords, sentences): Refs<'_, Self>) -> Result<Self::Writes, Error> {
        let run = || {
            let head = layer_head(&self.group, L::STEP)?;
            L::write(head, &self.group.input(view, tokens, sentences, subwords)?)
        };
        run().map(|l| (l,)).map_err(failed(Self::ID))
    }

    provenance!(L::STEP);
}

/// The learned sentence boundary head's sentences, from the group's
/// whole-view read.
pub struct SentenceStep(pub Arc<HeadGroup>);

impl Annotator for SentenceStep {
    const ID: &'static str = HeadStep::Sentences.name();
    type Reads = (View, Tokens, Option<Subwords>);
    type Writes = (Sentences,);

    fn annotate(&self, (view, tokens, subwords): Refs<'_, Self>) -> Result<Self::Writes, Error> {
        let run = || {
            let Head::Sentence { boundary } = layer_head(&self.0, HeadStep::Sentences)? else {
                return Err("the sentence step's head is not a sentence boundary head".into());
            };
            self.0.whole(view, tokens, subwords, |input| decode_sentences(boundary, input))
        };
        run().map(|s| (s,)).map_err(failed(Self::ID))
    }

    fn model(&self) -> Option<ModelRef> {
        Some(self.0.model.clone())
    }

    fn execution(&self) -> Option<Execution> {
        Some(self.0.execution.clone())
    }

    fn config_hash(&self) -> u64 {
        config(&self.0, HeadStep::Sentences)
    }
}

/// The span classifier head's entities, a sentence at a time.
pub struct SpanStep {
    pub group: Arc<HeadGroup>,
}

impl Annotator for SpanStep {
    const ID: &'static str = HeadStep::Spans.name();
    type Reads = (View, Tokens, Option<Subwords>, Option<Sentences>);
    type Writes = (Entities,);

    fn annotate(&self, (view, tokens, subwords, sentences): Refs<'_, Self>) -> Result<Self::Writes, Error> {
        let run = || {
            let GroupHead::Span(head) = self.group.head(HeadStep::Spans)? else { return Err("not a span head".into()) };
            let input = self.group.input(view, tokens, sentences, subwords)?;
            let f = self.group.features_for(HeadStep::Spans, &input)?;
            head.decode(view, tokens, &input.seqs, &f)
        };
        run().map(|e| (e,)).map_err(failed(Self::ID))
    }

    provenance!(HeadStep::Spans);
}

/// The relation classifier head's relations among the default entity
/// layer, a sentence at a time.
pub struct RelationStep {
    pub group: Arc<HeadGroup>,
}

impl Annotator for RelationStep {
    const ID: &'static str = HeadStep::Relations.name();
    type Reads = (View, Tokens, Entities, Option<Subwords>, Option<Sentences>);
    type Writes = (Relations, Edges<Entities, Entities>);

    fn annotate(&self, (view, tokens, entities, subwords, sentences): Refs<'_, Self>) -> Result<Self::Writes, Error> {
        let run = || {
            let GroupHead::Relation(head) = self.group.head(HeadStep::Relations)? else {
                return Err("not a relation head".into());
            };
            let input = self.group.input(view, tokens, sentences, subwords)?;
            let f = self.group.features_for(HeadStep::Relations, &input)?;
            head.relations_with(view, tokens, entities, input.seqs.clone(), |places, _| {
                Ok(places.iter().map(|&p| f[p].clone()).collect())
            })
        };
        run().map_err(failed(Self::ID))
    }

    provenance!(HeadStep::Relations);
}

/// The coreference head's chains and links over the default entity
/// layer, from the group's whole-view read.
pub struct CorefStep {
    pub group: Arc<HeadGroup>,
}

impl Annotator for CorefStep {
    const ID: &'static str = HeadStep::Coref.name();
    type Reads = (View, Tokens, Entities, Option<Subwords>);
    type Writes = (Groups<Entities>, Edges<Entities, Entities>);

    fn annotate(&self, (view, tokens, entities, subwords): Refs<'_, Self>) -> Result<Self::Writes, Error> {
        let run = || {
            let GroupHead::Coref(head) = self.group.head(HeadStep::Coref)? else {
                return Err("not a coreference head".into());
            };
            self.group.whole(view, tokens, subwords, |input| {
                let f = self.group.features_for(HeadStep::Coref, input)?;
                head.resolve_with(view, tokens, entities, |_, _| Ok(f))
            })
        };
        run().map_err(failed(Self::ID))
    }

    provenance!(HeadStep::Coref);
}

/// `builder` with the step of `step` added over the plan's group.
pub(super) fn add(builder: Builder, step: HeadStep, plan: &Plan) -> Builder {
    let g = || Arc::clone(&plan.group);
    match step {
        HeadStep::Pos => builder.add(GroupStep::<PosTags>::new(g())),
        HeadStep::Morphology => builder.add(GroupStep::<MorphFeatures<Tokens>>::new(g())),
        HeadStep::Entities => builder.add(GroupStep::<Entities>::new(g())),
        HeadStep::Dependencies => builder.add(GroupStep::<Dependencies<Tokens>>::new(g())),
        HeadStep::Lemmas => builder.add(GroupStep::<Lemmas>::new(g())),
        HeadStep::Words => builder.add(GroupStep::<Words>::new(g())),
        HeadStep::Sentences => builder.add(SentenceStep(g())),
        HeadStep::Spans => builder.step(Bound::new(SpanStep { group: g() }).write(0, plan.spans)),
        HeadStep::Relations => builder.add(RelationStep { group: g() }),
        HeadStep::Coref => builder.step(Bound::new(CorefStep { group: g() }).write(1, COREF_INSTANCE)),
    }
}
