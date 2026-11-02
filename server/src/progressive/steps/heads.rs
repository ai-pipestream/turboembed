//! Bundle head steps for `AnalyzeDocumentProgressive`: the learned heads
//! of a loaded `TASK_EMBED` bundle (`manifest.heads`, docs/bundle.md
//! "Heads") run as pipeline steps. The word-level heads of one request
//! (part of speech, morphology, entities, dependencies, lemmas) share one
//! encoder pass through a [`HeadGroup`]: whichever of their steps runs
//! first reads every window of the document once, pooling the states each
//! head asks for (`WordEncoder::token_features_pooled`), and the other
//! steps decode from those states. The span head writes the entity layer
//! as a token tagger would, and the relation head reads the entities, so
//! those two read on their own.
//!
//! Each layer is written exactly as the tagger crate's own annotator
//! writes it, from the same features; only the step's id differs.

mod group;

use std::path::Path;
use std::sync::Arc;

use turbo::bundle::Bundle;
use turbo::manifest::{HeadKind, HeadSpec, Scheme};
use turbo::status::Error as CoreError;
use turbo_analysis::Layer;
use turbo_analysis::analyzer::Builder;
use turbo_analysis_layers::{Dependencies, Entities, Lemmas, MorphFeatures, PosTags, Tokens};
use turbo_relation::Relations;
use turbo_tagger::annotate::same_layer;
use turbo_tagger::multi::{Head, HeadLayer};
use turbo_tagger::{RelationTagger, SpanTagger};

pub use group::{GroupLayer, GroupStep, HeadGroup, RelationStep, SpanStep};

use super::EmbedContext;
use crate::api::{Failure, TURBO_E_BUNDLE_NOT_FOUND, TURBO_E_INVALID_ARGUMENT};
use crate::state::Loaded;

/// The step that stands for every head step the request's bundle
/// provides (the entity step over the span step when it has both).
pub const ALL_STEP: &str = "turbo-tagger:heads";

/// One head step: the layer it writes and the head kind that writes it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum HeadStep {
    Pos,
    Morphology,
    Entities,
    Dependencies,
    Lemmas,
    Spans,
    Relations,
}

impl HeadStep {
    /// Every head step, in the order the all-heads step adds them.
    pub const ALL: [HeadStep; 7] = [
        HeadStep::Pos,
        HeadStep::Morphology,
        HeadStep::Entities,
        HeadStep::Dependencies,
        HeadStep::Lemmas,
        HeadStep::Spans,
        HeadStep::Relations,
    ];

    /// The step's id, as a request names it and events report it.
    pub const fn name(self) -> &'static str {
        match self {
            HeadStep::Pos => "turbo-tagger:pos",
            HeadStep::Morphology => "turbo-tagger:morphology",
            HeadStep::Entities => "turbo-tagger:entities",
            HeadStep::Dependencies => "turbo-tagger:dependencies",
            HeadStep::Lemmas => "turbo-tagger:lemmas",
            HeadStep::Spans => "turbo-tagger:spans",
            HeadStep::Relations => "turbo-tagger:relations",
        }
    }

    /// The bundle requirement a profile names for this step.
    pub const fn requirement(self) -> &'static str {
        match self {
            HeadStep::Pos => "head.pos",
            HeadStep::Morphology => "head.morphology",
            HeadStep::Entities => "head.entities",
            HeadStep::Dependencies => "head.dependencies",
            HeadStep::Lemmas => "head.lemmas",
            HeadStep::Spans => "head.spans",
            HeadStep::Relations => "head.relations",
        }
    }

    pub fn parse(name: &str) -> Option<HeadStep> {
        HeadStep::ALL.into_iter().find(|s| s.name() == name)
    }

    pub fn of_requirement(requirement: &str) -> Option<HeadStep> {
        HeadStep::ALL.into_iter().find(|s| s.requirement() == requirement)
    }

    /// Whether the step decodes from the group's shared pass.
    fn shared(self) -> bool {
        !matches!(self, HeadStep::Spans | HeadStep::Relations)
    }

    fn layer(self) -> &'static str {
        match self {
            HeadStep::Pos => PosTags::ID,
            HeadStep::Morphology => MorphFeatures::<Tokens>::ID,
            HeadStep::Entities | HeadStep::Spans => Entities::ID,
            HeadStep::Dependencies => Dependencies::<Tokens>::ID,
            HeadStep::Lemmas => Lemmas::ID,
            HeadStep::Relations => Relations::ID,
        }
    }

    fn kind(self) -> HeadKind {
        match self {
            HeadStep::Pos | HeadStep::Morphology | HeadStep::Entities => HeadKind::TokenTagger,
            HeadStep::Dependencies => HeadKind::BiaffineParser,
            HeadStep::Lemmas => HeadKind::EditTreeLemmatizer,
            HeadStep::Spans => HeadKind::SpanClassifier,
            HeadStep::Relations => HeadKind::RelationClassifier,
        }
    }

    /// Whether a head of the step's kind and layer under `scheme` is
    /// this step's: tags for part of speech and morphology, spans for
    /// entities.
    fn takes(self, scheme: Scheme) -> bool {
        match self {
            HeadStep::Pos | HeadStep::Morphology => scheme == Scheme::Tags,
            HeadStep::Entities => scheme != Scheme::Tags,
            _ => true,
        }
    }

    /// The one head of `bundle` this step runs, if the bundle has
    /// exactly one.
    fn head(self, bundle: &Bundle) -> Option<&HeadSpec> {
        let mut found = bundle
            .manifest
            .heads
            .iter()
            .filter(|h| h.kind == self.kind() && same_layer(&h.layer, self.layer()) && self.takes(h.scheme));
        match (found.next(), found.next()) {
            (Some(h), None) => Some(h),
            _ => None,
        }
    }

    /// The step's head of `bundle` loaded for the shared pass.
    fn open(self, bundle: &Bundle, spec: &HeadSpec) -> Result<Head, CoreError> {
        match self {
            HeadStep::Pos => PosTags::open(bundle, spec),
            HeadStep::Morphology => MorphFeatures::<Tokens>::open(bundle, spec),
            HeadStep::Entities => Entities::open(bundle, spec),
            HeadStep::Dependencies => Dependencies::<Tokens>::open(bundle, spec),
            HeadStep::Lemmas => Lemmas::open(bundle, spec),
            HeadStep::Spans | HeadStep::Relations => {
                Err(CoreError::new(TURBO_E_INVALID_ARGUMENT, format!("{} reads on its own", self.name())))
            }
        }
    }
}

/// The head steps the bundle at `dir` provides, in step order: the
/// bundle's `Capabilities` entry lists them, and the all-heads step
/// runs them.
pub fn available(dir: &str) -> Vec<String> {
    let Ok(bundle) = Bundle::open(Path::new(dir)) else { return Vec::new() };
    HeadStep::ALL.into_iter().filter(|s| s.head(&bundle).is_some()).map(|s| s.name().to_owned()).collect()
}

/// Whether `name` is a head step, so the pipeline needs an embed bundle.
pub(super) fn is_head_step(name: &str) -> bool {
    name == ALL_STEP || HeadStep::parse(name).is_some()
}

/// The head steps a request needs its bundle to provide: those it names
/// and those its profile requires.
pub(super) fn needed(step_names: &[&str], required_bundles: &[&str]) -> Vec<HeadStep> {
    let named = step_names.iter().filter_map(|n| HeadStep::parse(n));
    let required = required_bundles.iter().filter_map(|r| HeadStep::of_requirement(r));
    let mut out: Vec<HeadStep> = Vec::new();
    for s in named.chain(required) {
        if !out.contains(&s) {
            out.push(s);
        }
    }
    out
}

fn provides(loaded: &Loaded, step: HeadStep) -> bool {
    loaded.head_steps.iter().any(|n| n == step.name())
}

fn not_loaded(who: &str, step: HeadStep, where_: &str) -> Failure {
    Failure::new(TURBO_E_BUNDLE_NOT_FOUND, 0, format!("{who} requires bundle '{}', which {where_}", step.requirement()))
}

/// Refuses the named model when it lacks a needed head.
pub(super) fn check_model(loaded: &Loaded, who: &str, model: &str, needed: &[HeadStep]) -> Result<(), Failure> {
    match needed.iter().copied().find(|&s| !provides(loaded, s)) {
        Some(s) => Err(not_loaded(who, s, &format!("model '{model}' does not provide"))),
        None => Ok(()),
    }
}

/// The models providing every needed head, or a refusal naming the
/// requirement no loaded model meets.
pub(super) fn select(models: Vec<Arc<Loaded>>, who: &str, needed: &[HeadStep]) -> Result<Vec<Arc<Loaded>>, Failure> {
    if needed.is_empty() {
        return Ok(models);
    }
    let kept: Vec<Arc<Loaded>> = models.iter().filter(|l| needed.iter().all(|&s| provides(l, s))).cloned().collect();
    if kept.is_empty() {
        let unmet = needed.iter().copied().find(|&s| !models.iter().any(|l| provides(l, s)));
        return Err(not_loaded(who, unmet.unwrap_or(needed[0]), "is not loaded"));
    }
    Ok(kept)
}

/// `step_names` with the all-heads step replaced by the head steps the
/// request's bundle provides, none twice.
pub(super) fn expand<'a>(step_names: &[&'a str], cx: Option<&EmbedContext>) -> Result<Vec<&'a str>, Failure> {
    if !step_names.contains(&ALL_STEP) {
        return Ok(step_names.to_vec());
    }
    let cx = cx.ok_or_else(|| {
        Failure::new(
            TURBO_E_BUNDLE_NOT_FOUND,
            0,
            format!("{ALL_STEP} requires bundle 'TASK_EMBED', which is not loaded"),
        )
    })?;
    if cx.loaded.head_steps.is_empty() {
        return Err(Failure::new(
            TURBO_E_BUNDLE_NOT_FOUND,
            0,
            format!("{ALL_STEP}: the bundle {} carries no head", cx.loaded.bundle),
        ));
    }
    let mut out: Vec<&'a str> = Vec::new();
    for &name in step_names {
        if name != ALL_STEP {
            out.push(name);
            continue;
        }
        for s in HeadStep::ALL {
            let shadowed = s == HeadStep::Spans && provides(&cx.loaded, HeadStep::Entities);
            if provides(&cx.loaded, s) && !shadowed && !out.contains(&s.name()) {
                out.push(s.name());
            }
        }
    }
    Ok(out)
}

/// The group of the shared steps among `step_names`, when there are any.
pub(super) fn group(cx: Option<&EmbedContext>, step_names: &[&str]) -> Result<Option<Arc<HeadGroup>>, Failure> {
    let mut steps: Vec<HeadStep> = Vec::new();
    for s in step_names.iter().filter_map(|n| HeadStep::parse(n)).filter(|s| s.shared()) {
        if !steps.contains(&s) {
            steps.push(s);
        }
    }
    if steps.is_empty() {
        return Ok(None);
    }
    let cx = cx.expect("embed context resolved");
    HeadGroup::open(Path::new(&cx.loaded.bundle), &steps, &cx.encoder_options())
        .map(Some)
        .map_err(|e| Failure::new(e.code, 0, format!("bundle heads: {}", e.message)))
}

/// Adds the head step `name` to `builder`, when it is one.
pub(super) fn add_step(
    builder: Builder,
    name: &str,
    cx: Option<&EmbedContext>,
    group: Option<&Arc<HeadGroup>>,
) -> Result<(Builder, bool), Failure> {
    let Some(step) = HeadStep::parse(name) else { return Ok((builder, false)) };
    let cx = cx.expect("embed context resolved");
    let dir = Path::new(&cx.loaded.bundle);
    let bad = |e: CoreError| Failure::new(TURBO_E_INVALID_ARGUMENT, 0, format!("{name}: {}", e.message));
    let own_head = || {
        let bundle = Bundle::open(dir).map_err(bad)?;
        step.head(&bundle)
            .map(|h| h.name.clone())
            .ok_or_else(|| not_loaded(name, step, &format!("the bundle {} does not provide", cx.loaded.bundle)))
    };
    let group = || Arc::clone(group.expect("a group for the shared steps"));
    let builder = match step {
        HeadStep::Spans => {
            let head = own_head()?;
            builder.add(SpanStep(SpanTagger::open(dir, Some(&head), &cx.encoder_options()).map_err(bad)?))
        }
        HeadStep::Relations => {
            let head = own_head()?;
            builder.add(RelationStep(RelationTagger::open(dir, Some(&head), &cx.encoder_options()).map_err(bad)?))
        }
        HeadStep::Pos => builder.add(GroupStep::<PosTags>::new(group())),
        HeadStep::Morphology => builder.add(GroupStep::<MorphFeatures<Tokens>>::new(group())),
        HeadStep::Entities => builder.add(GroupStep::<Entities>::new(group())),
        HeadStep::Dependencies => builder.add(GroupStep::<Dependencies<Tokens>>::new(group())),
        HeadStep::Lemmas => builder.add(GroupStep::<Lemmas>::new(group())),
    };
    Ok((builder, true))
}
