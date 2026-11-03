//! Bundle head steps for `AnalyzeDocumentProgressive`: the learned heads
//! of a loaded `TASK_EMBED` bundle (`manifest.heads`, docs/bundle.md
//! "Heads") run as pipeline steps, every one of a request decoding from
//! one [`HeadGroup`] read of the document (group.rs has the read plan).
//!
//! A step resolves to one head of the bundle by kind, label scheme and
//! layer, or by name when the request selects one (`turbo-tagger:pos@tags`).
//! A selection that finds no head, several, or a head of another kind is
//! refused by name (`missing head`, `ambiguous head`, `unsupported head`),
//! and so are two steps writing one output (`ambiguous output`); see
//! docs/grpc-analysis.md "Learned head steps" for the output keys.

mod group;
mod steps;

use std::path::Path;
use std::sync::Arc;

use turbo::bundle::Bundle;
use turbo::manifest::{HeadKind, HeadSpec, Scheme};
use turbo_analysis::Layer;
use turbo_analysis::analyzer::Builder;
use turbo_analysis_layers::{Dependencies, Entities, Lemmas, MorphFeatures, PosTags, Sentences, Tokens, Words};
use turbo_relation::Relations;
use turbo_tagger::annotate::same_layer;
use turbo_tagger::coref::COREF_LAYER;

pub use group::HeadGroup;
pub use steps::{CorefStep, GroupLayer, GroupStep, RelationStep, SentenceStep, SpanStep};

use super::EmbedContext;
use crate::api::{Failure, TURBO_E_BUNDLE_NOT_FOUND, TURBO_E_INVALID_ARGUMENT, TURBO_E_UNSUPPORTED_TASK};
use crate::state::Loaded;

/// The step that stands for every head step the request's bundle
/// provides.
pub const ALL_STEP: &str = "turbo-tagger:heads";

/// The profile requirement met by a bundle with any head step.
pub const ANY_HEAD: &str = "head.any";

/// The instance the span step writes its entities to when another step
/// writes the default entity layer.
pub const SPANS_INSTANCE: &str = "spans";

/// The instance the coreference step writes its links to, beside the
/// relation step's entity-to-entity edges.
pub const COREF_INSTANCE: &str = "coref";

/// The sentence instance the tokenizer reads beside a learned sentence
/// step: nothing writes it, so the tokens carry no sentence column and
/// the learned sentences, which read the tokens, come after them.
pub(super) const UNREAD_SENTENCES: &str = "turbo-tagger:sentences-unread";

/// The rule sentence steps a learned sentence step replaces.
pub(super) const RULE_SENTENCES: [&str; 2] = ["turbo-sentdetect:rules", "turbo:sentences"];

/// The rule step that writes the default entity layer.
const RULE_ENTITIES: [&str; 2] = ["turbo-geo:names", "turbo:entities"];

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
    Sentences,
    Words,
    Coref,
}

impl HeadStep {
    /// Every head step, in the order the all-heads step adds them.
    pub const ALL: [HeadStep; 10] = [
        HeadStep::Pos,
        HeadStep::Morphology,
        HeadStep::Entities,
        HeadStep::Dependencies,
        HeadStep::Lemmas,
        HeadStep::Spans,
        HeadStep::Relations,
        HeadStep::Sentences,
        HeadStep::Words,
        HeadStep::Coref,
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
            HeadStep::Sentences => "turbo-tagger:sentences",
            HeadStep::Words => "turbo-tagger:words",
            HeadStep::Coref => "turbo-tagger:coref",
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
            HeadStep::Sentences => "head.sentences",
            HeadStep::Words => "head.words",
            HeadStep::Coref => "head.coref",
        }
    }

    /// The step a request names, without its head selection.
    pub fn parse(name: &str) -> Option<HeadStep> {
        let step = name.split_once('@').map_or(name, |(s, _)| s);
        HeadStep::ALL.into_iter().find(|s| s.name() == step)
    }

    pub fn of_requirement(requirement: &str) -> Option<HeadStep> {
        HeadStep::ALL.into_iter().find(|s| s.requirement() == requirement)
    }

    /// Whether the step reads the whole view as one sequence of words,
    /// not a sentence at a time.
    pub const fn whole_view(self) -> bool {
        matches!(self, HeadStep::Sentences | HeadStep::Coref)
    }

    pub(crate) fn layer(self) -> &'static str {
        match self {
            HeadStep::Pos => PosTags::ID,
            HeadStep::Morphology => MorphFeatures::<Tokens>::ID,
            HeadStep::Entities | HeadStep::Spans => Entities::ID,
            HeadStep::Dependencies => Dependencies::<Tokens>::ID,
            HeadStep::Lemmas => Lemmas::ID,
            HeadStep::Relations => Relations::ID,
            HeadStep::Sentences => Sentences::ID,
            HeadStep::Words => Words::ID,
            HeadStep::Coref => COREF_LAYER,
        }
    }

    pub(crate) fn kind(self) -> HeadKind {
        match self {
            HeadStep::Pos | HeadStep::Morphology | HeadStep::Entities => HeadKind::TokenTagger,
            HeadStep::Dependencies => HeadKind::BiaffineParser,
            HeadStep::Lemmas => HeadKind::EditTreeLemmatizer,
            HeadStep::Spans => HeadKind::SpanClassifier,
            HeadStep::Relations => HeadKind::RelationClassifier,
            HeadStep::Sentences => HeadKind::SentenceBoundary,
            HeadStep::Words => HeadKind::TokenSplit,
            HeadStep::Coref => HeadKind::CorefRanker,
        }
    }

    /// Whether `spec` is a head this step runs: of the step's kind and
    /// layer, with tags for part of speech and morphology and spans for
    /// BIO/BILOU entities.
    fn takes(self, spec: &HeadSpec) -> bool {
        let scheme = match self {
            HeadStep::Pos | HeadStep::Morphology => spec.scheme == Scheme::Tags,
            HeadStep::Entities => spec.scheme != Scheme::Tags,
            _ => true,
        };
        scheme && spec.kind == self.kind() && same_layer(&spec.layer, self.layer())
    }

    /// The head of `bundle` this step runs: the one named `selected`, or
    /// with no selection the one head it takes.
    pub fn resolve<'b>(self, bundle: &'b Bundle, selected: Option<&str>) -> Result<&'b HeadSpec, Selection> {
        let heads = &bundle.manifest.heads;
        if let Some(name) = selected {
            let spec =
                heads.iter().find(|h| h.name == name).ok_or_else(|| Selection::Missing(self, Some(name.into())))?;
            return if self.takes(spec) { Ok(spec) } else { Err(Selection::Unsupported(self, spec.name.clone())) };
        }
        let found: Vec<&HeadSpec> = heads.iter().filter(|h| self.takes(h)).collect();
        match found[..] {
            [one] => Ok(one),
            [] => Err(Selection::Missing(self, None)),
            _ => Err(Selection::Ambiguous(self, found.iter().map(|h| h.name.clone()).collect())),
        }
    }
}

/// Why a step's head could not be resolved.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Selection {
    /// No head of the step's kind, or none of the selected name.
    Missing(HeadStep, Option<String>),
    /// Several heads the step takes and no selection among them.
    Ambiguous(HeadStep, Vec<String>),
    /// The selected head is of another kind, layer or scheme.
    Unsupported(HeadStep, String),
}

impl Selection {
    /// The refusal, `who` naming the pipeline or profile and `bundle` the
    /// bundle's directory.
    pub fn failure(&self, who: &str, bundle: &str) -> Failure {
        match self {
            Selection::Missing(s, None) => not_loaded(who, *s, &format!("the bundle {bundle} does not provide")),
            Selection::Missing(s, Some(n)) => Failure::new(
                TURBO_E_BUNDLE_NOT_FOUND,
                0,
                format!("missing head: {} selects head {n:?}, which the bundle {bundle} does not carry", s.name()),
            ),
            Selection::Ambiguous(s, names) => Failure::new(
                TURBO_E_INVALID_ARGUMENT,
                0,
                format!(
                    "ambiguous head: {} matches heads {names:?} of the bundle {bundle}; select one as {}@<head>",
                    s.name(),
                    s.name()
                ),
            ),
            Selection::Unsupported(s, n) => Failure::new(
                TURBO_E_UNSUPPORTED_TASK,
                0,
                format!(
                    "unsupported head: head {n:?} is not a {:?} head writing {} for {}",
                    s.kind(),
                    s.layer(),
                    s.name()
                ),
            ),
        }
    }
}

/// The head steps the bundle at `dir` provides (at least one head it
/// takes; several need a selection), in step order: the bundle's
/// `Capabilities` entry lists them, and the all-heads step runs them.
pub fn available(dir: &str) -> Vec<String> {
    let Ok(bundle) = Bundle::open(Path::new(dir)) else { return Vec::new() };
    HeadStep::ALL
        .into_iter()
        .filter(|s| bundle.manifest.heads.iter().any(|h| s.takes(h)))
        .map(|s| s.name().to_owned())
        .collect()
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

fn no_head(who: &str, where_: &str) -> Failure {
    Failure::new(TURBO_E_BUNDLE_NOT_FOUND, 0, format!("{who} requires bundle '{ANY_HEAD}', which {where_}"))
}

/// Refuses the named model when it lacks a needed head, or any head
/// when `any` is required.
pub(super) fn check_model(
    loaded: &Loaded,
    who: &str,
    model: &str,
    needed: &[HeadStep],
    any: bool,
) -> Result<(), Failure> {
    if any && loaded.head_steps.is_empty() {
        return Err(no_head(who, &format!("model '{model}' does not provide")));
    }
    match needed.iter().copied().find(|&s| !provides(loaded, s)) {
        Some(s) => Err(not_loaded(who, s, &format!("model '{model}' does not provide"))),
        None => Ok(()),
    }
}

/// The models providing every needed head (and some head when `any`),
/// or a refusal naming the requirement no loaded model meets.
pub(super) fn select(
    models: Vec<Arc<Loaded>>,
    who: &str,
    needed: &[HeadStep],
    any: bool,
) -> Result<Vec<Arc<Loaded>>, Failure> {
    if needed.is_empty() && !any {
        return Ok(models);
    }
    let fits = |l: &Loaded| needed.iter().all(|&s| provides(l, s)) && (!any || !l.head_steps.is_empty());
    let kept: Vec<Arc<Loaded>> = models.iter().filter(|l| fits(l)).cloned().collect();
    if kept.is_empty() {
        return Err(match needed.iter().copied().find(|&s| !models.iter().any(|l| provides(l, s))) {
            Some(s) => not_loaded(who, s, "is not loaded"),
            None if any => no_head(who, "is not loaded"),
            None => not_loaded(who, needed[0], "is not loaded"),
        });
    }
    Ok(kept)
}

/// `step_names` with the all-heads step replaced by the head steps the
/// request's bundle provides, none twice. A learned sentence step added
/// this way replaces the rule sentence step.
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
    let named: Vec<HeadStep> = step_names.iter().filter_map(|n| HeadStep::parse(n)).collect();
    let learned = provides(&cx.loaded, HeadStep::Sentences) && !named.contains(&HeadStep::Sentences);
    let mut out: Vec<&'a str> = Vec::new();
    for &name in step_names {
        if learned && RULE_SENTENCES.contains(&name) {
            continue;
        }
        if name != ALL_STEP {
            out.push(name);
            continue;
        }
        for s in HeadStep::ALL {
            if provides(&cx.loaded, s) && !named.contains(&s) && !out.contains(&s.name()) {
                out.push(s.name());
            }
        }
    }
    Ok(out)
}

/// What the head steps of a request run on: the group they decode from
/// and where the span step writes.
pub(super) struct Plan {
    pub(super) group: Arc<HeadGroup>,
    /// The entity instance the span step writes: the default one, or
    /// [`SPANS_INSTANCE`] when another step writes the default.
    pub(super) spans: &'static str,
    /// Whether a learned sentence step writes the sentences.
    pub(super) learned_sentences: bool,
}

fn ambiguous_output(a: &str, b: &str, layer: &str) -> Failure {
    Failure::new(TURBO_E_INVALID_ARGUMENT, 0, format!("ambiguous output: {a} and {b} both write {layer}"))
}

/// The plan of the head steps among `step_names`, when there are any:
/// each step's head resolved against the bundle, and one group opened
/// for all of them.
pub(super) fn plan(cx: Option<&EmbedContext>, step_names: &[&str], who: &str) -> Result<Option<Plan>, Failure> {
    let mut steps: Vec<(HeadStep, Option<&str>)> = Vec::new();
    for &name in step_names {
        let Some(s) = HeadStep::parse(name) else { continue };
        if steps.iter().any(|(t, _)| *t == s) {
            return Err(ambiguous_output(name, s.name(), s.layer()));
        }
        steps.push((s, name.split_once('@').map(|(_, h)| h)));
    }
    if steps.is_empty() {
        return Ok(None);
    }
    let has = |s: HeadStep| steps.iter().any(|(t, _)| *t == s);
    if let (true, Some(rule)) = (has(HeadStep::Sentences), step_names.iter().find(|n| RULE_SENTENCES.contains(n))) {
        return Err(ambiguous_output(rule, HeadStep::Sentences.name(), Sentences::ID));
    }
    if has(HeadStep::Sentences) && step_names.iter().any(|n| super::is_normalizer_step(n)) {
        return Err(Failure::new(
            TURBO_E_INVALID_ARGUMENT,
            0,
            "turbo-tagger:sentences reads the source view: drop the normalizer steps",
        ));
    }
    let rule_entities = step_names.iter().find(|n| RULE_ENTITIES.contains(n));
    if let (true, Some(rule)) = (has(HeadStep::Entities), rule_entities) {
        return Err(ambiguous_output(rule, HeadStep::Entities.name(), Entities::ID));
    }
    let cx = cx.expect("embed context resolved");
    let bundle_dir = &cx.loaded.bundle;
    let bundle = Bundle::open(Path::new(bundle_dir))
        .map_err(|e| Failure::new(e.code, 0, format!("bundle heads: {}", e.message)))?;
    let mut heads: Vec<(HeadStep, String)> = Vec::with_capacity(steps.len());
    for &(s, selected) in &steps {
        let spec = s.resolve(&bundle, selected).map_err(|e| e.failure(who, bundle_dir))?;
        heads.push((s, spec.name.clone()));
    }
    let group = HeadGroup::open_resolved(Path::new(bundle_dir), &heads, &cx.encoder_options())
        .map_err(|e| Failure::new(e.code, 0, format!("bundle heads: {}", e.message)))?;
    let spans = if has(HeadStep::Entities) || rule_entities.is_some() { SPANS_INSTANCE } else { "" };
    Ok(Some(Plan { group, spans, learned_sentences: has(HeadStep::Sentences) }))
}

/// Adds the head step `name` to `builder`, when it is one.
pub(super) fn add_step(builder: Builder, name: &str, plan: Option<&Plan>) -> (Builder, bool) {
    let Some(step) = HeadStep::parse(name) else { return (builder, false) };
    let plan = plan.expect("a plan for the head steps");
    (steps::add(builder, step, plan), true)
}
