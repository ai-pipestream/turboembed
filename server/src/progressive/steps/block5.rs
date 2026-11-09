//! Block 5 opt-in progressive steps: SaT sentences, PII scanning,
//! per-document risk and masking, and the numeric annotators.
//!
//! These steps are never part of the default profile. A request names
//! them explicitly beside the existing steps. SaT writes the same
//! `Sentences` layer as the rule sentence detector, so one pipeline picks
//! one sentence tier. PII masking writes the named `masked` view and
//! requires the PII layer from the scanning step.

use std::any::Any;
use std::path::Path;
use std::sync::Arc;

use turbo_abbrev::AbbreviationAnnotator;
use turbo_analysis::analyzer::Builder;
use turbo_analysis::annotate::{AnchorUse, Step};
use turbo_analysis::{Annotator, Bound, Document, Error, Key, KeyId, View, ViewId};
use turbo_analysis_layers::LayoutBoxes;
use turbo_chunker::levels::LevelBuilder;
use turbo_chunker::terms::TermIndexer;
use turbo_chunker::{
    ChunkCitationAnnotator, Costs, LayoutBoundaries, LexicalBoundaries, STRUCTURE, StructureBoundaries,
    WeighLayoutBoundaries,
};
use turbo_keywords::{Stopwords as KeywordStopwords, YakeAnnotator};
use turbo_numeric::money::MoneyExtractor;
use turbo_numeric::quantity::QuantityExtractor;
use turbo_numeric::temporal::TemporalExtractor;
use turbo_numeric::{DocumentDateSelector, NumberScanner};
use turbo_pii::rewrite::MaskedView;
use turbo_pii::{Pack, PiiDetector, PiiRiskScorer};
use turbo_sat::{SatAnnotator, SatOptions};
use turbo_stem::{Algorithm, StemAnnotator, TermAnnotator};
use turbo_stopword::{ListChoice, StopwordAnnotator, StopwordLanguage};
use turbo_structure::LayoutProjector;

use super::super::normalize::{check_required_inputs, store_single};
use super::EmbedContext;
use crate::api::{Failure, TURBO_E_BUNDLE_NOT_FOUND, TURBO_E_INVALID_ARGUMENT};
use crate::pii_policy::{REDACT_STEP, RedactStep, Redaction};
use crate::proto::result::VocabularyBinding;
use crate::state::State;
use crate::vocabulary::{ResolvedVocabulary, VocabularyResolver};

/// Whether `name` selects the SaT sentence tier.
pub(super) fn is_sat_step(name: &str) -> bool {
    matches!(name, "turbo-sat:sat" | "turbo:sat" | "sat")
}

/// Whether `name` selects the redact step.
pub(super) fn is_redact_step(name: &str) -> bool {
    matches!(name, REDACT_STEP | "turbo:pii-redact" | "pii-redact")
}

fn is_glossary_step(name: &str) -> bool {
    matches!(name, "turbo-glossary:annotator" | "turbo-glossary:glossary" | "turbo:glossary-terms" | "glossary")
}

/// Appends `"turbo-glossary:annotator"` when a glossary artifact is bound and
/// the pipeline has a tokenizer step but no explicit glossary step yet.
pub(super) fn append_bound_glossary(steps: &mut Vec<&str>, vocabulary: &ResolvedVocabulary) {
    if vocabulary.glossary.is_some()
        && !steps.iter().copied().any(is_glossary_step)
        && steps.iter().any(|&s| matches!(s, "turbo-tokenize:uax29-untyped" | "turbo:tokens"))
    {
        steps.push("turbo-glossary:annotator");
    }
}

/// Resolves the request's `vocabulary` binding over any vocabulary bound to the
/// bundle actually selected for this request (`model_name` or `selected`).
pub(super) fn resolve_vocabulary(
    state: Option<&State>,
    model_name: &str,
    selected: Option<&crate::state::Loaded>,
    req_vocab: Option<&VocabularyBinding>,
) -> Result<ResolvedVocabulary, Failure> {
    let trimmed = model_name.trim();
    let base = match state {
        None => ResolvedVocabulary::default(),
        Some(_) if selected.is_some() => selected.unwrap().vocabulary.clone(),
        Some(st) if !trimmed.is_empty() => st.ready(trimmed, "").map(|(_, l)| l.vocabulary.clone()).unwrap_or_default(),
        Some(_) => ResolvedVocabulary::default(),
    };
    match state {
        Some(st) => st.vocabulary_resolver.resolve_merged(&base, req_vocab),
        None => VocabularyResolver::default().resolve_merged(&base, req_vocab),
    }
}

/// Adds one block 5 or lexical step to `builder` when `name` names one.
///
/// Returns the builder and whether the step was handled here.
#[allow(clippy::too_many_arguments)]
pub(super) fn add_step(
    builder: Builder,
    name: &str,
    embed_cx: Option<&EmbedContext>,
    norm_chain: &[&'static str],
    step_names: &[&str],
    redaction: Option<&Redaction>,
    vocabulary: &ResolvedVocabulary,
) -> Result<(Builder, bool), Failure> {
    let has_weigh_layout =
        step_names.iter().any(|&s| matches!(s, "turbo:weigh-layout-boundaries" | "weigh_layout_boundaries"));
    let builder = match name {
        _ if is_sat_step(name) => {
            let cx = embed_cx
                .ok_or_else(|| Failure::new(TURBO_E_BUNDLE_NOT_FOUND, 0, "turbo:sat needs a loaded SaT bundle"))?;
            let annotator = SatAnnotator::open(
                Path::new(&cx.loaded.bundle),
                &SatOptions { encoder: cx.encoder_options(), ..SatOptions::default() },
            )
            .map_err(|e| Failure::new(TURBO_E_INVALID_ARGUMENT, 0, format!("turbo:sat: {e}")))?;
            builder.add(annotator)
        }
        "turbo-abbrev:schwartz-hearst" | "turbo-abbrev:abbreviations" | "turbo:abbreviations" => {
            builder.add(AbbreviationAnnotator)
        }
        "turbo-stem:terms" | "turbo:terms" => builder.add(TermAnnotator::default()),
        "turbo-stem:stemmer" | "turbo:stems" => builder.add(StemAnnotator::new(Algorithm::Porter)),
        "turbo-stopword:bundled" | "turbo-stopword:stopwords" | "turbo:stopwords" => match &vocabulary.stopwords {
            Some(sw) => builder.add(sw.stopword_annotator()),
            None => builder.add(StopwordAnnotator::new(ListChoice::Fixed(StopwordLanguage::English))),
        },
        "turbo-keywords:yake" | "turbo:keywords" => match &vocabulary.stopwords {
            Some(sw) => builder.add(sw.yake_annotator()),
            None => builder.add(YakeAnnotator::new(KeywordStopwords::bundled(StopwordLanguage::English))),
        },
        _ if is_glossary_step(name) => match &vocabulary.glossary {
            Some(g) => builder.add(g.annotator()),
            None => {
                return Err(Failure::new(
                    TURBO_E_INVALID_ARGUMENT,
                    0,
                    "turbo-glossary:annotator needs a glossary artifact: set vocabulary.glossary",
                ));
            }
        },
        "turbo:structure-boundaries" | "turbo:boundaries" if has_weigh_layout => {
            builder.step(Bound::new(StructureBoundaries { costs: Costs::default() }).write(0, STRUCTURE))
        }
        "turbo:structure-boundaries" | "turbo:boundaries" => {
            builder.add(StructureBoundaries { costs: Costs::default() })
        }
        "turbo:lexical-boundaries" => builder.step(LexicalBoundaries::default().bound()),
        "turbo:term-indexer" | "turbo:term-index" => builder.add(TermIndexer),
        "turbo:level-builder" | "turbo:levels" => builder.add(LevelBuilder),
        "turbo-pii:pii" | "turbo:pii" | "pii" => match redaction {
            None => builder.add(PiiDetector::pack(Pack::AllStructured)),
            Some(r) => builder.add(
                r.policy
                    .detector(Pack::AllStructured)
                    .map_err(|e| Failure::new(TURBO_E_INVALID_ARGUMENT, 0, format!("turbo-pii:pii: {e}")))?,
            ),
        },
        "turbo-pii:risk" | "turbo:pii-risk" | "pii-risk" => builder.add(PiiRiskScorer::new()),
        "turbo-pii:masked" | "turbo:pii-masked" | "pii-masked" => {
            let masked = MaskedView::by_kind('*')
                .map_err(|e| Failure::new(TURBO_E_INVALID_ARGUMENT, 0, format!("turbo-pii:masked: {e}")))?;
            builder.step(masked.bound())
        }
        REDACT_STEP | "turbo:pii-redact" | "pii-redact" => {
            let r = redaction.ok_or_else(|| {
                Failure::new(TURBO_E_INVALID_ARGUMENT, 0, "turbo-pii:redact needs a policy: set pii_redaction")
            })?;
            let step = RedactStep::new(Arc::clone(&r.policy));
            if r.dry_run { builder.add(step.dry_run()) } else { builder.step(step.bound()) }
        }
        "turbo-numeric:numbers" | "turbo:numbers" | "numbers" => builder.add(NumberScanner::default()),
        "turbo-numeric:money" | "turbo:money" | "money" => builder.add(MoneyExtractor::default()),
        "turbo-numeric:quantities" | "turbo:quantities" | "quantities" => builder.add(QuantityExtractor::default()),
        "turbo-numeric:temporal" | "turbo:temporal" | "temporal" => builder.add(TemporalExtractor::new()),
        "turbo-numeric:document-date" | "turbo:document-date" | "document-date" => builder.add(DocumentDateSelector),
        "turbo-structure:project-layout" | "layout_project" => {
            if norm_chain.is_empty() {
                return Err(Failure::new(
                    TURBO_E_INVALID_ARGUMENT,
                    0,
                    "turbo-structure:project-layout requires a preceding normalizer step",
                ));
            }
            builder.given(Key::<LayoutBoxes>::DEFAULT).step(NormalizedLayoutProjectStep {
                norm_chain: norm_chain.to_vec(),
                source_name: "",
                target_name: "normalized",
            })
        }
        "turbo:layout-boundaries" | "layout_boundaries" => {
            builder.given(Key::<LayoutBoxes>::DEFAULT).step(LayoutBoundaries::default().bound())
        }
        "turbo:weigh-layout-boundaries" | "weigh_layout_boundaries" => {
            builder.step(WeighLayoutBoundaries::default().bound())
        }
        "turbo:chunk-citations" | "chunk_citations" => {
            builder.given(Key::<LayoutBoxes>::DEFAULT).add(ChunkCitationAnnotator)
        }
        _ => return Ok((builder, false)),
    };
    Ok((builder, true))
}

struct NormalizedLayoutProjectStep {
    norm_chain: Vec<&'static str>,
    source_name: &'static str,
    target_name: &'static str,
}

impl Step for NormalizedLayoutProjectStep {
    fn id(&self) -> &'static str {
        LayoutProjector::ID
    }

    fn reads(&self) -> Vec<KeyId> {
        let last_view = *self.norm_chain.last().unwrap_or(&"normalized");
        vec![Key::<View>::named(last_view).id(), Key::<LayoutBoxes>::named(self.source_name).id()]
    }

    fn required(&self) -> Vec<bool> {
        vec![true, true]
    }

    fn writes(&self) -> Vec<KeyId> {
        vec![Key::<LayoutBoxes>::named(self.target_name).id()]
    }

    fn anchors(&self) -> Vec<AnchorUse> {
        vec![AnchorUse::None]
    }

    fn compute(&self, docs: &[Document]) -> Box<dyn Any + Send> {
        let req_keys = self.reads();
        let src_key = Key::<LayoutBoxes>::named(self.source_name);
        let out: Vec<Option<Result<LayoutBoxes, Error>>> = docs
            .iter()
            .map(|doc| {
                if let Some(why) = check_required_inputs(doc, &req_keys) {
                    return Some(Err(Error::Skipped(why)));
                }
                let res = (|| {
                    let src = doc.require(&src_key)?;
                    let mut current: Option<LayoutBoxes> = None;
                    for &vname in &self.norm_chain {
                        let view = doc.require(&Key::<View>::named(vname))?;
                        let input = current.as_ref().unwrap_or(src);
                        let (projected,) = LayoutProjector.annotate((view, input))?;
                        current = Some(projected);
                    }
                    let final_layout = current.expect("non-empty norm_chain");
                    if self.target_name.is_empty() {
                        doc.project_layer(&final_layout, ViewId::SOURCE)
                    } else {
                        Ok(final_layout)
                    }
                })();
                Some(res)
            })
            .collect();
        Box::new(out)
    }

    fn store(&self, out: Box<dyn Any + Send>, docs: &mut [Document], status: &mut [Result<(), Error>]) {
        let reads = self.reads();
        store_single(
            out,
            Key::<LayoutBoxes>::named(self.target_name),
            &LayoutProjector,
            move |_| reads.clone(),
            docs,
            status,
        );
    }
}
