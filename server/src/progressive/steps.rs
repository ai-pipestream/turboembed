//! Built-in first-party [`turbo_analysis::Annotator`] step registration for
//! `AnalyzeDocumentProgressive` (issues `#1218`, `#1302`, `#1325`).

use std::marker::PhantomData;
use std::sync::OnceLock;

use turbo_analysis::analyzer::Builder;
use turbo_analysis::{Analyzer, Bound, Document, Key};
use turbo_analysis_embed::{ChunkVectors, LevelVectors, SentenceChunkVectors, SentenceVectors, SubwordVectors};
use turbo_analysis_layers::{
    Boundaries, Chunks, Entities, Levels, Sentences, Stems, Structure, Subwords, Terms, Tokens,
};
use turbo_chunker::{Budget, Chunker};
use turbo_geo::{BundledGazetteer, NameAnnotator, NameFinder};
use turbo_gliner::ExtractionSchema;
use turbo_normalize::{CaseFold, Chain, Invisible, NormalizeStep, Normalizer, UnicodeForm, Whitespace};
use turbo_relation::ZeroShotRelationConfig;
use turbo_sentdetect::RuleSentenceDetector;
use turbo_stopword_layers::Stopwords;
use turbo_structure::{StructureAnnotator, Syntax};
use turbo_tokenize::uax29::UntypedWordTokenizer;

use super::normalize::{
    NormalizedNameStep, NormalizedSourceStep, NormalizedStructureProjectStep, NormalizedSubwordsStep,
    NormalizedTokensStep,
};

mod block5;
pub mod heads;
mod relations;

use crate::api::{Failure, TURBO_E_INVALID_ARGUMENT};
use crate::pii_policy::Redaction;
use crate::proto::result::VocabularyBinding;
use crate::state::State;
use crate::vocabulary::ResolvedVocabulary;

pub(crate) const DEFAULT_STEPS: &[&str] = &[
    "turbo-structure:blocks",
    "turbo-sentdetect:rules",
    "turbo-tokenize:uax29-untyped",
    "turbo-abbrev:schwartz-hearst",
    "turbo-stem:terms",
    "turbo-stem:stemmer",
    "turbo-stopword:bundled",
    "turbo-keywords:yake",
    "turbo:structure-boundaries",
    "turbo:lexical-boundaries",
];

pub(crate) const MINIMAL_STEPS: &[&str] = &["turbo-sentdetect:rules", "turbo-tokenize:uax29-untyped"];

pub(crate) const CHUNKING_STEPS: &[&str] = &[
    "turbo-structure:blocks",
    "turbo-sentdetect:rules",
    "turbo-tokenize:uax29-untyped",
    "turbo-stem:terms",
    "turbo-stopword:bundled",
    "turbo:structure-boundaries",
    "turbo:lexical-boundaries",
];

const NORM_VIEWS: &[&str] = &["norm_1", "norm_2", "norm_3", "norm_4", "norm_5", "norm_6", "norm_7", "normalized"];

fn bad(msg: impl Into<String>) -> Failure {
    Failure::new(TURBO_E_INVALID_ARGUMENT, 0, msg)
}

fn bundled_name_finder() -> NameFinder<'static> {
    static IMAGE: OnceLock<Vec<u8>> = OnceLock::new();
    let bytes = IMAGE.get_or_init(|| BundledGazetteer::get().name_image().expect("bundled gazetteer name image"));
    NameFinder::from_bytes(bytes).expect("valid bundled name image")
}

pub(super) fn is_normalizer_step(name: &str) -> bool {
    matches!(
        name,
        "turbo:normalize"
            | "turbo:normalize:chain"
            | "normalize"
            | "turbo:normalize:unicode-form"
            | "turbo:normalize:nfkc"
            | "unicode_normalize"
            | "turbo:normalize:nfc"
            | "turbo:normalize:case-fold"
            | "case_fold"
            | "turbo:normalize:whitespace"
            | "whitespace"
            | "turbo:normalize:invisible"
            | "turbo:normalize:clean-text"
            | "clean_text"
    )
}

pub(super) fn step_needs_sparse(name: &str) -> bool {
    matches!(name, "turbo:sparse" | "turbo:sparse-terms" | "sparse_terms")
}

pub(super) fn step_needs_embed(name: &str) -> bool {
    block5::is_sat_step(name)
        || heads::is_head_step(name)
        || step_needs_sparse(name)
        || matches!(
            name,
            "turbo:subwords"
                | "subwords"
                | "turbo:sentence-vectors"
                | "sentence_vectors"
                | "turbo:chunk-vectors"
                | "chunk_vectors"
                | "turbo:sentence-chunk-vectors"
                | "sentence_chunk_vectors"
                | "turbo:subword-vectors"
                | "subword_vectors"
                | "turbo:level-vectors"
                | "level_vectors"
        )
}

fn is_structure_project_step(name: &str) -> bool {
    matches!(name, "turbo-structure:project" | "structure_project")
}

fn is_structure_blocks_step(name: &str) -> bool {
    matches!(name, "turbo-structure:blocks" | "turbo-structure:structure" | "turbo:structure")
}

/// Registers any layers already present on `initial_doc` as `given` on `builder`.
pub(super) fn register_given_layers(mut builder: Builder, initial_doc: &Document) -> Builder {
    macro_rules! given {
        ($($L:ty),+) => {
            $(if initial_doc.present(Key::<$L>::DEFAULT.id()) {
                builder = builder.given(Key::<$L>::DEFAULT);
            })+
        };
    }
    given!(Structure, Sentences, Tokens, Entities, Subwords, Terms, Stems, Stopwords, Boundaries, Chunks, Levels);
    builder
}

mod embed;
mod gliner;

pub(crate) use embed::EmbedContext;
use embed::resolve_embed_context;

fn add_norm<N: Normalizer>(builder: Builder, n: N, read_view: &'static str, write_view: &'static str) -> Builder {
    builder.step(Bound::new(NormalizeStep::new(n)).read(0, read_view).write(0, write_view))
}

#[derive(Default)]
pub(crate) struct StepOptions<'a> {
    pub(crate) redaction: Option<&'a Redaction>,
    pub(crate) open_span: Option<ExtractionSchema>,
    pub(crate) relations: Option<ZeroShotRelationConfig>,
    pub(crate) relation_model: &'a str,
    pub(crate) vocabulary: Option<&'a VocabularyBinding>,
}

/// Builds an [`Analyzer`] from `initial_doc` and the requested `step_names`
/// without model state.
pub fn compile_analyzer(initial_doc: &Document, step_names: &[&str]) -> Result<Analyzer, Failure> {
    compile_analyzer_with_state(initial_doc, step_names, None, "", "", &[], StepOptions::default())
}

/// [`compile_analyzer`] with the redaction policy `turbo-pii:redact` runs.
pub fn compile_analyzer_with_redaction(
    initial_doc: &Document,
    step_names: &[&str],
    redaction: &Redaction,
) -> Result<Analyzer, Failure> {
    let options = StepOptions { redaction: Some(redaction), ..StepOptions::default() };
    compile_analyzer_with_state(initial_doc, step_names, None, "", "", &[], options)
}

/// Builds an [`Analyzer`] from `initial_doc`, resolving model-backed steps and
/// required profile bundles against `state`.
#[allow(clippy::too_many_arguments)]
pub(crate) fn compile_analyzer_with_state(
    initial_doc: &Document,
    step_names: &[&str],
    state: Option<&State>,
    model_name: &str,
    profile_name: &str,
    required_bundles: &[&str],
    options: StepOptions<'_>,
) -> Result<Analyzer, Failure> {
    let StepOptions { redaction, open_span, relations, relation_model, vocabulary } = options;
    let redacts = step_names.iter().any(|&s| block5::is_redact_step(s));
    if redaction.is_some() && !redacts {
        return Err(bad("pii_redaction is set but the steps do not include turbo-pii:redact"));
    }
    // A step named by a legacy layer id runs as the step its current id names.
    let canonical: Vec<_> = step_names.iter().map(|&s| turbo_analysis::ids::canonical_layer_name(s)).collect();
    let step_vec: Vec<&str> = canonical.iter().map(|s| s.as_ref()).collect();
    let wants_gliner = step_vec.contains(&"turbo:gliner");
    let gliner = if wants_gliner {
        let schema = open_span.ok_or_else(|| bad("turbo:gliner requires open_span labels"))?;
        Some((gliner::open(state, model_name)?, schema))
    } else {
        if open_span.is_some() {
            return Err(bad("open_span is set but the steps do not include turbo:gliner"));
        }
        None
    };
    let wants_relations = step_vec.contains(&"turbo:zero-shot-relations");
    let relation = if wants_relations {
        let config = relations.ok_or_else(|| bad("turbo:zero-shot-relations requires relation labels"))?;
        Some((relations::open(state, relation_model)?, config))
    } else {
        if relations.is_some() || !relation_model.trim().is_empty() {
            return Err(bad("relations are set but the steps do not include turbo:zero-shot-relations"));
        }
        None
    };
    let embed_cx =
        resolve_embed_context(state, model_name, profile_name, required_bundles, &step_vec, gliner.is_some())?;
    let vocab = block5::resolve_vocabulary(
        state,
        model_name,
        relation_model,
        embed_cx.as_ref().map(|cx| &*cx.loaded),
        vocabulary,
    )?;
    compile_pipeline_with(initial_doc, step_names, embed_cx, gliner, relation, profile_name, redaction, &vocab)
}

pub(crate) fn compile_pipeline(
    initial_doc: &Document,
    step_names: &[&str],
    embed_cx: Option<EmbedContext>,
    profile_name: &str,
    redaction: Option<&Redaction>,
    vocabulary: &ResolvedVocabulary,
) -> Result<Analyzer, Failure> {
    compile_pipeline_with(initial_doc, step_names, embed_cx, None, None, profile_name, redaction, vocabulary)
}

#[allow(clippy::too_many_arguments)]
fn compile_pipeline_with(
    initial_doc: &Document,
    step_names: &[&str],
    mut embed_cx: Option<EmbedContext>,
    gliner: Option<(std::sync::Arc<turbo_gliner::Model>, ExtractionSchema)>,
    relation: Option<(std::sync::Arc<turbo_relation::Gliner2Relations>, ZeroShotRelationConfig)>,
    profile_name: &str,
    redaction: Option<&Redaction>,
    vocabulary: &ResolvedVocabulary,
) -> Result<Analyzer, Failure> {
    let redacts = step_names.iter().any(|&s| block5::is_redact_step(s));
    if redaction.is_some() && !redacts {
        return Err(bad("pii_redaction is set but the steps do not include turbo-pii:redact"));
    }
    let canonical: Vec<_> = step_names.iter().map(|&s| turbo_analysis::ids::canonical_layer_name(s)).collect();
    let mut step_vec: Vec<&str> = canonical.iter().map(|s| s.as_ref()).collect();
    block5::append_bound_glossary(&mut step_vec, vocabulary);
    let step_names = step_vec.as_slice();
    let step_names = &heads::expand(step_names, embed_cx.as_ref())?[..];
    let who = if profile_name.is_empty() { "pipeline".to_string() } else { format!("profile '{profile_name}'") };
    let head_plan = heads::plan(embed_cx.as_ref(), step_names, &who)?;
    let total_norms = step_names.iter().copied().filter(|&s| is_normalizer_step(s)).count();
    if total_norms > NORM_VIEWS.len() {
        return Err(bad(format!("pipeline exceeds maximum chained normalizer steps ({})", NORM_VIEWS.len())));
    }
    let has_structure_project = step_names.iter().copied().any(is_structure_project_step);
    let has_structure_blocks = step_names.iter().copied().any(is_structure_blocks_step);

    let mut builder = register_given_layers(Analyzer::builder(), initial_doc);
    let mut norm_index = 0usize;
    let mut norm_chain: Vec<&'static str> = Vec::new();
    let mut active_norm_view: Option<&'static str> = None;

    for &name in step_names {
        if name == "turbo:gliner" {
            let (model, schema) = gliner.as_ref().expect("GLiNER step resolved");
            builder = builder.step(gliner::GlinerStep::new(model.clone(), schema.clone()));
            continue;
        }
        if name == "turbo:zero-shot-relations" {
            let (model, config) = relation.as_ref().expect("relation step resolved");
            builder = relations::add(builder, model.clone(), config.clone())?;
            continue;
        }
        if is_normalizer_step(name) {
            let read_v = if norm_index == 0 { "" } else { NORM_VIEWS[norm_index - 1] };
            let write_v = if norm_index + 1 == total_norms { "normalized" } else { NORM_VIEWS[norm_index] };
            norm_index += 1;
            norm_chain.push(write_v);
            active_norm_view = Some(write_v);
            builder = match name {
                "turbo:normalize" | "turbo:normalize:chain" | "normalize" => add_norm(
                    builder,
                    Chain::new().then(Invisible).then(UnicodeForm::nfkc()).then(Whitespace::keep_line_breaks()),
                    read_v,
                    write_v,
                ),
                "turbo:normalize:unicode-form" | "turbo:normalize:nfkc" | "unicode_normalize" => {
                    add_norm(builder, UnicodeForm::nfkc(), read_v, write_v)
                }
                "turbo:normalize:nfc" => add_norm(builder, UnicodeForm::nfc(), read_v, write_v),
                "turbo:normalize:case-fold" | "case_fold" => add_norm(builder, CaseFold::full(), read_v, write_v),
                "turbo:normalize:whitespace" | "whitespace" => {
                    add_norm(builder, Whitespace::keep_paragraphs(), read_v, write_v)
                }
                "turbo:normalize:invisible" | "turbo:normalize:clean-text" | "clean_text" => {
                    add_norm(builder, Invisible, read_v, write_v)
                }
                _ => unreachable!(),
            };
            continue;
        }

        let (next_builder, handled) =
            block5::add_step(builder, name, embed_cx.as_ref(), &norm_chain, step_names, redaction, vocabulary)?;
        let (next_builder, handled) = match handled {
            true => (next_builder, true),
            false => heads::add_step(next_builder, name, head_plan.as_ref()),
        };
        builder = next_builder;
        if handled {
            continue;
        }

        builder = match name {
            "turbo-structure:blocks" | "turbo-structure:structure" | "turbo:structure" => {
                let ann = StructureAnnotator::new(Syntax::Markdown);
                if has_structure_project && total_norms > 0 {
                    builder.step(Bound::new(ann).write(0, "source"))
                } else if let Some(nv) = active_norm_view {
                    builder.step(NormalizedSourceStep { annotator: ann, norm_view: nv, _marker: PhantomData })
                } else {
                    builder.add(ann)
                }
            }
            "turbo-structure:project" | "structure_project" => {
                if norm_chain.is_empty() {
                    return Err(bad("turbo-structure:project requires a preceding normalizer step"));
                }
                let (src_name, dst_name) = if has_structure_blocks { ("source", "") } else { ("", "normalized") };
                builder.step(NormalizedStructureProjectStep {
                    norm_chain: norm_chain.clone(),
                    source_name: src_name,
                    target_name: dst_name,
                })
            }
            "turbo-sentdetect:rules" | "turbo:sentences" => {
                let ann = RuleSentenceDetector::new("eng");
                if let Some(nv) = active_norm_view {
                    builder.step(NormalizedSourceStep { annotator: ann, norm_view: nv, _marker: PhantomData })
                } else {
                    builder.add(ann)
                }
            }
            "turbo-tokenize:uax29-untyped" | "turbo:tokens" => {
                let ann = UntypedWordTokenizer::default();
                if head_plan.as_ref().is_some_and(|p| p.learned_sentences) {
                    // The learned sentences read the tokens, not the other way.
                    builder.step(Bound::new(ann).read(1, heads::UNREAD_SENTENCES))
                } else if let Some(nv) = active_norm_view {
                    builder.step(NormalizedTokensStep { annotator: ann, norm_view: nv })
                } else {
                    builder.add(ann)
                }
            }
            "turbo-geo:names" | "turbo:entities" => {
                let ann = NameAnnotator::new(bundled_name_finder());
                if let Some(nv) = active_norm_view {
                    builder.step(NormalizedNameStep { annotator: ann, norm_view: nv })
                } else {
                    builder.add(ann)
                }
            }
            "turbo:subwords" | "subwords" => {
                if let Some(cx) = embed_cx.as_ref() {
                    let ann = cx.subword_annotator()?;
                    if let Some(nv) = active_norm_view {
                        builder.step(NormalizedSubwordsStep { annotator: ann, norm_view: nv })
                    } else {
                        builder.add(ann)
                    }
                } else {
                    let (model, _) = gliner.as_ref().expect("GLiNER subwords resolved");
                    let ann = turbo_gliner::GlinerSubwords::new(model.clone());
                    if let Some(nv) = active_norm_view {
                        builder.step(NormalizedSubwordsStep { annotator: ann, norm_view: nv })
                    } else {
                        builder.add(ann)
                    }
                }
            }
            "turbo:chunker" | "turbo:chunks" => {
                let budget = if let Some(cx) = embed_cx.as_mut() {
                    let va = cx.vector_annotator()?;
                    let max = va.width();
                    let target = (max / 2).max(1);
                    Budget { max, target, size_weight: target.saturating_mul(2), overhead: va.overhead() }
                } else {
                    Budget { max: 256, target: 128, size_weight: 256, overhead: 0 }
                };
                builder.add(Chunker { budget })
            }
            "turbo:sentence-vectors" | "sentence_vectors" => {
                let va = embed_cx.as_mut().expect("embed context resolved").vector_annotator()?;
                builder.add(SentenceVectors(va))
            }
            "turbo:chunk-vectors" | "chunk_vectors" => {
                let va = embed_cx.as_mut().expect("embed context resolved").vector_annotator()?;
                builder.add(ChunkVectors(va))
            }
            "turbo:sentence-chunk-vectors" | "sentence_chunk_vectors" => {
                let va = embed_cx.as_mut().expect("embed context resolved").vector_annotator()?;
                builder.add(SentenceChunkVectors(va))
            }
            "turbo:subword-vectors" | "subword_vectors" => {
                let va = embed_cx.as_mut().expect("embed context resolved").vector_annotator()?;
                builder.add(SubwordVectors(va))
            }
            "turbo:level-vectors" | "level_vectors" => {
                let va = embed_cx.as_mut().expect("embed context resolved").vector_annotator()?;
                builder.step(LevelVectors(va).bound())
            }
            "turbo:sparse" | "turbo:sparse-terms" | "sparse_terms" => {
                let sa = embed_cx.as_ref().expect("embed context resolved").sparse_annotator()?;
                builder.add(sa)
            }
            unknown => return Err(bad(format!("unknown pipeline step '{unknown}'"))),
        };
    }

    builder.build().map_err(|e| bad(format!("invalid pipeline DAG: {e}")))
}
