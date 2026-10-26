//! Built-in first-party [`turbo_analysis::Annotator`] step registration for
//! `AnalyzeDocumentProgressive` (issues `#1218`, `#1302`, `#1325`).

use std::marker::PhantomData;
use std::path::Path;
use std::sync::{Arc, OnceLock};

use turbo::{TURBO_PRECISION_EXACT, TURBO_PRECISION_FASTEST, TURBO_TASK_EMBED};
use turbo_abbrev::AbbreviationAnnotator;
use turbo_analysis::analyzer::Builder;
use turbo_analysis::{Analyzer, Bound, Document, Key};
use turbo_analysis_embed::{
    ChunkVectors, Device, EncoderOptions, LevelVectors, Precision, SentenceChunkVectors, SentenceVectors,
    SparseAnnotator, SparseOptions, SubwordAnnotator, SubwordVectors, VectorAnnotator, VectorOptions,
};
use turbo_analysis_layers::{
    Boundaries, Chunks, Entities, Levels, Sentences, Stems, Structure, Subwords, Terms, Tokens,
};
use turbo_chunker::levels::LevelBuilder;
use turbo_chunker::terms::TermIndexer;
use turbo_chunker::{Budget, Chunker, Costs, LexicalBoundaries, StructureBoundaries};
use turbo_geo::{BundledGazetteer, NameAnnotator, NameFinder};
use turbo_keywords::{Stopwords as KeywordStopwords, YakeAnnotator};
use turbo_normalize::{Chain, Invisible, NormalizeStep, Normalizer, UnicodeForm, Whitespace};
use turbo_sentdetect::RuleSentenceDetector;
use turbo_stem::{Algorithm, StemAnnotator, TermAnnotator};
use turbo_stopword::{ListChoice, StopwordAnnotator, StopwordLanguage};
use turbo_stopword_layers::Stopwords;
use turbo_structure::{StructureAnnotator, Syntax};
use turbo_tokenize::uax29::UntypedWordTokenizer;

use super::normalize::{
    NormalizedNameStep, NormalizedSourceStep, NormalizedStructureProjectStep, NormalizedSubwordsStep,
    NormalizedTokensStep,
};
use crate::api::{Failure, TURBO_E_BUNDLE_NOT_FOUND, TURBO_E_INVALID_ARGUMENT};
use crate::state::{Loaded, State};

pub(super) const DEFAULT_STEPS: &[&str] = &[
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

pub(super) const MINIMAL_STEPS: &[&str] = &["turbo-sentdetect:rules", "turbo-tokenize:uax29-untyped"];

pub(super) const CHUNKING_STEPS: &[&str] = &[
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
    step_needs_sparse(name)
        || matches!(
            name,
            "turbo:subwords"
                | "opennlp:subwords"
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
    if initial_doc.present(Key::<Structure>::DEFAULT.id()) {
        builder = builder.given(Key::<Structure>::DEFAULT);
    }
    if initial_doc.present(Key::<Sentences>::DEFAULT.id()) {
        builder = builder.given(Key::<Sentences>::DEFAULT);
    }
    if initial_doc.present(Key::<Tokens>::DEFAULT.id()) {
        builder = builder.given(Key::<Tokens>::DEFAULT);
    }
    if initial_doc.present(Key::<Entities>::DEFAULT.id()) {
        builder = builder.given(Key::<Entities>::DEFAULT);
    }
    if initial_doc.present(Key::<Subwords>::DEFAULT.id()) {
        builder = builder.given(Key::<Subwords>::DEFAULT);
    }
    if initial_doc.present(Key::<Terms>::DEFAULT.id()) {
        builder = builder.given(Key::<Terms>::DEFAULT);
    }
    if initial_doc.present(Key::<Stems>::DEFAULT.id()) {
        builder = builder.given(Key::<Stems>::DEFAULT);
    }
    if initial_doc.present(Key::<Stopwords>::DEFAULT.id()) {
        builder = builder.given(Key::<Stopwords>::DEFAULT);
    }
    if initial_doc.present(Key::<Boundaries>::DEFAULT.id()) {
        builder = builder.given(Key::<Boundaries>::DEFAULT);
    }
    if initial_doc.present(Key::<Chunks>::DEFAULT.id()) {
        builder = builder.given(Key::<Chunks>::DEFAULT);
    }
    if initial_doc.present(Key::<Levels>::DEFAULT.id()) {
        builder = builder.given(Key::<Levels>::DEFAULT);
    }
    builder
}

struct EmbedContext {
    loaded: Arc<Loaded>,
    vectors: Option<Arc<VectorAnnotator>>,
}

impl EmbedContext {
    fn encoder_options(&self) -> EncoderOptions {
        let slot = &self.loaded.slots[0];
        let precision = match slot.desc.precision {
            TURBO_PRECISION_FASTEST => Precision::Fastest,
            TURBO_PRECISION_EXACT => Precision::Exact,
            _ => Precision::Model,
        };
        EncoderOptions {
            device: Device::Index(self.loaded.device),
            precision,
            max_batch: slot.info.max_batch,
            max_seq: slot.info.max_seq,
            max_ranges: 0,
            cpu_threads: 0,
        }
    }

    fn subword_annotator(&self) -> Result<SubwordAnnotator, Failure> {
        SubwordAnnotator::open(Path::new(&self.loaded.bundle)).map_err(|e| Failure::new(e.code, e.field, e.message))
    }

    fn vector_annotator(&mut self) -> Result<Arc<VectorAnnotator>, Failure> {
        if let Some(v) = &self.vectors {
            return Ok(Arc::clone(v));
        }
        let opts = VectorOptions { encoder: self.encoder_options(), ..VectorOptions::default() };
        let ann = VectorAnnotator::open(Path::new(&self.loaded.bundle), &opts)
            .map_err(|e| Failure::new(e.code, e.field, e.message))?;
        let arc = Arc::new(ann);
        self.vectors = Some(Arc::clone(&arc));
        Ok(arc)
    }

    fn sparse_annotator(&self) -> Result<SparseAnnotator, Failure> {
        let opts = SparseOptions { encoder: self.encoder_options(), ..SparseOptions::default() };
        SparseAnnotator::open(Path::new(&self.loaded.bundle), &opts)
            .map_err(|e| Failure::new(e.code, e.field, e.message))
    }
}

fn resolve_embed_context(
    state: Option<&State>,
    model_name: &str,
    profile_name: &str,
    required_bundles: &[&str],
    step_names: &[&str],
) -> Result<Option<EmbedContext>, Failure> {
    let needs_sparse = required_bundles.contains(&"embed.sparse") || step_names.iter().copied().any(step_needs_sparse);
    let needs_embed =
        needs_sparse || required_bundles.contains(&"TASK_EMBED") || step_names.iter().copied().any(step_needs_embed);

    if !needs_embed {
        if !model_name.trim().is_empty() {
            return Err(bad(format!(
                "unknown model bundle {:?} for built-in progressive pipeline; leave model empty",
                model_name
            )));
        }
        return Ok(None);
    }

    let who = if profile_name.is_empty() { "pipeline".to_string() } else { format!("profile '{profile_name}'") };
    let Some(st) = state else {
        return Err(Failure::new(
            TURBO_E_BUNDLE_NOT_FOUND,
            0,
            format!("{who} requires bundle 'TASK_EMBED', which is not loaded"),
        ));
    };

    let trimmed = model_name.trim();
    if !trimmed.is_empty() {
        let (_, loaded) = st
            .ready(trimmed, "")
            .map_err(|f| Failure::new(f.code, f.field, format!("{} (required bundle 'TASK_EMBED')", f.message)))?;
        if loaded.task != TURBO_TASK_EMBED {
            return Err(Failure::new(
                TURBO_E_BUNDLE_NOT_FOUND,
                0,
                format!("{who} requires bundle 'TASK_EMBED', which model '{trimmed}' does not provide"),
            ));
        }
        if needs_sparse {
            let embed = loaded.embed().map_err(|f| {
                Failure::new(f.code, f.field, format!("{} (required bundle 'embed.sparse')", f.message))
            })?;
            if embed.mlm.is_none() {
                return Err(Failure::new(
                    TURBO_E_BUNDLE_NOT_FOUND,
                    0,
                    format!("{who} requires bundle 'embed.sparse', which model '{trimmed}' does not provide"),
                ));
            }
        }
        return Ok(Some(EmbedContext { loaded, vectors: None }));
    }

    let embed_models: Vec<Arc<Loaded>> = st
        .listing()
        .into_iter()
        .filter_map(|m| m.loaded.get().cloned())
        .filter(|l| l.task == TURBO_TASK_EMBED)
        .collect();
    if embed_models.is_empty() {
        return Err(Failure::new(
            TURBO_E_BUNDLE_NOT_FOUND,
            0,
            format!("{who} requires bundle 'TASK_EMBED', which is not loaded"),
        ));
    }

    if needs_sparse {
        for loaded in embed_models {
            if loaded.embed().is_ok_and(|e| e.mlm.is_some()) {
                return Ok(Some(EmbedContext { loaded, vectors: None }));
            }
        }
        return Err(Failure::new(
            TURBO_E_BUNDLE_NOT_FOUND,
            0,
            format!("{who} requires bundle 'embed.sparse', which is not loaded"),
        ));
    }

    Ok(Some(EmbedContext { loaded: embed_models.into_iter().next().expect("non-empty"), vectors: None }))
}

fn add_norm<N: Normalizer>(builder: Builder, n: N, read_view: &'static str, write_view: &'static str) -> Builder {
    builder.step(Bound::new(NormalizeStep::new(n)).read(0, read_view).write(0, write_view))
}

/// Builds an [`Analyzer`] from `initial_doc` and the requested `step_names`
/// without model state.
pub fn compile_analyzer(initial_doc: &Document, step_names: &[&str]) -> Result<Analyzer, Failure> {
    compile_analyzer_with_state(initial_doc, step_names, None, "", "", &[])
}

/// Builds an [`Analyzer`] from `initial_doc`, resolving model-backed steps and
/// required profile bundles against `state`.
pub(crate) fn compile_analyzer_with_state(
    initial_doc: &Document,
    step_names: &[&str],
    state: Option<&State>,
    model_name: &str,
    profile_name: &str,
    required_bundles: &[&str],
) -> Result<Analyzer, Failure> {
    let mut embed_cx = resolve_embed_context(state, model_name, profile_name, required_bundles, step_names)?;
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
            "turbo-sentdetect:rules" | "turbo:sentences" | "opennlp:sentences" => {
                let ann = RuleSentenceDetector::new("eng");
                if let Some(nv) = active_norm_view {
                    builder.step(NormalizedSourceStep { annotator: ann, norm_view: nv, _marker: PhantomData })
                } else {
                    builder.add(ann)
                }
            }
            "turbo-tokenize:uax29-untyped" | "turbo:tokens" | "opennlp:tokens" => {
                let ann = UntypedWordTokenizer::default();
                if let Some(nv) = active_norm_view {
                    builder.step(NormalizedTokensStep { annotator: ann, norm_view: nv })
                } else {
                    builder.add(ann)
                }
            }
            "turbo-geo:names" | "turbo:entities" | "opennlp:entities" => {
                let ann = NameAnnotator::new(bundled_name_finder());
                if let Some(nv) = active_norm_view {
                    builder.step(NormalizedNameStep { annotator: ann, norm_view: nv })
                } else {
                    builder.add(ann)
                }
            }
            "turbo:subwords" | "opennlp:subwords" | "subwords" => {
                let cx = embed_cx.as_ref().expect("embed context resolved");
                let ann = cx.subword_annotator()?;
                if let Some(nv) = active_norm_view {
                    builder.step(NormalizedSubwordsStep { annotator: ann, norm_view: nv })
                } else {
                    builder.add(ann)
                }
            }
            "turbo-abbrev:schwartz-hearst" | "turbo-abbrev:abbreviations" | "turbo:abbreviations" => {
                builder.add(AbbreviationAnnotator)
            }
            "turbo-stem:terms" | "turbo:terms" => builder.add(TermAnnotator::default()),
            "turbo-stem:stemmer" | "turbo:stems" => builder.add(StemAnnotator::new(Algorithm::Porter)),
            "turbo-stopword:bundled" | "turbo-stopword:stopwords" | "turbo:stopwords" => {
                builder.add(StopwordAnnotator::new(ListChoice::Fixed(StopwordLanguage::English)))
            }
            "turbo-keywords:yake" | "turbo:keywords" => {
                builder.add(YakeAnnotator::new(KeywordStopwords::bundled(StopwordLanguage::English)))
            }
            "turbo:structure-boundaries" | "turbo:boundaries" => {
                builder.add(StructureBoundaries { costs: Costs::default() })
            }
            "turbo:lexical-boundaries" => builder.step(LexicalBoundaries::default().bound()),
            "turbo:term-indexer" | "turbo:term-index" => builder.add(TermIndexer),
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
            "turbo:level-builder" | "turbo:levels" => builder.add(LevelBuilder),
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
