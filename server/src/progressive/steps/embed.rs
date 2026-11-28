use std::sync::Arc;

use turbo::{TURBO_PRECISION_EXACT, TURBO_PRECISION_FASTEST, TURBO_TASK_EMBED};
use turbo_analysis::Bound;
use turbo_analysis::analyzer::Builder;
use turbo_analysis_embed::{
    ChunkVectors, Device, EncoderOptions, LevelVectors, Precision, SentenceChunkVectors, SentenceVectors,
    SparseAnnotator, SubwordAnnotator, SubwordVectors, VectorAnnotator,
};
use turbo_analysis_layers::VectorKind;
use turbo_chunker::{Budget, Chunker};
use turbo_gliner::ExtractionSchema;
use turbo_sat::SatAnnotator;

use super::super::cache::{CallScope, ScopedAnnotator};
use super::super::normalize::NormalizedSubwordsStep;
use super::heads::{self, HeadGroup, HeadStep};
use crate::api::{Failure, TURBO_E_BUNDLE_NOT_FOUND};
use crate::state::{Loaded, Served, State};

pub(crate) struct EmbedContext {
    pub(super) served: Arc<Served>,
    pub(super) loaded: Arc<Loaded>,
    pub(super) scope: CallScope,
}

impl EmbedContext {
    pub(super) fn encoder_options(&self) -> EncoderOptions {
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
            layers: 0,
        }
    }

    pub(super) fn scope<A>(&self, annotator: A) -> ScopedAnnotator<A> {
        ScopedAnnotator::new(annotator, self.scope.clone())
    }

    pub(super) fn subword_annotator(&self) -> Result<Arc<SubwordAnnotator>, Failure> {
        self.served.progressive.subword_annotator(&self.loaded)
    }

    pub(super) fn vector_annotator(&self) -> Result<Arc<VectorAnnotator>, Failure> {
        self.served.progressive.vector_annotator(&self.loaded, &self.encoder_options())
    }

    pub(super) fn sparse_annotator(&self) -> Result<Arc<SparseAnnotator>, Failure> {
        self.served.progressive.sparse_annotator(&self.loaded, &self.encoder_options())
    }

    pub(super) fn sat_annotator(&self) -> Result<Arc<SatAnnotator>, Failure> {
        self.served.progressive.sat_annotator(&self.loaded, &self.encoder_options())
    }

    pub(super) fn head_group(&self, steps: &[(HeadStep, Option<&str>)], who: &str) -> Result<Arc<HeadGroup>, Failure> {
        self.served.progressive.head_group(&self.loaded, &self.encoder_options(), steps, who, self.scope.clone())
    }
}

pub(super) fn add_step(
    builder: Builder,
    name: &str,
    embed_cx: Option<&EmbedContext>,
    active_norm_view: Option<&'static str>,
    gliner: Option<&(Arc<turbo_gliner::Model>, ExtractionSchema)>,
) -> Result<(Builder, bool), Failure> {
    let builder = match name {
        "turbo:subwords" | "subwords" => {
            if let Some(cx) = embed_cx {
                let ann = cx.subword_annotator()?;
                if let Some(nv) = active_norm_view {
                    builder.step(NormalizedSubwordsStep { annotator: ann, norm_view: nv })
                } else {
                    builder.add(ann)
                }
            } else {
                let (model, _) = gliner.expect("GLiNER subwords resolved");
                let ann = turbo_gliner::GlinerSubwords::new(model.clone());
                if let Some(nv) = active_norm_view {
                    builder.step(NormalizedSubwordsStep { annotator: ann, norm_view: nv })
                } else {
                    builder.add(ann)
                }
            }
        }
        "turbo:chunker" | "turbo:chunks" => {
            let budget = if let Some(cx) = embed_cx {
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
            let cx = embed_cx.expect("embed context resolved");
            builder.add(cx.scope(SentenceVectors(cx.vector_annotator()?)))
        }
        "turbo:chunk-vectors" | "chunk_vectors" => {
            let cx = embed_cx.expect("embed context resolved");
            builder.add(cx.scope(ChunkVectors(cx.vector_annotator()?)))
        }
        "turbo:sentence-chunk-vectors" | "sentence_chunk_vectors" => {
            let cx = embed_cx.expect("embed context resolved");
            builder.add(cx.scope(SentenceChunkVectors(cx.vector_annotator()?)))
        }
        "turbo:subword-vectors" | "subword_vectors" => {
            let cx = embed_cx.expect("embed context resolved");
            builder.add(cx.scope(SubwordVectors(cx.vector_annotator()?)))
        }
        "turbo:level-vectors" | "level_vectors" => {
            let cx = embed_cx.expect("embed context resolved");
            let step = cx.scope(LevelVectors(cx.vector_annotator()?));
            builder.step(Bound::new(step).write(0, VectorKind::Local.name()).write(1, VectorKind::Contextual.name()))
        }
        "turbo:sparse" | "turbo:sparse-terms" | "sparse_terms" => {
            let cx = embed_cx.expect("embed context resolved");
            builder.add(cx.scope(cx.sparse_annotator()?))
        }
        _ => return Ok((builder, false)),
    };
    Ok((builder, true))
}

#[allow(clippy::too_many_arguments)]
pub(super) fn resolve_embed_context(
    state: Option<&State>,
    model_name: &str,
    profile_name: &str,
    required_bundles: &[&str],
    step_names: &[&str],
    other_model: bool,
    scope: CallScope,
) -> Result<Option<EmbedContext>, Failure> {
    let needs_sparse =
        required_bundles.contains(&"embed.sparse") || step_names.iter().copied().any(super::step_needs_sparse);
    let needs_embed = needs_sparse
        || required_bundles.contains(&"TASK_EMBED")
        || step_names
            .iter()
            .copied()
            .filter(|name| !other_model || *name != "turbo:subwords")
            .any(super::step_needs_embed);

    if !needs_embed {
        let trimmed = model_name.trim();
        if !trimmed.is_empty() && !other_model {
            let has_vocab =
                state.and_then(|st| st.ready_served(trimmed).ok()).is_some_and(|s| !s.vocabulary.is_empty());
            if !has_vocab {
                return Err(super::bad(format!(
                    "unknown model bundle {:?} for built-in progressive pipeline; leave model empty",
                    model_name
                )));
            }
        }
        return Ok(None);
    }

    let who = if profile_name.is_empty() { "pipeline".to_string() } else { format!("profile '{profile_name}'") };
    let needed_heads = heads::needed(step_names, required_bundles);
    let any_head = required_bundles.contains(&heads::ANY_HEAD);
    let Some(st) = state else {
        return Err(Failure::new(
            TURBO_E_BUNDLE_NOT_FOUND,
            0,
            format!("{who} requires bundle 'TASK_EMBED', which is not loaded"),
        ));
    };

    let trimmed = model_name.trim();
    if !trimmed.is_empty() {
        let (served, loaded) = st
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
        heads::check_model(&loaded, &who, trimmed, &needed_heads, any_head)?;
        return Ok(Some(EmbedContext { served, loaded, scope }));
    }

    let served_embed: Vec<(Arc<Served>, Arc<Loaded>)> = st
        .listing()
        .into_iter()
        .filter_map(|m| m.loaded.get().cloned().filter(|l| l.task == TURBO_TASK_EMBED).map(|l| (m, l)))
        .collect();
    if served_embed.is_empty() {
        return Err(Failure::new(
            TURBO_E_BUNDLE_NOT_FOUND,
            0,
            format!("{who} requires bundle 'TASK_EMBED', which is not loaded"),
        ));
    }
    let embed_models: Vec<Arc<Loaded>> = served_embed.iter().map(|(_, l)| Arc::clone(l)).collect();
    let selected = heads::select(embed_models, &who, &needed_heads, any_head)?;
    let find_served = |target: &Arc<Loaded>| {
        served_embed.iter().find(|(_, l)| Arc::ptr_eq(l, target)).expect("selected from served_embed").0.clone()
    };

    if needs_sparse {
        for loaded in selected {
            if loaded.embed().is_ok_and(|e| e.mlm.is_some()) {
                let served = find_served(&loaded);
                return Ok(Some(EmbedContext { served, loaded, scope }));
            }
        }
        return Err(Failure::new(
            TURBO_E_BUNDLE_NOT_FOUND,
            0,
            format!("{who} requires bundle 'embed.sparse', which is not loaded"),
        ));
    }

    let loaded = selected.into_iter().next().expect("non-empty");
    let served = find_served(&loaded);
    Ok(Some(EmbedContext { served, loaded, scope }))
}
