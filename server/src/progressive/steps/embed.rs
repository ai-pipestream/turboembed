use std::path::Path;
use std::sync::Arc;

use turbo::{TURBO_PRECISION_EXACT, TURBO_PRECISION_FASTEST, TURBO_TASK_EMBED};
use turbo_analysis_embed::{
    Device, EncoderOptions, Precision, SparseAnnotator, SparseOptions, SubwordAnnotator, VectorAnnotator, VectorOptions,
};

use super::heads;
use crate::api::{Failure, TURBO_E_BUNDLE_NOT_FOUND};
use crate::state::{Loaded, State};

pub(crate) struct EmbedContext {
    pub(super) loaded: Arc<Loaded>,
    vectors: Option<Arc<VectorAnnotator>>,
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

    pub(super) fn subword_annotator(&self) -> Result<SubwordAnnotator, Failure> {
        SubwordAnnotator::open(Path::new(&self.loaded.bundle)).map_err(|e| Failure::new(e.code, e.field, e.message))
    }

    pub(super) fn vector_annotator(&mut self) -> Result<Arc<VectorAnnotator>, Failure> {
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

    pub(super) fn sparse_annotator(&self) -> Result<SparseAnnotator, Failure> {
        let opts = SparseOptions { encoder: self.encoder_options(), ..SparseOptions::default() };
        SparseAnnotator::open(Path::new(&self.loaded.bundle), &opts)
            .map_err(|e| Failure::new(e.code, e.field, e.message))
    }
}

pub(super) fn resolve_embed_context(
    state: Option<&State>,
    model_name: &str,
    profile_name: &str,
    required_bundles: &[&str],
    step_names: &[&str],
    other_model: bool,
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
        if !model_name.trim().is_empty() && !other_model {
            return Err(super::bad(format!(
                "unknown model bundle {:?} for built-in progressive pipeline; leave model empty",
                model_name
            )));
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
        heads::check_model(&loaded, &who, trimmed, &needed_heads, any_head)?;
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
    let embed_models = heads::select(embed_models, &who, &needed_heads, any_head)?;

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
