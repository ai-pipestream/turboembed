//! The capability listing: what every loaded bundle is and takes, so a
//! client rejects an incompatible request before sending it
//! (docs/grpc-analysis.md), and the LoadBundle request as a
//! configuration.

use crate::api::{self, Failure, TURBO_E_INVALID_ARGUMENT, TURBO_E_INVALID_ENUM};
use crate::config::{Device, ModelConfig};
use crate::proto::result::{BundleCapabilities, LoadBundleRequest, ModelRef, ResultLimits, ServerLimits};
use crate::results::{
    MAX_ANSWER_FRAGMENTS, MAX_ANSWERS, MAX_CHUNKS, MAX_FRAGMENTS, MAX_GRAPH_BYTES, MAX_LEVELS, MAX_RERANK_ITEMS,
    MAX_SPARSE_ENTRIES, MAX_SPARSE_OCCURRENCES, MAX_TOKEN_VECTOR_BYTES, MAX_WINDOWS, over,
};
use crate::state::{Loaded, MAX_LOADED_MODELS, MAX_SESSIONS_PER_MODEL};

fn bad(message: impl Into<String>) -> Failure {
    Failure::new(TURBO_E_INVALID_ARGUMENT, 0, message)
}

/// What one result message may carry, as the capability listing shows it.
pub(crate) fn result_limits() -> ResultLimits {
    ResultLimits {
        max_sparse_entries: MAX_SPARSE_ENTRIES,
        max_sparse_occurrences: MAX_SPARSE_OCCURRENCES,
        max_token_vector_bytes: MAX_TOKEN_VECTOR_BYTES,
        max_rerank_items: MAX_RERANK_ITEMS,
        max_answers: MAX_ANSWERS,
        max_answer_fragments: MAX_ANSWER_FRAGMENTS,
        max_windows: MAX_WINDOWS,
        max_chunks: MAX_CHUNKS,
        max_levels: MAX_LEVELS,
        max_fragments: MAX_FRAGMENTS,
        max_graph_bytes: MAX_GRAPH_BYTES,
    }
}

/// What LoadBundle may ask for, as the capability listing shows it.
pub(crate) fn server_limits() -> ServerLimits {
    ServerLimits {
        max_sessions_per_model: MAX_SESSIONS_PER_MODEL,
        max_loaded_models: MAX_LOADED_MODELS as u32,
        stream: Some(crate::stream::stream_limits()),
    }
}

/// What a loaded bundle is and takes: its name, identity, task, roles,
/// widths and the limits a request is refused over, from the library's
/// own facts. The limits are the session's, what a request is held to;
/// a server started below the model's limits lists the smaller ones.
pub(crate) fn bundle_capabilities(name: &str, loaded: &Loaded) -> api::Result<BundleCapabilities> {
    let mi = loaded.model.info()?;
    let si = loaded.slots[0].info;
    let prefix_query = crate::api::field(&mi.prefix_query);
    let prefix_document = crate::api::field(&mi.prefix_document);
    let mut roles = Vec::new();
    if !prefix_query.is_empty() {
        roles.push("PROMPT_QUERY".to_string());
    }
    if !prefix_document.is_empty() {
        roles.push("PROMPT_DOCUMENT".to_string());
    }
    if roles.is_empty() {
        roles.push("PROMPT_NONE".to_string());
    }
    let mut widths = if mi.task == turbo::TURBO_TASK_EMBED { vec![mi.dim] } else { Vec::new() };
    widths.extend(mi.output_dims.iter().take((mi.output_dims_count as usize).min(mi.output_dims.len())));
    Ok(BundleCapabilities {
        name: name.to_string(),
        model: Some(ModelRef {
            id: crate::api::field(&mi.model_id),
            revision: crate::api::field(&mi.revision),
            sha256: crate::api::field(&mi.manifest_sha256),
        }),
        task: crate::api::names::task(mi.task),
        roles,
        widths,
        context_limit: si.max_seq,
        batch_limit: si.max_batch,
        // No merged task reads an extraction schema yet (issue #715); a
        // bundle that does will fill this.
        schema: None,
    })
}

/// The LoadBundle request as the configuration the loader takes,
/// refusing what a request cannot mean. The name, when empty, is the
/// bundle path's last component, exactly as a startup `--model` names it.
pub(crate) fn load_config(req: &LoadBundleRequest) -> Result<(String, ModelConfig), Failure> {
    if req.bundle.is_empty() {
        return Err(bad("bundle: the bundle directory is required"));
    }
    let name = if req.name.is_empty() {
        let last = req.bundle.rsplit(std::path::is_separator).next().unwrap_or("");
        if matches!(last, "" | "." | "..") {
            return Err(bad(format!("bundle {}: the path names no bundle", req.bundle)));
        }
        last.to_string()
    } else {
        req.name.clone()
    };
    // The caps are refused before anything is allocated, with the
    // limit named.
    if req.sessions == 0 {
        return Err(bad("sessions: at least 1"));
    }
    if req.sessions > MAX_SESSIONS_PER_MODEL {
        return Err(over("max_sessions_per_model", "sessions", req.sessions as u64));
    }
    if crate::api::names::PRECISION.iter().all(|(v, _)| *v != req.precision) {
        return Err(Failure::new(
            TURBO_E_INVALID_ENUM,
            0,
            format!("precision {}: not PRECISION_MODEL, PRECISION_FASTEST or PRECISION_EXACT", req.precision),
        ));
    }
    let device = if req.select_device {
        if req.device != 0 {
            return Err(bad("device: cannot set both select_device and a non-zero device index"));
        }
        Device::Select
    } else {
        Device::Index(req.device)
    };
    Ok((
        name,
        ModelConfig {
            bundle: req.bundle.clone(),
            name: if req.name.is_empty() { None } else { Some(req.name.clone()) },
            device,
            precision: req.precision,
            max_batch: req.max_batch,
            max_seq: req.max_seq,
            sessions: req.sessions,
        },
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::api::TURBO_E_CAPACITY;
    use crate::state::MAX_LOADED_MODELS;

    fn req(sessions: u32) -> LoadBundleRequest {
        LoadBundleRequest {
            name: "x".into(),
            bundle: "b/x".into(),
            device: 0,
            sessions,
            precision: 0,
            max_batch: 0,
            max_seq: 0,
            select_device: false,
        }
    }

    #[test]
    fn sessions_are_capped_before_anything_is_allocated() {
        let (name, config) = load_config(&req(MAX_SESSIONS_PER_MODEL)).unwrap();
        assert_eq!(name, "x");
        assert_eq!(config.sessions, MAX_SESSIONS_PER_MODEL);
        let e = load_config(&req(MAX_SESSIONS_PER_MODEL + 1)).unwrap_err();
        assert_eq!(e.code, TURBO_E_CAPACITY);
        assert!(e.message.contains("max_sessions_per_model"), "{e:?}");
        assert!(e.message.contains(&(MAX_SESSIONS_PER_MODEL + 1).to_string()), "{e:?}");
        let e = load_config(&req(0)).unwrap_err();
        assert_eq!(e.code, TURBO_E_INVALID_ARGUMENT);
    }

    #[test]
    fn select_device_maps_to_select_and_rejects_non_zero_device() {
        let mut r = req(1);
        r.select_device = true;
        let (_, config) = load_config(&r).unwrap();
        assert_eq!(config.device, Device::Select);

        r.device = 2;
        let e = load_config(&r).unwrap_err();
        assert_eq!(e.code, TURBO_E_INVALID_ARGUMENT);
        assert!(e.message.contains("select_device"), "{e:?}");
    }

    #[test]
    fn the_listing_shows_the_server_limits() {
        let l = server_limits();
        assert_eq!(l.max_sessions_per_model, MAX_SESSIONS_PER_MODEL);
        assert_eq!(l.max_loaded_models, MAX_LOADED_MODELS as u32);
    }
}
