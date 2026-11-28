//! A query text placed in a `ProjectChunks` projection (#2249, #2609):
//! embedded by the served model that made the stored vectors (waiting on its
//! session pool up to the request deadline), laid on the projection's own
//! mean and components, and compared with every projected item.

use std::sync::Arc;

use turbo::{TURBO_PROMPT_QUERY, TURBO_TASK_EMBED};
use turbo_analysis::math::{dot_f64, sum_sq_f64};
use turbo_analysis::{Ix, Layer, ModelRef};
use turbo_analysis_layers::Vectors;
use turbo_chunker::VectorProjection;

use super::pca::{MAX_PROJECT_DIMS, MAX_PROJECT_ROWS, validate_finite_vectors};
use super::project::{finish_project, inspect_query_space, prepare_project, project_chunks_under};
use crate::api::{
    Failure, TURBO_E_BUNDLE_NOT_FOUND, TURBO_E_INTERNAL, TURBO_E_INVALID_ARGUMENT, TURBO_E_INVALID_STATE, field,
};
use crate::deadline::Deadline;
use crate::priority::Priority;
use crate::proto::result::{ProjectChunksRequest, ProjectChunksResponse, QueryPointProto};
use crate::results::disclosure::PiiOutputPolicy;
use crate::results::over;
use crate::state::{Loaded, State};
use crate::tasks::embed::exec::{embed_f32_batches, prompt_embed_options};

/// Longest query text accepted, in bytes.
pub const MAX_QUERY_BYTES: usize = 8_192;

/// Stored rows within this of length 1 count as L2-normalized.
const UNIT_TOLERANCE: f64 = 1e-3;

/// Vector space identity and normalization detected on the graph's stored vectors.
pub(super) struct QuerySpace {
    pub model: ModelRef,
    pub dim: u32,
    pub unit: bool,
}

/// A query vector and the served model name that made it.
pub(crate) type Embedded = (String, Vec<f32>);

fn bad(message: impl Into<String>) -> Failure {
    Failure::new(TURBO_E_INVALID_ARGUMENT, 0, message)
}

/// Validates `req.query` before reading the document graph.
pub(super) fn check_query_field(req: &ProjectChunksRequest, has_embed_runtime: bool) -> Result<(), Failure> {
    let Some(text) = req.query.as_deref() else { return Ok(()) };
    if text.trim().is_empty() {
        return Err(bad("query: a text to place is required when the field is set"));
    }
    if text.len() > MAX_QUERY_BYTES {
        return Err(over("max_query_bytes", "query bytes", text.len() as u64));
    }
    if !has_embed_runtime {
        return Err(Failure::new(
            TURBO_E_INVALID_STATE,
            0,
            "query: placing a query needs a server with its embedding model loaded",
        ));
    }
    Ok(())
}

fn vector_norms_and_unit<L: Layer>(vectors: &Vectors<L>) -> Result<(Vec<f64>, bool), Failure> {
    let n = vectors.len();
    let norms: Vec<f64> = (0..n)
        .map(|i| {
            let ix = Ix::new(i as u32);
            if vectors.is_empty(ix) { 0.0 } else { sum_sq_f64(vectors.row(ix)).sqrt() }
        })
        .collect();
    let mut present = norms.iter().filter(|&&norm| norm > 0.0).peekable();
    if present.peek().is_none() {
        return Err(bad("query: every stored vector is empty, so there is nothing to place a query among"));
    }
    let unit = present.all(|norm| (norm - 1.0).abs() <= UNIT_TOLERANCE);
    Ok((norms, unit))
}

/// Validates `vectors` against `ProjectChunks` caps and returns its [`QuerySpace`].
pub(super) fn space_for_vectors<L: Layer>(vectors: &Vectors<L>, n: usize) -> Result<QuerySpace, Failure> {
    let d = vectors.dim as usize;
    if n > MAX_PROJECT_ROWS {
        return Err(over("max_project_rows", "projected vector rows", n as u64));
    }
    if d > MAX_PROJECT_DIMS {
        return Err(over("max_project_dims", "projected vector dimensions", d as u64));
    }
    validate_finite_vectors(vectors, n)?;
    if n == 0 || d == 0 {
        return Err(bad("query: the document graph carries no vectors to place a query among"));
    }
    let (_, unit) = vector_norms_and_unit(vectors)?;
    Ok(QuerySpace { model: vectors.model.clone(), dim: vectors.dim, unit })
}

/// Places a pre-embedded query vector into `projection` and computes its cosine
/// similarity to every row of `vectors`.
pub(crate) fn place_query<L: Layer>(
    vectors: &Vectors<L>,
    projection: &VectorProjection<L>,
    embedded: &Embedded,
) -> Result<QueryPointProto, Failure> {
    let (norms, _) = vector_norms_and_unit(vectors)?;
    let (model_name, query) = embedded;
    let at = projection.place(query).map_err(|e| bad(format!("query: {e}")))?;
    let length = sum_sq_f64(query).sqrt();
    let cosines = norms
        .iter()
        .enumerate()
        .map(|(i, &norm)| {
            if norm == 0.0 || length == 0.0 {
                return 0.0;
            }
            (dot_f64(query, vectors.row(Ix::new(i as u32))) / (norm * length)).clamp(-1.0, 1.0) as f32
        })
        .collect();
    Ok(QueryPointProto {
        x: at[0],
        y: at[1],
        z: at.get(2).copied().unwrap_or(0.0),
        cosines,
        model_name: model_name.clone(),
    })
}

fn resolve_embed_model(state: &State, space: &QuerySpace) -> Result<(String, Arc<Loaded>), Failure> {
    let want = &space.model;
    for served in state.listing() {
        let Some(loaded) = served.loaded.get() else { continue };
        let info = &loaded.model_info;
        if loaded.task != TURBO_TASK_EMBED
            || field(&info.model_id) != want.id
            || field(&info.revision) != want.revision
            || field(&info.manifest_sha256) != want.sha256
        {
            continue;
        }
        let extra = &info.output_dims[..(info.output_dims_count as usize).min(info.output_dims.len())];
        if info.dim == space.dim || extra.contains(&space.dim) {
            return Ok((served.name.clone(), loaded.clone()));
        }
    }
    Err(Failure::new(
        TURBO_E_BUNDLE_NOT_FOUND,
        0,
        format!(
            "query: no loaded embedding model is {}@{}#{} at {} dimensions, the model of the graph's vectors",
            want.id, want.revision, want.sha256, space.dim
        ),
    ))
}

/// Runs `ProjectChunks` for `AnalysisService`, acquiring a query embedding
/// session asynchronously with `priority` and `deadline` when `req.query` is
/// set (`#2609`).
pub(crate) async fn project_chunks_served(
    state: Arc<State>,
    req: ProjectChunksRequest,
    policy: PiiOutputPolicy,
    priority: Priority,
    deadline: Deadline,
) -> Result<ProjectChunksResponse, Failure> {
    if req.query.is_none() {
        return tokio::task::spawn_blocking(move || project_chunks_under(&req, policy))
            .await
            .map_err(|e| Failure::new(TURBO_E_INTERNAL, 0, format!("project_chunks task failed: {e}")))?;
    }

    let req = Arc::new(req);
    let req_prep = Arc::clone(&req);
    let state_prep = Arc::clone(&state);
    let (prep, name, loaded, space) = tokio::task::spawn_blocking(move || {
        let prep = prepare_project(&req_prep, policy, true)?;
        let space = inspect_query_space(&prep.doc)?;
        let (name, loaded) = resolve_embed_model(&state_prep, &space)?;
        Ok::<_, Failure>((prep, name, loaded, space))
    })
    .await
    .map_err(|e| Failure::new(TURBO_E_INTERNAL, 0, format!("project_chunks task failed: {e}")))??;

    let held = loaded.acquire_async(priority, &deadline).await?;
    tokio::task::spawn_blocking(move || {
        let opts = prompt_embed_options(TURBO_PROMPT_QUERY, space.unit, loaded.model_info.dim, space.dim);
        let q_text = req.query.as_deref().expect("checked query is Some");
        let _docs = loaded.metrics.track_docs(1, Some(loaded.device));
        let mut vecs = embed_f32_batches(&state, &loaded, held.slot(), &[q_text], &opts, &deadline)?;
        drop(held);
        let embedded = (name, vecs.pop().expect("single query row"));
        finish_project(prep, &req, Some(&embedded))
    })
    .await
    .map_err(|e| Failure::new(TURBO_E_INTERNAL, 0, format!("project_chunks task failed: {e}")))?
}
