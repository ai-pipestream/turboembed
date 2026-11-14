//! A query text placed in a `ProjectChunks` projection (#2249): embedded by
//! the served model that made the stored vectors, laid on the projection's
//! own mean and components, and compared with every projected item.

use turbo::{TURBO_NORMALIZE_L2, TURBO_NORMALIZE_NONE, TURBO_PROMPT_QUERY, TURBO_TASK_EMBED, turbo_text};
use turbo_analysis::math::{dot_f64, sum_sq_f64};
use turbo_analysis::{Ix, Layer, ModelRef};
use turbo_analysis_layers::Vectors;
use turbo_chunker::VectorProjection;

use crate::api::{Failure, TURBO_E_BUNDLE_NOT_FOUND, TURBO_E_INVALID_ARGUMENT, TURBO_E_INVALID_STATE, field};
use crate::proto::result::{ProjectChunksRequest, QueryPointProto};
use crate::results::over;
use crate::state::State;

/// Longest query text accepted, in bytes.
pub const MAX_QUERY_BYTES: usize = 8_192;

/// Stored rows within this of length 1 count as L2-normalized.
const UNIT_TOLERANCE: f64 = 1e-3;

/// What the vectors a query must sit among look like.
pub(crate) struct Space<'a> {
    pub model: &'a ModelRef,
    pub dim: u32,
    pub unit: bool,
}

/// A query vector and the served model name that made it.
pub(crate) type Embedded = (String, Vec<f32>);

/// The server's way to embed a query text for a [`Space`].
pub(crate) type Embed<'a> = dyn Fn(&Space<'_>, &str) -> Result<Embedded, Failure> + 'a;

fn bad(message: impl Into<String>) -> Failure {
    Failure::new(TURBO_E_INVALID_ARGUMENT, 0, message)
}

/// The request's query text with the embedder it needs, checked before the
/// graph is read: `None` when the request carries no query.
pub(crate) fn wanted_query<'a>(
    req: &'a ProjectChunksRequest,
    embed: Option<&'a Embed<'a>>,
) -> Result<Option<(&'a str, &'a Embed<'a>)>, Failure> {
    let Some(text) = req.query.as_deref() else { return Ok(None) };
    if text.trim().is_empty() {
        return Err(bad("query: a text to place is required when the field is set"));
    }
    if text.len() > MAX_QUERY_BYTES {
        return Err(over("max_query_bytes", "query bytes", text.len() as u64));
    }
    let embed = embed.ok_or_else(|| {
        Failure::new(TURBO_E_INVALID_STATE, 0, "query: placing a query needs a server with its embedding model loaded")
    })?;
    Ok(Some((text, embed)))
}

/// Embeds `text` for `vectors` and returns its coordinates in `projection`
/// with its cosine to every row.
pub(crate) fn place_query<L: Layer>(
    vectors: &Vectors<L>,
    projection: &VectorProjection<L>,
    text: &str,
    embed: &Embed<'_>,
) -> Result<QueryPointProto, Failure> {
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
    let (model_name, query) = embed(&Space { model: &vectors.model, dim: vectors.dim, unit }, text)?;
    let at = projection.place(&query).map_err(|e| bad(format!("query: {e}")))?;
    let length = sum_sq_f64(&query).sqrt();
    let cosines = norms
        .iter()
        .enumerate()
        .map(|(i, &norm)| {
            if norm == 0.0 || length == 0.0 {
                return 0.0;
            }
            (dot_f64(&query, vectors.row(Ix::new(i as u32))) / (norm * length)).clamp(-1.0, 1.0) as f32
        })
        .collect();
    Ok(QueryPointProto { x: at[0], y: at[1], z: at.get(2).copied().unwrap_or(0.0), cosines, model_name })
}

/// Embeds `text` with the loaded embedding model of `state` whose identity
/// is `space.model`, in the query prompt role, at `space.dim`.
pub(crate) fn embed_with(state: &State, space: &Space<'_>, text: &str) -> Result<Embedded, Failure> {
    let want = space.model;
    let mut found = None;
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
            found = Some((served.name.clone(), loaded.clone()));
            break;
        }
    }
    let Some((name, loaded)) = found else {
        return Err(Failure::new(
            TURBO_E_BUNDLE_NOT_FOUND,
            0,
            format!(
                "query: no loaded embedding model is {}@{}#{} at {} dimensions, the model of the graph's vectors",
                want.id, want.revision, want.sha256, space.dim
            ),
        ));
    };
    let held = loaded.take()?;
    let slot = held.slot();
    let opts = turbo::turbo_embed_options {
        struct_size: size_of::<turbo::turbo_embed_options>() as u32,
        truncate: 0,
        max_tokens: 0,
        prompt_role: TURBO_PROMPT_QUERY,
        normalize: if space.unit { TURBO_NORMALIZE_L2 } else { TURBO_NORMALIZE_NONE },
        pooling: 0,
        output_dim: if space.dim != loaded.model_info.dim { space.dim } else { 0 },
        quantization: 0,
        int8_ranges: std::ptr::null(),
        layers: 0,
        reserved: 0,
    };
    let views = [turbo_text { ptr: text.as_ptr().cast(), len: text.len() as u64 }];
    slot.session.write_text(&views, &opts)?;
    let res = slot.session.run()?;
    let info = res.info()?;
    let bytes = res.read(info.bytes)?;
    Ok((name, bytes.as_chunks::<4>().0.iter().map(|b| f32::from_le_bytes(*b)).collect()))
}
