//! `SUBWORD_MAXSIM` highlights from an embed model's contextual token
//! vectors (`#2342`): the query's token vectors, then each candidate's,
//! scored by late interaction.

use std::sync::Arc;
use std::sync::atomic::AtomicBool;

use turbo::{TURBO_PROMPT_DOCUMENT, TURBO_PROMPT_QUERY};
use turbo_analysis::{Ix, ModelRef, Span};
use turbo_analysis_embed::{SourcePassage, clean_query_text, score_spans_contextual_maxsim};

use super::highlight::{build_passages, embed_opts};
use super::{bad, to_wire_highlights};
use crate::api::{Failure, Result, TURBO_E_INTERNAL};
use crate::deadline::Deadline;
use crate::offsets::OffsetMode;
use crate::proto::result::analyze_stream_response::Outcome as ProtoOutcome;
use crate::proto::result::{RerankAndHighlightRequest, RerankedCandidateProto};
use crate::results::f32s;
use crate::state::{Loaded, Slot, State};

/// The query's contextual token vectors, `dim` floats a token.
pub(super) struct QueryVectors {
    dim: usize,
    data: Vec<f32>,
}

/// One held session of the embed model and the call's deadline.
pub(super) struct Run<'a> {
    pub(super) state: &'a State,
    pub(super) loaded: &'a Loaded,
    pub(super) slot: &'a Slot,
    pub(super) deadline: &'a Deadline,
}

/// Encodes the query of `req` on the held session.
pub(super) fn query_vectors(run: &Run<'_>, req: &RerankAndHighlightRequest) -> Result<QueryVectors> {
    let Run { state, loaded, slot, deadline } = *run;
    crate::priority::check_before_exec(deadline, &loaded.metrics)?;
    let embed = loaded.embed()?;
    let dead: &AtomicBool = deadline.cancel_flag();
    let q_clean = clean_query_text(&req.query).map_err(|e| bad(e.to_string()))?;
    let q_opts = embed_opts(TURBO_PROMPT_QUERY, 0);
    let q_outcomes =
        embed.run_token_vectors_batch(&state.stream, dead, slot, &[q_clean.as_str()], &q_opts, OffsetMode::Utf8, 0)?;
    let Some(ProtoOutcome::TokenVectors(q_tv)) = q_outcomes.into_iter().next() else {
        return Err(Failure::new(TURBO_E_INTERNAL, 0, "expected TokenVectors outcome for query"));
    };
    Ok(QueryVectors { dim: q_tv.dim as usize, data: f32s(&q_tv.data) })
}

/// Encodes the candidates of `top` on the held session and writes their
/// highlights.
pub(super) fn candidate_highlights(
    run: &Run<'_>,
    req: &RerankAndHighlightRequest,
    offsets: OffsetMode,
    top: &mut [RerankedCandidateProto],
    max_snippets: usize,
    query: &QueryVectors,
) -> Result<()> {
    let Run { state, loaded, slot, deadline } = *run;
    // The call may have ended while the query ran: its flag, not one made
    // here, stops the candidate windows.
    crate::priority::check_before_exec(deadline, &loaded.metrics)?;
    let embed = loaded.embed()?;
    let dead: &AtomicBool = deadline.cancel_flag();
    let passages = build_passages(req, top, offsets)?;
    let d_texts: Vec<&str> = passages.iter().map(SourcePassage::clean_text).collect();
    let d_opts = embed_opts(TURBO_PROMPT_DOCUMENT, 0);
    let d_outcomes =
        embed.run_token_vectors_batch(&state.stream, dead, slot, &d_texts, &d_opts, OffsetMode::Utf8, 0)?;
    for ((entry, passage), outcome) in top.iter_mut().zip(passages).zip(d_outcomes) {
        let ProtoOutcome::TokenVectors(d_tv) = outcome else {
            return Err(Failure::new(TURBO_E_INTERNAL, 0, "expected TokenVectors outcome for candidate"));
        };
        let d_vecs = f32s(&d_tv.data);
        let sub = embed.tokenize_subwords(passage.clean_text())?;
        let tok_spans: Vec<Span> = (0..sub.spans.len()).map(|k| sub.spans.get(Ix::new(k as u32))).collect();
        let spans = score_spans_contextual_maxsim(&query.data, &d_vecs, query.dim, &tok_spans, &passage, max_snippets);
        let cand = &req.candidates[entry.input_index as usize];
        entry.highlights = to_wire_highlights(cand, offsets, spans);
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub(super) async fn run_contextual_maxsim_highlights(
    state: &Arc<State>,
    embed_name: &str,
    req: &Arc<RerankAndHighlightRequest>,
    offsets: OffsetMode,
    mut top: Vec<RerankedCandidateProto>,
    max_snippets: usize,
    priority: crate::priority::Priority,
    deadline: &Deadline,
) -> Result<(ModelRef, Vec<RerankedCandidateProto>)> {
    let (_, loaded) = state.ready(embed_name, "")?;
    let _ = loaded.embed()?;
    let held = loaded.acquire_async(priority, deadline).await?;
    let state = Arc::clone(state);
    let req = Arc::clone(req);
    let deadline = deadline.clone();
    tokio::task::spawn_blocking(move || {
        held.reserve_token_ranges()?;
        let run = Run { state: &state, loaded: &loaded, slot: held.slot(), deadline: &deadline };
        let query = query_vectors(&run, &req)?;
        candidate_highlights(&run, &req, offsets, &mut top, max_snippets, &query)?;
        drop(held);
        Ok((loaded.embed()?.model.clone(), top))
    })
    .await
    .map_err(|e| Failure::new(TURBO_E_INTERNAL, 0, format!("contextual_maxsim highlight task failed: {e}")))?
}

#[cfg(test)]
#[path = "maxsim_tests.rs"]
mod tests;
