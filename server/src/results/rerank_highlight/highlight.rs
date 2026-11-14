//! Top-K semantic highlight attribution (`RANGE_POOL`, `SUBWORD_MAXSIM`, and
//! `EXTRACTIVE_QA`) for `RerankAndHighlight` (`#2342`).

use std::path::Path;
use std::sync::atomic::{AtomicBool, Ordering};

use turbo::bundle::Bundle;
use turbo::tokenizer::Tokenizer;
use turbo::{
    TURBO_DTYPE_F32, TURBO_NORMALIZE_NONE, TURBO_PROMPT_DOCUMENT, TURBO_PROMPT_QUERY, TURBO_TASK_EXTRACTIVE_QA,
    turbo_embed_options, turbo_text, turbo_token_range,
};
use turbo_analysis::{Ix, ModelRef, Span};
use turbo_analysis_embed::window::{Placed, Plan, Target, Which, pieces_of};
use turbo_analysis_embed::{
    HighlightCandidateSpan, Rows, SourcePassage, clean_query_text, score_spans_contextual_maxsim,
    score_spans_pooled_cosine, score_spans_subword_tokenizer, select_top_non_overlapping,
};

use super::{bad, prepare_source_passage, to_wire_highlights};
use crate::api::{Failure, Result, TURBO_E_INTERNAL, TURBO_E_UNSUPPORTED_OPTION, field};
use crate::deadline::Deadline;
use crate::offsets::OffsetMode;
use crate::proto::result::analyze_stream_response::Outcome as ProtoOutcome;
use crate::proto::result::{RerankAndHighlightRequest, RerankHighlightMode, RerankedCandidateProto};
use crate::results::f32s;
use crate::state::{Loaded, State};
use crate::stream::run::read_output;
use crate::tasks::embed::exec::run_ranges_on_slot;

fn embed_opts(prompt_role: u32, normalize: u32) -> turbo_embed_options {
    turbo_embed_options {
        struct_size: std::mem::size_of::<turbo_embed_options>() as u32,
        truncate: 0,
        max_tokens: 0,
        prompt_role,
        normalize,
        pooling: 0,
        output_dim: 0,
        quantization: 0,
        int8_ranges: std::ptr::null(),
        layers: 0,
        reserved: 0,
    }
}

#[allow(clippy::too_many_arguments)]
pub(super) fn attribute_top_k(
    state: &State,
    rerank_loaded: &Loaded,
    rerank_mref: &ModelRef,
    req: &RerankAndHighlightRequest,
    offsets: OffsetMode,
    mode: RerankHighlightMode,
    top: &mut [RerankedCandidateProto],
    max_snippets: usize,
    priority: crate::priority::Priority,
    deadline: &Deadline,
) -> Result<Option<ModelRef>> {
    let embed_name = req.embed_model.trim();
    let qa_name = req.qa_model.trim();
    let resolved_mode = match mode {
        RerankHighlightMode::Auto => {
            if !qa_name.is_empty() && embed_name.is_empty() {
                RerankHighlightMode::ExtractiveQa
            } else if !embed_name.is_empty() {
                RerankHighlightMode::RangePool
            } else {
                RerankHighlightMode::SubwordMaxsim
            }
        }
        other => other,
    };

    let passages: Vec<SourcePassage<'_>> = top
        .iter()
        .map(|r| prepare_source_passage(&req.candidates[r.input_index as usize], offsets, &req.language))
        .collect::<Result<Vec<_>>>()?;

    match resolved_mode {
        RerankHighlightMode::None => Ok(None),
        RerankHighlightMode::ExtractiveQa => {
            if qa_name.is_empty() {
                return Err(bad("qa_model is required for RERANK_HIGHLIGHT_MODE_EXTRACTIVE_QA"));
            }
            run_qa_highlights(state, qa_name, req, offsets, &passages, top, max_snippets, priority, deadline).map(Some)
        }
        RerankHighlightMode::RangePool => {
            if embed_name.is_empty() {
                return Err(bad("embed_model is required for RERANK_HIGHLIGHT_MODE_RANGE_POOL"));
            }
            run_range_pool_highlights(state, embed_name, req, offsets, &passages, top, max_snippets, priority, deadline)
                .map(Some)
        }
        RerankHighlightMode::SubwordMaxsim => {
            if !embed_name.is_empty() {
                run_contextual_maxsim_highlights(
                    state,
                    embed_name,
                    req,
                    offsets,
                    &passages,
                    top,
                    max_snippets,
                    priority,
                    deadline,
                )
                .map(Some)
            } else {
                let map_err = |e: turbo::status::Error| Failure::new(e.code, e.field, e.message);
                let bundle = Bundle::open(Path::new(&rerank_loaded.bundle)).map_err(map_err)?;
                let tok = Tokenizer::load(&bundle).map_err(map_err)?;
                for (entry, passage) in top.iter_mut().zip(&passages) {
                    deadline.check_alive()?;
                    let spans = score_spans_subword_tokenizer(&tok, &req.query, passage, max_snippets).map_err(bad)?;
                    let cand = &req.candidates[entry.input_index as usize];
                    entry.highlights = to_wire_highlights(cand, offsets, spans);
                }
                Ok(Some(rerank_mref.clone()))
            }
        }
        RerankHighlightMode::Auto => unreachable!(),
    }
}

#[allow(clippy::too_many_arguments)]
fn run_range_pool_highlights(
    state: &State,
    embed_name: &str,
    req: &RerankAndHighlightRequest,
    offsets: OffsetMode,
    passages: &[SourcePassage<'_>],
    top: &mut [RerankedCandidateProto],
    max_snippets: usize,
    priority: crate::priority::Priority,
    deadline: &Deadline,
) -> Result<ModelRef> {
    let (_, loaded) = state.ready(embed_name, "")?;
    let embed = loaded.embed()?;
    let hidden_dim = embed.dim as usize;

    let held = loaded.acquire(priority, deadline)?;
    crate::priority::check_before_exec(deadline, &loaded.metrics)?;
    held.reserve_token_ranges()?;
    let slot = held.slot();

    let q_clean = clean_query_text(&req.query).map_err(|e| bad(e.to_string()))?;
    let q_raw = [turbo_text { ptr: q_clean.as_ptr().cast(), len: q_clean.len() as u64 }];
    let mut q_opts = embed_opts(TURBO_PROMPT_QUERY, TURBO_NORMALIZE_NONE);
    if let Err(f) = slot.session.write_text(&q_raw, &q_opts) {
        if f.code == TURBO_E_UNSUPPORTED_OPTION {
            q_opts.normalize = 0;
            slot.session.write_text(&q_raw, &q_opts)?;
        } else {
            return Err(f);
        }
    }
    state.stream.metrics.record_batch_size(1);
    state.stream.runs.fetch_add(1, Ordering::Relaxed);
    let q_res = slot.session.run()?;
    let (q_dim, q_dtype, q_bytes) = read_output(&q_res, 1)?;
    drop(q_res);
    if q_dtype != TURBO_DTYPE_F32 || q_dim as usize != hidden_dim {
        return Err(Failure::new(TURBO_E_INTERNAL, 0, "unexpected query embed shape in RerankAndHighlight"));
    }
    let query_vec = f32s(&q_bytes);

    let d_opts = embed_opts(TURBO_PROMPT_DOCUMENT, TURBO_NORMALIZE_NONE);
    let (head, tail) = embed.row_ends(TURBO_PROMPT_DOCUMENT)?;
    let width = embed.window_width(slot, &d_opts, head.len(), tail.len())?;
    let mut plan = Plan::new(width);
    let mut subwords = Vec::with_capacity(passages.len());
    for (d, p) in passages.iter().enumerate() {
        deadline.check_alive()?;
        let sub = embed.tokenize_subwords(p.clean_text())?;
        let (starts, ends) = sub.spans.columns();
        for (i, &sp) in p.clean_spans().iter().enumerate() {
            let r = pieces_of(starts, ends, sp.start, sp.end);
            plan.layout(Target { doc: d as u32, which: Which::Sentences, item: i as u32 }, r);
        }
        plan.place(d as u32, starts.len() as u32).map_err(|e| bad(e.to_string()))?;
        subwords.push(sub);
    }

    let mut sent_hidden: Vec<Vec<f32>> =
        passages.iter().map(|p| vec![0.0f32; p.clean_spans().len() * hidden_dim]).collect();
    let (max_batch, max_ranges) = (slot.info.max_batch, slot.max_ranges() as usize);
    let mut rows = Rows::new(slot.info.max_seq, embed.tokenizer.fill_id());
    let mut ranges: Vec<turbo_token_range> = Vec::new();
    let mut targets: Vec<u32> = Vec::new();
    let mut flat: Vec<f32> = Vec::new();
    let head_len = head.len() as u32;
    let dead = AtomicBool::new(false);
    let mut p_at = 0usize;

    let flush = |rows: &mut Rows,
                 ranges: &mut Vec<turbo_token_range>,
                 targets: &mut Vec<u32>,
                 flat: &mut Vec<f32>,
                 sent_hidden: &mut [Vec<f32>]|
     -> Result<()> {
        if rows.is_empty() {
            return Ok(());
        }
        crate::priority::check_before_exec(deadline, &loaded.metrics)?;
        run_ranges_on_slot(&state.stream, &dead, slot, rows, ranges, hidden_dim as u32, flat)?;
        for (v, &t) in flat.chunks_exact(hidden_dim).zip(targets.iter()) {
            let x: Placed = plan.placed[t as usize];
            let i = x.target.item as usize;
            let dst = &mut sent_hidden[x.target.doc as usize][i * hidden_dim..(i + 1) * hidden_dim];
            if x.whole {
                dst.copy_from_slice(v);
            } else {
                let n = (x.end - x.start) as f32;
                for (dv, &sv) in dst.iter_mut().zip(v) {
                    *dv += sv * n;
                }
            }
        }
        rows.clear();
        ranges.clear();
        targets.clear();
        Ok(())
    };

    for (w, win) in plan.windows.iter().enumerate() {
        let end = p_at + plan.placed[p_at..].iter().take_while(|x| x.window as usize == w).count();
        while p_at < end {
            if rows.len() == max_batch || ranges.len() == max_ranges {
                flush(&mut rows, &mut ranges, &mut targets, &mut flat, &mut sent_hidden)?;
            }
            let take = (end - p_at).min(max_ranges - ranges.len());
            let ids = &subwords[win.doc as usize].ids[win.start as usize..win.end as usize];
            let row = rows.push(&[&head, ids, &tail]);
            for (k, x) in plan.placed[p_at..p_at + take].iter().enumerate() {
                ranges.push(turbo_token_range { row, start: head_len + x.start, end: head_len + x.end });
                targets.push((p_at + k) as u32);
            }
            p_at += take;
        }
    }
    flush(&mut rows, &mut ranges, &mut targets, &mut flat, &mut sent_hidden)?;
    drop(held);

    for sp in &plan.split {
        let inv = 1.0 / sp.pieces as f32;
        let i = sp.target.item as usize;
        for val in &mut sent_hidden[sp.target.doc as usize][i * hidden_dim..(i + 1) * hidden_dim] {
            *val *= inv;
        }
    }

    for ((entry, passage), s_vecs) in top.iter_mut().zip(passages).zip(&sent_hidden) {
        let spans = score_spans_pooled_cosine(&query_vec, s_vecs, hidden_dim, passage, max_snippets);
        let cand = &req.candidates[entry.input_index as usize];
        entry.highlights = to_wire_highlights(cand, offsets, spans);
    }
    Ok(embed.model.clone())
}

#[allow(clippy::too_many_arguments)]
fn run_contextual_maxsim_highlights(
    state: &State,
    embed_name: &str,
    req: &RerankAndHighlightRequest,
    offsets: OffsetMode,
    passages: &[SourcePassage<'_>],
    top: &mut [RerankedCandidateProto],
    max_snippets: usize,
    priority: crate::priority::Priority,
    deadline: &Deadline,
) -> Result<ModelRef> {
    let (_, loaded) = state.ready(embed_name, "")?;
    let embed = loaded.embed()?;
    let held = loaded.acquire(priority, deadline)?;
    crate::priority::check_before_exec(deadline, &loaded.metrics)?;
    held.reserve_token_ranges()?;
    let slot = held.slot();
    let dead = AtomicBool::new(false);

    let q_clean = clean_query_text(&req.query).map_err(|e| bad(e.to_string()))?;
    let q_opts = embed_opts(TURBO_PROMPT_QUERY, 0);
    let q_outcomes =
        embed.run_token_vectors_batch(&state.stream, &dead, slot, &[q_clean.as_str()], &q_opts, OffsetMode::Utf8, 0)?;
    let Some(ProtoOutcome::TokenVectors(q_tv)) = q_outcomes.into_iter().next() else {
        return Err(Failure::new(TURBO_E_INTERNAL, 0, "expected TokenVectors outcome for query"));
    };
    let dim = q_tv.dim as usize;
    let q_vecs = f32s(&q_tv.data);

    let d_texts: Vec<&str> = passages.iter().map(SourcePassage::clean_text).collect();
    let d_opts = embed_opts(TURBO_PROMPT_DOCUMENT, 0);
    let d_outcomes =
        embed.run_token_vectors_batch(&state.stream, &dead, slot, &d_texts, &d_opts, OffsetMode::Utf8, 0)?;
    drop(held);

    for ((entry, passage), outcome) in top.iter_mut().zip(passages).zip(d_outcomes) {
        let ProtoOutcome::TokenVectors(d_tv) = outcome else {
            return Err(Failure::new(TURBO_E_INTERNAL, 0, "expected TokenVectors outcome for candidate"));
        };
        let d_vecs = f32s(&d_tv.data);
        let sub = embed.tokenize_subwords(passage.clean_text())?;
        let tok_spans: Vec<Span> = (0..sub.spans.len()).map(|k| sub.spans.get(Ix::new(k as u32))).collect();
        let spans = score_spans_contextual_maxsim(&q_vecs, &d_vecs, dim, &tok_spans, passage, max_snippets);
        let cand = &req.candidates[entry.input_index as usize];
        entry.highlights = to_wire_highlights(cand, offsets, spans);
    }
    Ok(embed.model.clone())
}

#[allow(clippy::too_many_arguments)]
fn run_qa_highlights(
    state: &State,
    qa_name: &str,
    req: &RerankAndHighlightRequest,
    offsets: OffsetMode,
    passages: &[SourcePassage<'_>],
    top: &mut [RerankedCandidateProto],
    max_snippets: usize,
    priority: crate::priority::Priority,
    deadline: &Deadline,
) -> Result<ModelRef> {
    let (_, loaded) = state.ready(qa_name, "")?;
    if loaded.task != TURBO_TASK_EXTRACTIVE_QA {
        return Err(Failure::new(
            TURBO_E_UNSUPPORTED_OPTION,
            0,
            format!("model {qa_name} is not a TASK_EXTRACTIVE_QA bundle"),
        ));
    }
    let qa = loaded.qa.as_ref().ok_or_else(|| Failure::new(TURBO_E_INTERNAL, 0, "missing QaModel state"))?;
    let mi = &loaded.model_info;
    let mref = ModelRef { id: field(&mi.model_id), revision: field(&mi.revision), sha256: field(&mi.manifest_sha256) };
    let map_err = |e: turbo::status::Error| Failure::new(e.code, e.field, e.message);

    let mut decode_opts = turbo::qa::DecodeOptions { top_k: max_snippets, non_overlapping: true, ..Default::default() };
    if req.max_answer_tokens > 0 {
        decode_opts.max_answer_tokens = req.max_answer_tokens as usize;
    }
    if let Some(th) = req.qa_null_threshold {
        if !th.is_finite() {
            return Err(bad("qa_null_threshold must be finite"));
        }
        decode_opts.null_threshold = th;
    }

    let q_clean = clean_query_text(&req.query).map_err(|e| bad(e.to_string()))?;
    let held = loaded.acquire(priority, deadline)?;
    let slot = held.slot();
    let mut cfg = qa.config.clone();
    cfg.max_seq = slot.info.max_seq;
    let fill = qa.tokenizer.fill_id();

    for (entry, passage) in top.iter_mut().zip(passages) {
        crate::priority::check_before_exec(deadline, &loaded.metrics)?;
        let ctx = passage.clean_text();
        if ctx.trim().is_empty() {
            entry.qa_abstained = true;
            entry.qa_null_probability = Some(1.0);
            continue;
        }
        let windows = turbo::qa::windows(&qa.tokenizer, qa.family, &cfg, &q_clean, ctx).map_err(map_err)?;
        if windows.is_empty() {
            entry.qa_abstained = true;
            entry.qa_null_probability = Some(1.0);
            continue;
        }
        let mut win_logits = Vec::with_capacity(windows.len());
        for chunk in windows.chunks((slot.info.max_batch as usize).max(1)) {
            crate::priority::check_before_exec(deadline, &loaded.metrics)?;
            loaded.metrics.record_batch_size(chunk.len());
            state.stream.runs.fetch_add(1, Ordering::Relaxed);
            let (batch_logits, _) =
                unsafe { turbo::qa::run_window_batch(slot.session.as_ptr(), fill, &qa.head, chunk) }
                    .map_err(map_err)?;
            win_logits.extend(batch_logits);
        }
        let decoded = turbo::qa::decode(ctx, &win_logits, decode_opts).map_err(map_err)?;
        match decoded {
            turbo::qa::Answer::None { .. } => {
                entry.qa_abstained = true;
                entry.qa_null_probability = Some(1.0);
            }
            turbo::qa::Answer::Spans { answers, null_probability, .. } => {
                entry.qa_abstained = false;
                entry.qa_null_probability = Some(null_probability);
                let mut scored = Vec::with_capacity(answers.len());
                for a in answers {
                    let clean_sp = Span::new(a.bytes.start as u32, a.bytes.end as u32);
                    let src_sp = passage.project_clean_span(clean_sp);
                    scored.push((src_sp, a.probability));
                }
                let spans: Vec<HighlightCandidateSpan> =
                    select_top_non_overlapping(passage.source_text(), scored, max_snippets);
                let cand = &req.candidates[entry.input_index as usize];
                entry.highlights = to_wire_highlights(cand, offsets, spans);
            }
        }
    }
    drop(held);
    Ok(mref)
}
