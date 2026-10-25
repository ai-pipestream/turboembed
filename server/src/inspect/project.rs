//! Handler for `ProjectChunks` (`server/proto/turbo_analysis.proto`,
//! issue #1220): computes 2D/3D PCA chunk embedding coordinates,
//! `BoundaryDriftCurveProto`, ranked `TermVectorStackProto`s for chunks and
//! the whole document, and `StrategyMatrixProto` chunking comparisons.

use turbo_analysis::export::{DocumentGraph, import};
use turbo_analysis::{Annotator, Document, Ix, IxRange, Layer, Span, Spans, ViewId};
use turbo_analysis_layers::{Boundaries, Chunks, Cut, Sentences, Structure, Subwords, Tokens, Vectors};
use turbo_chunker::compare::{Pricing, Run, compare};
use turbo_chunker::{
    Budget, Chunker, ClauseCuts, Costs, Drift, Rows, StructureBoundaries, cosines as gap_cosines,
    drift as compute_drift,
};

use super::pca::{compute_pca, validate_finite_vectors};
use super::stacks::build_term_stacks;
use crate::api::{Failure, TURBO_E_INVALID_ARGUMENT};
use crate::offsets::{OffsetMode, Offsetter};
use crate::proto::result::{
    BoundaryDriftCurveProto, BudgetSpec, DriftGapPointProto, ProjectChunksRequest, ProjectChunksResponse,
    StrategyComparisonEntryProto, StrategyMatrixProto, VectorProjectionProto,
};
use crate::results::document::codecs;
use crate::results::{MAX_CHUNKS, MAX_GRAPH_BYTES, MAX_SPARSE_ENTRIES, over};

const DEFAULT_PCA_DIMS: u32 = 2;
const DEFAULT_MAX_STACK_SEGMENTS: u32 = 10;

/// Maximum number of `strategy_budgets` entries accepted by `ProjectChunks`.
pub const MAX_STRATEGY_BUDGETS: usize = 16;

fn bad(message: impl Into<String>) -> Failure {
    Failure::new(TURBO_E_INVALID_ARGUMENT, 0, message)
}

/// Executes `ProjectChunks` on a binary [`DocumentGraph`].
pub fn project_chunks(req: &ProjectChunksRequest) -> Result<ProjectChunksResponse, Failure> {
    if req.graph.len() as u64 > MAX_GRAPH_BYTES {
        return Err(over("max_graph_bytes", "document graph bytes", req.graph.len() as u64));
    }
    if req.graph.is_empty() {
        return Err(bad("graph: document graph bytes are required"));
    }
    if req.strategy_budgets.len() > MAX_STRATEGY_BUDGETS {
        return Err(over("max_strategy_budgets", "strategy_budgets", req.strategy_budgets.len() as u64));
    }
    let offset_mode = OffsetMode::from_proto(req.offsets)?;
    let pca_dims = if req.pca_dims == 0 { DEFAULT_PCA_DIMS } else { req.pca_dims };
    if pca_dims != 2 && pca_dims != 3 {
        return Err(bad(format!("pca_dims {pca_dims}: must be 2 or 3 (or 0 for default 2)")));
    }
    let max_stack_segments =
        if req.max_stack_segments == 0 { DEFAULT_MAX_STACK_SEGMENTS } else { req.max_stack_segments };
    if max_stack_segments as u64 > MAX_SPARSE_ENTRIES {
        return Err(over("max_sparse_entries", "max_stack_segments", max_stack_segments as u64));
    }

    let graph = DocumentGraph::from_bytes(&req.graph).map_err(|e| bad(format!("document graph: {e}")))?;
    let doc = import(&graph, codecs()).map_err(|e| bad(format!("document graph import: {e}")))?;

    let view = active_view(&doc)?;
    let text = doc.text(view);
    let offsetter = Offsetter::new(text, offset_mode);

    let projection = build_projection(&doc, text, &offsetter, pca_dims)?;
    let drift_curve = if req.include_drift_curve { build_drift_curve(&doc, text, &offsetter)? } else { None };
    let (chunk_term_stacks, document_term_stack) =
        build_term_stacks(&doc, view, text, &offsetter, max_stack_segments as usize);
    let strategy_matrix = if req.include_strategy_matrix {
        build_strategy_matrix(&doc, text, &offsetter, &req.strategy_budgets)?
    } else {
        None
    };

    Ok(ProjectChunksResponse {
        projection: Some(projection),
        drift_curve,
        chunk_term_stacks,
        document_term_stack,
        strategy_matrix,
    })
}

fn active_view(doc: &Document) -> Result<ViewId, Failure> {
    let view = if let Some(chunks) = doc.get::<Chunks>() {
        chunks.spans.view()
    } else if let Some(sentences) = doc.get::<Sentences>() {
        sentences.spans.view()
    } else if let Some(tokens) = doc.get::<Tokens>() {
        tokens.spans.view()
    } else {
        ViewId::SOURCE
    };
    if let Some(sents) = doc.get::<Sentences>()
        && sents.spans.view() != view
    {
        return Err(bad(format!(
            "sentences view {} does not match active projection view {}",
            sents.spans.view().0,
            view.0
        )));
    }
    if let Some(toks) = doc.get::<Tokens>()
        && toks.spans.view() != view
    {
        return Err(bad(format!(
            "tokens view {} does not match active projection view {}",
            toks.spans.view().0,
            view.0
        )));
    }
    if let Some(subs) = doc.get::<Subwords>()
        && subs.spans.view() != view
    {
        return Err(bad(format!(
            "subwords view {} does not match active projection view {}",
            subs.spans.view().0,
            view.0
        )));
    }
    Ok(view)
}

fn build_projection(
    doc: &Document,
    text: &str,
    offsetter: &Offsetter,
    pca_dims: u32,
) -> Result<VectorProjectionProto, Failure> {
    if let Some(vecs) = doc.get::<Vectors<Chunks>>() {
        let chunks =
            doc.get::<Chunks>().ok_or_else(|| bad("document has turbo:vectors<turbo:chunks> without turbo:chunks"))?;
        if chunks.spans.len() as u64 > MAX_CHUNKS {
            return Err(over("max_chunks", "chunks", chunks.spans.len() as u64));
        }
        if vecs.len() != chunks.spans.len() {
            return Err(bad(format!(
                "turbo:vectors<turbo:chunks> length {} does not match turbo:chunks length {}",
                vecs.len(),
                chunks.spans.len()
            )));
        }
        let spans: Vec<Span> = (0..chunks.spans.len()).map(|i| chunks.spans.get(Ix::new(i as u32))).collect();
        return compute_pca(vecs, &spans, &chunks.level, text, offsetter, pca_dims);
    }
    if let Some(vecs) = doc.get::<Vectors<Sentences>>() {
        let sentences = doc
            .get::<Sentences>()
            .ok_or_else(|| bad("document has turbo:vectors<opennlp:sentences> without opennlp:sentences"))?;
        if sentences.spans.len() as u64 > MAX_CHUNKS {
            return Err(over("max_chunks", "sentences", sentences.spans.len() as u64));
        }
        if vecs.len() != sentences.spans.len() {
            return Err(bad(format!(
                "turbo:vectors<opennlp:sentences> length {} does not match opennlp:sentences length {}",
                vecs.len(),
                sentences.spans.len()
            )));
        }
        let spans: Vec<Span> = (0..sentences.spans.len()).map(|i| sentences.spans.get(Ix::new(i as u32))).collect();
        let levels = vec![0u8; spans.len()];
        return compute_pca(vecs, &spans, &levels, text, offsetter, pca_dims);
    }
    Ok(VectorProjectionProto {
        model: None,
        pca_dims,
        explained_variance: vec![0.0; pca_dims as usize],
        points: Vec::new(),
    })
}

fn cut_code(c: Cut) -> u32 {
    match c {
        Cut::DocumentStart => 0,
        Cut::Sentence => 1,
        Cut::Section => 2,
        Cut::Heading => 3,
        Cut::Paragraph => 4,
        Cut::TopicShift => 5,
        Cut::Clause => 6,
        Cut::Item => 7,
        Cut::Inside(_) => 8,
        Cut::Overlap(_) => 9,
    }
}

fn build_drift_curve(
    doc: &Document,
    text: &str,
    offsetter: &Offsetter,
) -> Result<Option<BoundaryDriftCurveProto>, Failure> {
    let Some(sentences) = doc.get::<Sentences>() else {
        return Ok(None);
    };
    let n = sentences.spans.len();
    let cfg = Drift::default();
    let chunks = doc.get::<Chunks>();

    let (raw_cos, sm_cos, depth_scores, costs, reasons) = if let Some(vecs) = doc.get::<Vectors<Sentences>>() {
        if vecs.len() != n {
            return Err(bad(format!(
                "turbo:vectors<opennlp:sentences> length {} does not match opennlp:sentences length {n}",
                vecs.len()
            )));
        }
        validate_finite_vectors(vecs, n)?;
        let rows = Rows::of(vecs);
        let raw = gap_cosines(&rows, cfg.window);
        let (sm, depth) = smooth_and_depth(&raw, cfg.smoothing as usize);
        let (cost, reason) = compute_drift(&rows, &cfg);
        (raw, sm, depth, cost, reason)
    } else if let Some(bounds) = doc.get::<Boundaries>() {
        if bounds.cost.len() != n {
            return Err(bad(format!(
                "turbo:boundaries length {} does not match opennlp:sentences length {n}",
                bounds.cost.len()
            )));
        }
        (vec![None; n], vec![None; n], vec![0i64; n], bounds.cost.clone(), bounds.reason.clone())
    } else {
        (vec![None; n], vec![None; n], vec![0i64; n], vec![0i32; n], vec![Cut::Sentence; n])
    };

    let mut gaps = Vec::with_capacity(n);
    for i in 0..n {
        let sent_end = sentences.spans.get(Ix::new(i as u32)).end;
        let gap_offset = offsetter.to_wire(text, Span::new(sent_end, sent_end)).0;
        let is_chunk_cut = chunks.is_some_and(|c| {
            (0..c.spans.len()).any(|ci| c.level[ci] == 0 && c.spans.get(Ix::new(ci as u32)).end == sent_end)
        });
        gaps.push(DriftGapPointProto {
            sentence: i as u32,
            gap_offset,
            raw_cosine_micro: raw_cos[i],
            smoothed_cosine_micro: sm_cos[i],
            depth_score_micro: depth_scores[i],
            cost: costs[i],
            reason: cut_code(reasons[i]),
            is_chunk_cut,
        });
    }

    Ok(Some(BoundaryDriftCurveProto { window: cfg.window.get(), smoothing: cfg.smoothing, gaps }))
}

fn smooth_and_depth(raw_opt: &[Option<i64>], smoothing: usize) -> (Vec<Option<i64>>, Vec<i64>) {
    let at: Vec<usize> = (0..raw_opt.len()).filter(|&g| raw_opt[g].is_some()).collect();
    let raw: Vec<i64> = at.iter().filter_map(|&g| raw_opt[g]).collect();
    let m = raw.len();
    let mut prefix = vec![0i64; m + 1];
    for (i, &x) in raw.iter().enumerate() {
        prefix[i + 1] = prefix[i] + x;
    }
    let sm: Vec<i64> = (0..m)
        .map(|i| {
            let r = smoothing.min(i).min(m - 1 - i);
            let (a, b) = (i - r, i + r + 1);
            let (sum, cnt) = (prefix[b] - prefix[a], (b - a) as i64);
            (2 * sum + cnt).div_euclid(2 * cnt)
        })
        .collect();
    let mut left = sm.clone();
    for i in 1..m {
        if sm[i - 1] >= sm[i] {
            left[i] = left[i - 1];
        }
    }
    let mut right = sm.clone();
    for i in (0..m.saturating_sub(1)).rev() {
        if sm[i + 1] >= sm[i] {
            right[i] = right[i + 1];
        }
    }
    let mut sm_out = vec![None; raw_opt.len()];
    let mut depth_out = vec![0i64; raw_opt.len()];
    for (k, &g) in at.iter().enumerate() {
        sm_out[g] = Some(sm[k]);
        depth_out[g] = left[k] + right[k] - 2 * sm[k];
    }
    (sm_out, depth_out)
}

fn greedy_sentence_chunks(sentences: &Sentences, subwords: &Subwords, budget: Budget) -> Chunks {
    let view = sentences.spans.view();
    let n = sentences.spans.len();
    let (sub_starts, sub_ends) = subwords.spans.columns();
    let mut spans = Spans::<Chunks>::new(view);
    let mut sent_ranges = Vec::new();
    let mut cuts = Vec::new();
    let mut costs = Vec::new();
    let mut model_tokens = Vec::new();
    let mut over_budget = Vec::new();

    if n > 0 {
        let mut cur_start = 0usize;
        let mut cur_tokens = budget.overhead;
        for i in 0..n {
            let sp = sentences.spans.get(Ix::new(i as u32));
            let s_sub = sub_ends.partition_point(|&e| e <= sp.start);
            let e_sub = sub_starts.partition_point(|&s| s < sp.end).max(s_sub);
            let tok_count = (e_sub - s_sub) as u32;
            if i > cur_start && cur_tokens.saturating_add(tok_count) > budget.max {
                let first = sentences.spans.get(Ix::new(cur_start as u32));
                let last = sentences.spans.get(Ix::new((i - 1) as u32));
                spans.push(Span::new(first.start, last.end));
                sent_ranges.push(IxRange::new(cur_start as u32, i as u32));
                cuts.push(if cur_start == 0 { Cut::DocumentStart } else { Cut::Sentence });
                costs.push(0);
                model_tokens.push(cur_tokens);
                over_budget.push(cur_tokens > budget.max);
                cur_start = i;
                cur_tokens = budget.overhead.saturating_add(tok_count);
            } else {
                cur_tokens = cur_tokens.saturating_add(tok_count);
            }
        }
        let first = sentences.spans.get(Ix::new(cur_start as u32));
        let last = sentences.spans.get(Ix::new((n - 1) as u32));
        spans.push(Span::new(first.start, last.end));
        sent_ranges.push(IxRange::new(cur_start as u32, n as u32));
        cuts.push(if cur_start == 0 { Cut::DocumentStart } else { Cut::Sentence });
        costs.push(0);
        model_tokens.push(cur_tokens);
        over_budget.push(cur_tokens > budget.max);
    }

    let len = spans.len();
    Chunks {
        spans,
        level: vec![0; len],
        parent: vec![None; len],
        sentences: sent_ranges,
        cut: cuts,
        cost: costs,
        model_tokens,
        over_budget,
        prompt_overhead: budget.overhead,
    }
}

fn build_strategy_matrix(
    doc: &Document,
    text: &str,
    offsetter: &Offsetter,
    requested: &[BudgetSpec],
) -> Result<Option<StrategyMatrixProto>, Failure> {
    let (Some(sentences), Some(subwords), Some(bounds)) =
        (doc.get::<Sentences>(), doc.get::<Subwords>(), doc.get::<Boundaries>())
    else {
        return Ok(None);
    };
    let structure = doc.get::<Structure>();
    let clauses = doc.get::<ClauseCuts>();

    let mut runs: Vec<(BudgetSpec, Budget, Chunks)> = Vec::new();
    if requested.is_empty() {
        let base_spec = BudgetSpec {
            name: "semantic_dp".into(),
            max_tokens: 256,
            target_tokens: 192,
            size_weight: 1000,
            overhead: 2,
        };
        let b = Budget { max: 256, target: 192, size_weight: 1000, overhead: 2 };
        let greedy = greedy_sentence_chunks(sentences, subwords, b);
        runs.push((BudgetSpec { name: "greedy_sentences".into(), ..base_spec.clone() }, b, greedy));

        let struct_bounds = StructureBoundaries { costs: Costs::default() }
            .annotate((sentences, structure))
            .map_err(|e| bad(format!("structure_only boundaries: {e}")))?
            .0;
        let struct_chunks = Chunker { budget: b }
            .annotate((sentences, subwords, &struct_bounds, structure, None))
            .map_err(|e| bad(format!("structure_only chunker: {e}")))?
            .0;
        runs.push((BudgetSpec { name: "structure_only".into(), ..base_spec.clone() }, b, struct_chunks));

        let sem_chunks = Chunker { budget: b }
            .annotate((sentences, subwords, bounds, structure, clauses))
            .map_err(|e| bad(format!("semantic_dp chunker: {e}")))?
            .0;
        runs.push((base_spec, b, sem_chunks));
    } else {
        for (idx, spec) in requested.iter().cloned().enumerate() {
            if spec.max_tokens == 0 {
                return Err(bad(format!("strategy_budgets[{idx}].max_tokens must be > 0")));
            }
            let b = Budget {
                max: spec.max_tokens,
                target: if spec.target_tokens == 0 { spec.max_tokens } else { spec.target_tokens },
                size_weight: spec.size_weight,
                overhead: spec.overhead,
            };
            let chunks = Chunker { budget: b }
                .annotate((sentences, subwords, bounds, structure, clauses))
                .map_err(|e| bad(format!("strategy_budgets[{idx}] chunker: {e}")))?
                .0;
            runs.push((spec, b, chunks));
        }
    }

    let baseline_chunks = doc.get::<Chunks>().or_else(|| runs.last().map(|(_, _, c)| c));
    let mut entries = Vec::with_capacity(runs.len());
    for (spec, b, chunks) in &runs {
        let (matched, disagreements) = if let Some(base) = baseline_chunks {
            let run_a = Run {
                chunks: base,
                execution: None,
                pricing: Some(Pricing { sentences, boundaries: bounds, budget: *b }),
            };
            let run_b =
                Run { chunks, execution: None, pricing: Some(Pricing { sentences, boundaries: bounds, budget: *b }) };
            let rep = compare(run_a, run_b).map_err(|e| bad(format!("strategy compare: {e}")))?;
            (rep.matched as u32, rep.disagreements.len() as u32)
        } else {
            (0, 0)
        };
        let over_budget_count = chunks.over_budget.iter().filter(|&&ob| ob).count() as u32;
        let total_cost: i64 = chunks.cost.iter().copied().fold(0i64, i64::saturating_add);
        let starts_bytes: Vec<u32> =
            (0..chunks.spans.len()).map(|i| chunks.spans.get(Ix::new(i as u32)).start).collect();
        let cut_starts = offsetter.column_to_wire(text, &starts_bytes);
        entries.push(StrategyComparisonEntryProto {
            name: if spec.name.is_empty() { format!("max_{}", spec.max_tokens) } else { spec.name.clone() },
            budget: Some(spec.clone()),
            chunk_count: chunks.spans.len() as u32,
            over_budget_count,
            total_cost,
            matched_cuts_with_baseline: matched,
            disagreements_with_baseline: disagreements,
            cut_starts,
        });
    }

    Ok(Some(StrategyMatrixProto { strategies: entries }))
}
