//! Handler for `ProjectChunks` (`server/proto/turbo_analysis.proto`,
//! issue #1220): computes 2D/3D PCA chunk embedding coordinates,
//! `BoundaryDriftCurveProto`, ranked `TermVectorStackProto`s for chunks and
//! the whole document, and `StrategyMatrixProto` chunking comparisons.

use turbo_analysis::export::{DocumentGraph, import};
use turbo_analysis::{Document, Ix, Layer, Span, ViewId};
use turbo_analysis_layers::{Boundaries, Chunks, Cut, Sentences, Subwords, Tokens, Vectors};
use turbo_chunker::{Drift, Rows, cosines as gap_cosines, drift as compute_drift};

use super::bindings::require_default_binding;
use super::pairwise::pairwise;
use super::pca::{compute_pca, validate_finite_vectors};
use super::query::{Embedded, QuerySpace, check_query_field, space_for_vectors};
use super::stacks::build_term_stacks;
use super::strategy::build_strategy_matrix;
use crate::api::{Failure, TURBO_E_INVALID_ARGUMENT};
use crate::offsets::{OffsetMode, Offsetter};
use crate::proto::result::{
    BoundaryDriftCurveProto, DriftGapPointProto, ProjectChunksRequest, ProjectChunksResponse, QueryPointProto,
    VectorProjectionProto,
};
use crate::results::disclosure::{Disclosure, PiiOutputPolicy};
use crate::results::document::codecs;
use crate::results::{MAX_CHUNKS, MAX_GRAPH_BYTES, MAX_SPARSE_ENTRIES, over};

const DEFAULT_PCA_DIMS: u32 = 2;
const DEFAULT_MAX_STACK_SEGMENTS: u32 = 10;

/// Maximum number of `strategy_budgets` entries accepted by `ProjectChunks`.
pub const MAX_STRATEGY_BUDGETS: usize = 16;

fn bad(message: impl Into<String>) -> Failure {
    Failure::new(TURBO_E_INVALID_ARGUMENT, 0, message)
}

pub(super) struct PreparedProject {
    pub doc: Document,
    pub view: ViewId,
    pub offset_mode: OffsetMode,
    pub pca_dims: u32,
    pub max_stack_segments: u32,
    pub disclosure: Disclosure,
}

pub(super) fn prepare_project(
    req: &ProjectChunksRequest,
    policy: PiiOutputPolicy,
    has_embed_runtime: bool,
) -> Result<PreparedProject, Failure> {
    let disclosure = Disclosure::decide(req.pii_output.as_ref(), policy)?;
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
    check_query_field(req, has_embed_runtime)?;

    let graph = DocumentGraph::from_bytes(&req.graph).map_err(|e| bad(format!("document graph: {e}")))?;
    let doc = import(&graph, codecs()).map_err(|e| bad(format!("document graph import: {e}")))?;
    let view = active_view(&doc)?;
    disclosure.require_view_text(&doc, view, "project_chunks")?;

    Ok(PreparedProject { doc, view, offset_mode, pca_dims, max_stack_segments, disclosure })
}

fn check_vector_target<'a, L: Layer>(
    doc: &'a Document,
    vecs: &'a Vectors<L>,
    spans_len: Option<usize>,
    what: &str,
) -> Result<usize, Failure> {
    require_default_binding::<Vectors<L>, L>(doc)?;
    let n = spans_len.ok_or_else(|| bad(format!("document has turbo:vectors<turbo:{what}> without turbo:{what}")))?;
    if n as u64 > MAX_CHUNKS {
        return Err(over("max_chunks", what, n as u64));
    }
    if vecs.len() != n {
        return Err(bad(format!(
            "turbo:vectors<turbo:{what}> length {} does not match turbo:{what} length {n}",
            vecs.len()
        )));
    }
    Ok(n)
}

pub(super) fn inspect_query_space(doc: &Document) -> Result<QuerySpace, Failure> {
    if let Some(vecs) = doc.get::<Vectors<Chunks>>() {
        let n = check_vector_target(doc, vecs, doc.get::<Chunks>().map(|c| c.spans.len()), "chunks")?;
        return space_for_vectors(vecs, n);
    }
    if let Some(vecs) = doc.get::<Vectors<Sentences>>() {
        let n = check_vector_target(doc, vecs, doc.get::<Sentences>().map(|s| s.spans.len()), "sentences")?;
        return space_for_vectors(vecs, n);
    }
    Err(bad("query: the document graph carries no chunk or sentence vectors to place a query among"))
}

pub(super) fn finish_project(
    prep: PreparedProject,
    req: &ProjectChunksRequest,
    query: Option<&Embedded>,
) -> Result<ProjectChunksResponse, Failure> {
    let text = prep.doc.text(prep.view);
    let offsetter = Offsetter::new(text, prep.offset_mode);

    let (projection, query) = build_projection(&prep.doc, text, &offsetter, prep.pca_dims, query)?;
    let drift_curve = if req.include_drift_curve { build_drift_curve(&prep.doc, text, &offsetter)? } else { None };
    let (chunk_term_stacks, document_term_stack) = build_term_stacks(
        &prep.doc,
        prep.view,
        text,
        &offsetter,
        prep.max_stack_segments as usize,
        prep.disclosure.graph(&prep.doc),
    )?;
    let strategy_matrix = if req.include_strategy_matrix {
        build_strategy_matrix(&prep.doc, text, &offsetter, &req.strategy_budgets)?
    } else {
        None
    };
    let pairwise = match &req.pairwise {
        Some(spec) => Some(pairwise(&prep.doc, spec, &projection.points)?),
        None => None,
    };

    Ok(ProjectChunksResponse {
        pairwise,
        query,
        projection: Some(projection),
        drift_curve,
        chunk_term_stacks,
        document_term_stack,
        strategy_matrix,
    })
}

/// Executes `ProjectChunks` on a binary [`DocumentGraph`], under
/// [`PiiOutputPolicy::ALL`].
pub fn project_chunks(req: &ProjectChunksRequest) -> Result<ProjectChunksResponse, Failure> {
    project_chunks_under(req, PiiOutputPolicy::ALL)
}

/// [`project_chunks`] under the server's `policy`: previews and term
/// stacks carry the projected view's words, so a document holding PII
/// needs that view's text disclosed (#1904).
pub fn project_chunks_under(
    req: &ProjectChunksRequest,
    policy: PiiOutputPolicy,
) -> Result<ProjectChunksResponse, Failure> {
    let prep = prepare_project(req, policy, false)?;
    finish_project(prep, req, None)
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
    query: Option<&Embedded>,
) -> Result<(VectorProjectionProto, Option<QueryPointProto>), Failure> {
    if let Some(vecs) = doc.get::<Vectors<Chunks>>() {
        let chunks = doc.get::<Chunks>();
        let n = check_vector_target(doc, vecs, chunks.map(|c| c.spans.len()), "chunks")?;
        let chunks = chunks.expect("checked Some");
        let spans: Vec<Span> = (0..n).map(|i| chunks.spans.get(Ix::new(i as u32))).collect();
        return compute_pca(vecs, &spans, &chunks.level, text, offsetter, pca_dims, query);
    }
    if let Some(vecs) = doc.get::<Vectors<Sentences>>() {
        let sents = doc.get::<Sentences>();
        let n = check_vector_target(doc, vecs, sents.map(|s| s.spans.len()), "sentences")?;
        let sents = sents.expect("checked Some");
        let spans: Vec<Span> = (0..n).map(|i| sents.spans.get(Ix::new(i as u32))).collect();
        let levels = vec![0u8; spans.len()];
        return compute_pca(vecs, &spans, &levels, text, offsetter, pca_dims, query);
    }
    if query.is_some() {
        return Err(bad("query: the document graph carries no chunk or sentence vectors to place a query among"));
    }
    let none = VectorProjectionProto {
        model: None,
        pca_dims,
        explained_variance: vec![0.0; pca_dims as usize],
        points: Vec::new(),
    };
    Ok((none, None))
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
        require_default_binding::<Vectors<Sentences>, Sentences>(doc)?;
        if vecs.len() != n {
            return Err(bad(format!(
                "turbo:vectors<turbo:sentences> length {} does not match turbo:sentences length {n}",
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
        require_default_binding::<Boundaries, Sentences>(doc)?;
        if bounds.cost.len() != n {
            return Err(bad(format!(
                "turbo:boundaries length {} does not match turbo:sentences length {n}",
                bounds.cost.len()
            )));
        }
        (vec![None; n], vec![None; n], vec![0i64; n], bounds.cost.clone(), bounds.reason.clone())
    } else {
        (vec![None; n], vec![None; n], vec![0i64; n], vec![0i32; n], vec![Cut::Sentence; n])
    };

    let mut gaps = Vec::with_capacity(n);
    // Where level-0 chunks end, once, rather than a scan of every chunk
    // per sentence.
    let mut chunk_ends: Vec<u32> = chunks.map_or_else(Vec::new, |c| {
        (0..c.spans.len()).filter(|&ci| c.level[ci] == 0).map(|ci| c.spans.get(Ix::new(ci as u32)).end).collect()
    });
    chunk_ends.sort_unstable();
    for i in 0..n {
        let sent_end = sentences.spans.get(Ix::new(i as u32)).end;
        let gap_offset = offsetter.to_wire(text, Span::new(sent_end, sent_end)).0;
        let is_chunk_cut = chunk_ends.binary_search(&sent_end).is_ok();
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
