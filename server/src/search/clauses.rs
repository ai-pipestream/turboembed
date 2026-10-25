//! Leaf clause evaluators (`TermClause`, `PhraseClause`, `SparseClause`,
//! `LateInteractionClause`, and `FilterClause`) for `TurboSearchService` (`#1221`).

use turbo_chunker::terms::Bm25;

use crate::api::Result;
use crate::proto::search::{
    FilterClause, LateInteractionClause, MatchedSpan, PhraseClause, SpanProto, SparseClause, TermClause, TermMatchMode,
};
use crate::search::ingest::analyze_chunk_terms;
use crate::search::registry::{HybridIndex, ScoreMap, bad_arg, parse_space};

pub const MAX_TOP_K: usize = 10_000;

pub(crate) fn bounded_top_k(raw: u32, fallback: usize) -> Result<usize> {
    if raw as usize > MAX_TOP_K {
        return Err(bad_arg(format!("top_k ({raw}) exceeds maximum allowed ({MAX_TOP_K})")));
    }
    let k = if raw == 0 { fallback.max(1) } else { raw as usize };
    Ok(k.min(MAX_TOP_K))
}

pub(crate) fn eval_term_clause(ix: &HybridIndex, c: &TermClause) -> Result<ScoreMap> {
    let mode = TermMatchMode::try_from(c.mode).map_err(|_| bad_arg(format!("invalid TermMatchMode {}", c.mode)))?;
    let k = bounded_top_k(c.top_k, ix.chunks.len().max(1))?;
    let normalized_terms = normalize_clause_terms(&c.terms);
    if normalized_terms.is_empty() {
        return Ok(ScoreMap::new());
    }
    let refs: Vec<&str> = normalized_terms.iter().map(String::as_str).collect();
    let hits = ix.bm25.top_k(Bm25::LUCENE, &refs, ix.chunks.len().max(1));

    let mut out = ScoreMap::new();
    for (row, score) in hits {
        let ch = &ix.chunks[row as usize];
        if mode == TermMatchMode::All {
            let all_present = normalized_terms.iter().all(|want| ch.terms.iter().any(|(t, _)| t == want));
            if !all_present {
                continue;
            }
        }
        let mut spans = Vec::new();
        for want in &normalized_terms {
            for (t, sp) in &ch.terms {
                if t == want {
                    spans.push(MatchedSpan {
                        source: "bm25_term".into(),
                        label: want.clone(),
                        span: Some(SpanProto { start: sp.start, end: sp.end }),
                        score: score as f32,
                    });
                }
            }
        }
        out.insert(row, (score, spans));
        if out.len() >= k {
            break;
        }
    }
    Ok(out)
}

pub(crate) fn eval_phrase_clause(ix: &HybridIndex, c: &PhraseClause, default_k: usize) -> Result<ScoreMap> {
    let k = bounded_top_k(c.top_k, default_k)?;
    let terms = normalize_clause_terms(&c.terms);
    if terms.is_empty() {
        return Ok(ScoreMap::new());
    }
    let refs: Vec<&str> = terms.iter().map(String::as_str).collect();
    let hits = ix.bm25.phrase(Bm25::LUCENE, &refs, c.slop, k);
    let label = terms.join(" ");
    let mut out = ScoreMap::new();
    for (row, score) in hits {
        let ch = &ix.chunks[row as usize];
        let mut min_start = ch.span.end;
        let mut max_end = ch.span.start;
        for (t, sp) in &ch.terms {
            if terms.contains(t) {
                min_start = min_start.min(sp.start);
                max_end = max_end.max(sp.end);
            }
        }
        let span = if min_start <= max_end {
            SpanProto { start: min_start, end: max_end }
        } else {
            SpanProto { start: ch.span.start, end: ch.span.end }
        };
        out.insert(
            row,
            (
                score,
                vec![MatchedSpan {
                    source: "phrase".into(),
                    label: label.clone(),
                    span: Some(span),
                    score: score as f32,
                }],
            ),
        );
    }
    Ok(out)
}

pub(crate) fn eval_sparse_clause(ix: &HybridIndex, c: &SparseClause, default_k: usize) -> Result<ScoreMap> {
    let k = bounded_top_k(c.top_k, default_k)?;
    let mut ranked = Vec::new();
    for (row, ch) in ix.chunks.iter().enumerate() {
        if ch.sparse_terms.is_empty() {
            continue;
        }
        let mut total = 0.0f64;
        let mut spans = Vec::new();
        for qt in &c.query_terms {
            for dt in &ch.sparse_terms {
                let matched = (qt.term_id != 0 && qt.term_id == dt.term_id)
                    || (!qt.term_text.is_empty() && qt.term_text == dt.term_text);
                if matched {
                    let prod = qt.weight * dt.weight;
                    if prod > 0.0 && prod.is_finite() {
                        total += f64::from(prod);
                        let sp = if dt.has_span {
                            SpanProto { start: dt.span_start, end: dt.span_end }
                        } else {
                            SpanProto { start: ch.span.start, end: ch.span.end }
                        };
                        let lbl =
                            if !dt.term_text.is_empty() { dt.term_text.clone() } else { format!("#{}", dt.term_id) };
                        spans.push(MatchedSpan { source: "sparse".into(), label: lbl, span: Some(sp), score: prod });
                    }
                }
            }
        }
        if total > 0.0 {
            ranked.push((row as u32, total, spans));
        }
    }
    ranked.sort_by(|a, b| b.1.total_cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
    ranked.truncate(k);
    Ok(ranked.into_iter().map(|(r, s, sp)| (r, (s, sp))).collect())
}

pub(crate) fn eval_late_clause(ix: &HybridIndex, c: &LateInteractionClause, default_k: usize) -> Result<ScoreMap> {
    let k = bounded_top_k(c.top_k, default_k)?;
    let q_space = parse_space(c.query_space.as_ref())
        .map_err(|f| bad_arg(format!("LateInteractionClause.query_space: {}", f.message)))?;
    let Some(tok_ix) = &ix.token_index else {
        return Ok(ScoreMap::new());
    };
    let hits = tok_ix.top_k(&q_space, &c.query_token_vectors, k).map_err(|e| bad_arg(e.to_string()))?;
    let mut out = ScoreMap::new();
    for h in hits {
        let chunk_row = *tok_ix.meta(h.candidate);
        let ch = &ix.chunks[chunk_row as usize];
        let (_, matches) =
            tok_ix.explain(&q_space, &c.query_token_vectors, h.candidate).map_err(|e| bad_arg(e.to_string()))?;
        let mut spans = Vec::with_capacity(matches.len());
        for (qi, m) in matches.iter().enumerate() {
            let sp = ch
                .token_vectors
                .get(m.token as usize)
                .filter(|tv| tv.has_span)
                .map(|tv| SpanProto { start: tv.span_start, end: tv.span_end })
                .unwrap_or(SpanProto { start: ch.span.start, end: ch.span.end });
            spans.push(MatchedSpan {
                source: "maxsim".into(),
                label: format!("q_tok[{qi}]->c_tok[{}]", m.token),
                span: Some(sp),
                score: m.score,
            });
        }
        out.insert(chunk_row, (f64::from(h.score), spans));
    }
    Ok(out)
}

pub(crate) fn eval_filter_clause(ix: &HybridIndex, c: &FilterClause) -> ScoreMap {
    let mut out = ScoreMap::new();
    let needle = c.contains_text.to_ascii_lowercase();
    for (row, ch) in ix.chunks.iter().enumerate() {
        if !c.doc_id.is_empty() && ch.doc_id != c.doc_id {
            continue;
        }
        if !c.level_kind.is_empty() && ch.level_kind != c.level_kind {
            continue;
        }
        if !c.attributes.is_empty() && !c.attributes.iter().all(|(k, v)| ch.attributes.get(k) == Some(v)) {
            continue;
        }
        let mut sp = SpanProto { start: ch.span.start, end: ch.span.end };
        if !needle.is_empty() {
            if let Some(pos) = ch.text.to_ascii_lowercase().find(&needle) {
                sp = SpanProto {
                    start: ch.span.start + pos as u32,
                    end: ch.span.start + (pos + c.contains_text.len()) as u32,
                };
            } else {
                continue;
            }
        }
        out.insert(
            row as u32,
            (
                1.0,
                vec![MatchedSpan {
                    source: "filter".into(),
                    label: if !c.contains_text.is_empty() { c.contains_text.clone() } else { "match".into() },
                    span: Some(sp),
                    score: 1.0,
                }],
            ),
        );
    }
    out
}

fn normalize_clause_terms(raw_terms: &[String]) -> Vec<String> {
    let joined = raw_terms.join(" ");
    let analyzed: Vec<String> = analyze_chunk_terms(&joined, 0).into_iter().map(|(t, _)| t).collect();
    if !analyzed.is_empty() {
        return analyzed;
    }
    raw_terms.iter().filter(|s| !s.is_empty()).map(|s| s.to_ascii_lowercase()).collect()
}
