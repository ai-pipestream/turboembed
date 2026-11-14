//! Calibrated cross-encoder reranking scores and bidirectional source-offset
//! semantic highlight span attribution (`#2342`).

use std::collections::HashMap;

use turbo::manifest::{PromptRole, Truncation};
use turbo::tokenizer::{Encode, Tokenizer};
use turbo_analysis::{Annotator, Document, Error, Key, ModelRef, Provenance, Span, View, ViewId};
use turbo_calibration::{Calibration as ProbCalibration, Identity, Task, Transform, sigmoid};
use turbo_normalize::{Invisible, NormalizeStep};
use turbo_sentdetect::RuleSplitter;

/// Binary relevance labels expected in a cross-encoder `TCAL0001` calibration artifact.
pub const RERANK_CALIBRATION_LABELS: [&str; 2] = ["not_relevant", "relevant"];

/// One candidate's raw cross-encoder logit, calibrated probability in `[0, 1]`,
/// and optional split-conformal relevance p-value in `(0, 1]`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CalibratedRerankScore {
    pub raw_logit: f32,
    pub calibrated_probability: f64,
    pub conformal_p_value: Option<f64>,
}

/// Calibrates raw cross-encoder logits into `[0.0, 1.0]` relevance probabilities.
#[derive(Clone, Debug)]
pub struct RerankCalibrator {
    transform: Transform,
    nonconformity: Option<Vec<f64>>,
}

impl Default for RerankCalibrator {
    fn default() -> Self {
        Self::uncalibrated()
    }
}

impl RerankCalibrator {
    /// Identity sigmoid mapping (`Temperature(1.0)`), mapping logit `z` to `sigmoid(z)`.
    pub fn uncalibrated() -> Self {
        Self { transform: Transform::Temperature(1.0), nonconformity: None }
    }

    /// Builds a calibrator from a validated [`Transform`] (`Temperature`, `Platt`, or `Isotonic`).
    pub fn from_transform(transform: Transform) -> Result<Self, String> {
        transform.check()?;
        Ok(Self { transform, nonconformity: None })
    }

    /// Builds a calibrator from a full split-conformal [`ProbCalibration`] artifact.
    pub fn from_calibration(cal: ProbCalibration) -> Result<Self, String> {
        if cal.identity().labels.len() != 2 {
            return Err(format!("rerank calibration requires 2 binary labels, got {}", cal.identity().labels.len()));
        }
        Ok(Self { transform: cal.transform().clone(), nonconformity: Some(cal.scores().to_vec()) })
    }

    /// Decodes and binds a `TCAL0001` binary artifact against `model` and `domain`.
    pub fn from_tcal_bytes(bytes: &[u8], model: &ModelRef, domain: &str, inference_hash: u64) -> Result<Self, String> {
        let expected = Identity {
            model: model.id.clone(),
            revision: model.revision.clone(),
            sha256: model.sha256.clone(),
            task: Task::Categorization,
            domain: domain.to_owned(),
            labels: RERANK_CALIBRATION_LABELS.iter().map(|&s| s.to_owned()).collect(),
            inference_hash,
        };
        let cal = ProbCalibration::from_bytes(bytes, &expected)?;
        Self::from_calibration(cal)
    }

    /// Maps a finite cross-encoder logit `raw_logit` to [`CalibratedRerankScore`].
    pub fn score(&self, raw_logit: f32) -> Result<CalibratedRerankScore, String> {
        if !raw_logit.is_finite() {
            return Err("cross-encoder logit must be finite".into());
        }
        let p_pos = sigmoid(f64::from(raw_logit)).clamp(1e-15, 1.0 - 1e-15);
        let raw_dist = [1.0 - p_pos, p_pos];
        let mut cal_dist = [0.0f64; 2];
        self.transform.apply(&raw_dist, &mut cal_dist)?;
        let prob = cal_dist[1].clamp(0.0, 1.0);
        let conformal_p_value = self.nonconformity.as_ref().map(|scores| {
            let nonconf = 1.0 - prob;
            // Sorted ascending: count scores >= nonconf via partition_point.
            let first_ge = scores.partition_point(|&s| s < nonconf);
            let ge_count = scores.len() - first_ge;
            (1.0 + ge_count as f64) / (1.0 + scores.len() as f64)
        });
        Ok(CalibratedRerankScore { raw_logit, calibrated_probability: prob, conformal_p_value })
    }
}

/// Strips invisible formatting characters (`U+00AD`, `U+200B`, etc.) via [`Invisible`].
pub fn clean_query_text(text: &str) -> Result<String, Error> {
    let doc = Document::new(text);
    let src = doc.require(&Key::<View>::DEFAULT)?;
    let step = NormalizeStep::new(Invisible);
    let (clean,) = step.annotate((src,))?;
    Ok(clean.text().to_owned())
}

/// A passage prepared with `Invisible` cleanup and bidirectional `ViewId::SOURCE`
/// alignment so every clean sentence or sub-span maps back to original source bytes.
pub struct SourcePassage<'a> {
    source: &'a str,
    doc: Document,
    clean_id: ViewId,
    clean_spans: Vec<Span>,
    source_spans: Vec<Span>,
}

impl<'a> SourcePassage<'a> {
    /// Prepares `source` with `Invisible` normalization and either caller-supplied
    /// `explicit_source_spans` (in `ViewId::SOURCE` UTF-8 bytes) or `RuleSplitter`
    /// sentence spans.
    pub fn new(source: &'a str, language: &str, explicit_source_spans: &[Span]) -> Result<Self, Error> {
        let mut doc = Document::new(source);
        let step = NormalizeStep::new(Invisible);
        let (clean_view,) = {
            let src = doc.require(&Key::<View>::DEFAULT)?;
            step.annotate((src,))?
        };
        let prov = Provenance {
            annotator: NormalizeStep::<Invisible>::ID,
            version: NormalizeStep::<Invisible>::VERSION,
            config: step.config_hash(),
            model: None,
            execution: None,
            reads: vec![Key::<View>::DEFAULT.id()],
            on: Vec::new(),
        };
        let mut clean_spans = Vec::new();
        let mut source_spans = Vec::new();
        if !explicit_source_spans.is_empty() {
            for &src_sp in explicit_source_spans {
                if (src_sp.end as usize) > source.len()
                    || src_sp.start > src_sp.end
                    || !source.is_char_boundary(src_sp.start as usize)
                    || !source.is_char_boundary(src_sp.end as usize)
                {
                    return Err(Error::Annotator {
                        annotator: "turbo-embed:rerank_highlight",
                        detail: format!(
                            "explicit sentence span [{}, {}) is invalid for passage",
                            src_sp.start, src_sp.end
                        ),
                    });
                }
                let ch = clean_view.alignment().span_to_child(src_sp);
                if !ch.is_empty() {
                    clean_spans.push(ch);
                    source_spans.push(src_sp);
                }
            }
            doc.add(&Key::<View>::named("invisible_clean"), clean_view, prov)?;
            let clean_id = doc.require(&Key::<View>::named("invisible_clean"))?.id();
            Ok(Self { source, doc, clean_id, clean_spans, source_spans })
        } else {
            let lang = if language.trim().is_empty() { "en" } else { language.trim() };
            let mut split_spans = Vec::new();
            RuleSplitter::new(lang).try_split(clean_view.text(), &mut split_spans)?;
            if split_spans.is_empty() && !clean_view.text().trim().is_empty() {
                let txt = clean_view.text();
                let s = txt.len() - txt.trim_start().len();
                let e = txt.trim_end().len();
                if s < e {
                    split_spans.push(Span::new(s as u32, e as u32));
                }
            }
            doc.add(&Key::<View>::named("invisible_clean"), clean_view, prov)?;
            let clean_id = doc.require(&Key::<View>::named("invisible_clean"))?.id();
            for ch in split_spans {
                let src_sp = doc.to_source(clean_id, ch);
                if !src_sp.is_empty() {
                    clean_spans.push(ch);
                    source_spans.push(src_sp);
                }
            }
            Ok(Self { source, doc, clean_id, clean_spans, source_spans })
        }
    }

    pub fn source_text(&self) -> &'a str {
        self.source
    }

    pub fn clean_text(&self) -> &str {
        self.doc.get_key(&Key::<View>::named("invisible_clean")).expect("clean view registered").text()
    }

    pub fn clean_spans(&self) -> &[Span] {
        &self.clean_spans
    }

    pub fn source_spans(&self) -> &[Span] {
        &self.source_spans
    }

    /// Maps an arbitrary span in `clean_text()` (e.g. from `ExtractiveQa`) back
    /// to exact UTF-8 byte offsets in `source_text()`.
    pub fn project_clean_span(&self, clean_span: Span) -> Span {
        self.doc.to_source(self.clean_id, clean_span)
    }
}

/// One attributed semantic highlight span in `ViewId::SOURCE` UTF-8 coordinates.
#[derive(Clone, Debug, PartialEq)]
pub struct HighlightCandidateSpan {
    pub source_span: Span,
    pub score: f64,
    pub surface: String,
}

/// Selects up to `max_snippets` highest-scoring non-overlapping spans.
pub fn select_top_non_overlapping(
    source: &str,
    mut scored: Vec<(Span, f64)>,
    max_snippets: usize,
) -> Vec<HighlightCandidateSpan> {
    if max_snippets == 0 || scored.is_empty() {
        return Vec::new();
    }
    scored.retain(|(sp, sc)| !sp.is_empty() && sc.is_finite());
    scored.sort_by(|a, b| b.1.total_cmp(&a.1).then_with(|| a.0.start.cmp(&b.0.start)));
    let mut chosen: Vec<HighlightCandidateSpan> = Vec::with_capacity(scored.len().min(max_snippets));
    for (sp, score) in scored {
        if chosen.len() >= max_snippets {
            break;
        }
        if chosen.iter().any(|c| sp.start < c.source_span.end && c.source_span.start < sp.end) {
            continue;
        }
        chosen.push(HighlightCandidateSpan { source_span: sp, score, surface: source[sp.range()].to_owned() });
    }
    chosen
}

/// Scores sentence spans using `tokenizer` subword IDF-weighted MaxSim alignment
/// (requires zero extra neural model load beyond the rerank or embed bundle's tokenizer).
pub fn score_spans_subword_tokenizer(
    tokenizer: &Tokenizer,
    query: &str,
    passage: &SourcePassage<'_>,
    max_snippets: usize,
) -> Result<Vec<HighlightCandidateSpan>, String> {
    if max_snippets == 0 || passage.clean_spans().is_empty() {
        return Ok(Vec::new());
    }
    let bare = Encode {
        add_special_tokens: false,
        truncation: Truncation::None,
        max_tokens: u32::MAX,
        prompt: PromptRole::None,
    };
    let q_clean = clean_query_text(query).map_err(|e| e.to_string())?;
    let q_ids = tokenizer.encode(&q_clean, bare).map_err(|e| e.message)?;
    if q_ids.is_empty() {
        return Ok(Vec::new());
    }
    let clean_text = passage.clean_text();
    let mut sent_ids: Vec<Vec<i32>> = Vec::with_capacity(passage.clean_spans().len());
    let mut df: HashMap<i32, usize> = HashMap::new();
    for &sp in passage.clean_spans() {
        let ids = tokenizer.encode(&clean_text[sp.range()], bare).map_err(|e| e.message)?;
        let mut uniq = ids.clone();
        uniq.sort_unstable();
        uniq.dedup();
        for &id in &uniq {
            *df.entry(id).or_insert(0) += 1;
        }
        sent_ids.push(ids);
    }
    let n_sents = passage.clean_spans().len() as f64;
    let mut q_weights: Vec<(i32, f64)> = Vec::with_capacity(q_ids.len());
    let mut total_w = 0.0f64;
    for &qid in &q_ids {
        let d = df.get(&qid).copied().unwrap_or(0) as f64;
        let w = ((n_sents + 1.0) / (d + 1.0)).ln() + 1.0;
        q_weights.push((qid, w));
        total_w += w;
    }
    let norm = total_w.max(1e-9);
    let mut scored = Vec::with_capacity(passage.source_spans().len());
    for (i, &src_sp) in passage.source_spans().iter().enumerate() {
        let ids = &sent_ids[i];
        if ids.is_empty() {
            continue;
        }
        let mut hit_w = 0.0f64;
        for &(qid, w) in &q_weights {
            if ids.contains(&qid) {
                hit_w += w;
            }
        }
        let density_bonus = (hit_w / (ids.len() as f64).sqrt()).min(1.0) * 0.05;
        let score = (hit_w / norm + density_bonus).clamp(0.0, 1.0);
        scored.push((src_sp, score));
    }
    Ok(select_top_non_overlapping(passage.source_text(), scored, max_snippets))
}

/// Scores sentence spans using L2-normalized pooled range vectors (`[n_sents, dim]`)
/// against `query_vec` (`[dim]`).
pub fn score_spans_pooled_cosine(
    query_vec: &[f32],
    sent_vecs: &[f32],
    dim: usize,
    passage: &SourcePassage<'_>,
    max_snippets: usize,
) -> Vec<HighlightCandidateSpan> {
    if dim == 0 || query_vec.len() != dim || max_snippets == 0 {
        return Vec::new();
    }
    let q_norm: f64 = query_vec.iter().map(|&v| f64::from(v) * f64::from(v)).sum::<f64>().sqrt();
    if q_norm == 0.0 || !q_norm.is_finite() {
        return Vec::new();
    }
    let mut scored = Vec::with_capacity(passage.source_spans().len());
    for (i, &src_sp) in passage.source_spans().iter().enumerate() {
        let Some(row) = sent_vecs.get(i * dim..(i + 1) * dim) else { break };
        let r_norm: f64 = row.iter().map(|&v| f64::from(v) * f64::from(v)).sum::<f64>().sqrt();
        if r_norm == 0.0 || !r_norm.is_finite() {
            continue;
        }
        let dot: f64 = query_vec.iter().zip(row).map(|(&a, &b)| f64::from(a) * f64::from(b)).sum();
        let cos = (dot / (q_norm * r_norm)).clamp(-1.0, 1.0);
        scored.push((src_sp, cos));
    }
    select_top_non_overlapping(passage.source_text(), scored, max_snippets)
}

/// Scores sentence spans by ColBERT-style contextual subword MaxSim:
/// `mean_{q} max_{d in sentence} cos(q_vec, d_vec)`.
pub fn score_spans_contextual_maxsim(
    query_vecs: &[f32],
    doc_vecs: &[f32],
    dim: usize,
    doc_clean_subword_spans: &[Span],
    passage: &SourcePassage<'_>,
    max_snippets: usize,
) -> Vec<HighlightCandidateSpan> {
    if dim == 0 || query_vecs.is_empty() || doc_vecs.is_empty() || max_snippets == 0 {
        return Vec::new();
    }
    let q_rows: Vec<&[f32]> = query_vecs.chunks_exact(dim).collect();
    let d_rows: Vec<&[f32]> = doc_vecs.chunks_exact(dim).collect();
    if q_rows.is_empty() || d_rows.len() != doc_clean_subword_spans.len() {
        return Vec::new();
    }
    let row_norm = |r: &[f32]| r.iter().map(|&v| f64::from(v) * f64::from(v)).sum::<f64>().sqrt();
    let q_norms: Vec<f64> = q_rows.iter().map(|r| row_norm(r)).collect();
    let d_norms: Vec<f64> = d_rows.iter().map(|r| row_norm(r)).collect();

    let mut scored = Vec::with_capacity(passage.clean_spans().len());
    for (&clean_sp, &src_sp) in passage.clean_spans().iter().zip(passage.source_spans()) {
        let tok_indices: Vec<usize> = doc_clean_subword_spans
            .iter()
            .enumerate()
            .filter_map(|(k, sp)| {
                (sp.start >= clean_sp.start && sp.end <= clean_sp.end && d_norms[k] > 0.0).then_some(k)
            })
            .collect();
        if tok_indices.is_empty() {
            continue;
        }
        let mut sum_max = 0.0f64;
        let mut valid_q = 0usize;
        for (qi, q_row) in q_rows.iter().enumerate() {
            let qn = q_norms[qi];
            if qn == 0.0 {
                continue;
            }
            valid_q += 1;
            let mut best = -1.0f64;
            for &dk in &tok_indices {
                let dot: f64 = q_row.iter().zip(d_rows[dk]).map(|(&a, &b)| f64::from(a) * f64::from(b)).sum();
                let cos = (dot / (qn * d_norms[dk])).clamp(-1.0, 1.0);
                if cos > best {
                    best = cos;
                }
            }
            sum_max += best;
        }
        if valid_q > 0 {
            scored.push((src_sp, sum_max / valid_q as f64));
        }
    }
    select_top_non_overlapping(passage.source_text(), scored, max_snippets)
}

#[cfg(test)]
mod tests {
    use super::*;
    use turbo_calibration::DataIdentity;

    #[test]
    fn calibrator_supports_temperature_platt_isotonic_and_tcal_conformal() {
        let uncal = RerankCalibrator::uncalibrated();
        let s0 = uncal.score(0.0).unwrap();
        assert!((s0.calibrated_probability - 0.5).abs() < 1e-9);
        assert!(s0.conformal_p_value.is_none());

        let platt = RerankCalibrator::from_transform(Transform::Platt { slope: 1.5, intercept: -0.2 }).unwrap();
        let lo = platt.score(-2.0).unwrap().calibrated_probability;
        let hi = platt.score(2.0).unwrap().calibrated_probability;
        assert!(lo < hi && (0.0..=1.0).contains(&lo) && (0.0..=1.0).contains(&hi));

        let mref = ModelRef { id: "bge-reranker".into(), revision: "1".into(), sha256: "a".repeat(64) };
        let id = Identity {
            model: mref.id.clone(),
            revision: mref.revision.clone(),
            sha256: mref.sha256.clone(),
            task: Task::Categorization,
            domain: "search".into(),
            labels: RERANK_CALIBRATION_LABELS.iter().map(|&s| s.into()).collect(),
            inference_hash: 42,
        };
        let cal = ProbCalibration::new(
            id,
            Transform::Temperature(1.5),
            DataIdentity { split: "val".into(), sha256: "b".repeat(64) },
            DataIdentity { split: "cal".into(), sha256: "c".repeat(64) },
            vec![0.05, 0.15, 0.35, 0.70],
        )
        .unwrap();
        let bytes = cal.to_bytes();
        let loaded = RerankCalibrator::from_tcal_bytes(&bytes, &mref, "search", 42).unwrap();
        let high = loaded.score(4.0).unwrap();
        assert!(high.calibrated_probability > 0.8);
        assert!(high.conformal_p_value.unwrap() > 0.7);
    }

    #[test]
    fn source_passage_preserves_exact_boundaries_across_invisible_chars_and_emoji() {
        let raw = "Intro \u{1F680} line. Sec\u{00AD}ond \u{200B} answer sentence here! Third trailing note.";
        let passage = SourcePassage::new(raw, "en", &[]).unwrap();
        assert_eq!(passage.source_spans().len(), 3);
        for &sp in passage.source_spans() {
            assert!(raw.is_char_boundary(sp.start as usize));
            assert!(raw.is_char_boundary(sp.end as usize));
        }
        assert!(raw[passage.source_spans()[1].range()].contains("Sec\u{00AD}ond"));

        let q = [0.0f32, 1.0];
        let sents = [1.0f32, 0.0, 0.1, 0.99, -0.5, 0.0];
        let top = score_spans_pooled_cosine(&q, &sents, 2, &passage, 1);
        assert_eq!(top.len(), 1);
        assert_eq!(top[0].surface, "Sec\u{00AD}ond \u{200B} answer sentence here!");
    }
}
