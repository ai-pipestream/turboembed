//! Toxic evidence span evaluation and Unicode codepoint-to-UTF-8/token span
//! projection (`#890`).
//!
//! The `ipavlopoulos/toxic_spans` corpus records character (Unicode codepoint)
//! offsets `[char_start, char_end)` into each comment's text. Inside this
//! workspace every span is a UTF-8 byte [`Span`] in a [`View`], and evidence
//! spans on [`Incivility`] are [`IxRange<Tokens>`] token ranges over
//! `opennlp:tokens`.
//!
//! [`char_span_to_utf8_span`] converts a Unicode codepoint span in `view` into
//! a UTF-8 byte [`Span`] via [`View::span_from_code_point`] and verifies the
//! exact reverse projection via [`View::span_to_code_point`].
//! [`char_span_to_token_range`] maps that UTF-8 byte [`Span`] to the smallest
//! contiguous [`IxRange<Tokens>`] covering all tokens in `tokens` whose byte
//! spans overlap `[byte_start, byte_end)`.

use turbo_analysis::{CodePointOffset, CodePointSpan, Ix, IxRange, Layer, Span, View};
use turbo_analysis_layers::{Incivility, Tokens};

/// Converts a half-open Unicode codepoint span `[char_start, char_end)` in
/// `view` into a UTF-8 byte [`Span`], verifying the bidirectional round trip
/// `codepoint -> utf8 -> codepoint` through `view.offsets()`.
pub fn char_span_to_utf8_span(view: &View, char_start: u32, char_end: u32) -> Result<Span, String> {
    let cp_span = CodePointSpan::try_new(CodePointOffset(char_start), CodePointOffset(char_end))
        .ok_or_else(|| format!("code point span {char_start}..{char_end} ends before it starts"))?;
    let utf8_span = view.span_from_code_point(cp_span).map_err(|e| e.to_string())?;
    let back = view.span_to_code_point(utf8_span);
    if back != cp_span {
        return Err(format!(
            "code point span {char_start}..{char_end} round-tripped to {}..{}",
            back.start.0, back.end.0
        ));
    }
    Ok(utf8_span)
}

/// Converts a half-open Unicode codepoint span `[char_start, char_end)` in
/// `view` to the overlapping token index range [`IxRange<Tokens>`] in `tokens`.
///
/// Returns `Ok(None)` when the span is empty or falls strictly on whitespace
/// between tokens. Returns `Err` when `tokens` belongs to a different [`View`]
/// or `[char_start, char_end)` is out of bounds.
pub fn char_span_to_token_range(
    view: &View,
    tokens: &Tokens,
    char_start: u32,
    char_end: u32,
) -> Result<Option<IxRange<Tokens>>, String> {
    if tokens.spans.view() != view.id() {
        return Err("tokens belong to another view".into());
    }
    let byte_span = char_span_to_utf8_span(view, char_start, char_end)?;
    if byte_span.is_empty() {
        return Ok(None);
    }
    let (tok_starts, tok_ends) = tokens.spans.columns();
    let tok_start = tok_ends.partition_point(|&end| end <= byte_span.start);
    let tok_end = tok_starts.partition_point(|&start| start < byte_span.end);
    if tok_start < tok_end { Ok(Some(IxRange::new(tok_start as u32, tok_end as u32))) } else { Ok(None) }
}

/// One gold-annotated comment from `ipavlopoulos/toxic_spans`, holding its
/// gold character spans (converted to UTF-8 byte [`Span`]s in `view`) and the
/// corresponding overlapping [`IxRange<Tokens>`] ranges.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct GoldToxicSpanUnit {
    /// Gold UTF-8 byte spans in the comment's [`View`].
    pub utf8_spans: Vec<Span>,
    /// Gold overlapping token index ranges in `opennlp:tokens`.
    pub token_ranges: Vec<IxRange<Tokens>>,
}

/// Corpus-level evidence-span precision, recall, and F1 on `toxic_spans` (`#890`),
/// measured both at token granularity (`IxRange<Tokens>`) and character
/// (codepoint) granularity, following Pavlopoulos et al. (SemEval-2021 Task 5 /
/// ACL 2022).
#[derive(Copy, Clone, Debug, Default, PartialEq)]
pub struct ToxicSpanMetrics {
    /// Number of evaluated units.
    pub units: usize,
    /// Number of units with at least one gold toxic span.
    pub units_with_gold_spans: usize,
    /// Number of units where the annotator predicted at least one evidence span.
    pub units_with_predicted_spans: usize,
    /// Macro-averaged per-unit token-level precision (SemEval-2021 Task 5 convention:
    /// when both gold and predicted are empty, `P = R = F1 = 1.0`).
    pub token_precision: f64,
    /// Macro-averaged per-unit token-level recall.
    pub token_recall: f64,
    /// Macro-averaged per-unit token-level F1.
    pub token_f1: f64,
    /// Micro-averaged token-level precision across all units with gold or predicted spans.
    pub micro_token_precision: f64,
    /// Micro-averaged token-level recall across all units with gold or predicted spans.
    pub micro_token_recall: f64,
    /// Micro-averaged token-level F1 across all units with gold or predicted spans.
    pub micro_token_f1: f64,
    /// Macro-averaged per-unit character (codepoint) precision on units with gold spans.
    pub gold_positive_char_precision: f64,
    /// Macro-averaged per-unit character (codepoint) recall on units with gold spans.
    pub gold_positive_char_recall: f64,
    /// Macro-averaged per-unit character (codepoint) F1 on units with gold spans.
    pub gold_positive_char_f1: f64,
    /// Fraction of positive predicted units whose union of evidence spans overlaps
    /// at least one gold toxic token (`0.0..=1.0`).
    pub predicted_positive_gold_overlap_rate: f64,
}

/// Evaluates the union of predicted [`Incivility`] evidence spans per unit
/// against `gold` across `views` and `tokens_per_unit`.
///
/// Only evidence spans whose label `l` satisfies `scores.get(l) >= thresholds[l.index()]`
/// are included in a unit's predicted span set.
pub fn evaluate_toxic_spans(
    gold: &[GoldToxicSpanUnit],
    views: &[&View],
    tokens_per_unit: &[&Tokens],
    predicted: &Incivility,
    thresholds: &[f64; 6],
) -> Result<ToxicSpanMetrics, String> {
    let n = gold.len();
    if views.len() != n || tokens_per_unit.len() != n || predicted.len() != n {
        return Err(format!(
            "mismatched lengths: {} gold, {} views, {} token layers, {} predictions",
            n,
            views.len(),
            tokens_per_unit.len(),
            predicted.len()
        ));
    }
    if n == 0 {
        return Ok(ToxicSpanMetrics::default());
    }

    let mut units_with_gold = 0usize;
    let mut units_with_pred = 0usize;
    let mut pred_pos_on_gold_pos = 0usize;
    let mut pred_pos_overlapping_gold = 0usize;

    let (mut sum_tok_p, mut sum_tok_r, mut sum_tok_f1) = (0.0, 0.0, 0.0);
    let (mut tot_tok_tp, mut tot_tok_fp, mut tot_tok_fn) = (0usize, 0usize, 0usize);
    let (mut sum_char_p, mut sum_char_r, mut sum_char_f1) = (0.0, 0.0, 0.0);

    for i in 0..n {
        let g = &gold[i];
        let view = views[i];
        let tokens = tokens_per_unit[i];
        if tokens.spans.view() != view.id() {
            return Err(format!("unit {i}: tokens belong to another view"));
        }
        let n_toks = tokens.len();
        let (tok_starts, tok_ends) = tokens.spans.columns();

        let mut gold_tok_mask = vec![false; n_toks];
        for r in &g.token_ranges {
            let s = r.start.get().min(n_toks);
            let e = r.end.get().min(n_toks);
            for flag in &mut gold_tok_mask[s..e] {
                *flag = true;
            }
        }

        let ix = Ix::<Incivility>::new(i as u32);
        let scores = predicted.scores(ix);
        let mut pred_tok_mask = vec![false; n_toks];
        for (lbl, r) in predicted.evidence_of(ix) {
            if scores.get(lbl) >= thresholds[lbl as usize] {
                let s = r.start.get().min(n_toks);
                let e = r.end.get().min(n_toks);
                for flag in &mut pred_tok_mask[s..e] {
                    *flag = true;
                }
            }
        }

        let gold_tok_count = gold_tok_mask.iter().filter(|&&b| b).count();
        let pred_tok_count = pred_tok_mask.iter().filter(|&&b| b).count();
        let tok_tp = gold_tok_mask.iter().zip(&pred_tok_mask).filter(|&(&a, &b)| a && b).count();
        let tok_fp = pred_tok_count - tok_tp;
        let tok_fn = gold_tok_count - tok_tp;

        tot_tok_tp += tok_tp;
        tot_tok_fp += tok_fp;
        tot_tok_fn += tok_fn;

        let (p_t, r_t, f1_t) = prf1_semeval(tok_tp, pred_tok_count, gold_tok_count);
        sum_tok_p += p_t;
        sum_tok_r += r_t;
        sum_tok_f1 += f1_t;

        if pred_tok_count > 0 {
            units_with_pred += 1;
        }
        if !g.utf8_spans.is_empty() || gold_tok_count > 0 {
            units_with_gold += 1;
            if pred_tok_count > 0 {
                pred_pos_on_gold_pos += 1;
                if tok_tp > 0 {
                    pred_pos_overlapping_gold += 1;
                }
            }

            let cp_len = view.offsets().code_point_len().0 as usize;
            let mut gold_cp_mask = vec![false; cp_len];
            for &sp in &g.utf8_spans {
                let cp = view.span_to_code_point(sp);
                let s = (cp.start.0 as usize).min(cp_len);
                let e = (cp.end.0 as usize).min(cp_len);
                for flag in &mut gold_cp_mask[s..e] {
                    *flag = true;
                }
            }
            let mut pred_cp_mask = vec![false; cp_len];
            for (t_idx, &flagged) in pred_tok_mask.iter().enumerate() {
                if flagged {
                    let sp = Span::new(tok_starts[t_idx], tok_ends[t_idx]);
                    let cp = view.span_to_code_point(sp);
                    let s = (cp.start.0 as usize).min(cp_len);
                    let e = (cp.end.0 as usize).min(cp_len);
                    for flag in &mut pred_cp_mask[s..e] {
                        *flag = true;
                    }
                }
            }
            let gold_cp = gold_cp_mask.iter().filter(|&&b| b).count();
            let pred_cp = pred_cp_mask.iter().filter(|&&b| b).count();
            let cp_tp = gold_cp_mask.iter().zip(&pred_cp_mask).filter(|&(&a, &b)| a && b).count();
            let (p_c, r_c, f1_c) = prf1_semeval(cp_tp, pred_cp, gold_cp);
            sum_char_p += p_c;
            sum_char_r += r_c;
            sum_char_f1 += f1_c;
        }
    }

    let nf = n as f64;
    let gf = units_with_gold.max(1) as f64;
    let micro_p =
        if tot_tok_tp + tot_tok_fp == 0 { 1.0 } else { (tot_tok_tp as f64) / ((tot_tok_tp + tot_tok_fp) as f64) };
    let micro_r =
        if tot_tok_tp + tot_tok_fn == 0 { 1.0 } else { (tot_tok_tp as f64) / ((tot_tok_tp + tot_tok_fn) as f64) };
    let micro_f1 = if micro_p + micro_r == 0.0 { 0.0 } else { 2.0 * micro_p * micro_r / (micro_p + micro_r) };
    let overlap_rate = if pred_pos_on_gold_pos == 0 {
        0.0
    } else {
        (pred_pos_overlapping_gold as f64) / (pred_pos_on_gold_pos as f64)
    };

    Ok(ToxicSpanMetrics {
        units: n,
        units_with_gold_spans: units_with_gold,
        units_with_predicted_spans: units_with_pred,
        token_precision: sum_tok_p / nf,
        token_recall: sum_tok_r / nf,
        token_f1: sum_tok_f1 / nf,
        micro_token_precision: micro_p,
        micro_token_recall: micro_r,
        micro_token_f1: micro_f1,
        gold_positive_char_precision: if units_with_gold == 0 { 0.0 } else { sum_char_p / gf },
        gold_positive_char_recall: if units_with_gold == 0 { 0.0 } else { sum_char_r / gf },
        gold_positive_char_f1: if units_with_gold == 0 { 0.0 } else { sum_char_f1 / gf },
        predicted_positive_gold_overlap_rate: overlap_rate,
    })
}

fn prf1_semeval(tp: usize, pred: usize, gold: usize) -> (f64, f64, f64) {
    match (pred, gold) {
        (0, 0) => (1.0, 1.0, 1.0),
        (0, _) | (_, 0) => (0.0, 0.0, 0.0),
        _ => {
            let p = (tp as f64) / (pred as f64);
            let r = (tp as f64) / (gold as f64);
            let f1 = if p + r == 0.0 { 0.0 } else { 2.0 * p * r / (p + r) };
            (p, r, f1)
        }
    }
}
