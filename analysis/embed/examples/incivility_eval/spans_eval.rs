//! Loading `ipavlopoulos/toxic_spans` (`beaf170`) and projecting character
//! (Unicode codepoint) spans through [`View`] and `opennlp:tokens` (`#890`).

use std::collections::HashMap;
use std::fs;
use std::path::Path;

use serde_json::{Value, json};
use turbo_analysis::{Document, Ix, Span, Spans, View, ViewId};
use turbo_analysis_layers::{Incivility, IncivilityItem, IncivilityTarget, Tokens};
use turbo_doccat::{
    GoldToxicSpanUnit, INCIVILITY_DIMENSIONS, IncivilityAnnotator, ToxicSpanMetrics, char_span_to_token_range,
    char_span_to_utf8_span, evaluate_toxic_spans,
};

use super::data::{self, Unit};

pub struct ToxicSpansDataset {
    pub acl_texts: Vec<String>,
    pub acl_views: Vec<View>,
    pub acl_tokens: Vec<Tokens>,
    pub acl_token_lower: Vec<Vec<String>>,
    pub acl_gold: Vec<GoldToxicSpanUnit>,
    pub acl_non_ascii: usize,
    pub raw_texts: Vec<String>,
    pub raw_views: Vec<View>,
    pub raw_tokens: Vec<Tokens>,
    pub raw_token_lower: Vec<Vec<String>>,
    pub raw_gold: Vec<GoldToxicSpanUnit>,
    pub raw_non_ascii: usize,
}

fn token_lower_of(text: &str, spans: &[Span]) -> Vec<String> {
    spans
        .iter()
        .map(|s| {
            text[s.start as usize..s.end as usize]
                .trim_matches(|c: char| !c.is_alphanumeric() && c != '\'')
                .to_ascii_lowercase()
        })
        .collect()
}

fn build_unit(
    text: String,
    cp_ranges: &[(u32, u32)],
) -> Result<(String, View, Tokens, Vec<String>, GoldToxicSpanUnit), String> {
    let doc = Document::new(text.as_str());
    let mut raw_spans = Vec::new();
    turbo_tokenize::simple_spans(&text, turbo_tokenize::TokenizerOptions::default(), &mut raw_spans);
    let mut t_spans = Spans::new(ViewId::SOURCE);
    for s in &raw_spans {
        t_spans.push(Span::new(s.start, s.end));
    }
    let tok_lower = token_lower_of(&text, &raw_spans);
    let view = doc.get::<View>().expect("source view").clone();
    let tokens = Tokens { spans: t_spans, sentence: None };

    let mut utf8_spans = Vec::with_capacity(cp_ranges.len());
    let mut token_ranges = Vec::with_capacity(cp_ranges.len());
    for &(cs, ce) in cp_ranges {
        let b_span = char_span_to_utf8_span(&view, cs, ce)?;
        utf8_spans.push(b_span);
        if let Some(tr) = char_span_to_token_range(&view, &tokens, cs, ce)? {
            token_ranges.push(tr);
        }
    }
    Ok((text, view, tokens, tok_lower, GoldToxicSpanUnit { utf8_spans, token_ranges }))
}

fn mask_to_cp_ranges(mask: &[bool]) -> Vec<(u32, u32)> {
    let mut ranges = Vec::new();
    let mut i = 0usize;
    while i < mask.len() {
        if mask[i] {
            let start = i;
            while i < mask.len() && mask[i] {
                i += 1;
            }
            ranges.push((start as u32, i as u32));
        } else {
            i += 1;
        }
    }
    ranges
}

fn parse_position_list(s: &str, cp_len: usize) -> Result<Vec<(u32, u32)>, String> {
    let trimmed = s.trim().trim_start_matches('[').trim_end_matches(']').trim();
    if trimmed.is_empty() {
        return Ok(Vec::new());
    }
    let mut mask = vec![false; cp_len];
    for part in trimmed.split(',') {
        let p: usize = part.trim().parse().map_err(|e| format!("bad position {part:?}: {e}"))?;
        if p < cp_len {
            mask[p] = true;
        }
    }
    Ok(mask_to_cp_ranges(&mask))
}

pub fn load_toxic_spans(dir: &Path) -> Result<ToxicSpansDataset, String> {
    let acl_path = dir.join("ACL2022/data/toxic_spans.csv");
    let acl_text = fs::read_to_string(&acl_path).map_err(|e| format!("{}: {e}", acl_path.display()))?;
    let acl_rows = data::csv_records(&acl_text)?;
    let h = &acl_rows[0];
    let pos_col = h.iter().position(|c| c == "position").ok_or("no position column")?;
    let txt_col = h.iter().position(|c| c == "text_of_post").ok_or("no text_of_post column")?;

    let mut acl_texts = Vec::with_capacity(acl_rows.len() - 1);
    let mut acl_views = Vec::with_capacity(acl_rows.len() - 1);
    let mut acl_tokens = Vec::with_capacity(acl_rows.len() - 1);
    let mut acl_token_lower = Vec::with_capacity(acl_rows.len() - 1);
    let mut acl_gold = Vec::with_capacity(acl_rows.len() - 1);
    let mut acl_non_ascii = 0usize;

    for r in &acl_rows[1..] {
        let text = r[txt_col].clone();
        if !text.is_ascii() {
            acl_non_ascii += 1;
        }
        let cp_len = text.chars().count();
        let cp_ranges = parse_position_list(&r[pos_col], cp_len)?;
        let (t, v, tok, tl, g) = build_unit(text, &cp_ranges)?;
        acl_texts.push(t);
        acl_views.push(v);
        acl_tokens.push(tok);
        acl_token_lower.push(tl);
        acl_gold.push(g);
    }

    let comments_rows = data::csv_records(
        &fs::read_to_string(dir.join("data/comments.csv")).map_err(|e| format!("comments.csv: {e}"))?,
    )?;
    let annotations_rows = data::csv_records(
        &fs::read_to_string(dir.join("data/annotations.csv")).map_err(|e| format!("annotations.csv: {e}"))?,
    )?;
    let spans_rows =
        data::csv_records(&fs::read_to_string(dir.join("data/spans.csv")).map_err(|e| format!("spans.csv: {e}"))?)?;

    let mut cid_to_anns: HashMap<String, Vec<u32>> = HashMap::new();
    for r in &annotations_rows[1..] {
        let aid: u32 = r[0].trim().parse().map_err(|e| format!("annotation id: {e}"))?;
        cid_to_anns.entry(r[1].trim().to_owned()).or_default().push(aid);
    }
    let mut spans_by_ann: HashMap<u32, Vec<(usize, usize)>> = HashMap::new();
    for r in &spans_rows[1..] {
        let aid: u32 = r[0].trim().parse().map_err(|e| format!("span annotation id: {e}"))?;
        let s: usize = r[2].trim().parse().map_err(|e| format!("span start: {e}"))?;
        let e: usize = r[3].trim().parse().map_err(|e| format!("span end: {e}"))?;
        spans_by_ann.entry(aid).or_default().push((s, e));
    }

    let mut raw_texts = Vec::with_capacity(comments_rows.len() - 1);
    let mut raw_views = Vec::with_capacity(comments_rows.len() - 1);
    let mut raw_tokens = Vec::with_capacity(comments_rows.len() - 1);
    let mut raw_token_lower = Vec::with_capacity(comments_rows.len() - 1);
    let mut raw_gold = Vec::with_capacity(comments_rows.len() - 1);
    let mut raw_non_ascii = 0usize;

    for r in &comments_rows[1..] {
        let cid = r[0].trim();
        let text = r[1].clone();
        if !text.is_ascii() {
            raw_non_ascii += 1;
        }
        let cp_len = text.chars().count();
        let anns = cid_to_anns.get(cid).map(Vec::as_slice).unwrap_or(&[]);
        let k = anns.len();
        let mut counts = vec![0usize; cp_len];
        for &aid in anns {
            let mut marked = vec![false; cp_len];
            if let Some(sp_list) = spans_by_ann.get(&aid) {
                for &(s, e) in sp_list {
                    if s <= e && e <= cp_len {
                        for flag in &mut marked[s..e] {
                            *flag = true;
                        }
                    }
                }
            }
            for (c, &m) in counts.iter_mut().zip(&marked) {
                if m {
                    *c += 1;
                }
            }
        }
        let maj_mask: Vec<bool> = counts.into_iter().map(|c| k > 0 && c * 2 > k).collect();
        let cp_ranges = mask_to_cp_ranges(&maj_mask);
        let (t, v, tok, tl, g) = build_unit(text, &cp_ranges)?;
        raw_texts.push(t);
        raw_views.push(v);
        raw_tokens.push(tok);
        raw_token_lower.push(tl);
        raw_gold.push(g);
    }

    Ok(ToxicSpansDataset {
        acl_texts,
        acl_views,
        acl_tokens,
        acl_token_lower,
        acl_gold,
        acl_non_ascii,
        raw_texts,
        raw_views,
        raw_tokens,
        raw_token_lower,
        raw_gold,
        raw_non_ascii,
    })
}

#[allow(clippy::too_many_arguments)]
pub fn eval_subset(
    annotator: &IncivilityAnnotator,
    views: &[View],
    tokens: &[Tokens],
    token_lower: &[Vec<String>],
    gold: &[GoldToxicSpanUnit],
    vectors: &[f32],
    dim: usize,
    thresholds: &[f64; INCIVILITY_DIMENSIONS],
) -> Result<Value, String> {
    let mut layer = Incivility::new();
    for (i, tl) in token_lower.iter().enumerate() {
        let v = &vectors[i * dim..(i + 1) * dim];
        let (scores, evidence) = annotator.score_unit_in_context(tl, 0, tl.len(), None, Some(v), None)?;
        layer.push(IncivilityItem {
            target: IncivilityTarget::Sentence(Ix::new(i as u32)),
            scores,
            evidence: &evidence,
        });
    }
    let view_refs: Vec<&View> = views.iter().collect();
    let tok_refs: Vec<&Tokens> = tokens.iter().collect();
    let m: ToxicSpanMetrics = evaluate_toxic_spans(gold, &view_refs, &tok_refs, &layer, thresholds)?;

    let gold_pos_indices: Vec<usize> =
        gold.iter().enumerate().filter(|(_, g)| !g.token_ranges.is_empty()).map(|(i, _)| i).collect();
    let mut gp_gold = Vec::with_capacity(gold_pos_indices.len());
    let mut gp_views = Vec::with_capacity(gold_pos_indices.len());
    let mut gp_toks = Vec::with_capacity(gold_pos_indices.len());
    let mut gp_layer = Incivility::new();
    for (new_idx, &orig_idx) in gold_pos_indices.iter().enumerate() {
        gp_gold.push(gold[orig_idx].clone());
        gp_views.push(&views[orig_idx]);
        gp_toks.push(&tokens[orig_idx]);
        let ix = Ix::new(orig_idx as u32);
        let ev: Vec<_> = layer.evidence_of(ix).collect();
        gp_layer.push(IncivilityItem {
            target: IncivilityTarget::Sentence(Ix::new(new_idx as u32)),
            scores: layer.scores(ix),
            evidence: &ev,
        });
    }
    let gp_m = evaluate_toxic_spans(&gp_gold, &gp_views, &gp_toks, &gp_layer, thresholds)?;

    Ok(json!({
        "units": m.units,
        "units_with_gold_spans": m.units_with_gold_spans,
        "units_with_predicted_spans": m.units_with_predicted_spans,
        "all_units_token_macro": {
            "precision": m.token_precision,
            "recall": m.token_recall,
            "f1": m.token_f1,
        },
        "gold_positive_token_macro": {
            "precision": gp_m.token_precision,
            "recall": gp_m.token_recall,
            "f1": gp_m.token_f1,
        },
        "micro_token": {
            "precision": m.micro_token_precision,
            "recall": m.micro_token_recall,
            "f1": m.micro_token_f1,
        },
        "gold_positive_char_macro": {
            "precision": m.gold_positive_char_precision,
            "recall": m.gold_positive_char_recall,
            "f1": m.gold_positive_char_f1,
        },
        "predicted_positive_gold_overlap_rate": m.predicted_positive_gold_overlap_rate,
    }))
}

pub fn build_parent_lookup<'a>(
    train: &[Unit],
    train_v: &'a [f32],
    val: &[Unit],
    val_v: &'a [f32],
    test: &[Unit],
    test_v: &'a [f32],
    dim: usize,
) -> HashMap<String, &'a [f32]> {
    let mut map = HashMap::with_capacity(train.len() + val.len() + test.len());
    for (units, vecs) in [(train, train_v), (val, val_v), (test, test_v)] {
        for (i, u) in units.iter().enumerate() {
            if !u.id.is_empty() {
                map.insert(u.id.clone(), &vecs[i * dim..(i + 1) * dim]);
            }
        }
    }
    map
}
