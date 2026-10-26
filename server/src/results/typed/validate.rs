//! Validation of `TypedDocumentResult` messages (`RoundTripTypedDocument`,
//! issue #1219). Re-proves offset validity (`UTF8`, `UTF16`, `CODE_POINT`),
//! span ordering, probability bounds (`[0.0, 1.0]`), `model_ref` resolution,
//! aligned anchor integrity, and `indexes_layer` bounds.

mod annotation;
mod evidence;
mod quality;
mod quotations;
mod shared;

use std::collections::{HashMap, HashSet};

use turbo_analysis::Span;

use super::bad;
use crate::api::Failure;
use crate::offsets::{OffsetMode, Offsetter};
use crate::proto::result::typed_annotation::Value;
use crate::proto::result::{
    AnchorKind, LayerWireStatus, SpanFragment, TypedAnnotation, TypedDocumentResult, TypedLayer,
};
use crate::results::{MAX_FRAGMENTS, MAX_GRAPH_BYTES, over};

fn check_prob(what: &str, layer: &str, index: usize, p: f64) -> Result<(), Failure> {
    if !p.is_finite() || !(0.0..=1.0).contains(&p) {
        return Err(bad(format!("{layer}[{index}] {what} {p} is outside [0.0, 1.0] or non-finite")));
    }
    Ok(())
}

fn check_f32_prob(what: &str, layer: &str, index: usize, p: f32) -> Result<(), Failure> {
    check_prob(what, layer, index, f64::from(p))
}

fn check_parent(what: &str, layer: &str, index: usize, parent: Option<u32>) -> Result<(), Failure> {
    if let Some(p) = parent
        && (p as usize) >= index
    {
        return Err(bad(format!("{layer}[{index}] {what} parent_index {p} must be < {index}")));
    }
    Ok(())
}

fn check_head_in_range(what: &str, layer: &str, index: usize, head: u32, start: u32, end: u32) -> Result<(), Failure> {
    if head < start || head >= end {
        return Err(bad(format!("{layer}[{index}] {what} head_token {head} is outside {start}..{end}")));
    }
    Ok(())
}

fn check_spans(
    off: &Offsetter,
    text: &str,
    layer: &str,
    i: usize,
    frags: &[SpanFragment],
) -> Result<Vec<Span>, Failure> {
    let mut out: Vec<Span> = Vec::with_capacity(frags.len());
    for (f_ix, frag) in frags.iter().enumerate() {
        let sp = off.from_wire(text, layer, frag.start, frag.end)?;
        if frags.len() > 1 && sp.is_empty() {
            return Err(bad(format!("{layer}[{i}] fragment {f_ix} is empty in a multi-fragment span")));
        }
        if let Some(&prev) = out.last()
            && sp.start < prev.end
        {
            return Err(bad(format!("{layer}[{i}] fragment {f_ix} overlaps or precedes previous fragment")));
        }
        out.push(sp);
    }
    Ok(out)
}

fn check_token_range(tlen: TargetLen, what: &str, lid: &str, i: usize, s: u32, e: u32) -> Result<(), Failure> {
    if s >= e {
        return Err(bad(format!("{lid}[{i}] {what} has empty or inverted token range {s}..{e}")));
    }
    // A projection always carries the tokens layer its token ranges index;
    // without it nothing bounds the range, so refuse rather than skip.
    let Some(lim) = tlen.get(lid)? else {
        return Err(bad(format!("{lid}[{i}] {what} has a token range but no opennlp:tokens layer to bound it")));
    };
    if (e as usize) > lim {
        return Err(bad(format!("{lid}[{i}] {what} token_end {e} exceeds opennlp:tokens length")));
    }
    Ok(())
}

fn check_aligned_index(what: &str, layer: &str, index: usize, target_index: u32, len: usize) -> Result<(), Failure> {
    if (target_index as usize) >= len || (target_index as usize) != index {
        return Err(bad(format!(
            "{layer}[{index}] {what} {target_index} does not match aligned row {index} (len {len})"
        )));
    }
    Ok(())
}

fn check_model_ref(known_models: &HashSet<&str>, where_desc: &str, mref: &str) -> Result<(), Failure> {
    if mref.trim().is_empty() {
        return Err(bad(format!("{where_desc} has an empty model_ref")));
    }
    if !known_models.contains(mref) {
        return Err(bad(format!("{where_desc} references unknown model_ref {mref:?}")));
    }
    Ok(())
}

/// Validates every layer, span fragment, probability, `model_ref`, and
/// `indexes_layer` bound on `msg`.
pub fn validate_typed_document(msg: &TypedDocumentResult) -> Result<(), Failure> {
    if msg.raw_text.len() as u64 > MAX_GRAPH_BYTES {
        return Err(over("max_graph_bytes", "typed document raw_text bytes", msg.raw_text.len() as u64));
    }
    let _ = u16::try_from(msg.view).map_err(|_| bad(format!("view {} exceeds u16::MAX", msg.view)))?;
    let offsets = OffsetMode::from_proto(msg.offsets)?;
    let offsetter = Offsetter::new(&msg.raw_text, offsets);

    let mut known_models: HashSet<&str> = HashSet::with_capacity(msg.models.len());
    for (i, m) in msg.models.iter().enumerate() {
        if m.model_ref.trim().is_empty() {
            return Err(bad(format!("models[{i}] has an empty model_ref")));
        }
        if m.annotator.trim().is_empty() {
            return Err(bad(format!("models[{i}] ({}) has an empty annotator", m.model_ref)));
        }
        if !known_models.insert(m.model_ref.as_str()) {
            return Err(bad(format!("duplicate model_ref {:?} in models", m.model_ref)));
        }
    }

    let mut layer_lengths: HashMap<(&str, &str), usize> = HashMap::with_capacity(msg.layers.len());
    let mut total_fragments = 0usize;

    for (l_ix, layer) in msg.layers.iter().enumerate() {
        if layer.id.trim().is_empty() {
            return Err(bad(format!("layers[{l_ix}] has an empty id")));
        }
        let Ok(_status) = LayerWireStatus::try_from(layer.status) else {
            return Err(bad(format!("layer {:?} has unknown status {}", layer.id, layer.status)));
        };
        let Ok(_scope) = AnchorKind::try_from(layer.scope) else {
            return Err(bad(format!("layer {:?} has unknown scope {}", layer.id, layer.scope)));
        };
        if layer_lengths.insert((layer.id.as_str(), layer.instance.as_str()), layer.annotations.len()).is_some() {
            return Err(bad(format!("duplicate layer ({:?}, {:?})", layer.id, layer.instance)));
        }
        total_fragments =
            total_fragments.saturating_add(layer.annotations.iter().map(|a| a.spans.len().max(1)).sum::<usize>());
        if total_fragments as u64 > MAX_FRAGMENTS {
            return Err(over("max_fragments", "typed document fragments", total_fragments as u64));
        }
    }

    for layer in &msg.layers {
        validate_layer(msg, &offsetter, &known_models, &layer_lengths, layer)?;
    }

    Ok(())
}

struct LayerCtx<'a> {
    text: &'a str,
    offsetter: &'a Offsetter,
    known_models: &'a HashSet<&'a str>,
    layer_lengths: &'a HashMap<(&'a str, &'a str), usize>,
    token_len: TargetLen,
    entity_len: TargetLen,
    target_lengths: HashMap<(&'a str, &'a str), usize>,
    table_ids: HashSet<u32>,
    target_bounds: Vec<usize>,
    layer_len: usize,
    lid: &'a str,
}

fn check_nonneg(what: &str, layer: &str, index: usize, v: f64) -> Result<(), Failure> {
    if !v.is_finite() || v < 0.0 {
        return Err(bad(format!("{layer}[{index}] {what} {v} is negative or non-finite")));
    }
    Ok(())
}

fn check_opt_tokens(
    tlen: TargetLen,
    what: &str,
    lid: &str,
    i: usize,
    s: Option<u32>,
    e: Option<u32>,
) -> Result<(), Failure> {
    match (s, e) {
        (Some(ts), Some(te)) => check_token_range(tlen, what, lid, i, ts, te)?,
        (None, None) => {}
        _ => return Err(bad(format!("{lid}[{i}] {what} token range needs both start and end"))),
    }
    Ok(())
}

fn check_index_slice(what: &str, lid: &str, i: usize, indices: &[u32], lim: Option<usize>) -> Result<(), Failure> {
    if indices.is_empty() {
        return Err(bad(format!("{lid}[{i}] {what} is empty")));
    }
    let l = lim.ok_or_else(|| bad(format!("{lid}[{i}] {what} has no index target")))?;
    if let Some(&bad_m) = indices.iter().find(|&&m| (m as usize) >= l) {
        return Err(bad(format!("{lid}[{i}] {what} index {bad_m} exceeds target {l}")));
    }
    Ok(())
}

fn unique_target_len(
    target_lengths: &HashMap<(&str, &str), usize>,
    lid: &str,
    layer: &str,
) -> Result<Option<usize>, Failure> {
    let mut lengths = target_lengths.iter().filter(|((id, _), _)| *id == layer).map(|(_, len)| *len);
    let len = lengths.next();
    if lengths.next().is_some() {
        return Err(bad(format!("{lid} has ambiguous {layer} index targets")));
    }
    Ok(len)
}

/// One target type's length for a layer, judged where an annotation reads
/// it: a layer may bound two instances of a type, as an edge between them
/// does, so long as nothing in it reads that type's length.
#[derive(Clone, Copy)]
enum TargetLen {
    One(Option<usize>),
    Ambiguous(&'static str),
}

impl TargetLen {
    fn of(target_lengths: &HashMap<(&str, &str), usize>, layer: &'static str) -> Self {
        match unique_target_len(target_lengths, "", layer) {
            Ok(len) => TargetLen::One(len),
            Err(_) => TargetLen::Ambiguous(layer),
        }
    }

    fn get(self, lid: &str) -> Result<Option<usize>, Failure> {
        match self {
            TargetLen::One(len) => Ok(len),
            TargetLen::Ambiguous(layer) => Err(bad(format!("{lid} has ambiguous {layer} index targets"))),
        }
    }
}

fn validate_layer(
    msg: &TypedDocumentResult,
    offsetter: &Offsetter,
    known_models: &HashSet<&str>,
    layer_lengths: &HashMap<(&str, &str), usize>,
    layer: &TypedLayer,
) -> Result<(), Failure> {
    let lid = layer.id.as_str();
    quality::layer(layer)?;
    quotations::layer(layer)?;
    evidence::layer(layer)?;
    for mref in &layer.model_refs {
        check_model_ref(known_models, &format!("layer {lid}.model_refs"), mref)?;
    }

    let mut target_bounds: Vec<usize> = Vec::with_capacity(layer.indexes_layer.len());
    for bound in &layer.indexes_layer {
        if bound.layer.trim().is_empty() {
            return Err(bad(format!("layer {lid} indexes_layer has an empty target layer id")));
        }
        let Some(&target_len) = layer_lengths.get(&(bound.layer.as_str(), bound.instance.as_str())) else {
            return Err(bad(format!("layer {lid} indexes_layer target {:?} is missing", bound.layer)));
        };
        if (bound.bound as usize) > target_len {
            return Err(bad(format!(
                "layer {lid} indexes_layer {:?} bound {} > {target_len}",
                bound.layer, bound.bound
            )));
        }
        target_bounds.push(target_len);
    }

    if AnchorKind::try_from(layer.scope) == Ok(AnchorKind::Aligned)
        && LayerWireStatus::try_from(layer.status) == Ok(LayerWireStatus::Ok)
    {
        let anchor = layer
            .anchor_layer
            .as_deref()
            .filter(|a| !a.trim().is_empty())
            .ok_or_else(|| bad(format!("aligned layer {lid} is missing its anchor_layer")))?;
        let (alid, ainst) = anchor.split_once('@').unwrap_or((anchor, ""));
        let alen = if alid == "turbo:view" {
            (msg.views.iter().any(|v| v.instance == ainst) || (ainst.is_empty() && !msg.views.is_empty())).then_some(1)
        } else {
            layer_lengths.get(&(alid, ainst)).copied()
        }
        .ok_or_else(|| bad(format!("aligned layer {lid} references missing anchor_layer {anchor:?}")))?;
        if layer.annotations.len() != alen {
            return Err(bad(format!(
                "aligned layer {lid} len {} != anchor {anchor:?} len {alen}",
                layer.annotations.len()
            )));
        }
    }

    let target_lengths: HashMap<(&str, &str), usize> = layer
        .indexes_layer
        .iter()
        .zip(&target_bounds)
        .map(|(b, &n)| ((b.layer.as_str(), b.instance.as_str()), n))
        .collect();
    let token_len = match TargetLen::of(&target_lengths, "opennlp:tokens") {
        TargetLen::One(None) => TargetLen::One(
            layer_lengths
                .get(&("opennlp:tokens", layer.instance.as_str()))
                .or_else(|| layer_lengths.get(&("opennlp:tokens", "")))
                .copied(),
        ),
        bound => bound,
    };
    let entity_len = TargetLen::of(&target_lengths, "opennlp:entities");

    let ctx = LayerCtx {
        text: &msg.raw_text,
        offsetter,
        known_models,
        layer_lengths,
        token_len,
        entity_len,
        target_lengths,
        table_ids: layer
            .indexes_layer
            .iter()
            .find(|b| b.layer == "turbo:tables")
            .and_then(|b| msg.layers.iter().find(|l| l.id == b.layer && l.instance == b.instance))
            .map(|tables| {
                tables
                    .annotations
                    .iter()
                    .filter_map(|a| match &a.value {
                        Some(Value::TableCell(cell)) => Some(cell.table_id),
                        _ => None,
                    })
                    .collect()
            })
            .unwrap_or_default(),
        target_bounds,
        layer_len: layer.annotations.len(),
        lid,
    };
    for (i, ann) in layer.annotations.iter().enumerate() {
        annotation::validate_annotation(&ctx, i, ann)?;
    }
    shared::salience_order(&ctx, layer)?;
    Ok(())
}
