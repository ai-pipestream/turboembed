//! Validation of `TypedDocumentResult` messages (`RoundTripTypedDocument`,
//! issue #1219). Re-proves offset validity (`UTF8`, `UTF16`, `CODE_POINT`),
//! span ordering, probability bounds (`[0.0, 1.0]`), `model_ref` resolution,
//! and `indexes_layer` bounds.

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
    if !p.is_finite() || !(0.0..=1.0).contains(&p) {
        return Err(bad(format!("{layer}[{index}] {what} {p} is outside [0.0, 1.0] or non-finite")));
    }
    Ok(())
}

fn check_spans(
    offsetter: &Offsetter,
    text: &str,
    layer: &str,
    index: usize,
    frags: &[SpanFragment],
) -> Result<Vec<Span>, Failure> {
    let mut out: Vec<Span> = Vec::with_capacity(frags.len());
    for (f_ix, frag) in frags.iter().enumerate() {
        let sp = offsetter.from_wire(text, layer, frag.start, frag.end)?;
        if frags.len() > 1 && sp.is_empty() {
            return Err(bad(format!("{layer}[{index}] fragment {f_ix} is empty in a multi-fragment span")));
        }
        if let Some(&prev) = out.last()
            && sp.start < prev.end
        {
            return Err(bad(format!(
                "{layer}[{index}] fragment {f_ix} ({}..{}) overlaps or precedes previous fragment ({}..{})",
                sp.start, sp.end, prev.start, prev.end
            )));
        }
        out.push(sp);
    }
    Ok(out)
}

fn check_token_range(
    layer_by_id: &HashMap<&str, usize>,
    what: &str,
    layer: &str,
    index: usize,
    start: u32,
    end: u32,
) -> Result<(), Failure> {
    if start >= end {
        return Err(bad(format!("{layer}[{index}] {what} has empty or inverted token range {start}..{end}")));
    }
    if let Some(&tlen) = layer_by_id.get("opennlp:tokens")
        && (end as usize) > tlen
    {
        return Err(bad(format!("{layer}[{index}] {what} token_end {end} exceeds opennlp:tokens length {tlen}")));
    }
    Ok(())
}

fn check_model_ref(
    known_models: &HashSet<&str>,
    enforce_known: bool,
    where_desc: &str,
    mref: &str,
) -> Result<(), Failure> {
    if mref.trim().is_empty() {
        return Err(bad(format!("{where_desc} has an empty model_ref")));
    }
    if enforce_known && !known_models.contains(mref) {
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
    let enforce_models = !msg.models.is_empty();

    let mut layer_lengths: HashMap<(&str, &str), usize> = HashMap::with_capacity(msg.layers.len());
    let mut layer_by_id: HashMap<&str, usize> = HashMap::with_capacity(msg.layers.len());
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
        layer_by_id.entry(layer.id.as_str()).or_insert(layer.annotations.len());
        total_fragments =
            total_fragments.saturating_add(layer.annotations.iter().map(|a| a.spans.len().max(1)).sum::<usize>());
        if total_fragments as u64 > MAX_FRAGMENTS {
            return Err(over("max_fragments", "typed document fragments", total_fragments as u64));
        }
    }

    for layer in &msg.layers {
        validate_layer(msg, &offsetter, &known_models, enforce_models, &layer_lengths, &layer_by_id, layer)?;
    }

    Ok(())
}

fn validate_layer(
    msg: &TypedDocumentResult,
    offsetter: &Offsetter,
    known_models: &HashSet<&str>,
    enforce_models: bool,
    layer_lengths: &HashMap<(&str, &str), usize>,
    layer_by_id: &HashMap<&str, usize>,
    layer: &TypedLayer,
) -> Result<(), Failure> {
    let lid = layer.id.as_str();
    for mref in &layer.model_refs {
        check_model_ref(known_models, enforce_models, &format!("layer {lid}.model_refs"), mref)?;
    }

    for bound in &layer.indexes_layer {
        if bound.layer.trim().is_empty() {
            return Err(bad(format!("layer {lid} indexes_layer has an empty target layer id")));
        }
        let target_len = layer_lengths
            .get(&(bound.layer.as_str(), bound.instance.as_str()))
            .copied()
            .or_else(|| bound.instance.is_empty().then(|| layer_by_id.get(bound.layer.as_str()).copied()).flatten())
            .ok_or_else(|| {
                bad(format!(
                    "layer {lid} indexes_layer target ({:?}, {:?}) is missing from layers",
                    bound.layer, bound.instance
                ))
            })?;
        if (bound.bound as usize) > target_len {
            return Err(bad(format!(
                "layer {lid} indexes_layer ({:?}, {:?}) bound {} exceeds target layer length {target_len}",
                bound.layer, bound.instance, bound.bound
            )));
        }
    }

    if AnchorKind::try_from(layer.scope) == Ok(AnchorKind::Aligned)
        && LayerWireStatus::try_from(layer.status) == Ok(LayerWireStatus::Ok)
        && let Some(anchor) = &layer.anchor_layer
        && let Some(&anchor_len) =
            layer_lengths.get(&(anchor.as_str(), "")).or_else(|| layer_by_id.get(anchor.as_str()))
        && layer.annotations.len() != anchor_len
    {
        return Err(bad(format!(
            "aligned layer {lid} has {} annotations, but anchor_layer {anchor:?} has {anchor_len}",
            layer.annotations.len()
        )));
    }

    for (i, ann) in layer.annotations.iter().enumerate() {
        validate_annotation(&msg.raw_text, offsetter, known_models, enforce_models, layer_by_id, lid, i, ann)?;
    }

    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn validate_annotation(
    text: &str,
    offsetter: &Offsetter,
    known_models: &HashSet<&str>,
    enforce_models: bool,
    layer_by_id: &HashMap<&str, usize>,
    lid: &str,
    i: usize,
    ann: &TypedAnnotation,
) -> Result<(), Failure> {
    if let Some(mref) = &ann.model_ref {
        check_model_ref(known_models, enforce_models, &format!("{lid}[{i}].model_ref"), mref)?;
    }
    if let Some(p) = ann.probability {
        check_prob("probability", lid, i, p)?;
    }
    let spans = check_spans(offsetter, text, lid, i, &ann.spans)?;

    let Some(val) = &ann.value else {
        return Err(bad(format!("{lid}[{i}] is missing its typed annotation value")));
    };

    match val {
        Value::Word(w) => {
            check_token_range(layer_by_id, "word", lid, i, w.token_start, w.token_end)?;
            if let Some(sub) = &w.subspan {
                offsetter.from_wire(text, lid, sub.start, sub.end)?;
            }
        }
        Value::PhraseChunk(pc) => check_token_range(layer_by_id, "phrase_chunk", lid, i, pc.token_start, pc.token_end)?,
        Value::Phrase(ph) => {
            check_token_range(layer_by_id, "phrase", lid, i, ph.token_start, ph.token_end)?;
            if ph.head_token < ph.token_start || ph.head_token >= ph.token_end {
                return Err(bad(format!(
                    "{lid}[{i}] phrase head_token {} is outside {}..{}",
                    ph.head_token, ph.token_start, ph.token_end
                )));
            }
            if let Some(p) = ph.parent_index
                && (p as usize) >= i
            {
                return Err(bad(format!("{lid}[{i}] phrase parent_index {p} must be < {i}")));
            }
        }
        Value::Entity(ent) => {
            check_token_range(layer_by_id, "entity", lid, i, ent.token_start, ent.token_end)?;
            for (v_ix, vote) in ent.votes.iter().enumerate() {
                if let Some(vm) = &vote.model_ref {
                    check_model_ref(known_models, enforce_models, &format!("{lid}[{i}].votes[{v_ix}]"), vm)?;
                }
                if let Some(vp) = vote.probability {
                    check_prob("vote.probability", lid, i, vp)?;
                }
                if let Some(vc) = vote.calibrated_probability {
                    check_prob("vote.calibrated_probability", lid, i, vc)?;
                }
                check_spans(offsetter, text, lid, i, &vote.spans)?;
            }
        }
        Value::CorefMention(cm) => {
            check_token_range(layer_by_id, "coref_mention", lid, i, cm.token_start, cm.token_end)?;
            if cm.head_token < cm.token_start || cm.head_token >= cm.token_end {
                return Err(bad(format!(
                    "{lid}[{i}] coref_mention head_token {} is outside {}..{}",
                    cm.head_token, cm.token_start, cm.token_end
                )));
            }
        }
        Value::CorefChain(cc) => {
            if cc.mention_indices.is_empty() {
                return Err(bad(format!("{lid}[{i}] coref_chain has no mention_indices")));
            }
        }
        Value::GlossaryTerm(gt) => {
            check_token_range(layer_by_id, "glossary_term", lid, i, gt.token_start, gt.token_end)?
        }
        Value::Number(num) => {
            if let Some(ns) = &num.numeral_span {
                offsetter.from_wire(text, lid, ns.start, ns.end)?;
            }
        }
        Value::RegionVote(rv) => {
            check_prob("share", lid, i, rv.share)?;
            if rv.share <= 0.0 {
                return Err(bad(format!("{lid}[{i}] region_vote share must be > 0.0")));
            }
        }
        Value::LanguageSpan(ls) => check_prob("confidence", lid, i, ls.confidence)?,
        Value::Category(cat) => {
            for sc in &cat.scores {
                if !sc.score.is_finite() || sc.score < 0.0 {
                    return Err(bad(format!("{lid}[{i}] category score {} is negative or non-finite", sc.score)));
                }
            }
        }
        Value::Emotion(em) => {
            for &s in &em.scores {
                check_prob("emotion score", lid, i, s)?;
            }
            for ev in &em.evidence {
                check_token_range(layer_by_id, "emotion evidence", lid, i, ev.token_start, ev.token_end)?;
            }
        }
        Value::Aspect(asp) => {
            if let (Some(ts), Some(te)) = (asp.token_start, asp.token_end) {
                check_token_range(layer_by_id, "aspect", lid, i, ts, te)?;
            }
            for (name, v) in [("positive", asp.positive), ("negative", asp.negative), ("neutral", asp.neutral)] {
                if !v.is_finite() || v < 0.0 {
                    return Err(bad(format!("{lid}[{i}] aspect {name} score {v} is negative or non-finite")));
                }
            }
        }
        Value::Opinion(op) => {
            check_token_range(layer_by_id, "opinion", lid, i, op.token_start, op.token_end)?;
            if !op.polarity.is_finite() || !op.intensity.is_finite() || op.intensity < 0.0 {
                return Err(bad(format!("{lid}[{i}] opinion has non-finite polarity or invalid intensity")));
            }
        }
        Value::Incivility(inc) => {
            for (name, v) in [
                ("insult", inc.insult),
                ("profanity", inc.profanity),
                ("threat", inc.threat),
                ("identity_attack", inc.identity_attack),
                ("intolerance", inc.intolerance),
                ("severe_toxicity", inc.severe_toxicity),
            ] {
                check_prob(name, lid, i, v)?;
            }
            for ev in &inc.evidence {
                check_token_range(layer_by_id, "incivility evidence", lid, i, ev.token_start, ev.token_end)?;
            }
        }
        Value::Pragmatics(pr) => {
            check_prob("score", lid, i, pr.score)?;
            if let (Some(ps), Some(pe)) = (pr.positive_token_start, pr.positive_token_end) {
                check_token_range(layer_by_id, "pragmatics positive", lid, i, ps, pe)?;
            }
            if let (Some(ns), Some(ne)) = (pr.negative_token_start, pr.negative_token_end) {
                check_token_range(layer_by_id, "pragmatics negative", lid, i, ns, ne)?;
            }
        }
        Value::Noise(n) => {
            check_prob("score", lid, i, n.score)?;
            if n.score <= 0.0 {
                return Err(bad(format!("{lid}[{i}] noise score must be > 0.0")));
            }
        }
        Value::Asset(asset) => {
            if let Some(ps) = &asset.payload_span {
                let p_span = offsetter.from_wire(text, lid, ps.start, ps.end)?;
                if p_span.is_empty() {
                    return Err(bad(format!("{lid}[{i}] asset payload_span is empty")));
                }
                if let Some(&outer) = spans.first()
                    && !outer.contains(p_span)
                {
                    return Err(bad(format!("{lid}[{i}] asset payload_span lies outside outer span")));
                }
            }
        }
        Value::RuleMatch(m) => check_token_range(layer_by_id, "rule_match", lid, i, m.token_start, m.token_end)?,
        Value::Boilerplate(bp) => check_f32_prob("score", lid, i, bp.score)?,
        Value::Confidence(c) => check_prob("confidence", lid, i, c.confidence)?,
        Value::Keyword(kw) => check_prob("score", lid, i, kw.score)?,
        Value::SourceEncoding(se) => check_f32_prob("confidence", lid, i, se.confidence)?,
        Value::Vector(vec_ann) => {
            if vec_ann.values.len() != vec_ann.dim as usize {
                return Err(bad(format!(
                    "{lid}[{i}] vector values length {} does not match dim {}",
                    vec_ann.values.len(),
                    vec_ann.dim
                )));
            }
        }
        Value::Chunk(ch) => {
            if ch.sentence_start > ch.sentence_end {
                return Err(bad(format!(
                    "{lid}[{i}] chunk sentence range {}..{} is inverted",
                    ch.sentence_start, ch.sentence_end
                )));
            }
            if let Some(p) = ch.parent_index
                && (p as usize) >= i
            {
                return Err(bad(format!("{lid}[{i}] chunk parent_index {p} must be < {i}")));
            }
        }
        Value::LevelUnit(lv) => {
            if let Some(p) = lv.parent_index
                && (p as usize) >= i
            {
                return Err(bad(format!("{lid}[{i}] level parent_index {p} must be < {i}")));
            }
        }
        Value::Structure(st) => {
            if let Some(p) = st.parent_index
                && (p as usize) >= i
            {
                return Err(bad(format!("{lid}[{i}] structure parent_index {p} must be < {i}")));
            }
        }
        Value::Answer(ans) => {
            check_prob("probability", lid, i, ans.probability)?;
            check_prob("null_probability", lid, i, ans.null_probability)?;
        }
        Value::Group(grp) => {
            if grp.members.is_empty() {
                return Err(bad(format!("{lid}[{i}] group has no members")));
            }
        }
        Value::LinkedEntity(le) => check_prob("score", lid, i, le.score)?,
        Value::Calibrated(cal) => {
            if cal.raw.len() != cal.calibrated.len() {
                return Err(bad(format!("{lid}[{i}] calibrated raw/calibrated length mismatch")));
            }
            for &p in &cal.calibrated {
                check_prob("calibrated", lid, i, p)?;
            }
        }
        Value::Utf16Span(u16s) if u16s.start16 > u16s.end16 => {
            return Err(bad(format!("{lid}[{i}] utf16_span {}..{} is inverted", u16s.start16, u16s.end16)));
        }
        _ => {}
    }

    Ok(())
}
