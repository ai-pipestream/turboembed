//! Validation of `TypedDocumentResult` messages (`RoundTripTypedDocument`,
//! issue #1219). Re-proves offset validity (`UTF8`, `UTF16`, `CODE_POINT`),
//! span ordering, probability bounds (`[0.0, 1.0]`), `model_ref` resolution,
//! aligned anchor integrity, and `indexes_layer` bounds.

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

fn check_token_range(tlen: Option<usize>, what: &str, lid: &str, i: usize, s: u32, e: u32) -> Result<(), Failure> {
    if s >= e {
        return Err(bad(format!("{lid}[{i}] {what} has empty or inverted token range {s}..{e}")));
    }
    if tlen.is_some_and(|lim| (e as usize) > lim) {
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
    token_len: Option<usize>,
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
    tlen: Option<usize>,
    what: &str,
    lid: &str,
    i: usize,
    s: Option<u32>,
    e: Option<u32>,
) -> Result<(), Failure> {
    if let (Some(ts), Some(te)) = (s, e) {
        check_token_range(tlen, what, lid, i, ts, te)?;
    }
    Ok(())
}

fn check_index_slice(what: &str, lid: &str, i: usize, indices: &[u32], lim: Option<usize>) -> Result<(), Failure> {
    if indices.is_empty() {
        return Err(bad(format!("{lid}[{i}] {what} is empty")));
    }
    if let Some(l) = lim
        && let Some(&bad_m) = indices.iter().find(|&&m| (m as usize) >= l)
    {
        return Err(bad(format!("{lid}[{i}] {what} index {bad_m} exceeds target {l}")));
    }
    Ok(())
}

fn validate_layer(
    msg: &TypedDocumentResult,
    offsetter: &Offsetter,
    known_models: &HashSet<&str>,
    layer_lengths: &HashMap<(&str, &str), usize>,
    layer: &TypedLayer,
) -> Result<(), Failure> {
    let lid = layer.id.as_str();
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

    let token_len = layer
        .indexes_layer
        .iter()
        .find(|b| b.layer == "opennlp:tokens")
        .and_then(|b| layer_lengths.get(&("opennlp:tokens", b.instance.as_str())).copied())
        .or_else(|| {
            layer_lengths
                .get(&("opennlp:tokens", layer.instance.as_str()))
                .or_else(|| layer_lengths.get(&("opennlp:tokens", "")))
                .copied()
        });

    let ctx = LayerCtx {
        text: &msg.raw_text,
        offsetter,
        known_models,
        layer_lengths,
        token_len,
        target_bounds,
        layer_len: layer.annotations.len(),
        lid,
    };
    for (i, ann) in layer.annotations.iter().enumerate() {
        validate_annotation(&ctx, i, ann)?;
    }
    Ok(())
}

fn validate_annotation(ctx: &LayerCtx, i: usize, ann: &TypedAnnotation) -> Result<(), Failure> {
    let LayerCtx { text, offsetter, known_models, layer_lengths, token_len, ref target_bounds, layer_len, lid } = *ctx;
    if let Some(mref) = &ann.model_ref {
        check_model_ref(known_models, &format!("{lid}[{i}].model_ref"), mref)?;
    }
    if let Some(p) = ann.probability {
        check_prob("probability", lid, i, p)?;
    }
    let spans = check_spans(offsetter, text, lid, i, &ann.spans)?;

    let Some(val) = &ann.value else {
        return Err(bad(format!("{lid}[{i}] is missing its typed annotation value")));
    };

    match val {
        Value::PosTag(p) => check_aligned_index("target_index", lid, i, p.target_index, layer_len)?,
        Value::MorphFeatures(m) => check_aligned_index("target_index", lid, i, m.target_index, layer_len)?,
        Value::Lemma(l) => check_aligned_index("target_index", lid, i, l.target_index, layer_len)?,
        Value::Stem(s) => check_aligned_index("token_index", lid, i, s.token_index, layer_len)?,
        Value::Term(t) => check_aligned_index("token_index", lid, i, t.token_index, layer_len)?,
        Value::ChunkTag(c) => check_aligned_index("token_index", lid, i, c.token_index, layer_len)?,
        Value::StopwordFlag(sw) => check_aligned_index("token_index", lid, i, sw.token_index, layer_len)?,
        Value::WordType(wt) => check_aligned_index("token_index", lid, i, wt.token_index, layer_len)?,
        Value::Reading(r) => check_aligned_index("token_index", lid, i, r.token_index, layer_len)?,
        Value::Boundary(b) => check_aligned_index("sentence_index", lid, i, b.sentence_index, layer_len)?,
        Value::SparseTerms(st) => check_aligned_index("target_index", lid, i, st.target_index, layer_len)?,
        Value::Nbest(nb) => check_aligned_index("target_index", lid, i, nb.target_index, layer_len)?,
        Value::Word(w) => {
            check_token_range(token_len, "word", lid, i, w.token_start, w.token_end)?;
            if let Some(sub) = &w.subspan {
                offsetter.from_wire(text, lid, sub.start, sub.end)?;
            }
        }
        Value::PhraseChunk(pc) => check_token_range(token_len, "phrase_chunk", lid, i, pc.token_start, pc.token_end)?,
        Value::Phrase(ph) => {
            check_token_range(token_len, "phrase", lid, i, ph.token_start, ph.token_end)?;
            check_head_in_range("phrase", lid, i, ph.head_token, ph.token_start, ph.token_end)?;
            check_parent("phrase", lid, i, ph.parent_index)?;
        }
        Value::Entity(ent) => {
            check_token_range(token_len, "entity", lid, i, ent.token_start, ent.token_end)?;
            for (v_ix, vote) in ent.votes.iter().enumerate() {
                if let Some(vm) = &vote.model_ref {
                    check_model_ref(known_models, &format!("{lid}[{i}].votes[{v_ix}]"), vm)?;
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
        Value::Dependency(dep) => {
            check_aligned_index("target_index", lid, i, dep.target_index, layer_len)?;
            if dep.head_index.is_some_and(|h| (h as usize) >= layer_len) {
                return Err(bad(format!("{lid}[{i}] dependency head_index exceeds layer length {layer_len}")));
            }
        }
        Value::Edge(edge) => {
            for (what, idx, lim) in [
                ("from_index", edge.from_index, target_bounds.first()),
                ("to_index", edge.to_index, target_bounds.get(1).or_else(|| target_bounds.first())),
            ] {
                if let Some(&l) = lim
                    && (idx as usize) >= l
                {
                    return Err(bad(format!("{lid}[{i}] edge {what} {idx} exceeds target {l}")));
                }
            }
        }
        Value::CorefMention(cm) => {
            check_token_range(token_len, "coref_mention", lid, i, cm.token_start, cm.token_end)?;
            check_head_in_range("coref_mention", lid, i, cm.head_token, cm.token_start, cm.token_end)?;
        }
        Value::CorefChain(cc) => {
            let lim = target_bounds.first().or_else(|| layer_lengths.get(&("turbo:mentions", ""))).copied();
            check_index_slice("coref_chain mention_indices", lid, i, &cc.mention_indices, lim)?;
        }
        Value::GlossaryTerm(gt) => check_token_range(token_len, "glossary_term", lid, i, gt.token_start, gt.token_end)?,
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
                check_nonneg("category score", lid, i, sc.score)?;
            }
        }
        Value::Emotion(em) => {
            for &s in &em.scores {
                check_prob("emotion score", lid, i, s)?;
            }
            for ev in &em.evidence {
                check_token_range(token_len, "emotion evidence", lid, i, ev.token_start, ev.token_end)?;
            }
        }
        Value::Aspect(asp) => {
            check_opt_tokens(token_len, "aspect", lid, i, asp.token_start, asp.token_end)?;
            for (name, v) in [("positive", asp.positive), ("negative", asp.negative), ("neutral", asp.neutral)] {
                check_nonneg(name, lid, i, v)?;
            }
        }
        Value::Opinion(op) => {
            check_token_range(token_len, "opinion", lid, i, op.token_start, op.token_end)?;
            if !op.polarity.is_finite() || !op.intensity.is_finite() || op.intensity < 0.0 {
                return Err(bad(format!("{lid}[{i}] opinion has non-finite polarity or invalid intensity")));
            }
        }
        Value::Incivility(inc) => {
            for (n, v) in [
                ("insult", inc.insult),
                ("profanity", inc.profanity),
                ("threat", inc.threat),
                ("identity_attack", inc.identity_attack),
                ("intolerance", inc.intolerance),
                ("severe_toxicity", inc.severe_toxicity),
            ] {
                check_prob(n, lid, i, v)?;
            }
            for ev in &inc.evidence {
                check_token_range(token_len, "incivility evidence", lid, i, ev.token_start, ev.token_end)?;
            }
        }
        Value::Pragmatics(pr) => {
            check_aligned_index("sentence_index", lid, i, pr.sentence_index, layer_len)?;
            check_prob("score", lid, i, pr.score)?;
            check_opt_tokens(token_len, "pragmatics positive", lid, i, pr.positive_token_start, pr.positive_token_end)?;
            check_opt_tokens(token_len, "pragmatics negative", lid, i, pr.negative_token_start, pr.negative_token_end)?;
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
                if p_span.is_empty() || spans.first().is_some_and(|o| !o.contains(p_span)) {
                    return Err(bad(format!("{lid}[{i}] asset payload_span is empty or outside outer span")));
                }
            }
        }
        Value::RuleMatch(m) => check_token_range(token_len, "rule_match", lid, i, m.token_start, m.token_end)?,
        Value::Boilerplate(bp) => {
            check_aligned_index("target_index", lid, i, bp.target_index, layer_len)?;
            check_f32_prob("score", lid, i, bp.score)?;
        }
        Value::Confidence(c) => {
            check_aligned_index("target_index", lid, i, c.target_index, layer_len)?;
            check_prob("confidence", lid, i, c.confidence)?;
        }
        Value::Keyword(kw) => check_prob("score", lid, i, kw.score)?,
        Value::SourceEncoding(se) => check_f32_prob("confidence", lid, i, se.confidence)?,
        Value::Vector(v) => {
            check_aligned_index("target_index", lid, i, v.target_index, layer_len)?;
            if v.values.len() != v.dim as usize {
                return Err(bad(format!("{lid}[{i}] vector values length {} != dim {}", v.values.len(), v.dim)));
            }
        }
        Value::Chunk(ch) => {
            if ch.sentence_start > ch.sentence_end {
                return Err(bad(format!(
                    "{lid}[{i}] chunk sentence range {}..{} is inverted",
                    ch.sentence_start, ch.sentence_end
                )));
            }
            check_parent("chunk", lid, i, ch.parent_index)?;
        }
        Value::LevelUnit(lv) => check_parent("level", lid, i, lv.parent_index)?,
        Value::Structure(st) => check_parent("structure", lid, i, st.parent_index)?,
        Value::Answer(ans) => {
            check_prob("probability", lid, i, ans.probability)?;
            check_prob("null_probability", lid, i, ans.null_probability)?;
        }
        Value::Group(grp) => check_index_slice("group members", lid, i, &grp.members, target_bounds.first().copied())?,
        Value::LinkedEntity(le) => {
            check_aligned_index("entity_index", lid, i, le.entity_index, layer_len)?;
            check_prob("score", lid, i, le.score)?;
        }
        Value::Calibrated(cal) => {
            check_aligned_index("target_index", lid, i, cal.target_index, layer_len)?;
            if cal.raw.len() != cal.calibrated.len() {
                return Err(bad(format!("{lid}[{i}] calibrated raw/calibrated length mismatch")));
            }
            for &p in &cal.calibrated {
                check_prob("calibrated", lid, i, p)?;
            }
        }
        Value::Utf16Span(u16s) => {
            check_aligned_index("target_index", lid, i, u16s.target_index, layer_len)?;
            if u16s.start16 > u16s.end16 {
                return Err(bad(format!("{lid}[{i}] utf16_span {}..{} is inverted", u16s.start16, u16s.end16)));
            }
        }
        Value::Modality(m) => {
            check_f32_prob("confidence", lid, i, m.confidence)?;
            if let Some(cue) = &m.cue {
                offsetter.from_wire(text, lid, cue.start, cue.end)?;
            }
        }
        Value::Event(ev) => {
            check_f32_prob("confidence", lid, i, ev.confidence)?;
            for sp in ev.arguments.iter().filter_map(|a| a.span.as_ref()) {
                offsetter.from_wire(text, lid, sp.start, sp.end)?;
            }
        }
        Value::LexicalExpansion(le) => check_prob("weight", lid, i, le.weight)?,
        Value::RecordValue(rv) => check_prob("score", lid, i, rv.score)?,
        _ => {}
    }

    Ok(())
}
