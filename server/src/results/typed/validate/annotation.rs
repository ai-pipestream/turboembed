use super::*;

pub(super) fn validate_annotation(ctx: &LayerCtx, i: usize, ann: &TypedAnnotation) -> Result<(), Failure> {
    let LayerCtx { text, offsetter, known_models, layer_lengths, token_len, ref target_bounds, layer_len, lid, .. } =
        *ctx;
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
        Value::Evidence(e) => evidence::record(ctx, i, ann, e)?,
        Value::TextQuality(m) => quality::metrics(ctx, ann, m, true)?,
        Value::SpanQuality(m) => quality::metrics(ctx, ann, m, false)?,
        Value::Quotation(q) => quotations::record(ctx, i, ann, q)?,
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
        Value::SparseTerms(st) => shared::sparse(ctx, i, st)?,
        Value::Nbest(nb) => shared::nbest(ctx, i, nb)?,
        Value::DocumentEntity(entity) => shared::document_entity(ctx, i, entity)?,
        Value::EntitySalience(salience) => shared::salience(ctx, i, salience)?,
        Value::ChunkTableHeader(header) => shared::table_header(ctx, i, header)?,
        Value::KeyValuePair(pair) => shared::key_value(ctx, i, pair)?,
        Value::Token(t) => {
            if let Some(sentence) = t.sentence_index {
                shared::target(ctx, "opennlp:sentences", sentence)?;
            }
        }
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
                let l = lim.copied().ok_or_else(|| bad(format!("{lid}[{i}] edge {what} has no index target")))?;
                if (idx as usize) >= l {
                    return Err(bad(format!("{lid}[{i}] edge {what} {idx} exceeds target {l}")));
                }
            }
        }
        Value::Relation(relation) => {
            shared::target(ctx, "opennlp:entities", relation.subject_entity)?;
            shared::target(ctx, "opennlp:entities", relation.object_entity)?;
            if relation.subject_entity == relation.object_entity {
                return Err(bad(format!("{lid}[{i}] relation has identical subject and object")));
            }
        }
        Value::CorefMention(cm) => {
            shared::target(ctx, "opennlp:sentences", cm.sentence_index)?;
            if let Some(entity) = cm.entity_index {
                shared::target(ctx, "opennlp:entities", entity)?;
            }
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
            if let Some(sentence) = asp.sentence_index {
                shared::target(ctx, "opennlp:sentences", sentence)?;
            }
            for &opinion in &asp.opinion_indices {
                shared::target(ctx, "turbo:opinions", opinion)?;
            }
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
            let ps = asset.payload_span.as_ref().ok_or_else(|| bad(format!("{lid}[{i}] asset has no payload_span")))?;
            let p_span = offsetter.from_wire(text, lid, ps.start, ps.end)?;
            let outer = spans.first().ok_or_else(|| bad(format!("{lid}[{i}] asset has no outer span")))?;
            // A deleted payload retains presence as an empty span, including
            // at the outer boundary. Containment still applies after projection.
            if !outer.contains(p_span) {
                return Err(bad(format!("{lid}[{i}] asset payload_span is outside outer span")));
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
            if ch.sentence_end > 0 && ch.sentence_end as usize > shared::bound(ctx, "opennlp:sentences")? {
                return Err(bad(format!("{lid}[{i}] chunk sentence_end exceeds its sentence target")));
            }
            check_parent("chunk", lid, i, ch.parent_index)?;
        }
        Value::LevelUnit(lv) => {
            check_parent("level", lid, i, lv.parent_index)?;
            match lv.kind.as_str() {
                "document" if i == 0 && lv.unit_index == 0 && lv.parent_index.is_none() => {}
                kind @ ("chunk" | "section") if i > 0 && lv.parent_index.is_some() => {
                    let target = if kind == "chunk" { "turbo:chunks" } else { "turbo:structure" };
                    shared::target(ctx, target, lv.unit_index)?;
                }
                _ => return Err(bad(format!("{lid}[{i}] invalid level kind, root or parent"))),
            }
        }
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
        Value::Event(ev) => shared::event(ctx, i, ev)?,
        Value::LexicalExpansion(le) => check_prob("weight", lid, i, le.weight)?,
        Value::RecordValue(rv) => check_prob("score", lid, i, rv.score)?,
        _ => {}
    }

    Ok(())
}
