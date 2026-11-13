//! Structural and offset-invariant validation for `SearchDocumentProjection`
//! (`RoundTripSearchProjection` and `TurboSinkService`, `#2345`).

use super::{Failure, bad};
use crate::offsets::OffsetMode;
use crate::proto::result::{SearchAssessmentState, SearchDocumentProjection, SearchLinkedEntity, SearchSketchRow};
use crate::results::typed::identity::check_binding;

fn check_wire_span(what: &str, start: u32, end: u32, text_len: u32) -> Result<(), Failure> {
    if start > end {
        return Err(bad(format!("{what} [{start}, {end}) ends before it starts")));
    }
    if end > text_len {
        return Err(bad(format!("{what} [{start}, {end}) exceeds target_text_len {text_len}")));
    }
    Ok(())
}

fn check_finite(what: &str, v: f64) -> Result<(), Failure> {
    if !v.is_finite() {
        return Err(bad(format!("{what} {v} is not finite")));
    }
    Ok(())
}

fn check_unit_interval(what: &str, v: f64) -> Result<(), Failure> {
    check_finite(what, v)?;
    if !(0.0..=1.0).contains(&v) {
        return Err(bad(format!("{what} {v} is outside [0.0, 1.0]")));
    }
    Ok(())
}

fn check_sha256_len(what: &str, bytes: &[u8], allow_empty: bool) -> Result<(), Failure> {
    if allow_empty && bytes.is_empty() {
        return Ok(());
    }
    if bytes.len() != 32 {
        return Err(bad(format!("{what} must be 32 SHA-256 bytes, got {}", bytes.len())));
    }
    Ok(())
}

fn check_sketch(what: &str, sk: &SearchSketchRow) -> Result<(), Failure> {
    if sk.shingle == 0 {
        return Err(bad(format!("{what}.shingle must be > 0")));
    }
    if sk.minhash.is_empty() {
        return Err(bad(format!("{what}.minhash must not be empty")));
    }
    Ok(())
}

fn check_entity_list(
    bucket_name: &str,
    expected_polarity: &str,
    items: &[SearchLinkedEntity],
    text_len: u32,
) -> Result<(), Failure> {
    for (i, e) in items.iter().enumerate() {
        check_wire_span(&format!("{bucket_name}[{i}].span"), e.span_start, e.span_end, text_len)?;
        check_finite(&format!("{bucket_name}[{i}].score"), e.score)?;
        if e.polarity != expected_polarity {
            return Err(bad(format!(
                "{bucket_name}[{i}].polarity is {:?}, expected {expected_polarity:?}",
                e.polarity
            )));
        }
        if e.entity_kind.is_empty() {
            return Err(bad(format!("{bucket_name}[{i}].entity_kind must not be empty")));
        }
    }
    Ok(())
}

/// Validates every structural, provenance, span, and scalar invariant on `msg`.
pub fn validate_search_projection(msg: &SearchDocumentProjection) -> Result<(), Failure> {
    let _ = OffsetMode::from_proto(msg.offset_mode)?;
    u16::try_from(msg.target_view).map_err(|_| bad(format!("target_view {} exceeds u16::MAX", msg.target_view)))?;
    let text_len = msg.target_text_len;

    let prov = msg.provenance.as_ref().ok_or_else(|| bad("provenance is required on SearchDocumentProjection"))?;
    let input = prov.input.as_ref().ok_or_else(|| bad("provenance.input is required"))?;
    check_binding(input.binding.as_ref(), "provenance.input.binding")?;
    check_sha256_len("provenance.input.raw_bytes_digest", &input.raw_bytes_digest, false)?;
    check_sha256_len("provenance.input.source_text_digest", &input.source_text_digest, false)?;
    let res = prov.result.as_ref().ok_or_else(|| bad("provenance.result is required"))?;
    check_sha256_len("provenance.result.content_digest", &res.content_digest, false)?;
    check_sha256_len("provenance.analyzer_config_digest", &prov.analyzer_config_digest, true)?;
    for (i, v) in prov.views.iter().enumerate() {
        u16::try_from(v.view_id).map_err(|_| bad(format!("provenance.views[{i}].view_id exceeds u16::MAX")))?;
        check_sha256_len(&format!("provenance.views[{i}].text_sha256"), &v.text_sha256, false)?;
    }

    // Sentences invariant (`sentences:body`): sorted, non-overlapping, in bounds.
    let mut prev_sent_end = 0u32;
    for (i, s) in msg.sentences.iter().enumerate() {
        if s.index != i as u32 {
            return Err(bad(format!("sentences[{i}].index is {}, expected {i}", s.index)));
        }
        check_wire_span(&format!("sentences[{i}].span"), s.span_start, s.span_end, text_len)?;
        if i > 0 && s.span_start < prev_sent_end {
            return Err(bad(format!(
                "sentences[{i}] [{}, {}) overlaps previous sentence ending at {prev_sent_end}",
                s.span_start, s.span_end
            )));
        }
        if let Some(sc) = s.boilerplate_score
            && (!sc.is_finite() || !(0.0..=1.0).contains(&sc))
        {
            return Err(bad(format!("sentences[{i}].boilerplate_score {sc} is outside [0.0, 1.0]")));
        }
        prev_sent_end = s.span_end;
    }

    for (i, c) in msg.chunks.iter().enumerate() {
        if c.index != i as u32 {
            return Err(bad(format!("chunks[{i}].index is {}, expected {i}", c.index)));
        }
        check_wire_span(&format!("chunks[{i}].span"), c.span_start, c.span_end, text_len)?;
        if c.sentence_start > c.sentence_end {
            return Err(bad(format!(
                "chunks[{i}] sentence range [{}..{}) ends before it starts",
                c.sentence_start, c.sentence_end
            )));
        }
        if !msg.sentences.is_empty() && c.sentence_end as usize > msg.sentences.len() {
            return Err(bad(format!(
                "chunks[{i}].sentence_end {} exceeds {} sentences",
                c.sentence_end,
                msg.sentences.len()
            )));
        }
        if let Some(p) = c.parent_index
            && p as usize >= msg.chunks.len()
        {
            return Err(bad(format!("chunks[{i}].parent_index {p} exceeds {} chunks", msg.chunks.len())));
        }
        if let Some(sk) = &c.sketch {
            check_sketch(&format!("chunks[{i}].sketch"), sk)?;
        }
    }

    for (i, t) in msg.tokens.iter().enumerate() {
        if t.term.is_empty() {
            return Err(bad(format!("tokens[{i}].term must not be empty")));
        }
        if t.pos_len == 0 {
            return Err(bad(format!("tokens[{i}].pos_len must be >= 1")));
        }
        check_wire_span(&format!("tokens[{i}].span"), t.span_start, t.span_end, text_len)?;
        if let Some(si) = t.sentence_index
            && si as usize >= msg.sentences.len()
        {
            return Err(bad(format!("tokens[{i}].sentence_index {si} exceeds {} sentences", msg.sentences.len())));
        }
    }

    if let Some(nt) = &msg.numeric_temporal {
        for (i, m) in nt.money.iter().enumerate() {
            check_wire_span(&format!("money[{i}].span"), m.span_start, m.span_end, text_len)?;
            check_finite(&format!("money[{i}].amount_f64"), m.amount_f64)?;
            if m.currency.is_empty() {
                return Err(bad(format!("money[{i}].currency must not be empty")));
            }
        }
        for (i, q) in nt.quantities.iter().enumerate() {
            check_wire_span(&format!("quantities[{i}].span"), q.span_start, q.span_end, text_len)?;
            check_finite(&format!("quantities[{i}].value_f64"), q.value_f64)?;
        }
        if let Some(dd) = &nt.document_date
            && (dd.month == 0 || dd.month > 12 || dd.day == 0 || dd.day > 31)
        {
            return Err(bad(format!("document_date {}-{}-{} has invalid month/day", dd.year, dd.month, dd.day)));
        }
        for (i, tr) in nt.temporals.iter().enumerate() {
            check_wire_span(&format!("temporals[{i}].span"), tr.span_start, tr.span_end, text_len)?;
            if tr.start_epoch_seconds > tr.end_epoch_seconds {
                return Err(bad(format!(
                    "temporals[{i}] epoch range [{}..{}) is inverted",
                    tr.start_epoch_seconds, tr.end_epoch_seconds
                )));
            }
        }
        for (i, tm) in nt.times.iter().enumerate() {
            check_wire_span(&format!("times[{i}].span"), tm.span_start, tm.span_end, text_len)?;
            if let (Some(s), Some(e)) = (tm.start_epoch_ms, tm.end_epoch_ms)
                && s > e
            {
                return Err(bad(format!("times[{i}] epoch_ms range [{s}..{e}) is inverted")));
            }
        }
    }

    for (i, g) in msg.geo_locations.iter().enumerate() {
        check_wire_span(&format!("geo_locations[{i}].span"), g.span_start, g.span_end, text_len)?;
        if !g.latitude.is_finite() || !(-90.0..=90.0).contains(&g.latitude) {
            return Err(bad(format!("geo_locations[{i}].latitude {} outside [-90, 90]", g.latitude)));
        }
        if !g.longitude.is_finite() || !(-180.0..=180.0).contains(&g.longitude) {
            return Err(bad(format!("geo_locations[{i}].longitude {} outside [-180, 180]", g.longitude)));
        }
        check_unit_interval(&format!("geo_locations[{i}].confidence"), g.confidence)?;
    }

    if let Some(cs) = &msg.categories_sentiment {
        if let Some(sc) = cs.sentiment_score {
            check_unit_interval("categories_sentiment.sentiment_score", sc)?;
        }
        for (i, c) in cs.categories.iter().enumerate() {
            if let Some(p) = c.probability {
                check_unit_interval(&format!("categories[{i}].probability"), p)?;
            }
        }
        for (i, a) in cs.aspects.iter().enumerate() {
            check_wire_span(&format!("aspects[{i}].span"), a.span_start, a.span_end, text_len)?;
            check_unit_interval(&format!("aspects[{i}].positive"), a.positive)?;
            check_unit_interval(&format!("aspects[{i}].negative"), a.negative)?;
            check_unit_interval(&format!("aspects[{i}].neutral"), a.neutral)?;
        }
    }

    if let Some(q) = &msg.quality {
        for (what, raw) in [
            ("quality.noise_state", q.noise_state),
            ("quality.artifacts_state", q.artifacts_state),
            ("quality.text_quality_state", q.text_quality_state),
        ] {
            SearchAssessmentState::try_from(raw)
                .map_err(|_| bad(format!("{what} {raw} is not a SearchAssessmentState")))?;
        }
        check_unit_interval("quality.worst_noise_score", q.worst_noise_score)?;
        if q.union_noise_chars < 0 {
            return Err(bad(format!("quality.union_noise_chars {} must be >= 0", q.union_noise_chars)));
        }
        if q.artifact_count < 0 {
            return Err(bad(format!("quality.artifact_count {} must be >= 0", q.artifact_count)));
        }
    }

    if let Some(ents) = &msg.entities {
        check_entity_list("entities.affirmed", "affirmed", &ents.affirmed, text_len)?;
        check_entity_list("entities.negated", "negated", &ents.negated, text_len)?;
        check_entity_list("entities.uncertain", "uncertain", &ents.uncertain, text_len)?;
    }

    if let Some(sk) = &msg.document_sketch {
        check_sketch("document_sketch", sk)?;
    }

    Ok(())
}
