//! PII mention and candidate formatting for JSON-LD and Markdown document
//! exports (#1904, #1942).
//!
//! Exporters write findings only: a mention's or candidate's surface value and
//! normalized value never leave through an export, and label evidence records
//! the label phrase's projected source span (and phrase index in JSON-LD)
//! without repeating the label text. Candidate rows (`turbo:pii-candidates`)
//! are emitted only when `detail` is requested.

use std::fmt::Write as _;

use turbo_analysis::{Document, Key, ViewId};
use turbo_analysis_layers::{
    AllowRule, AllowScope, Pii, PiiCandidates, PiiConfidence, PiiDetectorRef, PiiDetectors, PiiDisposition,
    PiiEvidence, PiiScore,
};

use super::jsonld_export::{format_json_f64, json_string};
use super::markdown_export::escape_markdown;
use super::{ExportOffsetMode, FormatExportError, get_bound_layer, sanitize_single_line};

/// Canonical JSON-LD context URI referenced by exported documents.
pub const JSONLD_CONTEXT_URI: &str = "https://turboembed.ai/ns/jsonld/v1";

/// Canonical JSON-LD context document mapping exported property terms to the
/// `turbo:` namespace (`https://turboembed.ai/ns/jsonld/v1#`).
pub const JSONLD_CONTEXT: &str = r#"{
  "@context": {
    "turbo": "https://turboembed.ai/ns/jsonld/v1#",
    "docId": "turbo:docId",
    "offsetMode": "turbo:offsetMode",
    "sourceText": "turbo:sourceText",
    "provenance": "turbo:provenance",
    "layer": "turbo:layer",
    "instance": "turbo:instance",
    "by": "turbo:by",
    "version": "turbo:version",
    "layers": "turbo:layers",
    "code": "turbo:code",
    "best": "turbo:best",
    "index": "turbo:index",
    "sentence": "turbo:sentence",
    "start": "turbo:start",
    "end": "turbo:end",
    "text": "turbo:text",
    "pos": "turbo:pos",
    "lemma": "turbo:lemma",
    "stem": "turbo:stem",
    "feats": "turbo:feats",
    "head": "turbo:head",
    "deprel": "turbo:deprel",
    "stopword": "turbo:stopword",
    "tokenStart": "turbo:tokenStart",
    "tokenEnd": "turbo:tokenEnd",
    "form": "turbo:form",
    "fragments": "turbo:fragments",
    "from": "turbo:from",
    "to": "turbo:to",
    "kind": "turbo:kind",
    "confidence": "turbo:confidence",
    "detector": "turbo:detector",
    "id": "turbo:id",
    "evidence": "turbo:evidence",
    "shape": "turbo:shape",
    "checksum": "turbo:checksum",
    "algorithm": "turbo:algorithm",
    "passed": "turbo:passed",
    "label": "turbo:label",
    "span": "turbo:span",
    "phrase": "turbo:phrase",
    "denyHit": "turbo:denyHit",
    "list": "turbo:list",
    "term": "turbo:term",
    "rule": "turbo:rule",
    "score": "turbo:score",
    "value": "turbo:value",
    "modelRef": "turbo:modelRef",
    "calibrated": "turbo:calibrated",
    "disposition": "turbo:disposition",
    "kept": "turbo:kept",
    "allowedByRule": "turbo:allowedByRule",
    "scope": "turbo:scope",
    "pattern": "turbo:pattern",
    "nestedIn": "turbo:nestedIn",
    "rejectedByRule": "turbo:rejectedByRule",
    "repeats": "turbo:repeats"
  }
}
"#;

pub(crate) fn append_jsonld_pii(
    doc: &Document,
    offsets: ExportOffsetMode,
    detail: bool,
    sections: &mut Vec<String>,
) -> Result<(), FormatExportError> {
    if let Some(pii) = doc.get::<Pii>() {
        let view = pii.spans.view();
        let mut s = String::from("    \"turbo:pii\": [");
        for (i, m) in pii.mentions().enumerate() {
            if i > 0 {
                s.push_str(", ");
            }
            let src = doc.to_source(view, m.span);
            let (start, end) = offsets.convert_source_span(doc, src)?;
            let det = format_json_detector(&pii.detectors, m.detector);
            let ev = format_json_evidence(doc, view, offsets, m.evidence)?;
            let score = format_json_score(&pii.detectors, m.score);
            let _ = write!(
                s,
                "{{\"@type\": \"turbo:PiiMention\", \"kind\": {}, \"confidence\": {}, \"start\": {start}, \"end\": {end}, \"detector\": {det}, \"evidence\": {ev}{score}}}",
                json_string(pii.kind_name(m.kind).unwrap_or("")),
                json_string(m.confidence().name()),
            );
        }
        s.push(']');
        sections.push(s);
    }

    if detail && let Some(candidates) = get_bound_layer::<PiiCandidates>(doc, Key::<Pii>::DEFAULT.id(), None)? {
        let view = candidates.spans.view();
        let mut s = String::from("    \"turbo:pii-candidates\": [");
        for (i, c) in candidates.candidates().enumerate() {
            if i > 0 {
                s.push_str(", ");
            }
            let src = doc.to_source(view, c.span);
            let (start, end) = offsets.convert_source_span(doc, src)?;
            let det = format_json_detector(&candidates.detectors, c.detector);
            let ev = format_json_evidence(doc, view, offsets, c.evidence)?;
            let score = format_json_score(&candidates.detectors, c.score);
            let disp = format_json_disposition(c.disposition);
            let _ = write!(
                s,
                "{{\"@type\": \"turbo:PiiCandidate\", \"kind\": {}, \"confidence\": {}, \"start\": {start}, \"end\": {end}, \"detector\": {det}, \"evidence\": {ev}{score}, \"disposition\": {disp}}}",
                json_string(candidates.kind_name(c.kind).unwrap_or("")),
                json_string(PiiConfidence::derive(c.evidence).name()),
            );
        }
        s.push(']');
        sections.push(s);
    }
    Ok(())
}

fn format_json_detector(detectors: &PiiDetectors, r: PiiDetectorRef) -> String {
    let (id, version) = detectors.get(r).unwrap_or(("", ""));
    format!("{{\"id\": {}, \"version\": {}}}", json_string(id), json_string(version))
}

fn format_json_evidence(
    doc: &Document,
    view: ViewId,
    offsets: ExportOffsetMode,
    evidence: &[PiiEvidence],
) -> Result<String, FormatExportError> {
    let mut out = String::from("[");
    for (i, fact) in evidence.iter().enumerate() {
        if i > 0 {
            out.push_str(", ");
        }
        match *fact {
            PiiEvidence::Shape => out.push_str("{\"shape\": {}}"),
            PiiEvidence::Checksum { algorithm, passed } => {
                let _ = write!(
                    out,
                    "{{\"checksum\": {{\"algorithm\": {}, \"passed\": {passed}}}}}",
                    json_string(algorithm.name()),
                );
            }
            PiiEvidence::Label { span, phrase } => {
                let src = doc.to_source(view, span);
                let (start, end) = offsets.convert_source_span(doc, src)?;
                let _ = write!(
                    out,
                    "{{\"label\": {{\"span\": {{\"start\": {start}, \"end\": {end}}}, \"phrase\": {phrase}}}}}"
                );
            }
            PiiEvidence::DenyHit { list, term } => {
                let _ = write!(out, "{{\"denyHit\": {{\"list\": {list}, \"term\": {term}}}}}");
            }
            PiiEvidence::Rule { rule } => {
                let _ = write!(out, "{{\"rule\": {{\"rule\": {rule}}}}}");
            }
        }
    }
    out.push(']');
    Ok(out)
}

fn format_json_score(detectors: &PiiDetectors, score: Option<PiiScore>) -> String {
    let Some(s) = score else { return String::new() };
    format!(
        ", \"score\": {{\"value\": {}, \"kind\": {}, \"modelRef\": {}, \"calibrated\": {}}}",
        format_json_f64(f64::from(s.value)),
        json_string(s.kind.name()),
        format_json_detector(detectors, s.model),
        s.calibrated,
    )
}

fn format_json_disposition(d: PiiDisposition) -> String {
    match d {
        PiiDisposition::Kept { mention } => format!("{{\"kept\": {}}}", mention.get()),
        PiiDisposition::AllowedByRule { scope, rule } => {
            let scope_str = match scope {
                AllowScope::Global => "global",
                AllowScope::Kind => "kind",
            };
            let rule_json = match rule {
                AllowRule::Term(i) => format!("{{\"term\": {i}}}"),
                AllowRule::Pattern(i) => format!("{{\"pattern\": {i}}}"),
            };
            format!("{{\"allowedByRule\": {{\"scope\": {}, \"rule\": {rule_json}}}}}", json_string(scope_str))
        }
        PiiDisposition::NestedIn { mention } => format!("{{\"nestedIn\": {}}}", mention.get()),
        PiiDisposition::RejectedByRule { reason } => {
            format!("{{\"rejectedByRule\": {}}}", json_string(reason.name()))
        }
        PiiDisposition::Repeats { mention } => format!("{{\"repeats\": {}}}", mention.get()),
    }
}

pub(crate) fn append_markdown_pii(
    doc: &Document,
    offsets: ExportOffsetMode,
    detail: bool,
    buf: &mut String,
) -> Result<(), FormatExportError> {
    if let Some(pii) = doc.get::<Pii>()
        && !pii.spans.is_empty()
    {
        buf.push_str("## PII Summary\n\n");
        let p_view = pii.spans.view();
        for m in pii.mentions() {
            let src_span = doc.to_source(p_view, m.span);
            let (s, e) = offsets.convert_source_span(doc, src_span)?;
            let det = format_md_detector(&pii.detectors, m.detector);
            let facts = format_md_evidence(doc, p_view, offsets, m.evidence)?;
            let score = format_md_score(&pii.detectors, m.score);
            let _ = writeln!(
                buf,
                "- **{}** [{s}..{e}] (confidence: `{}`, detector: `{det}`, evidence: {facts}{score})",
                escape_markdown(pii.kind_name(m.kind).unwrap_or("")),
                m.confidence().name(),
            );
        }
        buf.push('\n');
    }

    if detail
        && let Some(candidates) = get_bound_layer::<PiiCandidates>(doc, Key::<Pii>::DEFAULT.id(), None)?
        && !candidates.spans.is_empty()
    {
        buf.push_str("## PII Candidates\n\n");
        let c_view = candidates.spans.view();
        for c in candidates.candidates() {
            let src_span = doc.to_source(c_view, c.span);
            let (s, e) = offsets.convert_source_span(doc, src_span)?;
            let det = format_md_detector(&candidates.detectors, c.detector);
            let facts = format_md_evidence(doc, c_view, offsets, c.evidence)?;
            let score = format_md_score(&candidates.detectors, c.score);
            let disp = format_md_disposition(c.disposition);
            let _ = writeln!(
                buf,
                "- **{}** [{s}..{e}] (disposition: `{disp}`, detector: `{det}`, evidence: {facts}{score})",
                escape_markdown(candidates.kind_name(c.kind).unwrap_or("")),
            );
        }
        buf.push('\n');
    }
    Ok(())
}

fn format_md_detector(detectors: &PiiDetectors, r: PiiDetectorRef) -> String {
    let (id, version) = detectors.get(r).unwrap_or(("", ""));
    let id_esc = escape_markdown(&sanitize_single_line(id));
    if version.is_empty() { id_esc } else { format!("{id_esc}@{}", escape_markdown(&sanitize_single_line(version))) }
}

fn format_md_evidence(
    doc: &Document,
    view: ViewId,
    offsets: ExportOffsetMode,
    evidence: &[PiiEvidence],
) -> Result<String, FormatExportError> {
    let mut parts = Vec::with_capacity(evidence.len());
    for fact in evidence {
        let item = match *fact {
            PiiEvidence::Shape => "shape".to_owned(),
            PiiEvidence::Checksum { algorithm, passed } => {
                format!("{} {}", algorithm.name(), if passed { "ok" } else { "fail" })
            }
            PiiEvidence::Label { span, .. } => {
                let src = doc.to_source(view, span);
                let (ls, le) = offsets.convert_source_span(doc, src)?;
                format!("label {ls}..{le}")
            }
            PiiEvidence::DenyHit { list, term } => format!("deny-hit {list}:{term}"),
            PiiEvidence::Rule { rule } => format!("rule {rule}"),
        };
        parts.push(item);
    }
    Ok(parts.join(", "))
}

fn format_md_score(detectors: &PiiDetectors, score: Option<PiiScore>) -> String {
    let Some(s) = score else { return String::new() };
    format!(
        ", score: {:.4} ({}, model: `{}`, calibrated: `{}`)",
        s.value,
        s.kind.name(),
        format_md_detector(detectors, s.model),
        s.calibrated,
    )
}

fn format_md_disposition(d: PiiDisposition) -> String {
    match d {
        PiiDisposition::Kept { mention } => format!("kept {}", mention.get()),
        PiiDisposition::AllowedByRule { scope, rule } => {
            let scope_str = match scope {
                AllowScope::Global => "global",
                AllowScope::Kind => "kind",
            };
            let rule_str = match rule {
                AllowRule::Term(i) => format!("term {i}"),
                AllowRule::Pattern(i) => format!("pattern {i}"),
            };
            format!("allowed-by-rule ({scope_str} {rule_str})")
        }
        PiiDisposition::NestedIn { mention } => format!("nested-in {}", mention.get()),
        PiiDisposition::RejectedByRule { reason } => format!("rejected-by-rule ({})", reason.name()),
        PiiDisposition::Repeats { mention } => format!("repeats {}", mention.get()),
    }
}
