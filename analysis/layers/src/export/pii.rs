//! The `turbo:pii` and `turbo:pii-candidates` codecs. Both write their
//! tables (custom kinds, detectors), one `u32` kind code an item
//! ([`PiiKind::code`]), one `u16` detector index an item, the flat
//! evidence facts with per-item end offsets, and the optional scores; the
//! mention layer adds the normalized values and the candidate layer its
//! dispositions. An unknown kind code, evidence tag, algorithm, score
//! kind or disposition tag is refused on import.

use super::*;
use crate::pii_candidates::{AllowRule, AllowScope, PiiCandidates, PiiDisposition, RejectReason};
use crate::{
    ChecksumAlgorithm, EvidenceList, PiiDetectorRef, PiiDetectors, PiiEvidence, PiiKind, PiiScore, PiiScoreKind,
};

fn put_tables(w: &mut Writer, kinds: &CustomKinds, detectors: &PiiDetectors) -> Result<(), ExportError> {
    w.usize(kinds.len())?;
    for name in kinds.names() {
        w.str(name)?;
    }
    w.usize(detectors.len())?;
    for (id, version) in detectors.iter() {
        w.str(id)?;
        w.str(version)?;
    }
    Ok(())
}

fn get_tables(r: &mut Reader<'_>) -> Result<(CustomKinds, PiiDetectors), ExportError> {
    let n_custom = r.usize()?;
    if n_custom > CustomKinds::MAX {
        return Err(malformed(KindError::Full.to_string()));
    }
    let mut names = Vec::with_capacity(n_custom);
    for _ in 0..n_custom {
        names.push(r.str()?);
    }
    let custom_kinds = CustomKinds::from_names(names).map_err(|e| malformed(e.to_string()))?;
    let n_detectors = r.usize()?;
    if n_detectors > PiiDetectors::MAX {
        return Err(malformed("more pii detectors than the table holds"));
    }
    let mut detectors = PiiDetectors::new();
    for i in 0..n_detectors {
        let id = r.str()?;
        let version = r.str()?;
        let at = detectors.add(id, version).map_err(|e| malformed(e.to_string()))?;
        if at.index() as usize != i {
            return Err(malformed("a pii detector table repeats an entry"));
        }
    }
    Ok((custom_kinds, detectors))
}

fn put_kinds(w: &mut Writer, kinds: &[PiiKind], detector: &[PiiDetectorRef]) -> Result<(), ExportError> {
    w.usize(kinds.len())?;
    for &k in kinds {
        w.u32(k.code());
    }
    w.usize(detector.len())?;
    for &d in detector {
        w.u16(d.index());
    }
    Ok(())
}

fn get_kinds(
    r: &mut Reader<'_>,
    custom: &CustomKinds,
    detectors: &PiiDetectors,
) -> Result<(Vec<PiiKind>, Vec<PiiDetectorRef>), ExportError> {
    let mut kind = Vec::new();
    for _ in 0..r.usize()? {
        kind.push(custom.from_code(r.u32()?).map_err(|e| malformed(e.to_string()))?);
    }
    let mut detector = Vec::new();
    for _ in 0..r.usize()? {
        let d = PiiDetectorRef::new(r.u16()?);
        detectors.check(d).map_err(|e| malformed(e.to_string()))?;
        detector.push(d);
    }
    Ok((kind, detector))
}

fn put_evidence(w: &mut Writer, evidence: &EvidenceList) -> Result<(), ExportError> {
    w.usize(evidence.facts().len())?;
    for &f in evidence.facts() {
        w.u8(f.tag());
        match f {
            PiiEvidence::Shape => {}
            PiiEvidence::Checksum { algorithm, passed } => {
                w.u8(algorithm.id());
                w.u8(u8::from(passed));
            }
            PiiEvidence::Label { span, phrase } => {
                w.span(span);
                w.u32(phrase);
            }
            PiiEvidence::DenyHit { list, term } => {
                w.u32(list);
                w.u32(term);
            }
            PiiEvidence::Rule { rule } => w.u32(rule),
        }
    }
    w.usize(evidence.len())?;
    for &e in evidence.ends() {
        w.u32(e);
    }
    Ok(())
}

fn get_flag(r: &mut Reader<'_>, what: &str) -> Result<bool, ExportError> {
    match r.u8()? {
        0 => Ok(false),
        1 => Ok(true),
        v => Err(malformed(format!("a {what} flag of {v}"))),
    }
}

fn get_evidence(r: &mut Reader<'_>) -> Result<EvidenceList, ExportError> {
    let mut facts = Vec::new();
    for _ in 0..r.usize()? {
        let tag = r.u8()?;
        facts.push(match tag {
            1 => PiiEvidence::Shape,
            2 => {
                let id = r.u8()?;
                let algorithm = ChecksumAlgorithm::from_id(u32::from(id))
                    .ok_or_else(|| malformed(format!("an unknown pii checksum algorithm {id}")))?;
                PiiEvidence::Checksum { algorithm, passed: get_flag(r, "checksum")? }
            }
            3 => PiiEvidence::Label { span: r.span()?, phrase: r.u32()? },
            4 => PiiEvidence::DenyHit { list: r.u32()?, term: r.u32()? },
            5 => PiiEvidence::Rule { rule: r.u32()? },
            tag => return Err(malformed(format!("an unknown pii evidence kind {tag}"))),
        });
    }
    let mut ends = Vec::new();
    for _ in 0..r.usize()? {
        ends.push(r.u32()?);
    }
    EvidenceList::from_parts(facts, ends).map_err(malformed)
}

fn put_scores(w: &mut Writer, scores: &[Option<PiiScore>]) -> Result<(), ExportError> {
    w.usize(scores.len())?;
    for s in scores {
        match s {
            None => w.u8(0),
            Some(s) => {
                w.u8(1);
                w.u32(s.value.to_bits());
                w.u8(s.kind.id());
                w.u16(s.model.index());
                w.u8(u8::from(s.calibrated));
            }
        }
    }
    Ok(())
}

fn get_scores(r: &mut Reader<'_>, detectors: &PiiDetectors) -> Result<Vec<Option<PiiScore>>, ExportError> {
    let mut scores = Vec::new();
    for _ in 0..r.usize()? {
        if !get_flag(r, "score presence")? {
            scores.push(None);
            continue;
        }
        let value = f32::from_bits(r.u32()?);
        let id = r.u8()?;
        let kind =
            PiiScoreKind::from_id(u32::from(id)).ok_or_else(|| malformed(format!("an unknown pii score kind {id}")))?;
        let model = PiiDetectorRef::new(r.u16()?);
        detectors.check(model).map_err(|e| malformed(e.to_string()))?;
        let score = PiiScore { value, kind, model, calibrated: get_flag(r, "calibrated")? };
        score.check().map_err(malformed)?;
        scores.push(Some(score));
    }
    Ok(scores)
}

impl LayerCodec for Pii {
    fn export(&self, cx: &ExportCx) -> Result<Vec<u8>, ExportError> {
        for &k in &self.kind {
            self.custom_kinds.check(k).map_err(|e| malformed(e.to_string()))?;
        }
        let mut w = Writer::new();
        put_spans(&mut w, &self.spans, cx)?;
        put_tables(&mut w, &self.custom_kinds, &self.detectors)?;
        put_kinds(&mut w, &self.kind, &self.detector)?;
        put_evidence(&mut w, &self.evidence)?;
        put_scores(&mut w, &self.score)?;
        put_strs(&mut w, &self.normalized)?;
        Ok(w.finish())
    }
    fn import(data: &[u8], cx: &mut ImportCx, _doc: &Document) -> Result<Self, ExportError> {
        let mut r = Reader::new(data);
        let spans = get_spans(&mut r, cx)?;
        let (custom_kinds, detectors) = get_tables(&mut r)?;
        let (kind, detector) = get_kinds(&mut r, &custom_kinds, &detectors)?;
        let evidence = get_evidence(&mut r)?;
        let score = get_scores(&mut r, &detectors)?;
        let normalized = get_strs(&mut r)?;
        want_len(kind.len(), spans.len(), "kinds")?;
        want_len(detector.len(), spans.len(), "detectors")?;
        want_len(evidence.len(), spans.len(), "evidence lists")?;
        want_len(score.len(), spans.len(), "scores")?;
        want_len(normalized.len(), spans.len(), "normalized values")?;
        done(&r, "pii")?;
        Ok(Pii { spans, custom_kinds, detectors, kind, detector, evidence, score, normalized })
    }
}

fn put_dispositions(w: &mut Writer, dispositions: &[PiiDisposition]) -> Result<(), ExportError> {
    w.usize(dispositions.len())?;
    for &d in dispositions {
        w.u8(d.tag());
        match d {
            PiiDisposition::Kept { mention }
            | PiiDisposition::NestedIn { mention }
            | PiiDisposition::Repeats { mention } => w.usize(mention.get())?,
            PiiDisposition::AllowedByRule { scope, rule } => {
                w.u8(match scope {
                    AllowScope::Global => 1,
                    AllowScope::Kind => 2,
                });
                match rule {
                    AllowRule::Term(i) => {
                        w.u8(1);
                        w.u32(i);
                    }
                    AllowRule::Pattern(i) => {
                        w.u8(2);
                        w.u32(i);
                    }
                }
            }
            PiiDisposition::RejectedByRule { reason } => w.u8(reason.id()),
        }
    }
    Ok(())
}

fn get_mention_ix(r: &mut Reader<'_>) -> Result<Ix<Pii>, ExportError> {
    let i = r.usize()?;
    u32::try_from(i).map(Ix::new).map_err(|_| malformed("a pii mention index past u32"))
}

fn get_dispositions(r: &mut Reader<'_>) -> Result<Vec<PiiDisposition>, ExportError> {
    let mut out = Vec::new();
    for _ in 0..r.usize()? {
        let tag = r.u8()?;
        out.push(match tag {
            1 => PiiDisposition::Kept { mention: get_mention_ix(r)? },
            2 => {
                let scope = match r.u8()? {
                    1 => AllowScope::Global,
                    2 => AllowScope::Kind,
                    s => return Err(malformed(format!("an unknown pii allow scope {s}"))),
                };
                let rule = match r.u8()? {
                    1 => AllowRule::Term(r.u32()?),
                    2 => AllowRule::Pattern(r.u32()?),
                    k => return Err(malformed(format!("an unknown pii allow rule kind {k}"))),
                };
                PiiDisposition::AllowedByRule { scope, rule }
            }
            3 => PiiDisposition::NestedIn { mention: get_mention_ix(r)? },
            4 => {
                let id = r.u8()?;
                let reason = RejectReason::from_id(u32::from(id))
                    .ok_or_else(|| malformed(format!("an unknown pii reject reason {id}")))?;
                PiiDisposition::RejectedByRule { reason }
            }
            5 => PiiDisposition::Repeats { mention: get_mention_ix(r)? },
            tag => return Err(malformed(format!("an unknown pii disposition {tag}"))),
        });
    }
    Ok(out)
}

impl LayerCodec for PiiCandidates {
    fn export(&self, cx: &ExportCx) -> Result<Vec<u8>, ExportError> {
        for &k in &self.kind {
            self.custom_kinds.check(k).map_err(|e| malformed(e.to_string()))?;
        }
        let mut w = Writer::new();
        put_spans(&mut w, &self.spans, cx)?;
        put_tables(&mut w, &self.custom_kinds, &self.detectors)?;
        put_kinds(&mut w, &self.kind, &self.detector)?;
        put_evidence(&mut w, &self.evidence)?;
        put_scores(&mut w, &self.score)?;
        put_dispositions(&mut w, &self.disposition)?;
        Ok(w.finish())
    }
    fn import(data: &[u8], cx: &mut ImportCx, _doc: &Document) -> Result<Self, ExportError> {
        let mut r = Reader::new(data);
        let spans = get_spans(&mut r, cx)?;
        let (custom_kinds, detectors) = get_tables(&mut r)?;
        let (kind, detector) = get_kinds(&mut r, &custom_kinds, &detectors)?;
        let evidence = get_evidence(&mut r)?;
        let score = get_scores(&mut r, &detectors)?;
        let disposition = get_dispositions(&mut r)?;
        want_len(kind.len(), spans.len(), "kinds")?;
        want_len(detector.len(), spans.len(), "detectors")?;
        want_len(evidence.len(), spans.len(), "evidence lists")?;
        want_len(score.len(), spans.len(), "scores")?;
        want_len(disposition.len(), spans.len(), "dispositions")?;
        done(&r, "pii candidates")?;
        let layer = PiiCandidates { spans, custom_kinds, detectors, kind, detector, evidence, score, disposition };
        // The mention layer this record's provenance binds the references
        // to is restored first; a reference must land inside it. The
        // document repeats the check when it takes the layer.
        if layer.disposition.iter().any(|d| d.mention().is_some()) {
            let pii = LayerRef { layer: Pii::ID.to_owned(), instance: cx.resolved_instance(Pii::ID).to_owned() };
            cx.expect_target::<Pii>(&pii)?;
            let len = cx.layer_len(&pii)?;
            for (i, d) in layer.disposition.iter().enumerate() {
                if let Some(m) = d.mention()
                    && m.get() >= len
                {
                    return Err(malformed(format!("pii candidate {i} refers to mention {} of {len}", m.get())));
                }
            }
        }
        Ok(layer)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn evidence_of(bytes: &[u8]) -> Result<EvidenceList, ExportError> {
        let mut r = Reader::new(bytes);
        get_evidence(&mut r)
    }

    #[test]
    fn unknown_evidence_kinds_algorithms_and_dispositions_are_refused() {
        let mut w = Writer::new();
        w.usize(1).unwrap();
        w.u8(9);
        assert!(
            matches!(evidence_of(&w.finish()), Err(ExportError::Malformed(d)) if d.contains("unknown pii evidence kind 9"))
        );
        let mut w = Writer::new();
        w.usize(1).unwrap();
        w.u8(2);
        w.u8(14);
        w.u8(1);
        assert!(matches!(evidence_of(&w.finish()), Err(ExportError::Malformed(d)) if d.contains("algorithm 14")));
        let mut w = Writer::new();
        w.usize(1).unwrap();
        w.u8(1);
        w.usize(1).unwrap();
        w.u32(0);
        assert!(matches!(evidence_of(&w.finish()), Err(ExportError::Malformed(d)) if d.contains("no evidence")));
        let mut w = Writer::new();
        w.usize(1).unwrap();
        w.u8(1);
        w.usize(1).unwrap();
        w.u32(1);
        let list = evidence_of(&w.finish()).unwrap();
        assert_eq!(list.get(0), &[PiiEvidence::Shape]);
        let mut w = Writer::new();
        w.usize(1).unwrap();
        w.u8(6);
        let bytes = w.finish();
        let mut r = Reader::new(&bytes);
        assert!(matches!(get_dispositions(&mut r), Err(ExportError::Malformed(d)) if d.contains("disposition 6")));
        let mut w = Writer::new();
        w.usize(1).unwrap();
        w.u8(1);
        w.u32(1.5f32.to_bits());
        w.u8(1);
        w.u16(0);
        w.u8(0);
        let mut detectors = PiiDetectors::new();
        detectors.add("d", "1").unwrap();
        let bytes = w.finish();
        let mut r = Reader::new(&bytes);
        assert!(
            matches!(get_scores(&mut r, &detectors), Err(ExportError::Malformed(d)) if d.contains("outside 0..=1"))
        );
    }
}
