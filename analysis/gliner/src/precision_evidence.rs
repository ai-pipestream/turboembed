use turbo_analysis::{Document, Error, Evidence, EvidenceStep, Key};

use crate::{PrecisionCandidates, PrecisionDecision, PrecisionRejection};

/// Builds evidence with one explicit abstention reason per rejected candidate.
pub fn precision_evidence(
    doc: &Document,
    key: Key<PrecisionCandidates>,
) -> Result<Evidence<PrecisionCandidates>, Error> {
    let candidates = doc.require(&key)?;
    let mut evidence = EvidenceStep::new(key).build(doc)?;
    evidence.abstention = candidates
        .decision
        .iter()
        .enumerate()
        .filter_map(|(i, decision)| match decision {
            PrecisionDecision::Rejected(reason) => Some((i as u32, rejection(reason))),
            _ => None,
        })
        .collect();
    Ok(evidence)
}

fn rejection(reason: &PrecisionRejection) -> String {
    match reason {
        PrecisionRejection::NonEntity => "non-entity".into(),
        PrecisionRejection::NoCompatibleLink => "no-compatible-link".into(),
        PrecisionRejection::PiiConflict(kind) => format!("pii-conflict:{}", kind.code()),
    }
}
