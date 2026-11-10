use std::sync::Arc;

use turbo_analysis::evidence::read_of;
use turbo_analysis::{Document, Error, Evidence, EvidenceStep, Key};
use turbo_analysis_layers::Entities;

use crate::{PrecisionCandidates, PrecisionDecision, PrecisionRejection};

/// Builds evidence with one explicit abstention reason per rejected candidate.
pub fn precision_evidence(
    doc: &Document,
    key: Key<PrecisionCandidates>,
) -> Result<Evidence<PrecisionCandidates>, Error> {
    let candidates = doc.require(&key)?;
    let mut evidence = EvidenceStep::new(key).build(doc)?;
    // The stage has no model of its own: its records are the candidates of
    // the model that wrote the entities it read, so they name that model.
    let read = doc.provenance(key.id()).and_then(read_of::<Entities>).unwrap_or(Key::<Entities>::DEFAULT.id());
    if let Some(model) = doc.provenance(read).and_then(|p| p.model.clone()) {
        let mut producers = (*evidence.producers).clone();
        for producer in producers.iter_mut().filter(|p| p.model.is_none()) {
            producer.model = Some(model.clone());
        }
        evidence.producers = Arc::new(producers);
    }
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
