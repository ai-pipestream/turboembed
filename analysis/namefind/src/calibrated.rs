//! Opt-in CRF span-marginal calibration over caller-provided numerical
//! scores. CRF span prediction sets (`Calibrated<Entities>`) are
//! sequence-marginal entity/non-entity sets aligned 1-to-1 with decoded
//! `Entities`. The calibration identity binds the typed schema, the
//! caller's numerical-score identity and the explicit model provenance;
//! this crate reads no model archive.
use std::collections::BTreeSet;
use std::sync::Arc;
use turbo_analysis::{Annotator, Error, Fnv1a, Layer, ModelRef, Refs, Spans, Tag, View};
use turbo_analysis_layers::{Calibrated, Entities, Sentences, Tokens};
use turbo_calibration::{Calibration, Identity, Policy, Predictor, Task};
use turbo_sequence::crf::{EntityDecoder, EntityLattice};

use crate::crf::{CrfScoreFn, CrfSentenceScores, SentenceTokens, push_entity, sentence_token_ranges};

pub const DEFAULT_CRF_CANDIDATE_PATHS: usize = 1;

/// Producer identity for span-level CRF calibration (`Task::SpanTagging`)
/// over the decoder's span distributions (`["other", kind_0, ...]`), bound
/// to the caller's explicit provenance and numerical-score identity.
/// `scores_hash` must identify the numerical parameters in the same way
/// everywhere they appear (see [`crate::crf::CrfNameAnnotator::new`]).
pub fn crf_identity(
    provenance: &ModelRef,
    domain: &str,
    decoder: &EntityDecoder,
    scores_hash: u64,
) -> Result<Identity, String> {
    crf_identity_with_candidates(provenance, domain, decoder, scores_hash, DEFAULT_CRF_CANDIDATE_PATHS)
}

/// As [`crf_identity`], with the candidate path count part of the
/// inference configuration (`1` selects top-1 decoded entities).
pub fn crf_identity_with_candidates(
    provenance: &ModelRef,
    domain: &str,
    decoder: &EntityDecoder,
    scores_hash: u64,
    candidate_paths: usize,
) -> Result<Identity, String> {
    if candidate_paths == 0 || candidate_paths > 4096 {
        return Err("CRF candidate path count must be in 1..=4096".into());
    }
    let labels: Vec<String> = std::iter::once("other".to_owned())
        .chain((0..decoder.kinds().len()).map(|i| decoder.kinds().label(Tag(i as u16)).to_owned()))
        .collect();
    let inference_hash = Fnv1a::new().u64(decoder.schema_hash()).u64(scores_hash).u64(candidate_paths as u64).finish();
    let i = Identity {
        model: provenance.id.clone(),
        revision: provenance.revision.clone(),
        sha256: provenance.sha256.clone(),
        task: Task::SpanTagging,
        domain: domain.into(),
        labels,
        inference_hash,
    };
    i.check()?;
    Ok(i)
}

/// Span-calibrated CRF name annotator writing raw `Entities` and an aligned
/// `Calibrated<Entities>` sidecar computed from CRF forward-backward span
/// marginals over the caller's numerical scores.
pub struct CalibratedCrfNameAnnotator {
    decoder: EntityDecoder,
    scores: Arc<CrfScoreFn>,
    predictor: Predictor,
    candidate_paths: usize,
}

impl CalibratedCrfNameAnnotator {
    pub fn new(
        decoder: EntityDecoder,
        calibration: Calibration,
        domain: &str,
        policy: Policy,
        scores: impl Fn(&[&str]) -> Result<CrfSentenceScores, String> + Send + Sync + 'static,
        scores_hash: u64,
    ) -> Result<Self, String> {
        Self::with_candidates(decoder, calibration, domain, policy, scores, scores_hash, DEFAULT_CRF_CANDIDATE_PATHS)
    }

    pub fn with_candidates(
        decoder: EntityDecoder,
        calibration: Calibration,
        domain: &str,
        policy: Policy,
        scores: impl Fn(&[&str]) -> Result<CrfSentenceScores, String> + Send + Sync + 'static,
        scores_hash: u64,
        candidate_paths: usize,
    ) -> Result<Self, String> {
        let provenance = ModelRef {
            id: calibration.identity().model.clone(),
            revision: calibration.identity().revision.clone(),
            sha256: calibration.identity().sha256.clone(),
        };
        let expected = crf_identity_with_candidates(&provenance, domain, &decoder, scores_hash, candidate_paths)?;
        if calibration.identity() != &expected {
            return Err("calibration identity does not match decoder, domain, scores or candidate paths".into());
        }
        let predictor = Predictor::new(calibration, &expected, policy)?;
        Ok(Self { decoder, scores: Arc::new(scores), predictor, candidate_paths })
    }

    pub fn candidate_paths(&self) -> usize {
        self.candidate_paths
    }

    pub fn predictor(&self) -> &Predictor {
        &self.predictor
    }

    pub fn decoder(&self) -> &EntityDecoder {
        &self.decoder
    }

    /// Decodes one sentence: pushes its candidate entities onto `entities`
    /// and returns one span-distribution row for each. With the default
    /// single candidate path the entities are the top path's; with more,
    /// the unique spans of the top-`k` paths, in first-appearance order.
    fn decoded_rows(
        &self,
        lattice: &EntityLattice<'_>,
        range: &SentenceTokens,
        entities: &mut Entities,
        starts: &[u32],
        ends: &[u32],
    ) -> Result<Vec<Vec<f64>>, String> {
        let mut rows = Vec::new();
        if self.candidate_paths == 1 {
            let mut paths = lattice.best_paths(1).map_err(|e| e.to_string())?;
            for entity in paths.remove(0).entities {
                rows.push(lattice.span_distribution(entity.tokens).map_err(|e| e.to_string())?);
                push_entity(entities, starts, ends, range.base, &entity)?;
            }
            return Ok(rows);
        }
        let paths = lattice.best_paths(self.candidate_paths).map_err(|e| e.to_string())?;
        let mut seen = BTreeSet::new();
        for path in &paths {
            for entity in &path.entities {
                if seen.insert((entity.tokens.start.get(), entity.tokens.end.get(), entity.kind)) {
                    rows.push(lattice.span_distribution(entity.tokens).map_err(|e| e.to_string())?);
                    push_entity(entities, starts, ends, range.base, entity)?;
                }
            }
        }
        Ok(rows)
    }
}

impl Annotator for CalibratedCrfNameAnnotator {
    const ID: &'static str = "turbo-namefind:crf-calibrated";
    const VERSION: &'static str = "2";
    type Reads = (View, Sentences, Tokens);
    type Writes = (Entities, Calibrated<Entities>);

    fn config_hash(&self) -> u64 {
        self.predictor.config_hash()
    }
    fn model(&self) -> Option<ModelRef> {
        let i = self.predictor.artifact().identity();
        Some(ModelRef { id: i.model.clone(), revision: i.revision.clone(), sha256: i.sha256.clone() })
    }

    fn annotate(&self, (view, sentences, tokens): Refs<'_, Self>) -> Result<Self::Writes, Error> {
        let fail = |detail: String| Error::Annotator { annotator: Self::ID, detail };
        sentences.check().map_err(fail)?;
        tokens.check().map_err(fail)?;
        let ranges = sentence_token_ranges(sentences, tokens, view).map_err(fail)?;
        let (starts, ends) = tokens.spans.columns();
        let text = view.text();
        let mut entities = Entities {
            spans: Spans::new(view.id()),
            tokens: Vec::new(),
            kind: Vec::new(),
            kinds: self.decoder.kinds().clone(),
            score: Vec::new(),
        };
        let mut calibrated = Calibrated::new(&self.predictor);
        for range in &ranges {
            let words: Vec<&str> =
                (range.base..range.end).map(|i| &text[starts[i] as usize..ends[i] as usize]).collect();
            let scores = (self.scores)(&words).map_err(fail)?;
            let lattice = self.decoder.lattice(scores.as_scores()).map_err(|e| fail(e.to_string()))?;
            for row in self.decoded_rows(&lattice, range, &mut entities, starts, ends).map_err(fail)? {
                calibrated.push(&self.predictor, &row).map_err(fail)?;
            }
        }
        entities.check().map_err(fail)?;
        calibrated.check().map_err(fail)?;
        Ok((entities, calibrated))
    }
}
