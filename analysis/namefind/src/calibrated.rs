//! Opt-in NameFinderME token calibration and CRF span-marginal calibration.
//! Token prediction sets (`Calibrated<Tokens>`) are marginal token sets;
//! CRF span prediction sets (`Calibrated<Entities>`) are sequence-marginal
//! entity/non-entity sets aligned 1-to-1 with decoded `Entities`.
use std::sync::Arc;
use turbo_analysis::{Annotator, Error, Ix, IxRange, Layer, ModelRef, Refs, Spans, TagSet, View};
use turbo_analysis_layers::{Calibrated, Entities, Sentences, Tokens};
use turbo_calibration::{Calibration, Identity, Policy, Predictor, Task};
use turbo_maxent::{MaxentModel, ModelFile};
use turbo_sequence::namefind::{NameCodec, NameContextFactory};
use turbo_sequence::{
    PatternMode,
    crf::CrfSequenceModel,
    namefind::{CrfNameFinder, DEFAULT_BEAM_SIZE, MODEL_ENTRY_NAME, NameFinder},
};

pub fn identity(file: &ModelFile, domain: &str) -> Result<Identity, String> {
    let model = file.maxent(MODEL_ENTRY_NAME).map_err(|e| e.to_string())?;
    let labels = model
        .outcome_texts()
        .map_err(|e| format!("non-text name outcome: {e:?}"))?
        .into_iter()
        .map(str::to_owned)
        .collect();
    let i = Identity {
        model: "opennlp:NameFinderME".into(),
        revision: file.opennlp_version().into(),
        sha256: file.sha256().to_hex(),
        task: Task::TokenTagging,
        domain: domain.into(),
        labels,
        inference_hash: (turbo_sequence::manifest_beam_size(file, DEFAULT_BEAM_SIZE).map_err(|e| e.to_string())?
            as u64)
            .to_le_bytes()
            .into_iter()
            .chain(b"PatternMode=Utf16Units".iter().copied())
            .fold(0xcbf29ce484222325, |h, b| (h ^ u64::from(b)).wrapping_mul(0x100000001b3)),
    };
    i.check()?;
    Ok(i)
}
pub struct CalibratedNameAnnotator {
    context: NameContextFactory,
    beam: usize,
    codec: NameCodec,
    model: MaxentModel,
    predictor: Predictor,
    kinds: Arc<TagSet>,
}
impl CalibratedNameAnnotator {
    pub fn new(file: Arc<ModelFile>, calibration: Calibration, domain: &str, policy: Policy) -> Result<Self, String> {
        let expected = identity(&file, domain)?;
        let predictor = Predictor::new(calibration, &expected, policy)?;
        let model = file.maxent(MODEL_ENTRY_NAME).map_err(|e| e.to_string())?;
        let codec = NameCodec::for_model(&file);
        let context = NameContextFactory::for_model(&file, PatternMode::Utf16Units).map_err(|e| e.to_string())?;
        let beam = turbo_sequence::manifest_beam_size(&file, DEFAULT_BEAM_SIZE).map_err(|e| e.to_string())?;
        NameFinder::with_context(&model, beam, codec, context.create().map_err(|e| e.to_string())?)
            .map_err(|e| e.to_string())?;
        let kinds = crate::entity_kinds::from_labels(
            "turbo:calibrated-name-types",
            predictor.artifact().identity().labels.iter().map(String::as_str),
        );
        Ok(Self { context, beam, codec, model, predictor, kinds })
    }
    pub fn predictor(&self) -> &Predictor {
        &self.predictor
    }
}
impl Annotator for CalibratedNameAnnotator {
    const ID: &'static str = "turbo-namefind:calibrated";
    const VERSION: &'static str = "2";
    type Reads = (View, Sentences, Tokens);
    type Writes = (Entities, Calibrated<Tokens>);
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
        check_spans(&sentences.spans, view).map_err(fail)?;
        check_spans(&tokens.spans, view).map_err(fail)?;
        let mut entities = Entities {
            spans: Spans::new(view.id()),
            tokens: Vec::new(),
            kind: Vec::new(),
            kinds: self.kinds.clone(),
            score: Vec::new(),
        };
        let mut calibrated = Calibrated::new(&self.predictor);
        let mut context = self.context.create().map_err(|e| fail(e.to_string()))?;
        let mut finder = NameFinder::with_context(&self.model, self.beam, self.codec, &mut context)
            .map_err(|e| fail(e.to_string()))?;
        let (starts, ends) = tokens.spans.columns();
        let mut base = 0;
        for si in 0..sentences.len() {
            let sentence = sentences.spans.get(Ix::new(si as u32));
            let mut end = base;
            while end < tokens.len() && starts[end] < sentence.end {
                if starts[end] < sentence.start
                    || ends[end] > sentence.end
                    || tokens.sentence.as_ref().is_some_and(|s| s[end].get() != si)
                {
                    return Err(fail("name finder token/sentence mismatch".into()));
                }
                end += 1;
            }
            let words: Vec<_> = (base..end).map(|i| &view.text()[starts[i] as usize..ends[i] as usize]).collect();
            let (names, rows) = finder.find_scored_flat(&words);
            let width = self.predictor.artifact().identity().labels.len();
            if rows.len() / width != words.len() || rows.len() % width != 0 {
                return Err(fail("name finder produced no token sequence".into()));
            }
            for row in rows.chunks_exact(width) {
                calibrated.push(&self.predictor, row).map_err(fail)?;
            }
            for name in names {
                let start = base + name.span.start;
                let stop = base + name.span.end;
                if start >= stop || stop > end {
                    return Err(fail("invalid decoded name span".into()));
                }
                let kind = entities
                    .kinds
                    .tag(name.span.kind.as_deref().unwrap_or("default"))
                    .ok_or_else(|| fail("name kind absent from model labels".into()))?;
                entities.spans.push(turbo_analysis::Span::new(starts[start], ends[stop - 1]));
                entities.tokens.push(IxRange::new(start as u32, stop as u32));
                entities.kind.push(kind);
                entities.score.push(name.prob);
            }
            base = end;
        }
        if base != tokens.len() {
            return Err(fail("tokens outside sentence spans".into()));
        }
        Ok((entities, calibrated))
    }
}

pub const DEFAULT_CRF_CANDIDATE_PATHS: usize = 1;

/// Producer identity for span-level CRF calibration (`Task::SpanTagging`)
/// over `model.span_outcomes()` (`["other", kind_0, ...]`) on top-1 decoded spans.
pub fn crf_identity(file: &ModelFile, domain: &str) -> Result<Identity, String> {
    crf_identity_with_candidates(file, domain, DEFAULT_CRF_CANDIDATE_PATHS)
}

/// Producer identity for span-level CRF calibration over unique candidate spans
/// from the top-`k` Viterbi paths (`k == 1` selects top-1 decoded entities).
pub fn crf_identity_with_candidates(file: &ModelFile, domain: &str, k: usize) -> Result<Identity, String> {
    if k == 0 || k > 4096 {
        return Err("CRF candidate path count k must be in 1..=4096".into());
    }
    let model = CrfNameFinder::read_model(file).map_err(|e| e.to_string())?;
    CrfNameFinder::from_file(&model, file, PatternMode::Utf16Units).map_err(|e| e.to_string())?;
    let inference_hash = model
        .codec()
        .class_name()
        .bytes()
        .chain(b";PatternMode=Utf16Units;CandidatePaths=".iter().copied())
        .chain((k as u64).to_le_bytes())
        .fold(0xcbf29ce484222325, |h, b| (h ^ u64::from(b)).wrapping_mul(0x100000001b3));
    let i = Identity {
        model: "turbo-namefind:crf".into(),
        revision: file.opennlp_version().into(),
        sha256: file.sha256().to_hex(),
        task: Task::SpanTagging,
        domain: domain.into(),
        labels: model.span_outcomes(),
        inference_hash,
    };
    i.check()?;
    Ok(i)
}

/// Span-calibrated CRF name annotator writing raw `Entities` and an aligned
/// `Calibrated<Entities>` sidecar computed from CRF forward-backward span marginals.
pub struct CalibratedCrfNameAnnotator {
    file: Arc<ModelFile>,
    model: CrfSequenceModel,
    predictor: Predictor,
    kinds: Arc<TagSet>,
    candidate_paths: usize,
}
impl CalibratedCrfNameAnnotator {
    pub fn new(file: Arc<ModelFile>, calibration: Calibration, domain: &str, policy: Policy) -> Result<Self, String> {
        Self::with_candidates(file, calibration, domain, policy, DEFAULT_CRF_CANDIDATE_PATHS)
    }
    pub fn with_candidates(
        file: Arc<ModelFile>,
        calibration: Calibration,
        domain: &str,
        policy: Policy,
        candidate_paths: usize,
    ) -> Result<Self, String> {
        let expected = crf_identity_with_candidates(&file, domain, candidate_paths)?;
        let predictor = Predictor::new(calibration, &expected, policy)?;
        let model = CrfNameFinder::read_model(&file).map_err(|e| e.to_string())?;
        CrfNameFinder::from_file(&model, &file, PatternMode::Utf16Units).map_err(|e| e.to_string())?;
        let entity_kinds = model.entity_kinds();
        let refs: Vec<_> = entity_kinds.iter().map(String::as_str).collect();
        let kinds = TagSet::new("turbo:crf-entities", &refs);
        Ok(Self { file, model, predictor, kinds, candidate_paths })
    }
    pub fn candidate_paths(&self) -> usize {
        self.candidate_paths
    }
    pub fn predictor(&self) -> &Predictor {
        &self.predictor
    }
}
impl Annotator for CalibratedCrfNameAnnotator {
    const ID: &'static str = "turbo-namefind:crf-calibrated";
    const VERSION: &'static str = "1";
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
        check_spans(&sentences.spans, view).map_err(fail)?;
        check_spans(&tokens.spans, view).map_err(fail)?;
        let mut entities = Entities {
            spans: Spans::new(view.id()),
            tokens: Vec::new(),
            kind: Vec::new(),
            kinds: self.kinds.clone(),
            score: Vec::new(),
        };
        let mut calibrated = Calibrated::new(&self.predictor);
        let mut finder = CrfNameFinder::from_file(&self.model, &self.file, PatternMode::Utf16Units)
            .map_err(|e| fail(e.to_string()))?;
        let (starts, ends) = tokens.spans.columns();
        let mut base = 0;
        for si in 0..sentences.len() {
            let sentence = sentences.spans.get(Ix::new(si as u32));
            let mut end = base;
            while end < tokens.len() && starts[end] < sentence.end {
                if starts[end] < sentence.start
                    || ends[end] > sentence.end
                    || tokens.sentence.as_ref().is_some_and(|s| s[end].get() != si)
                {
                    return Err(fail("name finder token/sentence mismatch".into()));
                }
                end += 1;
            }
            let words: Vec<_> = (base..end).map(|i| &view.text()[starts[i] as usize..ends[i] as usize]).collect();
            let spans_and_rows: Vec<_> = if self.candidate_paths == 1 {
                let (names, rows) = finder.find_scored_spans(&words).map_err(|e| fail(e.to_string()))?;
                names.into_iter().zip(rows).map(|(n, r)| (n.span, Some(n.prob), r)).collect()
            } else {
                finder
                    .candidate_spans(&words, self.candidate_paths)
                    .map_err(|e| fail(e.to_string()))?
                    .into_iter()
                    .map(|(s, r)| (s, None, r))
                    .collect()
            };
            for (span, prob, row) in spans_and_rows {
                let start = base + span.start;
                let stop = base + span.end;
                if start >= stop || stop > end {
                    return Err(fail("invalid decoded name span".into()));
                }
                let kind = entities
                    .kinds
                    .tag(span.kind.as_deref().unwrap_or("default"))
                    .ok_or_else(|| fail("name kind absent from model labels".into()))?;
                let score = prob.unwrap_or_else(|| row.get(1 + kind.0 as usize).copied().unwrap_or(0.0));
                entities.spans.push(turbo_analysis::Span::new(starts[start], ends[stop - 1]));
                entities.tokens.push(IxRange::new(start as u32, stop as u32));
                entities.kind.push(kind);
                entities.score.push(score);
                calibrated.push(&self.predictor, &row).map_err(fail)?;
            }
            base = end;
        }
        if base != tokens.len() {
            return Err(fail("tokens outside sentence spans".into()));
        }
        entities.check().map_err(fail)?;
        calibrated.check().map_err(fail)?;
        Ok((entities, calibrated))
    }
}

fn check_spans<L: Layer>(spans: &Spans<L>, view: &View) -> Result<(), String> {
    if spans.view() != view.id() {
        return Err("calibrated name finder view mismatch".into());
    }
    let mut last = 0;
    for i in 0..spans.len() {
        let ix = Ix::new(i as u32);
        let span = spans.get(ix);
        if spans.is_discontinuous(ix)
            || span.start < last
            || span.start >= span.end
            || view.text().get(span.start as usize..span.end as usize).is_none()
        {
            return Err("name finder needs ordered nonempty contiguous UTF-8 spans".into());
        }
        last = span.end;
    }
    Ok(())
}
