//! Opt-in NameFinderME token calibration and CRF span-marginal calibration.
//! Token prediction sets (`Calibrated<Tokens>`) are marginal token sets;
//! CRF span prediction sets (`Calibrated<Entities>`) are sequence-marginal
//! entity/non-entity sets aligned 1-to-1 with decoded `Entities`.
use std::sync::Arc;
use turbo_analysis::{Annotator, Error, Ix, IxRange, Layer, ModelRef, Refs, Spans, TagSet, View};
use turbo_analysis_layers::{Calibrated, Entities, Sentences, Tokens};
use turbo_calibration::{Calibration, Identity, Policy, Predictor, Task};
use turbo_maxent::{MaxentModel, ModelFile};
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
    file: Arc<ModelFile>,
    model: MaxentModel,
    predictor: Predictor,
    kinds: Arc<TagSet>,
}
impl CalibratedNameAnnotator {
    pub fn new(file: Arc<ModelFile>, calibration: Calibration, domain: &str, policy: Policy) -> Result<Self, String> {
        let expected = identity(&file, domain)?;
        let predictor = Predictor::new(calibration, &expected, policy)?;
        let model = file.maxent(MODEL_ENTRY_NAME).map_err(|e| e.to_string())?;
        NameFinder::new(&file, &model, PatternMode::Utf16Units).map_err(|e| e.to_string())?;
        let mut types: Vec<_> =
            predictor.artifact().identity().labels.iter().filter_map(|s| s.rsplit_once('-').map(|(t, _)| t)).collect();
        if predictor.artifact().identity().labels.iter().any(|s| matches!(s.as_str(), "start" | "cont")) {
            types.push("default");
        }
        types.sort_unstable();
        types.dedup();
        let kinds = TagSet::new("turbo:calibrated-name-types", &types);
        Ok(Self { file, model, predictor, kinds })
    }
    pub fn predictor(&self) -> &Predictor {
        &self.predictor
    }
}
impl Annotator for CalibratedNameAnnotator {
    const ID: &'static str = "turbo-namefind:calibrated";
    const VERSION: &'static str = "1";
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
        let mut finder =
            NameFinder::new(&self.file, &self.model, PatternMode::Utf16Units).map_err(|e| fail(e.to_string()))?;
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

/// Producer identity for span-level CRF calibration (`Task::SpanTagging`)
/// over `model.span_outcomes()` (`["other", kind_0, ...]`).
pub fn crf_identity(file: &ModelFile, domain: &str) -> Result<Identity, String> {
    let model = CrfNameFinder::read_model(file).map_err(|e| e.to_string())?;
    CrfNameFinder::from_file(&model, file, PatternMode::Utf16Units).map_err(|e| e.to_string())?;
    let inference_hash = model
        .codec()
        .class_name()
        .bytes()
        .chain(b";PatternMode=Utf16Units".iter().copied())
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
}
impl CalibratedCrfNameAnnotator {
    pub fn new(file: Arc<ModelFile>, calibration: Calibration, domain: &str, policy: Policy) -> Result<Self, String> {
        let expected = crf_identity(&file, domain)?;
        let predictor = Predictor::new(calibration, &expected, policy)?;
        let model = CrfNameFinder::read_model(&file).map_err(|e| e.to_string())?;
        CrfNameFinder::from_file(&model, &file, PatternMode::Utf16Units).map_err(|e| e.to_string())?;
        let entity_kinds = model.entity_kinds();
        let refs: Vec<_> = entity_kinds.iter().map(String::as_str).collect();
        let kinds = TagSet::new("turbo:crf-entities", &refs);
        Ok(Self { file, model, predictor, kinds })
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
            let (names, rows) = finder.find_scored_spans(&words).map_err(|e| fail(e.to_string()))?;
            for (name, row) in names.into_iter().zip(rows) {
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
