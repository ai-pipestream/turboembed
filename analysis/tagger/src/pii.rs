//! Neural PII token-classification annotator (`turboembed:pii`, #1888):
//! decodes BIO or BILOU spans from a bundle's [`turbo::head::TokenTagger`]
//! head, maps the 55 `Nemotron-PII` types (or custom kebab-case types) to
//! [`PiiKind`], attaches numeric [`turbo_pii::PiiScore`] probabilities
//! with model provenance and optional temperature calibration, and yields
//! to checksum/format [`PiiDetector`] rules on span overlap.

use std::path::Path;

use turbo::manifest::Scheme;
use turbo::status::{Error as CoreError, INVALID_ARGUMENT};
use turbo_analysis::{Annotator, Error, Execution, Fnv1a, Ix, Layer, ModelRef, Refs, Span, View};
use turbo_analysis_embed::EncoderOptions;
use turbo_analysis_layers::{Pii, PiiCandidates, Subwords, Tokens};
use turbo_pii::{
    AllowList, CustomKinds, ModelCandidate, ModelProvenance, PiiDetector, PiiKind, register_nemotron_kinds,
    resolve_model_with_rules, resolve_nemotron_kind,
};

use crate::annotate::HeadRunner;

/// Detector id recorded in `Pii::detectors` for [`PiiTagger`] candidates.
pub const PII_TAGGER_DETECTOR: &str = "turbo-tagger:pii";

/// Writes `turbo:pii` and `turbo:pii-candidates` from a bundle's
/// `SCHEME_BIO` or `SCHEME_BILOU` token-classification head, optionally
/// combined with [`PiiDetector`] rules that win on overlap.
pub struct PiiTagger {
    run: HeadRunner,
    custom_kinds: CustomKinds,
    type_kinds: Vec<PiiKind>,
    model_id: String,
    model_version: String,
    temperature: Option<f32>,
    min_score: Option<f32>,
    rules: Option<PiiDetector>,
    allow: AllowList,
}

impl PiiTagger {
    /// Opens the PII head named `name` in the bundle at `dir`, or with no
    /// name the one head whose `layer` matches `turbo:pii`
    /// (`turboembed:pii`).
    pub fn open(dir: &Path, name: Option<&str>, o: &EncoderOptions) -> Result<PiiTagger, CoreError> {
        let run = HeadRunner::open(dir, name, Pii::ID, o)?;
        if run.scheme() == Scheme::Tags {
            return Err(CoreError::new(
                INVALID_ARGUMENT,
                "a PII token-tagger head reads SCHEME_BIO or SCHEME_BILOU labels",
            ));
        }
        let mut custom_kinds = CustomKinds::new();
        register_nemotron_kinds(&mut custom_kinds).map_err(|e| CoreError::new(INVALID_ARGUMENT, e.to_string()))?;
        let mut type_kinds = Vec::with_capacity(run.tagger().types().len());
        for raw in run.tagger().types() {
            let kind = resolve_nemotron_kind(raw, &mut custom_kinds)
                .map_err(|e| CoreError::new(INVALID_ARGUMENT, format!("pii head label {raw:?}: {e}")))?;
            type_kinds.push(kind);
        }
        let model_id = format!("{}:{}", run.model().id, run.tagger().spec().name);
        let model_version = run.model().revision.clone();
        Ok(PiiTagger {
            run,
            custom_kinds,
            type_kinds,
            model_id,
            model_version,
            temperature: None,
            min_score: None,
            rules: None,
            allow: AllowList::new(),
        })
    }

    /// Combines this neural tagger with `rules`: structured rule mentions
    /// carry `score: None` and win whenever their spans overlap a neural
    /// candidate.
    pub fn with_rules(mut self, rules: PiiDetector) -> PiiTagger {
        self.rules = Some(rules);
        self
    }

    /// Applies post-hoc temperature scaling (`softmax(logits / temperature)`)
    /// fitted on a held-out validation split and marks emitted [`turbo_pii::PiiScore`]s
    /// with `calibrated: true`.
    pub fn with_temperature(mut self, temperature: f32) -> Result<PiiTagger, CoreError> {
        if !temperature.is_finite() || !(1e-6..=1e6).contains(&temperature) {
            return Err(CoreError::new(
                INVALID_ARGUMENT,
                format!("calibration temperature {temperature} must be finite and in [1e-6, 1e6]"),
            ));
        }
        self.temperature = Some(temperature);
        Ok(self)
    }

    /// Rejects neural candidates whose probability is below `min`
    /// (`RejectReason::Threshold` in `turbo:pii-candidates`), leaving
    /// unscored rule detections untouched.
    pub fn with_score_threshold(mut self, min: f32) -> Result<PiiTagger, CoreError> {
        if !min.is_finite() || !(0.0..=1.0).contains(&min) {
            return Err(CoreError::new(INVALID_ARGUMENT, "probability threshold must be finite and in [0.0, 1.0]"));
        }
        self.min_score = Some(min);
        Ok(self)
    }

    /// Overrides the `(id, version)` recorded in `PiiScore::model`.
    pub fn with_model_identity(
        mut self,
        id: impl Into<String>,
        version: impl Into<String>,
    ) -> Result<PiiTagger, CoreError> {
        let (id, version) = (id.into(), version.into());
        if id.is_empty() || id.len() > 128 || version.len() > 128 {
            return Err(CoreError::new(INVALID_ARGUMENT, "model id must be 1..=128 bytes and version <= 128 bytes"));
        }
        self.model_id = id;
        self.model_version = version;
        Ok(self)
    }

    /// Merges `allow` into this tagger's global allow list.
    pub fn with_allow_list(mut self, allow: AllowList) -> PiiTagger {
        self.allow.merge(allow);
        self
    }

    /// Custom-kind table covering this head's non-built-in PII kinds.
    pub const fn custom_kinds(&self) -> &CustomKinds {
        &self.custom_kinds
    }

    /// Configured calibration temperature, when set.
    pub const fn temperature(&self) -> Option<f32> {
        self.temperature
    }

    /// Decodes neural PII candidates on `view` before rule resolution.
    pub fn model_candidates(
        &self,
        view: &View,
        tokens: &Tokens,
        subwords: Option<&Subwords>,
    ) -> Result<Vec<ModelCandidate>, String> {
        let seqs = HeadRunner::sequences(view, tokens, None)?;
        let decoded = self.run.decode_tokens_temperature(view, tokens, &seqs, subwords, self.temperature)?;
        let mut out = Vec::new();
        for (s, d) in seqs.iter().zip(&decoded) {
            for w in self.run.tagger().spans(&d.tags).map_err(|e| e.to_string())? {
                let (a, b) = (s[w.start as usize], s[w.end as usize - 1]);
                let start = tokens.spans.get(Ix::new(a as u32)).start;
                let end = tokens.spans.get(Ix::new(b as u32)).end;
                let p = &d.prob[w.start as usize..w.end as usize];
                let mean_prob = (p.iter().sum::<f64>() / p.len() as f64).clamp(0.0, 1.0) as f32;
                out.push(ModelCandidate {
                    span: Span::new(start, end),
                    kind: self.type_kinds[w.kind as usize],
                    probability: mean_prob,
                });
            }
        }
        Ok(out)
    }

    /// Runs the neural head (and configured `rules`, if any) over `view`.
    pub fn detect(
        &self,
        view: &View,
        tokens: &Tokens,
        subwords: Option<&Subwords>,
    ) -> Result<(Pii, PiiCandidates), String> {
        let raw = self.model_candidates(view, tokens, subwords)?;
        let provenance = ModelProvenance {
            detector_id: PII_TAGGER_DETECTOR,
            detector_version: env!("CARGO_PKG_VERSION"),
            model_id: &self.model_id,
            model_version: &self.model_version,
            calibrated: self.temperature.is_some(),
        };
        let extra_allow = (!self.allow.is_empty()).then_some(&self.allow);
        resolve_model_with_rules(
            view.text(),
            view.id(),
            self.rules.as_ref(),
            &self.custom_kinds,
            provenance,
            self.min_score,
            extra_allow,
            &raw,
        )
    }
}

impl Annotator for PiiTagger {
    const ID: &'static str = "turbo-tagger-pii";
    type Reads = (View, Tokens, Option<Subwords>);
    type Writes = (Pii, PiiCandidates);

    fn annotate(&self, (view, tokens, subwords): Refs<'_, Self>) -> Result<Self::Writes, Error> {
        self.detect(view, tokens, subwords).map_err(|detail| Error::Annotator { annotator: Self::ID, detail })
    }

    fn model(&self) -> Option<ModelRef> {
        Some(self.run.model().clone())
    }

    fn execution(&self) -> Option<Execution> {
        Some(self.run.execution().clone())
    }

    fn config_hash(&self) -> u64 {
        let mut h = Fnv1a::new();
        h.str(&self.run.tagger().spec().name).u64(self.run.width() as u64);
        h.str(&self.model_id).str(&self.model_version);
        if let Some(t) = self.temperature {
            h.u8(0x74).f32(t);
        }
        if let Some(m) = self.min_score {
            h.u8(0x6d).f32(m);
        }
        if let Some(r) = &self.rules {
            h.u8(0x72).u64(r.config_hash());
        }
        if !self.allow.is_empty() {
            h.u8(0x61).u64(self.allow.config_hash());
        }
        h.finish()
    }
}
