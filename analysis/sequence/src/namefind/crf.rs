//! Sequence-normalized name finding. Emissions depend only on observed
//! tokens and caller features. Previous-label generators receive constant
//! `other` history; adaptive state is cleared before every sentence.
use super::{DefaultNameContext, FoundName, NameCodec};
use crate::crf::{CrfError, CrfSequenceModel, FeatureRow};
use crate::{ContextGenerator, Features, PatternMode};
use std::borrow::BorrowMut;
use turbo_maxent::{JavaString, ModelFile};

pub const MODEL_ENTRY: &str = "nameFinder.crf";
pub const MODEL_TYPE: &str = "turbo-linear-chain-crf-v1";

impl DefaultNameContext {
    /// Build emissions without gold or decoded label history. Existing
    /// generators remain usable, including custom descriptors. Document
    /// adaptive features intentionally start empty for each sentence.
    pub fn sequence_features(&mut self, tokens: &[&str], additional: &[Vec<String>]) -> Vec<FeatureRow> {
        self.clear_adaptive_data();
        self.set_additional_context(additional);
        self.start_sequence();
        let prior = vec!["other"; tokens.len()];
        let mut features = Features::new();
        (0..tokens.len())
            .map(|i| {
                features.clear();
                for g in &mut self.generators {
                    g.create_features(&mut features, tokens, i, &prior);
                }
                features
                    .iter()
                    .map(|b| (JavaString::from_modified_utf8(b).expect("Features emits modified UTF-8"), 1.0))
                    .collect()
            })
            .collect()
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct CrfNameSequence {
    pub names: Vec<FoundName>,
    pub outcomes: Vec<String>,
    /// Probability of the complete path under the full constrained model.
    pub log_probability: f64,
    pub token_probabilities: Vec<f64>,
}

pub struct CrfNameFinder<'m, C = DefaultNameContext> {
    model: &'m CrfSequenceModel,
    context: C,
}
impl<'m> CrfNameFinder<'m> {
    pub fn from_file(model: &'m CrfSequenceModel, file: &ModelFile, mode: PatternMode) -> Result<Self, CrfError> {
        if file.manifest().get("Model-Type") != Some(MODEL_TYPE)
            || file.manifest().get(super::CODEC_PROPERTY) != Some(model.codec().class_name())
        {
            return Err(CrfError::Invalid("CRF package metadata".into()));
        }
        let context = DefaultNameContext::for_model(file, mode).map_err(|e| CrfError::Invalid(e.to_string()))?;
        Ok(Self::new(model, context))
    }
    pub fn read_model(file: &ModelFile) -> Result<CrfSequenceModel, CrfError> {
        if file.manifest().get("Model-Type") != Some(MODEL_TYPE) {
            return Err(CrfError::Invalid("CRF model type".into()));
        }
        let bytes = file.entry(MODEL_ENTRY).map_err(|e| CrfError::Invalid(e.to_string()))?;
        let model = CrfSequenceModel::from_bytes(&bytes)?;
        if file.manifest().get(super::CODEC_PROPERTY) != Some(model.codec().class_name()) {
            return Err(CrfError::Invalid("CRF codec metadata".into()));
        }
        Ok(model)
    }
}
impl<'m, C: BorrowMut<DefaultNameContext>> CrfNameFinder<'m, C> {
    pub fn new(model: &'m CrfSequenceModel, context: C) -> Self {
        Self { model, context }
    }
    pub fn into_context(self) -> C {
        self.context
    }
    pub fn find(&mut self, tokens: &[&str]) -> Result<Vec<FoundName>, CrfError> {
        Ok(self.best_sequences(tokens, &[], &[], 1)?.remove(0).names)
    }
    /// Top-1 decoded entities paired with their exact sequence-marginal
    /// distributions over `model.span_outcomes()` (`["other", kind_0, ...]`).
    pub fn find_scored_spans(&mut self, tokens: &[&str]) -> Result<(Vec<FoundName>, Vec<Vec<f64>>), CrfError> {
        if tokens.is_empty() {
            return Ok((Vec::new(), Vec::new()));
        }
        let rows = self.context.borrow_mut().sequence_features(tokens, &[]);
        let lattice = self.model.lattice(&rows)?;
        let path = lattice.best_paths(1)?.remove(0);
        let outcomes: Vec<_> = path.labels.iter().map(|&y| self.model.labels()[y].as_str()).collect();
        let decoded = self.model.codec().decode(&outcomes);
        let mut names = Vec::with_capacity(decoded.len());
        let mut distributions = Vec::with_capacity(decoded.len());
        for span in decoded {
            let forbidden = if self.model.codec() == NameCodec::Bio {
                let label = format!("{}-cont", span.kind.as_deref().unwrap_or("default"));
                self.model.labels().iter().position(|s| s == &label)
            } else {
                None
            };
            let prob = lattice.span_probability(span.start, &path.labels[span.start..span.end], forbidden)?;
            let dist = self.model.span_distribution(&lattice, span.start, span.end)?;
            names.push(FoundName { span, prob });
            distributions.push(dist);
        }
        Ok((names, distributions))
    }
    /// Ordered unique candidate spans from the top-`k` constrained paths paired
    /// with their exact sequence-marginal distributions over `model.span_outcomes()`.
    pub fn candidate_spans(&mut self, tokens: &[&str], k: usize) -> Result<Vec<(super::Span, Vec<f64>)>, CrfError> {
        if k == 0 {
            return Err(CrfError::Invalid("k must be positive".into()));
        }
        if tokens.is_empty() {
            return Ok(Vec::new());
        }
        let rows = self.context.borrow_mut().sequence_features(tokens, &[]);
        let lattice = self.model.lattice(&rows)?;
        let paths = lattice.best_paths(k)?;
        let mut intervals: Vec<(usize, usize)> = Vec::new();
        for path in &paths {
            let outcomes: Vec<_> = path.labels.iter().map(|&y| self.model.labels()[y].as_str()).collect();
            for span in self.model.codec().decode(&outcomes) {
                intervals.push((span.start, span.end));
            }
        }
        intervals.sort_unstable();
        intervals.dedup();
        let kinds = self.model.entity_kinds();
        let mut out = Vec::with_capacity(intervals.len());
        for (start, end) in intervals {
            let dist = self.model.span_distribution(&lattice, start, end)?;
            let best_kind = dist[1..]
                .iter()
                .enumerate()
                .max_by(|(ia, a), (ib, b)| a.total_cmp(b).then(ib.cmp(ia)))
                .and_then(|(i, _)| kinds.get(i).map(String::as_str));
            out.push((super::Span::new(start, end, best_kind), dist));
        }
        Ok(out)
    }
    /// Exact k-best paths, with optional extra signed numeric features per
    /// token. Name probabilities are exact span marginals, not means of
    /// token marginals or normalization over the returned k paths.
    pub fn best_sequences(
        &mut self,
        tokens: &[&str],
        additional: &[Vec<String>],
        numeric: &[FeatureRow],
        k: usize,
    ) -> Result<Vec<CrfNameSequence>, CrfError> {
        if (!additional.is_empty() && additional.len() != tokens.len())
            || (!numeric.is_empty() && numeric.len() != tokens.len())
        {
            return Err(CrfError::Invalid("per-token feature lengths".into()));
        }
        let mut rows = self.context.borrow_mut().sequence_features(tokens, additional);
        for (row, extra) in rows.iter_mut().zip(numeric) {
            row.extend_from_slice(extra);
        }
        let lattice = self.model.lattice(&rows)?;
        let paths = lattice.best_paths(k)?;
        paths
            .into_iter()
            .map(|path| {
                let outcomes: Vec<_> = path.labels.iter().map(|&y| self.model.labels()[y].clone()).collect();
                let refs: Vec<_> = outcomes.iter().map(String::as_str).collect();
                let names = self
                    .model
                    .codec()
                    .decode(&refs)
                    .into_iter()
                    .map(|span| {
                        let forbidden = if self.model.codec() == NameCodec::Bio {
                            let label = span.kind.as_ref().map_or_else(|| "cont".into(), |kind| format!("{kind}-cont"));
                            self.model.labels().iter().position(|s| s == &label)
                        } else {
                            None
                        };
                        let prob =
                            lattice.span_probability(span.start, &path.labels[span.start..span.end], forbidden)?;
                        Ok(FoundName { span, prob })
                    })
                    .collect::<Result<_, CrfError>>()?;
                Ok(CrfNameSequence {
                    names,
                    outcomes,
                    log_probability: path.log_probability,
                    token_probabilities: path.token_probabilities,
                })
            })
            .collect()
    }
}
