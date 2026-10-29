//! Binary feature-presence MI and Pearson chi-squared selection across classes.
//! Fit on training events only, then filter before indexing or optimization.
//! Repeated predicates count once for ranking; filtering preserves their original
//! multiplicity and numeric values for the chosen downstream trainer.
use crate::{DoccatFactory, doccat_events};
use std::collections::{BTreeMap, BTreeSet};
use turbo_formats::DocumentSample;
use turbo_train::{Event, JavaString};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SelectionMetric {
    MutualInformation,
    Chi2,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct FeatureSelection {
    pub metric: SelectionMetric,
    pub top_k: usize,
}
#[derive(Clone, Debug)]
pub struct SelectedFeatures {
    ranked: Vec<(JavaString, f64)>,
    kept: BTreeSet<JavaString>,
}
impl FeatureSelection {
    pub fn fit(&self, events: &[Event]) -> Result<SelectedFeatures, String> {
        if self.top_k == 0 || events.is_empty() {
            return Err("feature selection needs positive top_k and training events".into());
        }
        let classes: BTreeSet<_> = events.iter().map(Event::outcome).collect();
        if classes.len() < 2 {
            return Err("feature selection requires at least two classes".into());
        }
        let classes: BTreeMap<_, _> = classes.into_iter().enumerate().map(|(i, c)| (c, i)).collect();
        let mut totals = vec![0usize; classes.len()];
        let mut present: BTreeMap<JavaString, Vec<usize>> = BTreeMap::new();
        for e in events {
            let c = classes[e.outcome()];
            totals[c] += 1;
            if e.values().is_some_and(|v| v.iter().any(|v| !v.is_finite() || *v < 0.)) {
                return Err("feature selection needs finite nonnegative feature values".into());
            }
            let unique: BTreeSet<_> = e
                .context()
                .iter()
                .enumerate()
                .filter(|(i, _)| e.values().is_none_or(|v| v[*i] > 0.))
                .map(|(_, f)| f)
                .collect();
            for f in unique {
                present.entry(f.clone()).or_insert_with(|| vec![0; classes.len()])[c] += 1;
            }
        }
        let n = events.len() as f64;
        let mut ranked = Vec::with_capacity(present.len());
        for (feature, counts) in present {
            let has = counts.iter().sum::<usize>() as f64;
            let mut score = 0.;
            for (c, &nc) in totals.iter().enumerate() {
                for (observed, nf) in [(counts[c] as f64, has), ((nc - counts[c]) as f64, n - has)] {
                    let expected = nf * (nc as f64 / n);
                    if expected == 0. {
                        continue;
                    }
                    match self.metric {
                        SelectionMetric::MutualInformation => {
                            if observed > 0. {
                                score += observed / n * (observed / expected).ln();
                            }
                        }
                        SelectionMetric::Chi2 => {
                            let delta = observed - expected;
                            score += delta * (delta / expected);
                        }
                    }
                }
            }
            if !score.is_finite() {
                return Err("non-finite feature selection score".into());
            }
            ranked.push((feature, score.max(0.)));
        }
        ranked.sort_by(|a, b| b.1.total_cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
        let kept = ranked.iter().take(self.top_k).map(|(f, _)| f.clone()).collect();
        Ok(SelectedFeatures { ranked, kept })
    }
}
impl SelectedFeatures {
    pub fn ranked(&self) -> &[(JavaString, f64)] {
        &self.ranked
    }
    pub fn kept(&self) -> &BTreeSet<JavaString> {
        &self.kept
    }
    /// The output can feed MaxEnt or a Liblinear index builder unchanged.
    pub fn filter(&self, events: &[Event]) -> Result<Vec<Event>, String> {
        events
            .iter()
            .map(|e| {
                let mut context = Vec::new();
                let mut values = e.values().map(|_| Vec::new());
                for (i, f) in e.context().iter().enumerate() {
                    if self.kept.contains(f) {
                        context.push(f.clone());
                        if let Some(v) = &mut values {
                            v.push(e.values().unwrap()[i]);
                        }
                    }
                }
                Event::with_values(e.outcome().clone(), context, values).map_err(|e| e.to_string())
            })
            .collect()
    }
}
/// Prepares document events with optional selection before a downstream
/// optimizer. Use the retained model predicates at inference; unseen terms
/// have no learned weight and do not need a second feature filter.
#[derive(Clone, Debug)]
pub struct DoccatTrainer {
    factory: DoccatFactory,
    selection: Option<FeatureSelection>,
}
impl DoccatTrainer {
    pub fn new(factory: DoccatFactory) -> Self {
        Self { factory, selection: None }
    }
    pub fn with_feature_selection(mut self, selection: FeatureSelection) -> Self {
        self.selection = Some(selection);
        self
    }
    pub fn events<'a, 's>(
        &self,
        samples: impl IntoIterator<Item = &'s DocumentSample<'a>>,
    ) -> Result<Vec<Event>, String>
    where
        'a: 's,
    {
        let events = doccat_events(samples, &self.factory);
        match self.selection {
            Some(s) => s.fit(&events)?.filter(&events),
            None => Ok(events),
        }
    }
}
