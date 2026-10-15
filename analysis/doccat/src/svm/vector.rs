//! A document's feature vector, as `DocumentCategorizerSVM` builds it for
//! `categorize`: each feature's count, the features of the vocabulary
//! only, weighted, scaled, and the ones that come out zero dropped.

use super::libsvm::PredictScratch;
use super::sidecar::Sidecar;
use super::weighting::{Scaling, scale};
use crate::features::for_each_feature;
use crate::java::is_not_zero;

/// Buffers for one document, kept between calls so a categorizer run on
/// many documents allocates only while they grow.
#[derive(Debug, Default, Clone)]
pub struct Scratch {
    /// The count of each feature index; zero between documents.
    counts: Vec<i32>,
    /// The feature indexes counted for this document.
    touched: Vec<u32>,
    feature: String,
    pub(crate) index: Vec<i32>,
    pub(crate) value: Vec<f64>,
    pub(crate) predict: PredictScratch,
    pub(crate) outcome: Vec<f64>,
}

impl Scratch {
    pub fn new() -> Scratch {
        Scratch::default()
    }

    /// The last document's vector: feature indexes, ascending, and their
    /// values.
    pub fn vector(&self) -> (&[i32], &[f64]) {
        (&self.index, &self.value)
    }

    /// The decision values of the last prediction, one per class pair in
    /// the LIBSVM model's label order.
    pub fn decision_values(&self) -> &[f64] {
        self.predict.decision_values()
    }
}

/// `textToSvmDocument` and `buildFeatureVector`: fills `s.index` and
/// `s.value` with the vector of `tokens`.
pub(crate) fn vectorize(sidecar: &Sidecar, tokens: &[&str], s: &mut Scratch) {
    let vocabulary = &sidecar.vocabulary;
    if s.counts.len() <= vocabulary.len() {
        s.counts.resize(vocabulary.len() + 1, 0);
    }
    let (counts, touched) = (&mut s.counts, &mut s.touched);
    touched.clear();
    for_each_feature(&sidecar.generators, tokens, &mut s.feature, |f| {
        if let Some(i) = vocabulary.index(f) {
            let c = &mut counts[i as usize];
            if *c == 0 {
                touched.push(i);
            }
            *c = c.wrapping_add(1);
        }
    });
    // The JVM collects the weighted features in a TreeMap by index.
    touched.sort_unstable();
    s.index.clear();
    s.value.clear();
    for &i in touched.iter() {
        let tf = std::mem::take(&mut counts[i as usize]);
        let mut value = sidecar.weighting.weight(tf, vocabulary.idf(i));
        if !is_not_zero(value) {
            continue;
        }
        if let Scaling::Range { lower, upper } = sidecar.scaling {
            let (min, max) = vocabulary.range(i);
            value = scale(value, min, max, lower, upper);
        }
        if is_not_zero(value) {
            s.index.push(i as i32);
            s.value.push(value);
        }
    }
}
