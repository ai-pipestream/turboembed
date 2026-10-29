//! How a feature's count becomes its value: OpenNLP's
//! `TermWeightingStrategy`, the scaling `DocumentCategorizerSVM` applies
//! after it, and the `FeatureSelectionStrategy` the vocabulary was chosen
//! with.

/// `TermWeightingStrategy`.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum TermWeighting {
    /// 1 for a feature present, whatever its count.
    Binary,
    /// The count.
    TermFrequency,
    /// The count times the feature's inverse document frequency.
    TfIdf,
    /// `1 + Math.log(count)`.
    LogNormalizedTf,
}

impl TermWeighting {
    pub const ALL: [TermWeighting; 4] =
        [TermWeighting::Binary, TermWeighting::TermFrequency, TermWeighting::TfIdf, TermWeighting::LogNormalizedTf];

    /// The Java constant's name.
    pub const fn name(self) -> &'static str {
        match self {
            TermWeighting::Binary => "BINARY",
            TermWeighting::TermFrequency => "TERM_FREQUENCY",
            TermWeighting::TfIdf => "TF_IDF",
            TermWeighting::LogNormalizedTf => "LOG_NORMALIZED_TF",
        }
    }

    pub fn from_name(name: &str) -> Option<TermWeighting> {
        TermWeighting::ALL.into_iter().find(|w| w.name() == name)
    }

    /// `weight(termFrequency, inverseDocumentFrequency)`. The logarithm is
    /// `Math.log`, which `turbo_jvm::math::log` matches on integers.
    #[inline]
    pub fn weight(self, tf: i32, idf: f64) -> f64 {
        match self {
            TermWeighting::Binary => {
                if tf > 0 {
                    1.0
                } else {
                    0.0
                }
            }
            TermWeighting::TermFrequency => f64::from(tf),
            TermWeighting::TfIdf => f64::from(tf) * idf,
            TermWeighting::LogNormalizedTf => {
                if tf > 0 {
                    1.0 + turbo_jvm::math::log(f64::from(tf))
                } else {
                    0.0
                }
            }
        }
    }
}

/// `FeatureSelectionStrategy`: how training chose the vocabulary. It
/// changes nothing at prediction, which reads the vocabulary as chosen;
/// the sidecar records it so the model says how it was made.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum FeatureSelection {
    None,
    InformationGain,
    ChiSquare,
    TermFrequency,
    DocumentFrequency,
}

impl FeatureSelection {
    pub const ALL: [FeatureSelection; 5] = [
        FeatureSelection::None,
        FeatureSelection::InformationGain,
        FeatureSelection::ChiSquare,
        FeatureSelection::TermFrequency,
        FeatureSelection::DocumentFrequency,
    ];

    /// The Java constant's name.
    pub const fn name(self) -> &'static str {
        match self {
            FeatureSelection::None => "NONE",
            FeatureSelection::InformationGain => "INFORMATION_GAIN",
            FeatureSelection::ChiSquare => "CHI_SQUARE",
            FeatureSelection::TermFrequency => "TERM_FREQUENCY",
            FeatureSelection::DocumentFrequency => "DOCUMENT_FREQUENCY",
        }
    }

    pub fn from_name(name: &str) -> Option<FeatureSelection> {
        FeatureSelection::ALL.into_iter().find(|s| s.name() == name)
    }
}

/// `SvmDoccatConfiguration`'s scaling: off, or each value mapped from its
/// training range onto `lower..=upper`.
#[derive(Copy, Clone, Debug, PartialEq)]
pub enum Scaling {
    None,
    Range { lower: f64, upper: f64 },
}

/// `DocumentCategorizerSVM.scale`: `lower` for a feature whose training
/// minimum and maximum are equal (by `Double.compare`), otherwise the
/// value clamped to the training range and mapped linearly onto
/// `lower..=upper`.
#[inline]
pub fn scale(value: f64, min: f64, max: f64, lower: f64, upper: f64) -> f64 {
    if crate::java::double_compare(max, min).is_eq() {
        return lower;
    }
    let clamped = crate::java::max(min, crate::java::min(max, value));
    lower + (upper - lower) * (clamped - min) / (max - min)
}

#[cfg(test)]
mod tests {
    use super::*;

    // TermWeightingStrategyTest of OpenNLP.
    #[test]
    fn binary_is_one_for_any_count() {
        assert_eq!(TermWeighting::Binary.weight(1, 0.0), 1.0);
        assert_eq!(TermWeighting::Binary.weight(5, 2.0), 1.0);
        assert_eq!(TermWeighting::Binary.weight(0, 0.0), 0.0);
    }

    #[test]
    fn term_frequency_is_the_count() {
        assert_eq!(TermWeighting::TermFrequency.weight(3, 0.0), 3.0);
        assert_eq!(TermWeighting::TermFrequency.weight(0, 0.0), 0.0);
        assert_eq!(TermWeighting::TermFrequency.weight(1, 5.0), 1.0);
    }

    #[test]
    fn tf_idf_is_the_count_times_the_idf() {
        let idf = (10.0f64 / 2.0).ln();
        assert!((TermWeighting::TfIdf.weight(3, idf) - 3.0 * idf).abs() < 1e-9);
        assert_eq!(TermWeighting::TfIdf.weight(3, idf), 3.0 * idf);
        assert_eq!(TermWeighting::TfIdf.weight(0, idf), 0.0);
    }

    #[test]
    fn log_normalized_tf_is_one_plus_the_log() {
        assert!((TermWeighting::LogNormalizedTf.weight(3, 0.0) - (1.0 + 3f64.ln())).abs() < 1e-9);
        assert_eq!(TermWeighting::LogNormalizedTf.weight(1, 0.0), 1.0);
        assert_eq!(TermWeighting::LogNormalizedTf.weight(0, 0.0), 0.0);
        // Math.log(2) on the JVM.
        assert_eq!(TermWeighting::LogNormalizedTf.weight(2, 0.0), 1.0 + f64::from_bits(0x3fe6_2e42_fefa_39ef));
    }

    #[test]
    fn names_are_the_java_constants() {
        for w in TermWeighting::ALL {
            assert_eq!(TermWeighting::from_name(w.name()), Some(w));
        }
        for s in FeatureSelection::ALL {
            assert_eq!(FeatureSelection::from_name(s.name()), Some(s));
        }
        assert_eq!(TermWeighting::from_name("tf_idf"), None);
        assert_eq!(FeatureSelection::from_name("CHI2"), None);
    }

    #[test]
    fn scaling_clamps_to_the_training_range() {
        assert_eq!(scale(1.0, 0.0, 2.0, 0.0, 1.0), 0.5);
        assert_eq!(scale(5.0, 0.0, 2.0, 0.0, 1.0), 1.0);
        assert_eq!(scale(-5.0, 0.0, 2.0, -1.0, 1.0), -1.0);
        assert_eq!(scale(1.0, 0.0, 2.0, -1.0, 1.0), 0.0);
        // Equal bounds give the lower end, whatever the value.
        assert_eq!(scale(7.0, 3.0, 3.0, -1.0, 1.0), -1.0);
        // 0.0 and -0.0 differ by Double.compare, so the range is used and
        // divides zero by zero, as the JVM does.
        assert!(scale(0.0, -0.0, 0.0, 0.0, 1.0).is_nan());
    }
}
