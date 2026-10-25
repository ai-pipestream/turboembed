use turbo_analysis::{Document, Key, ViewId};

use super::{QualityFlags, TextQuality};

/// A view-specific predicate over the default quality instance. Bind its
/// quality dependency with `ConditionalStep::require(Key::<TextQuality>::DEFAULT)`
/// so the Analyzer schedules quality production before the predicate.
/// Missing quality, invalid thresholds or a mismatched view fail closed.
pub fn quality_gate(
    view: ViewId,
    min_score: f32,
    forbidden: QualityFlags,
) -> impl Fn(&Document) -> bool + Send + Sync + 'static {
    quality_gate_on(Key::DEFAULT, view, min_score, forbidden)
}

/// The same predicate over an explicitly named quality instance.
pub fn quality_gate_on(
    key: Key<TextQuality>,
    view: ViewId,
    min_score: f32,
    forbidden: QualityFlags,
) -> impl Fn(&Document) -> bool + Send + Sync + 'static {
    let valid = min_score.is_finite() && (0.0..=1.0).contains(&min_score) && forbidden.0 & !QualityFlags::ALL.0 == 0;
    move |doc| {
        valid
            && doc.get_key(&key).is_some_and(|q| {
                q.view() == view && q.metrics.composite_score >= min_score && !q.metrics.flags.intersects(forbidden)
            })
    }
}

/// Information-density and MTLD floors over the default quality instance.
/// The predicate must declare the same quality key as a conditional dependency.
pub fn lexical_quality_gate(
    view: ViewId,
    min_density: f32,
    min_mtld: f32,
) -> impl Fn(&Document) -> bool + Send + Sync + 'static {
    lexical_quality_gate_on(Key::DEFAULT, view, min_density, min_mtld)
}

pub fn lexical_quality_gate_on(
    key: Key<TextQuality>,
    view: ViewId,
    min_density: f32,
    min_mtld: f32,
) -> impl Fn(&Document) -> bool + Send + Sync + 'static {
    let valid =
        min_density.is_finite() && (0.0..=1.0).contains(&min_density) && min_mtld.is_finite() && min_mtld >= 0.0;
    move |doc| {
        valid
            && doc.get_key(&key).is_some_and(|q| {
                q.view() == view
                    && q.metrics.information_density >= min_density
                    && q.metrics.mtld_lexical_diversity >= min_mtld
            })
    }
}
