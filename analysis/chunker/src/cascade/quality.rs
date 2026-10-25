use super::Cascade;
use turbo_analysis_layers::{QualityFlags, SpanQuality};

impl Cascade<'_> {
    /// Intersect the eligible units with this quality floor, before either
    /// stage-1 shortlist consumes its candidate budget and before contextual
    /// reranking. Uncovered spans remain eligible; every bad actual fragment
    /// excludes the unit. Repeated calls can only remove more units.
    pub fn retain_by_quality(
        &mut self,
        quality: &SpanQuality,
        min_score: f32,
        forbidden: QualityFlags,
    ) -> Result<(), String> {
        let rejected = quality.rejected(&self.levels.spans, min_score, forbidden)?;
        let allowed = self.quality_allowed.get_or_insert_with(|| vec![true; self.levels.spans.len()]);
        for (keep, reject) in allowed.iter_mut().zip(rejected) {
            *keep &= !reject;
        }
        Ok(())
    }
}
