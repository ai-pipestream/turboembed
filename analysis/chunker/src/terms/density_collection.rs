use turbo_analysis::Layer;

use super::{Collection, DensityTermStats, Vocabulary};

/// One document's quality-weighted units as a collection. Empty effective
/// units do not enter N or mean length; df counts positive-density rows.
pub struct DensityCollection<'a, L> {
    stats: &'a DensityTermStats<L>,
    vocabulary: &'a Vocabulary,
    units: u64,
    mean: f64,
}

impl<'a, L: Layer> DensityCollection<'a, L> {
    pub fn new(stats: &'a DensityTermStats<L>, vocabulary: &'a Vocabulary) -> Result<Self, String> {
        stats.check()?;
        vocabulary.check()?;
        if stats.df.len() != vocabulary.len() {
            return Err("density collection vocabulary must match its df column".into());
        }
        let (mut units, mut mean) = (0u64, 0.0f64);
        for &length in &stats.length {
            if length > 0.0 {
                units += 1;
                // Online mean avoids overflowing the sum of finite lengths.
                mean += (length - mean) / units as f64;
            }
        }
        Ok(Self { stats, vocabulary, units, mean })
    }
}

impl<L> Collection for DensityCollection<'_, L> {
    fn units(&self) -> u64 {
        self.units
    }
    fn mean_length(&self) -> f64 {
        self.mean
    }
    fn df(&self, term: &str) -> u64 {
        self.vocabulary.lookup(term).map_or(0, |t| u64::from(self.stats.df[t.get()]))
    }
}
