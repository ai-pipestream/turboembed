//! A base gazetteer with a caller's additions and suppressions laid over it
//! (`opennlp.geo.OverlayGazetteer`).

use std::fmt;

use crate::entry::RegionCode;
use crate::fold::Folder;
use crate::gazetteer::Gazetteer;
use crate::geocode::by_prior_view;
use crate::linker::FuzzyStringScorer;
use crate::store::Entry;
use crate::suppression::Suppression;

/// One merged view of a base gazetteer, the records of a second gazetteer
/// added to it, and rules that hide base records.
///
/// Lookups list the additions' candidates first, then the base candidates
/// no rule hides, each side in its own order. A rule hides base records
/// only, never additions, so a rule and an addition with the same name
/// replace a base record. A hidden record is hidden from every query.
pub struct OverlayGazetteer<B, A> {
    base: B,
    additions: Option<A>,
    suppressions: Vec<Suppression>,
}

/// An overlay with neither additions nor suppressions, which would change
/// nothing.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct EmptyOverlay;

impl fmt::Display for EmptyOverlay {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(
            "An overlay without additions and without suppressions changes nothing; use the base gazetteer directly",
        )
    }
}

impl std::error::Error for EmptyOverlay {}

impl<B: Gazetteer, A: Gazetteer> OverlayGazetteer<B, A> {
    /// Lays `additions` and `suppressions` over `base`. At least one of
    /// them must change something.
    pub fn new(base: B, additions: Option<A>, suppressions: Vec<Suppression>) -> Result<Self, EmptyOverlay> {
        if additions.is_none() && suppressions.is_empty() {
            return Err(EmptyOverlay);
        }
        Ok(OverlayGazetteer { base, additions, suppressions })
    }

    pub fn base(&self) -> &B {
        &self.base
    }

    pub fn additions(&self) -> Option<&A> {
        self.additions.as_ref()
    }

    pub fn suppressions(&self) -> &[Suppression] {
        &self.suppressions
    }

    /// Whether a rule hides the base record `entry`.
    fn suppressed(&self, entry: &Entry<'_>, folder: &mut Folder) -> bool {
        self.suppressions.iter().any(|s| s.matches(entry, folder))
    }
}

impl<B: Gazetteer, A: Gazetteer> Gazetteer for OverlayGazetteer<B, A> {
    fn lookup(&self, name: &str) -> Vec<Entry<'_>> {
        let mut merged = self.additions.as_ref().map_or_else(Vec::new, |a| a.lookup(name));
        let mut folder = Folder::new();
        merged.extend(self.base.lookup(name).into_iter().filter(|e| !self.suppressed(e, &mut folder)));
        merged
    }

    /// The additions are asked first; a base record is returned only when
    /// no rule hides it.
    fn by_id(&self, source: &str, record_id: &str) -> Option<Entry<'_>> {
        if let Some(found) = self.additions.as_ref().and_then(|a| a.by_id(source, record_id)) {
            return Some(found);
        }
        self.base.by_id(source, record_id).filter(|e| !self.suppressed(e, &mut Folder::new()))
    }

    /// The more populous of the two sides' representatives, the additions'
    /// on a tie. When a rule hides the base's representative the base
    /// offers none, since it has no second choice to give.
    fn by_region(&self, code: RegionCode) -> Option<Entry<'_>> {
        let from_base = self.base.by_region(code).filter(|e| !self.suppressed(e, &mut Folder::new()));
        let Some(additions) = &self.additions else { return from_base };
        match (additions.by_region(code), from_base) {
            (Some(a), Some(b)) => Some(if a.population() >= b.population() { a } else { b }),
            (a, b) => a.or(b),
        }
    }

    /// The additions' sources, then the base's not among them.
    fn sources(&self) -> Vec<&str> {
        let mut union: Vec<&str> = self.additions.as_ref().map_or_else(Vec::new, |a| a.sources());
        for s in self.base.sources() {
            if !union.contains(&s) {
                union.push(s);
            }
        }
        union
    }

    fn lookup_fuzzy(
        &self,
        name: &str,
        scorer: &FuzzyStringScorer,
        min_similarity: f64,
        limit: usize,
    ) -> Vec<(Entry<'_>, f64)> {
        if limit == 0 {
            return Vec::new();
        }
        let mut merged =
            self.additions.as_ref().map_or_else(Vec::new, |a| a.lookup_fuzzy(name, scorer, min_similarity, limit));
        let mut folder = Folder::new();
        let base_limit = if self.suppressions.is_empty() { limit } else { usize::MAX };
        for (entry, sim) in self.base.lookup_fuzzy(name, scorer, min_similarity, base_limit) {
            if !self.suppressed(&entry, &mut folder) {
                merged.push((entry, sim));
            }
        }
        merged.sort_by(|(ea, sa), (eb, sb)| sb.total_cmp(sa).then_with(|| by_prior_view(ea, eb)));
        merged.truncate(limit);
        merged
    }
}
