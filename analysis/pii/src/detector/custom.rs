//! The detector's custom-kind table and the caller extractors checked
//! against it.

use turbo_analysis_layers::{CustomKinds, KindError, PiiKind};

use super::PiiDetector;
use crate::hits::PiiSink;
use crate::scan::{CustomExtractor, Scanner};

impl PiiDetector {
    /// This detector with `kinds` as its custom-kind table, which every
    /// [`PiiKind::Custom`] given to it afterwards indexes and every layer it
    /// writes carries. Replaces any earlier table.
    ///
    /// # Panics
    ///
    /// When the replacement would change what a rule already set means;
    /// see [`PiiDetector::try_with_custom_kinds`].
    pub fn with_custom_kinds(self, kinds: CustomKinds) -> PiiDetector {
        self.try_with_custom_kinds(kinds).expect("a custom-kind table every detector rule agrees with")
    }

    /// This detector with `kinds` as its custom-kind table, refused when a
    /// custom kind that a scanner, a deny list, a per-kind allow list or a
    /// label override already uses is missing from `kinds`
    /// ([`KindError::UnknownCustom`]), or is named in the earlier table and
    /// named otherwise in `kinds` ([`KindError::Renamed`]). Every rule keeps
    /// the kind it was set for, so no rule outlives its kind and no layer
    /// carries a kind outside its table.
    pub fn try_with_custom_kinds(mut self, kinds: CustomKinds) -> Result<PiiDetector, KindError> {
        for kind in self.custom_kinds_in_use() {
            kinds.check(kind)?;
            if let (PiiKind::Custom(id), Some(was)) = (kind, self.custom_kinds.name(kind))
                && kinds.name(kind) != Some(was)
            {
                return Err(KindError::Renamed(id.index()));
            }
        }
        self.custom_kinds = kinds;
        Ok(self)
    }

    /// Every custom kind a rule of this detector names: scanner kinds, deny
    /// lists, per-kind allow lists and label overrides.
    fn custom_kinds_in_use(&self) -> impl Iterator<Item = PiiKind> + '_ {
        let scanners = self.scanners.iter().flat_map(|s| s.custom_kinds().iter().copied());
        let allow = self.kind_allow.iter().map(|(k, _)| *k);
        let deny = self.deny_lists.iter().map(|d| d.kind());
        scanners.chain(allow).chain(deny).chain(self.window.kinds()).filter(|k| k.is_custom())
    }

    /// The custom-kind table every layer this detector writes carries.
    pub fn custom_kinds(&self) -> &CustomKinds {
        &self.custom_kinds
    }

    /// Whether `kind` is built-in or in [`PiiDetector::custom_kinds`].
    pub fn check_kind(&self, kind: PiiKind) -> Result<(), KindError> {
        self.custom_kinds.check(kind)
    }

    /// This detector with a caller-defined [`CustomExtractor`] appended in
    /// composition order, refused when it declares a custom kind outside
    /// [`PiiDetector::custom_kinds`].
    pub fn with_custom<E: CustomExtractor + 'static>(self, extractor: E) -> Result<PiiDetector, KindError> {
        self.try_with_scanner(Scanner::custom(extractor))
    }

    /// This detector with a stateless caller function for `kind` appended
    /// in composition order, refused when `kind` is a custom kind outside
    /// [`PiiDetector::custom_kinds`].
    pub fn with_custom_fn(
        self,
        name: &'static str,
        kind: PiiKind,
        scan: fn(&str, &mut PiiSink<'_>),
    ) -> Result<PiiDetector, KindError> {
        self.try_with_scanner(Scanner::custom_fn(name, kind, scan))
    }

    /// This detector with `scanner` appended, refused when it declares a
    /// custom kind outside [`PiiDetector::custom_kinds`].
    pub fn try_with_scanner(mut self, scanner: Scanner) -> Result<PiiDetector, KindError> {
        for &k in scanner.custom_kinds() {
            self.check_kind(k)?;
        }
        self.scanners.push(scanner);
        Ok(self)
    }
}
