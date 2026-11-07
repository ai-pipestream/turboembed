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
    pub fn with_custom_kinds(mut self, kinds: CustomKinds) -> PiiDetector {
        self.custom_kinds = kinds;
        self
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
