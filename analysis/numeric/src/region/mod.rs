//! Shared per-region number notation and currency symbol configuration, plus
//! region-aware annotators for numbers, quantities, and money over the
//! [`Regions`] ballot layer (`turbo:regions`).
//!
//! Resolves the document's winning ISO 3166-1 alpha-2 country code from the
//! [`Regions`] layer (`Regions::winner`) and configures [`NumberScanner`],
//! [`QuantityExtractor`], and [`MoneyExtractor`] with the country's
//! [`NumberNotation`] (and for money, its single-code-point `Sc` currency
//! symbol) as JDK 21's locale data resolves them. When a document's [`Regions`]
//! ballot carries votes for both [`NumberNotation::LatinEu`] and
//! [`NumberNotation::LatinUs`] regions, [`RegionAwareNumberAnnotator`] and
//! [`RegionAwareQuantityAnnotator`] try the winning region's notation first and
//! fall back to the secondary notation when a candidate numeral is written in
//! the other convention (`1.234,5` alongside `1,234.5`).

mod table;
#[cfg(test)]
mod tests;

pub(crate) use table::REGION_CONFIGS;

use turbo_analysis::{Annotator, Error, Refs, View};
use turbo_analysis_layers::{MarginError, Money, NumberNotation, Numbers, Quantities, RegionCode, Regions};

use crate::annotator::NumberScanner;
use crate::money::MoneyExtractor;
use crate::quantity::{DEFAULT_UNITS, QuantityExtractor, UnitError};

/// Resolves the [`NumberNotation`] for an ISO 3166-1 alpha-2 [`RegionCode`]
/// from the shared region configuration table, defaulting to
/// [`NumberNotation::LatinUs`].
pub fn region_notation(country_code: RegionCode) -> NumberNotation {
    let key = country_code.bytes();
    match REGION_CONFIGS.binary_search_by_key(&key, |&(k, _, _, _)| k) {
        Ok(idx) => REGION_CONFIGS[idx].1,
        Err(_) => NumberNotation::LatinUs,
    }
}

/// Resolves the [`NumberNotation`] for a BCP-47 language tag or two-letter
/// country code (`"en-US"`, `"de-DE"`, `"fr-FR"`, `"fr-CA"`, `"en-CA"`,
/// `"pt-BR"`, `"es-ES"`, `"es-419"`, `"nl-NL"`, `"it-IT"`, `""`), following
/// OpenNLP's `NumberNotation.forLocale(Locale)` on JDK 21.
pub fn locale_notation(locale: &str) -> NumberNotation {
    let trimmed = locale.trim();
    if trimmed.is_empty() {
        return NumberNotation::LatinUs;
    }
    let mut parts = trimmed.split(['-', '_']);
    let first = parts.next().unwrap_or("");
    let rest: Vec<&str> = parts.collect();
    if rest.is_empty()
        && first.len() == 2
        && first.bytes().all(|b| b.is_ascii_uppercase())
        && let Ok(rc) = RegionCode::parse(first)
    {
        return region_notation(rc);
    }
    let lang = first.to_ascii_lowercase();
    let country = rest
        .iter()
        .copied()
        .take_while(|s| s.len() > 1)
        .find(|s| {
            (s.len() == 2 && s.bytes().all(|b| b.is_ascii_alphabetic()))
                || (s.len() == 3 && s.bytes().all(|b| b.is_ascii_digit()))
        })
        .map(str::to_ascii_uppercase);
    match lang.as_str() {
        "af" | "az" | "be" | "bg" | "bs" | "ca" | "cs" | "da" | "el" | "et" | "eu" | "fi" | "fo" | "fr" | "gl"
        | "hr" | "hu" | "hy" | "id" | "is" | "ka" | "kk" | "ky" | "lb" | "lo" | "lt" | "lv" | "mk" | "nb" | "nl"
        | "nn" | "no" | "pl" | "pt" | "ro" | "ru" | "sk" | "sl" | "sq" | "sr" | "sv" | "tr" | "uk" | "vi" => {
            NumberNotation::LatinEu
        }
        "de" => match country.as_deref() {
            Some("CH" | "LI") => NumberNotation::LatinUs,
            _ => NumberNotation::LatinEu,
        },
        "it" => match country.as_deref() {
            Some("CH") => NumberNotation::LatinUs,
            _ => NumberNotation::LatinEu,
        },
        "es" => match country.as_deref() {
            Some("419" | "BR" | "BZ" | "CU" | "DO" | "GT" | "HN" | "MX" | "NI" | "PA" | "PE" | "PR" | "SV" | "US") => {
                NumberNotation::LatinUs
            }
            _ => NumberNotation::LatinEu,
        },
        "en" => match country.as_deref() {
            Some("AT" | "BE" | "DE" | "DK" | "FI" | "NL" | "SI") => NumberNotation::LatinEu,
            _ => NumberNotation::LatinUs,
        },
        "ar" => match country.as_deref() {
            Some("DZ" | "LY" | "MA" | "TN") => NumberNotation::LatinEu,
            _ => NumberNotation::LatinUs,
        },
        "ms" => match country.as_deref() {
            Some("BN" | "ID") => NumberNotation::LatinEu,
            _ => NumberNotation::LatinUs,
        },
        _ => country.and_then(|c| RegionCode::parse(&c).ok()).map_or(NumberNotation::LatinUs, region_notation),
    }
}

/// Resolves the primary and optional secondary [`NumberNotation`] from a
/// document's [`Regions`] ballot. When the ballot carries votes in both
/// conventions (`LatinEu` and `LatinUs`), returns `Some(secondary)` so a
/// mixed-region document reads both `1.234,5` and `1,234.5`.
fn ballot_notations(regions: &Regions, minimum_margin: f64) -> (NumberNotation, Option<NumberNotation>) {
    let primary = regions
        .winner(minimum_margin)
        .expect("minimum_margin validated at construction")
        .map_or(NumberNotation::LatinUs, |vote| region_notation(vote.country_code));
    let has_eu = regions.iter().any(|v| region_notation(v.country_code) == NumberNotation::LatinEu);
    let has_us = regions.iter().any(|v| region_notation(v.country_code) == NumberNotation::LatinUs);
    let secondary = if has_eu && has_us {
        Some(match primary {
            NumberNotation::LatinUs => NumberNotation::LatinEu,
            NumberNotation::LatinEu => NumberNotation::LatinUs,
        })
    } else {
        None
    };
    (primary, secondary)
}

impl NumberScanner {
    /// Builds a [`NumberScanner`] configured with the [`NumberNotation`] of
    /// `country_code` from the shared region table.
    pub fn for_region(country_code: RegionCode) -> NumberScanner {
        NumberScanner::new(region_notation(country_code))
    }

    /// Builds a [`NumberScanner`] configured with the [`NumberNotation`] of
    /// `locale` (BCP-47 language tag or country code).
    pub fn for_locale(locale: &str) -> NumberScanner {
        NumberScanner::new(locale_notation(locale))
    }
}

impl QuantityExtractor {
    /// Builds a [`QuantityExtractor`] configured with the default unit set,
    /// the [`NumberNotation`] of `country_code`, and [`MoneyExtractor::for_region`].
    pub fn for_region(country_code: RegionCode) -> QuantityExtractor {
        QuantityExtractor::with_units_for_region(DEFAULT_UNITS, country_code).expect("the default units are valid")
    }

    /// Builds a [`QuantityExtractor`] configured with custom `units`, the
    /// [`NumberNotation`] of `country_code`, and [`MoneyExtractor::for_region`].
    pub fn with_units_for_region<S: AsRef<str>>(
        units: impl IntoIterator<Item = S>,
        country_code: RegionCode,
    ) -> Result<QuantityExtractor, UnitError> {
        let notation = region_notation(country_code);
        Ok(QuantityExtractor::with_units(units, notation)?.with_money(MoneyExtractor::for_region(country_code)))
    }

    /// Builds a [`QuantityExtractor`] configured with the default unit set and
    /// the [`NumberNotation`] of `locale` (BCP-47 language tag or country code),
    /// using [`MoneyExtractor::default`] (with `locale`'s notation) to exclude
    /// money mentions. Use [`QuantityExtractor::for_region`] when the country's
    /// currency symbol override should also apply.
    pub fn for_locale(locale: &str) -> QuantityExtractor {
        QuantityExtractor::new(locale_notation(locale))
    }
}

impl MoneyExtractor {
    /// Builds a [`MoneyExtractor`] configured for `country_code` as OpenNLP's
    /// `RegionAwareMoneyAnnotator` and `CursorMoneyExtractor.forRegion(Locale)`
    /// do on JDK 21: selects the country's [`NumberNotation`] and overrides its
    /// single-code-point `Sc` currency symbol when one exists (`$` -> `AUD` for
    /// `AU`, `$` -> `CAD` for `CA`, `¥` -> `CNY` for `CN`, `\u{060B}` -> `AFN`
    /// for `AF`, `€` -> `EUR` with `LatinEu` for `DE`), keeping the rest of
    /// [`DEFAULT_SYMBOLS`](crate::money::DEFAULT_SYMBOLS).
    pub fn for_region(country_code: RegionCode) -> MoneyExtractor {
        let key = country_code.bytes();
        match REGION_CONFIGS.binary_search_by_key(&key, |&(k, _, _, _)| k) {
            Ok(idx) => match REGION_CONFIGS[idx] {
                (_, _, money_notation, Some((sym, cur))) => {
                    MoneyExtractor::new(money_notation).with_symbol(sym, cur).expect("checked JDK 21 currency symbol")
                }
                _ => MoneyExtractor::default(),
            },
            Err(_) => MoneyExtractor::default(),
        }
    }
}

/// Extracts money mentions using the currency symbol table and number notation
/// of the document's winning country (`opennlp.tools.money.RegionAwareMoneyAnnotator`).
///
/// Reads `(View, Regions)` and writes `(Money,)`. When the [`Regions`] ballot
/// is empty or its top vote does not lead the runner-up by at least
/// `minimum_margin`, falls back to [`MoneyExtractor::default`].
#[derive(Clone, Debug, PartialEq)]
pub struct RegionAwareMoneyAnnotator {
    minimum_margin: f64,
    default_extractor: MoneyExtractor,
}

impl Default for RegionAwareMoneyAnnotator {
    fn default() -> Self {
        Self::new()
    }
}

impl RegionAwareMoneyAnnotator {
    /// Creates an annotator with a minimum margin of `0.0`, electing the top
    /// ballot row whenever the ballot is non-empty.
    pub fn new() -> RegionAwareMoneyAnnotator {
        RegionAwareMoneyAnnotator { minimum_margin: 0.0, default_extractor: MoneyExtractor::default() }
    }

    /// Creates an annotator that elects the top ballot row only when its share
    /// leads the runner-up by at least `minimum_margin` (in `[0.0, 1.0]`).
    pub fn with_minimum_margin(minimum_margin: f64) -> Result<RegionAwareMoneyAnnotator, MarginError> {
        if !(0.0..=1.0).contains(&minimum_margin) {
            return Err(MarginError(minimum_margin));
        }
        Ok(RegionAwareMoneyAnnotator { minimum_margin, default_extractor: MoneyExtractor::default() })
    }

    /// The required lead margin in `[0.0, 1.0]`.
    pub fn minimum_margin(&self) -> f64 {
        self.minimum_margin
    }
}

impl Annotator for RegionAwareMoneyAnnotator {
    const ID: &'static str = "turbo-numeric:region-aware-money";
    type Reads = (View, Regions);
    type Writes = (Money,);

    fn annotate(&self, (view, regions): Refs<'_, Self>) -> Result<(Money,), Error> {
        let winner = regions.winner(self.minimum_margin).expect("minimum_margin validated at construction");
        match winner {
            Some(vote) => MoneyExtractor::for_region(vote.country_code).annotate((view,)),
            None => self.default_extractor.annotate((view,)),
        }
    }

    fn config_hash(&self) -> u64 {
        self.minimum_margin.to_bits().wrapping_mul(0x0100_0000_01b3) ^ 0xcbf2_9ce4_8422_2325
    }
}

/// Extracts number mentions (`turbo:numbers`) using the [`NumberNotation`] of
/// the document's [`Regions`] ballot.
///
/// Reads `(View, Regions)` and writes `(Numbers,)`. When the ballot carries
/// votes for both [`NumberNotation::LatinEu`] and [`NumberNotation::LatinUs`]
/// countries, candidate numbers are tried in the winning notation first and
/// fall back to the secondary notation when the primary notation rejects them.
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct RegionAwareNumberAnnotator {
    minimum_margin: f64,
    apply_scale: bool,
}

impl Default for RegionAwareNumberAnnotator {
    fn default() -> Self {
        Self::new()
    }
}

impl RegionAwareNumberAnnotator {
    /// Creates an annotator with a minimum margin of `0.0` and scale markers
    /// enabled.
    pub fn new() -> RegionAwareNumberAnnotator {
        RegionAwareNumberAnnotator { minimum_margin: 0.0, apply_scale: true }
    }

    /// Creates an annotator that elects the top ballot row only when its share
    /// leads the runner-up by at least `minimum_margin` (in `[0.0, 1.0]`).
    pub fn with_minimum_margin(minimum_margin: f64) -> Result<RegionAwareNumberAnnotator, MarginError> {
        if !(0.0..=1.0).contains(&minimum_margin) {
            return Err(MarginError(minimum_margin));
        }
        Ok(RegionAwareNumberAnnotator { minimum_margin, apply_scale: true })
    }

    /// Configures whether scale suffixes (`12k`) and scale words (`3 million`)
    /// are applied.
    pub fn with_scale(mut self, apply_scale: bool) -> RegionAwareNumberAnnotator {
        self.apply_scale = apply_scale;
        self
    }

    /// The required lead margin in `[0.0, 1.0]`.
    pub fn minimum_margin(&self) -> f64 {
        self.minimum_margin
    }

    /// Whether scale markers are applied.
    pub fn apply_scale(&self) -> bool {
        self.apply_scale
    }
}

impl Annotator for RegionAwareNumberAnnotator {
    const ID: &'static str = "turbo-numeric:region-aware-numbers";
    type Reads = (View, Regions);
    type Writes = (Numbers,);

    fn annotate(&self, (view, regions): Refs<'_, Self>) -> Result<(Numbers,), Error> {
        let text = view.text();
        if u32::try_from(text.len()).is_err() {
            return Err(Error::Annotator {
                annotator: Self::ID,
                detail: format!("a text of {} bytes is past the u32 offset range", text.len()),
            });
        }
        let (primary, secondary) = ballot_notations(regions, self.minimum_margin);
        let scanner = NumberScanner::new(primary).with_scale(self.apply_scale);
        let mut numbers = Numbers::new(view.id());
        scanner.scan_with_fallback(text, secondary, &mut numbers);
        Ok((numbers,))
    }

    fn config_hash(&self) -> u64 {
        [self.minimum_margin.to_bits(), u64::from(self.apply_scale)]
            .iter()
            .fold(0xcbf2_9ce4_8422_2325, |h, &x| (h ^ x).wrapping_mul(0x0100_0000_01b3))
    }
}

/// Extracts quantity and percentage mentions (`turbo:quantities`) using the
/// [`NumberNotation`] and currency symbol table of the document's [`Regions`]
/// ballot.
///
/// Reads `(View, Regions)` and writes `(Quantities,)`. When the ballot carries
/// votes for both [`NumberNotation::LatinEu`] and [`NumberNotation::LatinUs`]
/// countries, candidate quantities are tried in the winning notation first and
/// fall back to the secondary notation when the primary notation rejects them.
#[derive(Clone, Debug, PartialEq)]
pub struct RegionAwareQuantityAnnotator {
    minimum_margin: f64,
    us_extractor: QuantityExtractor,
    eu_extractor: QuantityExtractor,
}

impl Default for RegionAwareQuantityAnnotator {
    fn default() -> Self {
        Self::new()
    }
}

impl RegionAwareQuantityAnnotator {
    /// Creates an annotator with a minimum margin of `0.0` and the default
    /// unit set.
    pub fn new() -> RegionAwareQuantityAnnotator {
        RegionAwareQuantityAnnotator {
            minimum_margin: 0.0,
            us_extractor: QuantityExtractor::new(NumberNotation::LatinUs),
            eu_extractor: QuantityExtractor::new(NumberNotation::LatinEu),
        }
    }

    /// Creates an annotator that elects the top ballot row only when its share
    /// leads the runner-up by at least `minimum_margin` (in `[0.0, 1.0]`).
    pub fn with_minimum_margin(minimum_margin: f64) -> Result<RegionAwareQuantityAnnotator, MarginError> {
        if !(0.0..=1.0).contains(&minimum_margin) {
            return Err(MarginError(minimum_margin));
        }
        Ok(RegionAwareQuantityAnnotator { minimum_margin, ..Self::new() })
    }

    /// Replaces the recognized unit set across both notations.
    pub fn with_units<S: AsRef<str>>(
        mut self,
        units: impl IntoIterator<Item = S>,
    ) -> Result<RegionAwareQuantityAnnotator, UnitError> {
        let collected: Vec<String> = units.into_iter().map(|s| s.as_ref().to_string()).collect();
        self.us_extractor = QuantityExtractor::with_units(&collected, NumberNotation::LatinUs)?;
        self.eu_extractor = QuantityExtractor::with_units(&collected, NumberNotation::LatinEu)?;
        Ok(self)
    }

    /// The required lead margin in `[0.0, 1.0]`.
    pub fn minimum_margin(&self) -> f64 {
        self.minimum_margin
    }

    /// The unit tokens, sorted.
    pub fn units(&self) -> impl Iterator<Item = &str> {
        self.us_extractor.units()
    }
}

impl Annotator for RegionAwareQuantityAnnotator {
    const ID: &'static str = "turbo-numeric:region-aware-quantities";
    type Reads = (View, Regions);
    type Writes = (Quantities,);

    fn annotate(&self, (view, regions): Refs<'_, Self>) -> Result<(Quantities,), Error> {
        let text = view.text();
        if i32::try_from(text.len()).is_err() {
            return Err(Error::Annotator {
                annotator: Self::ID,
                detail: format!("a text of {} bytes is past the i32 scale range", text.len()),
            });
        }
        let winner = regions.winner(self.minimum_margin).expect("minimum_margin validated at construction");
        let (primary_notation, secondary_notation) = ballot_notations(regions, self.minimum_margin);
        let base_primary = match primary_notation {
            NumberNotation::LatinUs => &self.us_extractor,
            NumberNotation::LatinEu => &self.eu_extractor,
        };
        let primary = match winner {
            Some(vote) => base_primary.clone().with_money(MoneyExtractor::for_region(vote.country_code)),
            None => base_primary.clone(),
        };
        let fallback = secondary_notation.map(|alt| match alt {
            NumberNotation::LatinUs => &self.us_extractor,
            NumberNotation::LatinEu => &self.eu_extractor,
        });
        let mut quantities = Quantities::new(view.id());
        primary.scan_with_fallback(text, fallback, &mut quantities);
        Ok((quantities,))
    }

    fn config_hash(&self) -> u64 {
        let units = self.us_extractor.units().flat_map(|u| u.bytes().chain([0]).map(u64::from));
        std::iter::once(self.minimum_margin.to_bits())
            .chain(units)
            .fold(0xcbf2_9ce4_8422_2325, |h, x| (h ^ x).wrapping_mul(0x0100_0000_01b3))
    }
}
