//! Region-aware money extraction (`CursorMoneyExtractor.forRegion` and
//! `RegionAwareMoneyAnnotator` in `opennlp.tools.money`).
//!
//! Resolves the document's winning ISO 3166-1 alpha-2 country code from the
//! [`Regions`] layer (`Regions::winner`) and configures a [`MoneyExtractor`]
//! with the country's [`NumberNotation`] and single-code-point currency symbol
//! (category `Sc`) as JDK 21's locale data resolves them. When the ballot has
//! no decisive winner or the winning country's currency symbol is not a
//! single `Sc` code point (such as `CHF` for Switzerland), the default symbol
//! table is used.

use turbo_analysis::{Annotator, Error, Refs, View};
use turbo_analysis_layers::{Currency, MarginError, Money, NumberNotation, RegionCode, Regions};

use crate::money::MoneyExtractor;

/// A currency code from a literal the table knows to be well formed.
const fn code(c: &[u8; 3]) -> Currency {
    match Currency::new(*c) {
        Some(c) => c,
        None => panic!("a code of three ASCII capital letters"),
    }
}

type RegionConfig = ([u8; 2], NumberNotation, Option<(char, Currency)>);

/// Every two-letter country code for which `RegionAwareMoneyAnnotator.localeOf`
/// resolves a locale with a single-code-point `Sc` currency symbol on the
/// pinned JDK 21 runtime (`Temurin-21.0.12.1+1`,
/// `eclipse-temurin@sha256:3e3c176ffed168beb42c607be9bc1639b466cf00261a0fb04425562c9d0c5c2b`),
/// along with `CursorMoneyExtractor.forRegion(locale)`'s [`NumberNotation`]
/// and symbol override (`(symbol, currency)`).
///
/// `RegionAwareMoneyAnnotator.localeOf(cc)` tries `Locale.of("en", cc)` first
/// whenever `en-<CC>` writes the currency as a single `Sc` code point (so
/// `en-CA` writes `$` instead of `fr-CA`'s `CA$`), and otherwise scans
/// `Locale.getAvailableLocales()` for a usable locale with the
/// lexicographically smallest `toLanguageTag()`:
/// - Every Eurozone country resolves through `en-<CC>` because `en-<CC>` writes
///   the euro as `€` (`U+20AC`). In JDK 21's CLDR data, `en-AT`, `en-BE`,
///   `en-DE`, `en-FI`, `en-NL`, and `en-SI` use `,` as their decimal separator
///   ([`NumberNotation::LatinEu`]), while the remaining Eurozone `en-<CC>`
///   locales (`AD`, `AX`, `BG`, `BL`, `CY`, `EE`, `ES`, `FR`, `GF`, `GP`,
///   `GR`, `HR`, `IE`, `IT`, `LT`, `LU`, `LV`, `MC`, `ME`, `MF`, `MQ`, `MT`,
///   `PM`, `PT`, `RE`, `SK`, `SM`, `TF`, `VA`, `YT`) use `.`
///   ([`NumberNotation::LatinUs`]). `BG` maps to `EUR` because
///   `Temurin-21.0.12.1+1` includes Bulgaria's `2026-01-01` euro cutover.
/// - When `en-<CC>` writes a multi-character symbol (`RUB` for `en-RU`, `CN¥`
///   for `en-CN`, `AFN` for `en-AF`), the installed-locale scan selects the
///   lexicographically first usable language tag (`ce-Cyrl-RU` with `.` ->
///   `LatinUs` and `\u{20BD}` -> `RUB`; `bo-CN` with `.` -> `LatinUs` and
///   `\u{00A5}` -> `CNY`; `fa-AF` with `.` -> `LatinUs` and `\u{060B}` ->
///   `AFN`; `km-KH` with `,` -> `LatinEu` and `\u{17DB}` -> `KHR`).
///
/// Sorted by two-byte country code.
const REGION_CONFIGS: [RegionConfig; 122] = [
    (*b"AD", NumberNotation::LatinUs, Some(('\u{20AC}', code(b"EUR")))),
    (*b"AF", NumberNotation::LatinUs, Some(('\u{060B}', code(b"AFN")))),
    (*b"AG", NumberNotation::LatinUs, Some(('\u{0024}', code(b"XCD")))),
    (*b"AI", NumberNotation::LatinUs, Some(('\u{0024}', code(b"XCD")))),
    (*b"AM", NumberNotation::LatinEu, Some(('\u{058F}', code(b"AMD")))),
    (*b"AR", NumberNotation::LatinEu, Some(('\u{0024}', code(b"ARS")))),
    (*b"AS", NumberNotation::LatinUs, Some(('\u{0024}', code(b"USD")))),
    (*b"AT", NumberNotation::LatinEu, Some(('\u{20AC}', code(b"EUR")))),
    (*b"AU", NumberNotation::LatinUs, Some(('\u{0024}', code(b"AUD")))),
    (*b"AX", NumberNotation::LatinUs, Some(('\u{20AC}', code(b"EUR")))),
    (*b"AZ", NumberNotation::LatinEu, Some(('\u{20BC}', code(b"AZN")))),
    (*b"BB", NumberNotation::LatinUs, Some(('\u{0024}', code(b"BBD")))),
    (*b"BD", NumberNotation::LatinUs, Some(('\u{09F3}', code(b"BDT")))),
    (*b"BE", NumberNotation::LatinEu, Some(('\u{20AC}', code(b"EUR")))),
    (*b"BG", NumberNotation::LatinUs, Some(('\u{20AC}', code(b"EUR")))),
    (*b"BL", NumberNotation::LatinUs, Some(('\u{20AC}', code(b"EUR")))),
    (*b"BM", NumberNotation::LatinUs, Some(('\u{0024}', code(b"BMD")))),
    (*b"BN", NumberNotation::LatinEu, Some(('\u{0024}', code(b"BND")))),
    (*b"BQ", NumberNotation::LatinUs, Some(('\u{0024}', code(b"USD")))),
    (*b"BS", NumberNotation::LatinUs, Some(('\u{0024}', code(b"BSD")))),
    (*b"BZ", NumberNotation::LatinUs, Some(('\u{0024}', code(b"BZD")))),
    (*b"CA", NumberNotation::LatinUs, Some(('\u{0024}', code(b"CAD")))),
    (*b"CC", NumberNotation::LatinUs, Some(('\u{0024}', code(b"AUD")))),
    (*b"CK", NumberNotation::LatinUs, Some(('\u{0024}', code(b"NZD")))),
    (*b"CL", NumberNotation::LatinEu, Some(('\u{0024}', code(b"CLP")))),
    (*b"CN", NumberNotation::LatinUs, Some(('\u{00A5}', code(b"CNY")))),
    (*b"CO", NumberNotation::LatinEu, Some(('\u{0024}', code(b"COP")))),
    (*b"CR", NumberNotation::LatinEu, Some(('\u{20A1}', code(b"CRC")))),
    (*b"CU", NumberNotation::LatinUs, Some(('\u{0024}', code(b"CUP")))),
    (*b"CX", NumberNotation::LatinUs, Some(('\u{0024}', code(b"AUD")))),
    (*b"CY", NumberNotation::LatinUs, Some(('\u{20AC}', code(b"EUR")))),
    (*b"DE", NumberNotation::LatinEu, Some(('\u{20AC}', code(b"EUR")))),
    (*b"DM", NumberNotation::LatinUs, Some(('\u{0024}', code(b"XCD")))),
    (*b"EC", NumberNotation::LatinUs, Some(('\u{0024}', code(b"USD")))),
    (*b"EE", NumberNotation::LatinUs, Some(('\u{20AC}', code(b"EUR")))),
    (*b"ES", NumberNotation::LatinUs, Some(('\u{20AC}', code(b"EUR")))),
    (*b"FI", NumberNotation::LatinEu, Some(('\u{20AC}', code(b"EUR")))),
    (*b"FJ", NumberNotation::LatinUs, Some(('\u{0024}', code(b"FJD")))),
    (*b"FK", NumberNotation::LatinUs, Some(('\u{00A3}', code(b"FKP")))),
    (*b"FR", NumberNotation::LatinUs, Some(('\u{20AC}', code(b"EUR")))),
    (*b"GB", NumberNotation::LatinUs, Some(('\u{00A3}', code(b"GBP")))),
    (*b"GD", NumberNotation::LatinUs, Some(('\u{0024}', code(b"XCD")))),
    (*b"GE", NumberNotation::LatinEu, Some(('\u{20BE}', code(b"GEL")))),
    (*b"GF", NumberNotation::LatinUs, Some(('\u{20AC}', code(b"EUR")))),
    (*b"GG", NumberNotation::LatinUs, Some(('\u{00A3}', code(b"GBP")))),
    (*b"GI", NumberNotation::LatinUs, Some(('\u{00A3}', code(b"GIP")))),
    (*b"GP", NumberNotation::LatinUs, Some(('\u{20AC}', code(b"EUR")))),
    (*b"GR", NumberNotation::LatinUs, Some(('\u{20AC}', code(b"EUR")))),
    (*b"GS", NumberNotation::LatinUs, Some(('\u{00A3}', code(b"GBP")))),
    (*b"GU", NumberNotation::LatinUs, Some(('\u{0024}', code(b"USD")))),
    (*b"GY", NumberNotation::LatinUs, Some(('\u{0024}', code(b"GYD")))),
    (*b"HR", NumberNotation::LatinUs, Some(('\u{20AC}', code(b"EUR")))),
    (*b"IE", NumberNotation::LatinUs, Some(('\u{20AC}', code(b"EUR")))),
    (*b"IL", NumberNotation::LatinUs, Some(('\u{20AA}', code(b"ILS")))),
    (*b"IM", NumberNotation::LatinUs, Some(('\u{00A3}', code(b"GBP")))),
    (*b"IN", NumberNotation::LatinUs, Some(('\u{20B9}', code(b"INR")))),
    (*b"IT", NumberNotation::LatinUs, Some(('\u{20AC}', code(b"EUR")))),
    (*b"JE", NumberNotation::LatinUs, Some(('\u{00A3}', code(b"GBP")))),
    (*b"JM", NumberNotation::LatinUs, Some(('\u{0024}', code(b"JMD")))),
    (*b"JP", NumberNotation::LatinUs, Some(('\u{00A5}', code(b"JPY")))),
    (*b"KH", NumberNotation::LatinEu, Some(('\u{17DB}', code(b"KHR")))),
    (*b"KI", NumberNotation::LatinUs, Some(('\u{0024}', code(b"AUD")))),
    (*b"KN", NumberNotation::LatinUs, Some(('\u{0024}', code(b"XCD")))),
    (*b"KR", NumberNotation::LatinUs, Some(('\u{20A9}', code(b"KRW")))),
    (*b"KY", NumberNotation::LatinUs, Some(('\u{0024}', code(b"KYD")))),
    (*b"KZ", NumberNotation::LatinEu, Some(('\u{20B8}', code(b"KZT")))),
    (*b"LA", NumberNotation::LatinEu, Some(('\u{20AD}', code(b"LAK")))),
    (*b"LC", NumberNotation::LatinUs, Some(('\u{0024}', code(b"XCD")))),
    (*b"LR", NumberNotation::LatinUs, Some(('\u{0024}', code(b"LRD")))),
    (*b"LT", NumberNotation::LatinUs, Some(('\u{20AC}', code(b"EUR")))),
    (*b"LU", NumberNotation::LatinUs, Some(('\u{20AC}', code(b"EUR")))),
    (*b"LV", NumberNotation::LatinUs, Some(('\u{20AC}', code(b"EUR")))),
    (*b"MC", NumberNotation::LatinUs, Some(('\u{20AC}', code(b"EUR")))),
    (*b"ME", NumberNotation::LatinUs, Some(('\u{20AC}', code(b"EUR")))),
    (*b"MF", NumberNotation::LatinUs, Some(('\u{20AC}', code(b"EUR")))),
    (*b"MH", NumberNotation::LatinUs, Some(('\u{0024}', code(b"USD")))),
    (*b"MN", NumberNotation::LatinUs, Some(('\u{20AE}', code(b"MNT")))),
    (*b"MP", NumberNotation::LatinUs, Some(('\u{0024}', code(b"USD")))),
    (*b"MQ", NumberNotation::LatinUs, Some(('\u{20AC}', code(b"EUR")))),
    (*b"MS", NumberNotation::LatinUs, Some(('\u{0024}', code(b"XCD")))),
    (*b"MT", NumberNotation::LatinUs, Some(('\u{20AC}', code(b"EUR")))),
    (*b"MX", NumberNotation::LatinUs, Some(('\u{0024}', code(b"MXN")))),
    (*b"NA", NumberNotation::LatinUs, Some(('\u{0024}', code(b"NAD")))),
    (*b"NF", NumberNotation::LatinUs, Some(('\u{0024}', code(b"AUD")))),
    (*b"NG", NumberNotation::LatinUs, Some(('\u{20A6}', code(b"NGN")))),
    (*b"NL", NumberNotation::LatinEu, Some(('\u{20AC}', code(b"EUR")))),
    (*b"NR", NumberNotation::LatinUs, Some(('\u{0024}', code(b"AUD")))),
    (*b"NU", NumberNotation::LatinUs, Some(('\u{0024}', code(b"NZD")))),
    (*b"NZ", NumberNotation::LatinUs, Some(('\u{0024}', code(b"NZD")))),
    (*b"PH", NumberNotation::LatinUs, Some(('\u{20B1}', code(b"PHP")))),
    (*b"PM", NumberNotation::LatinUs, Some(('\u{20AC}', code(b"EUR")))),
    (*b"PN", NumberNotation::LatinUs, Some(('\u{0024}', code(b"NZD")))),
    (*b"PR", NumberNotation::LatinUs, Some(('\u{0024}', code(b"USD")))),
    (*b"PS", NumberNotation::LatinUs, Some(('\u{20AA}', code(b"ILS")))),
    (*b"PT", NumberNotation::LatinUs, Some(('\u{20AC}', code(b"EUR")))),
    (*b"RE", NumberNotation::LatinUs, Some(('\u{20AC}', code(b"EUR")))),
    (*b"RU", NumberNotation::LatinUs, Some(('\u{20BD}', code(b"RUB")))),
    (*b"SB", NumberNotation::LatinUs, Some(('\u{0024}', code(b"SBD")))),
    (*b"SG", NumberNotation::LatinUs, Some(('\u{0024}', code(b"SGD")))),
    (*b"SH", NumberNotation::LatinUs, Some(('\u{00A3}', code(b"SHP")))),
    (*b"SI", NumberNotation::LatinEu, Some(('\u{20AC}', code(b"EUR")))),
    (*b"SK", NumberNotation::LatinUs, Some(('\u{20AC}', code(b"EUR")))),
    (*b"SM", NumberNotation::LatinUs, Some(('\u{20AC}', code(b"EUR")))),
    (*b"SR", NumberNotation::LatinEu, Some(('\u{0024}', code(b"SRD")))),
    (*b"SS", NumberNotation::LatinUs, Some(('\u{00A3}', code(b"SSP")))),
    (*b"TF", NumberNotation::LatinUs, Some(('\u{20AC}', code(b"EUR")))),
    (*b"TH", NumberNotation::LatinUs, Some(('\u{0E3F}', code(b"THB")))),
    (*b"TK", NumberNotation::LatinUs, Some(('\u{0024}', code(b"NZD")))),
    (*b"TL", NumberNotation::LatinUs, Some(('\u{0024}', code(b"USD")))),
    (*b"TR", NumberNotation::LatinEu, Some(('\u{20BA}', code(b"TRY")))),
    (*b"TT", NumberNotation::LatinUs, Some(('\u{0024}', code(b"TTD")))),
    (*b"TV", NumberNotation::LatinUs, Some(('\u{0024}', code(b"AUD")))),
    (*b"TW", NumberNotation::LatinUs, Some(('\u{0024}', code(b"TWD")))),
    (*b"UA", NumberNotation::LatinEu, Some(('\u{20B4}', code(b"UAH")))),
    (*b"UM", NumberNotation::LatinUs, Some(('\u{0024}', code(b"USD")))),
    (*b"US", NumberNotation::LatinUs, Some(('\u{0024}', code(b"USD")))),
    (*b"UY", NumberNotation::LatinEu, Some(('\u{0024}', code(b"UYU")))),
    (*b"VA", NumberNotation::LatinUs, Some(('\u{20AC}', code(b"EUR")))),
    (*b"VC", NumberNotation::LatinUs, Some(('\u{0024}', code(b"XCD")))),
    (*b"VI", NumberNotation::LatinUs, Some(('\u{0024}', code(b"USD")))),
    (*b"VN", NumberNotation::LatinUs, Some(('\u{20AB}', code(b"VND")))),
    (*b"YT", NumberNotation::LatinUs, Some(('\u{20AC}', code(b"EUR")))),
];

impl MoneyExtractor {
    /// Builds a [`MoneyExtractor`] configured for `country_code` as OpenNLP's
    /// `RegionAwareMoneyAnnotator` and `CursorMoneyExtractor.forRegion(Locale)`
    /// do on JDK 21: selects the country's [`NumberNotation`] and overrides its
    /// single-code-point `Sc` currency symbol when one exists (`$` -> `AUD` for
    /// `AU`, `$` -> `CAD` for `CA`, `¥` -> `CNY` for `CN`, `\u{060B}` -> `AFN`
    /// for `AF`, `€` -> `EUR` with `LatinEu` for `DE`), keeping the rest of
    /// [`DEFAULT_SYMBOLS`](super::DEFAULT_SYMBOLS).
    pub fn for_region(country_code: RegionCode) -> MoneyExtractor {
        let key = country_code.bytes();
        match REGION_CONFIGS.binary_search_by_key(&key, |&(k, _, _)| k) {
            Ok(idx) => {
                let (_, notation, override_sym) = REGION_CONFIGS[idx];
                let extractor = MoneyExtractor::new(notation);
                match override_sym {
                    Some((sym, cur)) => extractor.with_symbol(sym, cur).expect("checked JDK 21 currency symbol"),
                    None => extractor,
                }
            }
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

#[cfg(test)]
mod tests {
    use super::*;
    use turbo_analysis::{Analyzer, Document, Ix, Key, Layer, Provenance};
    use turbo_analysis_layers::RegionVote;

    fn with_ballot(text: &str, votes: &[(&str, f64)]) -> Document {
        let mut doc = Document::new(text);
        let mut regions = Regions::new();
        for &(code, share) in votes {
            regions.push(RegionVote::parse(code, share).unwrap());
        }
        doc.add(&Key::DEFAULT, regions, Provenance::by("test")).unwrap();
        doc
    }

    fn run_one(a: &Analyzer, mut doc: Document) -> Document {
        a.run(std::slice::from_mut(&mut doc)).into_iter().next().unwrap().unwrap();
        doc
    }

    #[test]
    fn region_configs_table_is_sorted_and_valid() {
        assert!(REGION_CONFIGS.windows(2).all(|w| w[0].0 < w[1].0));
        for &(bytes, _, _) in &REGION_CONFIGS {
            let rc = RegionCode::new(bytes).unwrap();
            let _ = MoneyExtractor::for_region(rc);
        }
    }

    #[test]
    fn winning_region_resolves_the_symbol() {
        let doc = with_ballot("the deal is worth $5", &[("AU", 1.0)]);
        let a =
            Analyzer::builder().add(RegionAwareMoneyAnnotator::new()).given(Key::<Regions>::DEFAULT).build().unwrap();
        let out = run_one(&a, doc);
        let money = out.get::<Money>().unwrap();
        assert_eq!(money.len(), 1);
        assert_eq!(money.get(Ix::new(0)).currency.as_str(), "AUD");
    }

    #[test]
    fn empty_ballot_falls_back_to_default_table() {
        let doc = with_ballot("costs $5", &[]);
        let a =
            Analyzer::builder().add(RegionAwareMoneyAnnotator::new()).given(Key::<Regions>::DEFAULT).build().unwrap();
        let out = run_one(&a, doc);
        let money = out.get::<Money>().unwrap();
        assert_eq!(money.len(), 1);
        assert_eq!(money.get(Ix::new(0)).currency.as_str(), "USD");
    }

    #[test]
    fn country_with_minority_language_locales_resolves_symbol() {
        let doc = with_ballot("the grant is $5", &[("CA", 1.0)]);
        let a =
            Analyzer::builder().add(RegionAwareMoneyAnnotator::new()).given(Key::<Regions>::DEFAULT).build().unwrap();
        let out = run_one(&a, doc);
        let money = out.get::<Money>().unwrap();
        assert_eq!(money.get(Ix::new(0)).currency.as_str(), "CAD");
    }

    #[test]
    fn locale_scan_skips_unusable_symbol_candidates() {
        let a =
            Analyzer::builder().add(RegionAwareMoneyAnnotator::new()).given(Key::<Regions>::DEFAULT).build().unwrap();
        let af = run_one(&a, with_ballot("price 100\u{060B} today", &[("AF", 1.0)]));
        let af_money = af.get::<Money>().unwrap();
        assert_eq!(af_money.len(), 1);
        assert_eq!(af_money.get(Ix::new(0)).currency.as_str(), "AFN");
        assert_eq!(af_money.amount(Ix::new(0)).to_plain_string(), "100");

        let cn = run_one(&a, with_ballot("price \u{00A5}100 today", &[("CN", 1.0)]));
        let cn_money = cn.get::<Money>().unwrap();
        assert_eq!(cn_money.len(), 1);
        assert_eq!(cn_money.get(Ix::new(0)).currency.as_str(), "CNY");
        assert_eq!(cn_money.amount(Ix::new(0)).to_plain_string(), "100");
    }

    #[test]
    fn letter_symbol_country_keeps_default_table() {
        let doc = with_ballot("the invoice totals $5", &[("CH", 1.0)]);
        let a =
            Analyzer::builder().add(RegionAwareMoneyAnnotator::new()).given(Key::<Regions>::DEFAULT).build().unwrap();
        let out = run_one(&a, doc);
        let money = out.get::<Money>().unwrap();
        assert_eq!(money.get(Ix::new(0)).currency.as_str(), "USD");
    }

    #[test]
    fn unknown_country_falls_back_to_default_table() {
        let doc = with_ballot("costs $5", &[("ZZ", 1.0)]);
        let a =
            Analyzer::builder().add(RegionAwareMoneyAnnotator::new()).given(Key::<Regions>::DEFAULT).build().unwrap();
        let out = run_one(&a, doc);
        let money = out.get::<Money>().unwrap();
        assert_eq!(money.get(Ix::new(0)).currency.as_str(), "USD");
    }

    #[test]
    fn tie_below_margin_falls_back_to_default_table_and_zero_margin_elects_top_row() {
        let tie_doc = || with_ballot("the deal is worth $5", &[("AU", 0.5), ("CA", 0.5)]);

        let a_strict = Analyzer::builder()
            .add(RegionAwareMoneyAnnotator::with_minimum_margin(0.1).unwrap())
            .given(Key::<Regions>::DEFAULT)
            .build()
            .unwrap();
        let out_strict = run_one(&a_strict, tie_doc());
        assert_eq!(out_strict.get::<Money>().unwrap().get(Ix::new(0)).currency.as_str(), "USD");

        let a_zero =
            Analyzer::builder().add(RegionAwareMoneyAnnotator::new()).given(Key::<Regions>::DEFAULT).build().unwrap();
        let out_zero = run_one(&a_zero, tie_doc());
        assert_eq!(out_zero.get::<Money>().unwrap().get(Ix::new(0)).currency.as_str(), "AUD");
    }

    #[test]
    fn margin_outside_unit_interval_is_rejected() {
        assert!(RegionAwareMoneyAnnotator::with_minimum_margin(-0.1).is_err());
        assert!(RegionAwareMoneyAnnotator::with_minimum_margin(1.1).is_err());
        assert!(RegionAwareMoneyAnnotator::with_minimum_margin(f64::NAN).is_err());
    }
}
