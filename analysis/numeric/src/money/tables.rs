//! The tables the money extractor reads: currency symbols, ISO 4217
//! codes and English currency words, as OpenNLP's `CursorMoneyExtractor`
//! defines them.

use turbo_analysis_layers::Currency;
use turbo_unicode::java;

/// A currency code from a literal the tables know to be well formed.
const fn code(c: &[u8; 3]) -> Currency {
    match Currency::new(*c) {
        Some(c) => c,
        None => panic!("a code of three ASCII capital letters"),
    }
}

const USD: Currency = code(b"USD");
const EUR: Currency = code(b"EUR");
const GBP: Currency = code(b"GBP");
const JPY: Currency = code(b"JPY");
const INR: Currency = code(b"INR");
const RUB: Currency = code(b"RUB");
const ILS: Currency = code(b"ILS");
const THB: Currency = code(b"THB");
const CNY: Currency = code(b"CNY");

/// Dollar-prefixed region symbols (`HK$`, `US$`, `CA$`, `C$`, `A$`, `AU$`,
/// `NZ$`, `R$`, `S$`, `SG$`, `NT$`, `MX$`, `TW$`, `Z$`) and the currency
/// each denotes by convention when `$` is in the symbol table.
pub const DOLLAR_PREFIXES: [(&[u8], Currency); 14] = [
    (b"A$", code(b"AUD")),
    (b"AU$", code(b"AUD")),
    (b"C$", code(b"CAD")),
    (b"CA$", code(b"CAD")),
    (b"HK$", code(b"HKD")),
    (b"MX$", code(b"MXN")),
    (b"NT$", code(b"TWD")),
    (b"NZ$", code(b"NZD")),
    (b"R$", code(b"BRL")),
    (b"S$", code(b"SGD")),
    (b"SG$", code(b"SGD")),
    (b"TW$", code(b"TWD")),
    (b"US$", USD),
    (b"Z$", code(b"ZWL")),
];

/// Returns the currency denoted by a dollar-prefixed region symbol (`HK$`,
/// `US$`, `CA$`, `R$`, `A$`) starting at `prefix` (`1..=2` ASCII letters
/// followed by `'$'`).
pub(crate) fn dollar_prefix(prefix: &[u8]) -> Option<Currency> {
    DOLLAR_PREFIXES.binary_search_by_key(&prefix, |&(p, _)| p).ok().map(|i| DOLLAR_PREFIXES[i].1)
}

/// The default symbol table: each symbol mapped to the currency it most
/// commonly denotes. A caller in another convention passes its own table
/// to [`MoneyExtractor::with_symbols`](super::MoneyExtractor::with_symbols).
pub const DEFAULT_SYMBOLS: [(char, Currency); 11] = [
    ('$', USD),
    ('\u{20AC}', EUR),          // euro sign
    ('\u{00A3}', GBP),          // pound sign
    ('\u{00A5}', JPY),          // yen sign
    ('\u{20B9}', INR),          // rupee sign
    ('\u{20A9}', code(b"KRW")), // won sign
    ('\u{20BD}', RUB),          // ruble sign
    ('\u{20BA}', code(b"TRY")), // lira sign
    ('\u{20AA}', ILS),          // sheqel sign
    ('\u{0E3F}', THB),          // baht sign
    ('\u{20AB}', code(b"VND")), // dong sign
];

/// English currency words and the code each denotes by convention.
/// `peso`, `franc`, `krone` and `won` are left out, as are subunits such
/// as `cents`.
const CURRENCY_WORDS: [(&[u8], Currency); 19] = [
    (b"dollar", USD),
    (b"dollars", USD),
    (b"euro", EUR),
    (b"euros", EUR),
    (b"pound", GBP),
    (b"pounds", GBP),
    (b"sterling", GBP),
    (b"yen", JPY),
    (b"rupee", INR),
    (b"rupees", INR),
    (b"yuan", CNY),
    (b"renminbi", CNY),
    (b"ruble", RUB),
    (b"rubles", RUB),
    (b"rouble", RUB),
    (b"roubles", RUB),
    (b"shekel", ILS),
    (b"shekels", ILS),
    (b"baht", THB),
];

/// The longest currency word, `sterling` and `renminbi`, in UTF-16 units.
pub(crate) const MAX_CURRENCY_WORD_UNITS: usize = 8;

/// The currency a word names. The Java lowercases the word code point by
/// code point with `Character.toLowerCase` before the table lookup, so a
/// letter whose simple lowercase mapping is an ASCII letter matches it:
/// the Kelvin sign U+212A reads as `k`, and U+0130 as `i`.
pub(crate) fn currency_word(word: &str) -> Option<Currency> {
    let mut lower = [0u8; MAX_CURRENCY_WORD_UNITS];
    let mut n = 0;
    for c in word.chars() {
        let l = java::to_lower_char(c);
        // Every word in the table is ASCII, so another letter matches none.
        if !l.is_ascii() || n == lower.len() {
            return None;
        }
        lower[n] = l as u8;
        n += 1;
    }
    CURRENCY_WORDS.iter().find(|(w, _)| *w == &lower[..n]).map(|&(_, c)| c)
}

/// Whether `code` is a currency the JDK knows: the alphabetic codes of
/// `java.util.Currency.getAvailableCurrencies()` under JDK 21.0.12, which
/// include withdrawn codes such as `DEM` and `EEK` and the testing and
/// no-currency codes `XTS` and `XXX`. OpenNLP takes the set from the
/// running JDK, so another JDK can know a few codes more or fewer.
pub fn is_iso_currency(code: Currency) -> bool {
    ISO_CODES.binary_search(&code.code()).is_ok()
}

/// Whether `code` is an ISO 4217 testing (`XTS`) or no-currency (`XXX`)
/// code. By default [`MoneyExtractor`](super::MoneyExtractor) refuses these
/// codes as money mentions; [`MoneyExtractor::with_opennlp_parity`](super::MoneyExtractor::with_opennlp_parity)
/// accepts them to match OpenNLP.
pub fn is_test_currency(code: Currency) -> bool {
    matches!(&code.code(), b"XTS" | b"XXX")
}

/// Returns the ISO 4217 default fraction digits (minor-unit decimal places)
/// for `code` as `java.util.Currency.getDefaultFractionDigits()` reports it
/// on JDK 21 (`0` for `JPY`/`KRW`/`VND`, `2` for `USD`/`EUR`/`GBP`, `3` for
/// `BHD`/`JOD`/`KWD`, `4` for `CLF`), or `None` when the code has no minor
/// unit (`XAU`, `XDR`, `XTS`, `XXX`) or is not in the ISO table.
pub fn minor_units(code: Currency) -> Option<u8> {
    if !is_iso_currency(code) {
        return None;
    }
    let bytes = code.code();
    if NO_MINOR_UNIT_CODES.binary_search(&bytes).is_ok() {
        return None;
    }
    if ZERO_MINOR_UNIT_CODES.binary_search(&bytes).is_ok() {
        return Some(0);
    }
    if THREE_MINOR_UNIT_CODES.binary_search(&bytes).is_ok() {
        return Some(3);
    }
    if bytes == *b"CLF" {
        return Some(4);
    }
    Some(2)
}

/// Codes where `Currency.getDefaultFractionDigits()` returns `-1` (precious
/// metals, composite units, testing, and no-currency codes), sorted.
const NO_MINOR_UNIT_CODES: [[u8; 3]; 15] = [
    *b"XAG", *b"XAU", *b"XBA", *b"XBB", *b"XBC", *b"XBD", *b"XDR", *b"XFO", *b"XFU", *b"XPD", *b"XPT", *b"XSU",
    *b"XTS", *b"XUA", *b"XXX",
];

/// Codes where `Currency.getDefaultFractionDigits()` returns `0`, sorted.
const ZERO_MINOR_UNIT_CODES: [[u8; 3]; 30] = [
    *b"ADP", *b"BEF", *b"BIF", *b"BYB", *b"BYR", *b"CLP", *b"DJF", *b"ESP", *b"GNF", *b"GRD", *b"ISK", *b"ITL",
    *b"JPY", *b"KMF", *b"KRW", *b"LUF", *b"MGF", *b"PTE", *b"PYG", *b"ROL", *b"RWF", *b"TPE", *b"TRL", *b"UGX",
    *b"UYI", *b"VND", *b"VUV", *b"XAF", *b"XOF", *b"XPF",
];

/// Codes where `Currency.getDefaultFractionDigits()` returns `3`, sorted.
const THREE_MINOR_UNIT_CODES: [[u8; 3]; 7] = [*b"BHD", *b"IQD", *b"JOD", *b"KWD", *b"LYD", *b"OMR", *b"TND"];

/// The JDK's currency codes, sorted.
const ISO_CODES: [[u8; 3]; 233] = [
    *b"ADP", *b"AED", *b"AFA", *b"AFN", *b"ALL", *b"AMD", *b"ANG", *b"AOA", *b"ARS", *b"ATS", *b"AUD", *b"AWG",
    *b"AYM", *b"AZM", *b"AZN", *b"BAM", *b"BBD", *b"BDT", *b"BEF", *b"BGL", *b"BGN", *b"BHD", *b"BIF", *b"BMD",
    *b"BND", *b"BOB", *b"BOV", *b"BRL", *b"BSD", *b"BTN", *b"BWP", *b"BYB", *b"BYN", *b"BYR", *b"BZD", *b"CAD",
    *b"CDF", *b"CHE", *b"CHF", *b"CHW", *b"CLF", *b"CLP", *b"CNY", *b"COP", *b"COU", *b"CRC", *b"CSD", *b"CUC",
    *b"CUP", *b"CVE", *b"CYP", *b"CZK", *b"DEM", *b"DJF", *b"DKK", *b"DOP", *b"DZD", *b"EEK", *b"EGP", *b"ERN",
    *b"ESP", *b"ETB", *b"EUR", *b"FIM", *b"FJD", *b"FKP", *b"FRF", *b"GBP", *b"GEL", *b"GHC", *b"GHS", *b"GIP",
    *b"GMD", *b"GNF", *b"GRD", *b"GTQ", *b"GWP", *b"GYD", *b"HKD", *b"HNL", *b"HRK", *b"HTG", *b"HUF", *b"IDR",
    *b"IEP", *b"ILS", *b"INR", *b"IQD", *b"IRR", *b"ISK", *b"ITL", *b"JMD", *b"JOD", *b"JPY", *b"KES", *b"KGS",
    *b"KHR", *b"KMF", *b"KPW", *b"KRW", *b"KWD", *b"KYD", *b"KZT", *b"LAK", *b"LBP", *b"LKR", *b"LRD", *b"LSL",
    *b"LTL", *b"LUF", *b"LVL", *b"LYD", *b"MAD", *b"MDL", *b"MGA", *b"MGF", *b"MKD", *b"MMK", *b"MNT", *b"MOP",
    *b"MRO", *b"MRU", *b"MTL", *b"MUR", *b"MVR", *b"MWK", *b"MXN", *b"MXV", *b"MYR", *b"MZM", *b"MZN", *b"NAD",
    *b"NGN", *b"NIO", *b"NLG", *b"NOK", *b"NPR", *b"NZD", *b"OMR", *b"PAB", *b"PEN", *b"PGK", *b"PHP", *b"PKR",
    *b"PLN", *b"PTE", *b"PYG", *b"QAR", *b"ROL", *b"RON", *b"RSD", *b"RUB", *b"RUR", *b"RWF", *b"SAR", *b"SBD",
    *b"SCR", *b"SDD", *b"SDG", *b"SEK", *b"SGD", *b"SHP", *b"SIT", *b"SKK", *b"SLE", *b"SLL", *b"SOS", *b"SRD",
    *b"SRG", *b"SSP", *b"STD", *b"STN", *b"SVC", *b"SYP", *b"SZL", *b"THB", *b"TJS", *b"TMM", *b"TMT", *b"TND",
    *b"TOP", *b"TPE", *b"TRL", *b"TRY", *b"TTD", *b"TWD", *b"TZS", *b"UAH", *b"UGX", *b"USD", *b"USN", *b"USS",
    *b"UYI", *b"UYU", *b"UZS", *b"VEB", *b"VED", *b"VEF", *b"VES", *b"VND", *b"VUV", *b"WST", *b"XAD", *b"XAF",
    *b"XAG", *b"XAU", *b"XBA", *b"XBB", *b"XBC", *b"XBD", *b"XCD", *b"XCG", *b"XDR", *b"XFO", *b"XFU", *b"XOF",
    *b"XPD", *b"XPF", *b"XPT", *b"XSU", *b"XTS", *b"XUA", *b"XXX", *b"YER", *b"YUM", *b"ZAR", *b"ZMK", *b"ZMW",
    *b"ZWD", *b"ZWG", *b"ZWL", *b"ZWN", *b"ZWR",
];

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_iso_table_is_sorted_and_well_formed() {
        assert!(ISO_CODES.windows(2).all(|w| w[0] < w[1]));
        assert!(ISO_CODES.iter().all(|c| Currency::new(*c).is_some()));
        for (_, c) in DEFAULT_SYMBOLS {
            assert!(is_iso_currency(c), "{c}");
        }
        for (_, c) in CURRENCY_WORDS {
            assert!(is_iso_currency(c), "{c}");
        }
        assert!(is_iso_currency(code(b"CHF")));
        assert!(is_iso_currency(code(b"EEK")));
        assert!(!is_iso_currency(code(b"USX")));
        assert!(!is_iso_currency(code(b"AAA")));
        assert!(DOLLAR_PREFIXES.windows(2).all(|w| w[0].0 < w[1].0));
        assert!(NO_MINOR_UNIT_CODES.windows(2).all(|w| w[0] < w[1]));
        assert!(ZERO_MINOR_UNIT_CODES.windows(2).all(|w| w[0] < w[1]));
        assert!(THREE_MINOR_UNIT_CODES.windows(2).all(|w| w[0] < w[1]));
        assert!(is_test_currency(code(b"XTS")));
        assert!(is_test_currency(code(b"XXX")));
        assert!(!is_test_currency(USD));
        assert_eq!(minor_units(USD), Some(2));
        assert_eq!(minor_units(JPY), Some(0));
        assert_eq!(minor_units(code(b"KWD")), Some(3));
        assert_eq!(minor_units(code(b"CLF")), Some(4));
        assert_eq!(minor_units(code(b"XAU")), None);
        assert_eq!(minor_units(code(b"XXX")), None);
        assert_eq!(minor_units(code(b"USX")), None);
    }

    #[test]
    fn currency_words_lowercase_by_code_point() {
        assert_eq!(currency_word("dollars"), Some(USD));
        assert_eq!(currency_word("DoLLaR"), Some(USD));
        assert_eq!(currency_word("STERLING"), Some(GBP));
        assert_eq!(currency_word("cents"), None);
        assert_eq!(currency_word("won"), None);
        assert_eq!(currency_word("dollarsx"), None);
        assert_eq!(currency_word(""), None);
        // U+0130 lowercases to `i` with Character.toLowerCase.
        assert_eq!(currency_word("rupee\u{0130}"), None);
        assert_eq!(currency_word("sh\u{0130}kel"), None);
        assert_eq!(currency_word("\u{0130}"), None);
        assert_eq!(currency_word("RENM\u{0130}NB\u{0130}"), Some(CNY));
        assert_eq!(currency_word("BAHT"), Some(THB));
        assert_eq!(currency_word("y\u{00C9}n"), None);
    }
}
