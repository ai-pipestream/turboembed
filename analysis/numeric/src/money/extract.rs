//! Finds the money mentions in a text: a port of OpenNLP's
//! `CursorMoneyExtractor` (`opennlp.tools.money`, numeric add-on).
//!
//! The Java steps over a `CharSequence` in UTF-16 units; this steps over
//! the same characters in UTF-8 bytes. Every character a shape compares
//! against is in the Basic Multilingual Plane except, possibly, a symbol
//! in a caller's table, which both read as one code point, so both find
//! the same mentions (see scan.rs for the number itself).

use std::fmt;

use turbo_analysis::{Annotator, Error, Refs, Span, View};
use turbo_analysis_layers::{Currency, Money, MoneyMention, NumberNotation};
use turbo_unicode::GeneralCategory;
use turbo_unicode::category::general_category;

use super::tables::{DEFAULT_SYMBOLS, MAX_CURRENCY_WORD_UNITS, currency_word, is_iso_currency};
use crate::scan::{
    EXTENDED_START_BYTES, ScanResult, base_char_before, boundary_after, boundary_before, byte_at, char_at,
    continues_number_with_parity, is_nd_lead_byte, parse_with_parity, sign_boundary_before, whitespace_at, word_end,
};

/// Writes `turbo:money` over a text view: every amount written with a
/// currency symbol before or after it (`$1,234.56`, `50€`), an ISO 4217
/// code before or after it (`USD 100`, `100 USD`) or an English currency
/// word after it (`1.2 million dollars`), with an optional leading minus
/// and an optional scale marker (`$1.2M`, `$3 billion`). A bare number is
/// never money.
///
/// Amounts are read in one [`NumberNotation`], `LatinUs` by default. A
/// symbol maps to one currency, the one it most commonly denotes (`$` is
/// `USD`) unless the caller gives its own table.
///
/// OpenNLP's `forRegion(Locale)` derives a table and a notation from the
/// JDK's locale data; that lookup is not here. A caller that knows the
/// region builds the same extractor from [`MoneyExtractor::with_symbol`]
/// and [`MoneyExtractor::new`]: for Australia, the default table with `$`
/// as `AUD`, in `LatinUs`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MoneyExtractor {
    /// The symbol table, sorted by symbol.
    symbols: Box<[(char, Currency)]>,
    /// Whether a byte can start a mention: a minus, a digit, an ASCII
    /// capital, or the first byte of a symbol in the table.
    starts: [bool; 256],
    notation: NumberNotation,
    opennlp_parity: bool,
}

/// Why a symbol table was refused.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum SymbolError {
    /// The table has no symbol.
    Empty,
    /// The character is not in the Unicode category Sc, currency symbol.
    NotACurrencySymbol(char),
    /// The code is not one the JDK's currency table holds.
    UnknownCurrency(Currency),
}

impl fmt::Display for SymbolError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            SymbolError::Empty => f.write_str("a symbol table must not be empty"),
            SymbolError::NotACurrencySymbol(c) => write!(f, "not a currency symbol code point: U+{:04X}", *c as u32),
            SymbolError::UnknownCurrency(code) => write!(f, "not an ISO 4217 currency code: {code}"),
        }
    }
}

impl std::error::Error for SymbolError {}

impl Default for MoneyExtractor {
    fn default() -> MoneyExtractor {
        MoneyExtractor::new(NumberNotation::LatinUs)
    }
}

impl MoneyExtractor {
    /// The default symbol table, reading amounts in `notation`.
    pub fn new(notation: NumberNotation) -> MoneyExtractor {
        MoneyExtractor::with_symbols(DEFAULT_SYMBOLS, notation).expect("the default table is valid")
    }

    /// A custom symbol table, reading amounts in `notation`. Each symbol
    /// must be a currency symbol (category Sc) and each code one the JDK
    /// knows ([`is_iso_currency`]). When a symbol is given twice, the later
    /// entry wins, as a later `put` does in the Java map.
    pub fn with_symbols(
        symbols: impl IntoIterator<Item = (char, Currency)>,
        notation: NumberNotation,
    ) -> Result<MoneyExtractor, SymbolError> {
        let mut table: Vec<(char, Currency)> = Vec::new();
        for (symbol, currency) in symbols {
            if general_category(symbol) != GeneralCategory::Sc {
                return Err(SymbolError::NotACurrencySymbol(symbol));
            }
            if !is_iso_currency(currency) {
                return Err(SymbolError::UnknownCurrency(currency));
            }
            match table.iter_mut().find(|(s, _)| *s == symbol) {
                Some(entry) => entry.1 = currency,
                None => table.push((symbol, currency)),
            }
        }
        if table.is_empty() {
            return Err(SymbolError::Empty);
        }
        table.sort_unstable_by_key(|&(s, _)| s);
        let mut starts = EXTENDED_START_BYTES;
        for b in b'A'..=b'Z' {
            starts[usize::from(b)] = true;
        }
        for &(s, _) in &table {
            let mut buf = [0u8; 4];
            starts[usize::from(s.encode_utf8(&mut buf).as_bytes()[0])] = true;
        }
        Ok(MoneyExtractor { symbols: table.into_boxed_slice(), starts, notation, opennlp_parity: false })
    }

    /// The same extractor with `symbol` mapped to `currency`, the other
    /// entries kept: what `forRegion` does for a region whose own currency
    /// is written with one currency sign.
    pub fn with_symbol(self, symbol: char, currency: Currency) -> Result<MoneyExtractor, SymbolError> {
        let parity = self.opennlp_parity;
        let entries = self.symbols.iter().copied().chain([(symbol, currency)]);
        Ok(MoneyExtractor::with_symbols(entries, self.notation)?.with_opennlp_parity(parity))
    }

    /// Configures whether to enforce strict OpenNLP JVM parity (rejecting
    /// Indian grouping, space/apostrophe grouping, exponents, and non-ASCII
    /// digits).
    pub fn with_opennlp_parity(mut self, opennlp_parity: bool) -> MoneyExtractor {
        self.opennlp_parity = opennlp_parity;
        self
    }

    /// The symbol table, sorted by symbol.
    pub fn symbols(&self) -> &[(char, Currency)] {
        &self.symbols
    }

    pub fn notation(&self) -> NumberNotation {
        self.notation
    }

    /// Appends every money mention in `text` to `out`, in text order, with
    /// byte offsets into `text`. After a mention the scan goes on at its
    /// end, so mentions never overlap; after a failed candidate, at the
    /// next character.
    ///
    /// # Panics
    ///
    /// When `text` is longer than `i32::MAX` bytes: offsets are `u32`, and
    /// a scale is a Java `int`, which a text of that length can overflow.
    pub fn scan(&self, text: &str, out: &mut Money) {
        assert!(i32::try_from(text.len()).is_ok(), "a text of at most i32::MAX bytes");
        let b = text.as_bytes();
        let mut digits = String::new();
        let mut i = 0;
        while i < b.len() {
            match b[i..].iter().position(|&x| self.starts[usize::from(x)]) {
                Some(skip) => i += skip,
                None => break,
            }
            match self.match_at(text, i, &mut digits) {
                Some(found) => {
                    out.push(MoneyMention {
                        span: Span::new(i as u32, found.end as u32),
                        negative: found.negative && digits != "0",
                        coefficient: &digits,
                        scale: found.scale,
                        currency: found.currency,
                    });
                    i = found.end;
                }
                None => i += char_at(text, i).map_or(1, char::len_utf8),
            }
        }
    }

    /// `matchAt`: tries the three shapes at byte `start`, symbol first,
    /// then ISO code first, then number first.
    fn match_at(&self, text: &str, start: usize, digits: &mut String) -> Option<Found> {
        let negative = text.as_bytes()[start] == b'-';
        let i = start + usize::from(negative);
        let c = char_at(text, i);
        if let Some((symbol, currency)) = c.and_then(|c| self.symbol(c).map(|cur| (c, cur)))
            && !ascii_letter_before(text, i)
            && (!negative || sign_boundary_before(text, start))
        {
            return self.symbol_first(text, i + symbol.len_utf8(), currency, negative, digits);
        }
        let b = text.as_bytes();
        if !negative
            && b[i].is_ascii_uppercase()
            && boundary_before(text, i)
            && let Some(found) = self.iso_first(text, i, digits)
        {
            return Some(found);
        }
        let boundary = if negative { sign_boundary_before(text, start) } else { boundary_before(text, i) };
        let is_digit = byte_at(b, i).is_ascii_digit()
            || (!self.opennlp_parity
                && is_nd_lead_byte(byte_at(b, i))
                && char_at(text, i).is_some_and(turbo_unicode::java::is_digit));
        if is_digit && !continues_number_with_parity(text, i, self.notation, self.opennlp_parity) && boundary {
            return self.number_first(text, i, negative, digits);
        }
        None
    }

    /// `$1,234.56`, `-$5`, `$1.2M` and `$ 100`: one whitespace character
    /// may stand between the symbol at `after_symbol` and the number.
    fn symbol_first(
        &self,
        text: &str,
        after_symbol: usize,
        currency: Currency,
        negative: bool,
        digits: &mut String,
    ) -> Option<Found> {
        let number_start = skip_whitespace(text, after_symbol);
        let r = parse_with_parity(text, number_start, true, self.notation, self.opennlp_parity, digits)?;
        mention(text, r.end, &r, currency, negative)
    }

    /// `USD 100` and `USD 1.2 million`: a code at `code_start`, one
    /// whitespace character, then the number. No minus is taken.
    fn iso_first(&self, text: &str, code_start: usize, digits: &mut String) -> Option<Found> {
        let code = iso_code_at(text, code_start)?;
        if !whitespace_at(text, code_start + 3) {
            return None;
        }
        let r = parse_with_parity(
            text,
            skip_whitespace(text, code_start + 3),
            true,
            self.notation,
            self.opennlp_parity,
            digits,
        )?;
        mention(text, r.end, &r, code, false)
    }

    /// `100 USD`, `50€`, `3.5m USD` and `1.2 million dollars`: the number
    /// at `digit_start`, then a symbol directly, or one whitespace
    /// character and a code or a currency word.
    fn number_first(&self, text: &str, digit_start: usize, negative: bool, digits: &mut String) -> Option<Found> {
        let r = parse_with_parity(text, digit_start, true, self.notation, self.opennlp_parity, digits)?;
        let next = char_at(text, r.end)?;
        if let Some(currency) = self.symbol(next) {
            return mention(text, r.end + next.len_utf8(), &r, currency, negative);
        }
        if !whitespace_at(text, r.end) {
            return None;
        }
        let word_start = r.end + next.len_utf8();
        if let Some(code) = iso_code_at(text, word_start) {
            return mention(text, word_start + 3, &r, code, negative);
        }
        let (currency, end) = currency_word_at(text, word_start)?;
        mention(text, end, &r, currency, negative)
    }

    /// The currency a symbol denotes in this table.
    fn symbol(&self, c: char) -> Option<Currency> {
        self.symbols.binary_search_by_key(&c, |&(s, _)| s).ok().map(|i| self.symbols[i].1)
    }
}

/// A mention found by `matchAt`. Its digits are in the scan's buffer.
struct Found {
    end: usize,
    scale: i32,
    currency: Currency,
    negative: bool,
}

/// `mention`: a mention ending at byte `end`, when a word boundary follows.
fn mention(text: &str, end: usize, r: &ScanResult, currency: Currency, negative: bool) -> Option<Found> {
    if !boundary_after(text, end) {
        return None;
    }
    let scale = i32::try_from(r.scale).expect("a text of at most i32::MAX bytes has a scale that fits");
    Some(Found { end, scale, currency, negative })
}

/// The byte after one whitespace character at `i`, or `i` when there is
/// none there. Past the end of the text the Java reads a space and steps
/// one unit; the number scan after it then finds nothing.
fn skip_whitespace(text: &str, i: usize) -> usize {
    if !whitespace_at(text, i) {
        return i;
    }
    i + char_at(text, i).map_or(1, char::len_utf8)
}

/// `isoCodeAt`: three ASCII capitals at byte `start`, between word
/// boundaries, that make a code the JDK knows.
fn iso_code_at(text: &str, start: usize) -> Option<Currency> {
    let code: [u8; 3] = text.as_bytes().get(start..start + 3)?.try_into().ok()?;
    if !boundary_before(text, start) {
        return None;
    }
    let code = Currency::new(code)?;
    (boundary_after(text, start + 3) && is_iso_currency(code)).then_some(code)
}

/// `currencyWordAt`: the currency a word at byte `start` names, and where
/// the word ends. A hyphen right after the word makes it the first part of
/// a compound (`euro-cent`), which is no amount.
fn currency_word_at(text: &str, start: usize) -> Option<(Currency, usize)> {
    let end = word_end(text, start, MAX_CURRENCY_WORD_UNITS)?;
    if byte_at(text.as_bytes(), end) == b'-' {
        return None;
    }
    Some((currency_word(&text[start..end])?, end))
}

/// `asciiLetterBefore`: whether the character before byte `index`,
/// combining marks skipped, is an ASCII letter, so a symbol there is the
/// tail of a longer one (`HK$`, `US$`).
fn ascii_letter_before(text: &str, index: usize) -> bool {
    base_char_before(text, index).is_some_and(|c| c.is_ascii_alphabetic())
}

impl Annotator for MoneyExtractor {
    const ID: &'static str = "turbo-numeric:money";
    type Reads = (View,);
    type Writes = (Money,);

    fn annotate(&self, (view,): Refs<'_, Self>) -> Result<(Money,), Error> {
        let text = view.text();
        if i32::try_from(text.len()).is_err() {
            return Err(Error::Annotator {
                annotator: Self::ID,
                detail: format!("a text of {} bytes is past the i32 scale range", text.len()),
            });
        }
        let mut money = Money::new(view.id());
        self.scan(text, &mut money);
        Ok((money,))
    }

    /// FNV-1a over the notation and the symbol table, fixed so that stored
    /// provenance stays comparable across builds.
    fn config_hash(&self) -> u64 {
        let notation: u64 = match self.notation {
            NumberNotation::LatinUs => 1,
            NumberNotation::LatinEu => 2,
        };
        let symbols = self.symbols.iter().flat_map(|&(s, c)| {
            let [a, b, d] = c.code();
            [u64::from(s), u64::from(a), u64::from(b), u64::from(d)]
        });
        [notation, u64::from(self.opennlp_parity)]
            .into_iter()
            .chain(symbols)
            .fold(0xcbf2_9ce4_8422_2325, |h, x| (h ^ x).wrapping_mul(0x0100_0000_01b3))
    }
}
