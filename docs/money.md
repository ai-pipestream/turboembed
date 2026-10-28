# Money

The `money` module of `turbo-numeric` (`analysis/numeric/src/money/`)
and the shared `region` module (`analysis/numeric/src/region/`)
find amounts of money in a text view, write them as the `turbo:money`
layer, and convert them into one currency with the European Central
Bank's euro reference rates:

| Type | Role |
|---|---|
| `MoneyExtractor` | Extracts currency amounts into `Money` (`turbo:money`) |
| `MoneyExtractor::for_region` | Configures regional number notation and local currency symbols |
| `RegionAwareMoneyAnnotator` | Selects regional notation and symbols from `turbo:regions` |
| `EcbRates`, `FxRates` | Loads and queries ECB euro reference rates |
| `MoneyConverter`, `DatedMoneyConverter` | Converts `Money` into `CONVERTED_MONEY` at a fixed date or `DocumentDate` |
| `Decimal`, `Date` | Exact base-10 decimal and calendar date types |

The number inside a mention is read by the number scan of
[numbers.md](numbers.md).

## The layer

`Money`, in `turbo-analysis-layers`, has one item per mention:

| Column | What it holds |
|---|---|
| `spans` | The whole mention: a minus sign, the symbol or code, the number, its scale marker and a currency word after it |
| `negative` | Whether the amount is below zero; never true for zero |
| `coefficient` | The amount's digits without sign, decimal point or leading zeros, `0` for zero |
| `scale` | The decimal scale of the amount, an `i32` |
| `currency` | A `Currency`: three ASCII capital letters, stored as three bytes |

`coefficient` and `scale` store the unscaled integer value and decimal
scale: `$1.2M` has the coefficient `12000000` and the scale 1.
`Money::amount` returns a `DecimalRef`, which formats the value as a
plain decimal string. An extracted amount has a scale of zero or more.
A converted amount can have a scale below zero, since a decimal rounded
to 16 digits keeps its trailing zeros as a negative scale
(`1085600000000000` at the scale -5 is 108560000000000000000).

The extractor writes the default instance. `MoneyConverter` writes a
second instance of the same type under the key `CONVERTED_MONEY` (the
name `converted`). The converted layer has the spans of the mentions it
could convert and leaves the others out.

## Extraction

Offsets are UTF-8 byte offsets into the view, on character boundaries.
At each character the scan tries three shapes, in this order, and after
a mention goes on at its end:

- **Symbol first** (`$1,234.56`, `-$5`, `$1.2M`, `$ 100`, `HK$50`, `US$50`):
  a symbol from the table, or a dollar-prefixed region symbol (`HK$`, `US$`,
  `CA$`, `C$`, `A$`, `AU$`, `NZ$`, `R$`, `S$`, `SG$`, `NT$`, `MX$`, `TW$`,
  `Z$`) when `$` is in the symbol table. A single-character symbol is rejected
  when preceded by a letter (combining marks skipped). One `White_Space`
  character may follow the symbol. A minus before the symbol counts only
  where a minus may open a negative number (numbers.md), so the hyphen in
  `$100-$200` is not a sign.
- **Code first** (`USD 100`, `USD 1.2 million`, `-USD 5`): three ASCII capitals
  between word boundaries that make a known non-test code, one `White_Space`
  character, then the number, with an optional leading minus at a sign
  boundary.
- **Number first** (`100 USD`, `50€`, `1.2 million dollars`): a
  number, then a symbol directly after it, or one `White_Space` character
  and a code or a currency word. A digit that goes on with a number
  before it never starts one, so `1,00,000 USD` gives nothing rather than
  `000 USD`. Unscaled `pound` and `pounds` in a weight context (`3 pounds of
  flour`, `weighs 150 pounds`) are not read as GBP, while `3 pounds sterling`
  includes `sterling` in the mention span.

Every shape must end at a word boundary. The number follows the scan
rules of numbers.md with scale markers read, in the extractor's
`NumberNotation` (`LatinUs` unless the caller chooses `LatinEu`).

The default symbol table maps `$` to USD, `€` to EUR, `£` to GBP, `¥` to
JPY, `₹` to INR, `₩` to KRW, `₽` to RUB, `₺` to TRY, `₪` to ILS, `฿` to
THB and `₫` to VND. `MoneyExtractor::with_symbols` takes another table,
whose symbols must be in the Unicode category Sc and whose codes must be
known. The currency words are `dollar(s)` (USD), `euro(s)` (EUR),
`pound(s)` and `sterling` (GBP), `yen` (JPY), `rupee(s)` (INR), `yuan`
and `renminbi` (CNY), `ruble(s)` and `rouble(s)` (RUB), `shekel(s)` (ILS)
and `baht` (THB), in any case. A word followed by a hyphen is the start
of a compound (`euro-cent`) and is not a currency.

The known codes are 233 three-letter ISO 4217 currency codes compiled into
the crate.

`MoneyExtractor::for_region(RegionCode)` selects the region's number
notation (`LatinEu` when the region's decimal separator is `,`, otherwise
`LatinUs`) and overrides the region's local single-character `Sc` currency
symbol (`$` -> `AUD` for `AU`, `$` -> `CAD` for `CA`, `\u{07FF}` -> `GNF`
for `GN`) while preserving the default symbol mappings for all other
symbols. `RegionAwareMoneyAnnotator`
(`turbo-numeric:region-aware-money`) reads `(View, Regions)` and writes
`(Money,)`, delegating to `MoneyExtractor::for_region` when
`Regions::winner(min_margin)` elects a winning region and falling back to
`MoneyExtractor::default()` when the ballot is empty or tied within
`min_margin`.

## Rates

`EcbRates` loads the ECB's history CSV from a path or from bytes the
caller supplies; nothing is downloaded:

- The bytes must be UTF-8. Lines end at `\n`, `\r` or `\r\n`, and blank
  lines (whitespace only) are skipped.
- The header starts with `Date,` and names unique codes of three ASCII
  capitals. Fields are split at every comma and trimmed. A control
  character below U+0020 that is not whitespace refuses the file, so a
  NUL cannot be trimmed away. A header or row may end with one more,
  empty field.
- Each row has a strict ISO date (`YYYY-MM-DD`) that no other row has,
  and a field per currency. An empty field or `N/A` means no quote; a
  quote is parsed as a decimal (including an exponent and any decimal
  digits of the Basic Multilingual Plane) and must be above zero. An
  `EUR` quote must equal 1.

A lookup takes the latest row on or before the date asked for, at most
`MAX_STALENESS_DAYS` (7) days earlier, inclusive, so a weekend or holiday
uses the last reference day. Both currencies must be quoted in that
row; an older row is not searched for a missing one. The euro is always
1 on a usable row.

## Arithmetic

The arithmetic is exact and written by hand: unbounded naturals in base
10⁹ and a division rounded to 16 significant digits, half to even
(IEEE 754 decimal64 precision):

- A rate is the target quote divided by the source quote; the same
  currency gives exactly 1.
- A converted amount is the amount times the target quote, divided by
  the source quote, rounded once. An amount already in the target
  currency is kept as it is, digits and scale, but still needs a quote.
- A quotient's scale is the preferred scale (dividend scale minus
  divisor scale) when the exact quotient fits 16 digits there, and the
  16-digit scale otherwise. By default no rounding to a
  currency's minor unit follows (1 USD in EUR at 1.2 is
  `0.8333333333333333`); `MoneyConverter::with_minor_unit_rounding(true)`
  and `DatedMoneyConverter::with_minor_unit_rounding(true)` round converted
  amounts half to even to the target currency's ISO 4217 minor unit
  (`minor_units`).
- When an intermediate scale leaves the `i32` range, the converter
  divides the unscaled values instead and applies the combined scale,
  returning `ScaleRangeError` only when the result itself does not fit.
  A zero amount converts to zero at any scale.

`FxRates` is the trait `MoneyConverter` reads rates through.
`EcbRates` implements it; another provider implements `rate` and
inherits `convert`, which multiplies the amount by the rate.

`MoneyConverter` converts the default money instance into one target
currency as of one date and is added to a pipeline with
`MoneyConverter::bound`, which names its output instance.

`DatedMoneyConverter` converts as of the date in the `turbo:document-date`
layer, which `DocumentDateSelector` writes ([temporal.md](temporal.md)),
and is added with `DatedMoneyConverter::bound` after that step. A pipeline
without a step that writes the date layer is refused when it is built. A
document without a date gets an empty converted layer and the rates are
not asked.

## Extraction rules and edge cases

Default extraction rules:

- A minus before a code is kept (`-USD 5` is minus 5 USD with span `0..6`).
- Dollar-prefixed region symbols (`HK$50`, `US$50`, `CA$50`, `C$50`, `A$50`,
  `AU$50`, `NZ$50`, `R$50`, `S$50`, `SG$50`, `NT$50`, `MX$50`, `TW$50`,
  `Z$50`) resolve to their currencies when `$` is in the symbol table, and
  any Unicode letter before a single-character symbol (`Ü$5`) rejects the
  symbol.
- The ISO 4217 testing and no-currency codes `XTS` and `XXX` (`is_test_currency`)
  are refused by default, while historical codes (`DEM`, `FRF`) remain
  accepted.
- Unscaled `pound` and `pounds` in a weight context (`3 pounds of flour`,
  `weighs 150 pounds`) are refused by default, and `3 pounds sterling`
  includes `sterling` in the span.
- `MoneyConverter::with_minor_unit_rounding(true)` and
  `DatedMoneyConverter::with_minor_unit_rounding(true)` round converted
  amounts half to even to the target currency's ISO 4217 minor unit
  (`minor_units`); by default converted amounts keep all 16 digits of the
  division.
- The one gap a shape allows is any `White_Space` character, a line
  break included: `$` at the end of a line and `100` on the next are
  one mention.
- Currency words are lowercased code point by code point, so `RENMİNBİ`,
  with U+0130, is `renminbi`.

## Testing

`analysis/numeric/tests/money_extract.rs`,
`analysis/numeric/tests/money_extract_fixes.rs`,
`analysis/numeric/tests/money_convert.rs`, and
`analysis/numeric/tests/money_rates.rs` test extraction, regional symbol
disambiguation, weight-context exclusions, ECB rate parsing, and fixed-date
and dated currency conversion, alongside the `money` and `money-convert`
snapshot suites in both `LatinUs` and `LatinEu` notations.
