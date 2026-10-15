# Money

The `money` module of `turbo-numeric` (`analysis/numeric/src/money/`)
finds amounts of money in a text view, writes them as the `turbo:money`
layer, and converts them into one currency with the European Central
Bank's euro reference rates. It is a port of the money package of the
OpenNLP numeric add-on:

| Rust | OpenNLP |
|---|---|
| `MoneyExtractor` | `CursorMoneyExtractor` and `MoneyAnnotator` |
| `EcbRates` | `EcbFxRates` |
| `FxRates` | `FxRates` |
| `MoneyConverter` | `MoneyConversionAnnotator`, fixed-date mode |
| `DatedMoneyConverter` | `MoneyConversionAnnotator`, document-date mode |
| `Decimal`, `Date` | `BigDecimal`, `LocalDate`, as far as these need them |

The number inside a mention is read by the number scan of
[numbers.md](numbers.md).

## The layer

`Money`, in `turbo-analysis-layers`, has one item per mention:

| Column | What it holds |
|---|---|
| `spans` | The whole mention: a minus sign, the symbol or code, the number, its scale marker and a currency word after it |
| `negative` | Whether the amount is below zero; never true for zero |
| `coefficient` | The amount's digits without sign, decimal point or leading zeros, `0` for zero |
| `scale` | `BigDecimal.scale()` of the amount, an `i32` |
| `currency` | A `Currency`: three ASCII capital letters, stored as three bytes |

`coefficient` and `scale` are the unscaled value and the scale of the
`BigDecimal` OpenNLP computes: `$1.2M` has the coefficient `12000000`
and the scale 1. `Money::amount` returns a `DecimalRef`, which writes
the value as `BigDecimal.toPlainString` does. An extracted amount has a
scale of zero or more. A converted amount can have a scale below zero,
since a `BigDecimal` rounded to 16 digits keeps its trailing zeros as a
negative scale (`1085600000000000` at the scale -5 is
108560000000000000000).

The extractor writes the default instance. `MoneyConverter` writes a
second instance of the same type under the key `CONVERTED_MONEY` (the
name `converted`), as OpenNLP writes `money.converted` beside
`money-amounts`. The converted layer has the spans of the mentions it
could convert and leaves the others out.

## Extraction

Offsets are UTF-8 byte offsets into the view, on character boundaries.
At each character the scan tries three shapes, in this order, and after
a mention goes on at its end:

- **Symbol first** (`$1,234.56`, `-$5`, `$1.2M`, `$ 100`): a symbol
  from the table, not directly after an ASCII letter (combining marks
  skipped), so `HK$50` and `US$50` give nothing. One `White_Space`
  character may follow the symbol. A minus before the symbol counts only
  where a minus may open a negative number (numbers.md), so the hyphen in
  `$100-$200` is not a sign.
- **Code first** (`USD 100`, `USD 1.2 million`): three ASCII capitals
  between word boundaries that make a known code, one `White_Space`
  character, then the number. This shape takes no minus.
- **Number first** (`100 USD`, `50€`, `1.2 million dollars`): a
  number, then a symbol directly after it, or one `White_Space` character
  and a code or a currency word. A digit that goes on with a number
  before it never starts one, so `1,00,000 USD` gives nothing rather than
  `000 USD`.

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

The known codes are the 233 alphabetic codes of
`java.util.Currency.getAvailableCurrencies()` under JDK 21.0.12, written
into the crate. OpenNLP takes the set from the running JDK, so a JVM of
another version can know a few codes more or fewer.

OpenNLP's `forRegion(Locale)` builds a table and a notation from the
JDK's locale data. That lookup is not ported; a caller that knows the
region builds the same extractor, for example
`MoneyExtractor::new(LatinUs).with_symbol('$', AUD)` for Australia.

## Rates

`EcbRates` loads the ECB's history CSV from a path or from bytes the
caller supplies; nothing is downloaded. The format is that of
`EcbFxRates`:

- The bytes must be UTF-8. Lines end at `\n`, `\r` or `\r\n`, as
  `BufferedReader.readLine` splits them, and blank lines
  (`Character.isWhitespace` only) are skipped.
- The header starts with `Date,` and names unique codes of three ASCII
  capitals. Fields are split at every comma and trimmed as `String.trim`
  trims. A control character below U+0020 that is not
  `Character.isWhitespace` refuses the file, so a NUL cannot be trimmed
  away. A header or row may end with one more, empty field.
- Each row has an ISO date (`LocalDate.parse`, strict) that no other row
  has, and a field per currency. An empty field or `N/A` means no quote;
  a quote is read as `new BigDecimal(String)` reads it, including an
  exponent and any decimal digits of the Basic Multilingual Plane, and
  must be above zero. An `EUR` quote must equal 1.

A lookup takes the latest row on or before the date asked for, at most
`MAX_STALENESS_DAYS` (7) days earlier, inclusive, so a weekend or holiday
uses the last reference day. Both currencies must be quoted in that
row; an older row is not searched for a missing one. The euro is always
1 on a usable row.

## Arithmetic

The arithmetic is exact and written by hand: unbounded naturals in base
10⁹ and a division rounded to 16 significant digits, half to even, as
`BigDecimal.divide(divisor, MathContext.DECIMAL64)` divides. Every
result was compared with the JVM's, scale included.

- A rate is the target quote divided by the source quote; the same
  currency gives exactly 1.
- A converted amount is the amount times the target quote, divided by
  the source quote, rounded once. An amount already in the target
  currency is kept as it is, digits and scale, but still needs a quote.
- A quotient's scale is the preferred scale (dividend scale minus
  divisor scale) when the exact quotient fits 16 digits there, and the
  16-digit scale otherwise, as in `BigDecimal`. No rounding to a
  currency's minor unit follows: 1 USD in EUR at 1.2 is
  `0.8333333333333333`.
- When a scale leaves the `int` range the JDK throws, and `EcbFxRates`
  divides the unscaled values instead and applies the combined scale;
  that fails only when the result itself does not fit. The port follows
  both steps and returns `ScaleRangeError` where the Java throws
  `ArithmeticException`. A zero amount converts to zero at any scale.

`FxRates` is the trait `MoneyConverter` reads rates through.
`EcbRates` implements it; another provider implements `rate` and
inherits `convert`, which multiplies the amount by the rate exactly, as
the Java interface's default method does.

`MoneyConverter` converts the default money instance into one target
currency as of one date and is added to a pipeline with
`MoneyConverter::bound`, which names its output instance.

`DatedMoneyConverter` is the Java annotator's document-date mode: it
converts as of the date in the `turbo:document-date` layer, which
`DocumentDateSelector` writes ([temporal.md](temporal.md)), and is added
with `DatedMoneyConverter::bound` after that step. A pipeline without a
step that writes the date layer is refused when it is built. A document
without a date gets an empty converted layer and the rates are not
asked, as in the Java.

## Differences from OpenNLP

None in the results are intended and none are known. The tests port
the cases of the add-on's money tests, and a JVM oracle (outside the
repository) compared the port with the Java on 2,000,000 generated
texts in three extractor configurations, 200,000 conversions with
scales up to the `int` limits, 200,000 rate strings, 100,000 date
strings and 400,000 generated CSV files, with no difference.

Where the APIs differ:

- Null and blank arguments are not representable: a code is a
  `Currency`, a date a `Date`, a symbol a `char`.
- A provider returns only an amount, so it cannot change a mention's
  span or currency, which the Java checks at run time.
- The bytes are checked as UTF-8 before any line is read. The Java
  decodes in blocks of 8192 characters as it reads, so in a larger file
  with both a bad byte and a malformed row, which of the two is reported
  can differ. Both refuse the file.
- Error offsets for a control character are bytes into the line, not
  UTF-16 units.
- The character classes come from the Unicode 18.0 tables of
  `turbo-unicode`, as in numbers.md.

Behaviour that comes from the Java and is kept, though it looks wrong:

- A minus before a code is dropped: `-USD 5` is a mention `USD 5` of
  plus 5 USD, while `-5 USD` and `-$5` are minus 5.
- Only an ASCII letter before a symbol marks it as part of a longer
  symbol: `HK$50` gives nothing, `Ü$5` gives 5 USD.
- The one gap a shape allows is any `White_Space` character, a line
  break included: `$` at the end of a line and `100` on the next are
  one mention.
- Every code the JDK lists is a currency, withdrawn ones (`DEM`, `FRF`)
  and the testing and no-currency codes `XTS` and `XXX` included:
  `XXX 5` is money.
- `3 pounds of flour` is 3 GBP; the Java documents this.
- Currency words are lowercased code point by code point with
  `Character.toLowerCase`, so `RENMİNBİ`, with U+0130, is `renminbi`.
- A rate field accepts what `new BigDecimal(String)` accepts, a leading
  `+`, an exponent and Arabic-Indic or fullwidth digits included, though
  the file is described as ASCII.
- DEL (U+007F) and U+0085 are not refused as control characters; they
  stay in the field and make it an invalid code or rate.
- A converted amount keeps the 16 digits of the division, with no
  rounding to cents, and its scale depends on the scales of the inputs
  (`12.000` divided by `1.2` is `10.00`).

## Golden files

JVM goldens for money come from the goldens tool (`analysis/goldens-tool`),
as for the other OpenNLP ports ([analysis.md](analysis.md#golden-files)).

- Inputs: the `money` set, `analysis/goldens/data/inputs/money/`, 64 texts.
- Rates: `analysis/goldens/data/rates/ecb-snapshot.csv`, a small hand made
  file in the format of the ECB's `eurofxref-hist.csv` with invented
  values (columns USD, JPY, GBP, CHF, CNY, INR, RUB; 12 weekdays from
  2026-03-02 to 2026-03-17; `N/A` where there is no quote).
- Extraction: `analysis/goldens/data/money/<notation>/money/<id>.golden`,
  `latin-us` or `latin-eu`, from `MoneyAnnotator` over
  `CursorMoneyExtractor`. Layer `turbo:money`.
- Conversion: `analysis/goldens/data/money-convert/<notation>/money/<id>.golden`,
  from `MoneyConversionAnnotator` over `EcbFxRates` into EUR as of
  2026-03-15 (a Sunday, so the Friday 2026-03-13 row applies). Layer
  `turbo:money` as well; a mention without a rate has no row.

Both layers have the columns `start`, `end` (UTF-8 byte offsets of the
whole mention, sign and symbol included), `coefficient` (unscaled value
without sign), `negative` (0 or 1), `scale`, `plain` (`toPlainString()`)
and `currency` (ISO 4217). The value is
`(-1)^negative * coefficient / 10^scale`. The `config` records are
`notation` (`LATIN_US` or `LATIN_EU`) and, for conversions, `target`,
`as_of` and `snapshot` (`rates/ecb-snapshot.csv`, relative to
`analysis/goldens/data`).
