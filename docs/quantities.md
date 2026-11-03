# Quantities

`QuantityExtractor` in `turbo-numeric` (`analysis/numeric/src/quantity.rs`)
finds measured quantities and percentages in a text view and writes them
as the `turbo:quantities` layer:

| Type | Role |
|---|---|
| `QuantityExtractor` | Extracts measured quantities and percentages into `Quantities` (`turbo:quantities`) |
| `QuantityExtractor::for_region`, `QuantityExtractor::for_locale` | Configures regional number notation (`LatinUs` or `LatinEu`) |
| `RegionAwareQuantityAnnotator` | Region-aware quantity extraction over `turbo:regions` |
| `Quantities`, `QuantityMention` | Typed layer and mention accessor |
| `DEFAULT_UNITS` | Default unit symbol set |

Units and amounts written as words ("6 ft 2 in", "$3.5m") are also
read by the rule grammar in `turbo-grammar` (`analysis/grammar/`).
The number inside a mention is read by the number scan of
[numbers.md](numbers.md), and money is found by the extractor of
[money.md](money.md).

## The layer

`Quantities`, in `turbo-analysis-layers`, has one item per mention:

| Column | What it holds |
|---|---|
| `spans` | The whole mention: a minus sign, the number, the gap if there is one, and the unit |
| `negative` | Whether the value is below zero; never true for zero |
| `coefficient` | The value's digits without sign, decimal point or leading zeros, `0` for zero |
| `scale` | The decimal scale of the value, an `i32`, as in `Money` |
| `unit` | The unit token as written, or `%` (`PERCENT`) for a percentage |
| `normalized_negative` | Whether the normalized value is below zero; never true for zero |
| `normalized_coefficient` | The normalized value's digits in `base_unit` without sign, decimal point or leading zeros, `0` for zero |
| `normalized_scale` | The decimal scale of the normalized value |
| `base_unit` | The SI or stated base unit symbol (`m`, `kg`, `m³`, `K`, `m/s`, `s`, `B`, `Hz`, `W`, `J`, `m²`, `rad`), `unit` when no base conversion applies, or `""` when the value or its normalized form lies outside `Dec`'s range |

The written value is stored as in `Money`: `coefficient` and `scale` store
the unscaled integer value and decimal scale with grouping removed, so
`1,250.5 GB` has the coefficient `12505` and the scale 1.
`Quantities::value` returns a `DecimalRef`, which formats the value as a
plain decimal string. A value read from text has a scale of zero or more.

Alongside the written value and unit, `Quantities::normalized_value`,
`Quantities::normalized_dec` (`Option<Dec>`), and `Quantities::base_unit`
expose the quantity normalized to its SI or stated base unit through the
shared unit table in `analysis/layers/src/unit/mod.rs` (for example,
`2.5km` -> `2500` `m`, `500 g` -> `0.5` `kg`, `250 mL` -> `0.00025` `m³`,
`100 °C` -> `373.15` `K`, `36 km/h` -> `10` `m/s`), so quantities can be
compared and filtered across units while keeping the written columns
unchanged. A unit with no defined base conversion, including temperature
degrees without a scale (`deg`), keeps its stated value and unit. When a
written value or its converted form lies outside `Dec`'s
`i128` mantissa or `i16` scale range, `base_unit` is left empty (`""`) and
`Quantities::normalized_dec` returns `None`; `normalized_value` then retains
the written value and must not be treated as a converted value. Note that
bare unit symbols in `Unit::from_symbol` (such as `C`, `F`, `K`, `st`, and
`d` for Celsius,
Fahrenheit, Kelvin, stone, and day) map to typed units in the shared unit
table, so a caller passing a custom unit list with the same spelling gets
that unit's normalization.

The unit is a string column, not an enum, because callers can pass a custom
unit set in place of `DEFAULT_UNITS`.
`Quantities::is_percent` tells a percentage apart.

## Extraction

Offsets are UTF-8 byte offsets into the view, on character boundaries.

1. The money amounts of the text are found first, with `MoneyExtractor`
   in its default symbol table and the extractor's notation.
2. The scan then tries a mention at each character that is not inside an
   amount. A mention is an optional minus, a number, then a unit:
   - The minus and the first digit follow the rules of numbers.md: a
     digit that goes on with a number before it never starts one, so
     `1,00,000 kg` gives nothing rather than `0 kg`, and a minus counts
     only where it may open a negative number.
   - The number is read without scale markers, in the extractor's
     `NumberNotation` (`LatinUs` unless the caller chooses `LatinEu`).
     A number the scan rejects (bad grouping, an exponent) is no mention.
   - The unit comes directly after the number or after one `White_Space`
     character. It is a `%` followed by a word boundary, the word
     `percent` in any ASCII case, or a run of at most eight UTF-16 units
     of letters, followed by a word boundary, that is in the unit set.
     Units match exactly and case-sensitively.
3. After a mention the scan goes on at its end. A mention that would run
   into the next money amount is dropped, and the scan still goes on at
   its end.

The default units are `km`, `m`, `cm`, `mm`, `nm`, `mi`, `ft`, `yd`,
`kg`, `g`, `mg`, `t`, `lb`, `lbs`, `oz`, `L`, `mL`, `ml`, `ms`, `ns`,
`min`, `hr`, `KB`, `MB`, `GB`, `TB`, `PB`, `Hz`, `kHz`, `MHz`, `GHz`,
`kW`, `MW`, `GW`, `kWh`, `mph` and `kph`. `QuantityExtractor::with_units`
takes another set; each token must be one to six UTF-16 units, all
letters (`Character.isLetter`), or `UnitError` says which one is not. The
percent sign and word are read with any set.

## Edge cases and testing

Native extraction inherits the money exclusion rules described in
[money.md](money.md). Regional dollar forms such as `A$2m`, `US$2m` and
`HK$2m` are money, so the trailing `m` is not read as metres.

Edge-case rules:

- In `5-10%` the hyphen after a digit is no minus and `5` has no unit,
  so the one mention is `10%`. A hyphen after a letter, a currency
  symbol or a percent sign is no minus either: `a-5 kg` is plus 5 kg and
  `$-5%` is plus 5 %.
- Scale markers are not read on quantities: `3m` is 3 metres, whereas
  `$3m` is three million dollars, and `3 million km`, `5k kg` and
  `1.5k km` give nothing (spelled-out scaled quantities are handled by
  `turbo-grammar`).
- The one gap may be any `White_Space` character, a line break included:
  `5`, a newline, then `kg` is one mention.
- Short units are read after any number: `at 5 g`, `5 t`, `100 m USD`.
  Only the first token of a compound unit is read by `QuantityExtractor`
  (`1 MB/s` is `1 MB`; compound units are handled by `turbo-grammar`).
- `percent` matches in any case (`50 PERCENT`) while unit symbols match
  case-sensitively (`50 gb` and `2.5 KM` give nothing).
- Money comes first even where the text reads as a quantity: `$5 m` and
  `€5 kg` give no quantity. A custom unit that is a currency code (`USD`)
  is never a quantity, since `5 USD` is money.

`analysis/numeric/tests/quantity_extract.rs` and
`analysis/numeric/tests/quantity_normalize.rs` test extraction, SI base-unit
normalization, money-exclusion precedence, and the `quantity` snapshot suite
in both `LatinUs` and `LatinEu` notations.
