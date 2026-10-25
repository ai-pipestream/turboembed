# Quantities

`QuantityExtractor` in `turbo-numeric` (`analysis/numeric/src/quantity.rs`)
finds measured quantities and percentages in a text view and writes them
as the `turbo:quantities` layer. It is a port of the quantity package of
the OpenNLP numeric add-on:

| Rust | OpenNLP |
|---|---|
| `QuantityExtractor` | `CursorQuantityExtractor` and `QuantityAnnotator` |
| `QuantityExtractor::for_region`, `QuantityExtractor::for_locale` | `NumericPacks.quantity(Locale)` |
| `RegionAwareQuantityAnnotator` | Region-aware quantity extraction over `turbo:regions` |
| `Quantities`, `QuantityMention` | `Quantity` |
| `DEFAULT_UNITS` | `CursorQuantityExtractor.DEFAULT_UNITS` |

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
| `scale` | `BigDecimal.scale()` of the value, an `i32`, as in `Money` |
| `unit` | The unit token as written, or `%` (`PERCENT`) for a percentage |
| `normalized_negative` | Whether the normalized value is below zero; never true for zero |
| `normalized_coefficient` | The normalized value's digits in `base_unit` without sign, decimal point or leading zeros, `0` for zero |
| `normalized_scale` | `BigDecimal.scale()` of the normalized value |
| `base_unit` | The SI or stated base unit symbol (`m`, `kg`, `m³`, `K`, `m/s`, `s`, `B`, `Hz`, `W`, `J`, `m²`, `rad`), `unit` when no base conversion applies, or `""` when the value or its normalized form lies outside `Dec`'s range |

The written value is stored as in `Money`: `coefficient` and `scale` are the
unscaled value and the scale of the `BigDecimal` OpenNLP computes, with
grouping removed, so `1,250.5 GB` has the coefficient `12505` and the
scale 1. `Quantities::value` returns a `DecimalRef`, which writes the
value as `BigDecimal.toPlainString` does. A value read from text has a
scale of zero or more.

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

The unit is a string column, not an enum, because the unit set is open:
OpenNLP takes a caller's set in place of its own, and the port does too.
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

## Differences from OpenNLP

None in the results are intended and none are known. The tests port the
cases of the add-on's `CursorQuantityExtractorTest`, and a JVM oracle
(outside the repository) compared the port with the Java on 2,000,000
generated texts in three configurations (the default units in both
notations, and a custom set with a unit outside the Basic Multilingual
Plane), about 1,400,000 mentions, with no difference.

Where the APIs differ:

- Null arguments are not representable, and a mention's own checks
  (a unit that is not blank) are `Quantities::check`.
- A unit set given twice the same token keeps it once.
- The character classes come from the Unicode 18.0 tables of
  `turbo-unicode`, as in numbers.md.

## OpenNLP behaviour kept

Behaviour that comes from the Java and is kept, though it looks wrong:

- In `5-10%` the hyphen after a digit is no minus and `5` has no unit,
  so the one mention is `10%`. A hyphen after a letter, a currency
  symbol or a percent sign is no minus either: `a-5 kg` is plus 5 kg and
  `$-5%` is plus 5 %.
- Scale markers are not read: `3m` is 3 metres, though `$3m` is three
  million dollars, and `3 million km`, `5k kg` and `1.5k km` give
  nothing.
- The one gap may be any `White_Space` character, a line break included:
  `5`, a newline, then `kg` is one mention.
- Short units are read after any number: `at 5 g`, `5 t`, `100 m USD`.
  Only the first token of a compound unit is read: `1 MB/s` is `1 MB`.
- `percent` matches in any case (`50 PERCENT`) while units do not
  (`50 gb` and `2.5 KM` give nothing).
- Money comes first even where the text reads as a quantity: `$5 m` and
  `€5 kg` give no quantity, and `3 pounds` is money, not a weight. A
  custom unit that is a currency code (`USD`) is never a quantity, since
  `5 USD` is money.
- A number with grouping the scan cannot read gives nothing at all:
  `1 000 kg` with a thin or no-break space, `1'000 kg` and `1,2345 km`.

## Golden files

JVM goldens for quantities come from the goldens tool
(`analysis/goldens-tool`), as for the other OpenNLP ports
([analysis.md](analysis.md#golden-files)).

- Inputs: the `quantity` set, `analysis/goldens/data/inputs/quantity/`,
  61 texts.
- Goldens: `analysis/goldens/data/quantity/<notation>/quantity/<id>.golden`,
  `latin-us` or `latin-eu`, from `QuantityAnnotator` over
  `CursorQuantityExtractor` with its default units. Layer
  `turbo:quantities`.

The columns are `start`, `end` (UTF-8 byte offsets of the whole mention),
`coefficient` (unscaled value without sign), `negative` (0 or 1, from the
value's sign), `scale`, `plain` (`toPlainString()`) and `unit`. The value
is `(-1)^negative * coefficient / 10^scale`. The `config` record is
`notation` (`LATIN_US` or `LATIN_EU`). `analysis/numeric/tests/quantity_goldens.rs`
runs `QuantityExtractor` over the set in both notations and compares the
layer with the goldens row for row.
