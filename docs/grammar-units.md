# Unit grammar

`ENGLISH_UNITS` (`analysis/grammar/src/lang/en/unit/mod.rs`) is the
English number grammar of `docs/grammar.md` with rules for amounts in
units and for money: `the 6 ft 2 in player`, `120 km/h`, `minus 40
degrees`, `a 5-year-old`, `it costs $3.5m`, `5 dollars and 50 cents`.
Its rules follow the number rules and its categories start after theirs,
so it reads every number `ENGLISH` reads. `GrammarAnnotator::english`
runs it.

## Types

Units are enums grouped by what they measure, in
`analysis/grammar/src/unit/mod.rs`: `Length`, `Area`, `Volume`,
`Weight`, `Speed`, `Information`, `Angle`, `Temperature` and `Age`, with
`Unit` over them and `Unit::Currency` for money. `Area::Square`,
`Volume::Cubic` and `Speed::Per` take a `Length` (and a time), so `km²`,
`ft³` and `km/h` need no variant of their own. `Unit::measure` gives the
Recognizers-Text type (currency, dimension, temperature, age) and
`Unit::dimension` the subtype. `Display` writes the Recognizers-Text
unit name (`Square meter`, `Kilometer per hour`, `C`) and is used only to
compare with the specifications; `Unit::write_symbol` writes the symbol
the layers hold (`km`, `m²`, `km/h`, `°C`).

Units that share a name but differ in size are separate variants. A
tonne, `t` and a `metric ton` are `Weight::MetricTon` (symbol `t`); an
unqualified `ton`, a `short ton` and a `US ton` are `Weight::ShortTon`
(`sh tn`); a `long ton`, `British ton` or `UK ton` is `Weight::LongTon`
(`long tn`). An unqualified gallon, quart, pint or fluid ounce is the US
measure; with `imperial`, `UK` or `British` before it, it is the
imperial one (`Volume::ImperialGallon`, symbol `imp gal`, and so on).
`nautical miles` and `nmi` are `Length::NauticalMile`.

Money units are in `analysis/grammar/src/unit/currency.rs`:

- `CurrencyUnit::Code` holds an ISO 4217 `Currency` from
  `turbo-analysis-layers`, for codes, signs and named currencies (`EUR`,
  `€`, `euros`, `Australian dollars`).
- `CurrencyUnit::Generic` is a name several currencies share (`dollar`,
  `pound`, `franc`, `rupee`). `Generic::conventional` gives the code the
  name means when nothing else says: dollar USD, pound GBP, rupee INR,
  ruble RUB; the others have none.
- `CurrencyUnit::Minor` is a fraction of another unit (`cent`, `penny`,
  `satoshi`, `kopek`), with the power of ten that makes the major unit.

`CurrencyUnit::currency` gives the code and that power of ten: `cents`
is USD over 100, `pence` GBP over 100, `satoshi` XBT over 10^8. The
table of codes holds the 153 codes the specification names and a few
more (former European codes such as DEM and FRF, PHP, VND, and XBT for
bitcoin).

## Rules

The lexicon tables are in `analysis/grammar/src/lang/en/unit/lexicon.rs`
and the rules in `analysis/grammar/src/lang/en/unit/rules.rs`.

- A unit word is a name (`kilometers`), an abbreviation (`km`, `kg`,
  `mph`) or a sign (`°`, `"`, `£`). Abbreviations whose case matters are
  flagged: `Mb` and `MB` are bits and bytes, `mm` is a millimeter and
  `MM` a million.
- Composed units: a metric prefix and a base (`kilo` `meter`), `square`
  and `cubic` or a glued `2` and `3`, a length over a time (`km/h`,
  `meters per second`), `fluid ounces`, `light years`, and `degrees`
  followed by a scale (`degrees Celsius`, `°F`, `deg C`).
- A quantity is a number then a unit, or a unit sign then a number
  (`$5`, `€ 20`, `USD 30`), with an optional multiplier after either
  (`$3.5m`, `4.5 bn euros`, `2 m dollars`). `6 ft 2 in` and `5'11"` add
  the inches as twelfths of a foot. `5 dollars and 50
  cents` and `15 USD 50` add the minor unit to the major.
- A country or its adjective before a generic name names the currency
  (`Canadian dollars`, `HK$`, `US$`, `South African rand`); a code glued
  to `$` does the same (`CAD$`).
- An age is a number and a time unit with `old` (`30 years old`), a
  hyphenated `5-year-old`, or `aged 40` and `age 40`.
- A unit with no amount (`in kilometers`) is a mention of its own when it
  is a full unit name, a code or a currency word; a word that is also
  common English (`in`, `pound`, `all`, `cup`) is not.

`EnglishUnits::keep` settles readings the rules leave open:

- `degrees` is an angle when an angle word is within a few words
  (`turn`, `rotate`, `slope`) and no temperature word is (`hot`,
  `weather`, `oven`, `boils`); otherwise it is a temperature.
- `pounds` is money when `sterling` follows, when the amount is a
  million or more, or when a money word is nearby (`cost`, `paid`,
  `price`) and `of` does not follow; otherwise it is a weight (`2 pounds of apples`). `£` is always money.
- A code that is also an English word (`ALL`, `CUP`, `TRY`, `SOS`) must
  be upper case.
- A number in a hyphenated word (`scp2-173m`) or before a hyphen and a
  word (`3 t-shirts`) is not a quantity.
- A currency sign glued to a number belongs to that number: `$10
  dollars` is `$10`, and `5 $10` is 5 and `$10`.

## Layers

`GrammarAnnotator` selects `Output::Quantity` once and writes each
mention by its unit:

| Mention | Layer | Instance |
|---|---|---|
| Dimension, temperature, age | `Quantities` (`turbo:quantities`), unit symbol | `grammar` (`GRAMMAR_PERCENTS`) |
| Money | `Money` (`turbo:money`) | `grammar` (`GRAMMAR_MONEY`) |

Percentages and other quantities share the `Quantities` instance. The two
are selected apart, so a percentage and a quantity can overlap; the
longer is kept. Money is stated in the currency `CurrencyUnit::currency`
gives, divided by its power of ten: `50 cents` is 0.5 USD. Money with a
generic name that has no conventional code (`20 francs`) has no currency
to write and is left out of the layer. A unit named alone is not written;
neither layer has an item with no amount.

## Conformance

`analysis/grammar/tests/spec.rs` runs the English `AgeModel.json`,
`CurrencyModel.json`, `DimensionModel.json` and
`TemperatureModel.json`, copied into
`analysis/grammar/tests/data/recognizers-text/English/NumberWithUnit/`.
Eligibility, offsets and the pass lists work as in `docs/grammar.md`.
The harness is `analysis/grammar/tests/spec/units.rs`. Each file is
compared with the quantities of its kind among those selected, and a
case passes when they and the expected results are equal sets: the same
spans, type names, unit names, values (within a relative 1e-12, or both
absent for a unit named alone), a dimension's subtype, and the ISO code
where the case gives one. The specification files gallons under weight;
the harness takes that subtype as volume. It names a tonne, `t` and an
unqualified ton all `Ton`; the harness accepts a metric or a short ton
for that name.

| File | Cases | Eligible | Passing |
|---|---|---|---|
| `AgeModel.json` | 19 | 19 | 19 |
| `CurrencyModel.json` | 180 | 177 | 175 |
| `DimensionModel.json` | 100 | 100 | 99 |
| `TemperatureModel.json` | 43 | 43 | 43 |

The cases that fail:

- `DimensionModel.json` #52, `2 pm is 2 picometer`: `pm` is read as a
  time of day, not a unit.
- `CurrencyModel.json` #102, `one kuai and five mao five`: a trailing
  bare count is the next minor unit (fen); the grammar reads only one
  minor unit after the major.
- `CurrencyModel.json` #110, `sen 100,000`: `sen` is not read as a unit
  before a number.

## Speed

`cargo run --release -p turbo-grammar --example grammar_bench` prints
the parse time with and without the unit rules over the 100 KB sample
text. On the machine the numbers were taken on, a parse took about 5 ms
with the number rules and about 7.5 to 8.5 ms with the unit rules, and
`GrammarAnnotator::find` with all its outputs about 9.5 to 10.5 ms
(about 10 MB/s). The timings vary by several percent between runs.

## Not handled

- Rates other than speeds: `10 Mb/s` gives `10 Mb`, `$5 per hour` gives `$5`.
- Readings that need more than the next few words: `20 knots tied` reads
  as a speed, `3 cups` (trophies) as a volume, `300 m people` as a length
  and `version 2 g` as a weight.
- `6 foot 2` gives `6 foot` and leaves the 2.
