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
`analysis/layers/src/unit/physical.rs` and
`analysis/layers/src/unit/kinds.rs`, re-exported from
`analysis/layers/src/unit/mod.rs` and `analysis/grammar/src/unit/mod.rs`: `Length`, `Area`, `Volume`,
`Weight`, `Speed`, `Information`, `Angle`, `Temperature` and `Age`, with
`Unit` over them and `Unit::Currency` for money. `Area::Square`,
`Volume::Cubic` and `Speed::Per` take a `Length` (and a time), so `km²`,
`ft³` and `km/h` need no variant of their own. A `Rate` is information
or money over a time (`10 Mb/s`, `$5 per hour`), written by
`write_symbol` as `Mbit/s` and, on the money layer, stated in the
currency with the span covering the `per hour`. The grammar's short time
denominators use `wk`, `mo` and `yr` for week, month and year. Money keeps
the amount and currency without a separate period column; its source span
retains that wording. `Unit::measure` gives the
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

Money units are in `analysis/layers/src/unit/currency.rs`:

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

The lexicon tables are in `analysis/grammar/src/lang/en/unit/lexicon/mod.rs`
and the rules in `analysis/grammar/src/lang/en/unit/rules/mod.rs`.

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
  cents` and `15 USD 50` add the minor unit to the major. A bare
  trailing count of the next smaller imperial unit adds in the same
  way (`6 foot 2` in inches, `10 stone 4` in pounds, `5 pounds 8` in
  ounces) when it ends its clause, as a bare money hundredths count
  must. Ambiguous `pounds` keeps the existing context rule: `a baby of
  5 pounds 8` is weight, while `it costs 5 pounds 8` and `he paid 2 pounds
  6` are currency hundredths. Explicit `lb` remains a weight. An
  information unit or a money amount over a time (with
  `per`, `/`, `a` or `an`) is a rate, except that `a year earlier`
  and its kin stay a date wording, not a rate.
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
| `CurrencyModel.json` | 180 | 177 | 177 |
| `DimensionModel.json` | 100 | 100 | 100 |
| `TemperatureModel.json` | 43 | 43 | 43 |

Every eligible case of the four files passes.

## Spanish

`SPANISH_UNITS` composes the Spanish number grammar with quantities,
ages, temperatures and currencies. `GrammarAnnotator::spanish()` runs this
unit grammar and writes the same Money and Quantities instances as English.
Its provenance name is `es-units`, distinct from the number-only `es` grammar.
The existing lexer keeps the original byte spans, including accent folding,
decimal commas, grouped thousands and decomposed accents.

Rules include `veintidós kilómetros`, `metros cuadrados`, `pies cúbicos`,
`188 millas por hora`, `10 Mb/s`, `6 pies y 3 pulgadas`, `5 libras y 8 onzas`,
`37ºC`, `trece euros y cuarenta y cinco céntimos`, `15 USD 50`,
`2,38 mil millones de dólares`, `C$ 16,4 millones` and `20 dólares por hora`.
`Mb` and `MB` remain distinct. `º` requires a temperature scale or temperature
context, so ordinary ordinal indicators such as `3º` are not quantities.
Currency country qualifiers follow the name, as in `dólares canadienses` or
`francos suizos`. Generic dollars retain the existing USD convention; the
rules do not infer a national currency from the language alone. Bare minor
unit names with no conventional currency remain parser readings and are
not forced into the Money layer. Money rate spans retain the denominator;
the Money schema still stores the amount and currency, not a period column.

All 240 unchanged Spanish Recognizers-Text NumberWithUnit cases at the pinned
revision pass: 19 ages, 122 currencies, 54 dimensions and 45 temperatures.
`spanish_unit_specs.rs` uses the shared exact-span comparator, translating
only localized expected unit names and decimal commas. It compares every
expected result, value, type, subtype and explicit ISO code, with no pass-list
exclusions. Additional tests run the real Analyzer pipeline, source-offset
round trips, Unicode whitespace, compound bounds and ambiguous ordinal marks.

## Portuguese

`PORTUGUESE_UNITS` and `GrammarAnnotator::portuguese()` add Portuguese
quantities, ages, temperatures and currencies. The provenance name is
`pt-units`. Portuguese cardinal forms supply the unit amounts, including
`vinte e dois quilômetros`, gendered hundreds, `mil`, `milhões`, Brazilian
`bilhões` (10^9), Portuguese `biliões` (10^12), digit fractions and halves.
This unit-oriented number grammar does not claim full Portuguese ordinal,
percentage or spelled-decimal coverage.

Rules include `6 pés e 3 polegadas`, `5 libras e 8 onças`,
`3 metros quadrados`, `2,5 quilômetros por hora`, `10 Mb/s`, `37ºC`,
`R$25,50`, `três reais e cinquenta centavos` and `20 reais por hora`.
The original source spans survive accented and decomposed text and Unicode
whitespace; `Mb` and `MB` remain distinct. `um` is also a number/article, so
its micrometer abbreviation is accepted only after a digit amount; the
unambiguous `micrômetros` name accepts spelled amounts as well. Currency
nouns do not consume following articles. Explicit `real`, `reais` and `R$`
mean BRL, while generic dollars and unqualified minor units retain the
shared currency conventions described above. Money rate spans retain the
denominator; the existing Money layer stores no separate period column.

All **253 unchanged Portuguese NumberWithUnit cases** at the same pinned
Recognizers-Text revision pass: **18 Age, 123 Currency, 65 Dimension and
47 Temperature**. No cases are excluded. The comparator translates only
localized resolution labels and decimal commas, then checks exact spans,
values, types, subtypes and explicit currency codes. It retains the shared
comparison conventions for `Ton` and the upstream gallon subtype. Several
upstream Portuguese fixtures contain Spanish prose or unaccented aliases
(`yenes`, `oncas`); those original fixtures are retained, and separate native
Portuguese regressions cover `ienes`, `onças` and BRL compounds. The suite
also runs the real Analyzer, layer checks, UTF-16/codepoint offset round
trips, negative boundaries and the distinct `cem`/`cento` forms.

## Speed

`cargo run --release -p turbo-grammar --example grammar_bench` prints
the parse time with and without the unit rules over the 100 KB sample
text. On the machine the numbers were taken on, a parse took about 5 ms
with the number rules and about 7.5 to 8.5 ms with the unit rules, and
`GrammarAnnotator::find` with all its outputs about 9.5 to 10.5 ms
(about 10 MB/s). The timings vary by several percent between runs.

## Not handled

- Readings that need more than the next few words: `20 knots tied` reads
  as a speed, `3 cups` (trophies) as a volume, `300 m people` as a length
  and `version 2 g` as a weight.
- Glued rate words (`200 Mbps`) and rates over units other than time
  (`$5 per person`).
- Dutch, Italian and Japanese remain separate #929 language slices.

## Chinese

`CHINESE_UNITS` runs through `GrammarAnnotator::new(&CHINESE_UNITS)` with source
spans intact. Its 91-case pinned corpus, currency composition, CJK boundaries,
modern mainland conversion conventions and explicit limits are documented in
[Chinese unit conventions](grammar-units-chinese.md).
