# Number and time grammar

`turbo-grammar` (`analysis/grammar/`) reads numbers, ordinals and
percentages written in words or digits, with exact values: `one hundred
and three and two thirds`, `1,674.5 million`, `minus 5`, `twenty-first`,
`the second to last`, `per cent of 210`. It also reads dates, times,
durations and sets, resolved against a reference date-time
([below](#times)). It is a rule grammar over a chart parser, in two
parts:

- `engine/`, which knows no language: a lexer, a numeral reader, rules
  over categories, the chart parser, and the ranking and selection of
  readings.
- `lang/`, one module per language, each a `Grammar`: its lexicon, its
  rules, how it writes numerals and which categories become output.
  English (`lang/en/`, the `ENGLISH` static), Spanish (`lang/es/`,
  `SPANISH`, [below](#spanish)) and Dutch (`analysis/grammar/src/lang/nl/`,
  `DUTCH`, [below](#dutch)). `ENGLISH_TIMES`, `SPANISH_TIMES` and
  `DUTCH_TIMES` hold the same number rules and add the time rules;
  `ENGLISH_UNITS` adds the rules for amounts in units and money
  (`docs/grammar-units.md`).

`GrammarAnnotator` runs a grammar over a text view and writes layers
([below](#layers)). It is separate from `turbo-numeric`, which ports
OpenNLP's number scan ([numbers.md](numbers.md)) and keeps its behaviour
and goldens; the two write different layer instances.

## Lexer

The grammar reads the view's text, not `opennlp:tokens`: a mention
starts and ends on character boundaries a tokenizer may not cut at
(`10k`, `twenty-one`, `4/5`). One pass cuts the text into lexemes:

- a word: a run of alphabetic characters and marks, lowercased into one
  buffer shared by the text's lexemes;
- a numeral: ASCII digits with separators, read by the numeral reader
  below; a dot or comma followed by digits after whitespace or at the
  start of the text opens one too (`.25`);
- any other character that is not `White_Space`: a punctuation lexeme of
  its own, so a lexicon can hold `%` or `½`.

A grammar's `WordForms` can ask the lexer for two more things: that a
word's lowercase form drop its accents for lookup (`tweeëntwintig` is
looked up as `tweeentwintig`), and that a word be cut into glued words
(`tweeentwintig` into `twee`, `en` and `twintig`). The spans stay those
of the text as written. `Lexicon::has_prefix` tells a splitter whether a
piece can still grow into a word.

Each lexeme records whether whitespace comes before it ("glued" when
none does). A whitespace-delimited chunk that contains `://` or starts
with `www.` is a web address and gives no lexemes. Offsets are UTF-8 byte
offsets into the view.

## Numerals

The reader takes a whole chain of digit runs joined by single `.` or `,`
characters, and, with `space_groups`, by a single space, no-break space,
narrow no-break space or thin space before exactly three digits. It
decides what the chain is. With the English style (`mixed_separators`,
`space_groups` and `indian_groups` on):

| Chain | Reading |
|---|---|
| `1,234,567`, `12,567`, `1 234 567`, `1.234.567` | a grouped integer: one to three digits, then groups of three, all with the same separator; a dot groups only when there are two or more |
| `12,34,567` | an Indian grouped integer: one or two digits, groups of two, a last group of three |
| `4.800`, `1,3`, `5,00` | a decimal: a lone dot always marks a fraction; a lone comma does unless three digits follow |
| `1,234,567.89`, `1.111,11`, `12,34,567.89`, `1 234 567.89` | a grouped integer and a fraction after a different separator |
| `192.168.1.2`, `22,24,26,31` | no single number: each digit run is an integer of its own |

A separator not followed by a digit ends the chain (`192.` is 192).

With `decimal_comma` (Dutch), a lone comma is the decimal point and a
dot groups before three digits (`1.234,56`, `4.800` is 4800). A grammar
names the notation of its numbers for the numbers layer through
`Grammar::notation`.

## Rules and the chart

A rule has a name, the category it builds, a list of items, a weight and
a function that builds the value from its children's values or refuses
the match. An item is a node of a category, optionally with a test on its
value and optionally glued to what comes before, or a word: a lexeme's
lowercase form, or its text as written for an exact-case item (`M` for
a million, never `m`), optionally absent.

The parser fills positions from the last lexeme to the first. At each
position it seeds a node for each lexicon entry of the lexeme (or the
numeral's node), runs the rules that open with that word, and then takes
each node made at the position in turn as the first child of the rules
indexed under its category. A rule's other items start later, at
positions already finished, so one pass is complete.

A node's score is the sum of the weights of the rules in its tree plus
the lexemes it covers. Two nodes over the same lexemes with the same
category and value are one, the better kept; at most four nodes share a
span and a category. "Better" is a higher score, then fewer rule
applications, then the lower rule index.

## Selection

Each output (numbers, ordinals, percentages) is chosen on its own from
the nodes of the categories the grammar sends to it. A candidate may not
start or end between two glued alphanumeric lexemes (`29km`, `x10`, the
`4` of `4th`), and the grammar may drop it (below). The rest are sorted
by length in bytes, then score, fewer rule applications, earlier start
and lower rule index, and taken greedily when they overlap none taken
before. The result is the same for the same text and grammar.

Because the outputs are chosen separately, the number of a percentage is
also a number, and `forty two and a third` holds the ordinal `third` as
well as the number 42.333…, as the Recognizers-Text models report them.
Numbers are chosen after ordinals, and a number that lies inside a chosen
ordinal is left out: `the twenty first` gives the ordinal 21 and no
number 20. A grammar can let an ordinal keep its numbers; English does so
for a count before a scale ordinal, since `two millionth` also reads as
two millionths and the Recognizers-Text model reports the number 2.

## English

What the rules read, by kind:

- Integers in words: units, teens and tens (`fifty-two`, `fifty - two`,
  `fifty two`), `hundred` after a number below a hundred (`nineteen
  hundred`), scale words (`thousand`, `million`, `billion`, `trillion`,
  `lakh`, `crore`, `mln`, `bln`, `tln`, plurals included) after a number
  below the scale (`two hundred and two thousand`, `three lakh crore`),
  `a hundred`, `a million`, and sums of a number ending in a scale with a
  smaller part, with or without `and` (`twenty-one trillion and three
  hundred`). A part does not stop before a multiplier that could not take
  the sum: `3 hundred and 2 hundred` is two numbers.
- Digits with words: a digit integer takes any scale word (`322
  hundred`, `13 lakh`, `4 thousand 3 hundred and 21`), a decimal numeral
  too (`1,674.5 million`, `2.12 hundred`). Suffixes after the digits,
  glued or not: `k` and `K`, `M` and `MM`, `mil`, `G`, `b` and `B`, `T`.
  A lowercase `m` or `mm` is a length, not a million.
- Dozens: `a dozen`, `three dozens`, `3 dzs`, `half a dozen` (6).
- Negatives: `minus` or `negative` before a number, and `-` or U+2212
  before a digit number when the sign starts the text, follows whitespace
  or an opening bracket or quote, and does not follow a number (`5 - 10`
  is a range, `SCP-173` and `+1-222` hold no negative number).
- Fractions in words: a spelled numerator and an ordinal denominator,
  with one taking the singular and anything else the plural (`one
  fifth`, `three fifths`, `twenty twenty-fifths`); `a` or `an` and a
  singular denominator; `half` and `quarter` as denominators; a lone
  `half`. `first` and `second` are never denominators (`a second`).
- Mixed numbers and scaled fractions: `one and a half`, `two hundred and
  fifty-four hundredths` (a proper fraction after `and`), `one million and
  a half` (after a million or more, the fraction is of the scale), `one
  billion and three quarters million`, `two thirds of a million`, `half
  million`, `five and a half million`.
- Fractions in digits: `3/4`, `-4/5`, `2 1/4`, `½`; not inside a run of
  slashes or glued dashes (`5/17/2018`, `222-2222/2222`).
- Fractions with prepositions: `one over three`, `2000 over 3`, and with
  `in` or `out of` only when the part is below the whole (`1 in 5`; `12
  out of 10` is two numbers). Not after a currency symbol (`$2000 over 3
  years`) and not with a year after `in` (`3 in 2009`).
- Spoken decimals: `two point five`, `nine point nought four`, `two
  hundred point seventy-one`.
- Exponents: `1e10`, `12.32e+14`, `-12e-1`, `1.3x10^6`, `1.1^23`,
  `2.5^-1`, with a run of signs (`^--23` is `^23`). A power with `^` is
  read up to an exponent of 64.
- Ordinals: `first` to `nineteenth`, the tens, compounds (`twenty-first`,
  `thirty fifth`), scale ordinals (`hundredth` to `trillionth`, `two
  hundredth`, `a hundred trillionth`), sums (`one hundred twenty fifth`),
  digits with the suffix English gives them (`1st`, `22nd`, `13th`; not
  `21th`), and `nineth`.
- Relative ordinals: `last`, `next`, `previous`, `current`, `next one`,
  `previous one`, `current one`, `penultimate`, `antepenultimate`,
  `second to last` (and `third to last` and so on, with `the` when it is
  written), `next to last`, `the last but one`, `the one before the last
  one`.
- Percentages: a number followed by `%`, `percent`, `percents`,
  `percentage`, `per cent`; `percent of`, `per cent of` before a number.

Ties between readings of the same text go by weight: a mixed number
over an improper fraction (`one hundred and three quarters` is 100.75),
a fraction of a large scale over a sum, and a compound ordinal
denominator over a longer numerator (`one hundred thirty fifths` is
100/35).

The grammar drops two kinds of candidates: a lone `one` after `the`,
`this`, `that`, `which`, `each`, `every`, `any` or `no`, or after an
ordinal or relative ordinal (a pronoun: `which one`, `the last one`, `the
3rd one`), and from the ordinals a plural ordinal (a denominator) and
a scale ordinal alone after `a` or `an` (`a hundredth` is a fraction).

A number's subtype is `fraction` for every fraction, `power` for an
exponent, `decimal` for a spoken decimal, and otherwise `integer` or
`decimal` by whether the value has a fraction (`5.00` is an integer).

## Times

`ENGLISH_TIMES` reads dates, times, date-times, durations, sets and
ranges of them, and `TimeAnnotator` writes them as the `turbo:times`
layer (`analysis/layers/src/times.rs`). The code is in two parts:

- `analysis/grammar/src/time/` knows no language. `spec.rs` holds what a
  text states (a date such as the next Friday or May 29 of an unnamed
  year, a time of day, a part of the day, a shift, a range, a
  recurrence); `duration.rs` a length in calendar units; `calendar.rs`
  the date arithmetic; `timex.rs` the TIMEX3 text; and `resolve/` turns
  a value and a reference date-time into a kind, a grain, a modifier and
  one or more readings, each with its timex and its start and end.
- `analysis/grammar/src/lang/en/time/` holds the English rules, grouped
  by family: `clock.rs` (`7:56:30 pm`, `half past seven`, `in the
  morning`), `analysis/grammar/src/lang/en/time/date/mod.rs` (`04th Jan 2019`,
  `next Monday`, `the first Monday of next month`, `Q3 2019`, holidays),
  `duration.rs` (`3.5 years`, `an hour and a half`, `a few minutes`), `set.rs` (`weekly`, `every two
  days`, `tuesday afternoons`), `compose.rs` (a date with a time, `in 5
  minutes`, `2 days ago`, `now`) and
  `analysis/grammar/src/lang/en/time/range/mod.rs` (`from May to Oct 2020`,
  `5 to 6pm`, `next 3 days`, `within 9 months`, `before 2018`, `since the
  end of 1989`, `mid-November`). `lexicon.rs` holds the words and
  `analysis/grammar/src/lang/en/time/mod.rs` the checks that drop a
  candidate in its context.

Time values are interned in the parse, so a rule's value is a small id.

The rules follow the conventions of the Recognizers-Text specifications
where they choose a reading:

- A clock with no `am` or `pm` reads in both halves of the day; the first
  is preferred. A weekday or a day of an unnamed year reads before and
  after the reference; the later reading is preferred. A decade without
  its century (`the eighties`) prefers the last one that starts by the
  reference.
- `may`, `sun` and a few other words with a common sense need a cue to
  be a date (`in May`, `on Sun`); `last` after `will` or `to` is a verb.
- Bare `to date` needs an adverbial context: a clause boundary or a
  recognized finite predicate (`sales to date exceed ...`). An infinitive
  governor (`wanted to date`) or following object or noun modifier
  (`to date him`, `to date night`) suppresses both the standalone node
  and its use inside a larger time expression. These lexical checks are
  conservative; they do not perform general syntactic tagging.
- A yearless range with February 29 at either endpoint pairs both dates
  around the same leap-day occurrence. A range crossing December takes
  the neighboring year for its other endpoint; inclusive edges apply
  after pairing. This avoids combining independent year searches into
  multi-year ranges for adjacent calendar days.
- A date with a dot after a full month name (`April. One`) is two
  sentences, and a slash pair after a number or before `of` (`21 1/4`,
  `3/4 of`) is a fraction.
- `around` keeps the point and marks it approximate; `later this week`
  is clipped at the reference and shows no modifier; `end of tomorrow` is
  its last second.
- `early`, `mid` and `late` split a year into four-month thirds (`mid
  1989` is May to August), and a decade into 4, 3 and 3 years (`the late
  eighties` is 1987 to 1989), a split the specifications do not test.
- `every other Friday` and `every second Thursday` are sets with the
  weekday's value (`XXXX-WXX-5`), as Recognizers-Text writes them;
  `every other day` and `every second week` double the unit (`P2D`).

`TimeAnnotator` reads `turbo:view` and, when the pipeline has one, the
`DocumentDate`. The reference is the document's date at midnight, else
the one given to `TimeAnnotator::with_reference`. There is no clock: with
no reference, a mention whose value depends on one (`tomorrow`, `next
week`) is written with its kind and grain and with no start, no end and
an empty timex; `2016-11-07`, `at 5` and `3 hours` resolve as usual. The
preferred reading is written, and `ambiguous` says whether there was
more than one.

## Dutch

`DUTCH` reads numbers, ordinals and percentages, and `DUTCH_TIMES` adds
the time rules. The lexer folds accents, so the lexicon holds `een` for
`één` and `en` for `ën`, and cuts compound number words into glued
pieces (`analysis/grammar/src/lang/nl/split.rs`): a word the lexicon does
not hold is cut when every piece is a number word, `en` stands only
between two of them and an ordinal or fraction word comes last; the cut
with the fewest pieces wins. Builders that read the same as in English
are the English ones.

- Integers: units, teens, tens, compounds with `en` (`tweeëntwintig`,
  `vijfentwintig`), `honderd` and `duizend` with or without a count,
  hundreds of a count (`vijfentwintighonderd`), `miljoen`, `miljard`,
  `biljoen` and their plurals, written as one word or apart
  (`driehonderdvijftig`, `twee miljoen driehonderdduizend`), digits with
  a scale word (`3 miljoen`, `1,5 miljard`), and `dozijn`.
- Digits with a decimal comma (above); a lone `een` written without
  accents is the article, `één` is the number unless a determiner makes
  it a pronoun (`de één`).
- Fractions: `drie vijfde`, `een derde`, `driekwart`, `half`,
  `anderhalf`, `tweeënhalf`, `3/4`, `1 op 5`, `2 uit 10`; spoken decimals
  (`vijf komma vijf`); negatives (`min 5`).
- Ordinals: `eerste` to the scale ordinals, compounds
  (`honderdvijfentwintigste`), digits with the suffix Dutch gives them
  (`1e`, `1ste`, `2de`, `8ste`, `100ste`); relative ordinals (`laatste`,
  `voorlaatste`, `volgende`, `vorige`, `huidige`, `op twee na laatste`). A
  plural ordinal is a denominator and `een miljoenste` a fraction.
- Percentages: a number with `%`, `procent` or `percent`.

The time rules use the categories, values and most builders of the
English time grammar, which the crate shares
(`analysis/grammar/src/lang/en/time/`); the words, their order and the
readings of their own are Dutch:

- Clock times: `7.15 uur`, `om 8 uur`, `om 14u`, `half acht` (7:30),
  `kwart over drie`, `kwart voor drie`, `tien over half vijf` (4:40),
  `middaguur`, `middernacht`, with a part of the day (`'s avonds`, `in de
  ochtend`). A bare hour needs `om`, `rond`, `omstreeks` or `tegen`
  before it, or opens a range of hours ending in `uur` (`tussen 3 en 5
  uur`).
- Days and their parts written as one word are cut into glued words by
  the lexer (`zaterdagochtend`, `morgenavond`, `gisterochtend`,
  `vanmiddag`), and read like the words apart.
- Dates day first (`12-03-1985`, `3 maart`, `dinsdag 4 mei`, `op de
  18e`), relative days (`vandaag`, `morgen`, `overmorgen`, `gisteren`,
  `eergisteren`), relative weeks, months and years (`volgende week`,
  `vorig jaar`), seasons, decades and Dutch holidays (`kerst`, `pasen`,
  `koningsdag`).
- Durations (`twee uur en een kwartier`, `anderhalf jaar`, `een paar
  dagen`), shifts (`over een kwartier`, `drie jaar geleden`, `twee weken
  na pasen`), sets (`elke dinsdag om half tien`, `dinsdagen`, `om de twee
  weken`, `wekelijks`), ranges (`van 1 tot en met 5 juli`, `tussen 3 en 5
  uur`, `de komende drie dagen`) and modifiers (`voor`, `na`, `rond`,
  `sinds`, `begin`, `medio`, `eind van de maand`).

The short forms of months and weekdays that are also words (`zo`, `di`,
`jan`, `vrij`) need a cue before them (`op`, `van`, `tot`, `elke`) or an
abbreviation dot.

## Values

Values are `Dec` decimals: an `i128` mantissa and a power of ten. Sums,
products and powers are exact up to 36 significant digits and rounded
half away from zero past that; a quotient is cut at 34 significant
digits, so `one third` holds 34 threes.

## Layers

`GrammarAnnotator` reads `turbo:view` and writes:

| Output | Layer | Instance |
|---|---|---|
| Numbers | `Numbers` (`turbo:numbers`) | `grammar` (`GRAMMAR_NUMBERS`) |
| Percentages | `Quantities` (`turbo:quantities`), unit `%` | `grammar` (`GRAMMAR_PERCENTS`) |
| Ordinals | `Ordinals` (`turbo:ordinals`) | default |
| Quantities with units | `Quantities` (`turbo:quantities`), unit symbol | `grammar` (`GRAMMAR_PERCENTS`) |
| Money | `Money` (`turbo:money`) | `grammar` (`GRAMMAR_MONEY`) |

The unit rows come from `ENGLISH_UNITS`, which `GrammarAnnotator::english`
runs; `docs/grammar-units.md` describes the unit grammar, its layers and
its conformance.

The numbers and percentages use the existing layers under the instance
`grammar`, never the default instances `turbo-numeric` writes. A number's
numeral ends where its span ends, with the multiplier `One`, the marker
`None` and the notation `LatinUs`; `negative` says whether the value is
below zero, and the coefficient and scale hold the value, with the zeros
of a large value written out. `GrammarAnnotator::bound` names the
instances for a pipeline.

`Ordinals` is a layer of its own in `turbo-analysis-layers`: one item per
mention, in text order, with a base (`Start`, `End` or `Current`) and an
offset. `third` is the start plus 3, `last` the end plus 0, `penultimate`
the end minus 1, `next` the current item plus 1. That is the smallest
typed form that holds both kinds of ordinal; one layer with a kind column
for every output would hold columns most items leave empty.

## Conformance

`analysis/grammar/tests/spec.rs` runs the English `NumberModel.json`,
`OrdinalModel.json` and `PercentModel.json` of Recognizers-Text, copied
into `analysis/grammar/tests/data/recognizers-text/` with their licence
and a README naming the upstream commit.

- A case is eligible unless its `NotSupported` or `NotSupportedByDesign`
  field names `dotnet`.
- The expected `Start` and `End` are inclusive UTF-16 indices into the
  input; they are converted to a byte span through the input's
  `OffsetTable`, and spans are compared, never the `Text` field (which
  the specifications lowercase).
- A case passes when the mentions of the model's output and the expected
  results are equal sets: the same spans, type names (`number`,
  `ordinal`, `ordinal.relative`, `percentage`) and values. A number's
  value is equal within a relative 1e-12, since the specifications print
  .NET doubles to 15 digits; its subtype is compared when the case gives
  one. An ordinal's `offset` and `relativeTo` are compared exactly.
- `analysis/grammar/tests/data/pass/English/<File>.txt` lists the
  indices of the passing cases. A listed case that fails fails the test; a
  case that passes and is not listed is printed, and `BLESS=1 cargo test
  -p turbo-grammar --test spec` rewrites the lists. `SPEC_SHOW=1` prints
  every failing case with what the grammar found.

| File | Eligible | Passing |
|---|---|---|
| `NumberModel.json` | 234 | 233 |
| `OrdinalModel.json` | 54 | 54 |
| `PercentModel.json` | 20 | 17 |

The cases that fail:

- `NumberModel.json` #54, ` 2.33 k`: the value 2330 matches, but the
  specification gives the subtype `decimal` while `1.2b` and `1,674.5 G`
  are `integer`; the grammar calls every integral value of a suffix
  `integer`.
- `PercentModel.json` #17 to #19 (`a small percent`, `the % of
  increase`, `a percentage`): the specification reports a percentage
  without a value. The layer holds a value for every percentage, so the
  grammar reports none.

Every other case of the three files passes.

`analysis/grammar/tests/spec/datetime.rs` runs `DateTimeModel.json`
the same way, with `analysis/grammar/tests/data/pass/English/DateTimeModel.txt`
as its list. Each case is resolved against its `ReferenceDateTime`. A
found mention matches when its span and type name are equal and its
alternatives, each with its `timex`, `type`, `Mod`, `value`, `start` and
`end`, are the same set. The test also prints the cases whose spans
match and whose values do not, and the counts by the first expected
type (`none` for a case that expects no mention):

| Type | Eligible | Passing |
|---|---|---|
| `date` | 198 | 183 |
| `daterange` | 357 | 347 |
| `datetime` | 82 | 79 |
| `datetimerange` | 96 | 89 |
| `duration` | 46 | 46 |
| `set` | 46 | 46 |
| `time` | 34 | 34 |
| `timerange` | 47 | 42 |
| none | 64 | 61 |
| All | 970 | 927 |

Every remaining eligible case that does not pass across all 16
specification files is documented in
`analysis/grammar/tests/spec/reasons.rs` and verified by `ratchet`.
The rules are written for
English forms, not for single cases, and read forms the specifications
do not test (`third to last`, a U+2212 minus, `3 in 2009`); the unit
tests in `analysis/grammar/src/lang/en/tests.rs` cover them.

`analysis/grammar/tests/spec/dutch.rs` runs the Dutch models the same
way, with the lists in `analysis/grammar/tests/data/pass/Dutch/`:

| File | Eligible | Passing |
|---|---|---|
| `NumberModel.json` | 161 | 159 |
| `OrdinalModel.json` | 48 | 48 |
| `PercentModel.json` | 13 | 13 |
| `DateTimeModel.json` | 605 | 579 |

| Date-time type | Eligible | Passing |
|---|---|---|
| `date` | 126 | 119 |
| `daterange` | 205 | 199 |
| `datetime` | 51 | 49 |
| `datetimerange` | 74 | 66 |
| `duration` | 28 | 28 |
| `set` | 23 | 23 |
| `time` | 23 | 21 |
| `timerange` | 24 | 23 |
| none | 51 | 51 |

The two number cases that fail are `192.168.1.2`, which the
specification reads as 192.168 and 1.2, and `1 234 567,89`, where the
specification expects only `567,89`. The date-time cases that fail
include `op` taken into some spans and not others, `voor` meaning for,
and a few readings the specification gives in a form it does not give
for the same words elsewhere (each listed in
`analysis/grammar/tests/spec/reasons.rs`). The unit tests in `analysis/grammar/src/lang/nl/tests.rs` and
`analysis/grammar/src/lang/nl/time/tests.rs` cover forms the
specifications do not test.

## Spanish

`analysis/grammar/src/lang/es/` is the Spanish grammar. It differs from
the English one in four places the engine leaves to a grammar:

- Its numeral style sets `decimal_comma`: a lone comma always marks the
  fraction (`3,5`), a lone dot groups three digits after one to three
  digits that do not start with 0 (`1.234`), and `1.234,56` is
  1234.56. Its layers write the `LatinEu` notation.
- Its `WordForms` fold accents for lookup (`dieciséis` is looked up as
  `dieciseis`, `ñ` stays), keeping the spans of the text as written.
- Its `WordForms` cut a compound number word the lexicon does not hold
  into glued pieces (`analysis/grammar/src/lang/es/split.rs`):
  `tresmillonésimo`, `doscientosavos`, `treintaicincoavos`,
  `vigesimoprimero`.
- Its `keep` and `shadows` drop `cien` of `por ciento` and the ordinal
  of a fraction (`un quinto`, `tres quintos`).

The number rules read the long scale (`billón` is 10^12, `mil
millones` 10^9), gender (`veintiuna`, `doscientas una`), `-avo`
denominators, spoken decimals (`siete coma cinco`, `cinco con un
cuarto`), `menos` as a sign only where it opens a number (not `al menos
cinco`), ordinals in words and digits (`decimotercero`, `1.º`, `2da`),
relative ordinals (`penúltimo`, `antes del último`) and percentages (`por
ciento`, `por cien`, `%`). They reuse the English builders for the
values.

The time rules (`analysis/grammar/src/lang/es/time/`) have the English
categories and builders, with Spanish words and orders: day before month
(`27/11`, `3 de enero del 2019`), the relative word after its noun (`el
lunes que viene`, `la semana pasada`), `a las` before an hour, `y media`
and `menos cuarto`, `de la tarde`. A clock range with `a las` in the
middle wants its start said alike (`de las 5 a las 7`), so `martes 4 a
las siete` is a day and a time.

`analysis/grammar/tests/spec/spanish.rs` runs the Spanish models as the
English ones are run, reading a decimal comma in an expected value as a
dot. The pass lists are under `analysis/grammar/tests/data/pass/Spanish/`.

| File | Eligible | Passing |
|---|---|---|
| `NumberModel.json` | 206 | 204 |
| `OrdinalModel.json` | 82 | 82 |
| `PercentModel.json` | 66 | 66 |
| `DateTimeModel.json` | 744 | 706 |

`DateTimeModel.json` by the first expected type: `date` 149 of 160,
`daterange` 284 of 295, `datetime` 62 of 68, `datetimerange` 58 of 64,
`duration` 29 of 29, `set` 18 of 18, `time` 33 of 33, `timerange` 20
of 22, none 53 of 55.

The number cases that fail are `192.168.1.2` (the specification reads
192.168 and 1.2) and `un cientocincoavos` (a singular numerator with a
plural denominator). The time cases that fail include a date said with
`este` or `próximo` (`el próximo 6 de abril`) that the specification
resolves to one year, an article the specification takes for some
units and not others, and a few misspellings (all listed in
`analysis/grammar/tests/spec/reasons.rs`).

In a release build, a 100 KB text made of the Spanish date-time inputs
parsed and selected with `SPANISH` at about 13 MB/s and with
`SPANISH_TIMES` at about 4 MB/s, where `grammar_bench` gave 15.5 and 4.4
MB/s for English on the same machine.

## Speed

`cargo run --release -p turbo-grammar --example grammar_bench` times the
English grammar over a generated 100 KB text (the sentences are in
`analysis/grammar/tests/common/sample.rs`), and `analysis/grammar/tests/timing.rs` runs
the same text in the test build, checks the layers and prints the time.
The benchmark also times the parse with the unit rules and
`TimeAnnotator` over the same text.

## Not yet done

Time zones are available through `TimeAnnotator::with_time_zones`;
the default parser retains the existing zone-free specification contract.
See the explicit-zone section below.

## Parser boundary checks

The chart admits at most four live readings per span and category. A rule
cannot recur along a derivation without consuming another lexeme, which
bounds unary cycles even when optional words are absent. Recursion that
extends a span remains allowed. Candidates whose score, application count
or child offsets cannot be represented are skipped; node indexes reserve
the terminal sentinel and cannot wrap. This is not a global memory quota:
charts can still grow with input length and grammar ambiguity.

`Parser::new` checks numeral, lexicon and every rule category before parsing.
`Parser::select` must receive the original source text. Incompatible UTF-8
byte lengths yield no additional mentions; candidate endpoints are checked
for UTF-8 boundaries before the selection callback reads the context. The
grammar and time annotators record version `2` for these parser semantics.

Decimal normalization returns `None` if reducing the scale would underflow.
The existing unresolved-date convention, including `Feb 30`, is unchanged.


## Explicit time zones

Enable zone suffixes with
`TimeAnnotator::english().with_time_zones(Region::Unspecified)` using
`time::zone::Region`; Spanish and Dutch annotators use the same zone reader.
`3pm EST` writes `T15-05:00`; `14:00 UTC+2` writes `T14:00+02:00`.
The resulting source span includes the zone, and range TIMEX3 values carry
an offset on each endpoint. Zone suffixes participate in chart selection,
so `02:30 hora de Madrid` cannot be split into a bare hour and a duration.
`time::zone::at` also reads a standalone zone at a supplied byte offset.

Only expressions with an explicit clock consume a zone suffix, both during
selection and when writing the result. Part-of-day expressions keep their
local TIMEX3 value and source span: `this morning CST` keeps `this morning`
and its resolved range rather than consuming the unresolved abbreviation.

UTC/GMT accept signed hours, HHMM or HH:MM, within -14:00..+14:00.
Malformed offsets are refused as a whole. Standard/daylight abbreviations
have fixed offsets: EST/EDT, PST/PDT, MST/MDT, CET/CEST, EET/EEST and CDT.
`CST` needs `NorthAmerica` (-06:00) or `China` (+08:00); `IST` needs `India`
(+05:30) or `Ireland` (+01:00). Without that context the recognized zone
stays unresolved and the time has no numeric value or TIMEX3 string.
Unrecognized zones are not consumed. Generic Pacific and Mountain names
mean standard time, -08:00 and -07:00 respectively. They do not imply a
geographic daylight-saving rule. Other ambiguous abbreviations, including
`PT`, are not accepted. Callers can use an explicit numeric offset instead.

`hora de Madrid`, Madrid time, Amsterdam time forms, and London/Londres
forms use the modern EU rule for 1996 through 2099. A time-only city needs
a reference date; a dated expression uses that expression's date. A date
emitted as a separate mention does not override the clock's reference date.
The last Sundays of March and October change at 01:00 UTC. Missing spring
hours and repeated autumn hours are unresolved rather than guessed.
Dates outside that supported interval are refused. This is a bounded
rule implementation, not a complete historical IANA database; future
legislation can require an update. Rule source:
[IANA Europe](https://data.iana.org/time-zones/tzdb/europe).
Fixed North American abbreviations follow
[IANA North America](https://data.iana.org/time-zones/tzdb/northamerica).

The `Times` start/end columns retain their existing **local-clock** units;
the offset is in TIMEX3. Subtract the offset for an absolute UTC instant.
A time-only value has no absolute date. Range durations remain local-clock
durations, so a range crossing a daylight-saving transition can differ
from elapsed UTC duration. Zone policy and region enter the configuration
hash; the annotator version is 4, separating these results from the earlier
part-of-day suffix behavior. No external dependency is added.

### Specification status

Re-running the existing DateTimeModel suite gives 927/970 English,
579/605 Dutch and 706/744 Spanish: **43, 26 and 38 failures** remain.
The English `timerange` row previously claimed 43 passing (893 total)
in the table when the actual passing count on `main` was 42 of 47 (891
total); no `timerange` pass-list entry was removed, and the 5 remaining
`timerange` failures are documented in
`analysis/grammar/tests/spec/reasons.rs`. Zone-free counts are
unchanged by the explicit-zone reader.

All nine fixture cases carrying an expected `utcOffsetMins` are marked
unsupported by .NET, which the existing conformance test excludes. Thus
there is no eligible upstream zone case whose pass count can increase.
The added zone-offset audit runs those cases separately: six offsets agree,
two generic Mountain cases expect -06:00 in November rather than the
standard -07:00 policy, and the Dutch `9-6 pt` case uses an unsupported
abbreviation and expects -07:00 in November. This audit compares offsets,
not full span/type/alternative conformance. It does not turn disabled cases
into passing claims or modify the upstream fixture files.

New real-text tests cover English, Spanish and Dutch aligned zone suffixes,
invalid offsets, regional CST/IST, Madrid summer/winter dates, transition
boundaries, missing/repeated hours, missing reference dates and provenance.
Issue #927 stays open for full upstream zone-case resolution, including
bare `9-6 pt` ranges and an explicit decision about the contradictory
Mountain/PT expectations. Standalone zones are exposed by the reader,
not represented as a new `Times` layer kind.
