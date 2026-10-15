# Number grammar

`turbo-grammar` (`analysis/grammar/`) reads numbers, ordinals and
percentages written in words or digits, with exact values: `one hundred
and three and two thirds`, `1,674.5 million`, `minus 5`, `twenty-first`,
`the second to last`, `per cent of 210`. It is a rule grammar over a
chart parser, in two parts:

- `engine/`, which knows no language: a lexer, a numeral reader, rules
  over categories, the chart parser, and the ranking and selection of
  readings.
- `lang/`, one module per language, each a `Grammar`: its lexicon, its
  rules, how it writes numerals and which categories become output.
  English (`lang/en/`, the `ENGLISH` static) is the one there is.

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

Every other case of the three files passes. The rules are written for
English forms, not for single cases, and read forms the specifications
do not test (`third to last`, a U+2212 minus, `3 in 2009`); the unit
tests in `analysis/grammar/src/lang/en/tests.rs` cover them.

## Speed

`cargo run --release -p turbo-grammar --example grammar_bench` times the
English grammar over a generated 100 KB text (the sentences are in
`analysis/grammar/tests/common/sample.rs`), and `tests/timing.rs` runs
the same text in the test build, checks the layers and prints the time.

## Not yet done

Planned on #388: dates, times, durations and sets (the `turbo:times`
layer), numbers with units (currency, dimension, temperature, age), and
Spanish and Dutch grammars.
