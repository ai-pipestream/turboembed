# Numbers

`NumberScanner` in `turbo-numeric` (`analysis/numeric/`) finds the
numbers written in a text view and writes them as the `turbo:numbers`
layer. It is a port of the number scan in the OpenNLP numeric add-on:
`NumberScan` reads one number, and the candidate loop of
`CursorQuantityExtractor` finds where numbers start, here without the
money exclusion and without the unit the quantity extractor wants after
a number.

## The layer

`Numbers`, in `turbo-analysis-layers`, has one item per mention:

| Column | What it holds |
|---|---|
| `spans` | The whole mention: the minus sign, the digits and the scale marker |
| `numeral_end` | Where the digits, group separators and fraction end, before a scale marker; the span's end when there is none. The numeral starts at the span's start, or one byte later when the mention is negative |
| `negative` | Whether the mention starts with a minus |
| `coefficient` | The value's digits without sign, decimal point or leading zeros, `0` for zero |
| `scale` | How many of those digits are the fraction |
| `multiplier` | `One`, `Thousand`, `Million`, `Billion` or `Trillion` |
| `marker` | `None`, `Suffix` (`k`, `m`, `b`, `bn`) or `Word` (`thousand` to `trillion`) |
| `notation` | The notation the mention was read in |

`coefficient` and `scale` are the unscaled value and the scale of the
`BigDecimal` OpenNLP computes, with the multiplier applied: `1.5k` has
the coefficient `15000` and the scale 1. `Numbers::value` returns a
`DecimalRef`, which writes the value as `BigDecimal.toPlainString` does
(`-1500.0`, `0.05`) and converts it to the nearest `f64`. A negated zero
is zero, as in `BigDecimal`, though `negative` still records the sign in
the text.

## The scan rules

Offsets are UTF-8 byte offsets into the view, on character boundaries.

- A mention starts at a decimal digit (`Nd` in Unicode, such as ASCII
  `0..=9`, Arabic-Indic `U+0660..=U+0669`, Extended Arabic-Indic
  `U+06F0..=U+06F9`, Devanagari `U+0966..=U+096F`, or Fullwidth
  `U+FF10..=U+FF19`), or at a minus directly before one. All digits in a
  single numeral must belong to the same 10-digit Unicode block, and the
  value is normalized into canonical ASCII digits in `coefficient`.
  The character before the mention, skipping combining marks, must not be
  a letter, a number of any kind, a mark or a connector such as `_`. A
  minus also must not follow a currency symbol or `%`, so `5%-10%` is two
  positive numbers.
- A digit right after a group or decimal separator that follows a digit,
  or after an exponent marker (`1e-3`), never starts a mention: it goes
  on with the number before it.
- `NumberNotation::LatinUs` groups with `,` and marks the fraction with
  `.`; `LatinEu` the other way round. Callers can choose a fixed notation,
  resolve one from a `RegionCode` (`region_notation` /
  `NumberScanner::for_region`) or a BCP-47 locale tag (`locale_notation` /
  `NumberScanner::for_locale`), or run `RegionAwareNumberAnnotator`
  (`turbo-numeric:region-aware-numbers`) over `(View, Regions)`. When a
  document's `Regions` ballot carries votes for both `LatinEu` and
  `LatinUs` countries, `RegionAwareNumberAnnotator` tries the winning
  region's notation first and falls back to the secondary notation when a
  candidate numeral is written in the other convention (`1.234,5`
  alongside `1,234.5`).
- Grouping conventions supported by default:
  - Standard 3-digit grouping in the active notation (`1,234,567.89` in
    `LatinUs`, `1.234.567,89` in `LatinEu`).
  - Indian lakh and crore comma grouping in `LatinUs` (`1,00,000`,
    `12,34,567.89`, `1,23,45,678`, with 1 or 2 leading digits, one or
    more 2-digit comma groups, and a final 3-digit comma group).
  - Thin and no-break space 3-digit grouping (`U+00A0`, `U+2009`,
    `U+202F`, such as `1\u{00A0}000`, `1\u{2009}234\u{2009}567`,
    `1\u{202F}350,75`).
  - Swiss apostrophe 3-digit grouping (`'` and `U+2019`, such as `1'000`,
    `1\u{2019}234.56`, `1'234,56`).
  - Arabic thousands (`U+066C`) and decimal (`U+066B`) separators
    (`1\u{066C}234\u{066B}56` and `\u{0661}\u{066C}\u{0662}\u{0663}\u{0664}\u{066B}\u{0665}\u{0666}`).
- Scientific notation exponents (`1e3`, `1E3`, `1e+3`, `1e-3`, `1E-03`,
  `2.5e-3`, `2,500e+3`, and `2,5e-3` in `LatinEu`) adjust `coefficient`
  and `scale` directly. Malformed exponents (`1.e-3`, `1,e3`, `1e-3e-4`)
  are rejected whole.
- The scan fails, rather than returning the prefix it read, when it stops
  at a separator followed by a digit that does not form a valid group or
  fraction (`1.2.3`, `1,23`, `1'23`, `1\u{2009}3`, `1.234,5` read in
  `LatinUs`), or when a separator precedes an exponent (`1.e-3`).
- With scaling on, the default, a `k`, `m`, `b` or `bn` in either case
  directly after the digits multiplies the value and must be followed by
  a word boundary; any other letter there rejects the number (`5x`). One
  `White_Space` character and then `thousand`, `million`, `billion` or
  `trillion` in any case also multiplies it, when the word ends at a
  boundary; otherwise the number ends before the whitespace.
- After a mention the scan goes on at its end; after a failed candidate,
  at the next character.

## Differences from OpenNLP

By default (`opennlp_parity: false`), `NumberScanner`, `QuantityExtractor`,
`MoneyExtractor`, and `parse` accept the extended groupings, exponents,
and non-ASCII `Nd` digit scripts listed above. Calling
`.with_opennlp_parity(true)` (or `parse_with_parity(..., true, ...)`)
restores OpenNLP's strict ASCII-only `LatinUs`/`LatinEu` scan for the JVM
golden comparisons in `analysis/numeric/tests/goldens.rs`,
`analysis/numeric/tests/quantity_goldens.rs`, and
`analysis/numeric/tests/money_goldens.rs`.

Interleaved release-build timing (`cargo test --release -p turbo-numeric --test extended_numbers timing_receipt_interleaved_release -- --ignored --nocapture`, 6 interleaved rounds of 100 iterations on the same machine comparing `main` `97dee9b33` against this head):

- Existing ASCII/Latin corpus (`prose-english.txt`, `prose-german.txt`,
  `decimals-us.txt`, `eu-grouping.txt`, `both-notations.txt`, 112.8 KB x
  100 iterations, 6,000 mentions/iteration):
  - `main` (`97dee9b33`): median `21.15 ms` (`533.5 MB/s`, range
    `20.35 ms` to `23.09 ms`).
  - Default extended path (`with_opennlp_parity(false)`): median
    `20.71 ms` (`544.8 MB/s`, range `20.47 ms` to `21.28 ms`).
  - OpenNLP parity path (`with_opennlp_parity(true)`): median `21.00 ms`
    (`537.2 MB/s`, range `20.88 ms` to `25.05 ms`).
- Non-Latin corpus with existing-path ASCII numerals (Devanagari, Thai,
  Arabic, CJK fullwidth punctuation, and emoji prose, 1.58 MB x 100
  iterations, 15,000 mentions/iteration on both):
  - `main` (`97dee9b33`, ASCII-only scan): median `85.92 ms`
    (`1,836 MB/s`, range `85.48 ms` to `86.77 ms`).
  - Default extended path (`with_opennlp_parity(false)`): median
    `121.45 ms` (`1,299 MB/s`, range `120.43 ms` to `123.59 ms`).
- Non-Latin corpus with Devanagari, Arabic-Indic, and Fullwidth `Nd`
  numerals alongside ASCII numerals (1.70 MB x 100 iterations):
  - `main` (`97dee9b33`, skips non-ASCII numerals, 7,500
    mentions/iteration): median `61.46 ms` (`2,770 MB/s`, range
    `60.42 ms` to `62.93 ms`).
  - Default extended path (`with_opennlp_parity(false)`, extracts all
    15,000 ASCII and non-ASCII `Nd` mentions/iteration): median
    `154.86 ms` (`1,099 MB/s`, range `154.37 ms` to `161.51 ms`).

Behaviour that comes from the Java and is kept:

- A number with no scale marker is not checked for a boundary after it:
  `5` followed by U+0301 gives `5`, and without scaling `5x` gives `5`.
- Scale words are lowercased code point by code point with
  `Character.toLowerCase`, so `mİllion` (with U+0130) reads as `million`.

## Golden files

JVM goldens for this layer come from the goldens tool
(`analysis/goldens-tool`), as for the other OpenNLP ports
([analysis.md](analysis.md#golden-files)). The `num` input set
(`analysis/goldens/data/inputs/num/`) is scanned in both notations, and
`analysis/goldens/data/numbers/<notation>/num/<id>.golden` holds a
`turbo:numbers` block per text: `start`, `end` and `numeral_end` in
UTF-8 bytes, `negative` (0 or 1), `coefficient` and `scale` (the
`BigDecimal`'s unscaled value without sign and its scale) and `plain`
(`toPlainString`, for a person reading the file).
`analysis/numeric/tests/goldens.rs` runs `NumberScanner` over every text
in both notations and compares each row exactly.

The generator loads `NumberScan` from the opennlp-addons `numeric` jar.
The pinned OpenNLP fork also carries an older `NumberScan` in
`opennlp-runtime`, which accepts only an ASCII space before a scale
word; the generator refuses to run when the older class is the one
loaded (see `analysis/goldens-tool/README.md`).
