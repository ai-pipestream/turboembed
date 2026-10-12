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

- A mention starts at an ASCII digit, or at a minus directly before one.
  The character before it, skipping combining marks, must not be a
  letter, a number of any kind, a mark or a connector such as `_`. A
  minus also must not follow a currency symbol or `%`, so `5%-10%` is two
  positive numbers.
- A digit right after a group or decimal separator that follows a digit,
  or after an exponent marker (`1e-3`), never starts a mention: it goes
  on with the number before it.
- `NumberNotation::LatinUs` groups with `,` and marks the fraction with
  `.`; `LatinEu` the other way round. The caller chooses. OpenNLP can
  also pick the notation from a `java.util.Locale` with the JDK's locale
  data; that lookup is not ported, so a caller that knows the locale maps
  it to a notation itself.
- Grouping is strict: a group separator may follow at most three leading
  digits, and every further group has exactly three.
- The scan fails, rather than returning the prefix it read, when it stops
  at a group or decimal separator, a no-break, narrow no-break or thin
  space, an apostrophe, U+2019 or an Arabic decimal or thousands
  separator that is directly followed by a digit (`1,00,000`, `1.2.3`,
  `1.234,5` read in `LatinUs`), and when an exponent follows (`1e3`,
  `1.e-3`).
- With scaling on, the default, a `k`, `m`, `b` or `bn` in either case
  directly after the digits multiplies the value and must be followed by
  a word boundary; any other letter there rejects the number (`5x`). One
  `White_Space` character and then `thousand`, `million`, `billion` or
  `trillion` in any case also multiplies it, when the word ends at a
  boundary; otherwise the number ends before the whitespace.
- After a mention the scan goes on at its end; after a failed candidate,
  at the next character.

## Differences from OpenNLP

None are intended and none are known. The tests port the expectations
of the add-on's own tests and compare the scanner on generated texts
with a transcription of the Java over characters, so they need no JVM.
The Java works in UTF-16 units and this in bytes; every character the
scan compares against is in the Basic Multilingual Plane, so both stop
at the same characters. The character classes
come from the Unicode 18.0 tables of `turbo-unicode`, so a character
added to Unicode after the JDK's version can be classified differently
from that JVM (see `turbo_unicode::java::code_point`).

Behaviour that comes from the Java and is kept:

- A number with no scale marker is not checked for a boundary after it:
  `5` followed by U+0301 gives `5`, and without scaling `5x` gives `5`.
- Scale words are lowercased code point by code point with
  `Character.toLowerCase`, so `mİllion` (with U+0130) reads as `million`.

## Golden files

JVM goldens for this layer come from the goldens tool
(`analysis/goldens-tool`), as for the other OpenNLP ports
([analysis.md](analysis.md#golden-files)). Its `numbers` golden set is
planned and not in the tree yet.
