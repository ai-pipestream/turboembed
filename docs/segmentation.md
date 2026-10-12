# Sentence detection and tokenization

The segmentation crates write the first two layers of a document: the
sentences (`opennlp:sentences`) and the word tokens (`opennlp:tokens`).
They are ports of the OpenNLP 3.x sentence detectors and tokenizers onto
the analysis API (docs/analysis.md), reading OpenNLP's own models. A
model-free tokenizer gives tokens before any model is loaded, and the
UAX #29 tokenizer handles text in any script.

OpenNLP is the reference. Where the Rust side matches it, a golden file
made by the JVM proves it. Where it differs on purpose, this page says
where and why.

## Crates

| Crate | Path | Contents |
|---|---|---|
| `turbo-unicode` | `analysis/unicode/` | Character data: Java's character classes, the Unicode `White_Space` set, `Word_Break` and `Extended_Pictographic` properties, a UTF-16 code unit cursor over UTF-8 text |
| `turbo-maxent` | `analysis/maxent/` | OpenNLP model files: the zip container, the manifest, maxent and perceptron models and their evaluation, the dictionary XML |
| `turbo-sentdetect` | `analysis/sentdetect/` | The newline detector, the paragraph detector and the maxent sentence detector |
| `turbo-tokenize` | `analysis/tokenize/` | The simple and whitespace tokenizers, the UAX #29 word tokenizer, the maxent tokenizer, the dictionary detokenizer |
| `turbo-tokenize-layers` | `analysis/tokenize-layers/` | The tokenizers' own layer: the word type of each token |

They depend on the standard library, `turbo-analysis` and
`turbo-analysis-layers`, and on each other in that order. Nothing here
depends on the `turbo` core crate: the embedding model's subword
tokenizer is a separate annotator (`turbo-subwords`), so a program that
only segments text does not link the engine.

## Character classes and Java parity

OpenNLP's algorithms run on Java `char` values, which are UTF-16 code
units, and classify them with `java.lang.Character`. To produce the same
features and the same splits, the ports walk the text as code units too.
`turbo-unicode` provides:

- `Utf16Cursor`: steps over a `&str` one UTF-16 code unit at a time
  without copying the text. A character outside the Basic Multilingual
  Plane is two steps, a high surrogate then a low one; each step knows
  the byte offset of the character it belongs to. The cursor is the
  only place where "the character before position p" is defined, so
  every port agrees on it.
- Java's classes on a code unit or a code point: `is_letter` (general
  category `L*`), `is_digit` (`Nd` only), `is_upper_case` (`Lu` and
  `Other_Uppercase`), `is_alphabetic`, `is_whitespace` (Java's own
  definition), `is_space_separator` (`Zs`), and `to_lower_case` (the
  simple one-to-one mapping). A surrogate half is in no class, as in
  Java. Rust's `char::is_numeric` and `char::is_alphabetic` are wider
  than Java's `isDigit` and `isLetter`, so they are not used.
- The tables are generated from the Unicode 17.0 Character Database
  files under `analysis/unicode/ucd/` by `turbo-unicode-gen`, with each
  Java class derived the way the JDK defines it, and checked in as Rust
  source, so a build needs no data files. The JDK the golden files are
  made with (JDK 21) implements Unicode 15.0: a character added since is
  unassigned to it and in no class, while here it is classified by the
  17.0 data. For code points both assign, the answers differ for 34
  (U+0295, and the combining Latin small letters U+0363..U+036F and
  U+1DD3..U+1DE6), listed in the `java` module and checked by a test
  against a running JDK.

### Whitespace

OpenNLP has two definitions of whitespace, chosen by a global
`WhitespaceMode`: `Unicode`, the default since 3.0, is the Unicode
`White_Space` set (25 code points, `U+0085` included, `U+001C` to
`U+001F` excluded); `Legacy` is `Character.isWhitespace` or general
category `Zs`. The mode changes tokens, trimmed spans and model
features, so it is a parameter of every annotator here, never a global,
and a golden file records the mode it was made with. The default is
`Unicode`.

### Model features

A maxent model's predicates are Java strings read from the model file
in Java's modified UTF-8, which can hold a lone surrogate. The ports
generate features as modified UTF-8 bytes into one reusable buffer, and
the model's predicate map is keyed by those bytes, so a feature built
around a character outside the BMP matches exactly what the JVM looked
up. For text inside the BMP, modified UTF-8 is UTF-8.

## Model files

An OpenNLP `.bin` model is a zip holding `manifest.properties` (a Java
properties file), one `.model` entry (the classifier), and optional
artifacts such as `abbreviations.dictionary` (the dictionary XML).
`turbo-maxent` reads the zip (stored and deflated entries, with its own
inflate), the properties file as `java.util.Properties` does (escapes,
continuation lines, `\uXXXX`), and the model:

| Type string | Trainer | Evaluation |
|---|---|---|
| `GIS` | GIS maxent | sum the parameters of the context's predicates onto a uniform log prior, `exp`, normalize |
| `QN` | L-BFGS maxent | sum the parameters, then `exp(x - logsumexp)` |
| `Perceptron` | perceptron | sum the parameters, divide by the largest magnitude when it is above 1, `exp`, normalize |

A model's parameters are stored by predicate as (outcome, weight)
pairs. `MaxentModel::eval` takes the context's features and writes one
probability per outcome into a caller's `[f64]`; it allocates nothing.
Probabilities are `f64` end to end, as they are in Java, so a golden
comparison at 1e-9 holds. `NaiveBayes` models are refused with an
error that names the type; nothing here needs them.

The dictionary XML (`<dictionary case_sensitive="..."> <entry
attr="..."> <token>...</token> </entry>`) is read by a small reader
written for exactly that shape. A case-insensitive dictionary lowercases
each entry per code point with Java's `toLowerCase`, as `StringList`
does, and `contains` lowercases the probe the same way.

## Sentence detection

### `NewlineDetector`

Reads a `View`, writes `Sentences`. A sentence is each run between `\n`
or `\r` characters (not `U+2028`, `U+2029` or `U+0085`), trimmed of
whitespace in the configured mode, dropped if nothing remains. This is
`NewlineSentenceDetector` exactly; `str::lines` is not used because it
does not split on a lone `\r`.

### `ParagraphDetector`

Reads a `View`, writes `Structure` blocks of kind `Paragraph`. A line is
blank when it holds only Unicode `White_Space`. A paragraph is a maximal
run of non-blank lines; its span starts at the first non-whitespace
character of its first line and ends at the last non-whitespace
character of its last line, so a paragraph holds its inner line breaks
but no leading or trailing ones. OpenNLP has three slightly different
paragraph definitions (an exactly empty line in `ParagraphStream`, a
line that is empty after `String.trim` in `EmptyLinePreprocessorStream`,
Unicode line breaks in the whitespace normalizer); this is the one
`turbo:structure` uses.

### `SentenceDetector` (maxent)

Reads a `View`, writes `Sentences` and `Confidence<Sentences>`. It is
`SentenceDetectorME`:

1. The end-of-sentence scanner lists every position holding one of the
   model's end characters (`eosCharacters` in the manifest, else the
   language default: `. ! ?`, Portuguese adds `; : ( ) « » ' "`).
2. Leading parts of a run of end characters are skipped (`z.B.` is one
   candidate), unless the character after one is upper case.
3. For each candidate, `DefaultSDContextGenerator` builds the features:
   `sp` and `sn` for whitespace on either side, `eos=` the character,
   the prefix, previous, suffix and next words with their lengths,
   capitalization and abbreviation flags. Word boundaries here are
   whitespace in the configured mode, and `String.trim` semantics
   (every code unit at or below `U+0020`) apply to the words.
4. The model decides `s` or `n`. A split is refused when the candidate
   ends an entry of the abbreviation dictionary that starts after
   whitespace, an apostrophe or `(`.
5. Sentence starts are placed after the end character, or after the
   following whitespace run when `useTokenEnd` is set. Spans are
   trimmed of Unicode `White_Space` (this step always uses the Unicode
   set, as OpenNLP does since OPENNLP-205) and empty ones are dropped
   with their probability. The last sentence has probability 1.

The sentence detector works on code units for the features and on code
points for the spans, as the Java does. Its spans are UTF-8 byte
offsets on character boundaries by construction: an end character is
always a whole character.

## Tokenization

Every tokenizer reads a `View` and an optional `Sentences` layer. With
sentences, each sentence is tokenized on its own and the `Tokens` layer
carries the sentence of every token; without, the whole view is one
unit. All write `Tokens`; the maxent tokenizer also writes
`Confidence<Tokens>`, and the UAX #29 tokenizer also writes `WordTypes`.

### `SimpleTokenizer`

`SimpleTokenizer`: a token is a run of one character class (whitespace,
alphabetic, digit, other), and a run of "other" characters splits
wherever the character changes, so `...` is one token and `.,` is two.
Deviation: Java classifies each surrogate half on its own, so a
character outside the BMP (an emoji, a rare CJK character) becomes two
"other" tokens that each cover half a character. Half a character is
not a valid UTF-8 span, so here such a character is one "other"
character. This is noted as an OpenNLP finding.

The `keepNewLines` option emits each `\n` and `\r` as a token of its
own. (The Java compares the character with the general category
constants `LINE_SEPARATOR` and `LETTER_NUMBER`, whose values are 13 and
10, so it tests for CR and LF by accident. The result is the same.)
Two more Java defects in this option are fixed rather than copied: the
Java throws when a line break comes before the first token, and it
reuses a stale start offset for a line break that follows a space, so
the span of that line break points at the wrong place. Here every `\n`
and `\r` is a token at its real position. The goldens record what Java
produced, and the tests for the affected inputs check against the
corrected positions and say so.

### `WhitespaceTokenizer`

A token is a run of non-whitespace in the configured mode, with the same
`keepNewLines` option. It runs at memory speed on ASCII text: a byte
below 0x80 is whitespace only if it is one of the nine ASCII members.

### `WordTokenizer` (UAX #29)

Word boundaries by UAX #29 rules WB1 to WB999 over the Unicode 17.0
`Word_Break` property, with `Extended_Pictographic` for WB3c. The
segmenter is a state machine over the last two significant classes, the
current class and, where a rule needs it, the next significant class
and the regional indicator parity, so a document is one pass with no
backtracking. Segments that hold no letter, digit or script character
(punctuation, spaces) are not tokens. Each token's type is one of
alphanumeric, numeric, ideographic, hiragana, katakana, hangul,
southeast asian or emoji, by the script and classes of its characters,
and goes to the `WordTypes` layer. Tokens longer than `max_token_length`
UTF-16 code units (255 by default) are cut into pieces of that length,
never inside a character. The boundaries are checked against Unicode's
`WordBreakTest.txt`, every case.

### `TokenizerMe` (maxent)

`TokenizerME`: whitespace tokens are the units; a unit of one code unit,
or one the alphanumeric check accepts, is a token as it stands. In any
other unit, the model decides at every code unit position from the
second whether a token boundary falls there (`T` or `F`), with
`DefaultTokenContextGenerator`'s features: the prefix and suffix
strings, the two characters on each side with their classes (letter,
caps, digit, whitespace, end of sentence, quote, brackets), the
character pairs across the position, a flag for `&...;` entities and
one for a known abbreviation. A token's confidence is the product of the
decisions inside it. A split inside a unit that the abbreviation
dictionary lists keeps the unit whole and skips as many positions as
the unit has dots, as the Java does.

The alphanumeric check is the manifest's `alphaNumericPattern`, or the
language default when the manifest has none. Every pattern OpenNLP
ships has the form `^[characters]+$`; the check reads that form (single
characters and ranges) into a set and tests the unit by code point as
`java.util.regex` would. A pattern of any other form is refused when
the model loads, with the pattern in the error, rather than approximated.

Deviation: Java indexes a unit by code unit, so a split can fall between
the two halves of a character outside the BMP. Such a split is not a
valid UTF-8 boundary and is not taken here; the model is still evaluated
at that position so the probabilities stay identical.

### `Detokenizer`

`DictionaryDetokenizer`: the detokenizer dictionary (the same XML, with
an `operation` attribute per entry: `MOVE_LEFT`, `MOVE_RIGHT`,
`MOVE_BOTH`, `RIGHT_LEFT_MATCHING`) decides how each token attaches to
its neighbours; matching tokens (quotes) alternate right and left. The
detokenizer takes tokens as strings and gives the operations, or the
rebuilt text with an optional split marker. It is not an annotator: a
`Tokens` layer keeps its spans, so the text between tokens is never
lost and nothing needs rebuilding.

## Golden files

Each port is checked against files the OpenNLP 3.x JVM wrote on the same
input. The generator is a Java program under `analysis/golden/jvm/`,
with the command that ran it in its README; the files it wrote live in
each crate's `tests/golden/` directory. A file names the component, the
model and the whitespace mode, then the input text with `\n`, `\r`,
`\\` and `\uXXXX` escapes, then one line per span with its UTF-8 byte
offsets, its UTF-16 offsets and, when the component has one, its
probability printed with Java's `Double.toString`. The tests compare
spans exactly and probabilities within 1e-9.

The JVM is used for golden files only. Nothing here calls into it at
run time.

## OpenNLP findings

Things found while porting that OpenNLP could change, kept here so they
reach the OpenNLP side:

- `SimpleTokenizer` splits a character outside the BMP into two tokens
  of half a character each, because it classifies code units.
- `SimpleTokenizer.isLineSeparator` and `WhitespaceTokenizer.isLineSeparator`
  compare a `char` with `Character.LINE_SEPARATOR` and
  `Character.LETTER_NUMBER`, which are general category byte constants
  (13 and 10), not characters. The test is for CR and LF only by
  coincidence.
- `TokenizerME` can place a split between surrogate halves for the
  same reason.
- `SimpleTokenizer` with `keepNewLines` throws an exception when the
  text starts with a line break (or any line break comes before the
  first token), because it tries to close a token that was never
  opened.
- `SimpleTokenizer` with `keepNewLines` reports a wrong span for a line
  break that follows a space: the start offset is left over from the
  previous token instead of being set to the line break's position.
- `Dictionary(InputStream)` inserts the entries before its
  `isCaseSensitive` field is assigned, so every entry is stored as if the
  dictionary were case-insensitive. In a case-sensitive dictionary, entries
  that differ only in case collapse to the first one: with `Dr.` and
  `dr.` loaded, `contains("dr.")` is false. The Rust reader reproduces
  this so lookups match; a test records it.
- `DefaultSDContextGenerator` tests `Character.isUpperCase` on the
  first code unit of a word, so a word starting with an upper case
  letter outside the BMP (Deseret, Osage, mathematical letters) is
  never `xcap`.
