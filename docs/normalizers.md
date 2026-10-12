# Text normalizers

The normalizers rewrite a document's text into a new view without
touching the source: whitespace folded, quotes straightened, Unicode
composed, URLs replaced by placeholders, line-broken words joined. Every
view carries an alignment back to the text it was made from, so a span
found in any view maps to exact source bytes (docs/analysis.md, "Views").

They port what Apache OpenNLP 3.x has in `opennlp.tools.util.normalizer`
and in its add-ons. OpenNLP is the reference for output text; where this
implementation differs on purpose, the difference is listed under
"Differences from OpenNLP" in the section of the normalizer concerned.

## Crates

| Crate | Path | Contents |
|---|---|---|
| `turbo-unicode` | `analysis/unicode/` | Unicode 18.0 character data: general category, properties, scripts, case mapping and folding, normalization forms, confusables, emoji sequences. Tables generated from the UCD files in `analysis/unicode/ucd/` by `turbo-unicode-gen`. |
| `turbo-unicode-gen` | `analysis/unicode-gen/` | The table generator. |
| `turbo-normalize` | `analysis/normalize/` | The normalizers, the `Normalizer` trait, the pipeline step adapter and the chain. |
| `turbo-normalize-layers` | `analysis/normalize-layers/` | The layers normalizers write besides views: `Placeholders` (`turbo:placeholders`), `Emoji` (`turbo:emoji`). |

### turbo-unicode

`turbo-unicode` holds the Unicode 18.0.0 data the normalizers need and
the operations built on it. It depends on the standard library only and
reads nothing at run time.

**How the tables are made.** `turbo-unicode-gen` reads the Unicode
Character Database files in `analysis/unicode/ucd/` (their sources and
digests are in `analysis/unicode/ucd/README.md` and `analysis/unicode/ucd/SHA256SUMS`) and writes one Rust
file per table family into `analysis/unicode/src/tables/`: `category.rs`,
`props.rs`, `script.rs`, `case.rs`, `normalize.rs`, `confusables.rs`,
`emoji.rs`, `word_break.rs` and `mod.rs`. Each file starts with a notice naming its input
files and the Unicode version. The generator parses the files by hand and
stops with the file name and line number on any line it does not
understand. Its output is deterministic and already in rustfmt layout, so
the test `analysis/unicode/tests/generated.rs` regenerates the tables into a temporary
directory and fails when they differ from the committed ones by a single
byte. To move to a newer Unicode release, replace the files in `analysis/unicode/ucd/`,
update `UNICODE_VERSION` in `analysis/unicode/src/lib.rs` and run:

```sh
cargo run -p turbo-unicode-gen -- analysis/unicode/ucd analysis/unicode/src/tables
```

The generator reads the version from the file headers; the crate does not
compile when `UNICODE_VERSION` and the tables disagree.

**Lookup structure.** A value stored for every code point (general
category, script, combining class and normalization quick check values,
binary property bits) is a two-stage table: `INDEX` gives, for each block
of 256 code points, the number of a stored block, and `BLOCKS` holds the
distinct stored blocks. A lookup is two array reads,
`BLOCKS[INDEX[cp >> 8] << 8 | cp & 0xFF]`. A mapping (case mappings,
case folding, decompositions, compositions, confusable prototypes, emoji
names) is a sorted array of keys searched by binary search, with the
values in a second array at the same positions. The property bits include
one bit per mapping that says whether a code point has an entry, so the
search runs only for code points that have one. Decompositions share one
pool of characters. Text functions copy ASCII stretches eight bytes at a
time and look characters up only outside them.

**Modules.**

| Module | Contents |
|---|---|
| `category` | `GeneralCategory` and `general_category`. |
| `props` | White_Space, Default_Ignorable_Code_Point, Dash, Quotation_Mark, Alphabetic, Lowercase, Uppercase, Cased, Case_Ignorable, Regional_Indicator, Variation_Selector and the emoji properties. |
| `script` | `Script` (the Scripts.txt values; `OldItalic` for `Old_Italic`, which `java.lang.Character.UnicodeScript` calls `OLD_ITALIC`) and `script`. |
| `java` | `java.lang.Character` methods (`isWhitespace`, `isLetter`, `digit`, simple case mappings) and `String.toLowerCase(Locale.ROOT)`, on Unicode 18.0 data. |
| `java::code_point` | The `int` overloads of `java.lang.Character` the segmentation ports use (`getType` with Java's numbering, `isLetter`, `isDigit`, `isUpperCase`, `isAlphabetic`, `isWhitespace`, `isSpaceChar`, `toLowerCase`, `UnicodeScript.of`) on a code point given as `u32`, which may be a lone surrogate code unit. Read from the same tables as the rest of the crate. |
| `whitespace` | `WhitespaceMode`: OpenNLP's legacy (`Character.isWhitespace` or `Zs`) and Unicode (`White_Space`) whitespace. |
| `word_break` | `WordBreak` (UAX #29 Word_Break, from WordBreakProperty.txt, in the order of OpenNLP's enum), `word_break`, `is_extended_pictographic` and `is_regional_indicator`, on `u32` code points. OpenNLP 3.x bundles WordBreakProperty-17.0.0.txt; against these 18.0 tables 558 code points differ, all newly assigned in 18.0 and `Other` in 17.0 (515 `ALetter`, 37 `Extend`, 6 `Katakana` here), so golden mismatches on those characters are expected. |
| `utf16` | `Utf16Cursor`, which steps over UTF-8 text one UTF-16 code unit at a time, and offset conversions between the two. |
| `modified_utf8` | Java's modified UTF-8, the encoding of the strings in OpenNLP model files. |
| `casefold` | Full and simple case folding and their Turkic variants (CaseFolding.txt). |
| `normalize` | NFC, NFD, NFKC, NFKD, quick check, single-level decompositions, composition, and `for_each_run`, which reports the ranges a normalization form changes. |
| `confusables` | UTS #39 prototypes and skeletons. |
| `emoji` | `scan` (UTS #51 possible emoji), the RGI set, CLDR names, groups and subgroups, and flag region codes. |
| `chars` | ASCII helpers. |

**Normalization runs.** `for_each_run` splits text before each character
with combining class 0 whose quick check value for the form is Yes. No
character before such a point reorders or composes with one after it, so
each run normalizes on its own. A run whose characters all pass the quick
check and are in canonical order is reported as unchanged without being
normalized; consecutive unchanged runs are reported as one range. A
normalizer that writes a view calls `keep` for an unchanged range and
`replace` for a changed run, so the alignment maps each changed run to its
exact source bytes.

**Emoji scanning.** `emoji::scan` reports, left to right: a pair of
regional indicators (a flag); `0`-`9`, `#` or `*` with an optional U+FE0F
and U+20E3 (a keycap; these characters are never reported without
U+20E3); otherwise an element (a character with the Emoji property, or an
Extended_Pictographic character followed by U+FE0F) with an optional
U+FE0F, an optional skin tone modifier and an optional terminated tag
sequence, joined by U+200D to further elements. A character followed by
U+FE0E is not reported. Characters that are emoji but shown as text by
default, such as U+00A9 and U+2639, are reported without U+FE0F; callers
that want only emoji-presentation characters check
`props::emoji_presentation` and look for U+FE0F. Every row of
`emoji-test.txt` is found as exactly one sequence.

**Differences from the JDK.** The `java` functions follow the JDK's
definitions but use Unicode 18.0 data; the JDK 21 tables are Unicode 15.0,
so characters assigned since then get different answers from a JVM.
Where both assign a code point, JDK 21 and these tables differ on 35
code points (U+0295 is `Lo` here and `Ll` in JDK 21; 33 combining Latin
small letters are `Alphabetic` here and not in JDK 21; U+1171E AHOM
CONSONANT SIGN MEDIAL RA is `Mc` here and `Mn` in JDK 21, a Unicode 16.0
change). `analysis/unicode/tests/jdk.rs` lists them and, with
`TURBO_TEST_JDK` set, compares the general category, the six class tests,
the simple lowercase mapping and the script from `java::code_point` with
a running JDK 21 for every code point from U+0000 to U+10FFFF, skipping
those the JDK reports as unassigned. It fails on any other difference, on
a JDK other than 21, and when `java` or `javac` is missing.
`java::digit` accepts decimal digits only (`Character.digit(c, 10)`
without the fullwidth and ASCII letters of larger radixes).
`java::to_lowercase` implements Final_Sigma as Unicode section 3.13
defines it; on the cases in `analysis/unicode/tests/java.rs` it gives the same output as
OpenJDK 21.
`java::lowercase_at` gives the mapping `to_lowercase` applies to one
character, with U+03A3 decided from the text around it, so a normalizer
can lowercase character by character and replace only the characters
that change (`CaseFold::lowercase` does).

**Tests.** `analysis/unicode/tests/normalization.rs` checks every line of
NormalizationTest.txt (all five columns and the standard invariants), that
every code point not listed in its part 1 is unchanged by every form, that
`is_normalized` agrees with `normalize`, and that splitting into runs gives
the same result as normalizing the whole text at once. `analysis/unicode/tests/ucd_data.rs`
parses UnicodeData.txt, CaseFolding.txt, the property files, Scripts.txt
and confusables.txt again and compares every code point. `analysis/unicode/tests/emoji.rs`
checks every row of emoji-test.txt and every RGI sequence, the scanning
rules above, and region decoding with the cases of OpenNLP's
`EmojiFlagsTest`. `analysis/unicode/tests/java.rs` checks whitespace, letters, digits and
lowercasing against OpenJDK 21 output. `analysis/unicode/tests/bench.rs` (ignored by
default) measures normalization throughput:

```sh
cargo test --release -p turbo-unicode --test bench -- --ignored --nocapture
```

## How a normalizer works

A normalizer implements `Normalizer`: an id, and `normalize(text, out)`,
which copies `text` into a `ViewBuilder`, calling `keep` for what stays
and `replace` for what changes. The builder records the alignment as the
text is written, so no normalized text exists without one.

`NormalizeStep<N>` makes any normalizer a pipeline step that reads a
`View` and writes a `View`; the instance names are bound where the step
is added, so the same step can write `nfkc` on one pipeline and
`folded` on another. `normalize` and `apply` run one on a string, for
callers outside a pipeline. `Chain` runs several in order and composes
their alignments into one, as OpenNLP's `TextNormalizer.buildAligned()`
does; the intermediate texts are dropped.

## The normalizers

(Each normalizer's section follows: what it does, its OpenNLP
counterpart, differences from OpenNLP, what the tests check.)


Each normalizer is described here in the pull request that adds it.

## Notes for OpenNLP

(Things found while porting that OpenNLP could fix or improve. Each
pull request that adds a normalizer adds its notes here.)

## Tests

`cargo test -p turbo-unicode -p turbo-normalize -p turbo-normalize-layers`.
