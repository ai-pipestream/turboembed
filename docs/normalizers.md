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
| `turbo-encoding` | `analysis/encoding/` | Decoders, encoding detection and the mojibake repair under `Mojibake` (docs/analysis.md, "Source bytes"). |
| `turbo-normalize-layers` | `analysis/normalize-layers/` | The layers normalizers write besides views: `Placeholders` (`turbo:placeholders`), `Emoji` (`turbo:emoji`), `Artifacts` (`turbo:artifacts`), `Noise` (`turbo:noise`), `Corrections` (`turbo:corrections`). |
| `turbo-normalize-layers` | `analysis/normalize-layers/` | The layers normalizers write besides views: `Placeholders` (`turbo:placeholders`), `Emoji` (`turbo:emoji`), `Artifacts` (`turbo:artifacts`), `Noise` (`turbo:noise`). |
| `turbo-normalize-layers` | `analysis/normalize-layers/` | The layers normalizers write besides views: `Placeholders` (`turbo:placeholders`), `Emoji` (`turbo:emoji`), `Assets` (`turbo:assets`). |

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
| `props` | White_Space, Default_Ignorable_Code_Point, Dash, Quotation_Mark, Alphabetic, Lowercase, Uppercase, Cased, Case_Ignorable, Regional_Indicator, Variation_Selector, the emoji properties, and `complex_context` (Line_Break=SA, from LineBreak.txt, the one Line_Break value the segmenters use). |
| `script` | `Script` (the Scripts.txt values; `OldItalic` for `Old_Italic`, which `java.lang.Character.UnicodeScript` calls `OLD_ITALIC`) and `script`. |
| `java` | `java.lang.Character` methods (`isWhitespace`, `isLetter`, `digit`, simple case mappings) and `String.toLowerCase(Locale.ROOT)`, on Unicode 18.0 data. |
| `java::code_point` | The `int` overloads of `java.lang.Character` the segmentation ports use (`getType` with Java's numbering, `isLetter`, `isDigit`, `isUpperCase`, `isAlphabetic`, `isWhitespace`, `isSpaceChar`, `toLowerCase`, `UnicodeScript.of`) on a code point given as `u32`, which may be a lone surrogate code unit. Read from the same tables as the rest of the crate. |
| `whitespace` | `WhitespaceMode`: OpenNLP's legacy (`Character.isWhitespace` or `Zs`) and Unicode (`White_Space`) whitespace. |
| `word_break` | `WordBreak` (UAX #29 Word_Break, from WordBreakProperty.txt, in the order of OpenNLP's enum), `word_break`, `is_extended_pictographic` and `is_regional_indicator`, on `u32` code points. OpenNLP 3.x bundles WordBreakProperty-17.0.0.txt; against these 18.0 tables 558 code points differ, all newly assigned in 18.0 and `Other` in 17.0 (515 `ALetter`, 37 `Extend`, 6 `Katakana` here), so golden mismatches on those characters are expected. |
| `grapheme` | `GraphemeBreak` (UAX #29 Grapheme_Cluster_Break, from GraphemeBreakProperty.txt), `IndicConjunctBreak` (Indic_Conjunct_Break, from DerivedCoreProperties.txt), and `grapheme_boundaries`, the extended grapheme cluster boundaries of a text by rules GB1 to GB999 of Unicode 18.0, checked against every case of GraphemeBreakTest.txt. |
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
`java::to_lowercase` writes final sigma under the JDK's own condition, not
Unicode's Final_Sigma: the JDK walks the word the sigma is in, as its word
`BreakIterator` for the root locale delimits it, backwards to the first
cased character and forwards to the word's end, passing over every other
character, where Unicode skips only case-ignorable ones; and its cased
characters are the three cased letter categories plus a fixed list (U+02B0
to U+02B8, U+02C0 to U+02C1, U+02E0 to U+02E4, U+0345, U+037A, U+1D2C to
U+1D61, the Roman numerals and the circled letters), not the current
Other_Uppercase and Other_Lowercase; U+0295 counts as cased too, being Ll
in JDK 21's Unicode 15.0 and Lo since 16.0. That iterator's rules predate
UAX #29 and were modelled by probing JDK 21, never by reading it
(`java::word_class`, with the class of every code point in
`analysis/unicode/src/java/word_classes.rs`): letters and digits chain
into one word; a nonspacing or enclosing mark attaches to the character
before it, except after a danda or percent sign that joined a word, after
a control, and at the start of the text, where the mark is a word of its
own and the word starts over; a format character is invisible; `.`, `'`,
`"`, a dash, a connector, the soft hyphen or U+2027 stays in the word
between two letters, and `.`, `'`, `"`, `,` or U+066B between two digits,
when the character right after the joiner is one (a mark there breaks the
word); `.`, `#` and a currency sign start a number; `%`, `&`, U+00A2,
U+066A, U+2030 and U+2031 end one; a danda follows letters and digits may
follow it; the kana and the kanji up to U+9FA5 are words of their own;
supplementary characters have the classes of their categories (U+1D7CE is
a digit, U+E0020 a format character, U+20000 a letter), and `isBoundary`
right after a supplementary character answers true unless that character
starts the text, as the iterator resynchronizes from inside the surrogate
pair, while the word goes on as the rules say; the model lays that quirk
over the rules. So "ΑΣ1" and "Α1Σ" end in final sigma, as does "ΑΣ।ͅ" (the
mark after the danda is not in the sigma's word), and "ΑΣ1a", "ΑΣ.a" and
"ΑΣ\u0e02a" (Thai letters, then a cased one, all one word) do not; the
differential fuzzing found the last on the lemmatizer (`docs/analysis.md`,
"Differential fuzzing"). `analysis/unicode/tests/java.rs` holds the cases,
and `analysis/unicode/tests/jdk_sigma.rs`, with `TURBO_TEST_JDK` set,
checks the class of every code point and the decision on six hundred
thousand random strings (short ones from the Basic Multilingual Plane and
from every plane, and ones of up to twenty characters from an alphabet of
dandas, marks, joiners, controls and supplementary characters next to each
other) against a running JDK 21; both agree with OpenJDK 21.0.12.1 on
every one, over several seeds and more than a million strings. The
decision reads only the text around the sigma, without allocating:
backwards to the nearest cased character and on to the nearest character
whose class fixes what the word ends in whatever came before it, from
where the rules are replayed, and forwards to the word's end.
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


### Character tables (`tables.rs`) and the shared pass (`substitute.rs`)

The whitespace, dash, quote, bullet, ellipsis and invisible character
normalizers work from fixed lists in `turbo_normalize::tables`
(`WHITESPACE`, `LINE_BREAKS`, `DASHES`, `MATH_MINUS`, `SINGLE_QUOTES`,
`DOUBLE_QUOTES`, `CORNER_BRACKETS`, `BULLETS`, `ELLIPSES`, `INVISIBLE`),
each sorted by code point, with a predicate or fold function for each
(`is_whitespace`, `is_line_break`, `is_horizontal_whitespace`,
`is_default_dash`, `is_math_minus`, `quote_fold`, `is_bullet`,
`ellipsis_fold`, `is_invisible`). They start from OpenNLP's lists
(`UnicodeWhitespace`, `UnicodeDash`, the sets in
`QuoteCharSequenceNormalizer`, `BulletCharSequenceNormalizer`,
`EllipsisCharSequenceNormalizer` and `InvisibleCharSequenceNormalizer`)
and are checked against the Unicode 18.0 data in `analysis/unicode/ucd/`
by `analysis/normalize/tests/unicode_tables.rs`:

- `WHITESPACE` (25 code points) is exactly the `White_Space` property.
  `LINE_BREAKS` (LF, VT, FF, CR, NEL, U+2028, U+2029) is the set UAX #14
  gives the mandatory break classes BK, CR, LF and NL. Both equal
  OpenNLP's lists.
- `DASHES` (31 code points) is exactly the `Dash` property, and
  `MATH_MINUS` its three members in category Sm. Both equal OpenNLP's
  lists.
- The quote tables are the `Quotation_Mark` property without the two
  ASCII marks and the CJK corner brackets, plus U+02BC. That is
  OpenNLP's list plus U+2E42.
- `BULLETS`, `ELLIPSES` and `INVISIBLE` add characters to OpenNLP's
  lists; the sections below give each addition and the reason.

The tables are kept as literal lists rather than looked up through
`turbo_unicode::props` at run time so that the fold is a `match` the
compiler turns into range checks, and the tests catch any change in a
later Unicode version.

The predicates are `match` expressions over ranges. For the 25-entry
whitespace table a `match` took 1.3 ms per MB of mixed text against
2.7 ms for `binary_search` over the sorted list (release build, the
ignored test `tables::tests::whitespace_membership_timing`).

`substitute(text, out, f)` and `remove(text, out, f)` are the one pass
shared by the folds that change single code points. They skip ASCII
bytes without decoding them and ask `f` only about non-ASCII
characters, because none of the tables that use them has an ASCII
member (the hyphen-minus is in the dash table but maps to itself). Text
that is all ASCII costs one byte scan and produces no edits.

Every edit these normalizers make is written through `ViewBuilder`,
which records each `replace` call as its own edit (issue #265). Two
neighbouring replaced characters (two Arabic-Indic digits, a closing
quote followed by an ellipsis in a chain) are therefore two edits, and a
span inside the run maps back to exactly the source characters it
covers, as OpenNLP's `Alignment`, which records a source range per
output character, maps it too.

### Whitespace

`Whitespace` (id `turbo:normalize:whitespace`) collapses each run of
whitespace (members of the `UnicodeWhitespace` table) to one character
and deletes the runs at the start and end of the text. `WhitespaceMode`
says what an inner run becomes:

- `Collapse` (`Whitespace::collapse()`): one space.
- `KeepLineBreaks` (`Whitespace::keep_line_breaks()`): "\n" when the run
  contains a line break, one space otherwise.
- `KeepParagraphs` (`Whitespace::keep_paragraphs()`): "\n" when the run
  contains two or more line breaks, counting CR LF as one, and one space
  otherwise. A single line break inside prose is a hard wrap and is
  unwrapped; a blank line is a paragraph boundary and is kept.

The whitespace and line-break sets are the Unicode properties described
above and match OpenNLP. A text that is all whitespace becomes empty.
Each run is one edit that
maps to the whole source run, so the "\n" that stands for "\n\n\t" maps
back to all three characters. A run that already equals its replacement
(one space between two words, or one "\n" in `KeepLineBreaks`) is not an
edit, so ordinary text comes back borrowed from `apply`.

OpenNLP counterparts: `WhitespaceCharSequenceNormalizer`,
`LineBreakPreservingWhitespaceCharSequenceNormalizer` and
`ParagraphPreservingWhitespaceCharSequenceNormalizer`, built on
`CharClass.collapseAligned`, `collapsePreservingAligned`,
`collapseParagraphPreservingAligned`, `trimAligned` and
`countLogicalLineBreaks`.

OpenNLP also has a "legacy" whitespace definition (`WhitespaceMode` in
`opennlp.tools.util`: `Character.isWhitespace` or general category Zs),
selected by a system property for the tokenizers so that models trained
with OpenNLP 1.x and 2.x keep their behaviour. It is a tokenizer setting
and is not part of this normalizer.

Differences from OpenNLP:

- OpenNLP collapses, then trims the collapsed text, as two aligned passes
  whose alignments it composes. This normalizer does both in one pass:
  a leading or trailing run is replaced by "", an inner run by " " or
  "\n". The output text is the same, so this is not a difference in
  output.
- `KeepParagraphs` writes a paragraph boundary as a single "\n", as
  OpenNLP does. Running it twice therefore joins the paragraphs: the
  second run sees one line break and turns it into a space. The tests
  record this.

Tests (`analysis/normalize/tests/whitespace.rs`, `analysis/normalize/tests/folds_properties.rs`, unit tests in
`whitespace.rs` and `tables.rs`): every expectation of OpenNLP's
`LineBreakPreservingWhitespaceCharSequenceNormalizerTest`,
`ParagraphPreservingWhitespaceCharSequenceNormalizerTest`, the whitespace
cases of `UnicodeCharSequenceNormalizerTest` and
`AlignedNormalizerPipelineTest`, with UTF-16 offsets rewritten as UTF-8
byte offsets; every table member and every line break in each mode; the
edits of a mixed text exactly; and 3000 generated texts per mode compared
with a slow reference that follows OpenNLP's two-pass collapse then trim
over a character vector, with the alignment checks described under
"Alignment checks" below. One OpenNLP expectation, that the empty output
span of an all-whitespace text maps back to a valid empty span, is the
ignored test `empty_output_span_at_a_deletion`: `Alignment::span_to_parent`
in turbo-analysis returns a span whose start is after its end for an empty
span that sits on a deletion, and `Span::new` panics.

### Dashes

`Dashes` (id `turbo:normalize:dashes`) replaces each dash of OpenNLP's
default dash set (the `UnicodeDash` table without U+207B SUPERSCRIPT
MINUS, U+208B SUBSCRIPT MINUS and U+2212 MINUS SIGN) with U+002D. The
hyphen-minus is a member and is kept as it is, so it is never an edit.
`Dashes::with_math_minus()` also folds the three minus signs; the default
leaves them because in arithmetic such as "5 U+2212 3" the minus sign
is an operator.

OpenNLP counterpart: `DashCharSequenceNormalizer` (`CharClass.dashes()`).
The default matches it. `with_math_minus` is an addition.

Tests (`analysis/normalize/tests/punctuation_folds.rs`, `analysis/normalize/tests/folds_properties.rs`): the
dash cases of `UnicodeCharSequenceNormalizerTest` and
`AlignedNormalizerPipelineTest` (a four-byte supplementary dash maps back
to its four bytes), every table entry in both configurations, characters
outside the table (U+2043, U+2E3C) left alone, and generated texts against
a reference fold.

### Quotes

`Quotes` (id `turbo:normalize:quotes`) replaces the eight single quotation
marks and apostrophes of `SINGLE_QUOTES` with U+0027 and the ten double
quotation marks of `DOUBLE_QUOTES` with U+0022.

OpenNLP counterpart: `QuoteCharSequenceNormalizer`. OpenNLP runs the
single set and then the double set as two passes and composes their
alignments; the sets do not overlap and neither replacement is a member,
so the one pass here writes the same text.

Differences from OpenNLP: U+2E42 DOUBLE LOW-REVERSED-9 QUOTATION MARK has
the `Quotation_Mark` property and is folded to U+0022; OpenNLP leaves it.
U+2E42 "Wort" U+201C becomes "\"Wort\"" here. The other
`Quotation_Mark` characters OpenNLP leaves out, the CJK corner brackets
U+300C to U+300F, U+FE41 to U+FE44, U+FF62 and U+FF63, are left alone
here too: they come in opening and closing pairs that also mark titles
and emphasis, and an ASCII quotation mark would lose which side of the
pair a bracket is.

Tests: every expectation of the quote tests in `SetBasedNormalizerTest`
and `AlignedNormalizerPipelineTest`, every table entry, prime marks and
the grave accent left alone, the shared edit of two neighbouring quotes,
U+2E42 folded and every corner bracket left alone, and generated texts
against a reference fold.

### Bullets

`Bullets` (id `turbo:normalize:bullets`) replaces U+2022 BULLET, U+2023
TRIANGULAR BULLET, U+2043 HYPHEN BULLET, U+204C BLACK LEFTWARDS BULLET,
U+204D BLACK RIGHTWARDS BULLET, U+2219 BULLET OPERATOR, U+25D8 INVERSE
BULLET, U+25E6 WHITE BULLET, U+29BE CIRCLED WHITE BULLET and U+29BF
CIRCLED BULLET with one space each. U+00B7 MIDDLE DOT is left alone:
Catalan writes it inside words. Run `Whitespace` afterwards to collapse
the space with its neighbours.

OpenNLP counterpart: `BulletCharSequenceNormalizer`.

Differences from OpenNLP: OpenNLP folds five bullets. The five additions
are the other characters whose Unicode name calls them a bullet and that
are used as list markers, so U+204C "a" U+25D8 "b" becomes " a b" here and
is unchanged in OpenNLP. The heart and floral bullets (U+2619, U+2765,
U+2767) are ornaments and are left alone, as is U+1F7DB BULLET IN DOUBLE
CIRCLE, a geometric symbol.

Tests: the bullet cases of `SetBasedNormalizerTest`, every table entry,
the additions, the heart bullets left alone, a check that every entry's
Unicode name contains BULLET, and generated texts against a reference fold.

### Ellipsis

`Ellipsis` (id `turbo:normalize:ellipsis`) replaces U+2026 HORIZONTAL
ELLIPSIS with "...", U+2025 TWO DOT LEADER with "..", U+2024 ONE DOT
LEADER with ".", U+FE19 PRESENTATION FORM FOR VERTICAL HORIZONTAL
ELLIPSIS with "..." and U+FE30 PRESENTATION FORM FOR VERTICAL TWO DOT
LEADER with "..". Each dot maps back to the one source character. U+22EF
MIDLINE HORIZONTAL ELLIPSIS is a mathematical symbol without a
decomposition and is left alone.

OpenNLP counterpart: `EllipsisCharSequenceNormalizer`.

Differences from OpenNLP: OpenNLP expands U+2025 and U+2026 only. The
three additions expand to their compatibility decompositions, the same
text NFKC gives, so "a" U+2024 "b" becomes "a.b" and a vertical ellipsis
becomes "..." instead of passing through unchanged.

Tests: the ellipsis cases of `SetBasedNormalizerTest` and
`AlignedNormalizerPipelineTest` (each dot and the whole expansion map back
to the ellipsis), the additions, a check that every expansion is the
character's compatibility decomposition, the chain of quotes and ellipsis from
`substitutionFoldsComposeInAnAlignedPipeline`, and generated texts.

### Digits

`Digits` (id `turbo:normalize:digits`) replaces every decimal digit
(general category Nd) that is not ASCII with the ASCII digit of the same
value: U+0663 ARABIC-INDIC DIGIT THREE becomes "3", U+FF15 FULLWIDTH DIGIT
FIVE becomes "5", U+1D7CE MATHEMATICAL BOLD DIGIT ZERO becomes "0". The
value comes from `turbo_unicode::java::digit`, which follows
`Character.digit(c, 10)`. For radix 10 `Character.digit` maps no
character outside Nd: the Latin and fullwidth letters it also accepts
have values of 10 and above, which radix 10 rejects. Superscripts,
fractions, Roman numerals, circled digits and U+3007 IDEOGRAPHIC NUMBER
ZERO are not Nd and are left alone.

OpenNLP counterpart: `DigitCharSequenceNormalizer`.

Differences from OpenNLP: the digit data is Unicode 18.0. A JVM folds the
digits of the Unicode version it ships (15.0 for JDK 21), so digits added
in Unicode 16.0 and later are folded here and are not folded by OpenNLP
on JDK 21. For characters inside the 15.0 repertoire the output is the same.

Tests (`analysis/normalize/tests/digits.rs`): the digit cases of `SetBasedNormalizerTest` and
`AlignedNormalizerPipelineTest`, fullwidth letters left alone, every code
point checked against the Nd rows and decimal values of UnicodeData.txt,
the shared edit of neighbouring digits, and generated texts against a
reference built from the zero of each digit block.

### Invisible characters

`Invisible` (id `turbo:normalize:invisible`) deletes the 26 code points of
`INVISIBLE`: U+00AD SOFT HYPHEN, U+061C ARABIC LETTER MARK, U+200B ZERO
WIDTH SPACE, U+200E and U+200F (the directional marks), U+202A to U+202E
(embeddings, pop and overrides), U+2060 WORD JOINER, U+2061 to U+2064
(invisible operators), U+2066 to U+2069 (isolates), U+206A to U+206F
(deprecated format characters) and U+FEFF (byte order mark). Each deletion is its own edit to the empty string. The zero width joiner and non-joiner and the
variation selectors are kept: they change shaping and rendering, and the
joiner holds emoji ZWJ sequences together. U+180E MONGOLIAN VOWEL
SEPARATOR is kept for the same reason: it selects the form of the vowel
after it.

OpenNLP counterpart: `InvisibleCharSequenceNormalizer`.

Differences from OpenNLP: U+206A to U+206F (inhibit and activate
symmetric swapping, inhibit and activate Arabic form shaping, national
and nominal digit shapes) are deleted here and kept by OpenNLP. They are
format characters that were deprecated in Unicode 3.0 and have no effect
on display; like the other members they are `Default_Ignorable_Code_Point`
characters in category Cf. "a" U+206A "b" U+206F becomes "ab". Other
default-ignorable characters stay: the Hangul fillers are letters, the
tag characters build emoji flag sequences, and the variation selectors
choose glyphs.
Alignment: OpenNLP's `Alignment` lists the source range of each output
character and nothing for a deleted one, so two deleted neighbours are
one gap in it, where each deletion is its own edit here; the text and
every mapped span are the same.

Tests: every expectation of the invisible character tests in
`SetBasedNormalizerTest` and `AlignedNormalizerPipelineTest`, every table
entry, the additions, U+180E kept, a check that every entry is a
`Default_Ignorable_Code_Point` format character, the chain of invisible
characters, whitespace and dashes from
`pipelineComposesStripInvisibleWhitespaceAndDashesBackToOriginal` and its
line-break and paragraph variants, and generated texts against a
reference filter.

### Text artifacts (`analysis/normalize/src/artifacts/mod.rs`)

`ArtifactDetector` (an annotator, id `turbo:artifacts`) reads a `View` and
writes an `Artifacts` layer (`turbo:artifacts`): a span and an
`ArtifactKind` for every damaged or hidden character sequence. It does not
write a view; the spans are UTF-8 byte offsets of the view it read, in text
order and never overlapping (the layer's `check` refuses overlaps, empty
spans and a missing kind). `ArtifactKind::name` gives OpenNLP's type
string ("replacement", "mojibake", ...), which the golden files use.

The default detector reports OpenNLP's nine kinds, by OpenNLP's rules, on
code points:

- maximal runs of one class: U+FFFD (`ReplacementChar`); C0 and C1
  controls that are not `White_Space` (`Control`, so tab, line feed and
  U+0085 are not reported); noncharacters (`Noncharacter`); private-use
  code points (`PrivateUse`); U+202A..U+202E, U+2066..U+2069, U+200E,
  U+200F and U+061C (`BidiControl`);
- zero-width characters (U+200B, U+200C, U+200D, U+2060, U+FEFF): a run of
  two or more is reported; a single one is left alone between letters, with
  combining marks passed over on both sides (ZWNJ in Persian and
  Malayalam, ZWJ in Sinhala, ZWSP in Thai), and a single ZWJ is left alone
  next to an `Extended_Pictographic` character or U+FE0F (emoji ZWJ
  sequences);
- runs from the Tags block (`UnicodeTag`), except a run after U+1F3F4 that
  spells a region or subdivision code and ends with CANCEL TAG (the
  England, Scotland and Wales flags), by the syntax of UTS #51 and UTS #35;
- `Mojibake`: a maximal run of non-ASCII characters that Windows-1252 can
  encode (ISO-8859-1 for the C1 controls) whose bytes are strictly
  well-formed UTF-8 with at least one multi-byte sequence (RFC 3629: no
  overlong form, no surrogate, nothing past U+10FFFF, nothing cut off).
  A mojibake finding wins over the class findings it overlaps; the parts
  of a class run outside it are still reported.

`ArtifactDetector::builder()` selects kinds (`only`, `with`, `without`),
as OpenNLP's `CursorArtifactDetector(Set)` does.

OpenNLP counterpart: `CursorArtifactDetector`, `ArtifactAnnotator` and
`TextArtifact` in the text-artifacts add-on.

Differences from OpenNLP:

- Character data is Unicode 18.0 (`turbo_unicode`): `White_Space`,
  `Extended_Pictographic`, the letter test (`Character.isLetter`) and the
  combining-mark test. OpenNLP reads the JDK's tables (Unicode 15.0 in JDK
  21) and its own `ExtendedPictographic`. The answers differ only for
  characters assigned after Unicode 15.0.
- `UnpairedSurrogate` is never reported: a Rust `str` cannot hold an
  unpaired surrogate. The kind exists so a layer read from OpenNLP output
  can carry it.
- An empty kind set is a detector that reports nothing; OpenNLP refuses
  it with an exception. Unknown kinds cannot be written.
- Two kinds are added, both off by default so the defaults match OpenNLP:
  - `MojibakeSequence` (`analysis/normalize/src/artifacts/sequences.rs`,
    `ArtifactDetector::builder().with(ArtifactKind::MojibakeSequence)`):
    the character classes of ftfy's `badness.py` as tables of three code
    pages (`analysis/normalize/src/artifacts/codepage.rs`: Windows-1252
    from CP1252.TXT, ISO-8859-1 for its undefined bytes, MacRoman from
    ROMAN.TXT). Each character that is the image of a UTF-8 lead byte is
    tried with the images of the continuation bytes it needs, one sequence
    at a time, so mojibake is found where the round trip of a whole run
    fails: next to a genuine character ("cafÃ©…" with a typed ellipsis,
    "naÃ¯" before a typed dash), with a byte Windows-1252 leaves undefined replaced by
    U+FFFD ("Ã�LVAREZ" for "ÁLVAREZ", and Japanese, whose kana have 0x81
    in their UTF-8), and in UTF-8 read as MacRoman ("caf√©", "don‚Äôt"). To
    keep genuine text out, a sequence counts only when it decodes to what
    mojibake is made of: under Windows-1252 a two-byte sequence must
    decode to U+00A0..U+017F, Greek or Cyrillic, under MacRoman to
    U+00A0..U+00FF, and a MacRoman three-byte sequence must start with
    "‚" and decode to U+2000..U+214F. "“élan”", "«été»", U+2014 before "à", "CAFÉ–BAR"
    and the multilingual texts of OpenNLP's false-positive test report
    nothing. Round-trip mojibake keeps precedence: a sequence inside a
    `Mojibake` finding is not reported. A sequence wins over the class
    findings it overlaps (a U+FFFD or C1 control inside it).
  - `RepeatedLine` (`analysis/normalize/src/artifacts/repeated.rs`,
    `ArtifactDetector::builder().repeated_lines(RepeatedLines::default())`):
    every occurrence of a line that occurs three or more times
    (`min_occurrences`) in the view, the running headers, footers and page
    labels of extracted documents. This is the duplicate-line rule of the
    Gopher quality filters turned into spans. Lines end at the mandatory
    breaks of UAX #14; they are compared with `White_Space` trimmed, inner
    whitespace runs read as one space and digit runs (Nd) read as one
    digit, so "Page 3 of 12" and "Page 4 of  12" match. Lines without an
    alphabetic character are not compared: with digits folded every table
    cell and bare page number would match. A character finding inside a
    repeated line keeps its span and the line is reported around it. A
    multi-line block that repeats is reported line by line, since each of
    its lines repeats.
- A labelled eleven-page PDF evaluation now measures the default and optional
  detectors. See [Extracted PDF evaluation](#extracted-pdf-evaluation) for
  per-kind support, precision, recall and the limits of this sample.

Tests (`analysis/normalize/tests/artifacts.rs`): every expectation of
`CursorArtifactDetectorTest`, `ArtifactBoundaryTest`,
`ArtifactEncodingTest` (with an independent Windows-1252 decoder),
`ArtifactFalsePositiveTest`, `ArtifactUnicodeContextTest`,
`ArtifactsManualExampleTest` and `ArtifactAnnotatorTest`, offsets written
as UTF-8 bytes (the unpaired-surrogate cases are recorded as unable to
occur, the null-argument and custom-detector cases have no counterpart);
the layer added to a document and run through an `Analyzer`; the cases of
both added kinds above, their precedence and that they are off by default;
and property loops over 10000 generated texts mixing every class, the
contexts the rules read and mojibake images, checking order, bounds and
character boundaries through `Layer::check` and `Document::add`, that each
class span holds only its class (against a classifier written in the
test), that every classified character is reported, that each kind alone
equals the full detector filtered to it, and that the added kinds leave
every default finding covered. Unit tests check the merge, the kind sets,
the MacRoman and Windows-1252 tables and the UTF-8 validation against the
standard library on every two-byte pair.

### Noise scoring (`analysis/normalize/src/noise/mod.rs`)

`NoiseScorer` (an annotator, id `turbo:noise`) reads a `View`, and the
`Placeholders` layer when the document has one, and writes a `Noise` layer
(`turbo:noise`): spans with a `NoiseSeverity` and a score in (0, 1] within
that severity. It splits the view at `White_Space`, trims surrounding
punctuation (ASCII sentence marks, quotes and brackets; outside ASCII every
Ps, Pe, Pi, Pf and Po character) and scores cores of three or more ASCII
characters:

- `Binaryish`: 24 or more characters of the base64 alphabet with a `+`,
  `/` or `=`, twelve case changes, or digits and eight case changes; the
  score is the length over 48, at most 1;
- `Gibberish` (two or more signals, score signals over 4, at most 1) and
  `Damaged` (one signal, 0.5), the signals being: five or more letters and
  no vowel (a e i o u y); eight or more letters and a vowel share of at
  most 0.10; seven consonants in a row; four equal characters in a row;
  four letter-digit transitions in eight or more characters;
- `Misspelled` (0.9), only with a `Dictionary`: a core with no other
  finding that one OCR confusion (rn/m, vv/w, cl/d, 1/l, 0/o, 5/s) turns
  into a dictionary word. A dictionary word is never reported. No
  dictionary ships here.

Findings separated only by whitespace that no exclusion covers are joined,
keeping the worse severity and its score (the larger score when both have
the same severity). Tokens that overlap a placeholder span are not scored.
`NoiseScorer::score(text, exclude)` does the same for a string and a list
of exclusions.

OpenNLP counterpart: `StructuralNoiseScorer`, `NoiseAnnotator` and
`NoiseSpan` in the noise add-on. The rules and constants are the same.

Differences from OpenNLP:

- Surrounding punctuation is tested per code point, so punctuation outside
  the Basic Multilingual Plane is trimmed: "\u{1E95E}bcdfg\u{1E95F}"
  (Adlam exclamation marks) is damaged here; OpenNLP tests UTF-16 code
  units, a surrogate is not punctuation, the core keeps its non-ASCII
  ends and is skipped. The general categories are Unicode 18.0.
- The exclusions are the `Placeholders` layer (URLs, e-mail addresses,
  hashtags, numbers, encoded assets a normalizer replaced), an optional
  input: without it the whole view is scored. OpenNLP's default
  `NoiseAnnotator` requires its asset layer and refuses a document
  without one. Placeholders in a view other than the one read are refused
  with an error.
- Exclusions are sorted and joined once, so each test is a binary search;
  OpenNLP tests every exclusion for every token.
- `config_hash` records whether a dictionary is set, not its contents.
- Added: `NoiseStats` (`analysis/normalize/src/noise/stats.rs`), document
  statistics in the manner of Gopher (Rae et al. 2021) and C4 (Raffel et
  al. 2020): words, lines, the ratio of `#` and ellipses to words, the
  mean word length, the share of lines ending in terminal punctuation,
  and the share of lines (and of characters) in lines that repeat an
  earlier line exactly. They are measurements: no threshold is applied;
  the chunker decides. `NoiseScorer::stats` returns them.

Tests (`analysis/normalize/tests/noise.rs`): every expectation of
`StructuralNoiseScorerTest`, `NoiseScorerBoundaryTest`,
`NoiseManualExampleTest`, `NoiseAnnotatorTest` and `NoiseConcurrencyTest`
(the asset layer replaced by placeholders, UTF-16 offsets written as UTF-8
bytes), the score bounds of `NoiseSpanTest` as the layer's check, the
supplementary punctuation case, the statistics, and a property loop over
4000 generated texts with random exclusions: findings in order, trimmed,
clear of exclusions and accepted by `Document::add`, and each token scored
alone is noise exactly when its core lies inside a finding of the whole
text.

### Symbols (`symbols/`)

OpenNLP's `SymbolJoinerCharSequenceNormalizer` (in the runtime, and as an
identical copy in the `symbol-joiner` add-on) maps an input that is
exactly one of ten symbols to an English word: `&` and, `+` plus, `@`
at, `%` percent, `§` section, `¶` paragraph, `°` degree, `©` copyright,
`®` registered, `™` trademark. Every other input, including `R&D`,
`"& "`, `©️` and currency symbols, is returned unchanged. It is
meant to be called once per token, and in OpenNLP neither version joins
multi-token symbol sequences (`C++`, `.NET`, `->`).

- `SymbolJoiner` (an `Annotator`, id `turbo:normalize:symbol-joiner`)
  reads a `View` and the `Tokens` made on it, joins touching token spans
  whose combined slice matches a symbol sequence in its table
  (`DEFAULT_SYMBOL_SEQUENCES` by default, or caller-configured via
  `SymbolJoiner::with_sequences` and `SymbolJoiner::without_sequences`),
  replaces each remaining token that is a single mapped symbol with its
  word in a new `View`, and writes both the new `View` and the joined
  `Tokens` mapped through its alignment.
- `SymbolWords` (a `Normalizer`, id `turbo:normalize:symbol-words`)
  replaces each mapped symbol in running text that has whitespace (the
  OpenNLP whitespace table) or the edge of the text on both sides:
  `Dungeons & Dragons` becomes `Dungeons and Dragons`.

Differences from OpenNLP: `SymbolWords` finds the symbols in running text
instead of requiring the whole input to be the symbol, so `"& "` becomes
`"and "` and `" &"` becomes `" and"` (OpenNLP leaves both, as a
whole input, unchanged; tokenized, each gives the token `&` and
`and`). Test: `embedded_symbols_are_left_alone`.

Tests (`analysis/normalize/tests/symbols.rs`): every mapping and every unchanged input of
`SymbolJoinerCharSequenceNormalizerTest`, the `Dungeons & Dragons`
example of `SymbolJoinerUsageExampleTest` through `SymbolJoiner` with
its spans, and symbols in running text.

### English contractions (`contractions/`)

`EnglishContractions` (id `turbo:normalize:english-contractions`) ports
`EnglishContractionCharSequenceNormalizer`. A contraction is a run of
ASCII letters, an apostrophe (U+0027, U+2019, U+02BC or U+FF07) and a
run of ASCII letters. The forms and their expansions:

| Form | Expansion |
|---|---|
| `can't` | `can not` (the stem is kept, `'t` becomes ` not`) |
| `won't`, `shan't` | `will not`, `shall not` |
| `<aux>n't` for are, could, dare, did, do, does, had, has, have, is, may, might, must, need, ought, should, was, were, would | `<aux> not` |
| `let's` | `let us` |
| `y'all` | `you all` |
| `'re` after how, here, there, these, they, those, we, what, when, where, who, why, you | ` are` |
| `'ve` after could, I, may, might, must, should, there, these, they, those, we, what, where, who, would, you | ` have` |
| `'ll` after he, here, how, I, it, she, that, there, these, they, this, those, we, what, when, where, which, who, why, you | ` will` |
| `I'm` | `I am` |

Stems are compared ASCII case-insensitively and kept as written. An
all-uppercase word gets an uppercase expansion (`CAN'T` becomes
`CAN NOT`); the replaced stems of `won't`, `shan't` and `y'all` follow
the case of what they replace (`Won't` becomes `Will not`). `'s`, `'d`
and `ain't` have more than one reading and are left alone.

A contraction is expanded only when it is a complete word under the
UAX #29 word boundaries, as OpenNLP decides with
`opennlp.tools.tokenize.uax29.WordSegmenter`: `_can't`, `can't1`,
`écan't`, `écan't`, `I.don't`, `can't've` and `can't‍go` are
left alone, while `'can't'`, `(don't)`, `x-can't` and `can't,` are
expanded. The search follows OpenNLP: each apostrophe is tried as the
middle of a contraction, and after a candidate the search continues at
its end.

The word boundaries are in `contractions/word_break/`, reading the
`Word_Break` and `Extended_Pictographic` properties of Unicode 18.0.0
from `turbo_unicode::word_break` (`WordBreakProperty.txt` in
`analysis/unicode/ucd/`; OpenNLP carries the 17.0.0 file).
`WordBreakProperty-18.0.0.txt` and `WordBreakTest-18.0.0.txt` in
`analysis/normalize/tests/data/` are under the Unicode License (see
`analysis/unicode/ucd/LICENSE`). `is_boundary` applies rules WB1 to WB999
at one position. OpenNLP computes all the boundaries of the text in a
forward pass; the rules only read the characters around a position (the
one before, the one at it, the last two before it that WB4 does not skip,
the next one after it that WB4 does not skip, and the length of the run
of regional indicators before it), so `is_boundary` reads only those and
gives the same answer.

`EnglishContractionsWithPos` (an `Annotator`, id
`turbo:normalize:english-contractions-pos`) reads the `View`, and
`Tokens` and `PosTags` when the document has them, and writes a `View`.
Besides the forms above it expands `'s` and `'d` by the Penn Treebank
tags (looked up by label in the tag set):

- `'s` becomes ` has` when the token holding the apostrophe is `VBZ` and
  the next token `VBN` (`he's gone`), and ` is` when the token is `VBZ`
  and the next is anything else (`he's here`). A possessive (`POS`) is
  left alone.
- `'d` becomes ` would` before a `VB` token (`I'd go`) and ` had` before a
  `VBN` token (`I'd gone`); anything else is left alone (`I'd rather`).

The token holding the apostrophe is `'s` itself under Penn Treebank
tokenization or the whole word under a tokenizer that keeps it whole.
The tokens must be in the view the step reads; tokens of another view
are refused with an error. Without tokens and tags the step writes what
`EnglishContractions` writes.

The alignment keeps the stem: in `can't` to `can not`, `can` maps to
itself and ` not` to `'t`; in `won't` the whole expansion maps to the
whole word; in `y'all`, `all` maps to itself.

Differences from OpenNLP:

- `y'all` is expanded to `you all`, and the subject lists are longer:
  `here`, `these`, `those` for `'re`; `may`, `there`, `these`, `those`,
  `what`, `where` for `'ve`; `here`, `this`, `these`, `those`, `which`
  for `'ll`. Each of these forms has only one reading. OpenNLP leaves
  them unchanged. Test: `additions_beyond_opennlp`.
- `EnglishContractionsWithPos` resolves `'s` and `'d` from tags; OpenNLP
  has no counterpart and leaves them alone. Tests:
  `pos_tags_resolve_s_and_d`, `pos_step_without_tags_is_the_plain_normalizer`.
- Word_Break data is Unicode 18.0.0 rather than OpenNLP's 17.0.0. With
  the 18.0.0 data `is_boundary` also passes every line of the 17.0.0
  conformance file OpenNLP runs (checked once by hand); the 18.0.0 file
  is the one the tests run.

Tests (`analysis/normalize/tests/contractions.rs`, unit tests in
`analysis/normalize/src/contractions/word_break/mod.rs`): every expectation of
`EnglishContractionCharSequenceNormalizerTest`,
`EnglishContractionBoundaryTest` and `EnglishContractionContractTest`
(UTF-16 offsets rewritten as UTF-8 byte offsets), each form in both
cases, the additions, the POS resolution, concurrent use from several
threads, the word-break table against every code point of the property
file, and every line of `WordBreakTest-18.0.0.txt` (1944 lines, none with
surrogates).

### Placeholder normalizers and the shared scan (`placeholder.rs`)

`Urls`, `BoundedUrls`, `SocialMedia`, `BoundedSocialMedia` and `Numbers`
replace stretches of text with one space each: a URL, an e-mail address,
a hashtag, a run of digits. Each is both a `Normalizer` and an
`Annotator`:

- as a `Normalizer` (directly, through `NormalizeStep`, or in a `Chain`)
  it writes only the view;
- as an `Annotator` it reads a `View` and writes the normalized `View`
  and a `Placeholders` layer (`turbo:placeholders`, in
  turbo-normalize-layers) with one row per replacement: the span of the
  replaced text in the view it read and a `PlaceholderKind` (`Url`,
  `Email`, `Hashtag`, `Mention`, `RetweetMarker`, `Emoticon`, `Number`).

Both run the same scan function (`Urls::scan`, `SocialMedia::scan`, and
so on), which writes through the `ViewBuilder` and reports each
replacement to a sink; the `Normalizer` passes a sink that drops the
reports. A placeholder's span is the exact source text that was
replaced, and the view's alignment maps it to the space it became.

All of these scanners work on the bytes of the UTF-8 text where the
characters they look for are ASCII. A byte of a multi-byte character is
never ASCII, so this reads non-ASCII characters exactly as OpenNLP's
UTF-16 scans do (as characters outside every set).

### URLs and e-mail addresses (`url/`)

`Urls` (id `turbo:normalize:urls`) keeps the output of OpenNLP's
`UrlCharSequenceNormalizer`, the normalizer the language detector models
OpenNLP distributes were trained with:

- a URL is lowercase `http://` or `https://` followed by a greedy run of
  at least one of `[-_.?&~;+=/#0-9A-Za-z]`. Nothing is required before
  the scheme (`xhttp://y` becomes `"x "`), and the run stops at the first
  other character, so ports, percent escapes, userinfo and non-ASCII
  hosts cut a URL short (`http://x:8080/y` becomes `" :8080/y"`);
- an e-mail address (OpenNLP's `MailAddressScan`, in `analysis/normalize/src/url/mail.rs`) is a
  maximal run of `[-+_.0-9A-Za-z]` whose preceding character is not in
  that set, `@`, and a domain of `[-.0-9A-Za-z]` that does not start with
  a dot and is at least two characters long (`a@b.` swallows the dot).

OpenNLP removes URLs in one pass and addresses in a second pass over the
result. `Urls` does both in one pass, trying a URL and then an address at
each position, and produces the same text. The reasoning is in the
documentation of `Urls::scan`; in short, the URL matches do not depend
on what precedes them, an address can never contain the start of a URL
except at the end of its domain, and the character after a URL can never
start an address. The one case that needs care is a URL that begins
inside a domain run (`a@bchttp://x`): OpenNLP's first pass turns the URL
into a space, which ends the domain there, so `Urls` cuts the domain at
the URL and checks its length again (`a@bchttp://x` becomes two spaces,
`a@bhttp://x` becomes `"a@b "`).

`BoundedUrls` (id `turbo:normalize:bounded-urls`) ports
`BoundedUrlCharSequenceNormalizer`, which takes a URL whole as it is
bounded in running text:

- the scheme is any `[A-Za-z][A-Za-z0-9+.-]*` followed by `://`, not
  preceded by a letter or digit (`Character.isLetterOrDigit`) or one of
  `_ + - . :`, so the `https` in `git+https://` or `blob:https://` does
  not start a URL;
- the body runs to the first whitespace (OpenNLP's 25-character table,
  `tables::is_whitespace`), control character, `Pi` or `Pf` quotation
  mark, `<`, `>` or `"`;
- trailing `. , ; : ! ?` and closing brackets without an opening bracket
  in the body are given back to the text (`(http://x.y/a_(b))` becomes
  `"( )"`);
- `http` and `https` URLs (any letter case) become a space; URLs with
  other schemes are copied unchanged, and no address is matched inside
  them.

The general categories come from turbo-unicode, which reads Unicode
18.0.0; OpenNLP reads the JVM's (Unicode 15.0 on JDK 21). The only
characters this affects are those assigned or recategorized since 15.0.

Differences from OpenNLP:

- `BoundedUrls` matches internationalized e-mail addresses (RFC 6531
  local parts and U-label domains): the local-part and domain sets are
  widened to every non-ASCII letter, digit and combining mark.
  `José@example.com` and `用户@例子.广告` become one space each; OpenNLP
  leaves both unchanged, and turns `éa@b.co` into `"é "`, removing only
  the ASCII tail of the address, where `BoundedUrls` removes all of it.
  An address must still not be preceded by a character of its local
  part. `Urls` keeps the ASCII scan, because its output is the input of
  the trained language detector models. Test:
  `bounded_replaces_internationalized_addresses` in `analysis/normalize/tests/urls.rs`.

Tests (`analysis/normalize/tests/urls.rs`): every expectation of OpenNLP's
`UrlCharSequenceNormalizerCharacterizationTest` and
`BoundedUrlCharSequenceNormalizerTest` (the cases with unpaired
surrogates, which a Rust string cannot hold, are left out); the weird URL
list of the characterization test checked against a two-pass reference
that follows OpenNLP's passes one to one, with five of them also checked
by hand; 40,000 generated inputs, half from OpenNLP's pool and half from
pieces that make URLs and addresses touch, compared with the two-pass
reference; the adversarial cases (a URL at the end of a sentence, in
brackets, before a comma, two URLs one space apart, an address right
after a URL, a non-ASCII host, `http://` alone) with their exact
placeholder spans for both normalizers; linear work on long runs of
closing brackets; and a pipeline that runs `BoundedUrls` and then
`BoundedSocialMedia` through the analyzer and maps every placeholder
back to the source.

### Embedded assets (`analysis/normalize/src/assets/mod.rs`)

`AssetDetector` (an annotator, id `turbo:assets`) reads a `View` and
writes an `Assets` layer (`turbo:assets`, in
`analysis/normalize-layers/src/lib.rs`): one row per binary payload that
survived text extraction, with its span, the span of its encoded bytes,
an `AssetFormat` (`analysis/normalize-layers/src/asset_format.rs`, whose
`name` and `media_type` are OpenNLP's strings), the media type a `data:`
URI declared, the encoding, the decoded length and, for a PNG or GIF, the
pixel size. The spans are UTF-8 byte offsets of the view read, in text
order and never overlapping. `AssetDetector::detect` and `scan` give the
same rows as `Asset` values, and `Asset::decode` decodes a payload from
the text. Nothing allocates a whole decoded payload: a header is at most
45 bytes, and the SVG and calendar checks read the payload through a
streaming decoder (`analysis/normalize/src/assets/payload.rs`).

The default detector finds what OpenNLP's finds, scanning each position
for a `data:` URI, a PEM envelope, a JSON Web Token and a bare run, in
that order:

- `data:` URIs with a `;base64,` payload of any length, identified by
  their header or by the media type they declare (a declared type with a
  slash after its first character is reported in place of the sniffed
  one; with nothing sniffed, the format name is its subtype);
- RFC 7468 PEM envelopes with one of 18 labels, at a line start, with
  body lines of 1 to 64 characters;
- compact JWTs whose header and claims decode to JSON objects
  (`analysis/normalize/src/assets/json.rs`, depth 64, duplicate members
  refused), with a non-empty `alg` that is `none` exactly when the
  signature is empty;
- bare base64 runs of at least 32 characters, standard or URL-safe,
  possibly wrapped at 64 or 76 columns, whose decoded header a check
  recognizes. The checks are the signature table
  (`analysis/normalize/src/assets/magic.rs`, 219 signatures from
  OpenNLP's `KnownMagics`, longest first) and the structured checks of
  `analysis/normalize/src/assets/sniff.rs` (AIFF, DEX, DWF, EMF, iCalendar
  with folding, JPEG 2000 brands, pcapng, Excel BIFF, WebVTT, RIFF WebP,
  WAVE and AVI, timestamped data, XML), with the header lengths OpenNLP
  decodes (24 bytes, 45 for EMF, 36 for JPEG 2000 and PEM). An XML header
  is SVG when its root element is `svg` in the SVG namespace, read by the
  scanner of `analysis/normalize/src/assets/svg.rs`.

Three additions are off by default, so the defaults give OpenNLP's
output; each is a field of `AssetDetector`:

- `hex_dumps` (`analysis/normalize/src/assets/hex.rs`): runs of 32 or more
  hex byte pairs whose decoded header a check recognizes, written back to
  back, separated by a space, colon, comma or comma and space, with `0x`
  on every pair or none, and continued over lines (a CRLF or LF after the
  separator or its trimmed form, then indentation). The first gap fixes
  the separator. The run must not touch an ASCII letter, digit or `_`.
  Encoding `Hex`. A hash, a UUID or `deadbeef` is never reported, since
  none decodes to a signature.
- `report_unknown_base64` (`analysis/normalize/src/assets/unknown.rs`): a
  valid base64 run no check recognizes, as `AssetFormat::Unknown`
  ("unknown", "application/octet-stream"), when it has at least 32
  characters without padding, an upper case letter, a lower case letter
  and a digit, and at least 45% of its adjacent characters change class
  (upper, lower, digit, symbol). Random base64 changes class about 64% of
  the time; words and identifiers change at word starts only, and hex
  strings lack one case. `Attachments`, `deadbeef`,
  `AlphanumericRunsLikeThisOne0123456789` and hashes are never reported.
  The rule is tried only after a bare run fails, so it never hides an
  asset the defaults report.
- `soft_line_breaks`: a quoted-printable soft line break (`=` then CRLF or
  LF, then a standard base64 character) inside a bare run or a `data:`
  URI payload is part of the payload, for base64 that went through a
  quoted-printable mail part. A URL-safe character makes such a payload
  invalid, as MIME wrapping does. The cost is that a word on the line
  after a payload that ends in `=` is read as more payload, which is why
  it is off.

`AssetFold` (id `turbo:normalize:asset-fold`,
`analysis/normalize/src/assets/fold.rs`) is a normalizer that replaces
each asset with a caption, by default OpenNLP's `AssetFolder.caption()`:
`[png 640x480, 12345 bytes]`, `[pdf, 51200 bytes]` (`write_caption`).
`AssetFold::with_caption(id, f)` takes a caption function instead, as
`AssetFolder.fold(document, replacement)` does; `id` names it in the
configuration hash. `AssetFold::with_detector` folds with the additions
on. As an annotator it reads a `View` and writes the folded `View` and a
`Placeholders` layer with one `Asset` row per asset, spanning the asset in
the view read. Each asset is one `replace`, so a caption maps back to
exactly its asset.

OpenNLP counterpart: `CursorAssetDetector`, `KnownMagics`, `SvgHeader`,
`JsonObjectParser`, `Base64PayloadInputStream`, `EmbeddedAsset`,
`AssetAnnotator` and `AssetFolder` in the embedded-assets add-on. The
constants keep OpenNLP's names (`MIN_BARE_PAYLOAD`,
`MAX_JWT_SEGMENT_LENGTH`, `MAX_MEDIA_TYPE_OFFSET`, `PEM_TYPES`,
`EMF_SIGNATURE_OFFSET`, `TIMESTAMPED_DATA_OID` and the others).

Differences from OpenNLP:

- An unknown dimension is `None`, not -1, and the layer accepts both
  dimensions or neither. OpenNLP's record allows one without the other;
  its detector never writes one.
- The format is the `AssetFormat` enum. A `data:` URI whose format comes
  from its declared type is `AssetFormat::Declared`, and
  `Asset::format_name` and `Assets::format_name` give the subtype OpenNLP
  reports. Null and blank strings and negative lengths cannot be written,
  so `EmbeddedAsset`'s constructor checks become `Assets::check`. Decode
  errors are `DecodeError` values with OpenNLP's messages.
- The caption function takes the asset, the text it was found in and the
  string to write to (`Fn(&Asset, &str, &mut String)`): it can decode the
  payload, and no string is allocated per asset. An empty caption deletes
  the asset.
- `AssetFold` detects as it folds, so a missing layer and rows out of
  order (OpenNLP's `IllegalArgumentException` cases) cannot occur.
- Two assets that touch (two `data:` URIs back to back) keep their own
  edits, so each caption maps back to its own asset, and the
  `Placeholders` layer holds one row per asset
  (`touching_assets_keep_their_own_edits`).
- OpenNLP filters runs by base64 prefixes of the signatures before it
  decodes a header; here the first decoded byte is looked up in
  `FIRST_BYTES`. The results are the same: the checks after the filter
  decide.
- SVG is read by a hand-written scanner instead of the JDK's StAX parser.
  It follows the parser with DTD support off, as `SvgHeader` configures
  it, including its quirks: the internal subset is skipped unread up to
  its first `]` (so `<!DOCTYPE svg [<!A<TLIST>]>` is accepted and a `]`
  inside a comment there ends it), public identifiers hold PubidChar only,
  a processing instruction target may hold a colon, a leading colon is
  part of a name, and the JDK's encoding names and aliases are accepted
  (`UTF8` and `ISO8859-1` are not). On 120000 generated SVG and XML
  headers and 50000 other generated texts its answers equal OpenNLP's,
  apart from the inputs where OpenNLP throws (below). Where it differs:
  - only UTF-8, UTF-16, US-ASCII and ISO-8859-1 are read; a header that
    declares another encoding (`windows-1252`, `Shift_JIS`) is XML but
    not SVG here, while the JDK reads it
    (`svg_other_declared_encodings_are_not_read`);
  - names follow XML 1.0 fifth edition, which the current specification
    is; the JDK applies the fourth edition's tables to XML 1.0 and
    rejects names such as U+2C30 that the fifth edition allows
    (`svg_names_follow_the_fifth_edition`);
  - a character XML does not allow inside an internal subset makes the
    JDK throw `MissingResourceException`, which `SvgHeader` does not
    catch, so OpenNLP's `detect` throws; here the payload is not SVG
    (`svg_prolog_follows_the_jdk_parser`);
  - the scan stops at the end of the root start tag. The JDK decodes
    ahead and fails on a four-byte UTF-8 sequence above U+10FFFF up to
    about 8 KB after the tag, so OpenNLP's answer can depend on bytes
    after the root element (`svg_bytes_after_the_root_tag_do_not_matter`).
- In the JSON parser a lone surrogate escape in a captured member value
  becomes U+FFFD, since a Rust string cannot hold it. Only `alg` is
  captured and only its emptiness and `none` are tested, so detection is
  the same.
- The three additions above.

Tests (`analysis/normalize/tests/assets.rs`, helpers in
`analysis/normalize/tests/assets_support/mod.rs`): every expectation of
OpenNLP's 28 embedded-assets test classes, class by class
(`CursorAssetDetectorTest`, `EmbeddedAssetTest`, `AssetFolderTest`,
`AssetAnnotatorTest`, `AssetFalsePositiveTest`, `AssetsManualExampleTest`,
`AssetImageMetadataTest`, `AssetTransportValidationTest`, the signature
tests for EMF, PEM, MPX, Excel, pcapng, iCalendar, DEX, the PNG family,
AIFF, WebVTT, Android XML and GRIB, media types, timestamped data, format
variants, DWF, JPEG 2000 and SVG, and `KnownMagicsTest`), in every
transport they use, with fixtures written in the test (a stored zip for
the DWF zip case and the AU header Java Sound writes for the A-law case);
`Base64PayloadInputStreamTest` and `JsonObjectParserTest` are unit tests
in `analysis/normalize/src/assets/payload.rs` and
`analysis/normalize/src/assets/json.rs`. OpenNLP's null-argument cases
have no counterpart. Then the SVG scanner against the JDK parser's
answers for DOCTYPEs, processing instructions, encodings and qualified
names; lookalikes (hashes, UUIDs, identifiers, `Attachments`, `deadbeef`,
near-miss URIs, envelopes and tokens) with the defaults and every
addition on; UTF-8 offsets after multibyte text and the media type limit
counted in UTF-16 units; each addition's forms and edges and that it is
off by default; the fold's captions, custom captions, touching assets and
the annotator. Every detection builds the layer through a `Document`
(`Layer::check` and the document's checks) and checks each row and
decode, and every fold is checked both ways: the edits rebuild the
output, each edit is one asset or a run of touching ones, each asset maps
forward to its caption and the caption back to it, and kept text maps to
itself. Property loops run 600 generated texts mixing assets in every
transport, cut and corrupted payloads, hex dumps, soft breaks and prose
through five configurations, checking those invariants, that the unknown
addition only adds rows, and that each asset found is found the same in a
text holding only it.

### Social media (`social/`)

`SocialMedia` (id `turbo:normalize:social-media`) keeps the output of
OpenNLP's `SocialMediaCharSequenceNormalizer`, which is also part of the
language detector's preparation. OpenNLP runs four passes, each over the
previous one's output:

1. a hashtag or handle, `#` or `@` followed by a character that is not
   ASCII whitespace, up to the next ASCII whitespace, becomes a space
   (`#a#b` is one tag; `abc#def` becomes `"abc "`);
2. a run of retweet units, `rt` and a space or colon, ASCII
   case-insensitively, becomes a space when it starts on a word boundary
   as the JDK regular expression engine decides `\b`: not after an ASCII
   letter, digit or `_`, nor after a non-spacing mark that sits on a
   letter or digit (the engine walks back over the marks one UTF-16 unit
   at a time, so a supplementary-plane character in that walk never
   blocks);
3. an emoticon, eyes `: ; x X`, an optional `-` and a mouth
   `( ) d D o O p P`, becomes a space, inside words too (`expo` becomes
   `"e o"`);
4. laughter, a run of `h`/`j`, a run of vowels and at least one more pair
   of runs of the last consonant and the last vowel, shrinks to that
   consonant and vowel twice as written (`hAHa` becomes `hAhA`).

`SocialMedia` does all four in one pass and produces the same text. The
four patterns start with disjoint characters and none can contain the
first character of another, so they never compete for a position. The
one interaction between the passes is that the space a hashtag or handle
becomes can end a retweet unit: `rt#tag` is `"rt "` after the first pass
and one space after the second. The scan accepts a tag in that place;
the placeholders are a `RetweetMarker` for `rt` and a `Hashtag` or
`Mention` for the tag, both mapped to the one space. Laughter is not a
placeholder. A shrunk laugh keeps the part it shares with its source, so
`hahaha` keeps `haha` (mapped to itself) and deletes `ha`.

`BoundedSocialMedia` (id `turbo:normalize:bounded-social-media`) finds
the same kinds of markup by Unicode properties and word boundaries,
closer to how the platforms that define hashtags and mentions delimit
them. A word character is a letter or digit, a combining mark, or `_`.

- A hashtag is `#` or `＃` (U+FF03), not after a word character or `&`,
  followed by a run of word characters, ZWNJ and ZWJ that contains a
  letter or mark: `#café`, `#日本`, `#हिन्दी`, `#100DaysOfCode`. `#1` and
  `#2024` are not hashtags, nor is the `#39` of `&#39;`.
- A mention is `@` or `＠` (U+FF20), not after a word character or one
  of `! @ # $ % & *`, followed by a run of word characters, optionally
  continued by `@` and a domain of two or more labels (a fediverse
  handle such as `@ana@social.example`). Any other `@` right after the
  run means it is not a mention, so `a@b.com` and `@ana@x` are left
  alone.
- A retweet marker is a run of `rt` units not after a word character; a
  last `rt` directly before a mention (`RT@ana`) also ends the marker.
- An emoticon is the same eyes, nose and mouth, not followed by a word
  character, and with `x`/`X` eyes not after one.
- Laughter shrinks as in `SocialMedia`.

Differences from OpenNLP (`BoundedSocialMedia` only; `SocialMedia`
matches OpenNLP exactly because the language detector models were
trained on its output):

- A hashtag or mention ends where its word ends instead of at the next
  ASCII whitespace, so punctuation after it stays in the text:
  `#win!` becomes `" !"` (OpenNLP: `" "`), `RT @ana: hello` becomes
  `"  : hello"` (OpenNLP: `"   hello"`). Tests:
  `bounded::hashtags_end_at_the_word`,
  `bounded::retweet_markers_need_a_word_boundary`.
- Letters, digits and marks of every script count as word characters,
  for the start of a tag and for the word boundary of retweet markers,
  so `éRT x` keeps its `RT` (OpenNLP removes it, since `é` is not an
  ASCII word character).
- Number-only tags (`#1`, `#2024`), HTML entities (`&#39;`), tags inside
  words (`abc#def`) and bare `#`/`##` are not replaced. OpenNLP replaces
  `abc#def` with `"abc "` and `##` with a space.
- Fullwidth `＃` and `＠` start tags, and fediverse handles are one
  mention.
- Emoticons do not match inside words: `expo`, `taxi`, `xoxo` and
  `:people` are kept (OpenNLP: `"e o"`, `"taxi"`, two spaces, `" eople"`).
  Test: `bounded::emoticons_stand_alone`.

Tests (`analysis/normalize/tests/social_media.rs`): every expectation of OpenNLP's
`SocialMediaCharSequenceNormalizerCharacterizationTest` (a supplementary
non-spacing mark stands in for the lone surrogates); 50,000 generated
inputs from OpenNLP's pool compared with a four-pass reference that
follows OpenNLP's passes one to one; the tag that ends a retweet unit;
the placeholder kinds and spans of both normalizers; and the
`BoundedSocialMedia` rules above.

### Numbers (`number.rs`)

`Numbers` (id `turbo:normalize:numbers`) replaces each maximal run of
ASCII digits with one space (`12.34` becomes `" . "`) and records a
`Number` placeholder per run. Digits of other scripts are left alone, as
in OpenNLP, whose `\d` class was ASCII only; `Digits` is the normalizer
that folds them to ASCII.

OpenNLP counterpart: `NumberCharSequenceNormalizer`. No differences.

Tests (`analysis/normalize/tests/numbers_shrink.rs`): every expectation of
`NumberCharSequenceNormalizerCharacterizationTest`, 5000 generated
inputs from its pool against the digit-run rule, the placeholder spans,
and that digit-free text comes back borrowed.

### Shrink (`shrink.rs`)

`Shrink` (id `turbo:normalize:shrink`) ports
`ShrinkCharSequenceNormalizer`:

1. a run of two or more ASCII whitespace characters (U+0009..U+000D,
   U+0020) becomes one space; a single tab or newline is kept;
2. a run of three or more of one character, compared ASCII
   case-insensitively, becomes the run's first character twice (`aAAb`
   becomes `aab`, not `aAb`); runs of LF, CR, U+0085, U+2028 and U+2029
   are never shrunk, because Java's `.` does not match them;
3. leading and trailing characters at or below U+0020 are deleted, as
   `String.trim()` does, control characters included.

OpenNLP runs the three steps as passes; `Shrink` runs one and produces
the same text. After step 1 no two whitespace characters are adjacent,
so step 2 only shrinks runs that are the same in the original text, and
the characters step 3 deletes are exactly the images of the input's
leading and trailing characters at or below U+0020 (a run never mixes
those with other characters), which `Shrink` deletes directly. A shrunk
run keeps its first character, mapped to itself, and writes the second
copy over the rest of the run.

OpenNLP counterpart: `ShrinkCharSequenceNormalizer`. No differences.

Tests (`analysis/normalize/tests/numbers_shrink.rs`): every expectation of
`ShrinkCharSequenceNormalizerCharacterizationTest` (without its lone
surrogate cases), and 50,000 generated inputs from its pool compared with
a three-pass reference that follows OpenNLP's passes one to one.
### Text artifacts (`analysis/normalize/src/artifacts/mod.rs`)

`ArtifactDetector` (an annotator, id `turbo:artifacts`) reads a `View` and
writes an `Artifacts` layer (`turbo:artifacts`): a span and an
`ArtifactKind` for every damaged or hidden character sequence. It does not
write a view; the spans are UTF-8 byte offsets of the view it read, in text
order and never overlapping (the layer's `check` refuses overlaps, empty
spans and a missing kind). `ArtifactKind::name` gives OpenNLP's type
string ("replacement", "mojibake", ...), which the golden files use.

The default detector reports OpenNLP's nine kinds, by OpenNLP's rules, on
code points:

- maximal runs of one class: U+FFFD (`ReplacementChar`); C0 and C1
  controls that are not `White_Space` (`Control`, so tab, line feed and
  U+0085 are not reported); noncharacters (`Noncharacter`); private-use
  code points (`PrivateUse`); U+202A..U+202E, U+2066..U+2069, U+200E,
  U+200F and U+061C (`BidiControl`);
- zero-width characters (U+200B, U+200C, U+200D, U+2060, U+FEFF): a run of
  two or more is reported; a single one is left alone between letters, with
  combining marks passed over on both sides (ZWNJ in Persian and
  Malayalam, ZWJ in Sinhala, ZWSP in Thai), and a single ZWJ is left alone
  next to an `Extended_Pictographic` character or U+FE0F (emoji ZWJ
  sequences);
- runs from the Tags block (`UnicodeTag`), except a run after U+1F3F4 that
  spells a region or subdivision code and ends with CANCEL TAG (the
  England, Scotland and Wales flags), by the syntax of UTS #51 and UTS #35;
- `Mojibake`: a maximal run of non-ASCII characters that Windows-1252 can
  encode (ISO-8859-1 for the C1 controls) whose bytes are strictly
  well-formed UTF-8 with at least one multi-byte sequence (RFC 3629: no
  overlong form, no surrogate, nothing past U+10FFFF, nothing cut off).
  A mojibake finding wins over the class findings it overlaps; the parts
  of a class run outside it are still reported.

`ArtifactDetector::builder()` selects kinds (`only`, `with`, `without`),
as OpenNLP's `CursorArtifactDetector(Set)` does.

OpenNLP counterpart: `CursorArtifactDetector`, `ArtifactAnnotator` and
`TextArtifact` in the text-artifacts add-on.

Differences from OpenNLP:

- Character data is Unicode 18.0 (`turbo_unicode`): `White_Space`,
  `Extended_Pictographic`, the letter test (`Character.isLetter`) and the
  combining-mark test. OpenNLP reads the JDK's tables (Unicode 15.0 in JDK
  21) and its own `ExtendedPictographic`. The answers differ only for
  characters assigned after Unicode 15.0.
- `UnpairedSurrogate` is never reported: a Rust `str` cannot hold an
  unpaired surrogate. The kind exists so a layer read from OpenNLP output
  can carry it.
- An empty kind set is a detector that reports nothing; OpenNLP refuses
  it with an exception. Unknown kinds cannot be written.
- Two kinds are added, both off by default so the defaults match OpenNLP:
  - `MojibakeSequence` (`analysis/normalize/src/artifacts/sequences.rs`,
    `ArtifactDetector::builder().with(ArtifactKind::MojibakeSequence)`):
    the character classes of ftfy's `badness.py` as tables of three code
    pages (`analysis/normalize/src/artifacts/codepage.rs`: Windows-1252
    from CP1252.TXT, ISO-8859-1 for its undefined bytes, MacRoman from
    ROMAN.TXT). Each character that is the image of a UTF-8 lead byte is
    tried with the images of the continuation bytes it needs, one sequence
    at a time, so mojibake is found where the round trip of a whole run
    fails: next to a genuine character ("cafÃ©…" with a typed ellipsis,
    "naÃ¯" before a typed dash), with a byte Windows-1252 leaves undefined replaced by
    U+FFFD ("Ã�LVAREZ" for "ÁLVAREZ", and Japanese, whose kana have 0x81
    in their UTF-8), and in UTF-8 read as MacRoman ("caf√©", "don‚Äôt"). To
    keep genuine text out, a sequence counts only when it decodes to what
    mojibake is made of: under Windows-1252 a two-byte sequence must
    decode to U+00A0..U+017F, Greek or Cyrillic, under MacRoman to
    U+00A0..U+00FF, and a MacRoman three-byte sequence must start with
    "‚" and decode to U+2000..U+214F. "“élan”", "«été»", U+2014 before "à", "CAFÉ–BAR"
    and the multilingual texts of OpenNLP's false-positive test report
    nothing. Round-trip mojibake keeps precedence: a sequence inside a
    `Mojibake` finding is not reported. A sequence wins over the class
    findings it overlaps (a U+FFFD or C1 control inside it).
  - `RepeatedLine` (`analysis/normalize/src/artifacts/repeated.rs`,
    `ArtifactDetector::builder().repeated_lines(RepeatedLines::default())`):
    every occurrence of a line that occurs three or more times
    (`min_occurrences`) in the view, the running headers, footers and page
    labels of extracted documents. This is the duplicate-line rule of the
    Gopher quality filters turned into spans. Lines end at the mandatory
    breaks of UAX #14; they are compared with `White_Space` trimmed, inner
    whitespace runs read as one space and digit runs (Nd) read as one
    digit, so "Page 3 of 12" and "Page 4 of  12" match. Lines without an
    alphabetic character are not compared: with digits folded every table
    cell and bare page number would match. A character finding inside a
    repeated line keeps its span and the line is reported around it. A
    multi-line block that repeats is reported line by line, since each of
    its lines repeats.
- A labelled four-page PDF evaluation now measures the default and optional
  detectors. See [Extracted PDF evaluation](#extracted-pdf-evaluation) for
  per-kind support, precision, recall and the limits of this sample.

Tests (`analysis/normalize/tests/artifacts.rs`): every expectation of
`CursorArtifactDetectorTest`, `ArtifactBoundaryTest`,
`ArtifactEncodingTest` (with an independent Windows-1252 decoder),
`ArtifactFalsePositiveTest`, `ArtifactUnicodeContextTest`,
`ArtifactsManualExampleTest` and `ArtifactAnnotatorTest`, offsets written
as UTF-8 bytes (the unpaired-surrogate cases are recorded as unable to
occur, the null-argument and custom-detector cases have no counterpart);
the layer added to a document and run through an `Analyzer`; the cases of
both added kinds above, their precedence and that they are off by default;
and property loops over 10000 generated texts mixing every class, the
contexts the rules read and mojibake images, checking order, bounds and
character boundaries through `Layer::check` and `Document::add`, that each
class span holds only its class (against a classifier written in the
test), that every classified character is reported, that each kind alone
equals the full detector filtered to it, and that the added kinds leave
every default finding covered. Unit tests check the merge, the kind sets,
the MacRoman and Windows-1252 tables and the UTF-8 validation against the
standard library on every two-byte pair.

### Noise scoring (`analysis/normalize/src/noise/mod.rs`)

`NoiseScorer` (an annotator, id `turbo:noise`) reads a `View`, and the
`Placeholders` layer when the document has one, and writes a `Noise` layer
(`turbo:noise`): spans with a `NoiseSeverity` and a score in (0, 1] within
that severity. It splits the view at `White_Space`, trims surrounding
punctuation (ASCII sentence marks, quotes and brackets; outside ASCII every
Ps, Pe, Pi, Pf and Po character) and scores cores of three or more ASCII
characters:

- `Binaryish`: 24 or more characters of the base64 alphabet with a `+`,
  `/` or `=`, twelve case changes, or digits and eight case changes; the
  score is the length over 48, at most 1;
- `Gibberish` (two or more signals, score signals over 4, at most 1) and
  `Damaged` (one signal, 0.5), the signals being: five or more letters and
  no vowel (a e i o u y); eight or more letters and a vowel share of at
  most 0.10; seven consonants in a row; four equal characters in a row;
  four letter-digit transitions in eight or more characters;
- `Misspelled` (0.9), only with a `Dictionary`: a core with no other
  finding that one OCR confusion (rn/m, vv/w, cl/d, 1/l, 0/o, 5/s) turns
  into a dictionary word. A dictionary word is never reported. No
  dictionary ships here.

Findings separated only by whitespace that no exclusion covers are joined,
keeping the worse severity and its score (the larger score when both have
the same severity). Tokens that overlap a placeholder span are not scored.
`NoiseScorer::score(text, exclude)` does the same for a string and a list
of exclusions.

OpenNLP counterpart: `StructuralNoiseScorer`, `NoiseAnnotator` and
`NoiseSpan` in the noise add-on. The rules and constants are the same.

Differences from OpenNLP:

- Surrounding punctuation is tested per code point, so punctuation outside
  the Basic Multilingual Plane is trimmed: "\u{1E95E}bcdfg\u{1E95F}"
  (Adlam exclamation marks) is damaged here; OpenNLP tests UTF-16 code
  units, a surrogate is not punctuation, the core keeps its non-ASCII
  ends and is skipped. The general categories are Unicode 18.0.
- The exclusions are the `Placeholders` layer (URLs, e-mail addresses,
  hashtags, numbers, encoded assets a normalizer replaced), an optional
  input: without it the whole view is scored. OpenNLP's default
  `NoiseAnnotator` requires its asset layer and refuses a document
  without one. Placeholders in a view other than the one read are refused
  with an error.
- Exclusions are sorted and joined once, so each test is a binary search;
  OpenNLP tests every exclusion for every token.
- `config_hash` records whether a dictionary is set, not its contents.
- Added: `NoiseStats` (`analysis/normalize/src/noise/stats.rs`), document
  statistics in the manner of Gopher (Rae et al. 2021) and C4 (Raffel et
  al. 2020): words, lines, the ratio of `#` and ellipses to words, the
  mean word length, the share of lines ending in terminal punctuation,
  and the share of lines (and of characters) in lines that repeat an
  earlier line exactly. They are measurements: no threshold is applied;
  the chunker decides. `NoiseScorer::stats` returns them.

Tests (`analysis/normalize/tests/noise.rs`): every expectation of
`StructuralNoiseScorerTest`, `NoiseScorerBoundaryTest`,
`NoiseManualExampleTest`, `NoiseAnnotatorTest` and `NoiseConcurrencyTest`
(the asset layer replaced by placeholders, UTF-16 offsets written as UTF-8
bytes), the score bounds of `NoiseSpanTest` as the layer's check, the
supplementary punctuation case, the statistics, and a property loop over
4000 generated texts with random exclusions: findings in order, trimmed,
clear of exclusions and accepted by `Document::add`, and each token scored
alone is noise exactly when its core lies inside a finding of the whole
text.

### Spelling correction (`analysis/normalize/src/spellcheck/mod.rs`)

`SymSpell` is a symmetric delete spelling corrector over a frequency
dictionary. Each indexed word is reduced to the strings its first
`prefix_length` code points become when up to
`max_dictionary_edit_distance` of them are deleted; a query is reduced
the same way, and the words found under its deletes are verified with
the configured `Metric` (optimal string alignment by default, or
Levenshtein). `SymSpellConfig` holds OpenNLP's defaults (distance 2,
prefix 7, count threshold 1, corpus size derived from the counts) and
`build()` refuses inconsistent values. The engine offers:

- `add` (a word becomes visible once its count reaches the threshold;
  counts saturate at `Long.MAX_VALUE`, as OpenNLP's do), `add_bigram`,
  `count`, `contains`, `bigram`;
- `lookup(term, verbosity, max)` with `Top`, `Closest` and `All`, the
  suggestions ordered by distance, then count (larger first), then term
  in UTF-16 order, as OpenNLP's `SuggestItem` compares them;
- `lookup_compound(text, max)`, which corrects a phrase and also splits
  words run together and merges words split apart, scored with the
  bigram counts or the naive Bayes estimate. Besides the corrected
  phrase it returns `parts`: for each output word, the byte range of the
  input it came from and its range in the output;
- `crate::noise::Dictionary`, so the noise scorer can use the engine as
  its word list.

`FrequencyDictionary` reads OpenNLP's plain-text unigram and bigram
files (`word count`, `w1 w2 count`; `#` comments and blank lines
skipped; a malformed line gives a `MalformedLine` with OpenNLP's message
and line number). `SymSpellModel` reads and writes the binary models of
OpenNLP's `SymSpellModelSerializer` (magic `SYMS`, format version 1,
modified UTF-8 strings, variable-length counts) and builds an engine. No
dictionary or model ships with the crate.

Memory layout: the words are one `Vec<String>` in the order they were
first added, with their counts (`u64`) and code point lengths (`u32`) in
two vectors beside it; a word's position is its id, and an
open-addressing table of ids keyed by the word's hash finds it, so no
word is stored twice. The deletes index maps the 64-bit hash of a delete
string to its words: one id held inline when a single word produces the
delete (most deletes), otherwise the position of a list of ids. The
delete strings themselves are not kept. Because two deletes can share a
hash, every candidate is checked to really have the probed string as a
delete (a subsequence test on at most `prefix_length` code points), so
the candidates and their order are exactly OpenNLP's. A lookup allocates
only the arena of its own deletes and the code points of the query.

`SpellCorrect` (id `turbo:normalize:spell-correct`) is the normalizer.
Per token (the default), the text is split at `White_Space`; a token
loses its leading and trailing code points that are not letters or
numbers, and the core left is looked up lowercased when it is at least
`min_token_length` (4) code points long, not number-like
(`skip_numbers`), not URL- or e-mail-like (`skip_urls`) and has a
letter. A known core stays; a correction takes the core's casing (all
uppercase, or a capital first letter) and replaces the core alone, so
the punctuation and whitespace around it keep their offsets. In
compound mode (`SpellCorrectMode::Compound`) the text goes to
`lookup_compound`; each output word that differs from its input is one
replacement of that input (a merge covers both tokens and the
whitespace between them), whitespace at the ends is deleted, and a
whitespace run between words keeps one existing space and loses the
rest, or becomes a space when it has none. Keeping a space keeps the
edits of neighbouring words apart; a run with no space (`"teh\u{A0}wrold"`)
joins them into one edit, because the `ViewBuilder` merges edits that
touch. The engine is shared through an `Arc`.

OpenNLP counterparts: `SymSpell`, `SymSpellConfig`, `SuggestItem`,
`Verbosity`, `DamerauOSADistance` and `LevenshteinDistance`,
`FrequencyDictionaryLoader`, `SymSpellModel` and
`SymSpellModelSerializer`, and `SpellCheckingCharSequenceNormalizer`, all
in the spellcheck add-on.

Differences from OpenNLP:

- The minimum token length counts code points; OpenNLP counts UTF-16
  units, so it corrects "\u{10428}wr" (three code points) at the default
  minimum of 4. Test `token_length_counts_code_points`.
- `lookup_compound` splits a token only between code points and treats
  a supplementary character as one character. OpenNLP splits at every
  UTF-16 index, which looks up the halves of a surrogate pair, and tries
  to split "😀" because its `length()` is 2. Test
  `compound_splits_only_between_code_points`.
- `lookup_compound` checks the requested distance against the index
  before anything else; OpenNLP checks it only when it looks a token up,
  so a blank input with too large a distance returns a result there and
  `LookupError` here. Test `a_distance_above_the_index_is_refused`.
- The metric is the enum `Metric` of the two metrics the model format
  names; OpenNLP accepts any `EditDistance`, but its serializer writes
  only these two. `EditDistance` is still a trait for other uses.
- The model reader refuses bytes after the last bigram
  (`ModelError::TrailingBytes`), a count above `Long.MAX_VALUE` (which
  Java reads as a negative count and refuses later, with an
  `IllegalArgumentException`) and a string holding an unpaired surrogate,
  which a Rust string cannot hold. Test `corrupt_streams_are_refused`,
  `modified_utf8_and_vlong_rows`.
- `entry_count` is the number of distinct delete hashes, which equals
  OpenNLP's number of distinct delete strings unless two of them share a
  64-bit hash.
- There is no Java serialization of the normalizer and no
  `withSpellChecker`: the engine is held by `Arc` and the normalizer is
  `Clone`.
- Added, all off by default:
  - `keep_unknown_proper_nouns`: in per-token mode a core that starts
    with an uppercase letter is corrected only at edit distance 1, so
    names the dictionary does not know are not turned into common words
    two edits away ("Quinn" stays; OpenNLP writes "Quick"). Test
    `proper_nouns_two_edits_from_a_word_can_be_kept`.
  - `use_placeholders`: as an annotator, tokens that overlap a span of
    the `Placeholders` layer of the view read are skipped, in place of
    OpenNLP's URL and e-mail guess; in compound mode they are kept and
    the phrases between them are corrected apart. A layer of another
    view is refused with an error. Tests
    `placeholders_replace_the_url_guess`,
    `placeholders_split_compound_phrases`,
    `placeholders_of_another_view_are_refused`.
  - A `Corrections` layer (`turbo:corrections`, in
    `analysis/normalize-layers/src/lib.rs`): one row per replaced
    stretch of the source, with the replacement and its edit distance.
    Test `the_corrections_layer_records_each_change`.
  - `Compound::parts`, the provenance of each output word, which the
    alignment and the layer are built from. Test
    `compound_parts_say_where_each_word_came_from`.
  - `ContextSpellCorrector` / `ContextSpellAnnotator` (id
    `turbo:normalize:context-spell-correct`, in
    `analysis/normalize/src/spellcheck/context.rs`): generates per-token
    candidates from `SymSpell` within `max_edit_distance` together with
    real-word homophone and confusable groups (`ConfusableSet`, such as
    `there`/`their`/`they're`, `to`/`too`/`two`, `form`/`from`), builds a
    sentence trellis, and decodes the optimal candidate path via Viterbi
    under $\lambda \cdot d_{\text{edit}}(w_i, c_i) - \log P_{\text{LM}}(c_i \mid c_{i-n+1..i-1})$
    scored by an `NGramLanguageModel` (`turbo-ngram`). As an `Annotator`,
    it automatically protects any token overlapping `ProtectedSpans`
    (`turbo:protected_spans`), `Placeholders`, `Pii`, `Temporals`,
    `Money`, `Quantities`, `Entities`, or `Abbreviations`, writing a
    derived `View` with exact bidirectional `Alignment` and a
    `Corrections` layer. Tests in
    `analysis/normalize/tests/context_spellcheck.rs`.

Tests: `analysis/normalize/tests/spellcheck.rs` ports `SymSpellTest`,
`LookupTest`, the distance, configuration, loader and serializer tests,
and checks the engine against OpenNLP's output over the fixtures in
`analysis/normalize/tests/data/spellcheck/` (see
`analysis/normalize/tests/data/spellcheck/README.md`): 2511 lookups at
every verbosity and distance and 2015 compound lookups give the same
suggestions, distances and counts, and the two models OpenNLP wrote read
and write back byte for byte. `analysis/normalize/tests/spell_correct.rs`
ports the normalizer's tests, runs 4218 inputs in six configurations
against OpenNLP's output, checks the alignment of both modes with
`check_alignment`, and holds each difference above to its inputs. Unit
tests check the bounded distances against the full table over 20000
random pairs, the uppercase table against `SpecialCasing.txt`, and the
number and URL guards against matchers written from OpenNLP's patterns.

### Emoji (`analysis/normalize/src/emoji/mod.rs`)

Four steps work on emoji. All of them take their sequences from
`turbo_unicode::emoji::scan` (UTS #51 possible emoji on Unicode 18.0) or
their table from OpenNLP's data, and all of them write each replaced
stretch with its own replacement.

**Which sequences are emoji.** The scanner reports every character with
the Emoji property, so `©`, `®`, `™`, `↔`, `☹` and a bare `❤` come out
as `Basic` sequences: no character property separates them from emoji
shown as pictures. `emoji::is_emoji_presentation` applies the
presentation rule of UTS #51: keycap, flag, modifier, tag and ZWJ
sequences are emoji; a single character is an emoji when it has
Emoji_Presentation or U+FE0F follows it. That leaves out the 207
text-default characters of Unicode 18.0 (the keycap bases aside) when
they are written without U+FE0F, which in running text are symbols
(`© 2026`, `A ↔ B`). `EmojiRemove::sequences()` and `EmojiAnnotate` both
use this rule; `EmojiAnnotate::with_text_default()` turns it off.

**`EmojiRemove`** (id `turbo:normalize:emoji-remove`) replaces emoji with
U+0020. It is a `Normalizer` and, like the other placeholder normalizers,
an `Annotator` that also writes a `Placeholders` layer with kind `Emoji`.

- `EmojiRemove::sequences()` (the default): every emoji sequence becomes
  one space: a ZWJ family, a flag, a subdivision flag, a keycap, a
  skin-toned hand, a character with its U+FE0F. Sequences with no text
  between them become one space together; each is its own replacement
  (the first writes the space, the others write nothing) and its own
  placeholder, so a placeholder's span is exactly one sequence. A
  character followed by U+FE0E stays, as does a U+200D or U+FE0F that is
  not part of a sequence.
- `EmojiRemove::supplementary()`: every run of supplementary-plane code
  points becomes one space, with one placeholder per run. This is
  OpenNLP's `EmojiCharSequenceNormalizer` in its default mode, which the
  language detector models were trained with, so the language detector
  preparation uses this mode.

OpenNLP's `EmojiCharSequenceNormalizer` also has a legacy mode, selected
by `CompatibilityMode.LEGACY`, which reproduces the 1.x and 2.x releases:
it replaces `-` (U+002D) and every code point from U+D83C to U+10FC00.
The range was meant to catch the UTF-16 high surrogates that start most
emoji, but read as code points it also takes the low surrogates, every
character from U+E000 to U+FFFF (the private use area, the CJK
compatibility ideographs, the presentation forms, the variation
selectors, the halfwidth and fullwidth forms among them) and every
supplementary code point up to U+10FC00, and it removes every
hyphen-minus. It is not ported: it reproduces UTF-16 surrogate handling
that 3.0 corrected, and a Rust string has no surrogates. Models trained
with OpenNLP 1.x or 2.x saw that output; OpenNLP 3.x defaults to the
corrected mode, which `EmojiRemove::supplementary()` matches.

**`EmojiToEmoticon`** (id `turbo:normalize:emoji-to-emoticon`) and
**`EmoticonToEmoji`** (id `turbo:normalize:emoticon-to-emoji`) port
OpenNLP's `EmojiEmoticons` over emoji-emoticons.txt, which
`analysis/normalize/src/emoji/data.rs` holds as Rust constants with all six columns of the file
(source, target, fold type, standard, Unicode version, note) in the
file's order. At each position the longest source wins (candidates are
ordered longest first in UTF-16 units, as OpenNLP orders them; for this
table UTF-8 length gives the same order).

- Pictographs to emoticons (35 rows, many to one: `🙂` and `😊` to `:)`,
  every heart to `<3`): no fold after U+200D; one U+FE0F after the
  source is replaced with it; no fold when U+200D or U+FE0E follows, so
  `❤️‍🔥` and `☺︎` stay. Pictographs glued to words fold.
- Emoticons to pictographs (26 rows: `:-)` and `:)` to `🙂`, `<3` to
  `❤️`): only when the text edge or a `White_Space` character
  (`tables::WHITESPACE`, the no-break spaces included) is on both sides,
  so `1:3`, `https://` and `nice:)` stay.

OpenNLP counterparts: `EmojiToEmoticonCharSequenceNormalizer`,
`EmoticonToEmojiCharSequenceNormalizer`. No differences in the output.

**`EmojiAnnotate`** (id `turbo:normalize:emoji-annotate`) reads a `View`
and writes the `Emoji` layer (`turbo:emoji`): one row per emoji sequence
with its span, its kind (`Basic`, `Keycap`, `Flag`, `Modifier`, `Tag`,
`Zwj`), its CLDR short name from the Unicode 18.0 emoji-test.txt
(U+FE0F ignored, so a minimally-qualified form has the name of its
fully-qualified form; empty for a sequence Unicode does not list) and
the region of a flag (`DE`) or subdivision flag (`GB-ENG`), empty
otherwise.

OpenNLP's bundled annotations, emoji-annotations.txt (40 pictographs,
each with a name, a sentiment from -2 to 2, an entity type and a
category), are in `analysis/normalize/src/emoji/data/annotations.rs`, one entry per pictograph. The
sources and the notes of the name rows are the same on every row, so
they are constants next to the table. `emoji::opennlp_annotation`
looks a sequence up as `EmojiAnnotations.lookup` does, ignoring every
U+FE0F. `emoji::features` gives the feature strings of
`EmojiAnnotator.collectFeatures` in its order (`emojiSentiment=2`,
`emojiType=HEART`, `emojiCategory=SMILEYS_AND_EMOTION`,
`emojiRegion=DE`); a flag has no bundled row and gives `FLAG`, `FLAGS`
and its region, and a token that starts with an ASCII character gives
nothing.

Differences from OpenNLP:

- `EmojiRemove::sequences()` has no OpenNLP counterpart. OpenNLP's
  emoji remover replaces supplementary-plane code points, which removes
  text that is not emoji (`𝐀` U+1D400, `𠀀` U+20000, Gothic and other
  historic scripts), leaves the emoji of the Basic Multilingual Plane
  (`☺`, `❤`, `⌚`, `✈`) and breaks sequences into pieces: a ZWJ family
  becomes `" ‍ ‍ "` and `❤️` keeps its U+FE0F. The sequence mode
  removes exactly the emoji, each whole. Tests:
  `every_fully_qualified_emoji_becomes_one_space`,
  `zwj_families_skin_tones_keycaps_and_flags_are_one_space_each`,
  `text_default_symbols_with_and_without_the_selector`,
  `supplementary_mode_matches_opennlp`.
- `EmojiAnnotate` finds the sequences in the text. OpenNLP's
  `EmojiAnnotator` annotates one token at a time and depends on the
  tokenizer to hand it a whole sequence. It names every emoji Unicode
  lists, from emoji-test.txt, where OpenNLP names only its 40 bundled
  pictographs. Test: `annotator_reports_kind_name_and_region`.
- OpenNLP's `EmojiAnnotationJoin` (a caller's function that adds
  attributes per symbol and region) is not ported; a caller joins its
  own data on the layer's `name` and `region` columns.
- The data files are Rust constants, so OpenNLP's tests of the file
  parsers (wrong field counts, bad hexadecimal, unknown attributes) have
  no counterpart; the audits of the data (row counts, the two fold
  directions closing over each other, every fold source annotated,
  sentiment in -2..2, the attribute sources) are ported, and every name,
  group and subgroup of the bundled annotations is checked against the
  Unicode 18.0 emoji-test.txt.

Tests (`analysis/normalize/tests/emoji.rs`): the expectations of
`EmojiToEmoticonCharSequenceNormalizerTest`,
`EmoticonToEmojiCharSequenceNormalizerTest` (without the case that runs
OpenNLP's word tokenizer), `EmojiEmoticonsTest`, `EmojiAnnotationsTest`,
`EmojiAnnotatorTest`, the malformed flags of `EmojiFlagsTest` (through
`features`) and `EmojiAnnotatorUsageExampleTest`; every fully-qualified,
minimally-qualified and component row of emoji-test.txt removed as one
space with a placeholder covering the row, alone and between letters;
ZWJ families, skin tones, keycaps with and without U+FE0F, flags,
subdivision flags, text-default symbols with and without U+FE0F and
U+FE0E; 20,000 generated inputs for each mode (the sequence mode is
idempotent and its placeholders are the scanner's emoji sequences; the
supplementary mode equals a code-point-at-a-time reference of OpenNLP's
loop); the alignment of every output; and a pipeline that runs
`EmojiAnnotate` and `EmojiRemove`, reads both layers back and maps each
emoji into the view.

### Language detector preparation (`language_detection.rs`)

`language_detection::prepare(text)` prepares a text as OpenNLP's
`LanguageDetectorFactory.getContextGenerator()` does before counting
character n-grams: the five normalizers it chains, in its order, as one
`Chain` run through `apply`:

1. `EmojiRemove::supplementary()` (OpenNLP's deprecated
   `EmojiCharSequenceNormalizer`: every run of supplementary-plane code
   points becomes one space);
2. `Urls`;
3. `SocialMedia`;
4. `Numbers`;
5. `Shrink`.

The result is borrowed when nothing changed. `language_detection::chain()`
returns the chain itself for a pipeline that wants the view and its
alignment. The models OpenNLP distributes were trained on this
preparation, so each step keeps OpenNLP's output text; the improved
variants (`BoundedUrls`, `BoundedSocialMedia`, emoji sequences) are not
used here.

Tests (`analysis/normalize/tests/language_detection.rs`): the chain's ids in order, and
texts mixing emoji, URLs, addresses, hashtags, digits and repeated
characters compared with the five OpenNLP normalizers applied one after
another.

### Unicode normalization forms (`unicode_form.rs`)

`UnicodeForm` (id `turbo:normalize:unicode-form`) puts the text in NFC
(`UnicodeForm::nfc()`), NFKC (`nfkc()`), NFD (`nfd()`) or NFKD
(`nfkd()`), as UAX #15 defines them, on the Unicode 18.0 data of
`turbo-unicode`. The config hash is the hash of the form.

It runs `turbo_unicode::normalize::for_each_run` over the text, which
splits it before each character with combining class 0 whose quick check
value for the form is Yes. Nothing before such a point reorders or
composes with anything after it, so each run normalizes on its own. A run
the form leaves alone is kept; a run it changes is one `replace` with its
normalized text. One refinement: a run starts at the starter before the
characters that fail the quick check, since they may compose with it, and
when the normalized run begins with that same starter (nothing composed
with it, as with "x" before a Hangul syllable under NFD, or before U+2126
OHM SIGN under NFC) the starter is kept and only the rest of the run is
replaced. The starter has combining class 0, so no mark moves across it
and the rest of the run normalizes on its own to the rest of the output.

The alignment is therefore one edit per changed run, and a run cannot be
split further: "e" U+0301 becomes U+00E9, and no part of U+00E9 comes from
only one of the two source characters. Text already in the form (all
ASCII text, in every form) gives no edit and `apply` returns it borrowed.

OpenNLP counterparts: `NfcCharSequenceNormalizer` and
`NfkcCharSequenceNormalizer`. NFD and NFKD have no OpenNLP normalizer.

Differences from OpenNLP:

- Alignment. OpenNLP's two normalizers call `java.text.Normalizer` on the
  whole text and are not `OffsetAwareNormalizer`s, so
  `TextNormalizer.buildAligned()` rejects a pipeline that contains them
  (`AlignedNormalizerPipelineTest`). Here every form records an exact
  alignment and can sit anywhere in a `Chain`.
- Unicode version. The data is Unicode 18.0; a JVM normalizes with the
  data of its own Unicode version (15.0 for JDK 21), so characters
  assigned after that version can normalize differently. Inside the 15.0
  repertoire the output text is the same.

Cost, measured by the ignored test `cost_per_megabyte` in
`analysis/normalize/tests/unicode_form.rs` (release build, a 4-core container shared with
other builds; repeated runs vary by up to half):

| Text (1 MB) | NFC | NFD | NFKC | NFKD | NFC without the view |
|---|---|---|---|---|---|
| ASCII | 0.1 ms | 0.1 ms | 0.1 ms | 0.1 ms | 0.2 ms |
| mostly NFC (English with accented words, German, Greek, CJK) | 0.8 ms | 2.8 ms | 0.7 ms | 2.8 ms | 0.9 ms |
| the same text in NFD | 2.2 ms | 0.7 ms | 2.3 ms | 0.7 ms | 2.0 ms |

"NFC without the view" is `turbo_unicode::normalize::normalize` writing
the whole output as one replace. Writing the view per run costs about
the same as normalizing alone: the alignment adds one edit per changed
run and nothing per unchanged byte.

```sh
cargo test --release -p turbo-normalize --test unicode_form cost_per_megabyte -- --ignored --nocapture
```

Tests (`analysis/normalize/tests/unicode_form.rs`): every column of every line of
NormalizationTest.txt in all four forms, compared with
`turbo_unicode::normalize::normalize` and with the expected columns
(c2 = NFC(c1..c3), c3 = NFD(c1..c3), c4 = NFC(c4, c5) = NFKC(c1..c5),
c5 = NFD(c4, c5) = NFKD(c1..c5)); 300 texts joined from random
NormalizationTest columns; 3000 generated texts of combining marks,
Hangul jamo and syllables, compatibility characters and singletons. Each
output is checked for the invariants below (`check_view`). Also:
Hangul composition and decomposition, the kept starter before a Hangul
syllable, combining marks out of order, ligatures, fullwidth letters,
superscripts and circled digits (changed by NFKC and NFKD only), one edit
per changed run, no edits and a borrowed result for text already in the
form, the expectations of OpenNLP's `UnicodeCharSequenceNormalizerTest`,
and distinct config hashes.

The invariants every output of these views is checked for: the edits are
on character boundaries, in order, and rebuild the output; every edit is
the form of its own source text; every source span maps forward and back
to a span that covers it (`folds_support::check_alignment`); and for every
output span `s` (all spans of short outputs, 300 random ones of long
outputs), the source span `p` it maps back to maps forward to a span that
covers `s`, and the source text of `p` normalizes to exactly the output
text `p` maps forward to.

### Case folding (`case_fold.rs`)

`CaseFold` (id `turbo:normalize:case-fold`) lowercases or case folds the
text. The config hash is the hash of the mode (`CaseFoldMode`):

| Constructor | Mapping |
|---|---|
| `CaseFold::lowercase()` | `String.toLowerCase(Locale.ROOT)`: the simple lowercase mapping, U+0130 to "i" U+0307, and U+03A3 to final sigma U+03C2 where the Final_Sigma condition holds (`turbo_unicode::java::lowercase_at`). |
| `CaseFold::simple()` | Simple case folding, CaseFolding.txt status C and S: one character to one. |
| `CaseFold::full()` | Full case folding, status C and F: "ß" to "ss", U+FB00 to "ff", U+0390 to three characters. |
| `CaseFold::turkic_simple()`, `turkic_full()` | The same with the status T rows: "I" to dotless U+0131, U+0130 to "i". |

Each character whose mapping differs from itself is its own `replace`, so
an expansion maps back to exactly its source character. ASCII is tested
eight bytes at a time; a stretch of ASCII without uppercase letters is
skipped without decoding. Only U+03A3 under `lowercase` looks at its
neighbours (through `turbo_unicode::java::lowercase_at`, which applies the
JDK's final sigma condition to the whole text); the folds are context
free.

Folding does not keep a normalization form: the full fold of U+01F0 is
"j" U+030C, which is not NFC, and the combining mark U+0345 (class 240)
folds to the letter U+03B9 (class 0). Canonical caseless matching
(Unicode section 3.13, D145) is NFD, fold, NFD:
`Chain::new().then(UnicodeForm::nfd()).then(CaseFold::full()).then(UnicodeForm::nfd())`.

OpenNLP counterparts: `CaseFoldCharSequenceNormalizer` with
`Locale.ROOT` (`lowercase`) and `FullCaseFoldCharSequenceNormalizer`
(`full`). Parity finding: OpenNLP's `CaseFoldCharSequenceNormalizer` does
not case fold, it lowercases. Lowercasing and the simple fold of
CaseFolding.txt differ on: U+00B5 MICRO SIGN (folds to Greek mu), final
sigma U+03C2 (folds to sigma), U+0345 (folds to iota), U+03D0 and the
other Greek letter-symbols, U+1E9B, the Cyrillic small letter variants
U+1C80 to U+1C88, the Cherokee small letters (which fold to the capitals,
U+13F8 to U+13F0 and U+AB70 to U+13A0), and U+0130, which lowercases to
"i" U+0307 and has no simple fold. `CaseFold::simple()` is the simple
fold; `CaseFold::lowercase()` keeps OpenNLP's output.

Differences from OpenNLP:

- Alignment. `CaseFoldCharSequenceNormalizer` is not offset-aware, so
  OpenNLP's `buildAligned()` rejects it, although `Locale.ROOT`
  lowercasing changes the length of the text (U+0130 becomes two
  characters). Here it has an exact alignment.
- Turkish. OpenNLP's `CaseFoldCharSequenceNormalizer` takes a `Locale`
  and lowercases by its rules. `CaseFold` offers no locale-dependent
  lowercasing; for Turkish and Azerbaijani it offers the Turkic folds of
  CaseFolding.txt, which map "I" to "ı" and U+0130 to "i" as Turkish
  lowercasing does. A decomposed "I" U+0307 folds to "ı" U+0307 (Turkish
  lowercasing gives "i"); `UnicodeForm::nfc()` before the fold composes it
  to U+0130 first, and the chain gives "i" for both spellings. The
  Lithuanian rules of SpecialCasing.txt (adding U+0307 above "i" before
  accents) are not applied.
- The simple fold and the Turkic folds are additions.
- Unicode version. CaseFolding.txt 18.0.0 has 1501 C rows and 105 F rows
  (1606); OpenNLP bundles 17.0.0 (1585) for the full fold, and
  `toLowerCase` uses the JVM's own Unicode version (15.0 for JDK 21).
  Inside those repertoires the output is the same.
- Neighbouring changes. Each character is its own `replace`, as OpenNLP
  records each expansion on its own. `ViewBuilder` and `compose` currently
  join touching edits, so in "ßß" each "ss" maps back to both characters
  where OpenNLP's `consecutiveExpansionsEachMapToTheirOwnSource` maps it
  to its own; the mapped span always covers the character's own source.

Cost, measured by the ignored test `cost_per_megabyte` in
`analysis/normalize/tests/case_fold.rs` (same conditions as for the
normalization forms): lowercase 2 to 3 ms per MB and the full fold about
2 ms per MB on ASCII text with capitals; about 1.5 ms per MB for both on
mostly ASCII Latin text; on Greek, where nearly every character is
looked up, 6 ms per MB for lowercase and 4 ms per
MB for the full fold.

Tests (`analysis/normalize/tests/case_fold.rs`): every row of CaseFolding.txt (parsed in
the test) through `full`, `simple`, `turkic_full` and `turkic_simple`,
alone and between other characters; every code point through `full`,
`simple` and `turkic_full` against the rows (characters without a row are
unchanged); every code point through `lowercase` against
`java::to_lowercase` of the same text; Final_Sigma ("ΣΑΣ" to "σας", "Σ"
alone, sigma before and after a combining mark, sigma after a digit or a
space, two sigmas); U+0130, "I", "ß" and U+1E9E in every mode; the
characters on which lowercasing and the simple fold differ; the
NFD-fold-NFD chain and the facts behind it; the Turkic fold after NFC;
every expectation of `CaseFoldCharSequenceNormalizerTest`,
`FullCaseFoldCharSequenceNormalizerTest` (the alignment ones with UTF-8
offsets) and the case fold cases of `UnicodeCharSequenceNormalizerTest`;
the whitespace and full fold pipeline of `composesIntoTheOffsetAwarePipeline`
and `identityPipelinePreservesEveryOffset`; 3000 generated texts that
cross the eight-byte ASCII test at every offset, and 2000 generated texts
in all five modes against a reference fold, each checked for the
invariants of the previous section. A unit test in `analysis/normalize/src/case_fold.rs` checks
the eight-byte ASCII test with each stop byte at each position.

### Accent folding (`analysis/normalize/src/accent_fold/mod.rs`) and the run pass (`analysis/normalize/src/fold_runs.rs`)

`AccentFold` (id `turbo:normalize:accent-fold`) removes accents. It does
what `AccentFoldCharSequenceNormalizer` does to the whole text: NFD,
then one pass over the decomposed characters that drops every
non-spacing mark (general category Mn) whose base belongs to a fold
script and maps the stroke letters and ligatures to ASCII, then NFC. The
base of a mark is the last character before it that is not a
non-spacing mark; a mapped stroke letter counts as Latin; a mark with no
base before it is kept. "café" becomes "cafe", "Straße" "Strasse", Greek
ά α, Cyrillic й и. Devanagari vowel signs and Arabic and Hebrew points
are kept with the default scripts.

Options: `AccentFold::new()` (Latin, Greek and Cyrillic, OpenNLP's 19
stroke letters), `AccentFold::for_scripts(&[Script])`,
`with_stroke_letters(StrokeLetters::Off | OpenNlp | Extended)` and
`AccentFold::extended()` (the default scripts with the extended list).

The alignment is recorded per source character. The text is cut into
runs (`analysis/normalize/src/fold_runs.rs`, shared with the confusable
skeleton). A run starts at a boundary character: one whose decomposition
starts with a character that is not a non-spacing mark, has combining
class 0 and has NFC_Quick_Check Yes. That character is never removed,
so it is the base of the marks after it whatever came before, canonical
ordering cannot move anything across it, and NFC cannot compose it with
anything before it. The fold of the whole text is therefore the folds of
the runs put together. Inside a run every character is also folded on its own; when
those pieces put together equal the fold of the run, each character that
changed is one edit ("café": é to e; "e" U+0301: a deletion of the
U+0301), otherwise the run is one edit. The second case happens where
normalization works across characters: marks out of canonical order on
a base whose marks are kept (Hebrew alef, U+0301, U+0323 is written with
U+0323 first) and two characters that compose once a removed mark
between them is gone (with Hangul folded, U+1100 U+0301 U+1161 becomes
U+AC00). A run of one character that neither decomposes nor maps (all
of ASCII, Hangul syllables) is skipped without work.

Differences from OpenNLP:

- The output text is the same as OpenNLP's for every input on the same
  Unicode data. OpenNLP's class has no aligned form; here every edit is
  aligned.
- The data is Unicode 18.0 (general category, scripts, normalization),
  where OpenNLP uses the JVM's (Unicode 15.0 on JDK 21). Characters
  assigned after 15.0 fold here and are left alone there.
- `StrokeLetters::Extended`, off by default, adds 264 atomic Latin
  letters that NFD does not decompose, so mark removal cannot reach them
  and OpenNLP's list leaves them. `EXTENDED_MODIFIED` (244 letters) is
  every letter named "LATIN CAPITAL LETTER x WITH ..." or "LATIN SMALL
  LETTER x WITH ..." in UnicodeData.txt, x one of A to Z, with no
  decomposition: the letters with a stroke, bar, hook, tail, curl, belt,
  loop, descender or other modification of an ASCII letter (Ŧ, ƀ, Ɨ, ɓ,
  ƈ, ɗ, ɠ, ɦ, ɫ, ɲ, ʂ, ȥ and so on), mapped to x in the case the name
  gives. Lucene's ASCIIFoldingFilter folds letters of this kind as
  well. `EXTENDED_OTHER` (20 letters) has the atomic letters the naming
  rule does not cover: the ligature Ĳ ĳ (IJ ij) and the digraphs Ǆ ǅ ǆ
  Ǉ ǈ ǉ Ǌ ǋ ǌ Ǳ ǲ ǳ (DZ Dz dz LJ Lj lj NJ Nj nj DZ Dz dz), which are
  one letter in Unicode and two in ASCII; Ŀ ŀ (L l, the Catalan L with
  middle dot); ſ (s, the long s); Ŋ ŋ (N n, eng, mapped to its base
  letter as the hooked letters are); ĸ (q, kra, written q in the
  Greenlandic orthography of 1973 that replaced it). The dental click
  letter ǀ is not added: it is not a modified Latin letter. "Łódź"
  becomes "Lodz" with either list; "Ŧ", "ƀ", "Ĳsselmeer" and "ǅ" change
  only with the extended one.

Tests (`analysis/normalize/tests/accent_folds.rs`): every expectation of
`AccentFoldCharSequenceNormalizerTest` (Latin, Greek and Cyrillic
accents, all 19 stroke letters, the leading mark kept, Devanagari,
Arabic and Hebrew marks kept, script and stroke configuration; the case
folding pipeline test with the case fold done by `to_lowercase`); the
edits of single characters and of separate marks; the two run cases
above; NFC applied to text of other scripts (Devanagari NA and NUKTA
become U+0929) and Hangul left alone; `EXTENDED_MODIFIED` derived again
from UnicodeData.txt by the naming rule and compared, and
`EXTENDED_OTHER` checked to be Latin letters that NFD leaves alone; and
generated texts (4000 each in four configurations, from a pool of
precomposed letters, marks of several combining classes, Mn of class 0,
Mc, Me, Greek, Cyrillic, Hebrew, Arabic, Devanagari, Tamil, Hangul
syllables and jamo, Tibetan U+0F73, kana with the voicing mark) compared
with a reference that runs the Java algorithm on the whole text, with
every alignment checked.

### German umlauts (`analysis/normalize/src/german.rs`)

`GermanUmlauts` (id `turbo:normalize:german-umlauts`) writes the umlauts
and sharp s as DIN 5007-2 spells them in ASCII: ä ö ü to ae oe ue, Ä Ö Ü
to Ae Oe Ue, ß to ss, ẞ (U+1E9E) to SS. Each character is one edit;
the output "SS" maps back to the one U+1E9E. A vowel followed by U+0308
COMBINING DIAERESIS is not changed; composing first (NFC) makes it the
precomposed letter.

OpenNLP counterpart: `GermanUmlautCharSequenceNormalizer`. The default
matches it.

Differences from OpenNLP: `GermanUmlauts::uppercase_aware()` writes a
capital umlaut with two capitals (AE, OE, UE) when it is part of an
uppercase word: when the next letter is uppercase, or when no letter
follows and the letter before is uppercase. Combining marks are passed
over. "ÜBER" becomes "UEBER" instead of "UeBER", "MÜ" becomes "MUE",
while "Öl" stays "Oel" and a lone "Ö" "Oe". In an all-capitals word the transliteration is written in
capitals (the machine-readable zone of a passport writes MÜLLER as
MUELLER); "UeBER" mixes the two spellings.

Tests (`analysis/normalize/tests/accent_folds.rs`): every expectation of
`GermanUmlautCharSequenceNormalizerTest` (with the U+1E9E offsets
checked as UTF-8 bytes: output 1..3 maps to source 1..4), the
uppercase-aware cases, and 4000 generated texts compared with a
per-character reference in both modes, each edit checked to be one
character's expansion.

### Mojibake repair (`analysis/normalize/src/mojibake.rs`)

`Mojibake` (id `turbo:normalize:mojibake`) undoes text that was encoded
in one encoding and decoded in another: "cafÃ©" becomes "café",
"donâ€™t" becomes "don’t", and text encoded twice is undone
twice. The work is done by `turbo_encoding::repair`, a port of the
encoding fix of ftfy 6.3.1, by Robyn Speer (Apache-2.0; see NOTICE):
ftfy's badness heuristic decides whether a line looks like mojibake, and
the line is then encoded back with each of ftfy's single-byte code pages
(Latin-1, the sloppy windows-1252, 1251, 1250, 1253, 1254 and 1257,
ISO-8859-2, Mac OS Roman, code page 437) and decoded as UTF-8, with
ftfy's byte repairs in between. The text is split into lines as ftfy's
`fix_text` splits it, so one damaged line is fixed beside a fine one.

Every character of the result carries the span of the source characters
it came from, through every encode and decode. A fix is one edit over the
source characters it replaces ("Ã©", four bytes, to "é", two bytes); a
U+FFFD written for a UTF-8 sequence that lost a byte maps to the whole
damaged sequence; unchanged text is kept as it is.

The heuristic can be wrong in both directions, as ftfy's is: rare
mojibake goes unfixed, and rare legitimate text that looks like mojibake
is changed. `ArtifactDetector` with `Mojibake` or `MojibakeSequence`
reports suspect spans without changing them.

No OpenNLP counterpart. The rest of ftfy's `fix_text` (HTML entities,
quotes, width, ligatures, line breaks, control characters,
normalization) is not part of this normalizer; other normalizers here
cover most of it.

Tests: `analysis/encoding/tests/repair.rs` runs all 161 cases of ftfy's
`test_cases.json` (and 151 variants with an extra layer of mojibake),
checking the repaired text and the plan of steps against what ftfy
itself returns, every alignment edit against the source, and that every
character maps back to its source characters. The normalizer's own tests
check one edit per fix and that fine text in several scripts is left
unchanged.

### Confusable skeleton (`analysis/normalize/src/confusables.rs`)

`ConfusableSkeleton` (id `turbo:normalize:confusable-skeleton`) writes
the skeleton of UTS #39 section 4: NFD, each character replaced by its
prototype from confusables.txt, NFD again. Text that looks alike has the
same skeleton: "раураl" with Cyrillic а and "paypal" both become
"paypal", "m" becomes "rn", 1 and I become "l", U+2026 becomes "...". The
skeleton is for comparing and indexing.

The alignment is recorded per source character with the run pass of
accent folding. A boundary is a character whose decomposition starts
with a character of combining class 0 that has no prototype, or whose
prototype also decomposes to a class 0 character first; both NFD passes
then split the text before it. A character in NFD with no prototype is
kept without work, which covers all of ASCII except the eight characters
confusables.txt maps (" % 0 1 I ` m |), the fast path of
`Confusables.skeleton` done per character.

OpenNLP counterpart: `ConfusableSkeletonCharSequenceNormalizer` and
`Confusables.skeleton`.

Differences from OpenNLP:

- The data is confusables.txt 18.0.0 (through `turbo_unicode`); OpenNLP
  bundles 17.0.0. A line diff of the two files has about 2850 changed
  lines. For example 18.0.0 maps U+030B COMBINING DOUBLE ACUTE ACCENT
  to two acute accents, so "ő" becomes "o" U+0301 U+0301 here and "o" U+030B in
  OpenNLP.
- OpenNLP's class has no aligned form; here every edit is aligned.

The differential (`analysis/fuzz/src/runs/normalize.rs`) tells this
difference from any other with the 17.0.0 data itself: the 2526 source
code points whose 17.0.0 prototype is not their 18.0.0 one are listed in
`analysis/fuzz/src/runs/confusables_17.rs`, and a JVM text is the
documented difference only when it is the 17.0.0 skeleton of the source
while the port's is the 18.0.0 one.

Tests (`analysis/normalize/tests/confusables.rs`): every expectation of
`ConfusableSkeletonTest`
and `ConfusablesTest` that is about the skeleton (the null argument and
`TermAnalyzer` tests have no counterpart), the edits of "com" and of the
Cyrillic spoof, agreement with `turbo_unicode::confusables::skeleton`,
the Unicode 18.0 mapping above, every source of confusables.txt alone
and between marks compared with a reference written after
`Confusables.skeleton` (including its fast path), and 4000 generated
texts from confusables.txt sources, letters and marks of other
scripts and frequent combining marks compared with the same reference,
with every alignment checked.

### Dehyphenation

`Dehyphenate` (id `turbo:normalize:dehyphenate`) rejoins words split at a
line end: "litiga-\ntion" becomes "litigation". A join needs a letter,
optionally followed by combining marks (general category M), a hyphen (U+002D, U+00AD SOFT HYPHEN or U+2010 HYPHEN), exactly one line
break from the `UnicodeWhitespace` line-break set (CR LF counts as one),
any amount of non-breaking whitespace from the same table (the
indentation of the next line, including U+00A0), and a letter. Letters
are `Character.isLetter` letters (`turbo_unicode::java::is_letter`,
general category L). The hyphen, the break and the indentation are
deleted as one edit, so the joined word maps back to the whole source
stretch.

Nothing else joins: a hyphen without a line break after it, U+2011
NON-BREAKING HYPHEN (the writer said the word must not break there), a
second line break (a blank line, or a form feed page break), and a
continuation that starts with a digit, punctuation or a combining mark.

OpenNLP counterpart: `DehyphenationCharSequenceNormalizer`.

Differences from OpenNLP: OpenNLP requires the character right before
the hyphen to be a letter, so it leaves a word that ends in a combining
mark split. That misses decomposed (NFD) text, which PDF extractors and
macOS file systems produce ("cafe" U+0301 "-\nteria"), and Devanagari,
Bengali and other Indic words, which often end in a vowel sign such as
U+0940. Here the letter may be followed by any number of combining
marks; both examples join. A mark with no letter before it, or a
continuation that starts with a mark, still does not join. Apart from
this the default mode (`Dehyphenate::new()`) matches OpenNLP.

Without a dictionary the default mode cannot tell a word broken by the
typesetter from a compound that was broken at its own hyphen:
"well-\nknown" becomes "wellknown". `Dehyphenate::keep_compounds()` is an
addition to OpenNLP that keeps the hyphen when it is U+002D and the
letter after the break is uppercase (`turbo_unicode::props::uppercase`);
only the break and the indentation are deleted. It keeps
"Wi-\nFi" as "Wi-Fi" and "Müller-\nLüdenscheidt" as
"Müller-Lüdenscheidt". `Dehyphenate::with_lexicon` (along with
`Dehyphenate::with_hunspell` and `Dehyphenate::with_morfologik`, which
accept `turbo_hunspell::Dictionary` and
`turbo_morfologik::MorfologikDictionary` through `Dictionary`) decides
from a lexicon (`DehyphenateMode::Lexicon`): a U+002D hyphen-minus is
deleted when the joined word is in the lexicon ("litiga-\ntion" becomes
"litigation", "Mc-\nDonald" becomes "McDonald"), kept when the joined
word is not in the lexicon and either the hyphenated form or both halves
are in the lexicon ("well-\nknown" becomes "well-known"), and otherwise
falls back to the uppercase check. Soft hyphens (U+00AD) and U+2010
mark typesetter breaks and are always deleted.

Tests (`analysis/normalize/tests/dehyphenate.rs`): every expectation of OpenNLP's
`DehyphenationCharSequenceNormalizerTest` (each line-break form, the form
feed page break, the three hyphens and the non-breaking one, indentation,
several joins, the cases left alone, the alignment of the joined word,
of the text around it, of CR LF with indentation, of a supplementary
letter before the hyphen and of an emoji after the join, with UTF-16
offsets rewritten as UTF-8 byte offsets); Cyrillic and German words;
words ending in one or more combining marks, in Latin NFD and
Devanagari, and marks without a letter; the compound mode, including the
alignment of the kept hyphen; and 4000 generated texts in both modes
compared with a reference written over a character vector.

### Alignment checks

The tests of these normalizers run every input through
`tests/folds_support/check_alignment`, which checks that the edits are in
order on both sides and rebuild the output from the source, that every
edit is one the normalizer may make (a whitespace run to "", " " or "\n";
a code point to its table replacement; an invisible character to ""; a
hyphen, line break and indentation to ""), that every output character
maps back to itself outside the edits and to the whole parent of its edit
inside one, and that every span of whole source characters maps forward
and back to a span that covers it, apart from deleted bytes at its edges
(deleted bytes belong to neither neighbour), and no wider than the edits
it overlaps.

## Notes for OpenNLP

(Things found while porting that OpenNLP could fix or improve.)

- Whitespace, line-break and paragraph normalizers: the collapse and the
  trim are two aligned passes whose alignments are composed with
  `andThen`, and the quote normalizer runs its two sets the same way.
  Each can be one pass that records the final alignment directly; the
  output is the same and the intermediate string and alignment arrays
  are not needed.
- `ParagraphPreservingWhitespaceCharSequenceNormalizer` writes a
  paragraph boundary as a single "\n", which the same normalizer reads
  back as one line break. Running it twice therefore joins the
  paragraphs ("one\n\ntwo" becomes "one\ntwo", then "one two"). Writing
  "\n\n" for a boundary would make it idempotent.
- `QuoteCharSequenceNormalizer` misses U+2E42 DOUBLE LOW-REVERSED-9
  QUOTATION MARK, which has the `Quotation_Mark` property.
- `EllipsisCharSequenceNormalizer` misses U+2024 ONE DOT LEADER and the
  vertical forms U+FE19 and U+FE30, whose compatibility decompositions
  are full stops.
- `InvisibleCharSequenceNormalizer` keeps the deprecated format
  characters U+206A to U+206F, which have no effect on display.
- `DigitCharSequenceNormalizer` depends on the Unicode version of the JVM
  it runs on (`Character.digit`), so the same input gives different
  output on JDK 21 (Unicode 15.0) and later JDKs. Its tests do not state
  which Unicode version they expect.
- `CursorArtifactDetector` reports two characters of genuine text as
  mojibake when an accented capital stands before a typographic dash,
  quote or ellipsis whose Windows-1252 byte is a continuation byte:
  "CAFÉ–BAR" (C9 96, the UTF-8 of U+0256), "Ä…" (C4 85). Requiring a
  two-byte image to decode to the ranges mojibake comes from (Latin-1
  Supplement, Latin Extended-A, Greek, Cyrillic) removes these.
- `CursorArtifactDetector` needs a whole run of non-ASCII characters to
  decode, so one genuine character next to the image ("cafÃ©…") or one
  byte replaced by U+FFFD hides the mojibake; decoding one sequence at a
  time finds it.
- `StructuralNoiseScorer.isSurroundingPunctuation` takes a `char`, so
  supplementary-plane punctuation is never trimmed and the token is
  skipped as non-ASCII. Testing code points with
  `Character.getType(int)` fixes this.
- `StructuralNoiseScorer.overlapsAny` tests every exclusion for every
  token and every gap. Sorting and joining the exclusions once makes each
  test a binary search.
- `SymbolJoinerCharSequenceNormalizer` and the `symbol-joiner` add-on are
  the same class. It maps single symbols to words and joins nothing, so
  its name does not describe what it does.
- `EnglishContractionCharSequenceNormalizer` computes every word
  boundary of the text (`WordSegmenter.boundaries`) when it finds one
  candidate. The UAX #29 rules at one position read only a few
  characters around it, so checking the two ends of each candidate costs
  a constant amount of work instead of a pass over the whole text. It
  also leaves unambiguous forms such as `y'all`, `this'll` and `may've`
  unexpanded.
- `UrlCharSequenceNormalizer` and `SocialMediaCharSequenceNormalizer`
  run two and four passes, each copying the text. Each can be one pass
  with the same output. For URLs, the only care needed is to cut an
  address's domain at a URL that starts inside it. For social media, the
  only interaction is that the space a hashtag becomes can end an `rt`
  unit, which makes `rt#tag` one replacement; the class documentation
  does not mention it. `ShrinkCharSequenceNormalizer`'s three passes
  can likewise be one.
- `BoundedUrlCharSequenceNormalizer`'s address scan is ASCII only, so it
  leaves `José@example.com` in place and turns `éa@b.co` into `"é "`,
  removing part of an address and leaving the rest. Widening the
  local-part and domain sets to non-ASCII letters, digits and marks
  removes whole internationalized addresses.
- `SvgHeader` catches `XMLStreamException` only. With DTD support off,
  JDK 21's StAX parser throws `MissingResourceException`
  ("InvalidCharInDTD") for a character XML does not allow inside an
  internal subset, so `CursorAssetDetector.detect` throws on any text
  holding such a header as a bare run or a `data:` URI, for example
  `<!DOCTYPE svg [<!ENTITY a\0 'b'>]><svg .../>` encoded. Catching
  `RuntimeException` around the parse, or rejecting the payload when the
  parser fails in any way, keeps detection total.
- With `SUPPORT_DTD` false the JDK skips the internal subset unread up to
  its first `]`: `<!DOCTYPE svg [<!A<TLIST>]>` is accepted, while a `]`
  inside a comment or literal there ends the subset and fails the
  document. `SvgHeader`'s answer for a DOCTYPE thus follows the parser's
  shortcuts rather than XML. The JDK also reads XML 1.0 names by the
  fourth edition's tables.
- `SvgHeader`'s answer can depend on bytes after the root start tag: the
  JDK's UTF-8 reader decodes ahead and rejects a four-byte sequence above
  U+10FFFF up to about 8 KB later (`<svg xmlns='...'/>` followed by
  F4 90 80 80 is not SVG), while other malformed bytes there are ignored.
  Bounding the stream at the end of the start tag, or decoding leniently
  after it, would make the answer depend on the header alone.
- `CursorAssetDetector` matches base64 prefixes of every signature before
  decoding a header, and `KnownMagics.formatOf` then tests the decoded
  header against the same signatures. Decoding the first base64 quantum
  and looking its first byte up in a 256-entry table filters the same
  runs with one lookup and no prefix strings.
- `SocialMediaCharSequenceNormalizer` replaces emoticons inside words
  (`expo` becomes `"e o"`, `xoxo` two spaces, `:people` `" eople"`), and
  its hashtags swallow the punctuation after them (`#win!`). Its retweet
  word boundary depends on how the JDK regular expression engine walks
  back over UTF-16 units, so a supplementary-plane combining mark never
  blocks it.
- `CursorArtifactDetector` reports two characters of genuine text as
  mojibake when an accented capital stands before a typographic dash,
  quote or ellipsis whose Windows-1252 byte is a continuation byte:
  "CAFÉ–BAR" (C9 96, the UTF-8 of U+0256), "Ä…" (C4 85). Requiring a
  two-byte image to decode to the ranges mojibake comes from (Latin-1
  Supplement, Latin Extended-A, Greek, Cyrillic) removes these.
- `CursorArtifactDetector` needs a whole run of non-ASCII characters to
  decode, so one genuine character next to the image ("cafÃ©…") or one
  byte replaced by U+FFFD hides the mojibake; decoding one sequence at a
  time finds it.
- `StructuralNoiseScorer.isSurroundingPunctuation` takes a `char`, so
  supplementary-plane punctuation is never trimmed and the token is
  skipped as non-ASCII. Testing code points with
  `Character.getType(int)` fixes this.
- `StructuralNoiseScorer.overlapsAny` tests every exclusion for every
  token and every gap. Sorting and joining the exclusions once makes each
  test a binary search.
- `SpellCheckingCharSequenceNormalizer` compares `minTokenLength` with
  `String.length()`, UTF-16 units, so a token with a supplementary
  character is corrected one code point earlier than the documentation
  suggests. `SymSpell.lookupCompound` likewise splits a token at every
  UTF-16 index (looking up halves of surrogate pairs) and its
  `length() == 1` test treats one supplementary character as two.
  `codePointCount` and code point split points fix both.
- In compound mode `SpellCheckingCharSequenceNormalizer` applies none of
  the guards its documentation lists and does not lowercase: with the
  tiny test dictionary, "HER, user@example.com yt http://example.com/teh
  calendar" becomes "of of user@example.com it http://example.com/teh
  calendar", and "Hich user@example.com With to Mof" becomes "with
  user@example.com with to of". The documentation could say so, or the
  guarded tokens could be kept as fixed words between the phrases that
  are corrected.
- With `maxDictionaryEditDistance` 0, `editsPrefix` still indexes every
  one-deletion string of each word (the recursion adds a level before it
  tests the distance), which no lookup can use.
- `SymSpell` keeps every delete as a `String` key. Keying the deletes by
  a 64-bit hash and checking each candidate with a subsequence test
  gives the same candidates in the same order with a fraction of the
  memory.
- `SymSpellModelSerializer.create` ignores bytes after the last bigram,
  and a count above `Long.MAX_VALUE` (read as negative) fails with an
  `IllegalArgumentException` from `SymSpellModel` rather than the
  `IOException` the other corrupt fields give.
- `SymSpellModel` keys bigrams as `w1 + " " + w2`, which is ambiguous
  when a word contains a space (a word read from the model format can).
- The add-on at the commit the test data was made from does not compile
  against OpenNLP's `kristian-3.x-features` branch: it calls
  `ArgumentChecks`, `ResourceLimits.requireWithinMaxEntries`,
  `MailAddressScan.isAddress`, `StringUtil.stripByteOrderMark` and
  `StringUtil.splitNonEmpty`, which that branch does not have.
- `EmojiCharSequenceNormalizer` replaces supplementary-plane code points,
  not emoji: it removes mathematical alphanumerics, CJK Extension B to
  J ideographs and historic scripts, keeps every emoji of the Basic
  Multilingual Plane (`☺`, `❤`, `⌚`), and turns a ZWJ family into spaces
  around its joiners and `❤️` into a space and U+FE0F. A UTS #51 sequence
  scan with the Emoji_Presentation rule removes whole emoji and nothing
  else. Its legacy mode also removes every `-`.
- `EmojiAnnotator` annotates tokens, so a token that holds an emoji and
  other text, or a tokenizer that splits a ZWJ sequence, loses the
  emoji. Its names cover only the 40 bundled pictographs, while
  emoji-test.txt names every RGI emoji; the bundled names could be read
  from emoji-test.txt instead of copied, so they follow each Unicode
  version.
- `NfcCharSequenceNormalizer`, `NfkcCharSequenceNormalizer` and
  `CaseFoldCharSequenceNormalizer` are not offset-aware, so
  `buildAligned()` rejects them. All three can record an exact alignment:
  normalization changes text only inside runs that start at a starter
  whose quick check is Yes, and `Locale.ROOT` lowercasing maps each
  character on its own except U+03A3, whose context is only needed to
  choose between two characters of the same length.
- `CaseFoldCharSequenceNormalizer` is named a case fold but lowercases
  (`String.toLowerCase`). The two differ on U+00B5, final sigma, U+0345,
  the Greek letter-symbols, the Cyrillic small letter variants U+1C80 to
  U+1C88 and the Cherokee small letters. Either the name or the
  documentation could say so, and a simple fold (CaseFolding.txt C and S)
  could sit next to the full one.
- `FullCaseFoldCharSequenceNormalizer` says its input is expected in NFC.
  Canonical caseless matching (Unicode section 3.13, D145) needs NFD
  before and after the fold: the fold of U+01F0 is "j" U+030C, which is
  not NFC, and U+0345 folds from a combining mark to a letter.
- `AccentFoldCharSequenceNormalizer` and
  `ConfusableSkeletonCharSequenceNormalizer` have no aligned form. Both
  can be aligned per character: cut the text before each character whose
  decomposition starts with a starter that the fold cannot remove or
  combine (for the skeleton: whose prototype also starts with a
  starter), fold each piece, and attribute the result to single
  characters when their separate folds add up to it.
- `AccentFoldCharSequenceNormalizer`'s stroke list leaves the other
  atomic Latin letters that NFD does not decompose, such as Ŧ ŧ
  (Northern Sami), ƀ, Ɨ ɨ, Ĳ ĳ, Ŋ ŋ and the hooked letters of African
  orthographies (ɓ ɗ ƙ ƴ).
- `GermanUmlautCharSequenceNormalizer` writes "ÜBER" as "UeBER".
- `Confusables` bundles confusables.txt 17.0.0; 18.0.0 changes about
  2850 of its lines.
- `DehyphenationCharSequenceNormalizer` checks only the code unit before
  the hyphen with `Character.isLetter`, so a word that ends in a
  combining mark (NFD text, Indic vowel signs) is never joined. Skipping
  back over general category M before the letter test fixes this.

## Tests

`cargo test -p turbo-unicode -p turbo-normalize -p turbo-normalize-layers`.

### JVM golden tests

`analysis/normalize/tests/goldens.rs` runs every normalizer that has an
OpenNLP counterpart over the 61 texts of
`analysis/goldens/data/inputs/norm/` and compares the result with the
`turbo:view` layer in `analysis/goldens/data/normalize/<name>/norm/<id>.golden`
using `turbo_goldens::matches_golden_key`: the rebuilt text first, then
the edits one for one (Rust edit i against golden row i). Both sides keep
one edit per replaced stretch, so a normalizer whose golden comes from
OpenNLP's `normalizeAligned` must match exactly.

The goldens of `emoji`, `nfc`, `nfkc`, `case-fold`, `case-fold-tr`,
`accent-fold`, `confusable-skeleton`, `url`, `bounded-url`, `social-media`,
`number`, `shrink`, `symbol-joiner` and the default chain come from
OpenNLP methods without an alignment, so the generator rebuilt their edits
by a diff on code points and an edit boundary is the diff's choice. For
those the text must still be equal, and differing edit lists are accepted
when each region of the source that either list changes comes out the
same under both. The texts where an output differs on purpose are listed
in the test with the section of this file that records the difference.
`emoji` runs `EmojiRemove::supplementary()`, which is OpenNLP's output;
the sequence mode is a documented difference.

### HTML/XML markup (`markup.rs`) and character references (`entities.rs`)

`HtmlTagStripNormalizer` (`turbo:normalize:html-strip`) removes markup
with a cursor state machine: element tags (a quoted attribute value may
hold `>`), comments (including the abruptly closed `<!-->` and
`<!--->` forms and the `--!>` close), doctypes (a `>` inside the
internal subset does not end one), processing instructions, and the
bodies of `<script>` and `<style>` elements. CDATA wrappers drop but
their inner text stays, and `<title>`/`<textarea>` bodies stay as text.
A `<` that starts no valid markup is literal. Removed block-level tags
leave one space behind (never two in a row) unless configured off, so
words across `</p><p>` cannot fuse while `un<b>break</b>able` stays
one word; an unterminated construct drops to the end of input. This is
a text-extraction stripper, not a parser: it builds no tree, checks no
nesting, and a start tag closed with `/>` is treated as having no body
even for `script` (where HTML proper would keep reading raw text).

`HtmlEntityDecodeNormalizer` (`turbo:normalize:html-entities`) decodes
character references in one pass that never rescans its own output, so
`&amp;lt;` becomes the literal text `&lt;`. Named references cover the
full WHATWG table (`entities_table/`, 2,231 names sorted for binary
search; the 106 legacy names also match without their semicolon, and
the longest matching name wins). Numeric references, decimal or hex,
with or without the semicolon, apply the spec replacements: NUL,
surrogates and values above U+10FFFF become U+FFFD, and C1 controls
remap through the Windows-1252 table. An `&` starting no valid
reference is kept. Both normalizers record one edit per removed or
decoded stretch, so spans found on the cleaned view (entities,
chunks, answers) project back to the exact raw markup bytes through
the view alignment; `analysis/normalize/tests/markup.rs` holds the
adversary fixtures, including that byte-for-byte projection. There is
no OpenNLP counterpart to either normalizer.


## Extracted PDF evaluation

Licensed page extracts, hand-labelled UTF-8 spans, attribution, pinned hashes
and a Rust reproduction helper are in
[`analysis/normalize/data/extracted-pdf`](../analysis/normalize/data/extracted-pdf/README.md).
Eleven complete pages from four CC-BY-4.0 papers provide 50,994 bytes and
6,555 whitespace-delimited tokens. The two original ACL page pairs use
Poppler 26.07.0; an additional ACL page uses MuPDF 1.27.0; six JMLR pages
use Poppler 26.07.0. Extraction identity and page ranges are pinned separately.
No corruption was injected or source text repaired. Run:

```sh
cargo run --locked -p turbo-normalize --example reproduce_extracted_pdf -- target/pdf-reproduction
cargo test --locked -p turbo-normalize --test extracted_pdf -- --nocapture
```

Exact kind and both byte boundaries must match. TP/FP/FN counts are per
span, precision is TP/(TP+FP), and recall is TP/(TP+FN). N/A means a zero
denominator. The default and optional all-kinds configurations differ:

| Kind | Default TP/FP/FN | Default precision/recall | All-kinds TP/FP/FN | All-kinds precision/recall |
| --- | --- | --- | --- | --- |
| ReplacementChar | 1/0/0 | 1.0000/1.0000 | 1/0/0 | 1.0000/1.0000 |
| Control | 26/0/0 | 1.0000/1.0000 | 26/0/0 | 1.0000/1.0000 |
| Noncharacter | 0/0/0 | N/A/N/A | 0/0/0 | N/A/N/A |
| UnpairedSurrogate | 0/0/0 | N/A/N/A | 0/0/0 | N/A/N/A |
| PrivateUse | 4/0/0 | 1.0000/1.0000 | 4/0/0 | 1.0000/1.0000 |
| BidiControl | 0/0/0 | N/A/N/A | 0/0/0 | N/A/N/A |
| ZeroWidth | 0/0/0 | N/A/N/A | 0/0/0 | N/A/N/A |
| UnicodeTag | 0/0/0 | N/A/N/A | 0/0/0 | N/A/N/A |
| Mojibake | 0/0/0 | N/A/N/A | 0/0/0 | N/A/N/A |
| MojibakeSequence | 0/0/0 | N/A/N/A | 0/0/0 | N/A/N/A |
| RepeatedLine | 0/0/6 | N/A/0.0000 | 6/3/0 | 0.6667/1.0000 |

The six repeated-line gold spans are alternating running headers in JMLR
pages 2-7. The optional rule also reports three legitimate occurrences of
`u∼Υ z∼Qθ` inside mathematical expressions, at bytes 17650..17664,
22423..22437 and 23004..23018 of `jmlr-25-0035`. Those positions remain
negative labels; the three predictions are recorded as false positives.
The default disables this optional rule, so all six headers are false
negatives in its result. The replacement-character positive is an actual
MuPDF extraction of a summation sign on ACL printed page 174. This
measurement does not imply that Poppler produces the same damage.

The default NoiseScorer evaluation remains negative-only on the two
original ACL samples. It reports the legitimate acronym `FFNNs` at
5548..5553 in P19-1423, severity Damaged and score 0.5: precision 0/1 = 0,
recall N/A. The second sample has no noise prediction. The new pages
were annotated for artifacts, not for lexical noise, and do not expand
that auxiliary noise evaluation.

No defaults were tuned. **Mojibake and ZeroWidth remain unsupported by
positive corpus examples**, so issue #915's coverage work is incomplete.
Other absent kinds are also unmeasured; a Rust str cannot contain an
unpaired surrogate. Four scientific papers labelled in one pass cannot
establish general PDF or OCR quality. Column order, hyphenation and
flattened equations are outside this taxonomy. Further licensed damaged
extracts and independent annotation review are still required. The test
pins observed errors as well as successes; it does not certify unsupported
kinds by treating N/A as perfect precision or recall.
