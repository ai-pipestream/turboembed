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
`emoji.rs` and `mod.rs`. Each file starts with a notice naming its input
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

Tests: every expectation of the invisible character tests in
`SetBasedNormalizerTest` and `AlignedNormalizerPipelineTest`, every table
entry, the additions, U+180E kept, a check that every entry is a
`Default_Ignorable_Code_Point` format character, the chain of invisible
characters, whitespace and dashes from
`pipelineComposesStripInvisibleWhitespaceAndDashesBackToOriginal` and its
line-break and paragraph variants, and generated texts against a
reference filter.

### Accent folding (`analysis/normalize/src/accent_fold.rs`) and the run pass (`analysis/normalize/src/fold_runs.rs`)

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

## Tests

`cargo test -p turbo-unicode -p turbo-normalize -p turbo-normalize-layers`.
