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
which joins edits that touch. Two neighbouring replaced characters
(two Arabic-Indic digits, a closing quote followed by an ellipsis in a
chain) therefore form one edit, and a span inside it maps back to both
source characters. OpenNLP's `Alignment` records a source range per
output character, so there the second digit maps back to the second
source digit alone. Output text is the same; the source spans of
characters inside such a run are wider here.

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
(deprecated format characters) and U+FEFF (byte order mark). Each deletion is an edit to the empty string, and neighbouring
deletions are one edit. The zero width joiner and non-joiner and the
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
"Müller-Lüdenscheidt". This is a heuristic. It does not help
with lowercase compounds ("well-\nknown" is still joined), it keeps a
hyphen that the typesetter added before a capital inside a word
("Mc-\nDonald" becomes "Mc-Donald"), and it never keeps a soft hyphen or
U+2010, which mark typesetter breaks. Deciding from a lexicon whether the
joined form is a word is not implemented.

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
- `DehyphenationCharSequenceNormalizer` checks only the code unit before
  the hyphen with `Character.isLetter`, so a word that ends in a
  combining mark (NFD text, Indic vowel signs) is never joined. Skipping
  back over general category M before the letter test fixes this.

## Tests

`cargo test -p turbo-unicode -p turbo-normalize -p turbo-normalize-layers`.
