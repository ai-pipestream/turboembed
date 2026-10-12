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
- `SocialMediaCharSequenceNormalizer` replaces emoticons inside words
  (`expo` becomes `"e o"`, `xoxo` two spaces, `:people` `" eople"`), and
  its hashtags swallow the punctuation after them (`#win!`). Its retweet
  word boundary depends on how the JDK regular expression engine walks
  back over UTF-16 units, so a supplementary-plane combining mark never
  blocks it.

## Tests

`cargo test -p turbo-unicode -p turbo-normalize -p turbo-normalize-layers`.
