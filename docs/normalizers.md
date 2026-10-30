# Text normalizers (`turbo-normalize`, `turbo-normalize-layers`, `turbo-unicode`, `turbo-encoding`)

`turbo-normalize` (`analysis/normalize/`) transforms a source `View` into a
derived `View` while recording every character edit in a `ViewBuilder`
(`turbo_analysis::ViewBuilder`), so spans produced by downstream annotators over
the normalized view project back to exact UTF-8 byte offsets in the original
document text ([analysis.md](analysis.md), "Document, views, and alignments").
It also provides non-rewriting annotators and scanners that emit diagnostic and
extraction layers in `turbo-normalize-layers` (`analysis/normalize-layers/`):
`Placeholders` (`turbo:placeholders`), `Emoji` (`turbo:emoji`), `Artifacts`
(`turbo:artifacts`), `Noise` (`turbo:noise`), `Corrections`
(`turbo:corrections`), `Assets` (`turbo:assets`), and `ProtectedSpans`
(`turbo:protected_spans`), alongside `TextQuality` (`turbo:text_quality`) and
`SpanQuality` (`turbo:span_quality`) in `turbo-analysis-layers`.

## Crates

| Crate | Path | Contents |
|---|---|---|
| `turbo-unicode` | `analysis/unicode/` | Unicode 18.0 character data: `GeneralCategory`, properties, `Script`, case mapping and folding, `Form` (`Nfc`, `Nfd`, `Nfkc`, `Nfkd`), confusables, emoji sequences, and UAX #29 `WordBreak`. Tables generated from the UCD files in `analysis/unicode/ucd/` by `turbo-unicode-gen`. |
| `turbo-unicode-gen` | `analysis/unicode-gen/` | The deterministic Unicode Character Database table generator. |
| `turbo-normalize` | `analysis/normalize/` | Rewriting normalizers, the `Normalizer` trait, the `NormalizeStep<N>` pipeline step adapter, `normalize`, `apply`, and the composed `Chain`. |
| `turbo-encoding` | `analysis/encoding/` | Character-set decoders, encoding detection, and `Mojibake` repair writing `SourceEncoding` (`turbo:encoding` in `turbo-encoding-layers`). |
| `turbo-normalize-layers` | `analysis/normalize-layers/` | Layers written by normalizers and scanners: `Placeholders` (`turbo:placeholders`), `Emoji` (`turbo:emoji`), `Artifacts` (`turbo:artifacts`), `Noise` (`turbo:noise`), `Corrections` (`turbo:corrections`), `Assets` (`turbo:assets`), `ProtectedSpans` (`turbo:protected_spans`), and `TextQualityAnnotator` (`QualityScratch`). |

## View-rewriting normalizers (`Normalizer`, `NormalizeStep`, and `Chain`)

Every rewriting normalizer implements `Normalizer`
(`analysis/normalize/src/lib.rs`) and runs standalone via `normalize` / `apply`,
as a pipeline step via `NormalizeStep<N>`, or inside a composed `Chain`
(`analysis/normalize/src/chain.rs`):

| Normalizer | Module | Transformation |
|---|---|---|
| `UnicodeForm` | `analysis/normalize/src/unicode_form.rs` | Canonical and compatibility Unicode normalization (`Form::Nfc`, `Form::Nfd`, `Form::Nfkc`, `Form::Nfkd`) with per-character alignment |
| `CaseFold` (`CaseFoldMode`) | `analysis/normalize/src/case_fold.rs` | Full Unicode case folding or locale-aware lowercasing (`tr`, `az`, `lt`, `el`, `nl`) |
| `AccentFold`, `StrokeLetters` (`FoldScratch`) | `analysis/normalize/src/accent_fold/mod.rs` | Strips combining diacritics and folds Latin typographic ligatures and stroked letters (`ß`, `æ`, `œ`, `ø`, `ł`, `đ`) to ASCII |
| `GermanUmlauts` | `analysis/normalize/src/german.rs` | Folds German umlauts and Eszett (`ä` -> `ae`, `ö` -> `oe`, `ü` -> `ue`, `ß` -> `ss`) |
| `Quotes`, `Dashes`, `Ellipsis`, `Bullets` | `analysis/normalize/src/quotes.rs`, `analysis/normalize/src/dashes.rs`, `analysis/normalize/src/ellipsis.rs`, `analysis/normalize/src/bullets.rs` | Folds curly quotes, dash variants, horizontal/vertical ellipses, and typographic bullets to canonical ASCII forms |
| `Digits` | `analysis/normalize/src/digits.rs` | Folds non-ASCII Unicode decimal digit scripts (`Nd`) and fullwidth digits to ASCII `0..=9` |
| `ConfusableSkeleton` | `analysis/normalize/src/confusables.rs` | Maps UTS #39 mixed-script homoglyphs to canonical script skeletons |
| `Invisible` | `analysis/normalize/src/invisible.rs` | Strips zero-width characters, bidirectional controls, soft hyphens, and variation selectors |
| `Whitespace` (`WhitespaceMode`), `Shrink` | `analysis/normalize/src/whitespace.rs`, `analysis/normalize/src/shrink.rs` | Collapses Unicode whitespace runs and repeated character runs (`soooo` -> `soo`) |
| `Urls`, `BoundedUrls`, `SocialMedia`, `BoundedSocialMedia`, `Numbers` | `analysis/normalize/src/url/mod.rs`, `analysis/normalize/src/social/mod.rs`, `analysis/normalize/src/number.rs` | Strips or replaces URLs, e-mail addresses, `@handles`, `#hashtags`, and digit runs while recording `Placeholders` (`turbo:placeholders`, `PlaceholderKind`) |
| `EmojiRemove` (`EmojiRemoveMode`), `EmojiToEmoticon`, `EmoticonToEmoji` | `analysis/normalize/src/emoji/mod.rs` | Strips or converts emoji presentation sequences, ZWJ sequences, flags, keycaps, and ASCII emoticons |
| `HtmlEntityDecodeNormalizer`, `HtmlTagStripNormalizer` | `analysis/normalize/src/entities.rs`, `analysis/normalize/src/markup.rs` | Decodes named/numeric HTML5 character references and strips HTML/XML/Markdown formatting tags |
| `Dehyphenate` (`DehyphenateMode`) | `analysis/normalize/src/dehyphenate.rs` | Rejoins line-broken hyphenated words from PDF and OCR extraction while preserving lexical compound hyphens |
| `Mojibake` | `analysis/normalize/src/mojibake.rs` | Repairs UTF-8 text mis-decoded as Windows-1252, ISO-8859-1, or CP437 |
| `EnglishContractions`, `EnglishContractionsWithPos` | `analysis/normalize/src/contractions/mod.rs` | Expands English clitic contractions (`don't` -> `do not`, context-sensitive `'s` and `'d` using `PosTags` when available) |
| `SymbolJoiner`, `SymbolWords` (`DEFAULT_SYMBOL_SEQUENCES`) | `analysis/normalize/src/symbols/mod.rs` | Joins or expands multi-character symbol sequences (`&`, `%`, `+`, `@`) |
| `SpellCorrect` (`SpellCorrectMode`), `ContextSpellCorrector` (`SymSpell`, `ConfusableSet`) | `analysis/normalize/src/spellcheck/mod.rs` | Rewrites high-confidence spelling errors into a derived `View` and records `Corrections` (`turbo:corrections`) |
| `AssetFold` | `analysis/normalize/src/assets/mod.rs` | Folds inline Base64 data URIs, PEM blocks, hex dumps, inline SVGs, and JSON blobs out of derived views while recording `Assets` (`turbo:assets`) and `Placeholders` (`turbo:placeholders`) |

### Quotes

`Quotes` (`turbo:normalize:quotes`) replaces the single quotation marks and
apostrophes of `SINGLE_QUOTES` with `U+0027` (`'`) and the double quotation
marks of `DOUBLE_QUOTES` with `U+0022` (`"`).

Behavior notes: `U+2E42` (`DOUBLE LOW-REVERSED-9 QUOTATION MARK`) carries the
Unicode `Quotation_Mark` property and folds to `U+0022` (`"`). CJK corner
brackets (`U+300C..=U+300F`, `U+FE41..=U+FE44`, `U+FF62`, `U+FF63`) are
preserved so opening and closing title/emphasis bracket directionality is not
lost.

### Bullets

`Bullets` (`turbo:normalize:bullets`) replaces `U+2022` (`BULLET`), `U+2023`
(`TRIANGULAR BULLET`), `U+2043` (`HYPHEN BULLET`), `U+204C` (`BLACK LEFTWARDS
BULLET`), `U+204D` (`BLACK RIGHTWARDS BULLET`), `U+2219` (`BULLET OPERATOR`),
`U+25D8` (`INVERSE BULLET`), `U+25E6` (`WHITE BULLET`), `U+29BE` (`CIRCLED
WHITE BULLET`), and `U+29BF` (`CIRCLED BULLET`) with a single ASCII space.

Behavior notes: All ten Unicode list-marker bullet characters fold to `" "`,
while `U+00B7` (`MIDDLE DOT`, used inside Catalan words), ornamental floral
bullets (`U+2619`, `U+2765`, `U+2767`), and `U+1F7DB` remain untouched.

### Ellipsis

`Ellipsis` (`turbo:normalize:ellipsis`) replaces `U+2026` (`HORIZONTAL
ELLIPSIS`) and `U+FE19` (`PRESENTATION FORM FOR VERTICAL HORIZONTAL ELLIPSIS`)
with `"..."`, `U+2025` (`TWO DOT LEADER`) and `U+FE30` (`PRESENTATION FORM FOR
VERTICAL TWO DOT LEADER`) with `".."`, and `U+2024` (`ONE DOT LEADER`) with
`"."`.

Behavior notes: Each leader and vertical presentation ellipsis expands to its
NFKC compatibility dot sequence with per-character source alignment, while
`U+22EF` (`MIDLINE HORIZONTAL ELLIPSIS`) is preserved as a mathematical symbol.

### Invisible characters

`Invisible` (`turbo:normalize:invisible`) deletes the 26 format and ignorable
code points in `INVISIBLE`: `U+00AD` (`SOFT HYPHEN`), `U+061C` (`ARABIC LETTER
MARK`), `U+200B` (`ZERO WIDTH SPACE`), `U+200E..=U+200F`, `U+202A..=U+202E`,
`U+2060..=U+2064`, `U+2066..=U+206F` (isolates and deprecated format controls),
and `U+FEFF` (`BYTE ORDER MARK`).

Behavior notes: `U+206A..=U+206F` (deprecated Unicode 3.0 `Cf` format controls)
are stripped alongside zero-width spaces and directional controls, whereas
`U+200C` (`ZWNJ`), `U+200D` (`ZWJ`), `U+180E` (`MONGOLIAN VOWEL SEPARATOR`),
and variation selectors are preserved so shaping and emoji ZWJ sequences remain
intact.

### Dehyphenation

`Dehyphenate` (`turbo:normalize:dehyphenate`) rejoins words split across a
single line break after a hyphen (`U+002D`, `U+00AD`, or `U+2010`) and optional
next-line indentation when both sides are letters.

Behavior notes: The letter preceding the line-break hyphen may be followed by
combining marks (`General_Category = M`), so decomposed (`NFD`) Latin words
(`"cafe\u{0301}-\nteria"`) and Indic words ending in combining vowel signs
rejoin cleanly across line breaks.

### URLs and e-mail addresses (`url/`)

`Urls` and `BoundedUrls` (`analysis/normalize/src/url/mod.rs`) strip or replace
HTTP(S) URLs and e-mail addresses while recording `Placeholders`
(`turbo:placeholders`).

Behavior notes: `BoundedUrls` recognizes internationalized local parts and
domains (including non-ASCII letters and combining marks in e-mail addresses)
when delimited by word/whitespace boundaries.

### English contractions (`contractions/`)

`EnglishContractions` and `EnglishContractionsWithPos`
(`analysis/normalize/src/contractions/mod.rs`) expand English clitic
contractions (`can't` -> `can not`, `won't` -> `will not`, `shan't` -> `shall
not`, `<aux>n't` -> `<aux> not`, `let's` -> `let us`, `y'all` -> `you all`, and
unambiguous `'re`, `'ve`, `'ll`, `'m`) while preserving case.

Behavior notes: `y'all` expands to `you all` (matching stem capitalization) and
demonstrative/interrogative subjects (`here`, `there`, `these`, `those`,
`this`, `what`, `where`, `which`) expand with `'re`, `'ve`, and `'ll`.

### Symbols (`symbols/`)

`SymbolJoiner` and `SymbolWords` (`analysis/normalize/src/symbols/mod.rs`) join
multi-character technical symbol tokens (`DEFAULT_SYMBOL_SEQUENCES`) and expand
isolated symbols (`&`, `+`, `@`, `%`, `§`, `¶`, `°`, `©`, `®`, `™`) to English
words.

Behavior notes: `SymbolWords` and `SymbolJoiner` expand standalone mapped
symbols delimited by whitespace or text boundaries in running text while leaving
variation-selector emoji sequences (`©\u{FE0F}`, `™\u{FE0F}`) and compound
tokens (`AT&T`, `R&D`) intact.

### Confusable skeleton (`analysis/normalize/src/confusables.rs`)

`ConfusableSkeleton` (`turbo:normalize:confusable-skeleton`) computes the
UTS #39 section 4 skeleton (`NFD` -> confusable prototype mapping -> `NFD`)
with per-character alignment.

Behavior notes: Prototype mappings are generated from Unicode 18.0.0
`confusables.txt` via `turbo_unicode`.

### Case folding (`case_fold.rs`)

`CaseFold` (`turbo:normalize:case-fold`) performs root lowercasing
(`CaseFold::lowercase()`), Turkic locale lowercasing
(`CaseFold::turkish_lowercase()`, `CaseFold::azerbaijani_lowercase()`), or
Unicode simple/full case folding (`CaseFold::simple()`, `CaseFold::full()`,
`CaseFold::turkic_simple()`, `CaseFold::turkic_full()`).

Behavior notes: `CaseFold::turkish_lowercase()` and
`CaseFold::azerbaijani_lowercase()` evaluate the Unicode `Before_Dot`,
`After_I`, and `Final_Sigma` context conditions on the original source text so
`I + U+0307` folds to `i`, `I` without a dot above folds to `U+0131` (`ı`), and
Greek final sigma (`U+03C2`) is selected at word ends.

## Diagnostic and extraction layers

- **Spelling (`SymSpell`, `ConfusableSet`, `ContextSpellAnnotator`,
  `ContextSpellCorrector`, `SpellCorrect`,
  `analysis/normalize/src/spellcheck/mod.rs`)**: SymSpell delete-index candidate
  generation with weighted Damerau-Levenshtein keyboard distance, UTS #39
  confusable folding, run-on word splitting, and 5-gram language-model beam
  decoding into `Corrections` (`turbo:corrections`).
- **Emoji and emoticons (`EmojiAnnotate`, `EmojiRemove`, `EmojiToEmoticon`,
  `EmoticonToEmoji`, `analysis/normalize/src/emoji/mod.rs`)**: CLDR short-name,
  group, and sentiment polarity lookup for Unicode emoji and ASCII emoticons
  into `Emoji` (`turbo:emoji`, `EmojiKind`).
- **URLs and social media placeholders (`Urls`, `BoundedUrls`, `SocialMedia`,
  `BoundedSocialMedia`, `analysis/normalize/src/url/mod.rs`,
  `analysis/normalize/src/social/mod.rs`)**: records replaced `@mentions`,
  `#hashtags`, URLs, and e-mail spans into `Placeholders`
  (`turbo:placeholders`, `PlaceholderKind`).
- **Embedded assets (`AssetDetector`, `AssetFold`, `Asset`,
  `analysis/normalize/src/assets/mod.rs`)**: detects inline Base64 data URIs,
  PEM blocks, hex dumps, inline SVGs, and JSON blobs into `Assets`
  (`turbo:assets`, `AssetFormat`, `AssetEncoding`).
- **Encoding artifacts and structural noise (`ArtifactDetector`,
  `ArtifactDetectorBuilder`, `KindSet`, `RepeatedLines`, `NoiseScorer`,
  `Dictionary`, `NoiseFinding`, `NoiseStats`,
  `analysis/normalize/src/artifacts/mod.rs`,
  `analysis/normalize/src/noise/mod.rs`)**: detects replacement characters,
  control sequences, mojibake, OCR garbage, and low-alpha noise regions into
  `Artifacts` (`turbo:artifacts`, `ArtifactKind`) and `Noise` (`turbo:noise`,
  `NoiseSeverity`).
- **Text quality (`TextQualityAnnotator`, `QualityScratch`,
  `analysis/normalize-layers/src/quality.rs`)**: computes per-view and per-span
  readability, lexical diversity, and quality metrics into `TextQuality`
  (`turbo:text_quality`) and `SpanQuality` (`turbo:span_quality`,
  [text-quality.md](text-quality.md)).

## Testing and speed

- **Alignment and property tests**:
  - `analysis/normalize/tests/core.rs`,
    `analysis/normalize/tests/folds_properties.rs`,
    `analysis/normalize/tests/span_validation.rs`, and
    `analysis/normalize/tests/xray_chain.rs` verify that single (`NormalizeStep`)
    and chained (`Chain`) normalizers preserve UTF-8 character boundaries and
    round-trip span projections between derived views and the source document.
- **Component tests**:
  - `analysis/normalize/tests/unicode_form.rs`,
    `analysis/normalize/tests/case_fold.rs`,
    `analysis/normalize/tests/locale_lowercase.rs`,
    `analysis/normalize/tests/accent_folds.rs`,
    `analysis/normalize/tests/punctuation_folds.rs`,
    `analysis/normalize/tests/digits.rs`,
    `analysis/normalize/tests/confusables.rs`,
    `analysis/normalize/tests/markup.rs`,
    `analysis/normalize/tests/dehyphenate.rs`,
    `analysis/normalize/tests/contractions.rs`,
    `analysis/normalize/tests/spellcheck.rs`,
    `analysis/normalize/tests/spell_correct.rs`,
    `analysis/normalize/tests/context_spellcheck.rs`,
    `analysis/normalize/tests/emoji.rs`,
    `analysis/normalize/tests/social_media.rs`,
    `analysis/normalize/tests/urls.rs`,
    `analysis/normalize/tests/assets.rs`,
    `analysis/normalize/tests/artifacts.rs`, and
    `analysis/normalize/tests/noise.rs` test each normalizer and annotator.
- **Throughput**:
  - `analysis/normalize/examples/normalize_whitespace_timing.rs` and
    `analysis/normalize/examples/reproduce_extracted_pdf.rs` benchmark
    `Whitespace` and PDF cleanup pipelines in MiB/s.
