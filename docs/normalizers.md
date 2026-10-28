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
