# Text normalizers (`turbo-normalize`)

`turbo-normalize` (`analysis/normalize/`) transforms a source `View` into a
derived `View` while recording every character edit in a `ViewBuilder`
(`turbo_analysis::doc::ViewBuilder`), so spans produced by downstream
annotators over the normalized view project back to exact UTF-8 byte offsets in
the original document text. It also provides non-rewriting scanners that emit
diagnostic layers (`SpellingSuggestions`, `EmojiMentions`, `SocialEntities`,
`EmbeddedAssets`, `EncodingArtifacts`, `BoilerplateRegions`, `NoiseRegions`).

## View-rewriting normalizers (`Normalizer` and `NormalizeChain`)

Every rewriting normalizer implements `Normalizer`
(`analysis/normalize/src/lib.rs`) and can run standalone or inside a
single-pass composed `NormalizeChain` (`analysis/normalize/src/chain.rs`):

| Normalizer | Module | Transformation |
|---|---|---|
| `UnicodeFormNormalizer` | `analysis/normalize/src/unicode_form.rs` | Canonical and compatibility Unicode normalization (`NFC`, `NFD`, `NFKC`, `NFKD`) with per-character alignment |
| `CaseFoldNormalizer` | `analysis/normalize/src/case_fold.rs` | Full Unicode case folding or locale-aware lowercasing (`tr`, `az`, `lt`, `el`, `nl`) |
| `AccentFoldNormalizer` | `analysis/normalize/src/accent_fold/mod.rs` | Strips combining diacritics and folds Latin typographic ligatures (`ß`, `æ`, `œ`, `ø`, `ł`, `đ`) to ASCII |
| `GermanOrthographyNormalizer` | `analysis/normalize/src/german.rs` | Folds German umlauts and Eszett (`ä` -> `ae`, `ö` -> `oe`, `ü` -> `ue`, `ß` -> `ss`) |
| `QuotesNormalizer`, `DashesNormalizer`, `EllipsisNormalizer`, `BulletsNormalizer` | `analysis/normalize/src/quotes.rs`, `analysis/normalize/src/dashes.rs`, `analysis/normalize/src/ellipsis.rs`, `analysis/normalize/src/bullets.rs` | Folds curly quotes, dash variants, horizontal/vertical ellipses, and typographic bullets to canonical ASCII forms |
| `DigitsNormalizer` | `analysis/normalize/src/digits.rs` | Folds non-ASCII Unicode decimal digit scripts (`Nd`) and fullwidth digits to ASCII `0..=9` |
| `ConfusablesNormalizer` | `analysis/normalize/src/confusables.rs` | Maps UTS #39 mixed-script homoglyphs to canonical script skeletons |
| `InvisibleNormalizer` | `analysis/normalize/src/invisible.rs` | Strips zero-width characters, bidirectional controls, soft hyphens, and variation selectors |
| `WhitespaceNormalizer`, `ShrinkNormalizer`, `FoldRunsNormalizer` | `analysis/normalize/src/whitespace.rs`, `analysis/normalize/src/shrink.rs`, `analysis/normalize/src/fold_runs.rs` | Collapses Unicode whitespace runs and repeated character runs (`soooo` -> `soo`) |
| `UrlNormalizer`, `TwitterNormalizer`, `NumberNormalizer` | `analysis/normalize/src/url/mod.rs`, `analysis/normalize/src/social/mod.rs`, `analysis/normalize/src/number.rs` | Strips or replaces URLs, e-mail addresses, `@handles`, `#hashtags`, and digit runs |
| `EmojiNormalizer` | `analysis/normalize/src/emoji/remove.rs` | Strips emoji presentation sequences, ZWJ sequences, flags, and keycaps |
| `HtmlEntitiesNormalizer`, `MarkupStripper` | `analysis/normalize/src/entities.rs`, `analysis/normalize/src/markup.rs` | Decodes named/numeric HTML5 character references and strips HTML/XML/Markdown formatting tags |
| `DehyphenateNormalizer` | `analysis/normalize/src/dehyphenate.rs` | Rejoins line-broken hyphenated words from PDF and OCR extraction while preserving lexical compound hyphens |
| `MojibakeNormalizer` | `analysis/normalize/src/mojibake.rs` | Repairs UTF-8 text mis-decoded as Windows-1252, ISO-8859-1, or CP437 |
| `ContractionNormalizer` | `analysis/normalize/src/contractions/mod.rs` | Expands English clitic contractions (`don't` -> `do not`, context-sensitive `'s` and `'d` using POS tags when available) |
| `SpellCorrector` | `analysis/normalize/src/spellcheck/correct.rs` | Rewrites high-confidence spelling errors into a derived `View` |

## Diagnostic and extraction layers

- **Spelling (`SpellChecker`, `ContextSpellChecker`,
  `analysis/normalize/src/spellcheck/mod.rs`)**: SymSpell delete-index candidate
  generation with weighted Damerau-Levenshtein keyboard distance, UTS #39
  confusable folding, run-on word splitting, and 5-gram language-model beam
  decoding into `SpellingSuggestions` (`turbo:spelling`).
- **Emoji and emoticons (`EmojiAnnotator`,
  `analysis/normalize/src/emoji/annotate.rs`)**: CLDR short-name, group, and
  sentiment polarity lookup for Unicode emoji and ASCII emoticons into
  `EmojiMentions` (`turbo:emoji`).
- **Social entities (`SocialScanner`, `analysis/normalize/src/social/mod.rs`)**:
  extracts `@mentions`, `#hashtags`, `$cashtags`, URLs, and e-mail spans into
  `SocialEntities` (`turbo:social`).
- **Embedded assets (`EmbeddedAssetScanner`,
  `analysis/normalize/src/assets/mod.rs`)**: detects inline Base64 data URIs,
  PEM blocks, hex dumps, inline SVGs, and JSON blobs into `EmbeddedAssets`
  (`turbo:embedded-assets`) and folds them out of derived views.
- **Encoding artifacts and noise (`ArtifactScanner`, `NoiseScanner`,
  `analysis/normalize/src/artifacts/mod.rs`,
  `analysis/normalize/src/noise/mod.rs`)**: detects replacement characters,
  control sequences, mojibake, OCR garbage, and low-alpha noise regions into
  `EncodingArtifacts` (`turbo:encoding-artifacts`) and `NoiseRegions`
  (`turbo:noise-regions`).

## Testing and speed

- **Alignment and property tests**:
  - `analysis/normalize/tests/core.rs`,
    `analysis/normalize/tests/folds_properties.rs`,
    `analysis/normalize/tests/span_validation.rs`, and
    `analysis/normalize/tests/xray_chain.rs` verify that single and chained
    normalizers preserve UTF-8 character boundaries and round-trip span
    projections between derived views and the source document.
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
    `analysis/normalize/tests/context_spellcheck.rs`,
    `analysis/normalize/tests/emoji.rs`,
    `analysis/normalize/tests/social_media.rs`,
    `analysis/normalize/tests/assets.rs`,
    `analysis/normalize/tests/artifacts.rs`, and
    `analysis/normalize/tests/noise.rs` test each normalizer and scanner.
- **Throughput**:
  - `analysis/normalize/examples/normalize_whitespace_timing.rs` and
    `analysis/normalize/examples/reproduce_extracted_pdf.rs` benchmark
    `WhitespaceNormalizer` and PDF cleanup pipelines in MiB/s.
