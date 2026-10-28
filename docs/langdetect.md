# Language detection (`turbo-langdetect` and `turbo-langspan`)

`turbo-langdetect` (`analysis/langdetect/`) detects the primary language of a
document view and writes the ranked ballot as the `turbo:language` (`Language`)
layer. `turbo-langspan` (`analysis/langspan/`) segments mixed-language views
into contiguous language-labelled spans and writes them as the
`turbo:language-spans` (`LanguageSpans`) layer alongside the document-level
`Language` summary.

## Document-level language detection (`turbo-langdetect`)

`LanguageDetector` (`analysis/langdetect/src/lib.rs`) scores a `View` over
character $n$-grams extracted from a normalized view and produces a ranked
list of `(lang, confidence)` predictions.

### Normalization and context extraction

Before extracting character $n$-grams, `LanguageDetector` normalizes the view
through a `NormalizeChain` combining:

- Twitter/X and social token stripping (`@handle`, `#hashtag`, URLs, and
  e-mail addresses),
- Unicode character substitution and script-aware filtering,
- Number digit stripping, and
- Whitespace run collapsing (`ShrinkNormalizer`).

`DefaultLanguageDetectorContextGenerator` (`analysis/langdetect/src/context.rs`)
extracts character $n$-grams within the model's configured `(min_length,
max_length)` window from the normalized text.

### Adaptive probing (`ProbingLanguageDetector`)

On long documents, scoring every character $n$-gram across the entire text is
unnecessary once the leading language candidate has converged.
`ProbingLanguageDetector` (`analysis/langdetect/src/probing.rs`) samples fixed
windows of text under `ProbingConfig` (`chunk_chars`, `max_windows`,
`convergence_streak`, `min_margin`), accumulating $n$-gram log-probabilities
until the top language candidate leads the runner-up by at least `min_margin`
for `convergence_streak` consecutive windows.

### The `Language` layer

`Language`, in `turbo-analysis-layers` (`analysis/layers/src/lib.rs`), is a
document-anchored layer holding the ranked language predictions (`code`,
`confidence`), with `Language::top()` returning the highest-scoring language
code (`eng`, `deu`, `fra`, `spa`, `cmn`, `jpn`, `ara`, `rus`, etc.).

## Mixed-language span detection (`turbo-langspan`)

When a document mixes multiple languages or scripts (for example, quoted
passages, code-switched dialogue, or bilingual headings), a single document
tag obscures the interior switches. `LanguageSpanDetector`
(`analysis/langspan/src/lib.rs`) segments a `View` into contiguous
UTF-8 byte spans, each tagged with its language code and confidence, and
writes:

- `LanguageSpans` (`turbo:language-spans`): ordered, non-overlapping spans
  covering the text with per-span language code and confidence, and
- `Language` (`turbo:language`): the byte-length-weighted language distribution
  aggregated across the detected spans.

### Script priors and sequence decoding

`LanguageSpanDetector` combines character $n$-gram emission scores with
Unicode script priors (for example, Han, Hiragana/Katakana, Hangul, Arabic,
Cyrillic, Devanagari, Greek, Hebrew, Thai) and decodes the sequence of windows
with a switch penalty so isolated foreign proper nouns do not fragment a
monolingual sentence while genuine multi-word switches form clean span
boundaries snapped to token/whitespace boundaries.

For short inputs and single-sentence queries, Platt-calibrated emission models
and script-prior boosting sharpen top-1 language identification (`analysis/langspan/examples/short_text.rs`).

## Testing

- `analysis/langdetect/tests/detector.rs` and
  `analysis/langdetect/tests/context.rs` test $n$-gram context extraction,
  full-text and adaptive probing detection, convergence thresholds, and
  pipeline integration into `Language`.
- `analysis/langspan/examples/short_text.rs` exercises short-text and
  mixed-language span segmentation across multilingual passages.
