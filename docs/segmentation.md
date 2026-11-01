# Sentence and token segmentation (`turbo-sentdetect`, `turbo-sat`, `turbo-tokenize`)

Sentence segmentation and tokenization turn a `View` into the `Sentences`
(`turbo:sentences`), `Structure` (`turbo:structure`), `Tokens`
(`turbo:tokens`), `WordTypes` (`turbo:word-types`), and `Readings`
(`turbo:readings`) layers in `turbo-analysis-layers` and
`turbo-tokenize-layers`. All offsets are UTF-8 byte offsets on character
boundaries into the view they annotate.

## Paragraph and sentence segmentation (`turbo-sentdetect`)

`turbo-sentdetect` (`analysis/sentdetect/`) provides paragraph detection and
three sentence segmenters plus classified abbreviation inventories:

| Detector | Module | Description |
|---|---|---|
| `ParagraphDetector` (`paragraph_spans`) | `analysis/sentdetect/src/paragraph.rs` | Splits a view on blank lines into paragraph `Block` items in `Structure` (`turbo:structure`) |
| `RuleSentenceDetector` (`RuleSplitter`, `UnpunctuatedSplitter`) | `analysis/sentdetect/src/rules/mod.rs` | UAX #29 sentence boundaries refined with per-language `Rules`, `AbbreviationInventory` (`AbbreviationPreset`), bracket/quote balancing, list-marker suppression, unit/initial guards, and unpunctuated script fallback (`UNPUNCTUATED_SCRIPTS`) |
| `NewlineDetector` (`newline_spans`) | `analysis/sentdetect/src/newline.rs` | Line-oriented sentence splitter writing one sentence per non-blank line into `Sentences` (`turbo:sentences`) |
| `SentenceDetector` (`SentenceModel`) | `analysis/sentdetect/src/me/mod.rs` | Maximum-entropy end-of-sentence classifier over candidate punctuation positions (`EndOfSentence`) and abbreviation dictionaries, writing `Sentences` (`turbo:sentences`) and `Confidence<Sentences>` |

### Rule and abbreviation-inventory sentence detection

`RuleSentenceDetector` and `RuleSplitter` start from Unicode Standard Annex #29
(`UAX #29`) sentence breaks (`analysis/sentdetect/src/rules/uax29.rs`) and
refine them using:

- Per-language abbreviation and title tables (`Rules`,
  `analysis/sentdetect/src/rules/tables.rs` and
  `analysis/sentdetect/src/rules/languages.rs`),
- Classified abbreviation inventories (`AbbreviationInventory`,
  `AbbreviationInventories`, `AbbreviationClass`, `AbbreviationPreset`,
  `ContextRule`, `Verdict`, `analysis/sentdetect/src/inventory/mod.rs`),
- Balanced parentheses, brackets, and quotation marks
  (`analysis/sentdetect/src/rules/brackets.rs`),
- Numbered and bulleted list-item guards
  (`analysis/sentdetect/src/rules/lists.rs`), and
- Numeric unit and initialism suppression
  (`analysis/sentdetect/src/rules/units.rs`).

### Neural Segment Any Text (`turbo-sat`)

For unpunctuated, ASR, or noisy conversational text where punctuation rules
under-segment, `SatAnnotator` (`SatOptions`, `SatProvider`, `Scored`, `Head`,
`rule_detector_with_sat`, `rule_splitter_with_sat` in `analysis/sat/src/lib.rs`)
scores each token's sentence-boundary probability on the core's encoder over
wtpsplit's overlapping windows, moves each cut to a grapheme cluster boundary,
and writes `Sentences` (`turbo:sentences`) (with `Scored` also writing
`Confidence<Sentences>`).

## Tokenization (`turbo-tokenize` and `turbo-tokenize-layers`)

`turbo-tokenize` (`analysis/tokenize/`) segments a `View` (or each sentence in
`Sentences`) into `Tokens` (`turbo:tokens`), and optionally writes word-type
and morphological reading columns into `WordTypes` (`turbo:word-types`) and
`Readings` (`turbo:readings`):

| Tokenizer | Module | Description |
|---|---|---|
| `WhitespaceTokenizer` (`whitespace_spans`) | `analysis/tokenize/src/whitespace.rs` | Splits on Unicode `White_Space` runs |
| `SimpleTokenizer` (`simple_spans`) | `analysis/tokenize/src/simple.rs` | Splits on transitions between alphabetic, numeric, whitespace, and punctuation character classes |
| `WordTokenizer` / `UntypedWordTokenizer` (`word_tokens`, `word_boundaries`) | `analysis/tokenize/src/uax29/mod.rs` | Unicode Standard Annex #29 word-boundary segmenter writing `Tokens` and `WordTypes` (`turbo:word-types` with `WordType`) |
| `TokenizerMe` (`TokenizerModel`) | `analysis/tokenize/src/me/mod.rs` | Maximum-entropy boundary classifier inside whitespace-delimited units (`WhitespaceUnits`) with an `AlphaNumeric` skip optimization, writing `Tokens` and `Confidence<Tokens>` |
| `UnigramSegmenter` (`UnigramLexicon`, `UnigramMode`) | `analysis/tokenize/src/unigram/mod.rs` | Dictionary-plus-Viterbi unigram word segmenter with out-of-vocabulary character HMM for Chinese (`cmn`, [unigram-chinese.md](unigram-chinese.md)) |
| `LatticeTokenizer` / `LatticeAnalyzer` (`MecabDictionary`, `DictionaryFormat`) | `analysis/tokenize/src/lattice/mod.rs` | Double-array trie and connection-cost matrix Viterbi morphological segmenter for Japanese and Korean, writing `Tokens`, `PosTags`, `Lemmas`, and `Readings` (`turbo:readings`) |
| `ComplexSegmenter` / `DictionaryWordTokenizer` / `UntypedDictionaryWordTokenizer` (`ComplexScript`, `dictionary_word_tokens`) | `analysis/tokenize/src/complex/mod.rs` | UAX #29 word segmenter with ICU4X dictionary segmentation for Thai, Lao, Khmer, and Burmese runs |
| `Detokenizer` (`DetokenizerDictionary`, `DictionaryOperation`, `Operation`) | `analysis/tokenize/src/detok.rs` | Rebuilds untokenized text from `Tokens` using per-token merge operations (`MergeToRight`, `MergeToLeft`, `MergeBoth`, `NoOperation`, `RightLeftMatching`) |

### Script-specific tokenizers

- **Chinese (`UnigramSegmenter`, `analysis/tokenize/src/unigram/mod.rs`)**:
  builds a word lattice over `UnigramLexicon` and decodes the minimum-cost path
  with a 4-state (`B`, `M`, `E`, `S`) character HMM fallback for unknown words
  ([unigram-chinese.md](unigram-chinese.md)).
- **Japanese and Korean (`LatticeTokenizer`, `LatticeAnalyzer`,
  `analysis/tokenize/src/lattice/mod.rs`)**: builds a word lattice over a
  double-array trie, character-category grouping table (`chardef`), and
  token-to-token connection matrix (`matrix`), decodes the minimum-cost path
  with Viterbi, and writes surface tokens to `Tokens`, part-of-speech tags to
  `PosTags`, lemmas to `Lemmas`, and reading/pronunciation columns to
  `Readings` (`turbo:readings`). Zero-copy prebuilt images
  (`analysis/tokenize/src/lattice/prebuilt/mod.rs`) load directly from aligned
  byte slices without heap allocation during lookup.
- **Southeast Asian complex scripts (`ComplexSegmenter`,
  `DictionaryWordTokenizer`, `UntypedDictionaryWordTokenizer`,
  `analysis/tokenize/src/complex/mod.rs`)**: segments Thai, Lao, Khmer, and
  Burmese runs while preserving extended grapheme cluster boundaries
  (`analysis/tokenize/tests/complex_graphemes.rs`).

## Testing, accuracy, and speed

- **Sentence segmentation tests**:
  - `analysis/sentdetect/tests/conformance.rs`,
    `analysis/sentdetect/tests/abbreviations.rs`,
    `analysis/sentdetect/tests/rules_tables.rs`, and
    `analysis/sentdetect/tests/inventory_languages.rs` verify boundary
    precision across abbreviations, initials, ellipses, nested quotes, and 35+
    languages.
  - `analysis/sentdetect/examples/throughput.rs` and
    `analysis/sentdetect/examples/inventory_whitespace_timing.rs` measure
    sentence detection throughput in MiB/s.
  - `analysis/sat/tests/unpunctuated_eval.rs` and `analysis/sat/tests/speed.rs`
    verify `SatAnnotator` sentence boundary F1 on unpunctuated text and batch
    throughput.
- **Tokenization tests**:
  - `analysis/tokenize/tests/wordbreak.rs` verifies `WordTokenizer` and
    `word_boundaries` against the official Unicode `WordBreakTest.txt`
    conformance vectors.
  - `analysis/tokenize/tests/unigram_chinese.rs` and
    `analysis/tokenize/tests/unigram_sighan.rs` evaluate `UnigramSegmenter`
    Chinese word segmentation F1 on UD GSD and SIGHAN Bakeoff benchmarks
    ([unigram-chinese.md](unigram-chinese.md)).
  - `analysis/tokenize/tests/lattice_prebuilt.rs`,
    `analysis/tokenize/tests/lattice_readings.rs`, and
    `analysis/tokenize/tests/lattice_allocations.rs` verify `LatticeTokenizer`
    and `LatticeAnalyzer` segmentation, `Readings` (`turbo:readings`) columns,
    and zero-allocation prebuilt dictionary lookup.
  - `analysis/tokenize/tests/complex_eval.rs` and
    `analysis/tokenize/tests/complex_graphemes.rs` verify `ComplexSegmenter` and
    `DictionaryWordTokenizer` Thai, Lao, Khmer, and Burmese F1 and
    grapheme-cluster invariants.
  - `analysis/tokenize/tests/detok.rs` verifies `Detokenizer` and
    `DetokenizerDictionary` span and text reconstruction.
