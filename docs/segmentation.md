# Sentence and token segmentation (`turbo-sentdetect`, `turbo-sat`, `turbo-tokenize`)

Sentence segmentation and tokenization turn a `View` into the `Sentences`
(`opennlp:sentences`) and `Tokens` (`opennlp:tokens`) layers in
`turbo-analysis-layers`. All offsets are UTF-8 byte offsets on character
boundaries into the view they annotate.

## Paragraph and sentence segmentation (`turbo-sentdetect`)

`turbo-sentdetect` (`analysis/sentdetect/`) provides paragraph splitting and
four complementary sentence segmenters:

| Segmenter | Module | Description |
|---|---|---|
| `ParagraphSplitter` | `analysis/sentdetect/src/paragraph.rs` | Splits a view on blank lines into `Paragraphs` (`turbo:paragraphs`) |
| `RuleSentenceDetector` | `analysis/sentdetect/src/rules/mod.rs` | UAX #29 sentence boundaries refined with per-language abbreviation tables, bracket/quote balancing, list-marker suppression, and unit/initial guards across 40+ languages |
| `InventorySentenceDetector` | `analysis/sentdetect/src/inventory/mod.rs` | High-throughput state-machine sentence detector with compiled abbreviation and starter inventories across 35 languages (`InventoryLanguage`) |
| `NewlineSentenceDetector` | `analysis/sentdetect/src/newline.rs` | Line-oriented sentence splitter for line-delimited logs, transcripts, and verse |
| `SentenceDetectorME` | `analysis/sentdetect/src/me/mod.rs` | Maximum-entropy end-of-sentence classifier over candidate punctuation positions and abbreviation dictionaries |

### Rule and inventory sentence detection

`RuleSentenceDetector` starts from Unicode Standard Annex #29 (`UAX #29`)
sentence breaks (`analysis/sentdetect/src/rules/uax29.rs`) and refines them
using:

- Per-language abbreviation and title tables
  (`analysis/sentdetect/src/rules/tables.rs` and
  `analysis/sentdetect/src/rules/languages.rs`),
- Balanced parentheses, brackets, and quotation marks
  (`analysis/sentdetect/src/rules/brackets.rs`),
- Numbered and bulleted list-item guards
  (`analysis/sentdetect/src/rules/lists.rs`), and
- Numeric unit and initialism suppression
  (`analysis/sentdetect/src/rules/units.rs`).

`InventorySentenceDetector` (`analysis/sentdetect/src/inventory/mod.rs`) scans
sentence-final punctuation directly and evaluates each candidate boundary with
a compiled abbreviation and lower-case starter judge
(`analysis/sentdetect/src/inventory/judge.rs`).

### Neural Segment Any Text (`turbo-sat`)

For unpunctuated, ASR, or noisy conversational text where punctuation rules
under-segment, `SatSentenceDetector` (`analysis/sat/src/lib.rs`) runs a
12-layer XLM-RoBERTa token-classification encoder with a LoRA adapter to score
each token's sentence-boundary probability and emits `Sentences`
(`opennlp:sentences`) with per-sentence confidence scores (`SatConfidence`).

## Tokenization (`turbo-tokenize`)

`turbo-tokenize` (`analysis/tokenize/`) segments a `View` (or each sentence in
`Sentences`) into `Tokens` (`opennlp:tokens`), and optionally writes
morphological readings into `TokenReadings` (`turbo:token-readings`):

| Tokenizer | Module | Description |
|---|---|---|
| `WhitespaceTokenizer` | `analysis/tokenize/src/whitespace.rs` | Splits on Unicode `White_Space` runs |
| `SimpleTokenizer` | `analysis/tokenize/src/simple.rs` | Splits on transitions between alphabetic, numeric, whitespace, and punctuation character classes |
| `WordTokenizer` / `UntypedWordTokenizer` | `analysis/tokenize/src/uax29/mod.rs` | Unicode Standard Annex #29 word-boundary segmenter with `WordType` classification |
| `TokenizerME` | `analysis/tokenize/src/me/mod.rs` | Maximum-entropy boundary classifier inside whitespace-delimited tokens with an alphanumeric skip optimization |
| `UnigramTokenizer` | `analysis/tokenize/src/unigram/mod.rs` | Dictionary-plus-Viterbi unigram word segmenter with out-of-vocabulary character HMM for Chinese (`cmn`) |
| `LatticeTokenizer` | `analysis/tokenize/src/lattice/mod.rs` | Double-array trie + connection-cost matrix Viterbi morphological tokenizer for Japanese (`jpn`), emitting `Tokens` and `TokenReadings` |
| `ComplexScriptTokenizer` | `analysis/tokenize/src/complex/mod.rs` | Dictionary + bi-LSTM / grapheme-cluster word segmenter for Thai (`tha`), Lao (`lao`), Khmer (`khm`), and Myanmar (`mya`) |
| `DictionaryDetokenizer` | `analysis/tokenize/src/detok.rs` | Reconstructs untokenized text and derived `View` alignments from `Tokens` using per-token merge rules (`MergeRight`, `MergeLeft`, `MergeBoth`, `RightLeftMatching`) |

### Script-specific tokenizers

- **Chinese (`UnigramTokenizer`, `analysis/tokenize/src/unigram/mod.rs`)**:
  builds a word lattice over a prefix trie (`analysis/tokenize/src/unigram/trie.rs`)
  and decodes the minimum-cost path with a 4-state (`B`, `M`, `E`, `S`)
  character HMM fallback for unknown words (`docs/unigram-chinese.md`).
- **Japanese (`LatticeTokenizer`, `analysis/tokenize/src/lattice/mod.rs`)**:
  builds a word lattice over a double-array trie, character-category grouping
  table (`chardef`), and token-to-token connection matrix (`matrix`), decodes
  the minimum-cost path with Viterbi, and writes surface tokens to `Tokens` and
  part-of-speech/lemma/reading/pronunciation columns to `TokenReadings`.
  Zero-copy prebuilt images (`analysis/tokenize/src/lattice/prebuilt/mod.rs`)
  load directly from aligned byte slices without heap allocation during lookup.
- **Southeast Asian complex scripts (`ComplexScriptTokenizer`,
  `analysis/tokenize/src/complex/mod.rs`)**: segments Thai, Lao, Khmer, and
  Myanmar runs while preserving extended grapheme cluster boundaries
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
    verify neural sentence boundary F1 on unpunctuated text and batch
    throughput.
- **Tokenization tests**:
  - `analysis/tokenize/tests/wordbreak.rs` verifies `Uax29Tokenizer` against
    the official Unicode `WordBreakTest.txt` conformance vectors.
  - `analysis/tokenize/tests/unigram_chinese.rs` and
    `analysis/tokenize/tests/unigram_sighan.rs` evaluate Chinese word
    segmentation F1 on UD GSD and SIGHAN Bakeoff benchmarks
    (`docs/unigram-chinese.md`).
  - `analysis/tokenize/tests/lattice_prebuilt.rs`,
    `analysis/tokenize/tests/lattice_readings.rs`, and
    `analysis/tokenize/tests/lattice_allocations.rs` verify Japanese lattice
    segmentation, `TokenReadings` columns, and zero-allocation prebuilt dictionary
    lookup.
  - `analysis/tokenize/tests/complex_eval.rs` and
    `analysis/tokenize/tests/complex_graphemes.rs` verify Thai, Lao, Khmer, and
    Myanmar F1 and grapheme-cluster invariants.
  - `analysis/tokenize/tests/detok.rs` verifies `DictionaryDetokenizer` span
    and view-alignment reconstruction.
