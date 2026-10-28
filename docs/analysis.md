# Document analysis

The analysis crates hold a document's text and what annotators find in
it: views of the text, sentences, tokens, the embedder's subwords, tags,
entities, vectors and chunks. Each kind of result is a layer, and each
layer is its own Rust type, so reading one gives that type with its own
fields. Annotators declare the layer types they read and write, and the
compiler holds them to it.

## Design direction

It is the base the chunker and the analysis pipeline are built on.
Learned components are heads on one shared encoder pass trained with our
own Rust trainers; rule components are ours; each component is judged
against spaCy, Stanza and other current stacks on held-out data, never
against Java output. Two standing assumptions govern this direction
(#1509): JVM goldens are no longer the test oracle (no new code may load
a classical `.bin` model, reproduce Java platform behaviour or use JVM
output as an oracle), and English syntax models trained on Universal
Dependencies English EWT (CC BY-SA 4.0) ship in the accept-first
download tier because every English UD treebank carries a ShareAlike or
non-commercial licence.
The crates are Rust only; the gRPC server serves analysis documents and
layers over them (`server/src/analysis.rs`, [grpc-analysis.md](grpc-analysis.md)).

## Crates

| Crate | Path | Contents |
|---|---|---|
| `turbo-analysis` | `analysis/core/` | `Document`, `View`, `ViewBuilder`, bidirectional UTF-8 `Alignment` tables, `Layer`, `Key`, `KeyId`, `Provenance`, `Annotator`, `Analyzer`, `Registry`, and binary `DocumentGraph` export/import |
| `turbo-analysis-layers` | `analysis/layers/` | Shared columnar layers: `Sentences`, `Paragraphs`, `Tokens`, `TokenReadings`, `Subwords`, `PosTags`, `Lemmas`, `Entities`, `Chunks`, `ChunkTags`, `Dependencies`, `Phrases`, `Numbers`, `Money`, `Quantities`, `Temporals`, `DocumentDate`, `Regions`, `Times`, `Ordinals`, `Pii`, `Language`, `LanguageSpans`, `Categories`, `Sentiment`, `CorefClusters`, `CorefMentions`, `Keywords`, `GrammarIssues`, `TextQuality`, `SpellingSuggestions`, `EmojiMentions`, `SocialEntities`, `EmbeddedAssets`, `EncodingArtifacts`, `BoilerplateRegions`, `NoiseRegions`, `GeoLocations`, `WordNetSenses`, `LexicalExpansions`, `Vectors<L>`, `VectorCodes<L>`, `Levels`, `Groups<L>`, `Edges<From, To>`, `Confidence<L>`, `Evidence<L>`, and `Matches` |
| `turbo-chunker` | `analysis/chunker/` | Boundary-aware retrieval chunking (`Chunker`), hierarchical `Levels`, and per-chunk, per-section, and per-document term statistics and BM25 scoring ([chunker-text-signals.md](chunker-text-signals.md)) |
| `turbo-group` | `analysis/group/` | MinHash/SimHash sketches, near-duplicate grouping, version links inside a document and across a batch, and structural boilerplate detection |
| `turbo-vector-index` | `analysis/vector-index/` | Exact in-memory vector index over rows of one embedding space: top-$k$ by dot product or cosine, TurboQuant 2-to-4-bit quantized storage (feature `turboquant`), and MaxSim late interaction over token vectors ([turboquant.md](turboquant.md)) |
| `turbo-structure` | `analysis/structure/` | Document structure (`turbo:structure`) from plain text and Markdown (`StructureAnnotator`), and projecting `turbo:structure` into a normalized view (`StructureProjector`, [structure-projection.md](structure-projection.md)) |
| `turbo-abbrev` | `analysis/abbrev/` | Schwartz-Hearst abbreviation and acronym definition extraction (`turbo:abbreviations`), short-to-long-form edges, and Ab3P MED1250 evaluation |
| `turbo-matcher` | `analysis/matcher/` | Automaton token-pattern matcher (`Matcher`), token-trie phrase matcher (`PhraseMatcher`), and `SpanRuler` writing named `turbo:matches` instances |
| `turbo-analysis-embed` | `analysis/embed/` | Shared transformer encoder pass as analysis steps: subword tokenization, pooled sentence, chunk, and level vectors with optional Matryoshka `VectorOptions::output_dim` truncation, `VectorCodesStep<L>` writing `VectorCodes<L>` (`turbo:vector-codes<L>`) as `int8` or packed `ubinary` codes, and multi-task sequence/token classification heads (`BundleSequenceScorer`) |
| `turbo-sat` | `analysis/sat/` | Neural Segment Any Text (`SatSentenceDetector`) over a 12-layer XLM-RoBERTa encoder with LoRA adapter, snapping boundary cuts to grapheme cluster boundaries ([neural-annotators.md](neural-annotators.md)) |
| `turbo-sentdetect` | `analysis/sentdetect/` | `ParagraphSplitter`, `RuleSentenceDetector`, `InventorySentenceDetector`, `NewlineSentenceDetector`, and `SentenceDetectorME` ([segmentation.md](segmentation.md)) |
| `turbo-tokenize` | `analysis/tokenize/` | `WhitespaceTokenizer`, `SimpleTokenizer`, `Uax29Tokenizer`, `TokenizerME`, Chinese `UnigramTokenizer` ([unigram-chinese.md](unigram-chinese.md)), Japanese `LatticeTokenizer`, Southeast Asian `ComplexScriptTokenizer`, and `DictionaryDetokenizer` ([segmentation.md](segmentation.md)) |
| `turbo-normalize` | `analysis/normalize/` | Rewriting view normalizers (`NormalizeChain`, Unicode NFC/NFD/NFKC/NFKD, case/accent/punctuation/digit folding,UTS #39 confusables, HTML/Markdown stripping, PDF dehyphenation, mojibake repair, contraction expansion, SymSpell + 5-gram context spell correction) and diagnostic scanners ([normalizers.md](normalizers.md)) |
| `turbo-numeric` | `analysis/numeric/` | `NumberScanner` ([numbers.md](numbers.md)), `MoneyExtractor`, `EcbRates`, `MoneyConverter`, `DatedMoneyConverter` ([money.md](money.md)), `QuantityExtractor` ([quantities.md](quantities.md)), `TemporalExtractor`, `DocumentDateSelector` ([temporal.md](temporal.md)), and region-aware wrappers over `turbo:regions` |
| `turbo-grammar` | `analysis/grammar/` | Deterministic typed production grammar with chart parsing for numbers, ordinals, percentages, quantities, temperatures, money, regions, and temporal expressions across English, Spanish, and Dutch, plus clause-level grammar and style diagnostics ([grammar.md](grammar.md)) |
| `turbo-pii` | `analysis/pii/` | Structured PII detection across 20 built-in identifier kinds with checksum validation (Luhn, mod 97, mod 11, ISO 7064, base58check, bech32/bech32m, EIP-55 Keccak-256), caller `CustomExtractor` kinds, byte-preserving masking (`MaskedView`), pseudonymization, HMAC-SHA-256 tokenization, and redacted audit reports ([pii.md](pii.md)) |
| `turbo-postag` | `analysis/postag/` | Universal Dependencies and Penn Treebank part-of-speech tagging (`FeedforwardPosTagger`, `BiLstmCrfPosTagger`, and `PosTagger`) writing `PosTags` ([neural-annotators.md](neural-annotators.md)) |
| `turbo-lemmatizer` | `analysis/lemmatizer/` | Edit-tree neural lemmatizer (`EditTreeLemmatizer`), dictionary lemmatizer (`DictionaryLemmatizer`), and Morfologik FSA lemmatizer writing `Lemmas` ([lemma-case-evaluation.md](lemma-case-evaluation.md)) |
| `turbo-morfologik` | `analysis/morfologik/` | Zero-copy `CFSA2` and `FSA5` finite-state morphological dictionary lookup and lemmatization |
| `turbo-depparse` | `analysis/depparse/` | Arc-standard and Chen-Manning neural projective transition dependency parser and trainer writing `Dependencies` ([dependency-parser.md](dependency-parser.md), [feedforward-dependency-parser.md](feedforward-dependency-parser.md)) |
| `turbo-namefind` | `analysis/namefind/` | Dictionary, compiled regex, linear-chain CRF (`CrfNameFinder`, [crf-namefinder.md](crf-namefinder.md)), and ONNX transformer/bi-LSTM-CRF (`NameFinderDL`, [namefind-dl.md](namefind-dl.md)) named entity recognizers, `FilterEntities`, and `ConsensusEntityMerger` ([namefinder-evaluation.md](namefinder-evaluation.md)) |
| `turbo-relation` | `analysis/relation/` | Dependency-path and pattern relation extraction between entity heads writing `Edges<Entities, Entities>` ([relation.md](relation.md)) |
| `turbo-coref` | `analysis/coref/` | Deterministic rule sieve and neural mention-pair coreference resolution writing `CorefMentions` and `CorefClusters` ([coref.md](coref.md)) |
| `turbo-doccat` | `analysis/doccat/` | Transformer sequence classification (`DocumentCategorizerDL`, `DlSpanCategorizer`), multi-label (`MultiLabelDoccatModel`), and zero-shot (`ZeroShotCategorizer`) document categorization ([doccat.md](doccat.md)), plus polarity/intensity vector scoring (`SentimentPolarities`, `SentimentVectorScorer`, `SpanSentimentScorer`) and pragmatic negation/sarcasm/aspect refinement (`PragmaticAnalyzer`, `AspectSentimentExtractor`) ([sentiment.md](sentiment.md)) |
| `turbo-keywords` | `analysis/keywords/` | Unsupervised YAKE, TextRank, and RAKE keyword and keyphrase extraction writing `Keywords` (`turbo:keywords`, [keywords.md](keywords.md)) |
| `turbo-text-quality` | `analysis/text-quality/` | Per-view and per-sentence readability, lexical diversity, and quality heuristics writing `TextQuality` (`turbo:text-quality`, [text-quality.md](text-quality.md)) |
| `turbo-langdetect` | `analysis/langdetect/` | Document-level character $n$-gram language detection (`LanguageDetector`, `ProbingLanguageDetector`) writing `Language` ([langdetect.md](langdetect.md)) |
| `turbo-langspan` | `analysis/langspan/` | Code-switching and mixed-language span segmentation (`LanguageSpanDetector`) writing `LanguageSpans` and `Language` ([langdetect.md](langdetect.md)) |
| `turbo-geo` | `analysis/geo/` | Gazetteer toponym matching and hierarchical geographic disambiguation (`GeoEntityLinker`) writing `GeoLocations` (`turbo:geo-locations`) |
| `turbo-wordnet` | `analysis/wordnet/` | WNDB and WN-LMF WordNet lexicon reader, Morphy lemmatizer, synset sense disambiguation, and lexical expansion writing `WordNetSenses` and `LexicalExpansions` |
| `turbo-stem` | `analysis/stem/` | Porter, 21 Snowball, and language-specific light/minimal stemmers (`StemAnnotator`) writing `Stems` (`turbo:stems`) |
| `turbo-hunspell` | `analysis/hunspell/` | Hunspell `.aff`/`.dic` reader, prefix/suffix/compound stemmer (`HunspellStemmer`), and `HunspellAnnotator` writing `Stems` (`turbo:stems`) |
| `turbo-stopword` | `analysis/stopword/` | Multilingual stopword lists across 11 languages and `StopwordAnnotator` writing `Stopwords` (`turbo:stopwords`) |
| `turbo-glossary` | `analysis/glossary/` | CSV and TBX 2/3 glossary readers, Aho-Corasick character matcher, zero-copy double-array `MappedGlossaryMatcher`, and token/stem matcher writing `GlossaryTerms` (`turbo:glossary-terms`) |
| `turbo-bpe-trainer` | `analysis/bpe-trainer/` | Byte-level, SentencePiece, and word-bounded BPE vocabulary trainers ([bpe-trainer.md](bpe-trainer.md)) |
| `turbo-formats` | `analysis/formats/` | Corpus readers and evaluation scorers: Universal Dependencies CoNLL-U (`analysis/formats/src/conllu/` with byte-identical round trips), CoNLL-2003, GermEval 2014, WNUT-17, and MultiCoNER v2 NER (`analysis/formats/src/ner/`), LitBank brat entity and coreference standoff (`analysis/formats/src/ner/brat/`), MASC 3.0.0 standoff (`analysis/formats/src/masc/`), pinned dataset recipes (`analysis/formats/recipes/`), and precision/recall/$F_1$ evaluators (`analysis/formats/src/evaluate/`; true positives are a multiset intersection, so recall is at most 1) |
| `turbo-zoo` | `analysis/zoo/` | JSON model and lexicon catalog (`analysis/zoo/index/zoo.json`) with artifact byte-length, SHA-256, and SHA-512 content-digest verification, license-tier gating (`Permissive` vs `AcceptFirst`), and atomic cache layout ([zoo.md](zoo.md)) |

## Document, views, and alignments (`analysis/core/src/doc/mod.rs`)

A `Document` owns the source UTF-8 text (`ViewId::SOURCE`, key `Key::DEFAULT`)
and a typed map of named views and columnar layers.

### Derived views and `ViewBuilder`

When a normalizer or masker (`WhitespaceNormalizer`, `CaseFoldNormalizer`,
`MarkupStripper`, `MaskedView`, `DictionaryDetokenizer`) rewrites a `View`, it
writes into a `ViewBuilder` (`analysis/core/src/doc/view.rs`), recording each
inserted, deleted, or replaced span as an edit run. The resulting `View`
carries an `Alignment` table that maps UTF-8 byte offsets and half-open spans
bidirectionally between the derived view and its parent view (and transitively
back to `ViewId::SOURCE`) in $O(\log E)$ time over $E$ edit runs.

Every character boundary in a derived view maps cleanly to the source character
span it came from, so downstream annotators can run on normalized or masked
text while callers project any resulting entity, chunk, or keyword span back
to exact byte offsets in the original document.

## Columnar layers, anchors, and provenance (`analysis/core/src/layer.rs`)

Every analysis output implements `Layer` (`analysis/core/src/layer.rs`):

- **Columnar layout (SoA)**: layers store parallel `Vec` columns (`start`,
  `end`, `tag`, `head`, `score`) rather than per-item heap objects, keeping
  cache locality tight and eliminating per-token string allocations via
  interned `TagSet` indices (`Tag(u16)`).
- **Typed anchors (`Anchor`)**: each layer declares how its items anchor to the
  document:
  - `Anchor::Text`: half-open UTF-8 byte spans `[start..end)` (or discontinuous
    fragment spans) into a specific `View`, validated against `Self::OVERLAP`
    (`Overlap::None`, `Overlap::Nested`, or `Overlap::Any`) and UTF-8 character
    boundaries when inserted into `Document`.
  - `Anchor::Layer(LayerName)`: one item per item of another layer (for
    example, `PosTags`, `Lemmas`, `Stems`, and `Stopwords` anchored 1:1 to
    `Tokens`, or `Vectors<Sentences>` anchored 1:1 to `Sentences`).
  - `Anchor::Document`: document-wide layers such as `Language`, `DocumentDate`,
    `Categories`, `Sentiment`, and `Regions`.
- **Cross-layer index validation (`Layer::indexes`)**: layers that reference
  other layers by item index (such as `Tokens` referencing `Sentences`,
  `Entities` referencing `Tokens`, or `Edges<From, To>` referencing endpoint
  layers) report their maximum index via `indexes()`; `Document::put` verifies
  that every referenced layer exists and that all indices are in bounds before
  accepting the layer.
- **Provenance (`Provenance`)**: every stored layer records the annotator step
  name, model identifier, configuration hash, anchor instance, and upstream
  layer keys that produced it.

## Annotators, pipelines, and registry (`analysis/core/src/annotate.rs`)

An annotator implements `Annotator` with typed `Reads` and `Writes` tuples:

```rust
use turbo_analysis::{Analyzer, Document};
use turbo_sentdetect::RuleSentenceDetector;
use turbo_tokenize::WordTokenizer;

let analyzer = Analyzer::builder()
    .add(RuleSentenceDetector::new("en"))
    .add(WordTokenizer::new())
    .build()?;

let mut docs = vec![Document::new("Dr. Smith arrived at 5 p.m. He brought two samples.")];
for res in analyzer.run(&mut docs) {
    res?;
}
```

When `Analyzer::builder().build()` constructs a pipeline, it verifies that
every required input key `(LayerName, instance_name)` is produced by an earlier
step (or is the initial `ViewId::SOURCE`), rejecting missing dependencies with
a typed error before any document is processed. Named layer and view instances
(`Key::<Entities>::named("pii")`, `Key::<View>::named("normalized")`) allow
multiple recognizers or views of the same Rust type to coexist in one
`Document` without collision.

## Binary wire export and import (`analysis/core/src/export/`)

`DocumentGraph` (`analysis/core/src/export/mod.rs`) serializes a `Document`,
its derived `View`s and edit alignments, its columnar layers, and its
`Provenance` records into a deterministic, bounds-checked binary wire envelope
for zero-loss transport across gRPC (`server/src/analysis.rs`,
[grpc-analysis.md](grpc-analysis.md)) and persistent caching.

## Testing, accuracy, and speed

- **Core invariants (`analysis/core/src/`)**: unit and integration tests verify
  UTF-8 boundary enforcement, overlap rules (`Overlap::None`, `Overlap::Nested`,
  `Overlap::Any`), cross-layer index bounds, multi-hop `View` alignment
  projection, pipeline dependency validation, and `DocumentGraph` binary
  round-tripping.
- **Component accuracy and throughput**: each analysis crate documents its
  held-out evaluation benchmarks (UD English EWT and multilingual treebanks,
  CoNLL-2003 / GermEval 2014 / MultiCoNER v2 NER, LitBank brat entity and
  coreference, SIGHAN Bakeoff CWS, Ab3P MED1250 abbreviations,
  Recognizers-Text grammar suites, MASC and RFC/Wikipedia PII partitions) and
  CPU throughput measurements in its companion guide under `docs/` and crate
  `README.md`.

### Abbreviations and MED1250 (`turbo-abbrev`)

`AbbreviationExtractor` finds short/long form pairs in running text and
scores against the 1,203 distinct gold pairs of Ab3P's MED1250 corpus:

| Long form punctuation dropped | True positives | False positives | False negatives | Precision | Recall | F1 |
|---|---|---|---|---|---|---|
| yes (as shipped) | 956 | 45 | 247 | 0.9550 | 0.7947 | 0.8675 |
| no | 950 | 51 | 253 | 0.9491 | 0.7897 | 0.8621 |

### Rule matching (`turbo-matcher`)

A token pattern test reads one string attribute (`Attr`) or one lexical flag:

| `Attr` | Value | spaCy |
|---|---|---|
| `Text` | the token's text in its view | `ORTH`, `TEXT` |
| `Lower` | the text lowercased (`str::to_lowercase`) | `LOWER` |
| `Prefix`, `Suffix` | the first character, the last three | `PREFIX`, `SUFFIX` |
| `Shape` | `word_shape` ("Xxxxx", "dd.d") | `SHAPE` |
| `Norm` | a string column the caller aligns to the tokens | `NORM` |
| `Lemma`, `Stem` | `Lemmas`, `turbo:stems` | `LEMMA` |
| `Pos` | the tag in `PosTags` | `TAG`, `POS` |
| `Entity` | the kind of the first entity in `Entities` covering the token, or the empty string where none does | `ENT_TYPE` |

Measured on one million two- to four-word phrases (3.05 million phrase tokens,
17,023 distinct values) matched over `Attr::Lower`:

| Measure | Value |
|---|---|
| Build | 1.2 s |
| Trie states | 1.52 million |
| Heap the matcher holds (`heap_bytes`) | 64.9 MiB |
| Growth of the process's RSS over the build | 92.1 MiB |
| Dev file (24,788 tokens, 0.13 MB): 41,544 matches | 2.8 ms, 8.9 million tokens/s, 45 MB/s |
| Test file (24,740 tokens, 0.12 MB): 41,268 matches | 2.8 ms, 8.9 million tokens/s, 45 MB/s |

### Chunker boundary costs, sketches, and levels

Each inter-sentence gap in `turbo-chunker` takes its strongest structural break:

| Break at the gap | Reason | Default cost |
| --- | --- | --- |
| A section starts or ends | `Section` | 0 |
| A heading starts | `Heading` | 100 |
| A paragraph, list, table, code block, quote, caption, footnote or page header or footer starts or ends | `Paragraph` | 400 |
| A list item or table row starts or ends | `Item` | 600 |
| None of these | `Sentence` | 1000 |

Per item, `turbo-group`'s sketch layer holds:

| Column | What it is |
| --- | --- |
| `words` | The item's word count |
| `exact` | A 128-bit hash of the word sequence: equal for the same words in the same order, whatever lies between them |
| `shape` | A 64-bit hash of the sequence with every number word replaced by one placeholder |
| `simhash` | The 64-bit SimHash of the item's shingles, each occurrence counted |
| `minhash` | 64 MinHash values of the item's set of shingles by one-permutation hashing with optimal densification |

Multi-level retrieval (`analysis/eval/records/turbo-chunker-254-200-level-*`,
`all-MiniLM-L6-v2`, `--max 254 --target 200`, plain structure) nDCG@10:

| Level, vectors | scifact | nfcorpus | arguana | qasper | gutenqa |
| --- | --- | --- | --- | --- | --- |
| chunk, text | 0.6506 (9126) | 0.3176 (6819) | 0.4955 (11145) | 0.2817 (10578) | 0.8635 (69226) |
| chunk, local | 0.6504 | 0.3172 | 0.4918 | 0.2817 | 0.8637 |
| chunk, contextual | 0.6599 | 0.3230 | 0.4932 | 0.2750 | 0.8563 |
| section, text | 0.6506 (9126) | 0.3176 (6819) | 0.4955 (11145) | 0.2841 (10207) | 0.8570 (67572) |
| section, local | 0.6504 | 0.3172 | 0.4918 | 0.2838 | 0.8569 |
| section, contextual | 0.6599 | 0.3230 | 0.4932 | 0.2771 | 0.8506 |
| document, text | 0.6402 (5183) | 0.3097 (3633) | 0.5033 (8674) | 0.2519 (416) | 0.3161 (100) |
| document, local | 0.6518 | 0.3167 | 0.4872 | 0.2328 | 0.6238 |

### Hybrid BM25 + dense retrieval, structure, and larger embedders

Hybrid fusion sweep across `scifact`, `nfcorpus`, `arguana`, and `qasper`:

| Chunker | Fusion | `scifact` | `nfcorpus` | `arguana` | `qasper` | Mean | Gate vs stage-0 (3 baselines) | Gate vs `bm25-document` |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: | :--- | :--- |
| `turbo-chunker-254-200` | `weighted` `w = 0.35` (default) | `69.24` | `34.28` | `51.65` | `37.85` | `48.26` | PASS (4/4, 4/4, 4/4) | PASS (3/4, `qasper` `+0.09`) |
| `turbo-chunker-254-200` | `weighted` `w = 0.38` | `69.13` | `34.28` | `51.22` | `38.36` | `48.25` | PASS (4/4, 4/4, 4/4) | PASS (3/4, `qasper` `+0.60`) |
| `turbo-chunker-254-200` | `weighted` `w = 0.40` | `69.04` | `34.11` | `51.12` | `38.44` | `48.18` | PASS (4/4, 3/4, 4/4) | PASS (3/4, `qasper` `+0.68`) |
| `turbo-chunker-254-200` | `weighted` `w = 0.45` | `68.79` | `33.96` | `50.86` | `39.11` | `48.18` | PASS (3/4, 3/4, 4/4) | PASS (3/4, `qasper` `+1.35`) |
| `turbo-chunker-254-200` | `weighted` `w = 0.50` (untuned 50/50) | `68.45` | `33.62` | `50.48` | `39.42` | `47.99` | PASS (3/4, 3/4, 4/4) | PASS (3/4, `qasper` `+1.66`) |
| `turbo-chunker-254-200` | `rrf` `k = 60, w = 0.50` (untuned RRF) | `68.13` | `34.17` | `49.69` | `36.82` | `47.20` | PASS (2/4, 3/4, 4/4) | FAIL (3/4, `qasper` `-0.94`) |

Held-out `bm25_weight` selection (`Hybrid::choose_bm25_weight_held_out`,
`analysis/eval/tests/heldout_hybrid.rs`,
`analysis/eval/records/heldout-hybrid-955/heldout-hybrid-955.json`):

| Set | Chosen `bm25_weight` | Selection queries | Selection nDCG@10 | Held-out queries | Held-out nDCG@10 | Held-out dense (`0.0`) | Held-out BM25 (`1.0`) |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `scifact` | 0.20 | 150 | 0.6860 | 150 | 0.7269 | 0.6667 | 0.6506 |
| `nfcorpus` | 0.30 | 162 | 0.3438 | 161 | 0.3472 | 0.3188 | 0.2868 |
| `arguana` | 0.30 | 703 | 0.5162 | 703 | 0.5179 | 0.4938 | 0.4474 |
| `qasper` | 0.60 | 726 | 0.4050 | 725 | 0.3854 | 0.2706 | 0.3678 |

Structure ablation (`--structure none`, `plain`, and `markdown`) at `--max 254 --target 200`:

| Structure | scifact | nfcorpus | arguana | qasper | gutenqa |
| --- | --- | --- | --- | --- | --- |
| none | 0.6506 (9126) | 0.3176 (6819) | 0.4955 (11145) | 0.2792 (9453) | 0.8631 (69197) |
| plain | 0.6506 (9126) | 0.3176 (6819) | 0.4955 (11145) | 0.2817 (10578) | 0.8635 (69226) |
| markdown | 0.6506 (9126) | 0.3176 (6819) | 0.4955 (11145) | 0.2825 (10544) | 0.8634 (69228) |

Larger embedders with `turbo-chunker --max 254 --target 200`, nDCG@10:

| Embedder | Retrieval | `scifact` | `nfcorpus` | `arguana` | `qasper` | Mean |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: |
| all-MiniLM-L6-v2 | dense | `65.06` | `31.76` | `49.55` | `28.17` | `43.63` |
| bge-small-en-v1.5 | dense | `70.32` | `35.21` | `62.86` | `30.32` | `49.68` |
| bge-base-en-v1.5 | dense | `73.74` | `36.65` | `64.51` | `30.82` | `51.43` |
| all-MiniLM-L6-v2 | hybrid, `w = 0.35` | `69.24` | `34.28` | `51.65` | `37.85` | `48.25` |
| bge-small-en-v1.5 | hybrid, `w = 0.35` | `70.55` | `34.90` | `58.57` | `38.62` | `50.66` |
| bge-base-en-v1.5 | hybrid, `w = 0.35` | `72.26` | `36.05` | `60.47` | `39.21` | `52.00` |

Query instruction ablation on `bge-small-en-v1.5` and `bge-base-en-v1.5`, nDCG@10:

| Embedder | Retrieval | Query | `scifact` | `nfcorpus` | `arguana` | `qasper` | Mean |
| :--- | :--- | :--- | ---: | ---: | ---: | ---: | ---: |
| bge-small-en-v1.5 | dense | none | `70.32` | `35.21` | `62.86` | `30.32` | `49.68` |
| bge-small-en-v1.5 | dense | instruction | `70.21` | `35.15` | `61.39` | `31.28` | `49.51` |
| bge-base-en-v1.5 | dense | none | `73.74` | `36.65` | `64.51` | `30.82` | `51.43` |
| bge-base-en-v1.5 | dense | instruction | `73.90` | `36.70` | `62.84` | `31.64` | `51.27` |
| bge-small-en-v1.5 | hybrid, `w = 0.35` | none | `70.55` | `34.90` | `58.57` | `38.62` | `50.66` |
| bge-small-en-v1.5 | hybrid, `w = 0.35` | instruction | `70.44` | `35.16` | `57.66` | `38.63` | `50.47` |
| bge-base-en-v1.5 | hybrid, `w = 0.35` | none | `72.26` | `36.05` | `60.47` | `39.21` | `52.00` |
| bge-base-en-v1.5 | hybrid, `w = 0.35` | instruction | `71.77` | `36.42` | `59.29` | `39.22` | `51.68` |

### Encoder audit and optional Granite download

The licence audit does not establish a permissive training lineage for
`all-MiniLM-L6-v2`, `BAAI/bge-base-en-v1.5`, or `minishlab/potion-base-8M`
([licences.md](licences.md)). `ibm-granite/granite-embedding-30m-english` is an
optional local encoder (`bundle/recipes/granite-embedding-30m-english.json`):
Apache-2.0 weights, CC BY-SA training inputs, and web-scraped and IBM-internal
text requiring explicit user acceptance ([zoo.md](zoo.md), "Granite requires an
explicit download decision"). Under the document-chunker dense retrieval gate on
CPU at `MODEL` precision (`analysis/eval/records/granite-embedding-30m-document`,
`all-minilm-l6-v2-document`, and `potion-base-8m-document`), Granite leads
`all-MiniLM-L6-v2` on all four BEIR sets:

| set | granite-embedding-30m-english | all-MiniLM-L6-v2 | potion-base-8M |
| --- | ---: | ---: | ---: |
| scifact | 0.7172 | 0.6402 | 0.4914 |
| nfcorpus | 0.3282 | 0.3094 | 0.2305 |
| arguana | 0.5562 | 0.5015 | 0.3625 |
| qasper | 0.2767 | 0.2519 | 0.1970 |
| mean | 0.4696 | 0.4258 | 0.3204 |

`turbo-train-recipes static-embed` ships as experimental tooling for static
embedding table initialization and contrastive training (`0.3354` untrained on
`scifact`).

### Evaluation sets and static-to-contextual cascade records (#712)

Evaluation datasets registered under `analysis/eval/src/sets/`:

| Set | Source | Licence | Spans | Digests |
|---|---|---|---|---|
| `scifact`, `nfcorpus`, `arguana` | BEIR layout, from MTEB's plain copies at a recorded revision | CC BY-SA 4.0 (BEIR dataset cards) | none | recorded |
| `legalbench-rag-mini` | a shared folder, placed by hand | MIT for the benchmark; the documents under their sources' terms | code points | none (no stable file URLs) |
| `conteb-covid-qa` | Parquet | none for the article text, so refused | the judged chunk | not recorded yet |
| `conteb-squad` | Parquet, ConTEB's SQuAD set (`illuin-conteb/squad-conteb-eval`, revision 8f81cbd), validation split | CC BY-SA 4.0 (SQuAD over Wikipedia; downloaded at use, never bundled) | the judged chunk | recorded |
| `gutenqa` | Parquet | MIT (dataset card) | the quoted passage | recorded |
| `hicbench` | no copy that can be fetched | none stated, so refused | quoted sentences | none |
| `qasper` | gzip tar, test split | CC BY 4.0 | quoted evidence, located by paragraph or text | recorded |

Static-to-contextual cascade records (`#712`) across BEIR sets and GutenQA:

| Set | Arm | Recall@10 | nDCG@10 | Contextual scored/query |
|---|---|---|---|---|
| scifact | Static only | 0.3325 | 0.2564 | 0 |
| scifact | Contextual only | 0.4650 | 0.3726 | 12387 |
| scifact | Cascade C=5 | 0.4650 | 0.3726 | 8534 |
| scifact | Cascade C=64 | 0.4650 | 0.3726 | 12387 |
| nfcorpus | Static only | 0.0537 | 0.1255 | 0 |
| nfcorpus | Contextual only | 0.1254 | 0.2655 | 21088 |
| nfcorpus | Cascade C=5 | 0.1254 | 0.2660 | 11841 |
| nfcorpus | Cascade C=64 | 0.1254 | 0.2655 | 21088 |
| arguana | Static only | 0.3600 | 0.2280 | 0 |
| arguana | Contextual only | 0.3700 | 0.2502 | 9564 |
| arguana | Cascade C=5 | 0.3700 | 0.2502 | 7659 |
| arguana | Cascade C=64 | 0.3700 | 0.2502 | 9564 |

GutenQA (100 documents, 339,809 level units, 50 queries):

| Arm | Recall@10 | nDCG@10 | Contextual scored/query |
|---|---|---|---|
| Static only | 0.4800 | 0.2806 | 0 |
| Contextual only | 0.6200 | 0.4374 | 339809 |
| Cascade C=5 | 0.5800 | 0.4105 | 968 |
| Cascade C=64 | 0.6000 | 0.4268 | 11651 |
