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
a legacy `.bin` model, reproduce Java platform behaviour or use JVM
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
| `turbo-analysis-layers` | `analysis/layers/` | Shared columnar layers: `Sentences`, `Paragraphs`, `Tokens`, `TokenReadings`, `Subwords`, `PosTags`, `Lemmas`, `Entities`, `Chunks`, `ChunkTags`, `Dependencies`, `Phrases`, `Numbers`, `Money`, `Quantities`, `Temporals`, `DocumentDate`, `Regions`, `Times`, `Ordinals`, `Pii`, `Language`, `LanguageSpans`, `Categories`, `Sentiment`, `CorefClusters`, `CorefMentions`, `Keywords`, `GrammarIssues`, `TextQuality`, `SpellingSuggestions`, `EmojiMentions`, `SocialEntities`, `EmbeddedAssets`, `EncodingArtifacts`, `BoilerplateRegions`, `NoiseRegions`, `GeoLocations`, `WordNetSenses`, `LexicalExpansions`, `Vectors<L>`, `Levels`, `Groups<L>`, `Edges<From, To>`, `Confidence<L>`, `Evidence<L>`, and `Matches` |
| `turbo-chunker` | `analysis/chunker/` | Boundary-aware retrieval chunking (`Chunker`), hierarchical `Levels`, and per-chunk, per-section, and per-document term statistics and BM25 scoring ([chunker-text-signals.md](chunker-text-signals.md)) |
| `turbo-group` | `analysis/group/` | MinHash/SimHash sketches, near-duplicate grouping, version links inside a document and across a batch, and structural boilerplate detection |
| `turbo-vector-index` | `analysis/vector-index/` | Exact in-memory vector index over rows of one embedding space: top-$k$ by dot product or cosine, TurboQuant 2-to-4-bit quantized storage (feature `turboquant`), and MaxSim late interaction over token vectors ([turboquant.md](turboquant.md)) |
| `turbo-structure` | `analysis/structure/` | Document structure (`turbo:structure`) from plain text and Markdown (`StructureAnnotator`), and projecting `turbo:structure` into a normalized view (`StructureProjector`, [structure-projection.md](structure-projection.md)) |
| `turbo-abbrev` | `analysis/abbrev/` | Schwartz-Hearst abbreviation and acronym definition extraction (`turbo:abbreviations`), short-to-long-form edges, and Ab3P MED1250 evaluation |
| `turbo-matcher` | `analysis/matcher/` | Automaton token-pattern matcher (`Matcher`), token-trie phrase matcher (`PhraseMatcher`), and `SpanRuler` writing named `turbo:matches` instances |
| `turbo-analysis-embed` | `analysis/embed/` | Shared transformer encoder pass as analysis steps: subword tokenization, pooled sentence and chunk vectors, and multi-task sequence/token classification heads (`BundleSequenceScorer`) |
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
| `turbo-zoo` | `analysis/zoo/` | Signed model manifest registry (`zoo.toml`), SHA-256 verification, license-tier gating (`Permissive` vs `AcceptFirst`), and atomic cache layout ([zoo.md](zoo.md)) |

## Document, views, and alignments (`analysis/core/src/doc/mod.rs`)

A `Document` owns the source UTF-8 text (`View::ORIGINAL`, key `Key::DEFAULT`)
and a typed map of named views and columnar layers.

### Derived views and `ViewBuilder`

When a normalizer or masker (`WhitespaceNormalizer`, `CaseFoldNormalizer`,
`MarkupStripper`, `MaskedView`, `DictionaryDetokenizer`) rewrites a `View`, it
writes into a `ViewBuilder` (`analysis/core/src/doc/view.rs`), recording each
inserted, deleted, or replaced span as an edit run. The resulting `View`
carries an `Alignment` table that maps UTF-8 byte offsets and half-open spans
bidirectionally between the derived view and its parent view (and transitively
back to `View::ORIGINAL`) in $O(\log E)$ time over $E$ edit runs.

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

An annotator implements `Annotator` with typed `In` (`Inputs`) and `Out`
(`Outputs`) tuples:

```rust
use turbo_analysis::{Analyzer, Document};
use turbo_sentdetect::RuleSentenceDetector;
use turbo_tokenize::Uax29Tokenizer;

let analyzer = Analyzer::builder()
    .add(RuleSentenceDetector::english())
    .add(Uax29Tokenizer::new())
    .build()?;

let mut docs = vec![Document::new("Dr. Smith arrived at 5 p.m. He brought two samples.")];
for res in analyzer.run(&mut docs) {
    res?;
}
```

When `Analyzer::builder().build()` constructs a pipeline, it verifies that
every required input key `(LayerName, instance_name)` is produced by an earlier
step (or is the initial `View::ORIGINAL`), rejecting missing dependencies with
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
  CoNLL-2003 / OntoNotes NER, SIGHAN Bakeoff CWS, Ab3P MED1250 abbreviations,
  Recognizers-Text grammar suites, MASC and RFC/Wikipedia PII partitions) and
  CPU throughput measurements in its companion guide under `docs/` and crate
  `README.md`.
