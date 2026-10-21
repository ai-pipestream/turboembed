# Document analysis

The analysis crates hold a document's text and what annotators find in
it: views of the text, sentences, tokens, the embedder's subwords, tags,
entities, vectors and chunks. Each kind of result is a layer, and each
layer is its own Rust type, so reading one gives that type with its own
fields. Annotators declare the layer types they read and write, and the
compiler holds them to it.

It is the base the chunker is built on, and the base OpenNLP 3.x is to
be rewritten on, as add-ons: OpenNLP is the reference, and the Rust side
follows it on its own schedule. The crates are Rust only for now;
nothing here is in the C header or the gRPC server.

## Crates

| Crate | Path | Contents |
|---|---|---|
| `turbo-analysis` | `analysis/core/` | The document, views, offset tables, layers, keys, provenance, annotators, the analyzer and the registry |
| `turbo-analysis-layers` | `analysis/layers/` | The layers add-ons share (below) |
| `turbo-chunker` | `analysis/chunker/` | The chunker, and term statistics and BM25 per chunk, section and document |
| `turbo-group` | `analysis/group/` | Sketches, near-duplicate groups and version links inside a document and across a batch, and boilerplate detection (Near duplicates and boilerplate, below) |
| `turbo-vector-index` | `analysis/vector-index/` | An exact in-memory vector index over rows of one embedding space: top-k by dot product or cosine, ties in insertion order, and a typed refusal of vectors from another model, dimension or normalization; TurboQuant storage at 2 to 4 bits a coordinate behind the same queries (feature `turboquant`); and late interaction (MaxSim) over candidates of token vectors from `Vectors<Subwords>` and `Levels` (Exact vector index, below; [turboquant.md](turboquant.md)) |
| `turbo-structure` | `analysis/structure/` | Document structure (`turbo:structure`) from plain text and Markdown (`StructureAnnotator`), and projecting `turbo:structure` into a normalized view (`StructureProjector`, [structure-projection.md](structure-projection.md)) |
| `turbo-abbrev` | `analysis/abbrev/` | Abbreviation and acronym definitions found with the Schwartz and Hearst algorithm, and the later mentions of each short form, as `turbo:abbreviations` and the edges from each short form to its long form; the reader and scoring of Ab3P's MED1250 set (Abbreviations, below) |
| `turbo-matcher` | `analysis/matcher/` | Rule matching after spaCy's `Matcher`, `PhraseMatcher` and `SpanRuler`: token patterns over any token layer, compiled to an automaton simulated without backtracking; phrase lists as a trie over token ids; and the span ruler step, which writes `turbo:matches` under a named instance (Rule matching, below) |
| `turbo-analysis-embed` | `analysis/embed/` | The embedding model as analysis steps: the subwords of a text view from the bundle's tokenizer, and a vector per sentence and per chunk |
| `turbo-sat` | `analysis/sat/` | Sentence boundaries from a SaT model (wtpsplit's Segment any Text) on the core's encoder: a score per token from the bundle's classifier over the encoder's final states, wtpsplit's windows and cuts, each cut moved to a grapheme cluster boundary, written as `opennlp:sentences` ([neural-annotators.md](neural-annotators.md); `analysis/sat/README.md` describes its reference fixtures) |
| `turbo-numeric` | `analysis/numeric/` | The number scanner, which writes `turbo:numbers` ([numbers.md](numbers.md)); the money extractor (`MoneyExtractor`, `MoneyExtractor::for_region`) and `RegionAwareMoneyAnnotator` (reading `turbo:regions`), which write `turbo:money`, the ECB rate table and the conversion step, which writes the `converted` instance of `turbo:money` ([money.md](money.md)); the quantity extractor, which writes `turbo:quantities` ([quantities.md](quantities.md)); the temporal extractor, which writes `turbo:temporal`, the document date selector, which writes `turbo:document-date`, and the conversion step that reads it ([temporal.md](temporal.md)) |
| `turbo-pii` | `analysis/pii/` | PII detection ported from OpenNLP's `opennlp.tools.pii`: e-mail, phone, IBAN, card, ABA routing, network, credential, Bitcoin and Ethereum, national and device identifiers, each checked with its check digit or checksum (Luhn, mod 97, mod 11, ISO 7064, base58check, bech32, EIP-55 over Keccak-256, all written in the crate), in the packs of `PiiPacks`; writes `turbo:pii` with a kind, a confidence and a normalized form per mention; masking in place or as the `masked` view (`MaskedView`), numbered and keyed (HMAC-SHA-256) labels and audit reports ([pii.md](pii.md)) |
| `turbo-grammar` | `analysis/grammar/` | A rule grammar with a chart parser for numbers, ordinals and percentages in words or digits (English), which writes the `grammar` instances of `turbo:numbers` and `turbo:quantities` and the `turbo:ordinals` layer, with its Recognizers-Text conformance tests ([grammar.md](grammar.md)) |
| `turbo-jvm` | `analysis/jvm/` | JVM behaviour the OpenNLP ports must reproduce bit for bit |
| `turbo-goldens` | `analysis/goldens/` | Golden layer files from the JVM: the reader, the writer, the comparison, loading golden layers as test inputs, SHA-256 and SHA-512 |
| `turbo-depparse` | `analysis/depparse/` | Typed dependency graphs, arc-standard parsing/training, UAS/LAS and annotation ([dependency-parser.md](dependency-parser.md)) |
| `turbo-relation` | `analysis/relation/` | Dependency-path relation extraction over entity heads ([relation.md](relation.md)) |
| `turbo-sequence` | `analysis/sequence/` | OpenNLP's beam search, BIO and BILOU codecs and feature generation framework (generators, `StringPattern` and the XML descriptors), with the context generators and validators of the part-of-speech tagger, chunker and name finder, the part-of-speech tag dictionary (`tags.tagdict`) and the Penn Treebank and Universal Dependencies tag mapping. Its crate documentation lists the OpenNLP behaviour it reproduces that is surprising, and the one it does not yet (QN models in a beam search); `analysis/sequence/README.md` describes its JVM golden files |
| `turbo-postag` | `analysis/postag/` | OpenNLP's `POSTaggerME`: `PosModel` opens a `POSModel` file with its descriptor, tag dictionary, beam size and tag format, and `PosTagger` writes `opennlp:pos` (a tag and its probability per token, aligned to `opennlp:tokens`, one beam search per sentence). Tags and probabilities are the JVM's bit for bit; the crate documentation lists the OpenNLP behaviour that is surprising, and `analysis/postag/README.md` describes its goldens and its accuracy on UD English EWT |
| `turbo-lemmatizer` | `analysis/lemmatizer/` | OpenNLP's `LemmatizerME` (`LemmatizerModel`, `MaxentLemmatizer`: a beam search per sentence over the tagged tokens whose outcomes are lemma classes, shortest edit scripts applied to the lowercased token) and `DictionaryLemmatizer` (a `word<tab>tag<tab>lemma` lookup). Both read `opennlp:tokens` and `opennlp:pos` and write `opennlp:lemmas`; the maxent one also writes `turbo:confidence<opennlp:lemmas>`. Lemmas and probabilities are the JVM's bit for bit; the crate documentation lists the OpenNLP behaviour that is surprising, and `analysis/lemmatizer/README.md` describes its goldens |
| `turbo-phrase-chunker` | `analysis/phrase-chunker/` | OpenNLP's `ChunkerME`: `ChunkerModel` opens a `ChunkerModel` file and `PhraseChunker` writes `opennlp:chunk-tags` (a chunk tag and its probability per token, aligned to `opennlp:tokens`, one beam search per sentence over the tokens and their part-of-speech tags) and `opennlp:chunks` (the phrases those tags make, as `chunkAsSpans` returns them). Tags and probabilities are the JVM's bit for bit; the crate documentation lists the OpenNLP behaviour that is surprising, and `analysis/phrase-chunker/README.md` describes its goldens. It is not `turbo-chunker`, which cuts retrieval units |
| `turbo-namefind` | `analysis/namefind/` | OpenNLP's `DictionaryNameFinder`, `RegexNameFinder` and the patterns of `RegexNameFinderFactory`, which write `opennlp:entities` from `opennlp:tokens`, `FilterEntities` to resolve overlapping or nested entities by `FilterRule`, and the merge of several entity layers by `dropOverlappingSpans`. Built-in patterns use hand-written UTF-8 scanners and user patterns in Java syntax are compiled by `JavaRegex` to a direct bytecode matcher or refused with a typed error; the crate documentation lists what is refused and the OpenNLP behaviour that is surprising, and `analysis/namefind/README.md` describes its JVM golden files |
| `turbo-stem` | `analysis/stem/` | OpenNLP's stemmers: `PorterStemmer`, the 21 Snowball stemmers (generated from the Snowball sources OpenNLP's classes come from, over UTF-16 code units as in Java), the light and minimal stemmers, `CachingStemmer`, and `StemAnnotator` (OpenNLP's `StemmerAnnotator`), which writes `turbo:stems` from `opennlp:tokens`. Its crate documentation lists the OpenNLP behaviour that is surprising; `analysis/stem/README.md` describes its JVM golden files and how OpenNLP's Snowball stems differ from current Snowball's |
| `turbo-stem-layers` | `analysis/stem-layers/` | The stemmers' layer, `turbo:stems` |
| `turbo-hunspell` | `analysis/hunspell/` | OpenNLP's Hunspell stemmer, from the `OPENNLP-1927-hunspell-compatibility` branch: a reader of `.aff` and `.dic` files (`HunspellDictionary` with its affix, condition, compound rule and conversion tables), `HunspellStemmer`, and `HunspellAnnotator`, which writes `turbo:stems` from `opennlp:tokens`. No dictionary ships with it; `analysis/hunspell/README.md` lists the licences of the ones the goldens download |
| `turbo-stopword` | `analysis/stopword/` | OpenNLP's eleven bundled stopword lists (Bulgarian, Danish, German, English, Spanish, Finnish, French, Italian, Dutch, Portuguese, Russian) and `StopwordFilter.isStopword`, and the step that reads `opennlp:tokens` and, optionally, `opennlp:language` and writes `turbo:stopwords`, a flag per token; it flags and never removes. `analysis/stopword/data/README.md` gives each list's origin and license, the crate documentation lists the OpenNLP behaviour that is surprising, and `analysis/stopword/README.md` describes its JVM golden files |
| `turbo-stopword-layers` | `analysis/stopword-layers/` | The `turbo:stopwords` layer and the languages that have a list (`StopwordLanguage`) |
| `turbo-glossary` | `analysis/glossary/` | OpenNLP's glossary matchers and readers: `TextMatcher` (`AhoCorasickGlossaryMatcher`, an Aho-Corasick automaton over code points whose matches must start and end on given boundaries), `TokenMatcher` (`TermAnalyzingGlossaryMatcher`, on turbo-matcher's phrase matcher, over token text or stems), their composite (`CompositeGlossaryMatcher`), `GlossaryAnnotator`, which reads a view, its `opennlp:tokens` and optionally `turbo:stems` and writes `turbo:glossary-terms`, and the comma or tab separated and TBX 2 and 3 readers (`CsvGlossaryReader`, `TbxGlossaryReader`, with a small XML parser of its own) and a TBX writer; the character automaton can also be written as a flat double-array image and read in place from a slice or a memory-mapped file (`MappedGlossaryWriter`, `MappedGlossaryMatcher`), with the same matches. The crate documentation lists where it differs from OpenNLP and the OpenNLP behaviour that is surprising; `analysis/glossary/README.md` describes its JVM golden files and the timing of the two matchers |
| `turbo-glossary-layers` | `analysis/glossary-layers/` | The glossary (`Glossary`, an entry's id and term) and the `turbo:glossary-terms` layer, so a consumer of the matches does not depend on the matchers |
| `turbo-doccat` | `analysis/doccat/` | OpenNLP's SVM document categorizer (`DocumentCategorizerSVM` of `opennlp-ml-libsvm`) at prediction: the LIBSVM text model reader and LIBSVM's prediction (C-SVC and nu-SVC, the linear, polynomial, RBF and sigmoid kernels, probability estimates), the sidecar that holds the rest of the model, the feature generators and term weightings, and the step that reads `opennlp:tokens` and writes `turbo:categories` ([doccat.md](doccat.md)); OpenNLP's maximum entropy document categorizer (`DocumentCategorizerME`) with its model file, feature generators and training events, and the steps that write `turbo:categories` for the document and `turbo:categories<L>` for each sentence, structure block or chunk; and OpenNLP's sentiment detector (`SentimentME` with `SentimentFactory` or `EmojiSentimentFactory`): the `SentimentModel` file, the token and emoji contexts, and the step that reads `opennlp:tokens` and writes the `sentiment` instance of `turbo:categories` ([sentiment.md](sentiment.md)) |
| `turbo-langspan` | `analysis/langspan/` | Language by span (code switching): script runs, then OpenNLP's language detector model scoring words and clauses, written as `turbo:language-spans` with the document summary in `opennlp:language` ([langdetect.md](langdetect.md), "Language by span") |
| `turbo-bpe-trainer` | `analysis/bpe-trainer/` | BPE trainers, a tool and not an annotator: byte-level BPE as upstream `tokenizers` trains it, SentencePiece BPE as the sentencepiece library trains it, and OpenNLP's `BPETokenizerTrainer` with its `BPETokenizer` ([bpe-trainer.md](bpe-trainer.md)) |
| `turbo-train` | `analysis/train/` | Training OpenNLP maxent models, the event side: `Event`, the `FileEventStream` and `RealValueFileEventStream` formats, the CRC-32C event checksum, the three data indexers with their cutoffs, sorting and merging, `TrainingParameters` with typed values and OpenNLP's parameters file, `AlgorithmType` and the trainer types, the progress monitors and stop criteria, the GIS trainer (`GISTrainer`, parameters bit for bit the JVM's after every iteration), the liblinear trainer of the OpenNLP addons with liblinear-java's logistic regression solvers ([liblinear.md](liblinear.md)), the binary model writer (`GenericModelWriter`, bytes the JVM's) and the `.bin` packaging with its manifest (`BaseModel.serialize`; the entries are the JVM's bytes, the archive around them is written here and loads in both). Sample types belong to the formats crate; this one starts at events. Its crate documentation lists the OpenNLP behaviour that is surprising, and "Training goldens" below describes its JVM reference |
| `turbo-train-components` | `analysis/train-components/` | Training the sentence detector, the tokenizer, the part-of-speech tagger, the lemmatizer, the phrase chunker, the language detector, the sentiment detector (`SentimentEventStream` and the `SentimentModel` file, [sentiment.md](sentiment.md)) and the chunking and tree-insertion constituency parsers (their `ParserEventStream`s, `Parser.train`, the `ParserModel` file, `ParserEvaluator` and `ParserCrossValidator`): `SDEventStream`, `TokSpanEventStream`, `POSSampleEventStream`, `LemmaSampleEventStream`, `ChunkerEventStream` and `LanguageDetectorEventStream` with their factories (end characters, alphanumeric pattern, abbreviation dictionary, feature generator descriptor, tag dictionary), the `SentenceModel`, `TokenizerModel`, `POSModel`, `LemmatizerModel`, `ChunkerModel` and `LanguageDetectorModel` files with every manifest entry and artifact the JVM writes (the dictionaries as `Dictionary.serialize` and `POSDictionary.serialize` write them, zip entries in the JVM's order), and the recipes that train the eighteen published UD 1.3 models again from their treebanks and compare them with the published files. Its own crate because `turbo-formats` depends on `turbo-tokenize` and `turbo-parser`; "Training the components" below |
| `turbo-train-recipes` | `analysis/train-recipes/` | Recipes that train models on permissive data and gate them: MASC 3.0.0 (CC BY 3.0 US) with the record of its release and fixed split, the English sentence detector, tokenizer, part-of-speech tagger and chunker trained on it, their gates against the OpenNLP English UD models and OpenNLP 1.5's Penn tagger and chunker (`en-pos-maxent.bin`, `en-chunker.bin`); the CC BY 4.0 UD treebanks at release 2.16 (Spanish and Catalan AnCora, Portuguese Porttinari, Finnish FTB, Italian MarkIT) with their file digests and licence check, the sentence detectors, tokenizers, taggers and lemmatizers trained on them and their gates against the OpenNLP UD 1.3 models; the language detector trained on Tatoeba's CC BY 2.0 FR sentences and gated against `langdetect-183.bin`; the web language detector trained on those sentences and on FineWeb-2 and FineWeb (ODC-By 1.0) windows read by range request, gated on held-out sentences and held-out windows; the model cards and the permissive zoo entries, and the `turbo-train-recipes` binary ("Models trained on permissive data", below) |
| `turbo-formats` | `analysis/formats/` | OpenNLP's training samples (name, part-of-speech, lemma, chunk, token, sentence, document category and language samples), a reader per native format that reproduces OpenNLP's sample stream over the same text, the conversion of name and part-of-speech samples into token and sentence samples, the CoNLL-U readers of part-of-speech, lemma, sentence and token samples, the CoNLL-2002 and CoNLL-2003 name readers and the CoNLL-X part-of-speech reader, the MASC corpus reader and its part-of-speech, token, sentence and name samples, the named entity corpus readers (GermEval 2014, BioNLP 2004, Evalita, the 1990 census names, MUC-6 and brat, including discontinuous entity standoff round-tripping), the sentence corpus readers (Moses, LetsMT, NKJP, the Irish Sentence Bank, which also gives token samples), the Leipzig language corpus reader and the 20 Newsgroups reader, the Arvores Deitadas readers of the Floresta Sinta(c)tica treebank (name, part-of-speech, chunk, sentence and token samples), the OntoNotes name, parse and part-of-speech readers and the French Treebank parse reader (`turbo_parser::Parse` trees), the format registry, and the evaluators and cross-validation folds (`FMeasure`, `Mean`, an evaluator per component). Every token and tag keeps its byte span in the source; the crate documentation lists the OpenNLP behaviour that is surprising, and its tests compare with the `formats`, `masc`, `ner`, `corpora`, `ontonotes`, `frenchtreebank` and `evaluate` goldens |
| `turbo-parser` | `analysis/parser/` | OpenNLP's constituency `Parse` tree and Penn Treebank bracket I/O (`Parse::parse`, `Parse::show`, `ParseSampleStream`, `ParserPosSample`, `ParserChunkSample`, `space_untokenized_parens`), the `HeadRules` and `GapLabeler` traits, `EnglishHeadRules` (`opennlp.tools.parser.lang.en.HeadRules`), and `AncoraSpanishHeadRules` (`opennlp.tools.parser.lang.es.AncoraSpanishHeadRules`), whose tags are Java regular expressions (`TagPattern`); and the `CHUNKING` and `TREEINSERT` bottom-up parsers: reading a `ParserModel` file, the context generators, the shared search of `AbstractBottomUpParser`, `ParserTool.parseLine` and `ParserAnnotator` (the `Phrases` layer). Training the parsers is `turbo-train-components`. `analysis/parser/README.md` describes where it differs from OpenNLP and its JVM golden files |
| `turbo-coref` | `analysis/coref/` | OpenNLP's coreference resolver: mention detection from entities, noun phrases (chunks or parse phrases) and pronouns, the precision-ordered sieves (speaker, exact, relaxed, acronym, person name, the strict and relaxed head sieves, pronoun match), the pair classifier and mention ranker read from a `CorefModel` file, word vectors for head similarity (or trained from a corpus by the skip-gram and CBOW `Word2VecTrainer`) and token vectors for span features, `CorefAnnotator` writing `opennlp:mentions` (`Mentions`), the chains as `Groups<Mentions>` and the antecedent links as `Edges<Mentions, Mentions>` tagged with the deciding sieve, the pairwise and ranking trainers, and `CorefScorer` (MUC, B-cubed, CEAF-m, CEAF-e, mention detection and the CoNLL average); [coref.md](coref.md) |
| `turbo-encoding` | `analysis/encoding/` | Bytes to a document: decoders for the 40 encodings of the WHATWG Encoding Standard, each recording an alignment from the bytes to the text; encoding detection (a port of chardetng); `open`, which settles the encoding and writes `turbo:encoding`; and mojibake repair (a port of ftfy's encoding fix, by Robyn Speer), one edit per fix. `analysis/encoding/README.md` lists the generated files and their sources |
| `turbo-encoding-layers` | `analysis/encoding-layers/` | The `Encoding` enum with the standard's labels, and the `turbo:encoding` layer (`SourceEncoding`) |
| `turbo-encoding-gen` | `analysis/encoding-gen/` | The generator of the WHATWG tables, and `ftfy.py`, which generates the repair's character classes and test fixture from ftfy |
| `turbo-geo` | `analysis/geo/` | The gazetteer, geocoding, region-ballot, containment and place-profile APIs of the OpenNLP addons `geo` module (`Gazetteer`, `GazetteerEntry`, `GeoPoint`, `GeoBoundingBox`, `AttributeValue`, `Geocoder`, `GeoResolution`, `PopulationPriorGeocoder`, `SpatialCoherenceGeocoder`, `MentionPolicy`, `GeocodeAnnotator` writing `turbo:locations`, `DocumentRegionAnnotator` writing `turbo:regions`, `PlaceHierarchy`, `ContainmentSpine` and `HierarchyAnnotator` writing `turbo:containment`, and `PlaceProfiles` / `PlaceProfile`) and its gazetteers: the bundled Natural Earth table (public domain, embedded), GeoNames and Overture tables the caller supplies, user tables with suppression rules, records built in code, and the overlay of additions and suppressions on a base. All share one columnar index with OpenNLP's match key (UAX #29 words, NFC, `toLowerCase(Locale.ROOT)`, the accent fold) and its population ranking; large tables load on several threads and save to a snapshot that reads back without folding again. A gazetteer's match keys can also be written as one mapped automaton image (`name_image`), which `NameFinder` reads in place to find the gazetteer's names in running text and `NameAnnotator` writes as location entities for the geocoder and the linker. The crate documentation lists the OpenNLP behaviour that is surprising, `analysis/geo/data/README.md` the bundled table's origin, and "Gazetteer goldens" in `analysis/goldens-tool/README.md` its JVM golden files |
| `turbo-entity-linker` | `analysis/entity-linker/` | Generic entity disambiguation and linking (`EntityLinker`, `LinkerConfig`, `LinkerWeights`) against pluggable `KbProvider` backends (`CatalogProvider` over `KnowledgeBase` CSV/TSV catalogs and binary snapshots with optional `VectorIndex` nearest-neighbor retrieval, `GazetteerProvider` over `turbo-geo` gazetteers, `WordNetProvider` over `turbo-wordnet` synsets, and `GlossaryProvider` over `turbo-glossary-layers` glossaries), scoring alias and popularity priors, NER entity-type compatibility, local and document context overlap, and document-level coherence with coreference cluster propagation (`Groups<Mentions>`, `Groups<Entities>`), writing `LinkedEntities` (`turbo:linked_entities`) aligned to `Entities` with explicit `NilReason` and `nil_cluster` assignments |
| `turbo-eval` | `analysis/eval/` | The evaluation harness: the evaluation sets with their fetch by checksum and typed readers, the pipeline, the baseline chunkers, measures, significance tests, records, the gate and the `turbo-eval` binary (Evaluation, below) |
| `turbo-zstd` | `analysis/zstd/` | A Zstandard decoder after RFC 8878: frames one after another, skippable frames, raw, RLE and compressed blocks, every kind of literals and sequences section, and the content checksum (XXH64); frames that need a dictionary or a window over 128 MiB are refused. The Parquet reader of `turbo-eval` uses it for ZSTD pages |

They need nothing beyond the standard library and each other, except
`turbo-analysis-embed`, which runs the embedding model and so depends on
the core crate, `turbo`; `turbo-sat`, which runs a SaT model through it
and the core; and `turbo-eval`, which uses serde for its
records and set files, the core for turbo-bench's provenance types,
`turbo-analysis-embed` for the embedder, `ureq` to fetch the sets, and
`flate2` (already in the build through `ureq`) for the one set published
as a gzip tar; and `turbo-train-recipes`, which uses `turbo-eval`'s gate
and serde for its model cards. A shared
layer lives in `turbo-analysis-layers`, so a consumer of a layer never
depends on the add-on that produces it; an add-on's own layers live in a
small schema crate beside it, for the same reason.

## JVM parity

Some of OpenNLP's results depend on the Java platform rather than on
OpenNLP, and `turbo-jvm` reproduces those parts exactly, so a port can
match the JVM bit for bit:

- `chars`: Java sees text as UTF-16 code units. `Utf16Units` yields the
  units of a `&str` with the UTF-8 byte offset of each unit's character,
  for context generators, `StringPattern` and character n-grams;
  `utf16_len` is `String.length()`; `compare_to` is `String.compareTo`,
  which orders by code unit and so differs from `str::cmp` for
  characters outside the Basic Multilingual Plane; `is_java_whitespace`
  is `Character.isWhitespace`.
- `random`: `JavaRandom` is `java.util.Random` from a seed, and its
  `shuffle` is `Collections.shuffle(List, Random)`, as OpenNLP's readers,
  cross-validation and trainers use them.
- `strict_math`: `exp` and `log` port fdlibm, which `StrictMath` computes
  and which the maxent, perceptron and naive Bayes models call.
- `hash_map`: `JavaHash` (`String.hashCode`, `CharBuffer.hashCode`,
  record/pair and `List.hashCode`), `spread` (`HashMap.hash`),
  `HashMapOrder`, `put_order` and `JavaHashMap` reproduce JDK 21's
  `HashMap` table sizing, bin splitting, red-black tree bins and
  iteration order for `new HashMap<>()`, `new HashMap<>(n)`, `put`,
  `merge`, `putAll` and `clear`, used by `turbo-langdetect`,
  `turbo-namefind`, `turbo-sequence`, `turbo-bpe-trainer` and
  `turbo-formats`.

It is checked against vectors a JDK 21 wrote; `analysis/jvm/README.md`
says how to regenerate them.

## Offsets

Every offset is a UTF-8 byte offset into one text view, half-open, as a
`u32`: the unit the tokenizer's spans already use (docs/tokenizer.md).
Both ends of a span must be on character boundaries: a layer with a
span that cuts a character is refused when it is added, not found later
by a panic.
UTF-16 and code point offsets, for Java and the wire, are conversions
at the edge, never a second coordinate system inside. Each view has an
`OffsetTable` (`View::offsets`, `Document::offsets`), built the first
time it is asked for and kept: a checkpoint every 256 bytes, stored by
column, that holds the byte, UTF-16 and code point offsets of the first
character boundary there. A conversion is a binary search over the
checkpoints and a forward count over at most one stride of the text; a
view that is all ASCII keeps no checkpoints, since all three units
agree. A UTF-16 offset between the two halves of a surrogate pair, or
any offset past the end of the view, is refused with `Error::Offset`.

`Utf16Offset` and `CodePointOffset` are their own types, so neither can
be passed where a byte offset is expected. `OffsetCursor` converts a
column of offsets in text order (a layer's starts, or its ends) in one
forward pass over the text, and `OffsetTable::column_to_utf16` and
`column_to_code_point` use it to fill a vector the caller provides.

## Views

The source text never changes. A step that changes text (Unicode
normalization, dehyphenation) writes a new `View`, with an `Alignment`
to its parent: the edits, in text order on both sides. Outside the edits
the two texts are the same byte for byte. Mapping an offset either way
is a binary search over the edits. An offset inside an edit widens to
the whole edit (a start to its start, an end to its end), so a mapped
span covers everything the original covered.

A normalizer writes its text through a `ViewBuilder`, which records the
alignment as it goes, so no view exists without one. Each `replace` call
is one edit, never merged with the edit before it, so the caller chooses
the granularity: three characters replaced one call each keep a
one-to-one mapping per character, and a run replaced in one call is one
edit over the whole run. A layer's spans
name the view they are in, and `Document::to_source` maps them back
through any chain of views to exact source bytes.

turbo-pii's `MaskedView` writes the view named `masked` this way: the
text with the mentions of the `turbo:pii` layer masked, the same byte
length as its parent, and one edit per masked character, so each mask
byte maps back to its own source character and every other offset maps
to itself ([pii.md](pii.md#the-masked-view)).

Inserted or deleted bytes (an edit with one empty side) belong to
neither neighbour: a start offset at them maps to after them and an end
offset to before them. An empty span at them stays empty and lands
before them, at the smaller of the two mapped offsets, so mapping any
span either way never builds a span that ends before it starts. Deleting
the whole text, for example, maps the empty view span at 0 to the empty
source span at 0. An empty span strictly inside an edit widens to the
whole edit like any other span. `Document::offset_from_source` and
`Document::offset_to_source` map one offset through the chain of views
for the side given, so a start at an insertion point goes after every
byte inserted there and an end goes before them.

### Source bytes

View 0 is text, but a file is bytes, and in any encoding but UTF-8 the
two disagree: "é" is one byte in windows-1252 and two in the text, a
Shift_JIS character is two bytes and three in the text, a UTF-16
surrogate pair is four bytes and four, and a byte order mark is bytes
with no text at all. A document opened from bytes
(`Document::decoded`, which `turbo_encoding::open` calls) keeps them
below view 0 as `SourceBytes`: the bytes and an `Alignment` whose parent
side is offsets in the bytes and whose child side is offsets in view 0.
It is the same `Alignment` every view has, so the mapping rules above
hold for it unchanged, and the byte view is one more step at the bottom
of the chain:

```text
bytes  --decoder alignment-->  view 0  --alignment-->  view 1  --> ...
```

`Document::to_source_bytes` and `offset_to_source_bytes` map a span or an
offset of any view through `to_source` (or `offset_to_source`) to view 0
and then through the decoder's alignment to the bytes;
`from_source_bytes` and `offset_from_source_bytes` go the other way. A
document made from text (`Document::new`) has no byte view: its bytes
are the UTF-8 of view 0, and the byte functions return the same numbers
as the text ones. Offsets stay `u32` on both sides; bytes or a decoded
text longer than `u32::MAX` are refused with `Error::Source` before any
offset is computed.

A decoder writes view 0 through a `DecodeBuilder`, the byte-level twin
of `ViewBuilder`, so no decoded text exists without the alignment that
produced it:

- Bytes that are already the text's UTF-8 (ASCII in an ASCII-compatible
  encoding, a valid stretch of a UTF-8 file) are kept, with no edit.
- Every other decoded character is one edit over its bytes: the smallest
  unit an offset maps to, so a span that cuts a character's bytes widens
  to the whole character. Four Big5 codes decode to two characters, which
  share one edit. An edit takes 16 bytes, so mostly non-ASCII text in a
  legacy encoding, or a UTF-16 file, carries about 16 bytes of alignment
  per character.
- A byte order mark and an ISO-2022-JP escape sequence are deletions.
- A U+FFFD is one edit over exactly the bytes it replaced: the decoders
  follow the WHATWG algorithms, which say which bytes an error consumes
  and which are read again, so a lead byte followed by ASCII is a U+FFFD
  for the lead byte alone, and a multi-byte sequence cut off at the end
  of the input is one U+FFFD for the bytes that are there.

`SourceBytes::new` checks the alignment against the bytes and the text
(every edit in bounds, in order and on character boundaries, every
unchanged stretch the same bytes on both sides), so a byte view cannot
place text at bytes it did not come from outside an edit. The bytes
inside an edit are not decoded again: that they encode the edit's text is
the decoder's guarantee.

The encoding a document was read with is the document layer
`turbo:encoding` (`SourceEncoding`: the encoding, a confidence in
[0, 1], and the evidence: a byte order mark, a declared label, valid
UTF-8, ASCII, or the detector's guess). The confidence of a detected
encoding is a heuristic from how far the guess's score is ahead of the
next encoding's, not a calibrated probability.

Mojibake repair (`turbo_normalize::Mojibake`, docs/normalizers.md) is an
ordinary view above view 0. A file in UTF-8 that was opened as
windows-1252 has "cafÃ©" in view 0; the repair writes "café" in its view
with one edit over "Ã©", and the "é" maps through both alignments to the
two bytes of its UTF-8 in the file.

### Markup

The analysis crates parse no HTML or XML. An extractor that does hands
over a `MarkupText` (`turbo_analysis::markup`): the text it produced, a
run per stretch of that text with the span of view 0 it came from, and
the spans of the tags it dropped. A run is copied (character data,
unchanged bytes), decoded (an entity such as `&amp;` standing for "&",
one edit over the whole reference) or inserted (a line break the
extractor adds for a block element, an edit with an empty source).
`MarkupText::check` refuses runs that leave gaps, overlap, run out of
bounds, cut a character or claim to be copied when they are not;
`MarkupText::view` turns the runs into a view whose alignment deletes
everything between them. A word found in the extracted text then maps
through that view, view 0 and the byte view to the file's bytes, and
the tags stay available as spans of view 0. Lucene's
`HTMLStripCharFilter` keeps the same information as offset corrections;
an `Alignment` maps both ways and composes with the other views.

## Layers

A layer implements `Layer`: an id, a schema version, what its items are
anchored to, how many there are, and a check of its own invariants. Its
data is stored by column (a `Vec` of starts, a `Vec` of tags), never as
an object per item. An item is anchored one of three ways:

- `Text`: each item carries a span of one view (sentences, tokens).
- `Aligned`: item i describes item i of another layer (a tag per
  token, a vector per sentence). The document checks the lengths match.
- `Document`: the layer describes the whole document (its language).

`Ix<L>` is an index into layer `L` and no other: an `Ix<Tokens>` used
as an index into sentences does not compile. It is stored as i + 1 in a
`NonZeroU32`, so an index that may be absent (a dependency head that may
be the root) is still four bytes.

Model probabilities and scores (`PosTags.prob`, `Entities.score`,
`Confidence<L>`, `Language.score`) are stored as f64. OpenNLP computes
them as Java doubles, and the parity tests compare them with the JVM's
output to within 1e-9, which f32 (about seven significant digits) cannot
hold. Vectors (`Vectors<L>`) are f32: they are computed in Rust and
never compared with OpenNLP. The chunker's costs (`Boundaries.cost`,
`Chunks.cost`) are integers, so every backend picks the same cuts.

A layer is added once and never changed. Before it is taken the
document checks that the layer is not already there, that no other
layer type has its id, the layer's own invariants, that its spans are
in bounds and on character boundaries, that an aligned layer is as long
as its anchor, and that every index into another layer (a token's
sentence, an entity's tokens) is inside it. A document clones
cheaply, since its layers are shared, so branches of a pipeline can
grow copies and merge them; a merge where both branches wrote the same
layer, or both made a view, is refused and changes nothing.

### Overlapping, nested and discontinuous spans

A positional layer (`Anchor::Text`) stores its spans in `Spans<L>` and
declares what overlap its items may have through `Layer::OVERLAP`
(`Overlap` in `turbo-analysis`), which `Document::add` checks on
`Layer::text_spans()` (`TextSpans`):

- `Overlap::None` (the default, used by `Sentences`, `Numbers`, `Money`,
  `Quantities` and the normalizer layers): continuous spans in text
  order (`start` non-decreasing) that do not overlap
  (`end[i - 1] <= start[i]`).
- `Overlap::Nesting` (`Structure`): continuous spans and individual
  discontinuous fragments never partially cross (`a.start < b.start &&
  b.start < a.end && a.end < b.end`), and whenever two items share
  covered bytes, every fragment of the inner item lies inside a fragment
  of the outer item. A separate item may lie inside the uncovered gap
  between fragments of a discontinuous item (so their bounding envelopes
  may overlap or cross while their covered fragments remain disjoint).
  The check does not need text order: unordered spans are sorted first.
- `Overlap::Any` (`Tokens`, `Subwords`, `PhraseChunks`, `Chunks`,
  `Levels`, `Entities`, or multi-tagger candidate layers): any spans in
  bounds. `Tokens` and `PhraseChunks` are here because two UTF-16 tokens
  that split a character outside the Basic Multilingual Plane are both
  widened to it; `Chunks` and `Levels` because a level 0 chunk can run
  across the edge of a nested section; `Entities` for `"New York"`
  inside `"Bank of New York"`, or overlapping matches from a
  multi-pattern `RegexNameFinder` before `MergeEntities` or
  `FilterEntities` resolves them. Each layer's own `check` adds what
  it promises beyond that (`Tokens` and `Subwords` are in text order,
  `Chunks` lie inside their parents).

For discontinuous spans (such as split verbs `"take ... off"`, clinical
mentions `"pain in the left ... and right knee"`, or brat standoff
`T1\tEntity 0 4;10 13\t...`), `Spans::push_fragments`
(`Spans::try_push_fragments`) stores the item's bounding envelope
(`first.start..last.end`) in `start` and `end` and populates three flat
columns (`frag_start: Vec<u32>`, `frag_end: Vec<u32>`,
`frag_item_end: Vec<u32>`). Layers whose items are all continuous keep
those three vectors empty and pay no allocation overhead. When a layer
holds any discontinuous item, `TextSpans` exposes the fragment columns
alongside the envelope columns so `Document::add` and
`turbo_goldens::invariants` check every fragment's bounds, character
boundaries, UTF-16 and code point round-trips, and view-to-source
alignment. Projecting a discontinuous item across views maps each
fragment individually (`spans.fragments(i).map(|f| doc.to_source(view, f))`)
rather than mapping the bounding envelope alone, because an edit inside
the gap between fragments would otherwise shift or widen the gap.

Retrieval chunking (`turbo-chunker`) cuts only at sentence boundaries
and never reads `Entities` as a split constraint: a named entity,
continuous or discontinuous, stays inside its sentence, and
`Spans::overlapping` and `Spans::covering` find the entities whose
fragments touch a chunk.

Entities are a cost, not a constraint. `turbo_chunker::EntityBoundaries`
keys the entities of `opennlp:entities` (normalized surface: NFKC, full
case folding, whitespace collapsed, edge punctuation dropped; merged by
`Groups<Entities>` or `Groups<Mentions>` chains when present, whose
pronoun and nominal mentions then count as mentions of the entity) and
writes the `entity` instance of `turbo:boundaries`: each entity whose
run of mentions (two mentions at most `max_gap` sentences apart) spans
a gap adds `per_entity` to it, up to `cap`. `DistilledBoundaryScorer`
sums that instance into the default one at `entity_runs` (`Weight::ONE`
by default); a pipeline that does not write it is unchanged, hashes
included. `ChunkEntityHeaders` writes `ChunkEntities`
(`turbo:chunk-entities`, aligned to `Chunks`): per chunk the entity
keys it mentions and the ones carried from earlier in its section, each
key named by the index of its first entity, never a copied string;
`ChunkEntities::header` projects them onto the text.

Query-time routing (`turbo_chunker::LevelRouter`) ranks the chunk,
section and document units of `Levels` in one list, fusing BM25 over
`TermStats<Levels>` with dense cosine scores over `Vectors<Levels>`
when present. A query profile (chunk-level term rarity and length)
weights the levels (factoid toward chunks, topical toward sections,
summary toward the document), sibling chunks that co-activate promote
their parent section, and hits carry their parent context with exact
view spans. `ContentHashes<L>` (`turbo:content-hashes<L>`) gives every
unit a deterministic content ID over its level, heading path and
fragment text, so a re-chunked document diffs cleanly.

The retrieval cascade (`turbo_chunker::Cascade`) spends the expensive
space only where it helps: static vectors and BM25 each admit a
shortlist of level units, a contextual `Vectors<Levels>` space
reranks just that pool, and structured `UnitFilter`s (built from any
span layer's overlap with the units) then require or boost. Every
query vector carries its space's `ModelRef`, and building a cascade
refuses a query scored against another space's rows (or the same
space twice). Hits are deterministic, name the admitting stage and
each signal's score, carry the unit's exact fragments, and the
result reports how many units the contextual stage actually scored;
`recall_at_k` records the recall side of a candidate-count sweep.
Units the contextual stage cannot score (empty rows) rank strictly
below every unit it did score, so a max-normalized stage-1 score is
never compared against a contextual cosine. Measured on the
synthetic pipeline fixture in `analysis/chunker/tests/cascade/main.rs`
(721 level units, 512-wide contextual rows, release, best of
alternating rounds): the cascade at 64 candidates scores 64 units
contextually in about 70 microseconds per query against about 342
microseconds for a direct contextual-only scan of all 721 units,
and recall@2 against the contextual stage's own top-2 is monotone
in the candidate count (1, 2, 3, 5, 64), reaching 1.0 at a full
pool. The external harness sets named for this work (GutenQA,
ConTEB, HiCBench) are not vendored in this repository, so no
numbers are claimed on them; recording them is the same sweep
(`recall_at_k` per candidate count against contextual-only gold,
plus the direct-scan latency baseline) once the sets are wired in.

Multi-representation indexing (`turbo_chunker::MultiRepChunkIndex`)
registers several indexable views of one chunk unit (its surface
text, a summary or header, its keyphrases from `TermStats<Levels>`,
its sentence sub-spans), each carrying provenance back to the
canonical unit and its `ChunkId`, and ranks them lexically in one
list. Parent expansion (`turbo_chunker::ParentExpander`) then
promotes ranked leaf or representation hits to the parent unit that
should enter the context ("small-to-big" retrieval): promotion
climbs the `Levels` tree toward a target level while the parent
fits `max_parent_tokens`, scores of hits resolving to one parent
combine by max, sum, mean or reciprocal rank fusion with
deterministic tie-breaking, an accepted unit blocks its ancestors
and its whole subtree so a parent and its own child are never both
emitted, and `max_total_tokens` caps the emitted context.
`unit_tokens` derives every unit's token count from the chunk
rows' `model_tokens`.

Late pooling (`turbo_chunker::LateChunkAnnotator<L>`) pools the
already-contextualized `Vectors<Subwords>` into `Vectors<L>` for any
span layer `L` with mean, max or masked-mean pooling (masked mean
drops punctuation pieces and configured stopwords first) and optional
L2 normalization, without re-running the encoder per item. It reads
`TokenVectorWindows` (`turbo:token-vector-windows`), the encoder
window spans the token vectors were produced under, and refuses
vectors whose model differs from the subwords' model or pieces no
window holds; an item straddling a window seam is refused, split and
recombined by piece count, or pooled across the seam, according to
its `WindowPolicy` (`Reject` by default). Items selecting no piece
get an empty vector row, and discontinuous items pool only their
fragments. Because the output is the ordinary `Vectors<L>` layer,
bind the step at a named instance to keep late and naive embeddings
side by side on one document.

`Spans<L>` provides lookup and conflict-resolution helpers that work
across continuous, nested, overlapping and discontinuous spans:

- `Spans::covering(offset)` returns all `Ix<L>` covering byte `offset`
  (for a discontinuous item, only when one of its fragments covers
  `offset`, never the gap between fragments).
- `Spans::overlapping(span)` returns all `Ix<L>` overlapping `span`
  (again testing each fragment of a discontinuous item).
- `Spans::in_range(outer)` returns all `Ix<L>` whose bounding span lies
  inside `outer`, narrowed by binary search on both ends when `start` is
  ordered.
- `Spans::filter(scores, rule)` (`filter_spans`), `Entities::filter` and
  `FilterEntities` resolve overlapping or nested items in $O(n \log n)$
  time to a non-overlapping subset in text order (`!a.overlaps(b)`)
  under `FilterRule::LongestFirst` (matching spaCy's
  `spacy.util.filter_spans`), `FilterRule::HighestScore` or
  `FilterRule::LeftmostLongest`.

### Edges and groups

Two generic layers in `turbo-analysis` represent graph structure over any
layers on the document:

- `Edges<From, To>` (`turbo:edges<From, To>`) holds directed, labeled,
  scored edges between items of `From` and items of `To` (`From` and `To`
  may be the same layer type or two different ones). Its rows are stored
  by column: `from: Vec<Ix<From>>`, `to: Vec<Ix<To>>`, `kind: Vec<Tag>`
  (interned against `kinds: Arc<TagSet>`), and `score: Vec<f64>`, alongside
  a secondary row-index column `by_to: Vec<Ix<Edges<From, To>>>`.
  - **Sort order**: the primary columns (`from`, `to`, `kind`, `score`) are
    stored in strictly increasing `(from, to, kind)` order (`Edges::new` and
    `Edges::between` sort by `(from, to, kind)`), so a triple appears at
    most once; two rows for one triple would be one relation with two
    scores, and a layer holding them is refused when it is added. The secondary
    index `by_to` holds a permutation of `0..len` in strictly increasing
    `(to[r], r)` order. Lookups by source (`from_source`, `outgoing`) are
    two binary searches over `from` returning a contiguous `IxRange`, and
    lookups by target (`to_target`, `incoming`) are two binary searches
    over `by_to` returning a contiguous slice of row indices in `(from,
    kind)` order. `Layer::check` verifies column lengths, tag bounds,
    strictly increasing `(from, to, kind)` order, and strictly increasing
    `(to[r], r)` order across `by_to` (which, with every entry below the
    row count, makes it a permutation).
  - **Naming the two layer instances**: `Edges<From, To>` carries
    `from_key: Key<From>` and `to_key: Key<To>`. `Edges::new` sets both to
    `Key::DEFAULT`, so when the layer is added to a document each endpoint
    resolves through provenance (`on`, then `reads`, then the default
    instance) just like any other cross-layer index; when `From` and `To`
    are one type, both ends resolve to the same instance. `Edges::between`
    takes explicit `Key<From>` and `Key<To>` (including `Key::named`), so an
    edge layer can link two named instances (`opennlp:entities@gold` to
    `opennlp:tokens@predicted`, or two named instances of the same layer
    type such as `opennlp:tokens@a` to `opennlp:tokens@b`). A named
    instance must be one the producing step read or wrote, listed in its
    provenance's `reads` or `on`; `Document::add` refuses any other, so the
    analyzer orders the step after that instance and the provenance names
    every instance the edges point into. `Layer::indexes` reports both keys
    with one past the largest source and target index, so `Document::add`
    checks both bounds against their respective instances and records each
    instance once in `Document::made_on`. A step that reads the edges
    beside a different instance of an endpoint layer fails with
    `Error::Unaligned`, as for any other layer made on an instance.
- `Groups<L>` (`turbo:groups<L>`) holds sets of `Ix<L>` (coreference
  chains, multi-argument clusters, duplicate groups) in two flat columns:
  `members: Vec<Ix<L>>` and `end: Vec<u32>`, with `of_key: Key<L>` naming
  the target instance (`Groups::new` for `Key::DEFAULT`, `Groups::of` for a
  named key). Group `g` is the slice `members[start..end[g]]`, kept in
  strictly increasing `Ix<L>` order with no duplicates, so membership
  (`Groups::contains`) is a binary search over the group's slice and no
  group allocates its own `Vec`. Every group has at least one member
  (a singleton is a valid chain); a layer with an empty group is refused.
  A named `of_key` follows the same rule as an edge endpoint.

A generic layer's name holds the full names of the layers it is over,
so nested generic layers stay distinct:
`Groups<Edges<Entities, Tokens>>` is
`turbo:groups<turbo:edges<opennlp:entities, opennlp:tokens>>`, not the
same name as `Groups<Edges<Tokens, Tokens>>`.

### Alternatives, lattices and n-best sequences

Two generic layers in `turbo-analysis` hold lattice alternatives and
top-k sequence hypotheses without changing the flat column layout of
ordinary single-path layers:

- `Alternatives<L>` (`turbo:alternatives<L>`, anchored to `Text`) holds a
  position-indexed word lattice for any layer type `L` (most commonly
  `Alternatives<Tokens>`).
  - **Positions and position length**: Positions are integer lattice nodes
    `0..=num_positions` (for an empty lattice, `num_positions == 0`). Each
    arc `i` has a byte `Span` in the view, a starting `position[i]`, a
    `position_length[i] >= 1` so it lands at `position[i] +
    position_length[i]`, a `mode` bitset (`ModeSet` over
    `SplitMode::{Best, A, B, C}`), an optional `score` (`f64`, finite), and
    an optional `form` (the normalized or expanded surface string when a
    synonym or decompounded arc differs from its source slice). A producer
    pushes rows in non-decreasing `position` order (`push`, `push_full`
    append as given, and `Layer::check` refuses any other order), so
    `at_position(p)` finds all arcs leaving node `p` by binary search.
  - **Coexistence with a flat layer**: An annotator that produces a
    lattice (such as a morphological analyzer with Sudachi/Kuromoji A/B/C
    split modes, or a compound splitter) can write the single chosen
    segmentation (`Best`, or a configured split mode) to
    `Key::<Tokens>::DEFAULT` and the full lattice to
    `Key::<Alternatives<Tokens>>::DEFAULT` on the same document. Readers
    that want one path keep reading `Tokens` directly, or call
    `Alternatives::best_path`, `Alternatives::path_for_mode(SplitMode::A)`,
    or `Tokens::from_alternatives(&alt, mode)` to materialize a linear
    `Tokens` layer from any mode without re-running the tokenizer; a mode
    with several complete paths gives the one with the highest sum of
    scores. Indexers and query builders that want every decomposition
    walk `Alternatives::at_position`; `all_paths` enumerates a sentence's
    paths and grows exponentially with the lattice's length.
  - **Byte-tiling and reachability invariant (`Layer::check`)**: Every arc
    must lie on a path from position `0` to `num_positions`; every
    position `0..=num_positions` must be visited; every position `p` has
    one start byte `start_at[p]` shared by every arc leaving it and one
    end byte `end_at[p]` shared by every arc entering it, with
    `end_at[p] <= start_at[p]` at each internal position (the gap between
    two words) and `start_at[p] <= end_at[p + 1]` over each unit position.
    So every path from `0` to `num_positions` covers the same unit
    position bytes; an arc over several positions also covers the gaps
    between them.
  - **What fits**: a reading where each position starts and ends at one
    byte, such as split modes, compounds beside their parts, and a
    contraction beside the words it stands for (`"ny"` over `"new york"`,
    the synonym on the longer arc). A Lucene synonym expansion that gives
    every output token the input's offsets (`"nyc"` as `"new"` plus
    `"york"`, all at bytes `0..3`) does not fit as such: the second word
    takes an empty span at the end of the first (`"york"` as `3..3` with
    `form: Some("york")`), and `Tokens::from_alternatives` then yields an
    empty token without the form. A position no arc reaches (a removed
    stop word) is not representable; the positions are renumbered.
- `NBest<L>` (`turbo:nbest<L>`, anchored `Aligned` to `L`) holds top-k
  ranked sequences over an existing tagged layer `L` of length `n` (such as
  `NBest<PosTags>` or `NBest<ChunkTags>` from a beam search).
  - **Columnar `[candidates, n]` layout**: `tags: Vec<Tag>` (interned
    against `tagset: Arc<TagSet>`) and optional per-token `prob:
    Option<Vec<f64>>` have length `candidates * n`, where candidate `k`'s
    slice is `k * n .. (k + 1) * n` (`tags_of(k)`, `probs_of(k)`).
    `score: Vec<f64>` holds the sequence-level score for each of the
    `candidates` hypotheses in non-increasing order (best first).
    `Layer::len` returns `n` so `Document::add` verifies alignment
    against the `n` items of `L`.

## Keys

A `Key<L>` names one instance of layer `L` on a document. Most layers
have one, `Key::DEFAULT`; gold tokens beside predicted ones, or a
normalized view beside the source, are named instances
(`Key::named("gold")`). `KeyId` is a key with its type erased, for
pipelines built at run time; it prints as the layer's id and the name,
`opennlp:tokens@gold`.

A layer keeps the instances it was made on: an aligned layer's anchor,
and the layers it indexes. They are the instances its step wrote before
it or read (for a layer added by hand, those its provenance lists, or
the default ones),
so tags made on the gold tokens say so (`Document::made_on`). A step
that would read such a layer beside another instance of the one it was
made on (those tags, with the predicted tokens) fails on that document
with `Error::Unaligned` instead of being handed items that do not line
up. Reading both instances, as an evaluation does, is fine.

One id names one layer type. A second type under an id is refused on
the document, whatever its instance name, and a pipeline holding both is
refused when it is built: on the wire they would be one layer.

## Provenance and status

Every layer records where it came from: the annotator and its version,
a hash of its configuration, the model (bundle id, revision and
sha256), where the model ran (backend, device, precision), and the
layers it read. An optional input is listed only when the document had
it. A layer made on another output of its own step (a confidence column
on the tokens written beside it) lists that output apart, under `on`,
since the step did not read it.

A layer's status is kept apart from its data. A layer never written has
no status; one written with no items is `Complete` and empty; one its
annotator declined is `Skipped`, with the reason; one its annotator
failed to make is `Failed`, with the error. An annotator declines a
document by returning `Error::Skipped`. A step that needs a skipped or
failed layer is skipped in turn, and says why: `opennlp:language was
skipped: no language evidence`, or `opennlp:tokens failed: ...`. So
"skipped" is never mistaken for "found nothing", and a stored document
says why a layer is missing.

### Evidence

`Evidence<L>` (`turbo:evidence<L>`) is one record per item of a result
layer `L`, so an entity, a category or any other result can be audited
the same way whatever produced it. A record names the producing
annotator and, when it ran a model, the model's identity; holds the
item's spans mapped through the views back to the source text; holds
the raw score, and the calibrated score and abstention reason when the
producer gives them (a field a producer does not give is absent, never
zero); and points at the units of other layers that support it (the
tokens an entity covers) by layer instance and item range, without
copying them. The result layers keep their own encoding.

`EvidenceStep<L>` builds the layer after the step that writes `L`, for
any `L` that implements `Evidenced`: `Entities`, `Categories` and
`Categories<L>` for sentences, structure blocks and chunks. The
dictionary and regex finders run no model and write 0 where they have
no score, so an entity's raw score is recorded only when a model (the
CRF finder's, say) produced it. A raw score that is not finite fails
the step on that document. `Evidence::verify`
checks a layer against its document and fails with a typed
`EvidenceError`: a model no layer of the document was made with, a span
outside the source or inside a character, or support outside a layer
the document holds. The graph import applies the same check.

## Document graph export

A search index or a coordinator that is not the process which ran the
pipeline holds a document as a document graph
(`turbo_analysis::export`): one typed, versioned record with the
source bytes and the alignment that decoded them, every view with its
parent and alignment edits, and one record per layer instance in the
order the layers were added, each with its full layer name, instance
name, `SCHEMA`, status and provenance. An item is named by a `NodeRef`
(layer name, instance name and index, printed as
`turbo:abbreviations@gold#2`), so an edge endpoint or a group member
resolves against exactly the instance it was made on, and every span
still maps through its view's alignment to exact source bytes.

Internal layers stay typed Rust types on both sides. A layer crosses
the boundary only through a `LayerCodec` registered for its type in a
`Codecs` set; its record body is that codec's bytes, written and read
with the boundary's `Writer` and `Reader` (`put_spans` and
`put_tagset` cover the shared span and tag columns). Codecs ship for
`Edges<From, To>` and `Groups<L>` in core, and
`turbo_analysis_layers::export::register_codecs` registers the shared
pipeline layers (sentences, tokens, words, subwords, tags, lemmas,
stems, terms, dependencies, entities, structure, boundaries, chunks,
categories, language spans, temporals and the document date, plus the
abbreviation relations); an add-on registers its own layer types the
same way. A layer with no codec refuses the whole export or import
with `ExportError::NoCodec` rather than being dropped or flattened,
and a skipped or failed layer needs a codec too, so its status and
provenance are never silently lost either.

The implementation lives in `analysis/core/src/export/mod.rs` (wire records,
codec contexts, and document reconstruction) and
`analysis/layers/src/export/mod.rs` (codecs grouped by layer family).
Import validates raw range endpoints before constructing typed indexes.
Skipped and failed views cannot carry text, parents, or alignment edits,
and skipped and failed layers must have zero item counts, so import never
discards those fields silently.

Two numbers version a graph: `FORMAT_VERSION` for the boundary (the
record shapes and the canonical byte encoding of
`DocumentGraph::to_bytes` together) and each layer record's schema,
copied from `Layer::SCHEMA`. Import is atomic and strict. The format
version, every layer schema, codec coverage, duplicate instances,
view parents, provenance references and, through the codecs, every
edge and group endpoint are checked, and any failure returns a typed
`ExportError` (`UnsupportedVersion`, `IncompatibleSchema`, `NoCodec`,
`DanglingReference`, `DanglingEndpoint`, `UnknownView`,
`InvalidSource`, `InvalidLayer`, `Malformed`) with no document: a
consumer that meets a newer format or schema upgrades, and there is
no partial recovery mode. Skipped and failed entries import back as
skipped and failed with the same reasons and the same errors, variant
for variant. A complete layer's provenance references must name
instances the graph holds; a failed layer's may name the required
input whose absence failed it, which no record holds, and those
rebuild by layer type alone. The source view's provenance is fixed
by the format (`turbo:source`) and a record claiming another is
refused instead of being silently replaced. Rebuilding provenance
turns its name strings into the `&'static str` the types hold through
a process-wide interner that leaks one copy of each distinct name
ever imported, so an importer that runs without end leaks the
vocabulary of annotator, version and instance names, not a copy per
document. The byte encoding is deterministic for a given graph,
so a round trip through export, bytes and import re-exports to the
same bytes (`analysis/core/tests/graph_export.rs` pins this against a
checked-in fixture, and the `document_graph` example in
`turbo-analysis-layers` shows a caller resolving nodes after the
round trip). The encoding is this boundary's transport form at
`FORMAT_VERSION`, not a persistence format with migration promises,
and the diagnostic trace JSON under `docs/receipts/document-trace` is
neither: it does not feed this boundary.

## Annotators

An `Annotator` declares what it reads and writes as tuples of layer
types, up to eight of each. One that reads `(View, Tokens)` and writes
`(PosTags,)` is handed a `&View` and a `&Tokens` and must return a
`PosTags`. It is handed nothing else, not the document, so it cannot
read a layer it did not declare and its provenance is complete; text
comes the same way, through a declared `View`. An input it uses only
when it is there is written `Option<L>`, and arrives as `Option<&L>`.

`annotate` handles one document. `annotate_batch` handles a whole batch
at once: by default it calls `annotate` on each, and a neural annotator
overrides it to pack every document's input into one device run. An
annotator that returns `Error::Annotator` fails that document alone. A
panic does too: it is caught, and a batch that panics is run again one
document at a time to find the documents that cause it.

`Bound` binds an annotator to the instances it reads and writes:
the default ones, or named ones (`.read(0, "folded")`).

An aligned output may hang on an earlier output of the same step: a
tokenizer that writes `(Tokens, Confidence<Tokens>)` scores the tokens it
wrote, without reading them. Outputs are added in the order of the
`Writes` tuple, so the base must come before the layer aligned to it,
and the aligned layer is made on the instance the step wrote under its
own key (named or default). The same holds for an index into an earlier
output: tokens written with their sentences point into those sentences.

## The analyzer

`Analyzer::builder()` takes the steps in any order, and the layers the
documents arrive with (`given`), such as structure from the parser
upstream. `build` refuses a pipeline that cannot run, naming why: two
layer types under one id, a layer two steps write, a required input
nothing writes, an aligned output whose anchor the step neither reads
nor writes before it, an aligned output written before its own base, or
a cycle. It then orders the steps by what they read and write into
stages.

`run` takes a batch of documents. Each stage runs over the whole batch
before the next starts, so a batching annotator sees every document at
once; steps in one stage compute side by side on threads, then store
their layers. A step that fails on a document fails only its own layers
there: the steps that need them are skipped and say why, the others
still run, and no other document is touched. Each document's result is
its first failure. `plan` lists the stages.

## The registry

A pipeline can also be assembled from configuration: a list of (name,
spec) lines, where a spec is a model location and options in order. Each
name is resolved through a `Registry<dyn Step>`, whose providers each
have a name, a priority, a cheap availability check and a claim on a
spec. The best available provider that supports the spec wins, and two
that tie at the top priority are an error, not a silent pick. Add-ons
are registered by the program that links them; nothing is found by
scanning. The shape is OpenNLP 3.x's `Provider` SPI (`Provider`,
`ProviderSpec`, `Providers`), and what a provider makes is the
registry's type parameter, as the service interface is in Java.

## Layer ids

A layer the OpenNLP gRPC standard defines has the standard's id (the
layer ids in `opennlp_document.proto`), so Java, Rust and the wire call
it one thing. A layer the standard lacks is `turbo:` until the standard
adds it. An add-on's own layers are in its own namespace. A layer
generic over another names both: `opennlp:embeddings<opennlp:sentences>`
is a vector per sentence, a different layer from
`opennlp:embeddings<turbo:chunks>`. A change to what a shared layer's
fields mean bumps its `SCHEMA`.

| Layer | Id | Anchored to | Crate |
|---|---|---|---|
| `View` | `turbo:view` | the document | `turbo-analysis` |
| `Structure` | `turbo:structure` | text | `turbo-analysis-layers` |
| `Sentences` | `opennlp:sentences` | text | `turbo-analysis-layers` |
| `Tokens` | `opennlp:tokens` | text | `turbo-analysis-layers` |
| `Words` | `turbo:words` | the document (indexes `Tokens`) | `turbo-analysis-layers` |
| `Subwords` | `opennlp:subwords` | text | `turbo-analysis-layers` |
| `PosTags` | `opennlp:pos` | tokens | `turbo-analysis-layers` |
| `WordPosTags` | `opennlp:pos<turbo:words>` | words | `turbo-analysis-layers` |
| `Lemmas` | `opennlp:lemmas` | tokens | `turbo-analysis-layers` |
| `WordLemmas` | `opennlp:lemmas<turbo:words>` | words | `turbo-analysis-layers` |
| `MorphFeatures<A>` | `turbo:feats<turbo:words>` / `turbo:feats` | `Words` or `Tokens` | `turbo-analysis-layers` |
| `ChunkTags` | `opennlp:chunk-tags` | tokens | `turbo-analysis-layers` |
| `PhraseChunks` | `opennlp:chunks` | text | `turbo-analysis-layers` |
| `Stems` | `turbo:stems` | tokens | `turbo-analysis-layers` (`turbo-stem-layers`) |
| `Terms` | `turbo:terms` | tokens | `turbo-analysis-layers` (`turbo-stem-layers`) |
| `Dependencies<A>` | `opennlp:dependencies` / `opennlp:dependencies<turbo:words>` | `Tokens` or `Words` | `turbo-analysis-layers` |
| `Entities` | `opennlp:entities` | text | `turbo-analysis-layers` |
| `Boundaries` | `turbo:boundaries` | sentences | `turbo-analysis-layers` |
| `Chunks` | `turbo:chunks` | text | `turbo-analysis-layers` |
| `Levels` | `turbo:levels` | text | `turbo-analysis-layers` |
| `Numbers` | `turbo:numbers` | text | `turbo-analysis-layers` |
| `Money` | `turbo:money` (and the instance `converted`) | text | `turbo-analysis-layers` |
| `Quantities` | `turbo:quantities` | text | `turbo-analysis-layers` |
| `Ordinals` | `turbo:ordinals` | text | `turbo-analysis-layers` |
| `Temporals` | `turbo:temporal` | text | `turbo-analysis-layers` |
| `DocumentDate` | `turbo:document-date` | the document (indexes `Temporals`) | `turbo-analysis-layers` |
| `Pii` | `turbo:pii` | text | `turbo-analysis-layers` |
| `Abbreviations` | `turbo:abbreviations` | text | `turbo-analysis-layers` |
| `Matches` | `turbo:matches` | text | `turbo-analysis-layers` |
| `Opinions` | `turbo:opinions` | the document (indexes `Tokens`) | `turbo-analysis-layers` |
| `Aspects` | `turbo:aspects` | the document (indexes `Tokens`, `Sentences`, `Opinions` when used) | `turbo-analysis-layers` |
| `Incivility` | `turbo:incivility` | the document (indexes `Tokens`, `Sentences`, `Chunks`, `Structure` when used) | `turbo-analysis-layers` |
| `Pragmatics` | `turbo:pragmatics` | sentences (indexes `Tokens` when there is evidence) | `turbo-analysis-layers` |
| `Stopwords` | `turbo:stopwords` | tokens | `turbo-stopword-layers` |
| `GlossaryTerms` | `turbo:glossary-terms` | text (indexes `Tokens`) | `turbo-glossary-layers` |
| `Vectors<L>` | `opennlp:embeddings<L>` | `L` | `turbo-analysis-layers` |
| `Confidence<L>` | `turbo:confidence<L>` | `L` | `turbo-analysis-layers` |
| `Utf16Spans<L>` | `turbo:utf16<L>` | `L` | `turbo-analysis-layers` |
| `Edges<From, To>` | `turbo:edges<From, To>` | the document (indexes `From` and `To`) | `turbo-analysis` |
| `Groups<L>` | `turbo:groups<L>` | the document (indexes `L`) | `turbo-analysis` |
| `Alternatives<L>` | `turbo:alternatives<L>` | text | `turbo-analysis` |
| `NBest<L>` | `turbo:nbest<L>` | `L` | `turbo-analysis` |
| `Language` | `opennlp:language` | the document | `turbo-analysis-layers` |
| `SourceEncoding` | `turbo:encoding` | the document | `turbo-encoding-layers` |
| `LanguageSpans` | `turbo:language-spans` | text | `turbo-analysis-layers` |
| `Categories` | `turbo:categories` | the document | `turbo-analysis-layers` |
| `Categories<L>` (`L` is `Sentences`, `Structure` or `Chunks`) | `turbo:categories<L>` | `L` | `turbo-analysis-layers` |
| `Evidence<L>` (`L` is `Entities`, `Categories` or `Categories<L>`) | `turbo:evidence<L>` | `L` | `turbo-analysis` |
| `Locations` | `turbo:locations` | text | `turbo-geo` |
| `Regions` | `turbo:regions` | the document | `turbo-analysis-layers` |
| `Containment` | `turbo:containment` | text | `turbo-analysis-layers` |
| `Vocabulary` | `turbo:vocabulary` | the document | `turbo-chunker` |
| `IndexedTerms` (`Terms` alias) | `turbo:indexed-terms` | tokens | `turbo-chunker` |
| `TermStats<L>` | `turbo:term-stats<L>` | `L` | `turbo-chunker` |
| `WeightedTerms<L>` | `turbo:weighted-terms<L>` | `L` | `turbo-chunker` |
| `ContentHashes<L>` | `turbo:content-hashes<L>` | `L` | `turbo-chunker` |
| `ChunkEntities` | `turbo:chunk-entities<turbo:chunks>` | `Chunks` (indexes `Entities`) | `turbo-chunker` |
| `TokenVectorWindows` | `turbo:token-vector-windows` | text | `turbo-analysis-layers` |
| `Mentions` | `opennlp:mentions` | text (indexes `Sentences`, `Tokens`, `Entities` when lifted from entities) | `turbo-coref` |
| `Speakers` | `opennlp:speakers` | text | `turbo-coref` |
| `LinkedEntities` | `turbo:linked_entities` | `Entities` | `turbo-entity-linker` |
| `Sketches<L>` | `turbo:sketches<L>` | `L` | `turbo-group` |
| `Boilerplate<L>` | `turbo:boilerplate<L>` | `L` | `turbo-analysis-layers` |
| `ProtectedSpans` | `turbo:protected_spans` | text | `turbo-normalize-layers` |

`Chunks` are retrieval chunks. The standard's `opennlp:chunks` are
phrase chunks (noun and verb groups), a different layer, which a phrase
chunker will write under that id.

`Categories<L>` is the one categories layer every categorizer writes:
the document categorizers (maximum entropy and SVM) and the sentiment
detector. Each item holds the best category and the score of every
category in the categorizer's order; `CategoryRow::ranked` lists them
best first, ties in category order. `L` defaults to `View`, the whole
document: `Categories` is `turbo:categories`, anchored to the document,
with one item. `Categories<Sentences>`, `Categories<Structure>` and
`Categories<Chunks>` are aligned to their layer, an item per sentence,
block or chunk. A crate categorizing units of its own layer implements
`CategoryScope` for it.

## Sentiment layers (`turbo:opinions`, `turbo:aspects`, `turbo:incivility`, `turbo:pragmatics`)

Four layers in `turbo-analysis-layers` (`sentiment`) hold opinion
expressions, aspect-based sentiment, multi-label incivility, and
sentence-level sarcasm/irony. All scores are `f64`, and every span is an
`Ix` or `IxRange` into `Tokens`, `Sentences`, `Chunks`, or `Structure`
so byte offsets follow view alignment without duplicating `Spans`:

- `Opinions` (`turbo:opinions`, anchored to `Document`, indexing
  `Tokens`): `tokens: Vec<IxRange<Tokens>>` (non-empty ranges in
  non-decreasing `(start, end)` order), `polarity: Vec<f64>` (prior
  signed polarity), `intensity: Vec<f64>` (finite, `>= 0.0`), and
  `negated: Vec<bool>`.
- `Aspects` (`turbo:aspects`, anchored to `Document`, indexing, when
  used, `Tokens`, `Sentences` and `Opinions`): `tokens:
  Vec<Option<IxRange<Tokens>>>` (`Some` for explicit aspect terms,
  `None` for implicit aspects), `sentence: Vec<Option<Ix<Sentences>>>`
  (required when `tokens[i]` is `None`), `category: Vec<Option<Tag>>`
  interned in `categories: Arc<TagSet>`, three polarity score columns
  (`positive`, `negative`, `neutral` in `f64`, finite and `>= 0.0`), and
  backing opinion links stored in CSR form (`opinions: Vec<Ix<Opinions>>`
  and `opinion_end: Vec<u32>`, kept in strictly increasing order per
  aspect).
- `Incivility` (`turbo:incivility`, anchored to `Document`, indexing
  `Tokens` when a row has evidence, and whichever of `Sentences`,
  `Chunks`, or `Structure` its rows target): `target: Vec<IncivilityTarget>` in strictly increasing
  target order, six `f64` label score columns (`insult`, `profanity`,
  `threat`, `identity_attack`, `intolerance`, `severe_toxicity` in
  `0.0..=1.0`), and CSR evidence spans (`evidence_label:
  Vec<IncivilityLabel>`, `evidence_tokens: Vec<IxRange<Tokens>>`,
  `evidence_end: Vec<u32>`) sorted by `(label, start, end)` within each
  row so `Incivility::evidence_for_label` returns a contiguous
  `&[IxRange<Tokens>]` slice by binary search.
- `Pragmatics` (`turbo:pragmatics`, aligned to `Sentences` and indexing
  `Tokens` when a sentence has evidence): `score: Vec<f64>` (`0.0..=1.0` per sentence) and `evidence:
  Vec<Option<Incongruity>>` where `Incongruity { positive:
  IxRange<Tokens>, negative: IxRange<Tokens> }` holds the two contrasted
  token spans (`size_of::<Option<Incongruity>>() == 16`).

## Syntactic words and multi-word tokens (`turbo:words`)

In Universal Dependencies, a single surface token can hold two or more
syntactic words whose forms are not substrings of the text: Spanish
`"del"` is `"de"` + `"el"`, Catalan `"pels"` is `"per"` + `"els"`,
French `"au"` is `"à"` + `"le"`, Portuguese `"pelos"` is `"por"` +
`"os"`, Italian `"nella"` is `"in"` + `"la"`, and German `"zum"` is
`"zu"` + `"dem"`.

`Words` (`turbo:words` in `turbo-analysis-layers`) sits beside
`Tokens`:

- Each syntactic word points at a non-empty `IxRange<Tokens>` (normally
  one surface token `t..t + 1`) in non-decreasing token order, so
  `Words::of_token` and `Words::of_sentence` find a token's or
  sentence's syntactic words by binary search.
- When a language or document has no multi-word tokens,
  `Words::from_tokens(&tokens)` sets `form: None` and `subspan: None`,
  costing only the single `Vec<IxRange<Tokens>>` column (`8` bytes per
  word) and slicing each word's surface form directly from the view.
- When a word's form is not a substring of the surface token (`"del"` ->
  `"de"` + `"el"`), `Words.form` carries the explicit form and `subspan`
  is `None`. When a split word is a contiguous substring of the surface
  token (`"can't"` -> `"ca"` + `"n't"`), `Words.subspan` carries
  `Some(Span)` and `form` stays `None`.
- Every word maps to its surface token's span (`Words::token_span`), so
  UTF-16 and code-point conversions at the edge (`Words::utf16_span`,
  `Words::code_point_span`) never split a surface token between words.
- Word-based layers (`WordPosTags` as `opennlp:pos<turbo:words>`,
  `WordLemmas` as `opennlp:lemmas<turbo:words>`, `MorphFeatures<Words>`
  as `turbo:feats<turbo:words>`, and `WordDependencies` /
  `Dependencies<Words>` as `opennlp:dependencies<turbo:words>`) anchor
  on `Words`, while token-based layers (`PosTags`, `Lemmas`,
  `Dependencies<Tokens>`) keep anchoring on `Tokens`; both can coexist
  on the same `Document`.
- The ids are `turbo:words` and `turbo:feats`, as the rule at the top of
  `analysis/layers/src/lib.rs` asks of a layer the OpenNLP gRPC standard
  does not have yet.
- `RuleMwtExpander` (`turbo:mwt-expander`) expands Spanish, Catalan,
  French, Portuguese, Italian, German and English contractions from
  `Tokens` into `Words`: closed-class tables written from each language's
  grammar, which split a homograph too (Portuguese `nos`, Italian `dai`).
  `ConlluWordReader` and `SyntacticSentence::write_conllu` in
  `turbo-formats::conllu` read and write CoNLL-U with its multi-word
  token ranges (`1-2 del`), empty nodes, comments and `MISC` kept, so a
  file in the format's own shape (LF line ends, one blank line per
  sentence) comes back as the same bytes. `SyntacticSentence::to_document`
  and `write_document_conllu` go through a `Document`, which keeps the
  form, lemma, part of speech, features, head and relation of each word
  and the spacing between tokens (`SpaceAfter=No`, `SpacesAfter=...`),
  not `XPOS`, `DEPS`, the rest of `MISC` or the comments.

## Subwords

`SubwordAnnotator` in `turbo-analysis-embed` writes `opennlp:subwords`:
it reads a text view (the source, or a named view through `Bound::read`)
and, when the document has them, the word tokens, and runs the bundle's
own tokenizer (`Tokenizer::encode_spans`, docs/tokenizer.md) over the
view. `SubwordAnnotator::open` loads the bundle and checks its
tokenizer against the reference ids as the core does; `SubwordsProvider`
offers it to a registry as `subwords`, the spec's location being the
bundle directory. The bundle's id and revision and the SHA-256 of its
manifest are the layer's `model` and its provenance's.

- The view is encoded as one row, with no truncation, so the layer
  covers the whole view however long it is; a text past the model's
  `max_seq` is split into chunks later, from these pieces.
- The row has no special tokens from the template and no prompt prefix,
  since neither is in the text. `SubwordAnnotator::overhead(role)` is
  what they add to every row the model sees (the template's special
  tokens and the prefix's tokens, counted on the prefix alone), for the
  chunker's `Budget::overhead`. A byte-level BPE or Unigram prefix can
  merge with the text's first piece, so a row can come out one token
  shorter than the pieces and the overhead.
- Each piece's span is the tokenizer's, a UTF-8 byte span of the view. A
  byte-level piece holding part of a character spans the whole
  character, so two pieces can share a span, and every span is on
  character boundaries; so does a SentencePiece byte-fallback piece. A
  special token written in the text (`[SEP]` typed by the author) is a
  piece of the text like any other.
- A bundle whose tokenizer is a SentencePiece `.model` (the
  `sentencepiece` kind, docs/tokenizer.md) gives the ids Google's
  sentencepiece library gives, and its offsets under the same span
  rules: whitespace at a piece's ends left out, a piece made for part
  of one character spanning that character.

With word tokens in the same view and in text order, every piece gets
the token it belongs to (`align`), in one pass over both span lists:

- a piece inside a token belongs to it;
- a piece over two or more tokens (a byte-level `'s` or `.)` across the
  word tokenizer's split) belongs to the first of them;
- a piece in no token (whitespace or punctuation the word tokenizer left
  out) belongs to the token before it, or to the first token when none
  comes before it.

So the pieces of one token are a run, which `Subwords::of` finds by
binary search. Tokens in another view, or out of order, fail the
document. A document with pieces but no tokens gets no token column.

### Against OpenNLP's WordPiece encoder

The WordPiece pieces are compared with OpenNLP's `WordpieceEncoder`
on the JVM (`analysis/embed/tests/goldens.rs`), from the `subwords`
goldens: the small BERT's bundle and the encoder read the same
vocabulary, `analysis/goldens/data/fixtures/all-minilm-l6-v2-vocab.txt`, uncased, with words
of up to 100 code points. On the 76 golden inputs, 65 give the same ids
and spans. The core's tokenizer follows upstream `tokenizers`, not the
JVM: the pieces feed embedding models that were trained on upstream's
tokenization, so matching upstream is what keeps the vectors right, and
the WordPiece pieces are also compared with upstream directly. The
other 11 inputs differ from the JVM where upstream does:

- A nonspacing mark that accent stripping removes (a combining acute,
  a variation selector U+FE0F, a virama, an Arabic vowel sign) is in
  the JVM's span of the piece before it, and not in ours; the ids are
  the same. Seven inputs.
- A capital sigma at the end of a word, after a cased letter, is
  lowercased to final sigma on the JVM, as the reference BERT's Python
  `lower()` does, and to sigma here, since upstream lowercases one
  character at a time. Final sigma has pieces of its own, so such words
  segment differently.
- Combining marks out of canonical order: the JVM gives every piece of
  the whitespace-delimited run the whole run's span, punctuation
  included. The ids are the same and our spans lie inside the JVM's.
- The ideographs U+2B820 to U+2B91F are CJK for the JVM, which uses the
  reference BERT's ranges, so each is a word of its own; upstream's
  range starts at U+2B920, so here they join the word around them.
- A special token's string in the text, such as `[CLS]`, is that special
  token here, because upstream matches its added tokens before it
  normalizes, and ordinary text on the JVM (`[`, `cl`, `##s`, `]`).

The test lists the inputs each difference touches, so a new one fails.

## Vectors

`VectorAnnotator` in `turbo-analysis-embed` writes a vector per sentence
(`opennlp:embeddings<opennlp:sentences>`) and per chunk
(`opennlp:embeddings<turbo:chunks>`). It embeds windows of the subword
pieces once each and pools every item's vector from the encoder's final
states over its pieces, through the core's late pooling
(`turbo_embed_write_ranges` in include/turbo/turbo.h). It drives the core through its
C interface, as any caller does, behind a small safe type (`Encoder`):
a runtime, the model on the device asked for (the CPU, a backend by
name, a device index, or the one `turbo_runtime_select` picks), and one
session at the precision asked for.

Three steps run it, since an aligned layer hangs on a layer the step
reads: `SentenceVectors` reads the subwords and the sentences,
`ChunkVectors` the subwords and the chunks, and `SentenceChunkVectors`
all three and writes both from the same windows. They can share one
loaded model (`Arc<VectorAnnotator>`). `VectorsProvider` offers them to
a registry as `vectors`, the spec's location being the bundle directory
and its options `layers` (`sentences`, `chunks`, `both`, `levels`),
`device`, `precision` and `prompt`. `LevelVectors`, for `levels`, is
described under "Levels" below.

The subwords must be the same bundle's pieces (the layer's model is
compared) and every layer must be in the subwords' view; a document
where either fails fails alone.

### Rows and windows

A row is the template's leading special tokens, the prompt's prefix
(the document role by default), the pieces of one window, then the
template's trailing special tokens. The overhead around the pieces is
`SubwordAnnotator::overhead` for the same role, the chunker's budget
overhead, so a chunk the chunker kept within a budget of at most
`max_seq` fits one window. The prefix is the prefix's own tokens, as
the overhead counts them; a byte-level BPE or Unigram prefix that the
tokenizer would merge with the text's first piece is not merged here.
A window holds at most `width` pieces: `max_seq` less the overhead. The
special tokens and the prefix are never pooled; an item is pooled over
its pieces alone.

Every window runs `width` pieces from its first piece, or to the end of
the text. The window policy, for one document:

1. An item covers the pieces whose span overlaps its own, found by
   binary search over the pieces' starts and ends. A piece that an
   item's edge cuts is covered, as the chunker counts it on both sides.
   A piece of no bytes at offset `x` is covered by an item `[start, end)`
   when `start <= x < end`: at the item's first byte or inside it, not at
   its end. An empty item covers no piece, whatever sits at its offset.
2. An item that covers no piece (an empty span, or whitespace a
   WordPiece tokenizer drops) is not placed.
3. The windows are laid out by the finest layer the step has: the
   sentences for `SentenceVectors` and `SentenceChunkVectors`, the chunks
   for `ChunkVectors`. Only the items of that layer that fit in `width`
   pieces take part. Sorted by their first piece, then their last, each
   goes in the current window when that window holds all of it;
   otherwise a new window starts at its first piece and becomes the
   current window.
4. Every other item is placed once those windows exist: first the items
   of the laying layer longer than `width`, then the items of the other
   layer (the chunks, beside sentences), each group in the same order.
   When an existing window holds the item whole, it is pooled there, in
   the window with the lowest first piece that does. Otherwise it is
   covered by the windows: from its first piece `p`, the window holding
   `p` that reaches furthest gives the part from `p` to that window's
   end, or to the item's end if that comes first, and the next part
   starts there. Only where no window holds `p` is a new window added,
   starting at `p`. Each of the item's pieces is in exactly one part.

So the windows depend on the laying layer and never on the longer items
beside it. A sentence is pooled from the same window, with the same
bits, by `SentenceVectors` and by `SentenceChunkVectors`. A chunk over
sentences that cover all of its pieces adds no window, however long it
is; a chunk over text that no sentence covers (a gap between two
sentences longer than what the windows reach) adds windows there. A
sentence that would cross a window's end starts the next window, so
windows overlap where that happens.

A chunk's vector depends on which layer laid out the windows. The same
chunk can get one vector from `ChunkVectors`, where the chunks lay out
the windows, and another from `SentenceChunkVectors`, where the
sentences do: the windows differ, so the context each piece is encoded
in differs. The configuration hash says which layer laid out the
windows, so the two are told apart in provenance.

The windows of every document in a batch go through the same runs: up
to the session's `max_batch` rows and `max_ranges` ranges a run (16 per
row unless set). A window whose ranges do not all fit in a run goes
again, with the rest, in the next.

### What a pooled vector is

A vector is the mean of the encoder's final states over the item's own
pieces. The template's special tokens and the prompt's prefix are in
every row, so they shape the context each piece is encoded in, but they
are never pooled. So a sentence's vector here is not the vector the
core's text interface (`turbo_embed_write_text`) gives for the same
span: that pools over every token of its row, the special tokens and
the prefix included, and encodes the span with only its own text in
context.

The prompt of the vector step (`VectorOptions::prompt`, the `prompt`
option) must be the one the chunker's budget was counted with
(`SubwordAnnotator::overhead` for that role, as `Budget::overhead`).
The chunker takes the overhead as a number and the chunks layer does not
record the prompt, so the vector step cannot check this: a pipeline that
counts the budget for one role and embeds with another gets rows longer
or shorter than the chunker planned for, and a chunk the chunker kept
within `max_seq` may no longer fit one window.

### Items longer than a window, and items with no pieces

An item that no window holds whole cannot be pooled from one pass of
the encoder. Its vector is the mean of its parts' vectors weighted by
their pieces: the mean of all its pieces' final states, each piece
encoded in the context of its own part's window. The model cannot read
more than `max_seq` tokens in one pass, so there is no single pass to
match for an item longer than a window, and refusing such an item would
fail every document with a chunk at a coarse level of the hierarchy (a
section over many windows). Attention never crosses from one part's
window to the next.

An item that covers no piece has no token to pool. Its row is the zero
vector, so the layer stays aligned to its items, and `Vectors::empty`
lists it (in order, each once; the layer's check holds every listed row
to zeros). Cosine similarity is undefined for a zero vector: a reader
skips the items `Vectors::is_empty` names rather than comparing them or
indexing them.

### Normalization and provenance

The core is asked for the plain means, and the vectors are normalized
as the bundle says on the host, with the arithmetic of the core's CPU
backend (the norm summed in f64, at least 1e-12), so a split item is
normalized after its parts are combined. A bundle pooled by its first
or last token is refused: late pooling is the mean, which is not the
vector such a model was trained to give.

Every vector layer's provenance has the bundle's id, revision and
manifest hash as its model, and as its execution the backend, the
device's name, and the precision as the dtype the session computes in
and the tier asked for, such as `f32 (model)`. The configuration hash
covers the row length, the prompt, and the layer that lays out the
windows: the sentences for `SentenceVectors` and `SentenceChunkVectors`,
the chunks for `ChunkVectors`, the levels for `LevelVectors`.

The annotator checks what it is given, since `VectorAnnotator::embed`
takes its input from any caller: an id for every piece, the pieces'
starts and ends in text order, no span that ends before it starts, and
levels that pass their own check (a parent for every unit, before it
and around it); a document that fails any of these fails alone. A
backend that pools no ranges (one without `embed_write_ranges`) is
refused when the session is made or at the first write, with an error
naming the backend and the range write the vector steps need. A result
of another shape than the run was written for, or a read of fewer
bytes, is an error, never a shorter set of vectors.

## Structure

`Structure` (`turbo:structure`) is a document's blocks as a tree:
sections, headings, paragraphs, lists and list items, tables with their
rows and cells, code blocks and block quotes (`Block` also has captions,
footnotes and page headers and footers, for extractors of HTML and PDF
layout to write). Each block has a span, a kind, a parent and a depth:
a heading's or a section's level, a list's, a list item's or a quote's
nesting (1 for the outermost), and 0 for the rest. The blocks are in
text order with each parent before its children; the layer's check
refuses a parent at or after its child, a block outside its parent's
span, a block that starts before the one before it, and two siblings
that overlap. A shared column of kinds rather than a layer per kind keeps
the nesting (a code block inside a list item inside a quote) in one
place for the chunker to read.

`StructureAnnotator` in `turbo-structure` reads a `View` and writes the
layer; `detect` does the same outside a pipeline, and
`StructureProvider` offers it to a registry as `structure`, with no model
location and one option, `syntax`: `markdown` (the default) or `plain`.
`StructureProjector` projects a `Structure` layer from a view's direct
parent into the normalized view ([structure-projection.md](structure-projection.md)).
`ParagraphDetector` in `turbo-sentdetect` writes the same layer with
paragraphs alone (docs/segmentation.md); a pipeline runs one of the two,
since a layer has one writer.
It is written by hand, with no Markdown library: one pass over the
lines, keeping the open quotes and list items and at most one open
paragraph, code block or table inside them, as CommonMark's reference
parser does. A line ends at `\n`, `\r\n` or a lone `\r`, as for
`ParagraphDetector`. Every span starts at its block's first character
that is not whitespace (the marker, for a quote, a list item, a heading
or a code fence) and ends after its last one, so line breaks, trailing
blank lines and the indentation of a first line are outside it. A line
is blank when it holds only whitespace (`char::is_whitespace`, so a line
of U+3000 or U+00A0 is blank too). A byte order mark at the start of the
view is in no block. A thematic break (`***`, `---`) ends the block before
it and is no block itself.

Sections come from the headings at the top level: a section runs from
its heading to the end of the last top-level block before the next
heading of the same or a higher level, and holds the heading, the blocks
after it and the sections of lower levels. A document whose first
heading is level 2 has level 2 sections at the top. Blocks before the
first heading are in no section, and a heading inside a quote or a list
item opens none. The chunker cuts each top-level section on its own.

With `markdown`, the blocks are CommonMark's: ATX and setext headings,
paragraphs, block quotes with lazy continuation lines, bullet and ordered
lists (an item continues while a line is indented to its content, an
ordered item interrupts a paragraph only when it starts at 1, an empty
item never does, and a list holds the items of one bullet character or
one delimiter), and fenced and indented code. Tabs stop every four
columns. A list item's indentation is measured from where its parent's
content starts on each line, as CommonMark does, so `> > 1.  one` and
`>>     two` continue one item. The hand-labelled fixtures in
`analysis/structure/tests/fixtures.rs` pin these rules down, including
the cases of the CommonMark specification that are easy to get wrong;
the crate is not tested against the specification's examples or against
another Markdown parser, since it takes no dependency on one.
Front matter is skipped: when the view starts (after a byte order mark,
if any) with a line `---` and a later line is `---` or `...`, the lines
up to and including that one are in no block, as static site generators
read YAML metadata. CommonMark has no front matter and reads such a
start as a thematic break and a setext heading. Without the closing
line, or anywhere but the start, `---` is a thematic break as usual.
What it does not do: HTML blocks and link reference definitions are
paragraphs, footnote definitions are not recognized, and a list is not
marked tight or loose.

GFM pipe tables are tables: a delimiter row (`|---|:-:|`, with at least
one `|`) under a line with as many cells turns that line into the
header, splitting it from the paragraph above when there is one. The
rows run to a blank line or a line that starts another block, and every
row (the header included, the delimiter row not) is a `TableRow` with a
`TableCell` per cell, the cell's span trimmed, an empty cell an empty
span where the cell starts. A leading and a trailing `|` are not cells.
A `|` escaped with a backslash or inside a backtick code span does not
split a cell. That differs from GFM, where a pipe inside a code span
splits the cell unless it is escaped: a pipe inside code (`a | b`, a
shell pipeline) is almost always the code's own, and splitting there
gives a row more cells than its header. A body row keeps the cells it
has; it is not padded or cut to the header's width.

With `plain`, a text is read as a letter or a contract: paragraphs
between blank lines, the same list items (and `•` as a bullet; an
ordered item's number has at most three digits, so `2024. A great year`
is text), quotes marked with `>`, and headings of two kinds. A line of
its own underlined with three or more `=` (level 1) or `-` (level 2):
an email's signature line `-- ` is no underline. And a line at the top
level in capitals (at least three letters, an upper case one among
them, no lower case one, so `OK` is not one), at most 80 characters
long, not ending with `.`, `,`, `;` or `:` (level 1), when it is a
paragraph of its own or the first line of a paragraph whose second line
has lower case letters (`INTRODUCTION` over `This report covers the
year.`), from which it is split off. Such headings in capitals with
nothing after them but more of the same, like a name and a country
under a letter's closing, are paragraphs, so they open no section. An
underline under a paragraph of several lines ends the paragraph and is
no heading. There are no `#` headings, no code, no tables and no front
matter, so an indented line is text.
Tables of aligned columns in plain text are not detected.

## Abbreviations

`AbbreviationAnnotator` in `turbo-abbrev` finds abbreviation and acronym
definitions with the algorithm of Schwartz and Hearst (2003), "A simple
algorithm for identifying abbreviation definitions in biomedical text",
and links every later use of a defined short form to its long form, so
a chunk that only says "WHO" can still be matched with "World Health
Organization". It reads a `View`, its `opennlp:sentences` and, when the
document has them, its `opennlp:tokens`, and writes two layers:

- `Abbreviations` (`turbo:abbreviations`, in `turbo-analysis-layers`): a
  span and a form per item, `Short` (the short form where it is
  defined), `Long` (its long form) or `Mention` (a later use of the
  short form), in text order by start and then end. Spans may nest: a
  long form can hold a mention of an earlier abbreviation ("the WHO
  Regional Office (WRO)"). The check refuses an empty span, two items
  with one span, and items out of order.
- `Edges<Abbreviations, Abbreviations>`
  (`turbo:edges<turbo:abbreviations, turbo:abbreviations>`): one edge
  from every short form and every mention to the long form it stands
  for, of kind `definition` or `mention` (`abbreviation_links()`), with
  score 1. `Abbreviations::long_form` follows it.

`detect` does the same outside a pipeline, from a text, its sentence
spans and, optionally, the token columns. There is no OpenNLP original
and no model; the rules are the paper's and its authors' reference
implementation's (`ExtractAbbrev.java`), and scispaCy's
`AbbreviationDetector` (Apache-2.0) supplied edge cases for the tests.

Within a sentence, each `(` after whitespace is a candidate. The text
inside, to the matching `)` (the next `)` when the inside opens another
parenthesis), cut at the first `, ` and then the first `; `, is the
short form candidate; the text before it, back to the end of the
previous candidate or to the last `.` or `,` followed by whitespace, is
the long form candidate. When the inside has more than two words or is
longer than the text before, they swap: the word just before the
parenthesis is the short form and must hold an uppercase letter (the
reverse form, "WHO (World Health Organization)"). A short form holds a
letter and starts with a letter, a digit or `(`. Its letters and digits
are then matched against the long form candidate from the right,
case-insensitively, the first of them at the start of a word, and the
long form runs from the start of that word to the parenthesis. A pair is
kept when the short form has two characters or more and at most ten
letters and digits (`n`), and the long form is at least as long, neither
ends with the short form nor holds it followed by a space, and has at
most `min(n + 5, 2n)` words, split at whitespace and hyphens.
Characters are compared as `Character.isLetterOrDigit` and
`Character.toLowerCase` see them (`turbo-unicode`), and an uppercase
letter is general category Lu. Where it departs from the Java: any
whitespace counts where the Java looks for a space, the candidates are
read per sentence of the document where the Java reads a line at a
time, and the long form loses the punctuation its word boundary can leave at its ends
(leading characters that are neither letters nor digits, unless an
opening bracket the long form closes, and trailing quotation marks),
so "[genioglossus (GG)" gives "genioglossus" and "(FEV1)/forced vital
capacity (FVC)" gives "forced vital capacity".

A mention is an exact, case-sensitive occurrence of a short form after
its definition's closing parenthesis, with no letter or digit right
before or after it ("WHOLE" holds none), that overlaps no defining short
form, is not itself a long form and is not inside a longer mention ("IL"
in a later "IL-2" when both are defined). With tokens, a mention must
also start and end on token boundaries, so "WHO's" as one token holds
none. A mention refers to the latest definition of its short form
before it, so a short form defined again takes its new long form from
there on; uses before the first definition are not linked. Finding them
is one pass over the text after the first definition: a position is
compared only with the short forms that start with its first two bytes.

### Measured on MED1250

`analysis/abbrev/examples/med1250.rs` measures the detector on MED1250,
the 1250 MEDLINE titles and abstracts Ab3P's authors annotated by hand
(Sohn, Comeau, Kim and Wilbur 2008), from the Ab3P repository
(`github.com/ncbi-nlp/Ab3P`, commit
`41130cddfcba1449ba612905d4a51274f8f565a8`, file `MED1250_labeled`,
SHA-256
`5093fa8f130ee250add0d0fbde7fc736478e18fbcc4447b00b4179db47f4cf53`,
which the example checks):

```bash
git clone https://github.com/ncbi-nlp/Ab3P <dir>
git -C <dir> checkout 41130cddfcba1449ba612905d4a51274f8f565a8
cargo run --release -p turbo-abbrev --example med1250 -- <dir>/MED1250_labeled
```

Each line of a record (title, abstract) is one sentence, as the
reference implementation reads a line at a time. A detected pair is the
text of its two spans, compared exactly with the gold pairs after
whitespace runs are made one space; the file's synonyms (`//*` lines,
which its authors leave out of the gold standard) count neither way.
The file's 1221 gold lines hold 1203 pairs once a pair repeated within
a record is counted once.

| Long form punctuation dropped | True positives | False positives | False negatives | Precision | Recall | F1 |
|---|---|---|---|---|---|---|
| yes (as shipped) | 956 | 45 | 247 | 0.9550 | 0.7947 | 0.8675 |
| no (as the Java) | 950 | 51 | 253 | 0.9491 | 0.7897 | 0.8621 |

Among the misses are pairs the algorithm cannot find by design: a long
form whose letters are out of order or missing ("cAMP" for "adenosine
3',5'-cyclic monophosphate"), a long form in another sentence, a short
form of one character, and a gold long form that starts before the
shortest match ("PPIs" gives "pump inhibitors", where the gold is
"proton pump inhibitors"). The mentions are not scored: the set marks
definitions only. Two other rules were tried and measured worse:
bounding a long form candidate by the candidate before it only when that
one defined something (precision 0.9401, recall 0.7963), and starting a
long form at a hyphen as well as at whitespace (0.9332, 0.7781).

On licences: the Ab3P repository's README carries a public domain notice
(a United States Government Work, with no restriction placed on its use
or reproduction), so MED1250 is used. The abstracts themselves are
MEDLINE records, whose text may be under the publishers' copyright, so
the file is not committed; the example reads a local clone. The
Medstract gold standard and Schwartz and Hearst's own evaluation corpus
(BioText) could not be fetched from here, and no licence for them was
confirmed, so neither is used. PLOS-derived sets were not tried.

`cargo test -p turbo-abbrev` holds the candidate and matching rules on
small texts worked out by hand (scispaCy's cases among them),
`analysis/abbrev/tests/pipeline.rs` runs the step in an analyzer and
checks every span's start and end against the source bytes (ASCII and
non-ASCII), the links and their kinds, mentions with and without tokens,
redefinition, nested short forms and named instances, that random
strings of brackets and punctuation give valid layers, and that 50,000
definitions take one pass, and
`analysis/abbrev/tests/gold.rs` reads a small sample in MED1250's format
with its quirks (carriage returns, a record split in two, synonyms,
comments in Windows-1252) and checks the scoring. None needs the
network.

## Rule matching

`turbo-matcher` matches rules over a document's tokens and the layers
aligned to them, after spaCy's `Matcher`, `PhraseMatcher` and
`SpanRuler` (MIT): token patterns, large phrase lists, and a step that
writes their matches as a span layer. Patterns are Rust values built
with builders; there is no pattern language read at run time (a reader
for spaCy's JSON pattern files could come later). A match is a range of
tokens (`IxRange<Tokens>`), so its bytes are its tokens' bytes, in the
view the tokens were made in, and `Document::to_source` takes them to
the source.

### What a pattern reads

An `Input` is one document's tokens, the text of their view, and the
token layers the caller has: `Input::new(text, &tokens)`, then
`with_pos`, `with_lemmas`, `with_norms`, `with_stems`, `with_entities`,
`with_stopwords` and `with_numbers`, each checked to be one item per
token. A test reads one string attribute (`Attr`) or one flag (`Flag`):

| `Attr` | Value | spaCy |
|---|---|---|
| `Text` | the token's text in its view | `ORTH`, `TEXT` |
| `Lower` | the text lowercased (`str::to_lowercase`) | `LOWER` |
| `Prefix`, `Suffix` | the first character, the last three | `PREFIX`, `SUFFIX` |
| `Shape` | spaCy's `word_shape` ("Xxxxx", "dd.d") | `SHAPE` |
| `Norm` | a string column the caller aligns to the tokens | `NORM` |
| `Lemma`, `Stem` | `opennlp:lemmas`, `turbo:stems` | `LEMMA` |
| `Pos` | the tag in `opennlp:pos` | `TAG`, `POS` |
| `Entity` | the kind of the first entity in `opennlp:entities` covering the token, or the empty string where none does | `ENT_TYPE` |

The flags are spaCy's lexical flags, defined as its `lex_attrs.py`
defines them with Python's `str` methods (`Alpha`, `Ascii`, `Digit`,
`Lower`, `Upper`, `Title`, `Punct`, `Space`, `Bracket`, `Quote`,
`LeftPunct`, `RightPunct`, `Currency`, `LikeNum`), and four read from
the token's context: `SpaceAfter` (whitespace follows it, spaCy's
`SPACY`), `SentStart` (from the tokens' sentence column), `Stop` (from
`turbo:stopwords`) and `Number` (it overlaps a `turbo:numbers` mention).
`Digit` is general category Nd, where Python's `isdigit` also counts
superscripts, and `LikeNum` is spaCy's language-neutral version, without
any language's number words. A test can also read the entity position
(`Iob`, spaCy's `ENT_IOB`), the length in characters (`Cmp`, spaCy's
`LENGTH` with `==` and the other comparisons), or call the caller's own
function on the token's text (`Tok::text_where`), which covers what
spaCy's custom extension attributes and `REGEX` do.

A pattern that reads a layer the input lacks fails the call with
`MatchError::Missing`, naming the layer, as spaCy refuses a `POS` pattern
on an untagged document. Normalized text is matched by making tokens on
the normalized view (`nfkc`, `dehyphenated`) and binding the matcher's
`View` and `Tokens` reads to that instance; `Text` is then the
normalized form, and the offsets map back through the view's alignment.

### Token patterns

`Matcher::add(label, &[Tok])` adds one pattern under a label. A `Tok`
holds tests that must all pass (`Tok::text("a")`, `Tok::lower("a")`,
`Tok::attr(Attr::Pos, "NN")`, `Tok::attr_in`, `Tok::attr_not_in`,
`Tok::flag`, `Tok::iob`, `Tok::length`, `Tok::text_where`, joined with
`and`; `Tok::any()` is spaCy's `{}`), and the operators: `!tok` (spaCy's `!`,
one token that fails the tests), `.opt()` (`?`), `.star()` (`*`),
`.plus()` (`+`) and `.repeat(min, max)` (`{n}`, `{n,m}`, `{n,}`,
`{,m}`). A repeat whose most is zero, or below its least, is refused
(`PatternError::Repeat`), where spaCy drops a `{0}` token from the
pattern. Values are compared as written: `Tok::attr(Attr::Lower,
"Boston")` matches nothing, as in spaCy.

`Matcher::find` reports every run of tokens the whole pattern matches,
every start and every end, as spaCy's `Matcher` does: `a b*` over
"a b b" gives "a", "a b" and "a b b". Empty runs are not reported, a
label matches a run once however many of its patterns do, and matches
come in token order (start, end, label). `Matcher::set_greedy` keeps,
per label, the first (`Greedy::First`) or longest (`Greedy::Longest`)
matches that share no token, as spaCy's `greedy`. `find_in` matches
inside a range of tokens only, as spaCy matches a `Span`.
`Matcher::align(input, &match)` says which `Tok` of the match's pattern
took each of its tokens, as spaCy's `with_alignments`. It is a pass of
its own over a match already found, so matching keeps no paths: a
table of which pattern tokens can take the rest of the match from
where is filled from the end, then read forward, each `Tok` taking as
many tokens as it can while the rest still fits (a backtracking
regular expression's greedy choice). That is spaCy's alignment except
in two of its test cases, where spaCy's state machine picks another
valid one (`a+ a a?` over three `a`s, and a run of `a` under
`a* b* a*`); the tests name both. As in spaCy,
`find` does not stop at sentence ends, so a match can cross one;
calling `find_in` with each sentence's tokens keeps matches inside
sentences. `remove`
drops a label and its patterns; adding the name again starts afresh.

The patterns compile to one Thompson automaton: a state takes one token
that passes a `Tok`'s tests, split states branch without taking one,
and each pattern has one accepting state. A repeat `{n,m}` is `n`
copies and `m - n` nested optional ones, so a run of tokens has one path
through it. Matching keeps the set of states the automaton can be in
after each token, each state once, and never backs up: from one start,
`k` tokens cost at most `2 * k * states` state visits whatever the
operators. Patterns that make a backtracking matcher exponential (`a?`
forty times then `a` forty times over forty `a`s; ten `a*` and a `b`
the text never has) stay inside that bound, which
`Matcher::find_in_counting` reports and `analysis/matcher/tests/backtracking.rs` checks,
and doubling the tokens a start reads doubles its visits. Since every
start runs, a pattern that can run to the end of the document (`{}*`)
costs time quadratic in the tokens, as the matches it reports can be. A
call looks up each token's values in the matcher's vocabulary once
(one hash of each attribute the patterns read), so the tests themselves
compare `u32` ids. The states a pattern can start in are indexed by the
value their token must have (the first `Tok::text`, `Tok::lower`,
`Tok::attr` or `Tok::attr_in` test of a token that is not negated), so a
start tests only the patterns its token can begin, plus those whose
first token has no such test. A token that does begin many patterns
still tests each of their next tokens: thousands of patterns that share
a first token are cheaper as phrases.

### Phrase lists

`PhraseMatcher::new(field)` matches phrases over one `Field`: an
attribute or a flag. `add(label, tokens)` adds a phrase as its tokens
and returns its `PhraseId`; a phrase added again under the same label
keeps its id, and under another label it matches once for each, as in
spaCy. Phrase tokens are turned into the field's value first: over
`Lower`, `Prefix`, `Suffix` and `Shape` they are lowercased or shaped,
as spaCy reads a pattern document's attributes; over `Norm`, `Lemma`,
`Stem`, `Pos` and `Entity` they are the values (lemmas, tags); over a
flag, each phrase token's flag comes from its text, and the flags that
come from a layer or the context are refused for phrases. Phrases are
tokenized by the caller, the same way the documents are, or they will
not line up with the tokens. `find` reports every occurrence, overlapping
ones too, in token order; `remove` drops a label's phrases. The token
matcher of `turbo-glossary` sits on this: it keeps the glossary entry of
each `PhraseId`, and a match names the entry.

The phrase list is a trie over token ids, written here rather than taken
from daachorse, aho-corasick or fst. The root's edges are an array
indexed by id; every other edge is a `(state, id)` pair packed into a
`u64`, in one open addressing table with linear probing, kept below
seven tenths full. Each state holds the head of a list of the phrases
ending there. A call looks each token up once, then each start walks
the trie as far as the tokens follow it, so it costs one lookup per
token plus one edge per token a walk takes (at most the longest
phrase's length per start). There are no failure links, as an
Aho-Corasick automaton has: walks from most starts end after one or two
tokens.

Measured with `analysis/matcher/examples/phrase_bench.rs` (release build, one core of a
4-core 2.1 GHz Xeon) on the Universal Dependencies English Web Treebank
(CC BY-SA 4.0), cloned for the run and not in the repository: the
phrases are the 479,356 distinct lowercased one- to four-token n-grams
of the training file, topped up to 1,000,000 with 520,644 random
two- to four-word phrases over its words from a fixed seed (3.05 million
phrase tokens, 17,023 distinct values). Matched over `Attr::Lower`:

| Measure | Value |
|---|---|
| Build | 1.2 s |
| Trie states | 1.52 million |
| Heap the matcher holds (`heap_bytes`) | 64.9 MiB |
| Growth of the process's RSS over the build | 92.1 MiB |
| Dev file (24,788 tokens, 0.13 MB): 41,544 matches | 2.8 ms, 8.9 million tokens/s, 45 MB/s |
| Test file (24,740 tokens, 0.12 MB): 41,268 matches | 2.8 ms, 8.9 million tokens/s, 45 MB/s |

Most of the heap is the edge table: 1.5 million edges in 4.2 million
slots of twelve bytes, since the table had just doubled. Most of a
call's time is lowercasing each token and looking it up. The same run
reports three token patterns (a run of titlecase tokens, a number and
a word, a short quotation) at about 8 million tokens a second.

### The span ruler

`SpanRuler` holds token patterns (`add_pattern`) and phrases
(`add_phrase`) under labels, each with a rule number in the order they
were added, and runs as a step. Phrases go to one phrase matcher per
`Field`: `add_phrase` to the field the ruler was made with (spaCy's
`phrase_matcher_attr`), `add_phrase_on(field, ...)` to any other, so
one ruler can hold, say, phrases by lowercase and phrases by lemma. It reads a `View` and its `Tokens`, and
whichever of `opennlp:pos`, `opennlp:lemmas`, `opennlp:entities`,
`turbo:stopwords`, `turbo:numbers` and `turbo:stems` the document has,
and writes `Matches` (`turbo:matches`, in `turbo-analysis-layers`): a
span, a token range, a label (tag set `rules`, the labels in the order
they were first added) and a rule number per match, in token order.
`SpanRuler::bound(name)` binds it to write the instance `Key::named(name)`,
so several rule sets write side by side, as spaCy's `spans_key`. Its
`Policy` is the overlap policy: `Keep` writes every match, overlapping
and nested ones too (the layer is `Overlap::Any`); `Filter(rule)` keeps
a subset that does not overlap with `filter_spans`, where
`FilterRule::LongestFirst` is spaCy's `filter_spans` except that it
compares lengths in bytes rather than tokens; `LongestTokens` is
spaCy's `filter_spans` exactly, lengths in tokens, the earliest first
among equally long matches, and of matches over the same tokens the
label added first. The two differ where a long token meets several
short ones ("Internationalization" against "New York City"). A label
matches a run of tokens once, with the first rule added.

### Against spaCy's tests

`analysis/matcher/tests/spacy_matcher.rs`,
`analysis/matcher/tests/spacy_phrase_matcher.rs` and
`analysis/matcher/tests/span_ruler.rs` port spaCy's matcher, phrase matcher and span
ruler tests (from spaCy's main branch of 2026-09-30, commit
`c2dabfce`) with the same expected matches; each test names the spaCy
test it ports, beside a few of their own (bad repeats, the ruler as a
pipeline step, a missing layer). A
document spaCy builds with its tokenizer is built here from the tokens
that tokenizer makes, and a text spaCy shows with its own spacing shows
here with the words joined by a space. Not ported:

- Callbacks (`on_match`: `test_issue615`, `test_matcher_from_usage_docs`,
  `test_matcher_callback`, `test_phrase_matcher_callback` and the like):
  a caller acts on the returned matches instead.
- `MORPH`, `IS_SUBSET`, `IS_SUPERSET`, `INTERSECTS` and list-valued
  extension attributes: morphological features and set-valued
  attributes are not attributes here.
- `REGEX` and `FUZZY`: no regular expression engine or edit distance is
  built in; `Tok::text_where` takes the caller's own test.
- `LIKE_URL`, `LIKE_EMAIL`, `DEP` and dependency patterns: the first two
  are spaCy's heuristics, not ported; dependency-tree patterns are a
  separate effort.
- Pattern validation, unknown `greedy` values and passing a pattern for a
  list of patterns (`test_issue3549`, `test_invalid_greediness`,
  `test_matcher_basic_check`, `test_phrase_matcher_validation`): the types
  rule these out at compile time.
- Pickling, `pipe`, warnings, multiprocessing, the scorer, entity
  annotation (`annotate_ents`, `ents_filter`) and pattern `id`s of the
  span ruler: there is no serialization or Python pipeline here, the
  ruler writes `turbo:matches` rather than entities, and a rule number
  stands in for an `id`.

## Term analyzer and per-language normalization profiles (`turbo:terms`, `turbo:stems`)

`Stems` (`turbo:stems`) and `Terms` (`turbo:terms`) in
`turbo-analysis-layers` (re-exported by `turbo-stem-layers` and
`turbo-stem`) hold token-aligned stem and layered term forms:

- `Stems` (`turbo:stems`, aligned to `Tokens`): one stem per token in a
  packed `Strs` buffer. Written by `StemAnnotator` in `turbo-stem`.
- `Terms` (`turbo:terms`, aligned to `Tokens`): `surface: Strs` (the
  original token text), `folded: Strs` (the character-normalized form),
  and optional `stem: Option<Strs>` and `lemma: Option<Strs>` columns
  when a stemmer or lemmatizer produced them. `Terms::get(i)` returns a
  borrowed `TermSlot<'_>` with the four slots for token `i`, and
  `Terms::check` verifies that `folded` and any present `stem` or `lemma`
  column match `surface.len()`.

`Dimension`, `Term`, `TermAnalyzer`, `NormalizationProfile`,
`NormalizationProfiles`, and `TermAnnotator` in `turbo-stem` port OpenNLP
3.x's `opennlp.tools.util.normalizer.{Dimension, Term, TermAnalyzer,
NormalizationProfile, NormalizationProfiles}`:

- `Dimension` has twelve variants in canonical application order:
  `Original`, `Nfc`, `Nfkc`, `Whitespace`, `Dash`, `CaseFold`,
  `FullCaseFold`, `AccentFold`, `EmojiFold`, `ConfusableFold`, `Stem`,
  and `Lemma`.
- `TermAnalyzer` applies the configured dimensions in canonical order,
  eagerly computing and caching each configured dimension in `Term` and
  evaluating any unconfigured character-level dimension lazily on first
  `Term::at(dim)` call (and memoizing it in a `OnceLock<String>` slot).
  `Term::peel()` returns the character-folded layer below `Stem` or
  `Lemma` when the final configured dimension is token-level.
- `NormalizationProfiles` registers the 22 built-in language profiles by
  ISO 639-3 code (`ara`, `cat`, `dan`, `deu`, `ell`, `eng`, `fin`,
  `fra`, `gle`, `hun`, `ind`, `ita`, `nld`, `nno`, `nob`, `nor`, `por`,
  `ron`, `rus`, `spa`, `swe`, `tur`) and resolves ISO 639-1 two-letter
  codes (`en` -> `eng`, `de` -> `deu`, `fr` -> `fra`, `es` -> `spa`,
  `nb` -> `nob`, `nn` -> `nno`, `no` -> `nor`, `in` -> `ind`, etc.) as
  JDK 21's `Locale.of(code).getISO3Language()` does. Each profile pairs a
  `SnowballAlgorithm` with `AccentFoldKind::Latin` (`AccentFold`, for
  `cat`, `eng`, `fra`, `ita`, `por`, `spa`), `AccentFoldKind::German`
  (`GermanUmlauts`, for `deu`), or `None` (for the Nordic and other
  languages), and `profile.matching_analyzer()` builds the corresponding
  `[Nfc, CaseFold, AccentFold?, Stem]` chain.
- `TermAnnotator` runs in an `Analyzer` pipeline, reading `(View, Tokens,
  Option<Language>, Option<PosTags>, Option<Lemmas>)` and writing
  `Terms` (`turbo:terms`), selecting the profile from the top-ranked
  code in `opennlp:language` when present, then its configured fallback
  language, and otherwise an `[Nfc, CaseFold]` analyzer (the default has
  no fallback language). The analyzer for each language is built once
  and shared by every document in that language, so its stem cache
  carries over between documents.

## The chunker

`turbo-chunker` reads the sentences, the embedder's subwords, the
boundary costs and, when the document has it, its structure, and writes
the chunks. Every length is counted in the embedder's own tokens, over
each chunk's whole span: what lies between two of its sentences (a
paragraph break, which byte-level BPE makes a token of) is in the chunk
too, and a subword a chunk's edge cuts counts on both sides, as each
chunk's own tokenization makes a token of its part. `Subwords` holds
only the text's own pieces; the special tokens a template adds and a
prompt prefix are the budget's overhead.

It chooses every boundary of a run of sentences together, the way TeX
breaks a paragraph into lines, rather than filling each chunk to the
budget and cutting wherever it lands. A chunk costs the boundary that
closes it plus a size penalty: `Budget::size_weight` times the squared
distance from the target size over the squared target, rounded down to
an integer. The optimizer finds the cut of least total cost. No chunk
is longer than the budget's maximum, unless one sentence alone is: that
sentence is then a chunk of its own, marked in `Chunks::over_budget`,
for a clause splitter to cut further; its `cut` still says why it starts
where it does (a section, say). Time is the number of sentences times the most
that fit in one chunk.

Every cost is an integer, so the same layers give the same cuts on
every machine and backend, with no float summed in a different order to
tip a choice. Two cuts tie only when they cost exactly the same; then
the one whose first boundary comes earliest wins, then its second, and
so on. The table runs from the end of each run back, so each entry
keeps the earliest end of equal least cost. A chunk's size penalty is
exact, never capped: a size is at most 2^33 tokens and the weight under
2^32, so a penalty is under 2^98 and fits an i128, and the optimizer
sums chunk costs in i128. Two different cuts never look equal for a cap
or an overflow, and a chunk of more than one sentence is never let past
the maximum to escape a huge penalty. A total could saturate only past
2^28 chunks each billions of tokens from the target. The `Chunks` layer
stores each chunk's cost saturated to the i64 range. A unit test checks the optimizer
against trying every cut of up to eleven sentences, with negative and
zero costs, a target of zero and several weights and overheads.

`segment` also takes a chunk term: a cost of a whole chunk, added beside
its size penalty and clamped to plus or minus 2^60 units, since the
caller supplies it. It is how a signal over whole
chunks, such as how tightly a chunk's sentences cluster, enters the
optimizer; `Chunker` passes `no_term`, and `CohesiveChunker` passes
cohesion (see Drift and cohesion below).

### Boundary costs

What a boundary costs is the `Boundaries` layer (`turbo:boundaries`),
aligned to the sentences: entry i is the gap after sentence i, the last
one the end of the text. Each entry is an `i32` cost and a `Cut`
reason. Costs are in `Boundaries::UNIT`s (1000); a plain sentence
boundary costs one unit, a stronger break less, and a signal that
argues for a cut may write a negative cost. A signal computed in
floating point is rounded to units once, when it is written.

The chunker reads only the default instance, and a layer has one
writer. With one boundary signal, that signal writes the default
instance itself, as `StructureBoundaries` does. With several (structure,
drift, topic shifts, coreference, discourse markers), each should write
a named instance of this same shape, and a step that weighs them with
integer weights and sums them should write the default instance,
computing in i64 and saturating to the i32 range so no weight or sum
wraps around. A new signal is one more instance, never a new column. A
term that belongs to a whole chunk rather than to a gap (how
tightly its sentences cluster) is not a boundary cost; it is the
optimizer's chunk term.

`StructureBoundaries` writes the structural term from the sentences and
`turbo:structure`. A block holds the sentences it has a byte of, found
with two binary searches; it starts at the gap before the first of them
and ends at the gap after the last, so a block edge inside a sentence
moves to the nearest gap. A section is the exception: it holds the
sentences that start in it, the same rule (one shared function) the
chunker splits its runs by, so a section start inside a sentence is the
gap after that sentence, and every chunk marked `Section` starts a run.
Each gap takes its strongest break:

| Break at the gap | Reason | Default cost |
| --- | --- | --- |
| A section starts or ends | `Section` | 0 |
| A heading starts | `Heading` | 100 |
| A paragraph, list, table, code block, quote, caption, footnote or page header or footer starts or ends | `Paragraph` | 400 |
| A list item or table row starts or ends | `Item` | 600 |
| None of these | `Sentence` | 1000 |

A gap where a heading ends costs 4000 more, so a heading stays with what
it heads. With no structure every gap is a sentence boundary. It writes
the default instance when it is the only signal, and the `structure`
instance beside drift.

### Hard limits

A chunk boundary never falls inside a sentence: boundaries are only
between sentences. Nor does it fall inside a table row, a code block or
a quote, unless that unit alone is over the budget's maximum. These
limits depend on the budget, so they are not costs: the chunker reads
the structure itself. A unit holds the sentences it has a byte of. A
gap is held when any unit that holds sentences on both sides of it fits
the budget, and the sentences joined by held gaps are one unit to the
optimizer. When units are around a gap and none fits, the gap may be
cut, and the chunk that starts there is marked `Cut::Inside` with the
smallest such unit's kind. Units nest: a code block that fits inside a
quote that does not is held whole while the quote is cut around it. No
boundary cost, however low, overrides a limit.

Two units whose bytes do not overlap can still share a sentence, when
an edge falls inside it, and a chain of such units can join more
sentences than the budget holds though each unit fits. That run must be
cut somewhere, so every gap in it is freed and a chunk that starts at
one is marked `Cut::Overlap` with the kind of the smallest unit that
held it. A section's edge is always a cut: when a sentence that
straddles a section's start belongs to the section before (by its
start) and a unit in the new section holds it, the unit is cut at the
section's edge and the chunk there says `Section`.

### Sections

With structure, no chunk crosses a top-level section. The chunker sorts
the top-level sections by start, so sections listed out of text order
still make runs of their own. Each section, and
each run of sentences outside every section, is its own run, so an edit
re-chunks only its own section, and each run becomes a chunk one level
up, the parent of the chunks cut from it. Every chunk records its level,
its parent, its sentences, why it starts where it does, what it cost the
optimizer (an integer, in units), its length in model tokens and
whether it is over the budget. The chunks layer refuses a parent that is
not at a level above its child or does not hold the child's span.

Not measured yet: the chunker against the stage 0 baselines on the
evaluation harness, with significance: the harness runs it (Evaluation,
below), and no record of it is committed yet.

### Drift and cohesion

Two signals come from the sentence vectors (`Vectors<Sentences>`, late
pooled). Both scale each vector to length one (`Rows`, which reads the
layer in place and keeps one over each row's length). A sentence with
no vector (one the layer's `empty` column lists, or a row of zeros or of
non-finite numbers) adds nothing and is not counted: it has no direction
to compare. Floating point (f64, summed in a fixed order) computes two
quantities only, drift's cosines and cohesion's spread; each is rounded
once to an integer, and every cost after that is integer arithmetic, so
the same vectors give the same costs.

Costs below are in the units `Boundaries` is written in: a plain
sentence boundary is `Boundaries::UNIT`, 1000, so one of these units is
a thousandth of a sentence boundary.

**Drift** (`DriftBoundaries`, the `drift` instance of the boundaries) is
a cost per gap. For the gap after sentence g:

1. The cosine between the sum of the `window` (3) sentences with a
   vector before the gap and the sum of the `window` after it, in
   millionths, summed directly: drift needs memory for a few vectors,
   not for the document. Only a gap with a full window on both sides is
   scored, as NLTK's TextTiling leaves the gaps at the ends unscored.
   With a window cut short at an end, a single odd first or last
   sentence was compared alone, its gap was a valley with a wall on one
   side only, and the optimizer set it apart as a chunk of its own (a
   cost of -2000 at the first gap, against about -420 for the same
   sentence in the middle). The price is no drift within `window - 1`
   sentences of either end, where a cut would leave a chunk of fewer
   than `window` sentences anyway. A gap that is not scored costs zero,
   and the rest are scored among themselves, in order, skipping it. The
   two gaps around an empty sentence see the same windows, so they cost
   the same. `window` is a `NonZeroU32`.
2. A moving average over `smoothing` (1) cosines on each side, rounded
   to an integer. Near an end it takes as many on each side as the
   nearer end allows, so the average stays centred and a valley next to
   an end stays where it is.
3. The score, by default the TextTiling depth (Hearst 1997): the
   highest smoothed cosine reached walking left while it does not fall,
   plus the same to the right, less twice the gap's own. It takes
   linear time, since a walk continues the walk from its neighbour. A
   steady slow drift has no valleys and scores nothing.
4. A score under `floor` (0.01) counts as zero; the gap then costs
   `-score * scale / 10^6`, truncated toward zero, where `scale` (2000)
   is what a score of one takes off the gap: a depth of 0.5 takes off
   one sentence boundary. `scale` is unsigned, so drift never makes a
   cut dearer.
5. A gap at the bottom of a valley (lower than the gap before, not
   higher than the one after, so a flat bottom marks its first gap)
   whose score is at least `shift` (0.1) gives `Cut::TopicShift`. A
   single outlier sentence in the middle of a stretch passes it (its
   dip is about 0.21 deep), and so does the first or last scored gap
   when the window there still holds an outlier at an end (about 0.11):
   the reason is named, and the cost (about -420 and -210) decides
   whether a cut is made there. The default is untuned; the tests pin
   this behaviour so a change to it is deliberate.

Comparing groups of sentences rather than single sentences keeps one
odd sentence from looking like a topic of its own: with a window of one,
both gaps around an outlier are full-depth valleys and the optimizer
sets it apart; with a window of three its dip is spread over six gaps
at about a tenth of that depth, and it stays in its chunk.

**Weighing.** `WeighBoundaries` reads the `structure` and `drift`
instances and writes the default one: `(structure * ws + drift * wd) /
1000` per gap, with each `Weight` a factor in thousandths (1000, the
costs as they are, for each by default), products in i64, their sum
saturating in i64, rounded down and saturated to the i32 range. The
reason is the structure's, unless the structure saw only a sentence
boundary and the drift a topic shift (and the drift weight is not zero).
A further signal is one more read and one more weight. In a pipeline
from configuration the three steps are `structure-boundaries` (option
`instance`: `default` or `structure`), `drift-boundaries` (options
`window`, `smoothing`, `score`, `floor`, `scale`, `shift`) and
`weigh-boundaries` (options `structure`, `drift`); an unknown option, a
value that does not parse, a negative `scale` and a `window` of zero are
refused.

**Cohesion** (`Cohesion`) is a chunk term. A chunk pays, for each of its
sentences with a vector, one less its cosine to the chunk's centroid.
With unit vectors summing to S over m sentences that sum is `m - |S|`,
so the term is `weight * (m - |S|)`, computed in f64 and rounded once to
the nearest integer, with `weight` (unsigned, 1000 by default) what a
sentence at right angles to its chunk costs: zero for a chunk whose
sentences all point one way, up to `weight * m` when they cancel.
Splitting a chunk never raises the total (`|A| + |B| >= |A + B|`) and
lowers it by exactly how far the parts point apart, so the term pulls
cuts toward where the vectors part, and the boundary costs and the size
penalty decide how many cuts that is worth.

S comes from running sums of the unit vectors (`Sums`), so each chunk
the optimizer tries costs one pass over the dimensions, whatever its
length, and the optimizer's time is multiplied by the dimension. The
sums take `(n + 1) * dim` f64s, so they are built for one run of the
optimizer (a top-level section) at a time and dropped after it: memory
follows the largest section, not the document. A document with no
structure is one run, so its sums cover all of it (about 820 MB at
100,000 sentences of dimension 1024); such a document needs sections, or
the plain `Chunker`.

`CohesiveChunker` is `Chunker` with this term. It is a step of its own,
not an option of `Chunker`, because it reads the sentence vectors: a
chunker that only might read them would still be ordered after them,
and a pipeline whose one vector step pools both sentence and chunk
vectors (`SentenceChunkVectors`, which reads the chunks) would then be a
cycle. It is not used unless chosen.

Alternatives in the code and tests:

- Score by distance (`Score::Distance`, one less the smoothed cosine),
  the quantity common semantic splitters threshold. On a slow drift it
  lowers every gap alike, so no gap stands out and every cut is cheaper,
  which favours more, smaller chunks. Depth is the default.
- A window of one sentence: it sets an outlier sentence apart (a test
  shows it).
- Windows cut short at the ends of the document: they set an outlier
  first or last sentence apart (step 1).
- Smoothing cut short on one side only at the ends: it moved a topic
  shift two sentences from the start onto the first gap.

Not tried: a percentile threshold over the document (it makes a gap's
cost depend on the whole document, so an edit far away moves it, and
the optimizer needs costs, not a yes or no), C99's rank matrix, lexical
TextTiling over tokens, windows that stop at section edges, and
pairwise cohesion (mean cosine between all pairs, which needs a banded
table of dot products rather than running sums).

The defaults are not tuned. Whether drift or cohesion helps, and at
what weights, needs the evaluation harness: retrieval records for the
stage 1 chunker on structure alone, structure with drift at a few
weights, and the cohesive chunker, at matched mean chunk length, with
significance, plus the boundary suite (Pk and WindowDiff) and the time
per document. None of that exists yet, so no gain is claimed.

## Term statistics and BM25

`turbo_chunker::terms` counts terms per retrieval unit for lexical
search, from the word tokens alone: no model and no vectors, so a BM25
baseline can be measured before any vector exists.

`TermIndexer` reads a text view and the tokens and writes the document's
`Vocabulary` and a term for each token (`Terms`, aligned to the tokens).
A term is the token's text in the view read, byte for byte: nothing is
lowercased here. To match without case, read a case folded view
(`NormalizeStep::new(CaseFold::full())` bound to write `folded`, and the
indexer bound with `.read(0, "folded")`); the tokens can stay on the
source, since a token's span is mapped through the folded view's
alignment to its parent. A view further down a chain than that is
refused. The view's edits must not cross a token edge: with one edit
turning "don't" into "do not" over the tokens "do" and "n't", each
token would map to the whole replacement, so a document with a token
edge strictly inside an edit is refused. Case folding replaces one
character per edit and never crosses one.

A token is a term when it holds at least one character
`char::is_alphanumeric` accepts (an alphabetic character, or a number
of category Nd, Nl or No). A token of punctuation or symbols alone (a
period) is not, and counts toward no length. This is close to Lucene's
`StandardTokenizer`, which drops punctuation, but not the same: that
tokenizer splits by UAX #29 word boundaries, keeps emoji as tokens
(no terms here) and makes no token of a lone No character such as "½"
(a term here), and its splits differ from the word tokenizer's. The vocabulary is in byte order, each term once, so a
term's id depends only on which terms the document holds, and a lookup
is a binary search.

`TermCounter<L>` reads the tokens, their terms, the vocabulary and the
units of one level, and writes `TermStats<L>`, aligned to the units:

- `TermCounter<Chunks>`: a row per chunk, at every level of the chunk
  hierarchy;
- `TermCounter<Structure>`: a row per structure block; the sections are
  the blocks of kind `Section`, and a nested section's terms count in it
  and in every section around it;
- `TermCounter<View>`: one row for the whole document;
- `TermCounter<Levels>`: a row per unit of the chunk, section and
  document levels (see "Levels" below), so every level gets its counts
  from the same code.

`ContentTermCounter<L>` writes the same layer from the same inputs plus
the items of a level and their `Boilerplate` flags (structure blocks by
default), counting only the tokens that start outside every flagged
item: a footer repeated on every page adds to no unit's length or
counts, and a unit that is all boilerplate has length 0. It is a step of
its own, so `TermCounter` does no work for boilerplate it is not given.

A token is in the unit its start is in. Each row is sparse: the unit's
term ids in ascending order with a count for each, by column, with
offsets per unit. The layer also holds each unit's length in terms and,
per term, how many of the level's units hold it (`df`). That df is
within one document; it is not the corpus document frequency BM25's idf
is defined over.

`Query` scores units with BM25 in Lucene's form (`BM25Similarity`,
`Bm25::LUCENE`: k1 = 1.2, b = 0.75):

```text
score = sum over query terms t of idf(t) * tf / (tf + k1 * (1 - b + b * length / mean_length))
idf(t) = ln(1 + (N - n(t) + 0.5) / (n(t) + 0.5))
```

The classic `(k1 + 1)` factor is left out, as Lucene has since version
8; it scales every score alike and leaves the ranking unchanged. N, the
mean length and n(t) are the collection's, and the caller supplies them
through `Collection`: from an index over the corpus for corpus search.
`LocalCollection` gives one document's own units as the collection (all
of a level's units, or those a filter keeps, such as the sections or
the chunks of one level), for ranking a document's chunks against each
other. As in Lucene, a unit with no terms is not counted in N or the
mean length. Lengths here are exact, where Lucene stores a length in
one byte, exactly only for short units, so Lucene's scores for long
units can differ slightly. Query terms are matched byte for byte and
must be normalized as the document's terms were; a term given twice
counts twice. `Query::rank` lists the units holding a query term, best
first, equal scores in unit order.

`CollectionStatisticsBuilder` gathers a corpus's statistics over the
`TermStats` of many documents (each unit with terms one unit of N, its
length added to the total, each term's df added) and merges builders
made apart, one per shard or thread; `build` gives `CollectionStatistics`,
terms in byte order, the same whatever order documents were added or
merged in. It is a `Collection`, has each term's `idf`, and has a binary
form (`to_bytes`, `from_bytes`: a magic and version, the counts, then
each term with its df) whose reader refuses truncation, trailing bytes,
counts past the payload, terms empty or out of byte order, and a df of
zero or past N.

`Bm25PostingIndexBuilder` adds the units of many documents as rows,
numbered in the order added, and `build` gives a `Bm25PostingIndex`: each
term's rows with its count, in row order, and the rows' own statistics.
`top_k` (with those statistics) and `top_k_with` (with a larger corpus's)
walk the query's lists a row at a time with MaxScore pruning: once k rows
are held, terms whose bounds (idf times the weight of the term's largest
count in its shortest row), added up from the weakest, cannot beat the
k-th score stop bringing in rows. Each row returned is scored term by
term in query order, as `Query::score` scores it, so the scores are the
full scan's to the bit and equal scores go to the earlier row; the tests
compare both over four `k1`/`b` settings, and
`analysis/chunker/examples/bm25_posting_bench.rs` times them.
`WeightedTerms<L>` holds, for each entry of a `TermStats<L>`, the weight
it adds to a query holding its term once.

The layers live in `turbo-chunker` beside the code that makes them while
only it reads them; they move to a schema crate of their own once
another crate does.

### Comparing two runs

`turbo_chunker::compare` checks whether two runs of the chunker over one
document cut it in the same places: two backends, or two precision tiers
on one backend. It takes each run's `Chunks`, where the model behind its
costs ran (the `Execution`: backend, device and precision tier, which the
caller takes from the layer the costs came from, since the chunker runs
no model) and, when the caller has them, the sentences, boundary costs
and budget the run was cut with. Runs whose chunks are in different
views are refused, as is a run with two rows at one place or with level
0 rows that overlap under one parent, and a run whose costs do not fit
it: sentences in another view or out of text order, a boundary cost
missing, or a chunk that does not start and end where a sentence does.

A cut is a chunk's start, as an exact byte offset, at its level, under
its parent (named by the parent's span). Cuts both runs make are
counted; the rest are reported in stretches between two shared cuts, in
order of level, parent and offset. A chunk's cost depends only on its
own sentences and the boundary that closes it, so the optimizer's
choices on either side of a shared cut are independent, and each
stretch is one choice: a moved cut is a stretch with a cut on each side,
an added cut one with a cut on one side only. Every cut in a stretch
carries why its chunk starts there, whether it is over the budget,
what the chunk cost its run (from the layer) and what each run's
boundary costs charge at that offset.

The layer records no runner-up, so each stretch's margin is computed:
how much more a run's own costs charge for the other run's cuts there
than for its own, in units. Each chunk from the shared cut on is priced
with the optimizer's own cost of one chunk (the boundary that closes it,
none at the end of its parent, plus its integer size penalty), so the
margin is exact and is what the run's whole cut would cost more had it
cut the stretch as the other did: at least the gap between its best and
second best choice, and zero for a tie. A test checks this against
changes of one to three gaps of the optimizer's cut on generated runs
of one and two sections. As a rule of thumb, a margin smaller than what
the precision tiers move a boundary cost by points at precision as the
cause.

A margin is computed only at level 0 (the level the optimizer cuts),
only after a shared cut, only where the other run's cuts are sentence
starts of this run's, and only where both choices end at the same byte.
Children of parents whose spans differ are never in one stretch, so
they share no cut and have no margin. It adds no chunk term, since the
layer records none, and it prices as if allowed a cut the run's own
limits would forbid (inside a unit it keeps whole) and a chunk over the
run's maximum. Each choice is priced with the pricing run's budget but
sized by its own run's token counts, which agree when the runs share a
tokenizer; runs with different tokenizers mix two counts. There is no
command for it: no record format holds chunks yet.

## Near duplicates and boilerplate

`turbo-group` finds copies, edited versions and page furniture among
the units of a level (structure blocks, sentences or chunks), inside
one document and across a batch. Everything it compares is a sketch, so
documents sketched apart, in other processes or on other machines,
compare: every hash constant is fixed, and a unit test pins one
sketch's value.

### Sketches

`Sketcher<L>` reads a view and the items of `L` and writes
`Sketches<L>` (`turbo:sketches<L>`), aligned to the items. An item's
words are its maximal runs of characters `char::is_alphanumeric`
accepts, read fragment by fragment for a discontinuous item, so no word
joins two fragments, and whitespace and punctuation are no part of any
word. Nothing is case folded: to compare without case, read a folded
view (`.read(0, "folded")`), as for terms. The items are in the view
read or in its parent; then each span is mapped through the view's
alignment, so blocks found on the source sketch from the folded text.
Per item the layer holds:

| Column | What it is |
| --- | --- |
| `words` | The item's word count |
| `exact` | A 128-bit hash of the word sequence: equal for the same words in the same order, whatever lies between them |
| `shape` | A 64-bit hash of the sequence with every number word replaced by one placeholder, so "Page 3 of 10" and "Page 4 of 10" share a shape |
| `simhash` | The 64-bit SimHash of the item's shingles, each occurrence counted |
| `minhash` | 64 MinHash values of the item's set of shingles, by one permutation hashing with optimal densification (Shrivastava, ICML 2017): one hash per shingle chooses a lane and a value there, and a lane no shingle chose borrows from the first chosen lane in its own fixed random order |

A shingle is a run of `shingle` consecutive words (3 by default); an
item with fewer words is one shingle of all of them, and an item with no
words has none (SimHash 0, every MinHash lane `u32::MAX`). Time is
linear in the bytes of the items, with a few operations per shingle.

### Duplicates and versions

`NearDuplicates<L>` reads the items and their sketches and writes the
groups among them (`Groups<L>`) and the links between them
(`Edges<L, L>`, with the tag set `turbo:duplicate-kinds`: `exact`,
`near-duplicate`, `version-edit`). Items with the same word sequence
are exact copies: the first copy links to each later one with score 1.
The first copies are then banded on their MinHash rows, 16 bands of 4
lanes, and any two that share a band are compared. An estimated
Jaccard similarity of at least `Thresholds::near` (0.9) is a near
duplicate, at least `Thresholds::version` (0.5) a version edit, and the
score is that estimate. Two items of similarity j share a band with
probability 1 - (1 - j^4)^16: 0.51 at 0.4, 0.88 at 0.6, 0.996 at 0.8.

Every link runs from the earlier item to the later one, so a version
link points from the older text to the newer, `Edges::outgoing` lists an
item's later copies and versions, and `Edges::incoming` its earlier
ones, each by two binary searches. Items that overlap never link: a
list item and the paragraph inside it hold one text and are not
duplicates, nor are a chunk and its parent. A group is the items joined
by exact and near-duplicate links; versions stay links and join no
group. The items must be in start order.

`cross_duplicates` does the same over a batch, given each document's
items and sketches, as one list in (document, item) order: links run
from earlier documents to later ones, pairs inside one document are
found too, and the result names each item by its document's place in
the batch and its index. Sketches of different shingle widths are
refused, since their MinHash rows do not compare.

### Boilerplate

`Boilerplate<L>` (`turbo:boilerplate<L>`, in `turbo-analysis-layers`)
flags the items of `L`, aligned to them: a `BoilerplateKind` (`Header`,
`Footer`, `Navigation`, `Disclaimer`, `Template`) or none for content,
and a score in `[0, 1]`. Readers that exclude boilerplate use the kind;
readers that down-weight it use the score.

`BoilerplateDetector<L>` reads the items and their sketches and writes
the flags. The analyzer hands it a whole batch at once. An item whose
shape is held by at least `min_docs` documents (2) and at least
`min_share` of the batch (half) is boilerplate, scored by the share of
the batch's documents that hold it. Its kind is what the layer says
(`BoilerplateUnit::hint`: a structure block that is or is inside a page
header, a page footer or a list), else a disclaimer when it holds at
least `disclaimer_words` (25) words and is in the second half of its
document's items, else a header in the first half and a footer in the
second. Inside one document, a shape held by two or more items that do
not overlap is a template, scored `1 - 1/n` for `n` copies. A document
alone is a batch of one and gets only templates. Items of fewer than
`min_words` (2) words are never boilerplate.

`BoilerplateProjector<L, S>` carries flags from one level (`S`, the
structure by default) to another in the same view: an item of `L` is
boilerplate when at least half of its bytes lie inside flagged items of
`S`, of the kind that covers the most of its bytes added up over every
flagged block and fragment (the first in kind order on a tie). Each
byte keeps the kind and score of the flagged block it lies in, touching
blocks included; where flagged blocks overlap, the innermost decides.
Its score is the byte-weighted mean of the covering scores, so a chunk that is a third footer
scores a third of the footer's score. `ContentTermCounter<L>` in
`turbo-chunker` counts terms as `TermCounter<L>` does with the tokens
inside boilerplate left out (see "Term statistics and BM25").

`turbo_group::export::register_codecs` registers the sketches, groups,
edges and boilerplate of structure blocks, sentences and chunks for the
document graph. The codecs refuse a truncated payload, a count past the
payload, trailing bytes, a shingle of no words, an empty item with
shingle hashes, an unknown boilerplate kind and a score outside
`[0, 1]` or not a number.

## Exact vector index

`turbo-vector-index` holds a bounded set of vectors next to inference,
such as label or schema-field candidates, the entities of a loaded
catalog or glossary candidates before a heavier scorer, and answers the
exact top-k for a query. It is not a corpus index: rows live in memory,
and there is no persistence, update or delete.

An `IndexBuilder` takes rows from `Vectors<L>` layers (`push_layer`,
which skips the items the layer lists as empty) or from slices (`push`),
each with the caller's metadata, and `build` returns a `VectorIndex`,
which is immutable and shared across threads. Rows are numbered from 0
in the order they were added.

- **Embedding space.** Every index records a `Space`: the `ModelRef` of
  the model that made its vectors, their dimension, and whether they are
  unit length (`Normalization::Unit` or `None`). A layer or a query
  names its space, and one from another model (id, revision or manifest
  hash), another dimension or another normalization is refused with
  `IndexError::Model`, `Dimension` or `Normalization`, never scored.
  The dimension is the width the vectors were cut to: a Matryoshka
  model's vectors at two declared widths (docs/bundle.md) are vectors
  of two dimensions, and do not compare.
- **Vectors refused.** A vector of zeros, a value that is not finite, a
  vector longer than `MAX_LENGTH` (`2^50`), and in a unit space a
  vector whose length is not within `UNIT_TOLERANCE` (`1e-3`) of 1, on
  insert and on query, each with its own error naming the row or query.
  A layer refused for any row adds none of its rows.
- **Scores.** `Metric::Dot` scores the dot product of the query and the
  row. `Metric::Cosine` stores each row divided by its length (in
  `f64`), divides the query the same way and scores their dot product,
  clamped to `[-1, 1]`, so a tiny vector keeps its direction. A dot
  product is summed in `f32` in one fixed order (16 lanes, then the
  lanes added pairwise), and the AVX-512, AVX2 and scalar paths all
  follow it, so a score is the same bits on every path; a unit test
  checks the three against each other.
- **Top-k.** `top_k` scores every row and keeps the best `k` in a heap;
  equal scores (by `==`, so `0.0` and `-0.0` tie) keep insertion order.
  `top_k_batch` answers several queries with the same results, scoring
  sixteen queries at a time against tiles of rows that fit in the
  core's cache, so the rows are read from memory once for every sixteen
  queries rather than once a query.

The tests compare top-k with a full sort of every row's score on random
rows of 7 to 1024 dimensions and on ties, duplicates, scaled copies and
`k` beyond the collection, compare scores with the metric computed in
`f64`, and run queries from eight threads against the single-thread
answer. `analysis/vector-index/examples/index_bench.rs` times one query
at a time and a batch over random unit rows.

### TurboQuant storage

With the `turboquant` feature, `IndexBuilder::build_quantized` stores the
same rows as TurboQuant codes (arXiv 2504.19874) and returns a
`QuantizedIndex` with the same `Space` checks, metadata, `score`, `top_k`
and `top_k_batch`; its scores are estimates of the dot product. The codes
and search kernels are the `turbovec` crate's, pinned to one commit of
`ai-pipestream/turbovec`: a fixed random rotation, Lloyd-Max codes for the
rotated coordinates, a stored length a row that removes the codes'
shrinkage from the estimate, and SIMD scans on AVX-512, AVX2 and NEON,
searched on rayon's pool. The feature brings in turbovec's dependencies
(rayon, memmap2, rand, rand_chacha, statrs, ordered-float); without it
the crate depends on nothing new.

`TurboQuant` sets the bits a coordinate (2 to 4, others refused with
`IndexError::Bits`) and whether to fit turbovec's per-coordinate
calibration (TQ+) first, on up to 1024 rows taken at an even stride so
the same rows give the same calibration (an index of one row is not
calibrated). turbovec codes a vector no longer than `1e-10` as zero, so
the rows are coded after one common scale that makes the longest of
length 1, and each query after division by its own length; scores are
scaled back, so a short `Metric::Dot` row keeps its dot product. A row
shorter than `2e-10` times the longest, a dimension that is not a
positive multiple of 8, or one past 16384, is refused at build with
`IndexError::Quantized`. `to_bytes` returns turbovec's `.tv` image, and a
test pins its bytes for fixed rows. The tests also check the refusals,
top-k against a full sort of the estimated scores, and the paper's error
bound and the estimate's mean at each width.
`analysis/vector-index/examples/quant_bench.rs` measures recall, error
and speed against the exact index; [turboquant.md](turboquant.md) is the
record, with the OpenNLP Java reference on the same rows.

### Late interaction

A `TokenIndex` holds candidates of several token vectors each and scores
a query of several token vectors against them by MaxSim, as ColBERT
does: for each query token the highest dot product over the candidate's
tokens, and the sum of those maxima over the query tokens. It reranks a
shortlist (`rerank`, the candidates given) or the whole index (`top_k`),
best first with ties in insertion order; `score` gives one candidate's
score, and `explain` also gives, for each query token, the candidate
token that won it (the first with the highest dot product), its dot
product and its source: the row it came from and, from a layer, its
bytes, so a hit can be highlighted.

`TokenIndexBuilder::push` adds a candidate from a slice of token vectors.
`push_levels` adds one candidate a unit of `Levels` (chunk, section or
document) from a text's `Vectors<Subwords>`: a unit's tokens are the
subword pieces inside its span (inside each fragment, for a
discontinuous unit; an empty piece counts where it lies before the
fragment's end, or at the end when no non-empty piece follows, so one on
the boundary of two units is the second's and one at the end of the text
the last unit's; an empty unit takes none), in text order, less the items the vectors list as
empty and the pieces a `Prune` rule drops (punctuation, stopwords). The
index records the rule's name and how many tokens it dropped for each
candidate (`pruning`). A unit left with no token is not added. The layers
must be of the index's model and dimension and of one view, and a layer
refused for any unit adds none of its units.

The space checks are the exact index's: a query from another model
(id, revision or manifest hash, so another projection of the same encoder
too), dimension or normalization is refused, and so are zero, non-finite
and overlong token vectors, each error naming the candidate and token
(`Which::Token`) or the query token. A candidate or query of no tokens is
refused with `IndexError::NoTokens`. The dot products are the exact
index's kernel, computed a candidate at a time as a block of query by
candidate scores, four query tokens by four candidate tokens a step on
AVX-512; the maximum of each query token is exact in any order, and the
maxima are added in query order, so a score is the same bits on every
path. `analysis/vector-index/tests/late.rs` checks scores and winning tokens against a naive
double loop bit for bit, reranking against a full sort, the refusals and
eight threads against one; `analysis/vector-index/tests/late_levels.rs` checks the tokens of
each unit, pruning, and the refusal of layers of another model or view.
`analysis/vector-index/examples/late_bench.rs` times a rerank.

## Levels

`Levels` (`turbo:levels`, in `turbo-analysis-layers`) holds three levels
of retrieval units over one view as a tree: the chunks, the sections
they sit in, and the whole document. Each row has its exact span in the
view, a `Unit` that names where it comes from (`Chunk` with its row of
`Chunks`, `Section` with its block of `Structure`, or `Document`), and
its parent. Row 0 is the document, over the whole view, leading and
trailing whitespace included; the other rows follow in text order, a
parent before its children: by start, then sections in structure order,
then chunks. The layer refuses a second document, a unit with no parent
or with a chunk as its parent, and a unit outside its parent's span.

It is a layer of its own rather than more rows of `Chunks`. `Chunks` is
the chunker's output and has one writer; its rows carry the optimizer's
cost, the reason for the cut and a length in model tokens, which a
section or the document does not have. A unit here points into
`Chunks` or `Structure` instead, so nothing of those is copied but the
spans that a positional layer needs.

`LevelBuilder` in `turbo-chunker` reads the view, the sentences, the
chunks and the structure when there is one, and writes the levels:

- The chunks are the level 0 rows of `Chunks`. The rows the chunker
  puts above them (one per top-level section run) are not repeated; the
  sections come from the structure, nested ones included.
- A section is every block of kind `Section`. Its parent is the nearest
  section around it, or the document.
- A chunk belongs to a section when every sentence of it starts in the
  section, the rule the chunker splits its runs by. Its parent is the
  innermost such section, or the document when there is none: text
  before the first heading, after the last section, or a whole document
  with no structure, whose levels are the document and its chunks.
- The chunker never lets a chunk cross a top-level section, but the edge
  of a nested section is only a cost, so a chunk can run across it.
  Such a chunk belongs to the section around both, and shares bytes with
  the nested section beside it: siblings may overlap, a unit never
  leaves its parent.
- A section's span is its block's, widened at the end to the end of the
  last sentence that starts in it, when that sentence runs past the
  block. The span depends on the sentences alone, not on how they were
  chunked, never ends inside a sentence, and holds every chunk that
  belongs to the section and every section nested in it. Its start
  never moves, so every offset is a block's, a sentence's or the
  view's.
- An empty block or an empty chunk keeps its row, with its empty span.

The time is the number of sections times the chunks that start in each,
after three binary searches per section.

Term statistics need no code of their own: `TermCounter<Levels>` counts
the terms of every unit at every level (see "Term statistics and BM25"),
and its df counts the units of all three levels that hold a term.

### Local and contextual vectors

`LevelVectors` in `turbo-analysis-embed` reads the subwords and the
levels and writes two instances of `Vectors<Levels>`, both aligned to
the levels, named by `VectorKind`: `local` and `contextual`
(`VectorKind::key`). It is added to a pipeline with
`LevelVectors::bound`, which gives each output its name; under the
default name for both the pipeline is refused.

- The local vector is pooled from rows that hold the unit's own pieces
  and nothing else: the windows start at its first piece and stop at its
  last.
- The contextual vector is pooled from the rows of the unit's parent
  (its section, a section around it, or the document), which stop at the
  parent's last piece: the late chunking of Günther et al. (2024) within
  the section or document. The document has no parent, so its
  contextual vector is pooled from its own rows and equals its local
  one. So does a unit whose parent holds no piece it does not, such as
  a chunk that is its whole section.

Both go through the window planner and the late pooling of the other
vector steps, one unit at a time: the unit lays out windows over its
own pieces, with the planner bounded by its last piece, its local vector
is pooled from them, and each of its children is placed in them for its
contextual vector. A unit that fits one window is one row. A longer one
is cut into windows of the full width from its first piece, the last
one shorter; its local vector is the mean of its parts weighted by
pieces, and a child is pooled from the window that holds it whole, or
in parts when it crosses from one window to the next. A child of a
long parent therefore sees the context of its own window, not the
whole parent. A child adds no window to its parent's, so each piece is
encoded once for every unit that holds it: once per level, more where
sections nest. A unit with no pieces has the zero vector in both and is
listed in both layers' `empty`.

The tests hold every vector of both kinds to the same unit pooled by
hand, each range run alone in a session of its own, in rows of 64
tokens (the whole text one window) and of 24 (the document and a
section in parts), with and without structure, with empty units, and
twice for the same bits.

What each level and each kind of vector adds to retrieval is not
measured yet: the evaluation harness retrieves with one vector per
chunk, and has no pipeline for levels yet.

## Golden files

An annotator that ports an OpenNLP component is checked against what
OpenNLP 3.x computes on the JVM. The JVM runs once, from a recorded
OpenNLP commit with fixed models, over fixed input texts, and writes
what each annotator produced as a golden file: typed layers, the same
ones the Rust annotator writes. Rust tests read the files and never
need a JVM. `turbo-goldens` reads and writes the files and compares a
layer on a `Document` with its golden layer.

The input texts are `analysis/goldens/data/inputs/<set>/<id>.txt`, the
file's bytes being the text, and each golden is
`analysis/goldens/data/<annotator>/<model>/<set>/<id>.golden`: one per
annotator (`sentdetect`, `tokenize`, `detokenize`, `langdetect`, ...),
run and input. The run directory is the first model file's name without
`.bin` (or `.xml` for a dictionary), with `+` and the names of the run's
options appended when it has any (`+legacy`, `+keep-new-lines`,
`+alphanumeric`, `+english-dictionary`); a run without a model names
itself there (`simple`, `whitespace`, `uax29`, `newline`, or a
normalizer's name). `turbo_goldens::data_dir()`
is that directory, for tests in any crate.

### The format

A golden file is UTF-8 text with LF line ends and one tab between
fields, ending with a line end. Everything is written in a fixed order,
so making a file again gives the same bytes. The first line is
`golden 1`, with a space: the format version, and a reader refuses any
other. Then one header record per line, a keyword and its fields:

- `generator <name> <version>`: what wrote the file.
- `opennlp <commit> <version>`: the OpenNLP commit, as 40 lowercase hex
  digits, and its Maven version.
- `jdk <java.version> <java.vm.version> <java.vendor>`.
- `unicode <major>.<minor>`: the Unicode version the JDK's character
  tables implement (15.0 for JDK 21).
- `input <set>/<id> <bytes> <units> <sha256>`: the input's length in
  UTF-8 bytes and in UTF-16 units, and its SHA-256 in lowercase hex.
- `model <file name> <sha512>`: zero or more, one per model file the
  annotator loaded, in the order loaded.
- `annotator <class> <method>`: the Java class and method that ran,
  such as `opennlp.tools.sentdetect.SentenceDetectorME` and
  `sentPosDetect`.
- `config <key> <value>`: zero or more, one per option the annotator
  was run with, sorted by key, each key once, so two goldens of one class
  run with different options are told apart. Key and value are `str`
  fields, escaped as below. The keys written now are `whitespace`
  (`legacy` for a run in OpenNLP's legacy `WhitespaceMode`; the default
  mode writes no record), `keepNewLines` (`true`),
  `useAlphaNumericOptimization` (`true`) with `alphaNumericPattern` (the
  pattern the tokenizer uses), `splitMarker` (the detokenizer's marker)
  and the normalizers' options.
- `error <class> <message>`: only for a run where the annotator threw
  on the JVM, after the config records and the last line of the file.
  The class is the exception's, such as
  `java.lang.IllegalArgumentException`, and the message is a `str`
  field, what `getMessage()` returned (empty for null). OpenNLP's
  `SimpleTokenizer` with `keepNewLines`, for one, throws on a text that
  starts with a line break, where the Rust port tokenizes it.

The other header fields are written as they are; none holds a tab or a
line end.
A file with an error record has no layer blocks, and the reader refuses
any line after that record. Otherwise one or more layer blocks follow. Each starts with
`layer <id> <anchor> <rows> <name:type>...`, where the anchor is `text`,
`aligned:<layer id>` or `document` and the list names every column with
its type. Then come the rows, one per line, a field per column, and
`end <id>` closes the block. A text anchored layer's rows are in text
order, and an aligned layer has as many rows as the layer it is aligned
to, which comes earlier in the file.

A column is one of three types. A `u32` is decimal, with no sign and no
leading zero (except `0` itself). An `f64` is the 16 lowercase hex
digits of its IEEE 754 bits, a space, and the decimal Java's
`Double.toString` prints, as in `3fe0000000000000 0.5`; the bits are the
value, and the decimal is there for a person reading the file. A `str`
is UTF-8 text in which a tab, a line feed, a carriage return and a
backslash are written `\t`, `\n`, `\r` and `\\`, and every other
character below U+0020 is `\uXXXX`, four lowercase hex digits naming one
UTF-16 unit. Nothing else is escaped. A reader also takes a character
outside the Basic Multilingual Plane written as two `\u` units (a
surrogate pair), and refuses a lone surrogate.

Every `start` and `end` is a UTF-8 byte offset into the input text,
half open, the unit the Rust layers use. The generator computes it from
OpenNLP's UTF-16 span. OpenNLP's `SimpleTokenizer` and `TokenizerME`
classify UTF-16 units, so they can split a character outside the Basic
Multilingual Plane, a surrogate pair, between two tokens, and a byte
offset cannot fall between its halves. For such a row the generator
widens the byte span to the enclosing character (the start rounds down,
the end rounds up), so the two halves of one character are two rows
over the same bytes, and it writes the aligned layer
`turbo:utf16<opennlp:tokens>` (or `turbo:utf16<opennlp:sentences>`)
whenever at least one row cuts a character. That layer has, for every
row of the layer it is aligned to, the JVM's exact span in UTF-16 units
(`start16`, `end16`, half open) and `cut`, 1 when the row starts or ends
inside a surrogate pair and 0 otherwise. Overlapping rows on the text
are allowed as they are for nested entities (a row may not start before
the row before it), so the widened rows need no rule of their own. The
reader checks that a UTF-16 span does not end before it starts or past
the input's length in UTF-16 units.

The layers and their columns, in order:

| Layer id | Anchor | Columns |
|---|---|---|
| `opennlp:sentences` | `text` | `start:u32`, `end:u32` |
| `turbo:confidence<opennlp:sentences>` | `aligned:opennlp:sentences` | `value:f64` |
| `turbo:utf16<opennlp:sentences>` | `aligned:opennlp:sentences` | `start16:u32`, `end16:u32`, `cut:u32` |
| `opennlp:tokens` | `text` | `start:u32`, `end:u32` |
| `turbo:confidence<opennlp:tokens>` | `aligned:opennlp:tokens` | `value:f64` |
| `turbo:utf16<opennlp:tokens>` | `aligned:opennlp:tokens` | `start16:u32`, `end16:u32`, `cut:u32` |
| `opennlp:pos` | `aligned:opennlp:tokens` | `tag:str`, `prob:f64` |
| `opennlp:lemmas` | `aligned:opennlp:tokens` | `lemma:str` |
| `opennlp:chunk-tags` | `aligned:opennlp:tokens` | `tag:str`, `prob:f64` |
| `opennlp:chunks` | `text` | `start:u32`, `end:u32`, `token_start:u32`, `token_end:u32`, `kind:str` |
| `turbo:utf16<opennlp:chunks>` | `aligned:opennlp:chunks` | `start16:u32`, `end16:u32`, `cut:u32` |
| `opennlp:entities` | `text` | `start:u32`, `end:u32`, `token_start:u32`, `token_end:u32`, `kind:str`, `prob:f64` |
| `opennlp:language` | `document` | `code:str`, `confidence:f64` |
| `turbo:categories` | `document` | `category:str`, `score:f64` |
| `turbo:numbers` | `text` | `start:u32`, `end:u32`, `numeral_end:u32`, `negative:u32`, `coefficient:str`, `scale:u32`, `plain:str` |
| `turbo:money` | `text` | `start:u32`, `end:u32`, `coefficient:str`, `negative:u32`, `scale:u32`, `plain:str`, `currency:str` |
| `turbo:quantities` | `text` | `start:u32`, `end:u32`, `coefficient:str`, `negative:u32`, `scale:u32`, `plain:str`, `unit:str` |
| `turbo:temporal` | `text` | `start:u32`, `end:u32`, `value:str`, `granularity:str`, `origin:str` |
| `turbo:document-date` | `document` | `date:str`, `mention:u32` |
| `turbo:pii` | `text` | `start:u32`, `end:u32`, `type:str`, `normalized:str` |
| `turbo:opinions` | `document` | `token_start:u32`, `token_end:u32`, `polarity:f64`, `intensity:f64`, `negated:u32` |
| `turbo:aspects` | `document` | `token_start:u32`, `token_end:u32`, `sentence:str`, `category:str`, `positive:f64`, `negative:f64`, `neutral:f64`, `opinions:str` |
| `turbo:incivility` | `document` | `target_kind:str`, `target_index:u32`, `insult:f64`, `profanity:f64`, `threat:f64`, `identity_attack:f64`, `intolerance:f64`, `severe_toxicity:f64`, `evidence:str` |
| `turbo:pragmatics` | `aligned:opennlp:sentences` | `score:f64`, `pos_start:u32`, `pos_end:u32`, `neg_start:u32`, `neg_end:u32` |
| `turbo:view` | `text` | `start:u32`, `end:u32`, `with:str` |
| `opennlp:subwords` | `text` | `start:u32`, `end:u32`, `id:u32` |
| `turbo:word-types` | `aligned:opennlp:tokens` | `kind:str` |
| `turbo:detokenization` | `aligned:opennlp:tokens` | `operation:str` |
| `turbo:detokenized` | `document` | `text:str`, `marked:str` |
| `turbo:bpe-merges` | `document` | `left:str`, `right:str` |
| `turbo:morphemes` | `aligned:opennlp:tokens` | `unknown:u32`, `features:str` |
| `turbo:events` | `document` | `outcome:str`, `context:str`, `values:str` |
| `turbo:indexer-error` | `document` | `class:str`, `message:str` |
| `turbo:event-checksum` | `document` | `crc32c:u32` |
| `turbo:indexer-outcomes` | `document` | `label:str` |
| `turbo:indexer-predicates` | `document` | `label:str`, `count:u32` |
| `turbo:indexer-events` | `document` | `outcome:u32`, `context:str`, `values:str`, `seen:u32` |
| `turbo:gis-training` | `document` | `iterations:u32`, `message:str` |
| `turbo:gis-iterations` | `document` | `iteration:u32`, `likelihood:f64`, `correct:u32`, `events:u32`, `parameters:str` |
| `turbo:gis-parameters` | `document` | `iteration:u32`, `predicate:u32`, `outcome:u32`, `value:f64` |
| `turbo:gis-model` | `document` | `sha256:str`, `length:u32` |

`opennlp:language` has a row per language the detector ranked, best
first, and `turbo:categories` a row per category of a document
categorizer, best first, ties in category order. A `turbo:view` block is what a normalizer produced: a row per
edit, in text order, where `start..end` is the span of the source text
the edit replaces (empty for an insertion) and `with` is the
replacement (empty for a deletion). The JVM writes adjacent one-to-one
replacements as separate rows, one per character.

An `opennlp:subwords` block is what OpenNLP's `WordpieceEncoder`
returned: a row per piece of the text, in order, with its byte span and
its id in the vocabulary. The `[CLS]` and `[SEP]` that `encode` puts
around the pieces, with empty spans at the two ends of the text, are
not written, as the Rust layer has no special tokens.

A `turbo:bpe-merges` block is the merges OpenNLP's
`BPETokenizerTrainer` learned from the input, a row per merge in the
order made, each side as a string in which a lone surrogate is written
`\uXXXX`; the `opennlp:tokens` block beside it is what `BPETokenizer`
with those merges gave for the same text. No layer type has the id, so
`analysis/bpe-trainer/tests/opennlp.rs` reads these goldens itself.

`turbo:word-types` holds the UAX #29 word type of each token, the
`WordType` name OpenNLP gives it (`ALPHANUMERIC`, `NUMERIC`, `EMOJI`, ...). A
detokenizer golden's `opennlp:tokens` are the lines of its input file, one
token per line, each row the line's bytes without its line feed;
`turbo:detokenization` is the `DetokenizationOperation` name
`DictionaryDetokenizer.detokenize(String[])` gave each token, and the one
`turbo:detokenized` row is what `detokenize(String[], String)` returned
with no split marker (`text`) and with the `splitMarker` config value
(`marked`).

The ids are the layer types' own (`Layer::ID`, and `Layer::OF`
for a generic layer, printed as `turbo:confidence<opennlp:tokens>`), and
the reader refuses an id no layer type has, a second block for one id,
and a block whose anchor or columns, by name and type, are not its
type's. Every error names the line and what was expected there.

The layer types the reader knows are a parameter. `Golden::read`,
`Golden::parse` and `Golden::parse_bytes` check blocks against
`known_shapes()`, the types of `turbo-analysis` and
`turbo-analysis-layers`; `read_with`, `parse_with` and
`parse_bytes_with` take the set. A crate that owns a layer type with a
golden shape (a tokenizer crate's `turbo:word-types`, a word type per
token in one `kind:str` column) implements `GoldenCompare` for it and
reads its goldens with `known_shapes()` plus `shape_of::<ItsLayer>()`.
`segmentation_shapes()` is `known_shapes()` plus the layers the
generator writes whose types live in other crates or in no crate
(`turbo:word-types`, `turbo:detokenization`, `turbo:detokenized`,
`turbo:morphemes`), so the tokenizer goldens can be read without those
crates depending on each other.

A `turbo:morphemes` block is what OpenNLP's `LatticeTokenizer` gave each
morpheme of the `opennlp:tokens` block beside it: `unknown` is 1 for a
morpheme that unknown word handling made, and `features` is its
dictionary entry's feature columns joined with commas. No layer type has
the id; `analysis/tokenize/tests/lattice_golden.rs` compares the rows
with the dictionary.

### Comparing a layer with its golden

```rust
use turbo_goldens::{Golden, Tolerance, assert_matches_golden, data_dir};

// tokenize/<model>/<set>/<id>.golden, for one model and one input
let golden = Golden::read(data_dir().join("tokenize").join(model).join(set).join(format!("{id}.golden")))?;
let text = golden.input_text(data_dir())?; // checks the length and SHA-256
let doc = run_the_tokenizer(text);
assert_matches_golden::<Tokens>(&doc, &golden, Tolerance::exact());
assert_matches_golden::<Confidence<Tokens>>(&doc, &golden, Tolerance::abs(1e-9));
```

The default instance of the layer is compared; `matches_golden_key`
and `assert_matches_golden_key` take a `Key` for a named one, and
`matches_golden` returns the difference as a `Mismatch` instead of
panicking. The document's text must be the golden's input, by length
and SHA-256. When the golden has an error record, the comparison
reports it as `Mismatch::JvmError` with the class and the message
instead of comparing anything, so a test for an input the JVM could not
handle asserts that error on purpose and then checks the Rust output
itself. Spans are compared in source bytes, so a layer on a
normalized view is mapped back through the view's alignment first. Item
by item, the span, then an entity's token range, then the label (a tag
through its tag set, a lemma, a kind, a language code) must be equal,
and a probability must be within the `Tolerance`: `Exact` compares the
bits (so -0.0 and 0.0 differ), and `Abs(d)` takes two finite values at
most `d` apart, or the same infinity, or two NaNs. `Language` holds
every language the detector ranked, best first, and is compared row by
row, so the ranking must match; so is `Categories`, every category of
its one item and its score, ranked best first, ties in category order. POS tags without
probabilities do not match a golden that has them.

`Utf16Spans<L>` (`turbo:utf16<L>`) is compared exactly, row by row: the
UTF-16 span, then the cut flag, the Rust layer's `bool` against the
golden's 0 or 1 (a golden with any other value is reported as one that
cannot be applied). Where the JVM split a character the Rust tokenizer
keeps whole, the byte comparison of the tokens shows that as an
expected deviation the test asserts on purpose: the golden has two rows
over the character's bytes where Rust has one token, so the report is
a count mismatch when the character ends the text (`Smile 😀`: 2 items
in Rust, 3 in the golden) and otherwise the first token after it, which
the golden still has over the character. The UTF-16 layer keeps the
JVM's own answer, the two halves with `cut` 1.

A normalized `View` is compared by its key (`Key::named("nfkc")`):
the source with the golden's edits applied must be the view's text,
byte for byte, and then the golden's rows must be the view's alignment
edits, one for one and in order, so row i is edit i and a report names
the first row that differs. The JVM writes one row per character for
one-to-one replacements, so a normalizer matches it by making one edit
per character too. Only a view made from the source is compared: a
view made from another view is reported as such, since composing a
chain's edits into edits against the source is not built.

The first difference is reported for a person: the layer, the item, the
spans on both sides with the text they cover (for an aligned layer, the
spans of the tokens or sentences it describes), both labels, and both
values in hex bits and decimal. When one side has more items, the
report gives both counts and the first item past the shorter side:

```
turbo:confidence<opennlp:tokens> item 3: the value differs (by 3.000000026176508e-9, tolerance 1e-9)
  rust:   bytes 15..17 "to", value 3fef5c28f75ee07a (0.980000003)
  golden: bytes 15..17 "to", value 3fef5c28f5c28f5c (0.98)
```

The crate writes golden files too (`Golden::write`), byte for byte as
the generator does: its `Double.toString` follows the JDK 19 and later
definition, which picks the nearest of the shortest decimals and breaks
a tie to an even digit. SHA-256 and SHA-512 are written in the crate
from FIPS 180-4.

### Golden layers as test inputs

A port is often written before the annotators in front of it: a POS
tagger before the Rust tokenizer, a name finder before the tagger. To
test it alone, `turbo-goldens` builds a document from the layers the JVM
produced, and the port runs on the JVM's own input layers:

```rust
use turbo_goldens::{GOLDEN_KEY, Golden, Tolerance, data_dir, golden_key, load_document, matches_golden_key};

let golden = Golden::read(path)?;
let text = golden.input_text(data_dir())?;
let mut doc = load_document(&golden, &text)?; // checks the length and SHA-256
// The tagger reads the JVM's tokens and writes the default instance of the tags.
let tagger = Bound::new(Tagger::new(model)).read(1, GOLDEN_KEY);
Analyzer::builder().given(golden_key::<Tokens>()).step(tagger).build()?.run(slice::from_mut(&mut doc));
assert_eq!(matches_golden_key(&doc, &Key::<PosTags>::DEFAULT, &golden, Tolerance::abs(1e-9)), Ok(()));
```

`load_document` adds every layer block of the golden, in file order,
each under the key name `golden` (`GOLDEN_KEY`; `golden_key::<L>()` is
`Key::<L>::named("golden")`). The JVM's instance of a layer is thus a
named instance, and the default key stays free for what the Rust
annotator writes, so the two are never confused and can be compared.
`load_layer` adds one layer under a key the caller picks; loading the
tokens under two names gives two separate instances.

A layer aligned to another (tags, lemmas and confidences) is loaded
onto the instance of its anchor with the same name, and the entities
onto the tokens with the same name, since their `token_start` and
`token_end` columns index those tokens. That instance must be loaded
first; otherwise the load fails with `LoadError::MissingAnchor`. Spans
are source bytes, values are the f64 values as the file holds them, and
the labels of a tag or kind column become a tag set of the distinct
labels in the order they first appear. Loaded tokens get their
sentence column from the sentences loaded under the same name: each
token's sentence is the one whose span contains it, and a token inside
no sentence is refused (`LoadError::OutsideSentences`), since the JVM
makes its tokens inside its sentences. The format writes sentences
before tokens, so `load_document` finds them; without sentences under
that name the column is `None`. `Language` holds every row of the
golden, best first; `Categories` is one item whose categories are the
golden's rows in their order, the first the best.

A block is checked against its layer type's shape before it is loaded,
the same check the comparison makes, and the document then checks the
layer as it checks any other (spans in bounds and on character
boundaries, an aligned layer as long as its anchor). Each refusal is a
`LoadError`: the text is not the golden's input, the golden has no such
block, the block has the wrong anchor or columns, the anchor is not
loaded, an entity's token range is outside the tokens, a block of a type
nothing loads, or the document refused the layer.

A loaded layer's provenance names the annotator `golden:<class>`, after
the Java class in the header, with the OpenNLP version as its version
and the layers it is made on as its reads. Its configuration hash is
taken from the SHA-256 of the annotator, the OpenNLP commit, each model's
file name and SHA-512 and each option, so goldens made with different
models or options have different hashes. It records no model:
`ModelRef` holds one model by SHA-256, and a golden names any number by
SHA-512.

A `turbo:view` block is not loaded. A view golden is only compared with
the view a Rust normalizer makes, so `load_document` on one gives the
source text alone, for the normalizer to run on.

## Training goldens

`analysis/goldens/data/train/<source>-<indexer>-cutoff<n>[-unsorted]/<set>/<id>.golden`
is what OpenNLP's training stack does below the sample types, for the
Rust trainers in `analysis/train` to be compared with. Each golden holds
the events an event stream made from the input (`turbo:events`: the
outcome, the contexts joined with single spaces as `FileEventStream.toLine`
writes them, and the values as `Float.toString` joined the same way, empty
when the event has none; in the contexts and in the predicate labels a
backslash is doubled and a lone surrogate, which the tokenizer's features
hold and a `str` column cannot, is the six characters `\uXXXX`), the CRC-32C `ChecksumEventStream` computed over
them (`turbo:event-checksum`, the manifest's `Training-Eventhash`), and
what the named data indexer made of them with the run's `Cutoff` and
`sort`: the outcome labels by id (`turbo:indexer-outcomes`), the predicate
labels and counts by id (`turbo:indexer-predicates`) and one row per
indexed event (`turbo:indexer-events`: outcome id, predicate ids joined
with single spaces, the values the indexer kept, and how often the event
was seen). The sources are the sentence detector's `SDEventStream`, the
tokenizer's `TokSpanEventStream`, a real valued source made from the
lines of the input and a part-of-speech shaped source with many outcomes
(`analysis/goldens-tool/README.md`, "Runs"); the
`sentdetect` runs on `analysis/goldens/data/inputs/train/sentences.txt` with `TwoPass` and cutoff
1 is the training of the fixture model `en-plain.bin`, and its checksum is
that model's `Training-Eventhash`.

Then the GIS trainer runs on the index with the run's `Iterations`, its
`Threads` (one, or two) and its smoothing (`Smoothing` with
`SmoothingObservation`, or `GaussianSmoothing` with
`GaussianSmoothingSigma`), each a `config` record when the run sets it:
`turbo:gis-training` says how many iterations it ran and what
its progress monitor recorded last (the stop criterion's message or the
finished message; with one outcome the trainer refuses the index, the row
says zero iterations and the exception, and the rest is left out);
`turbo:gis-iterations` has, per iteration, the log likelihood the monitor
was given, the correct and total events, and the SHA-256 of every model
parameter after the iteration (each predicate's parameters in order, each
as the 8 bytes of its IEEE 754 bits, big endian), where the parameters
after iteration k are those of a run of k iterations;
`turbo:gis-parameters` lists every parameter after the first and after the
last iteration with its predicate and outcome ids; and `turbo:gis-model`
is the SHA-256 and length of what `GenericModelWriter` writes for the
final model. An index with no event left makes the indexer throw
`InsufficientTrainingDataException`; the golden then has the events and,
in place of the indexer's layers, one `turbo:indexer-error` row with the
exception's class and message. No layer type has these ids, so
`analysis/train/tests/goldens.rs` reads these goldens itself, and compares the Rust indexers, trainer and writer with every one of them. With two threads the JVM adds the two chunks' log likelihoods in the order the threads finish, which gives the same sum either way, so those goldens are reproducible and compared like the others; no run uses more threads. `analysis/train/tests/roundtrip.rs` trains the fixture models `analysis/maxent/tests/fixtures/models/gis.model` and `en-plain.bin` again and checks that `gis.model`, and the `sent.model` entry and the manifest of `en-plain.bin`, come out byte for byte, and that the `.bin` it writes around them is the fixture `en-plain-rust.bin`, which the generator loads in the JVM as a sentence model (`sentdetect/en-plain-rust`) so that `analysis/sentdetect/tests/golden.rs` can compare the JVM's spans and probabilities with it against the Rust detector's.

The runs of the L-BFGS trainer (`QNTrainer`, `Algorithm=MAXENT_QN`;
directories with `-qn` appended, then `-l1`, `-l2` or `-plain` when only
one cost, or neither, is on) hold the events and the trainer's layers, the
index being that of the GIS run with the same source and indexer settings.
`QNTrainer` reports nothing to a progress monitor; `QNMinimizer` logs one
debug line per iteration and one naming the test that stopped it, which
the generator captures with its own SLF4J provider. `turbo:qn-training`
says how many iterations ran and which test stopped them (`change-rate`,
`gradient-norm`, `step-size`, `fct-eval`, or `iterations` for the limit;
`refused` with the exception for one outcome); `turbo:qn-iterations` has,
per iteration, the objective value at the new point, the function change
rate, the training accuracy and the parameter digest, where the parameters
after iteration k are again those of a run of k iterations (so after the
rescaling the minimizer applies at the end); `turbo:qn-parameters` and
`turbo:qn-model` are as for GIS. The `config` records add `Algorithm`,
`L1Cost`, `L2Cost`, `NumOfUpdates`, `MaxFctEval` and `Threads`. The runs cover one cost, both and neither, several threads, and the `postag` source with many outcomes; `analysis/train/tests/goldens.rs` compares the QN trainer with every one of them the same way. The run `sentdetect-twopass-cutoff1-qn` on `analysis/goldens/data/inputs/train/sentences.txt` is the JVM's training of the fixture `analysis/goldens/data/fixtures/en-plain-qn-rust.bin`, the `.bin` the Rust QN trainer writes for those events: `analysis/train/tests/roundtrip.rs` checks the model bytes against that golden and the `.bin` against the fixture, and the generator loads the fixture in the JVM as a sentence model (`sentdetect/en-plain-qn-rust`) for `analysis/sentdetect/tests/golden.rs` to compare with the Rust detector.

The runs of the perceptron trainers (`PerceptronTrainGolden`;
`Algorithm=PERCEPTRON`, directories with `-perceptron` appended, then
`-raw`, `-skipped` and `-decrease` for no averaging, skipped averaging and
a step size decrease; `Algorithm=PERCEPTRON_SEQUENCE`, directories with
`-perceptron-sequence` appended, then `-raw`) hold the events and the
trainer's layers as well. `turbo:perceptron-training` says how many
iterations ran and what the monitor recorded last (empty for the sequence
trainer, which has no monitor; with one outcome the event trainer refuses
the index, and the sequence trainer fails after its iterations when the
cutoff dropped an event: zero iterations and the exception, and the rest
left out); `turbo:perceptron-monitor` has, per iteration the monitor heard
of (1 to 9 and every tenth), the correct and total events and the
accuracy; `turbo:perceptron-iterations` has the parameter digest after
every iteration, from a run of that many iterations; `turbo:perceptron-parameters`
is as for GIS; and `turbo:perceptron-model` is the written model's SHA-256
and length, or an empty digest, zero length and the exception's class in
`error` when `PerceptronModelWriter` fails on a model without a parameter
left. The sequence runs cut the source's events into sequences of five
consecutive events and tag each event by its own context. The `config`
records add `Algorithm`, `Iterations`, `UseAverage` and, for the event
trainer, `UseSkippedAveraging`, `StepSizeDecrease` and `Tolerance`.
Both trainers also run on the `postag` source with its many outcomes.
`analysis/train/tests/goldens.rs` compares both trainers with every one of them, and
`analysis/train/tests/roundtrip.rs` trains `analysis/maxent/tests/fixtures/models/qn.model` and `analysis/maxent/tests/fixtures/models/perceptron.model` again byte for byte. As for QN, the run `sentdetect-twopass-cutoff1-unsorted-perceptron` on `analysis/goldens/data/inputs/train/sentences.txt` is the JVM's training of the fixture `analysis/goldens/data/fixtures/en-plain-perceptron-rust.bin`, which `analysis/train/tests/roundtrip.rs` writes and checks and the generator loads as a sentence model (`sentdetect/en-plain-perceptron-rust`) for `analysis/sentdetect/tests/golden.rs`.

`analysis/goldens/data/train-events/<run>/<set>/<id>.golden` (`EventGolden`
and `TaggerEventGolden`)
extends the event side to the factory options `TrainGolden` leaves at
their defaults and to the tagger and the lemmatizer, for
`analysis/train-components` to be compared with.
`EventGolden`'s five runs are the sentence detector with the English abbreviation
dictionary and the Portuguese end characters the language supplies
(`sentdetect-dictionary-pt`), with Thai's space and line feed
(`sentdetect-th`), and over CoNLL-U input with ten sentences a sample
and explicit end characters (`sentdetect-conllu`); and the tokenizer with
the dictionary, the alphanumeric optimization and the German pattern
(`tokenize-dictionary-alnum-de`) and over CoNLL-U with an explicit
pattern (`tokenize-conllu`). The `config` records name the run, the
source, the language, the format, the dictionary fixture and every
option given. Each golden has the events and the checksum as the `train`
goldens do, `turbo:stream-errors` (one row per exception the sample
stream threw: the events made before it, its class and its message; the
generator reads on, as the format goldens do), `turbo:manifest` (the
manifest entries, sorted, without `Timestamp`, of a model file built
from the factory around a one-predicate model), `turbo:package-entries`
(that file's zip entries in order with their lengths), with a dictionary
`turbo:dictionary` (its `abbreviations.dictionary` entry as text), and
on the `train` input of the dictionary run `turbo:dictionary-cases`:
`Dictionary.serialize` of dictionaries built in the generator (the
English fixture, tokens with the characters the serializer escapes, an
empty one, a case-sensitive one, forty entries, entries of several
tokens, entries that differ only in case). `TaggerEventGolden`'s six runs
are `POSSampleEventStream` over the context generator a
`POSTaggerFactory` with no descriptor makes (the default feature
generators), over word and tag lines without and with the tag dictionary
fixture `tags.tagdict` (`postag-wordtag`, `postag-wordtag-tagdict`) and
over CoNLL-U in the universal and the language tag set
(`postag-conllu-u`, `postag-conllu-x`), and `LemmaSampleEventStream` over
word, tag and lemma lines and over CoNLL-U (`lemma-text`,
`lemma-conllu-u`), on the `formats` inputs and the `tagged` inputs the
Rust fixtures were trained from. Their goldens have the same layers, the
model file being a `POSModel` or `LemmatizerModel` around a placeholder
model whose outcomes are the dictionary's tags, with `turbo:tagdict`
(the `tags.tagdict` entry) in place of `turbo:dictionary`, and on the
`tagged` inputs of the dictionary run `turbo:tagdict-cases`:
`POSDictionary.serialize` of dictionaries built in the generator (the
fixture as read, the same word in three cases in a case-sensitive and a
case-insensitive dictionary, words and tags with the characters the
serializer escapes and an entry with no tags, an empty one, forty
entries, words whose lowercase is not ASCII).

`analysis/goldens/data/train-liblinear/<run>/<set>/<id>.golden`
(`LiblinearTrainGolden`) is the liblinear trainer of the OpenNLP addons on
the same sources' events, with each logistic regression solver: the
events, then the exception or the text model, the serialized model's
SHA-256 and the probabilities `eval` gives. [liblinear.md](liblinear.md)
lists the runs, and `analysis/train/tests/liblinear_goldens.rs` compares
with them.

## Training the components

`analysis/train-components` is `SentenceDetectorME.train`,
`TokenizerME.train`, `POSTaggerME.train` and `LemmatizerME.train`: it
takes the samples `analysis/formats` reads (native one-per-line files or
CoNLL-U), makes the events `SDEventStream`, `TokSpanEventStream`,
`POSSampleEventStream` and `LemmaSampleEventStream` make, with the
factory options a training tool can set (`useTokenEnd`, end characters,
abbreviation dictionary; alphanumeric optimization and pattern; feature
generator descriptor and tag dictionary), using the context generators
the annotators infer with (`turbo-sentdetect`, `turbo-tokenize`,
`turbo-sequence`, `turbo-lemmatizer`), trains them with
`analysis/train`'s GIS trainer and writes the `.bin` file the JVM would
write. It is a crate of its own because `turbo-formats` depends on
`turbo-tokenize`, so a training module inside the annotator crates would
close a cycle, and one inside `turbo-formats` would make the sample
readers depend on the trainer. Its README lists the OpenNLP behaviour
that is surprising: the factories resolve a missing end character set or
pattern to the language's before the context generator or the manifest
sees them, while an empty end character array makes the JVM train with
the language's context generator (Thai's `SentenceContextGenerator`) and
record `eosCharacters=` empty, which the crate refuses; the dictionary's entries are written in a Java `HashSet`'s
order and the zip entries in a Java `HashMap`'s order, both simulated;
`new Dictionary(InputStream)` collapses case variants before it reads
`case_sensitive`; the JDK's XML serializer writes characters outside the
Basic Multilingual Plane, U+007F and the C1 controls as decimal character
references and refuses the other C0 controls; an empty training token asks the tokenizer's
context generator for index 0; a `POSTaggerFactory` with no descriptor
loads the default one when its context generator is first made, so every
trained tagger model carries `generator.featuregen`; `POSDictionary`
lowercases a case-insensitive dictionary into a second `HashMap` in the
first one's order and `serialize` writes that order, so a trained model's
`tags.tagdict` is not the file it was read from, and the dictionary comes
before the manifest among the zip entries; `POSModel` refuses a
dictionary tag the model has no outcome for; and a lemma class is an edit
script over UTF-16 units, which can hold half of a surrogate pair for a
token or lemma outside the Basic Multilingual Plane, so an outcome label
is a `JavaString` from the event to the model file, as it is on the JVM.

`analysis/train-components/tests/goldens.rs` compares the events with
every `train` and `train-events` golden, and the stream errors, the
manifest, the zip entries and the dictionaries with the `train-events`
goldens. `analysis/train-components/tests/fixtures.rs` trains the
fixture models `en-plain.bin`, `en-abbrev.bin`,
`tokenizer-abbreviations.bin` and `pos-fixture.bin` again from their
training texts and checks the manifest, the dictionaries, the descriptor
and the model entries byte for byte, and trains the fixtures the Rust
trainer wrote, `pos-fixture-rust.bin` (with the tag dictionary) and
`lemma-fixture-rust.bin`, from `analysis/goldens/data/inputs/tagged/`,
checking they are the checked in files; the generator loads those two in
the JVM (`postag/pos-fixture-rust`, `lemmatize/lemma-fixture-rust`) so
that `analysis/postag/tests/golden.rs` and
`analysis/lemmatizer/tests/golden.rs` compare the JVM's tags, lemmas and
probabilities with them against the Rust annotators'.

The `chunk` module is `ChunkerME.train` with `ChunkerFactory`: the
events of `ChunkerEventStream` over the default context generator of
`turbo-sequence`, which `turbo-phrase-chunker` infers with, trained by
`MAXENT` (the default), `MAXENT_QN`, `PERCEPTRON` or `NAIVEBAYES`, in a
model file with `chunker.model` before `manifest.properties`, as
the JVM's artifact map orders them. `ChunkerME.train` refuses the
sequence perceptron, an event model sequence trainer, with
`IllegalArgumentException`, and so does the port. The goldens of
`ChunkerTrainGolden` (`train-events/chunker-*`) are `ChunkerME.train`
itself over the chunk inputs with OpenNLP's default parameters, the
perceptron, `MAXENT_QN` and the sequence perceptron;
`analysis/train-components/tests/chunker.rs` compares the events, the
stream errors, what training threw, and the trained model's outcomes,
predicates, every parameter (the JVM's double, bit for bit), the model
entry's length and SHA-256, the manifest and the zip entries. The fixture
`chunker-fixture.bin` is trained again byte for byte, and
`chunker-fixture-rust.bin`, trained from
`analysis/goldens/data/inputs/tagged/chunks.txt`, is loaded and run in
the JVM (`chunk/chunker-fixture-rust`), where its tags, probabilities
and decision points are those `turbo-phrase-chunker` gives.

The language detector is trained as `LanguageDetectorME.train` does (the
`langdetect` module, #429). `LanguageTrainGolden` writes
`analysis/goldens/data/train-langdetect/<run>/<set>/<id>.golden`: the
events, the index, the GIS training iteration by iteration, and the
manifest and zip entries of the model file, with cutoff 5 and 1, on
`analysis/goldens/data/inputs/langdetect/samples.txt`,
`analysis/goldens/data/inputs/langdetect/tree-bucket.txt` (where
training's n-gram order differs from prediction's) and the language
format input. `analysis/train-components/tests/langdetect.rs`
compares all of it, the weights bit for bit (as for any event in the
JVM's predicate order; one with equal-hash n-grams in a tree bucket can
differ in the last bits, and the goldens hold none), and
`analysis/train-components/tests/fixtures.rs`
checks that `langdetect-fixture-rust.bin` is the file the Rust trainer
writes from the samples; the generator loads it in the JVM
(`langdetect/langdetect-fixture-rust`) for
`analysis/langdetect/tests/fixture.rs` (docs/langdetect.md, "Training").

The sentiment detector is trained as `SentimentME.train` does (the
`sentiment` module, #215). `SentimentGolden` writes
`analysis/goldens/data/sentiment/<run>/sentiment-train/feedback.golden`
(the events, the digest of the classifier entry, the manifest and the zip
entries) and `analysis/goldens/data/sentiment/<run>/sentiment/<id>.golden`
(each document's tokens, context and `turbo:categories`), for the token
and emoji factories with GIS, L-BFGS and the perceptron.
`analysis/train-components/tests/sentiment.rs` trains each model, compares
it, and scores the documents with the file it wrote; docs/sentiment.md
describes the runs.

The `parser` module is the training side of OpenNLP's bottom-up
constituency parsers: the events of the chunking and tree-insertion
`ParserEventStream`s (the context generators are `turbo-parser`'s, the
ones the parsers decode with), `chunking.Parser.train` and
`treeinsert.Parser.train` (the build, check and attach models with the
trainer the parameters name, the tagger with `train_pos_model`, the
chunker with `ParserChunkerFactory`), the `ParserModel` file with each
model's training report in the manifest, and `ParserEvaluator` and
`ParserCrossValidator`. `turbo-parser` reads the model and parses; the
training side is here because `turbo-formats` reads treebanks into
`turbo_parser::Parse` trees. The `parse-treebank` goldens of
`ParserTrainGolden` are compared in two places that share
`analysis/parser/tests/goldens/treebank_runs.rs`:
`analysis/train-components/tests/parser_goldens.rs` compares the event
runs and the model runs in full (the model file, the parses of the JVM's
model and the evaluation) and trains both models from the JVM's corpus,
checking every entry but the timestamps against the JVM's file;
`analysis/parser/tests/goldens/treebank.rs` compares the dictionary and
generator runs and the parses of the JVM's models.

The `ud` module holds the recipes of the eighteen published
`opennlp-{en,de,es,fr,pt}-ud-{ewt,gsd}-{sentence,tokens,pos,lemmas}-1.3-2.5.4.bin`
models as typed data: the treebank (`UniversalDependencies/UD_*` at
release 2.16, its `train` file, size and digests, CC BY-SA 4.0), the
component, the factory the 2.5.4 trainer tool builds and the training
parameters of `ud-train.conf` in `apache/opennlp-models` (`MAXENT`,
cutoff 5, four threads) with OpenNLP 2.5.4's whitespace definition, the
universal tag set and the publisher's time zone. The treebanks are CC
BY-SA, so the recipes fetch them through `turbo-zoo` by digest after the
licence is accepted and never check them in; `ud::datasets()` is their
zoo index, and a published model not under `TURBO_GOLDENS_MODELS` is
fetched through the zoo's built-in index. The reproduction is
`analysis/train-components/tests/reproduce.rs`, ignored by default:

```text
cargo run -p turbo-zoo -- accept CC-BY-SA-4.0 --yes
cargo run -p turbo-goldens-tool -- fetch-models <dir>
TURBO_GOLDENS_MODELS=<dir> cargo test -p turbo-train-components --release --test reproduce -- --ignored --nocapture
```

It reports per model whether the event checksum, the manifest and the
zip entries agree, and how the model entry compares: byte for byte, or
the same predicates and patterns with the number of differing parameters
and the largest difference. Two things were found. The published models
were trained with 300 iterations, not the 150 the committed
`ud-train.conf` says (the English sentence model's parameters after
every iteration of a longer run were compared with the file; 300 agrees,
299 and 301 do not), so the recipes use 300. And with 300 iterations the
events, manifests, predicates and patterns of all eighteen models come out
exactly while the parameters differ in their last bits, by 2.2e-15 to
9.8e-15 at most for the sentence and token models, by 1.6e-14 to 1.4e-13
for the taggers and by 8.0e-15 to 1.8e-14 for the lemmatizers, depending
on the model. For the eight sentence and token models the JVM of the
goldens container, running the fork's trainer on the same events, writes
the parameters this port writes and shows the same residual against the
published file (checked in #394), so for them the residual is not between
the port and the fork's trainer; the taggers and lemmatizers were not run
through the fork's trainer in the container, only compared with the
published files. Its cause is not known (`GISTrainer` uses `StrictMath` and
merges its threads' sums in a fixed order, so platform `Math` intrinsics
and thread timing do not explain it), and the next experiment is to run
the OpenNLP 2.5.4 command line itself in the container on the same
treebank. The test
asserts agreement to within `1e-12` and says which models match byte for
byte (none does); a model that did not come out this close would fail it.

## Generating golden files

The goldens in `analysis/goldens/data` are made by `turbo-goldens`
(crate `turbo-goldens-tool` in `analysis/goldens-tool`). It builds a
container from a recorded `eclipse-temurin:21-jdk` digest with OpenNLP
built from a recorded commit of the fork, runs the Java generator in it
without a network over the inputs the models (each model checked
against its published SHA-512) and the fixtures (checked against the
SHA-512 `recipe.rs` lists), and writes
`<annotator>/<model>/<set>/<id>.golden` for every run and every input
of the sets it takes. `turbo-goldens verify` regenerates them into a scratch directory and
fails on any file whose bytes differ. The checked in goldens were made
by `turbo-goldens generate` in that container, and `verify` reproduces
them byte for byte.
`analysis/goldens-tool/README.md` has the commands and lists every run.

The generator writes these layers, in this order within a golden:

| Annotator | Layers |
|---|---|
| `sentdetect` (`SentenceDetectorME.sentPosDetect`) | `opennlp:sentences`, `turbo:utf16<opennlp:sentences>` when a sentence cuts a character, `turbo:confidence<opennlp:sentences>` |
| `tokenize` (`TokenizerME.tokenizePos`) | `opennlp:tokens`, `turbo:utf16<opennlp:tokens>` when a token cuts a character, `turbo:confidence<opennlp:tokens>` |
| `langdetect` (`LanguageDetectorME.predictLanguages`) | `opennlp:language` |
| `postag` (`POSTaggerME.tag` on each sentence `SentenceDetectorME` and `TokenizerME` cut) | `opennlp:sentences` and `opennlp:tokens` as above, then `opennlp:pos` aligned to the tokens (`tag:str`, `prob:f64` from `probs()`) |
| `lemmatize` (`LemmatizerME.lemmatize` on each sentence, tagged by `POSTaggerME`) | `opennlp:sentences`, `opennlp:tokens` and `opennlp:pos` as above, then `opennlp:lemmas` aligned to the tokens (`lemma:str`) and `turbo:confidence<opennlp:lemmas>` (`value:f64` from `probs()`) |
| `lemmatize-dictionary` (`DictionaryLemmatizer.lemmatize`) | the same without the confidence; `O` where the dictionary has no entry |
| `chunk` (`ChunkerME.chunk` on each sentence, tagged by `POSTaggerME` asked for Penn Treebank tags) | `opennlp:sentences`, `opennlp:tokens` and `opennlp:pos` as above, then `opennlp:chunk-tags` aligned to the tokens (`tag:str`, `prob:f64` from `probs()`) and `opennlp:chunks`, the phrases `ChunkSample.phrasesAsSpanList` reads off those tags (what `chunkAsSpans` returns): the span from the first token's start to the last token's end, `token_start:u32`, `token_end:u32` and `kind:str`, with `turbo:utf16<opennlp:chunks>` when a phrase's token cuts a character |
| `sentdetect` (`NewlineSentenceDetector.sentPosDetect`) | `opennlp:sentences` |
| `tokenize` (`SimpleTokenizer`, `WhitespaceTokenizer` `tokenizePos`) | `opennlp:tokens`, `turbo:utf16<opennlp:tokens>` when a token cuts a character |
| `tokenize` (`WordTokenizer.tokenizeTyped`, UAX #29) | `opennlp:tokens`, `turbo:utf16<opennlp:tokens>` when a token cuts a character, `turbo:word-types` |
| `detokenize` (`DictionaryDetokenizer.detokenize`) | `opennlp:tokens`, `turbo:detokenization`, `turbo:detokenized` |
| `normalize` (each normalizer) | `turbo:view` |
| `numbers` (`NumberScan.parse`, per notation) | `turbo:numbers` |
| `money` (`CursorMoneyExtractor.extract` in `MoneyAnnotator`, per notation) | `turbo:money` |
| `money-convert` (`EcbFxRates.convert` in `MoneyConversionAnnotator`, per notation) | `turbo:money` |
| `quantity` (`CursorQuantityExtractor.extract` in `QuantityAnnotator`, per notation) | `turbo:quantities` |
| `temporal` (`CursorTemporalExtractor.extract` in `TemporalAnnotator`, then `DocumentDateAnnotator`, per reference mode) | `turbo:temporal`, `turbo:document-date` |
| `pii` (the `PiiPacks` extractor in `PiiAnnotator`, per pack) | `turbo:pii` |
| `pii-rewrite` (`Masker`, `Pseudonymizer`, `HmacTokenizer` and `PiiAuditReport` over those mentions, per pack) | `turbo:view`, `turbo:pii-mask-all`, `turbo:pii-rewrites`, `turbo:pii-labels`, `turbo:pii-audit`, `turbo:pii-audit-report` |
| `subwords` (`WordpieceEncoder.encode`) | `opennlp:subwords` |
| `unigram` (`UnigramSegmenter.tokenizePos`) | `opennlp:tokens` |
| `lattice` (`LatticeTokenizer.analyze`) | `opennlp:tokens`, `turbo:morphemes` |
| `train-liblinear` (`LiblinearTrainer.train`, `LiblinearModel.serialize`, `eval`) | `turbo:events`, then `turbo:liblinear-error`, or `turbo:liblinear-text`, `turbo:liblinear-model`, `turbo:liblinear-scores`, `turbo:liblinear-best` ([liblinear.md](liblinear.md)) |
| `svm-doccat` (`DocumentCategorizerSVM.train` on a corpus) | `turbo:svm-model`, `turbo:svm-sidecar` |
| `svm-doccat` (`DocumentCategorizerSVM.categorize` on a document) | `opennlp:tokens`, `turbo:svm-vector`, `turbo:svm-decisions`, `turbo:svm-vote`, `turbo:categories` |
| `sentiment` (`SentimentME.train` on a corpus) | `turbo:sentiment-events`, `turbo:sentiment-predicates`, `turbo:sentiment-model`, `turbo:manifest`, `turbo:package-entries` |
| `sentiment` (`SentimentME.probabilities` on a document) | `opennlp:tokens`, `turbo:sentiment-context`, `turbo:categories` |
| `corpora` (the Moses, LetsMT, NKJP and Irish Sentence Bank sentence streams) | `turbo:samples`, `turbo:read-errors`, `turbo:sample-texts`, `turbo:sample-spans` |
| `corpora` (`IrishSentenceBankTokenSampleStream`) | `turbo:samples`, `turbo:read-errors`, `turbo:sample-texts`, `turbo:sample-spans` |
| `corpora` (`LeipzigLanguageSampleStream`, shuffled and skipped) | `turbo:samples`, `turbo:read-errors`, `turbo:sample-labels`, `turbo:sample-texts` |
| `corpora` (`TwentyNewsgroupSampleStream`) | `turbo:samples`, `turbo:read-errors`, `turbo:listing`, `turbo:files`, `turbo:sample-labels`, `turbo:sample-tokens` |
| `ad` (`ADNameSampleStream`, and `NameToTokenSampleStream` over it) | `turbo:samples`, `turbo:read-errors`, `turbo:sample-tokens`, `turbo:names`; the token run `turbo:samples`, `turbo:read-errors`, `turbo:sample-texts`, `turbo:sample-spans` |
| `ad` (`ADPOSSampleStream`) | `turbo:samples`, `turbo:read-errors`, `turbo:sample-tokens`, `turbo:sample-tags` |
| `ad` (`ADChunkSampleStream`) | `turbo:samples`, `turbo:read-errors`, `turbo:sample-tokens`, `turbo:sample-tags`, `turbo:sample-chunks`, `turbo:phrases`, `turbo:phrase-errors` |
| `ad` (`ADSentenceSampleStream`) | `turbo:samples`, `turbo:read-errors`, `turbo:sample-texts`, `turbo:sample-spans` |
| `ontonotes` (`OntoNotesNameSampleStream`) | `turbo:samples`, `turbo:read-errors`, `turbo:sample-tokens`, `turbo:names`, `turbo:listing` |
| `ontonotes` (`OntoNotesParseSampleStream`) | `turbo:samples`, `turbo:read-errors`, `turbo:parse-texts`, `turbo:parse-nodes`, `turbo:pos-samples`, `turbo:chunk-samples`, `turbo:listing` |
| `ontonotes` (`ParseToPOSSampleStream`) | `turbo:samples`, `turbo:read-errors`, `turbo:sample-tokens`, `turbo:sample-tags`, `turbo:listing` |
| `ontonotes` (`DocumentToLineStream`) | `turbo:samples`, `turbo:read-errors`, `turbo:lines`, `turbo:listing` |
| `frenchtreebank` (`ConstitParseSampleStream`) | `turbo:samples`, `turbo:read-errors`, `turbo:parse-texts`, `turbo:parse-nodes`, `turbo:pos-samples`, `turbo:chunk-samples`, `turbo:listing` |

`numbers/<notation>/num/<id>.golden` (`latin-us`, `latin-eu`) holds
`turbo:numbers`, anchored on the text, with one row per number mention:
`start:u32` and `end:u32` (UTF-8 byte offsets, from an optional `-` to the
end of a `k`, `m`, `b`, `bn` or `thousand` to `trillion` scale marker),
`numeral_end:u32` (the end of the digits, before the scale marker),
`negative:u32` (1 when the mention starts with `-`), `coefficient:str` (the
unscaled value without sign), `scale:u32` (digits after the decimal point)
and `plain:str` (`toPlainString()`, for a human reading the file). The value
is `(-1)^negative * coefficient / 10^scale`. The `annotator` line names
`opennlp.tools.extraction.NumberScan` and `parse`, and a `config notation`
record gives `LATIN_US` or `LATIN_EU`. The scan is the mention loop of
`CursorQuantityExtractor` of the `numeric` module of `opennlp-addons`
(recorded in `analysis/goldens-tool/src/recipe/mod.rs`) without the money
exclusion and without units; there is no model and no dump.

`money/<notation>/money/<id>.golden` and
`money-convert/<notation>/money/<id>.golden` both hold
`turbo:money`, one row per mention, with the columns `start`,
`end` (UTF-8 byte offsets, from the `-`, symbol or first digit to the end
of the mention), `coefficient`, `negative`, `scale`, `plain` (as for
`turbo:numbers`) and `currency` (the ISO 4217 code). Both have `config
notation`; the conversion goldens also have `config target` (`EUR`),
`config as_of` (`2026-03-15`) and `config snapshot`
(`rates/ecb-snapshot.csv`, relative to `analysis/goldens/data`). A mention
without a rate has no row in a conversion golden. The generator reads
the snapshot from the `rates` directory beside the inputs; `turbo-goldens
generate` mounts it at `/rates`. See [money.md](money.md#golden-files).

`quantity/<notation>/quantity/<id>.golden` holds `turbo:quantities`, one
row per mention, with the columns `start`, `end` (UTF-8 byte offsets, from
the `-` or first digit to the end of the unit), `coefficient`, `negative`,
`scale`, `plain` (as for `turbo:money`) and `unit` (the unit token, `%` for
a percentage), and a `config notation` record. See
[quantities.md](quantities.md#golden-files).

`temporal/<mode>/temporal/<id>.golden` (`dateline`, `fixed`) holds
`turbo:temporal`, one row per mention, with the columns `start`, `end`
(UTF-8 byte offsets; for a timestamp, its date only), `value`
(`TemporalExpression.value()`, such as `2026-07-14`, `2026-W29`,
`2026-07`, `2024-Q3` or `2026`), `granularity` and `origin` (the enum
constants' names), and `turbo:document-date`, anchored on the document,
with no row or one: `date` and `mention`, the index of the mention it
came from. The `config reference` record is `dateline` (the reference date
of relative mentions is the text's first absolute day) or `2026-03-15`.
See [temporal.md](temporal.md#golden-files).

`pii/<pack>/pii/<id>.golden` (`payment`, `contact`, `network`,
`secrets`, `crypto`, `us-identity`, `eu-identity`, `ca-identity`,
`device`, `all-structured`) holds `turbo:pii`, one row per mention in
`PiiAnnotator`'s order, with the columns `start`, `end` (UTF-8 byte
offsets), `type` (`PiiMention.type()`) and `normalized`. The `annotator`
line names `opennlp.tools.pii.PiiPacks` and the factory method, and the
`config pack` record names the pack. See [pii.md](pii.md#golden-files).

`pii-rewrite/<pack>/pii/<id>.golden` holds what OpenNLP makes of those
mentions under a fixed test key (`config key`): the masked text's edits as
`turbo:view` (one row per changed character, a masked character written as
one mask byte per UTF-8 byte), the text with every mention masked, the
numbered and keyed rewrites or the exception each threw, their labels, and
the audit report by type and as text. `analysis/goldens-tool/README.md`
lists the columns. See [pii.md](pii.md#golden-files).

When the OpenNLP annotator throws on an input, the golden holds the
header records, the `config` records and then, as its last line,
`error <class> <message>`: the exception's fully qualified class and
what `getMessage()` returned (empty for null), escaped as a `str`
field. It has no layer blocks, and that input has no decision point
dump. Only the call into the annotator is caught this way; any other
exception (an input or model that cannot be read, one of the
generator's own checks) stops the generator. Of the checked in goldens,
only `SimpleTokenizer` with `keepNewLines` throws, on the whitespace only
input.

`TokenizerME` decides at every UTF-16 unit inside a whitespace token, so
it can split a character outside the Basic Multilingual Plane between
two tokens (`'😀.` gives `'`, the high surrogate, the low surrogate and
`.`). A byte offset cannot fall between the two halves, so the row of a
span that starts or ends inside a surrogate pair holds the whole
character (the start rounds down to the character's first byte, the
end up to the byte after it), and the two halves are two rows over the
same bytes. Right after the layer, the generator then writes
`turbo:utf16<opennlp:tokens>` (or `turbo:utf16<opennlp:sentences>`),
aligned to it, with one row per row: `start16:u32` and `end16:u32`, the
JVM's exact half open span in UTF-16 units, and `cut:u32`, 1 for a row
that cut a character and 0 otherwise. The layer is written only when at
least one row cuts a character, so every other golden is as it was.
`analysis/goldens/data/inputs/en/emoji-split.txt` and the
`inputs/surrogate` set are such inputs for the tokenizer models.

## Decision point dumps

Beside each golden of `sentdetect`, `tokenize`, `postag`, `lemmatize`, `chunk`
and `langdetect` that names a model and has no error record,
`<annotator>/<model>/<set>/<id>.dump` records every
decision the annotator asked its maxent model for on that input: the
context strings the context generator produced and the probability of
each outcome that `eval` returned. A Rust context generator and the
Rust maxent and perceptron evaluation can be checked against it one
decision at a time, before a whole annotator's golden lines up. The
generator writes them for `SentenceDetectorME`, `TokenizerME`,
`POSTaggerME`, `LemmatizerME`, `ChunkerME` and `LanguageDetectorME`. `turbo_goldens::Dump::read` parses one into its
header (the golden's `Header` type) and a `Point` per decision, with the
contexts as UTF-16 units, since a context can hold half of a surrogate
pair, and the outcome labels as `JavaString`s, since a lemma class can.

### The format

UTF-8 text, LF line ends, fields separated by one tab, and a line end at
the end of the file, as in a golden. The cells have the golden format's
types: `u32` (decimal, no sign, no leading zeros), `f64` (the 16
lowercase hex digits of the raw IEEE 754 bits, a space, then
`Double.toString`, for example `3fe0000000000000 0.5`) and `str`.

```
dump 1
generator	turbo-goldens-java	1
opennlp	<40 hex commit>	<maven version>
jdk	<java.version>	<java.vm.version>	<java.vendor>
unicode	<Unicode version>
input	<set>/<id>	<length in UTF-8 bytes>	<length in UTF-16 units>	<sha256 hex>
model	<file name>	<sha512 hex>
annotator	<fully qualified Java class>	<method name>
point	<index u32>	<position u32>	<contexts u32>	<outcomes u32>	<unit u32>	<values>
context	<str>
...
outcome	<label str>	<f64>
...
acceptable	<0 or 1>                    (sentence detector only, see below)
end	<index u32>
point	...
```

- The first line is exactly `dump 1`; the `1` is the format version.
- The header records are the ones the golden at the same path has, with
  the same values: the same `generator`, `opennlp`, `jdk`, `unicode` and
  `input` lines, one `model` line per model loaded, the same
  `annotator` class and method, and the same `config` records.
- Then one block per `eval` call, in the order the annotator made them,
  numbered from 0. A dump with no decision points (a text without an
  end of sentence character) is its header alone.
- `position` is the UTF-8 byte offset into the input that the decision is
  about, or 0 for a decision about the whole document. For the sentence
  detector it is the offset of the end of sentence character the scanner
  found (the `position` argument of `SDContextGenerator.getContext`); for
  the tokenizer it is the offset of the character a split would go before,
  inside a whitespace separated token; for the part-of-speech tagger it is
  the offset of the start of the token the context is about (the beam
  search asks about a token once per path it extends, so a token has as
  many points as the beam kept paths, in the order the search made them),
  and the same for the maxent lemmatizer and the chunker; for the language
  detector it is
  0.
- `unit` is 0 when the UTF-16 index OpenNLP decided at starts a code
  point, and 1 when it lies between the two halves of a surrogate pair.
  The tokenizer walks UTF-16 units, so it asks about such positions inside
  an emoji or any other character outside the Basic Multilingual Plane;
  `position` is then the byte offset of that pair's code point. The two
  fields together name every decision exactly without a UTF-16 offset in
  the file.
- `contexts` lines follow, `context	<str>`, in the order the model received
  them. `values` is `none` when the annotator called `eval(String[])`, and
  `given` when it called `eval(String[], float[])`; each context line then
  has a third field, the context's feature value widened to `f64`.
- `outcomes` lines follow, `outcome	<label>	<f64>`, one per outcome of the
  model in its outcome index order (`getOutcome(0)`, `getOutcome(1)`,
  ...), with the probability `eval` returned for it.
- `acceptable	<0|1>` follows the outcomes only in a sentence detector
  dump, and only for a point whose best outcome is the split outcome
  `s`: it is what `SentenceDetectorME.isAcceptableBreak` returned for
  that candidate, 0 when the abbreviation dictionary refused the split.
  A point without the record is one the model did not want to split at.
- `end	<index>` closes the block; the index must match.

The `str` escapes are the golden format's: `\t`, `\n`, `\r`, `\\`, and
`\uXXXX` (four lowercase hex digits, one UTF-16 unit) for any other
character below U+0020. A surrogate that is not half of a pair is
written as `\uXXXX` as well, because it has no UTF-8 form. The language
detector's n-grams and the tokenizer's contexts are cut at UTF-16 units
and can hold such halves. The tokenizer dumps of `en/emoji`,
`en/emoji-split` and `multi/mixed` have them. The language detector
dumps have none, because its first normalizer replaces every character
outside the Basic Multilingual Plane with a space before the n-grams
are cut; a reader that
decodes into Rust strings has to keep such a field as UTF-16 units or as
its escaped text. The generator writes goldens through the same escape,
and its UTF-8 encoder fails rather than writing a replacement character.

### How the decisions are captured

The generator builds a second instance of each annotator over a
recording model. `MaxentRecorder` wraps the loaded maxent model, forwards
every call to it and records each `eval` call: copies of the context
array, of the feature values when there are any, and of the returned
probabilities (the annotators and models reuse their buffers). It extends
OpenNLP's `AbstractModel` rather than only implementing `MaxentModel`,
because `TokenizerModel` and `LanguageDetectorModel` refuse a maxent model
that is not an `AbstractModel`. It is built over the loaded model's own
parameters, predicate map and outcome names, so the methods
`AbstractModel` makes final (`getBestOutcome`, `getAllOutcomes`,
`getOutcome`) answer exactly as the loaded model's would.

The recording models are made with the constructors that take a maxent
model: `SentenceModel(String, MaxentModel, Map, SentenceDetectorFactory)`,
`TokenizerModel(MaxentModel, Map, TokenizerFactory)` and
`LanguageDetectorModel(MaxentModel, Map, LanguageDetectorFactory)`. The
map is every manifest entry of the loaded model except `factory`, so the
language, versions, timestamp, training entries and factory settings are
the loaded model's; the generator checks that the recording model's
manifest equals the loaded one's in every entry but `factory`, and that
the loaded model's factory is exactly OpenNLP's default factory class.
A model constructor writes the factory's class name into the manifest
and makes its own factory instance from it through OpenNLP's
`ExtensionLoader`, so the recording factories are public classes in the
generator's package, which the generator registers with
`ExtensionLoader.registerAllowedPackage`.

How a decision's position is found, per annotator:

| Annotator | Approach | Why |
|---|---|---|
| `SentenceDetectorME.sentPosDetect` | The context generator is wrapped (`RecordingSentenceDetectorFactory`). Each `getContext(text, position)` call is reported to the recorder with the array it returned; the next `eval` must receive that very array, and the point takes its `position`. | The detector skips a candidate when it lies before the last accepted sentence start, which depends on earlier decisions, so the candidates cannot be listed from outside without re-implementing the loop. |
| `TokenizerME.tokenizePos` | The context generator is wrapped (`RecordingTokenizerFactory`) as above. It sees only one whitespace token (a new substring per token) and a UTF-16 index into it, so the requests of one token are grouped by that substring's identity, and each group is paired with the next span of `WhitespaceTokenizer` (without kept line breaks, the tokenizer's default) whose text equals the token. The position is the span's start plus the index. | After an accepted split at an abbreviation the tokenizer jumps over the abbreviation's dots, which depends on the decision, so the candidates cannot be listed from outside. Whether a whitespace token is evaluated at all depends on its text alone (its length and the alphanumeric pattern), so a skipped span can never have the text of an evaluated one; the generator checks this and fails otherwise. |
| `LanguageDetectorME.predictLanguages` | No context generator is wrapped. The detector makes exactly one `eval(String[], float[])` call per document (checked), and the point has position 0. | There is nothing to pair. The contexts are the distinct n-grams of the normalized text in the iteration order of the `HashMap` the detector counts them in (keyed by `CharBuffer`, whose hash depends only on the characters, so the order is the same on every run of one JDK), and every value is 1: the detector does not pass the counts (OPENNLP-1261 in its source). |

Nothing in a dump is inferred: every context, value and probability is
what `eval` received or returned, and every position is what the
annotator passed to its context generator. For `acceptable`, the
recording sentence detector overrides `isAcceptableBreak`, records its
argument and result, and passes the call on; the generator pairs each
point whose best outcome is a split with the next such call, checks that
it names the same text and candidate, and fails if any call is left
over. The layers the recording
annotator produces are written as a golden and compared byte for byte
with the plain annotator's golden for the same input; the generator
fails if they differ, so a dump is only written when recording changed
nothing the annotator computed.

### Making them

`Main` writes the dumps when given `--dumps` after its five arguments.
The driver always passes it (`GENERATOR_ARGS` in
`analysis/goldens-tool/src/recipe/mod.rs`, forwarded by the container's
entrypoint), so `turbo-goldens generate` writes and installs goldens and
dumps together, and `turbo-goldens verify` compares both byte for byte.
`verify` and `generate` also check, at the header level only, that every
golden of these three annotators with a `model` record has a dump beside
it, except one with an error record, which has none (and that a golden
without a model has none), that every dump has a golden, and that
each dump's header parses and names the input and model of its path. The
checked in dumps were made in the same container run as the goldens.

## Models trained on permissive data

`analysis/train-recipes` (`turbo-train-recipes`) trains models only on
data the licence policy lets the project ship, and gates each one
against the OpenNLP model for the same task (#401). The first recipes
are the English sentence detector, tokenizer, part-of-speech tagger and chunker
trained on MASC 3.0.0 (CC BY 3.0 US), gated against
`opennlp-en-ud-ewt-sentence-1.3-2.5.4`,
`opennlp-en-ud-ewt-tokens-1.3-2.5.4` and OpenNLP 1.5's
`en-pos-maxent.bin` and `en-chunker.bin`:

- The data: MASC is fetched through the zoo by digest as a permissive
  dataset and read from its extracted directory with the MASC reader of
  `turbo-formats`; it is not checked in. The digest of the directory
  (`masc::corpus_digest`, a `sha256sum` manifest of every file) must be
  the recorded one. `masc::RELEASE` records the archive
  `MASC-3.0.0.zip`, a release asset of `github.com/nancyide/anc-website`
  (tag `datasets-2026-09-25`), as downloaded and checked: 36,358,521
  bytes, SHA-256
  `0b300d96016233c7c79b43559b477a6bb0ed16bb416d5c517eccb108a6228493`, and
  the digest of the `MASC-3.0.0` directory it extracts to
  (`445449f535cac62e74b2712ad975d7b727e72b8276720dd27492266565ae0afd`, 4,425
  files). That anc.org's MASC download page links to the asset was
  reported but not checked: anc.org could not be reached from the
  machine the recipes were run on. A run on any other directory needs `--unrecorded-corpus` and
  marks its cards as not a release.
- The split: the documents in the order the reader reads them; every
  tenth, from the tenth, is held out. No randomness, so no seed. A
  corpus with a document the reader skips is refused, since the skip
  would move the documents after it. MASC's Twitter genre (the
  documents whose header's `textClass` names the category `TW`, the two
  of `data/written/twitter`) is left out of both sides, as Kristian
  Rickert decided on 2026-10-15: whether Twitter's terms reach the
  tweets is not settled, and the licence policy refuses data under a
  platform's terms. The places count every document read, so no other
  document changes side, and the card lists the documents left out.
- The training: the factory defaults and the settings of the UD 1.3
  models (GIS, 300 iterations, cutoff 5, 4 threads, OpenNLP 2.5.4's
  whitespace, 10 sentences per sentence detector sample, the default
  feature generator descriptor and no tag dictionary for the tagger). The
  sentence detector and the tokenizer differ from their references in
  their data, not their settings. The tagger's reference was trained
  otherwise (its manifest records OpenNLP 1.5.0, 100 iterations and
  cutoff 5, and it holds a tag dictionary and no feature generator
  descriptor), so its gate compares data and settings together. The
  token stream skips a sentence without a sample alone
  (`EmptySentence::Skipped`) rather than end the corpus there as
  OpenNLP's does, and a sentence whose tokens are out of text order (a
  token inside another, which MASC 3.0.0 has once) alone, which the
  port's `TokSpanEventStream` refuses; the tagger's skips a sentence
  with a token without a tag, or with no token, alone rather than
  throw. Held-out sentences are skipped the same way; the card records
  both counts, and a recipe fails when its reader skips more than one
  sentence in a thousand on either side. The chunker is trained as
  `ChunkerME.train` trains with `TrainingParameters.defaultParams()`
  (GIS, 100 iterations, cutoff 5, one thread, beam size 10) on MASC's
  noun and verb chunks (`f.nc` and `f.vc`, which OpenNLP has no reader
  for, read by `turbo-formats`); a sentence with chunks that chunk tags
  cannot express (part of a chunk that crosses sentences, or chunks that
  share a token, which MASC's chunk layers, made automatically and then
  validated, hold in about one sentence in fourteen) is skipped, with a
  bound of one in ten of its own, and on the recorded release the
  recipe checks the counts it records (1,872 training sentences, 349
  held out). A chunk in no sentence the reader keeps is in no sample;
  the card counts these on each side. In the recorded release 16,597 of
  the 221,931 chunks the split reads are outside every kept sentence,
  because MASC's sentence layer does not cover all of the text and the
  reader drops a sentence region that overlaps the next, as OpenNLP's
  does; the other MASC recipes read the same sentences and lose the same
  text. The manifest's `Timestamp` is fixed, so a run gives the same
  bytes.
- The gate: both models scored with OpenNLP's `SentenceDetectorEvaluator`,
  `TokenizerEvaluator` and `POSEvaluator` on the same held-out samples,
  gated by `turbo_eval::model_gate`. MASC's tags are Penn tags, so the
  tagger's reference is OpenNLP's English maxent tagger trained on Penn
  tags, `en-pos-maxent.bin` of the 1.5 models, scored in its own
  outcomes. The trained model's tags are compared with the held-out
  tags as written; the reference's through a mapping of its tags
  (`postag::REFERENCE_MAPPINGS`) where MASC follows another convention
  (`-LRB-` and `-RRB-` for MASC's `(` and `)`; `:` for MASC's `-`,
  `...` and `;`; ``` `` ``` for an opening quotation mark, which MASC tags
  `''`). The mapping is by tag, so it also counts the reference's
  ``` `` ``` right on a closing quotation mark MASC tags `''` (34
  held-out tokens, and 30 neither opening nor closing plainly); it only
  ever favours the reference, by at most 0.08 points. Only the held-out
  tokens whose tag it can give, as written or
  through the mapping, are scored: the others are left out of both
  models' scores and listed on the card, with the reference's ten most
  frequent mismatches, and more than one in a thousand fails the
  recipe. Its training data is not published and the zoo does not list
  it; the recipe records its size and SHA-256, reads it only to score
  it, and never ships or bundles it. The
  English UD tagger asked for Penn tags would be no reference: it gives
  one Penn tag per UD tag, never `VBD`, `NNS` or `,`. The chunker's
  reference is `en-chunker.bin` of the 1.5 models, trained on CoNLL-2000
  chunks (the Wall Street Journal part of the Penn Treebank, an LDC
  corpus), recorded and only scored like the tagger's. Both chunkers are
  scored with `ChunkerEvaluator` on the chunk types both give, `NP` and
  `VP`, the only ones MASC has; the reference's chunks of its other
  types are made `O` before it is scored, and it is given `:` for MASC's
  `-`, `...` and `;` (the gate record's `reference_inputs`; worth 0.02
  points to it on the held-out set). MASC's chunks follow other conventions than
  CoNLL-2000's (verb groups split where CoNLL-2000 keeps them whole,
  predicate adjectives inside verb chunks, coordinations of single nouns
  split), and every such difference counts against the reference, so
  this gate leans towards the trained model.
- The output: `<out>/<model id>/` with the model, `card.json` (the
  data, its licence and attribution, the split, the parameters, the
  training counts, the tag formats a tagger was scored in, a chunker's
  chunks in no sentence its reader keeps and the gate's outcome) and
  `gate.json` (the gate record). `ModelCard::artifact`
  is the model's permissive zoo entry once the file is published; it
  refuses a card not gated against its recipe's reference (its id and
  SHA-256), whose gate failed, or whose corpus is not the recorded
  release.

```
cargo run --release -p turbo-train-recipes -- masc --corpus <masc-dir> \
    --references <dir with the OpenNLP models> --out <dir> [--unrecorded-corpus] \
    [--recipe <model id>]...
```

Before it trains anything, the binary checks every reference the run
needs, every model directory it would write, and every recipe's samples
as training and the gate check them (the skipped sentences of each side
and the held-out tokens whose tag the tagger's reference cannot give),
and writes nothing when one of them fails.

On the recorded release (the Twitter genre left out) every gate passes:
the sentence detector scores 79.94 against 77.67 (F-measure, 639 units,
difference +2.28, 95% interval +1.66 to +2.92), the tokenizer 98.83
against 94.61 (F-measure, 6,384 units, +4.21, +3.97 to +4.47), and the
tagger 95.16 against 91.70 (accuracy over 85,286 tokens of 6,384
units, +3.46, +3.21 to +3.71). About 1.10 points of the tagger's lead
come from 1,309 held-out punctuation tokens MASC tags `NN` (`[`, `]`,
curly quotation marks, dashes, `*`), on which the reference, which does
not know punctuation outside ASCII, is mostly wrong; scored without
them the lead is +2.39 (+2.18 to +2.63), and the gate still passes. The
chunker scores 89.74 against 82.46 (F-measure of `NP` and `VP` chunks,
6,036 units, +7.29, +6.84 to +7.75), a lead that measures how closely
each model follows MASC's chunk conventions as well as how well it
chunks. Two runs write the same model bytes. The crate's README has the counts, the
models' digests and the rest.

The four models go out as the assets of this repository's release
`models-masc-3.0.0-1`, and the zoo's built-in index lists them there as
permissive models. `.github/workflows/models.yml` trains them in CI,
checks with `turbo-train-recipes zoo-index --check` that they are the
bytes the index lists, and only then, run by hand on main, creates the
release with each model's card and gate record beside it; until it has,
fetching them through the zoo fails. A release names the models it
publishes, so a recipe added later goes out in a release of its own.

### The UD treebanks

The UD recipes (#415) train the sentence detector, tokenizer, tagger
and lemmatizer on the Universal Dependencies treebanks of release 2.16
whose own `LICENSE.txt` and README say CC BY 4.0: Spanish AnCora,
Catalan AnCora and Portuguese Porttinari (all four components), and
Finnish FTB and Italian MarkIT (tagger and lemmatizer; their `text`
comments put a space between almost every two tokens, and MarkIT's
files have 9 `SpaceAfter=No`, FTB's none, so they hold no sentence ends
or tokenization of running text). Portuguese DANTEStocks is CC BY
4.0 too, but its text is tweets, and the policy refuses platform-terms
data, as it does MASC's Twitter genre (docs/licences.md), so it is
refused and has no recipe. Porttinari's recipes stay, so a user can
train its models locally, but its CC BY 4.0 grant is the treebank
authors', for their annotations, and its text is Folha de São Paulo
articles that its README says are on Kaggle, with no licence from Folha
shown, so no Porttinari model is shipped (`UdModelCard::artifact`
refuses one whatever its gate says).

The ten models that may ship, Spanish and Catalan AnCora's four and
Finnish FTB's tagger and lemmatizer, go out as the assets of this
repository's release `models-ud-2.16-1`, as the MASC models go out in
theirs, and the zoo's built-in index lists them there as permissive
models. `.github/workflows/models-ud.yml` trains them in CI, checks with
`turbo-train-recipes zoo-index --check` that they are the bytes the
index lists, and only then, run by hand on main, creates the release.
`UdModelCard::artifact` makes an entry only for a card that names its
recipe's treebank at release 2.16, with the sizes and digests of its
files, and its recipe's component and language.

- The data: each treebank's `LICENSE.txt`, README and UD `train`, `dev`
  and `test` files, with sizes, SHA-256 and SHA-512 recorded in
  `treebank`, fetched through the zoo by digest from the repository's
  `r2.16` tag. The licence check reads `LICENSE.txt` (it must grant CC BY
  4.0 and name no NonCommercial, ShareAlike or NoDerivatives element,
  and a GNU licence only as a recorded alternative, as FTB's LGPL is)
  and the README's `License: CC BY 4.0` line, before every file is
  checked against its record.
- The split: UD's own, trained on `train` and gated on `test`, `dev`
  not read. The training has the settings of the MASC recipes (those of
  the UD 1.3 models) with the factory defaults of the treebank's
  language. The samples are those of OpenNLP's CoNLL-U streams, except
  that the tagger's and lemmatizer's leave out empty nodes, which are
  not words of the text.
- The gate: against the OpenNLP UD 1.3 model for the language and
  component (`es-ud-gsd`, `ca-ud-ancora`, `pt-ud-gsd`, `fi-ud-tdt`,
  `it-ud-vit`) on the `test` samples, by the MASC recipes' gates: the
  sentence detector's and tokenizer's as the MASC ones', and each
  tagger's as the MASC tagger's, both models' tags (their outcomes, the
  universal tags) compared with the held-out tag as written, with no
  mapping of the reference's tags. The lemmatizer, given the held-out
  tags, is scored on its lemmas the same way. A tagger's tokens whose
  tag the reference has no outcome for (FTB's `DET` and `PART`, which the
  Finnish reference never gives), and a lemmatizer's tokens whose gold
  lemma class (the edit script from the token to its lemma) gives the
  lemma while no class of the reference does, are tokens it cannot get
  right, so they are left out of both models' scores and listed on the
  card; over 5 in 100 and the recipe fails. The limit is one for every UD
  tagger and lemmatizer, a floor on how much of the `test` file the gate
  covers; the MASC tagger's limit, 1 in 1,000, would refuse FTB, where
  3.9 in 100 (tagger) and 1.1 in 100 (lemmatizer) are left out. Every token left
  out is one the reference gets wrong: when the candidate trails, leaving
  tokens out always lowers its lead, in the point estimate and in each
  resample, so it cannot make a failing gate pass whenever the resamples
  the interval's low end is read from have the candidate trailing; when
  it leads, it raises its lead only when the candidate's accuracy on the
  left-out tokens is below its lead, and by at most 5 in 100 of that
  lead. Except for Catalan, the reference was trained on another
  treebank and is scored against conventions it was not trained on,
  which favours the candidate (the Finnish TDT reference tags 303 of
  FTB's scored `VERB` tokens `AUX`); the Catalan gates are the only ones
  that compare models trained on the same treebank. Before any model is
  trained, the run checks every reference, every treebank and every
  recipe's samples (the skipped sentences and the tagger's and
  lemmatizer's left-out tokens), as the MASC run does. OpenNLP's Italian
  models are trained on UD_Italian-VIT, which is non-commercial, so
  they are local-only and not in the zoo: the Italian recipes record
  each one's size and SHA-256, read it from the references directory
  and only score it, as the MASC tagger's gate does
  `en-pos-maxent.bin`.
- The output: `<out>/<model id>/` with the model, `card.json` (the
  treebank, its files and licence, the attribution and the citation its
  README asks for, the parameters, the counts, and the MASC card's
  held-out and gate parts) and `gate.json`, the MASC recipes' gate
  record. The run prints each card's summary, as the MASC run does.

```
cargo run --release -p turbo-train-recipes -- fetch-ud --home <zoo-home> [--recipe <model id>]...
cargo run --release -p turbo-train-recipes -- ud --treebanks <zoo-home>/artifacts \
    --references <dir with the OpenNLP UD models> --out <dir> [--recipe <model id>]...
```

Results of a release-mode run of all sixteen recipes, with the five
lemmatizers run again after the last change to the tokens their gates
leave out (each model file is the one the runs before the review wrote,
by SHA-256), in points (F-measure for sentences and tokens, accuracy for
tags and lemmas), with the 95 percent interval of the difference from
the gate's paired bootstrap; the README has every interval, the times
and each gate's lead with its left-out tokens counted:

| Treebank | Sentences | Tokens | Tags | Lemmas |
|---|---|---|---|---|
| Spanish AnCora vs `es-ud-gsd` | 97.99 vs 96.27, +0.85 to +2.72 | 99.96 vs 99.89, +0.04 to +0.11 | 97.40 vs 92.73, +4.44 to +4.91 | 91.37 vs 90.64, +0.63 to +0.82 |
| Catalan AnCora vs `ca-ud-ancora` | 99.16 vs 99.16, +0.00 to +0.00 | 99.92 vs 99.92, +0.00 to +0.00 | 97.28 vs 97.24, -0.01 to +0.08 | 89.77 vs 89.77, -0.01 to +0.01 |
| Portuguese Porttinari vs `pt-ud-gsd` | 99.11 vs 98.26, +0.06 to +1.70 | 99.84 vs 99.20, +0.52 to +0.77 | 95.82 vs 91.49, +3.99 to +4.68 | 91.37 vs 81.38, +9.59 to +10.39 |
| Finnish FTB vs `fi-ud-tdt` |  |  | 89.25 vs 87.25, +1.47 to +2.53 | 88.87 vs 83.97, +4.35 to +5.45 |
| Italian MarkIT vs `it-ud-vit` |  |  | 89.96 vs 92.17, -2.92 to -1.49, fail | 82.57 vs 83.55, -1.44 to -0.53, fail |

The fourteen models gated against references of the zoo pass, and the
two Italian models fail: they lose 2.21 points (tagger) and 0.98
(lemmatizer) to OpenNLP's Italian models, so they get no zoo entry. The
gates leave out the test tokens the reference cannot get right: the
Finnish tagger's 635 of 16,261 (3.91 percent: `DET`, `PART` and
`SCONJ+VERB`), the Portuguese one's 6 of 31,451 and the Italian one's 3
of 9,654, and the lemmatizers' 57 (Spanish), 18 (Catalan), 100
(Portuguese), 177 (Finnish) and 33 (Italian). Every such gate's lead is
lower with them left out than with them counted, or the same: most for
the Finnish tagger, +2.00 against +4.55. No model gives a lemma with a
capital letter, as no OpenNLP lemmatizer does (it applies a class to the
lowercased token), so both models get wrong the test tokens whose lemma
has one: from 7.14 percent of the Portuguese ones to 12.12 percent of
the Italian ones, and 1 Finnish token. Giving them would be an
improvement over OpenNLP, found here and not made. The Catalan references are OpenNLP's models of the same
treebank: the sentence detector and tokenizer recipes make OpenNLP's
events exactly and their models differ from the references only in the
last bits of some parameters, so those gates check that the recipes
reproduce OpenNLP's models, and the tagger and lemmatizer are trained on
OpenNLP's samples less the treebank's 4,999 empty nodes. The others are
trained on another treebank of the language, so a gap there can come
from the treebanks' annotation as well as from the models, which favours
the candidate.

The language detector recipe (#429) trains `LanguageDetectorME` on
Tatoeba's CC BY 2.0 FR sentences (each sentence's licence read from
`sentences_detailed.csv` and `sentences_CC0.csv`, the CC0 ones left
out) of the languages `langdetect-183.bin` also covers, through an
explicit table of Tatoeba's and the model's ISO 639-3 codes, with the
command line trainer's default parameters, and gates it against
`langdetect-183.bin` on 500 held-out sentences per language, one unit
each (`Task::LanguageDetector`, `Scoring::Labels`). The model's
directory lists the ids of the sentences it was trained on
(`training-sentences.txt`), the credit CC BY 2.0 FR asks for, as
Tatoeba's sentence addresses. Tatoeba's download host refused the
machine it was written on, so the full run was made on GitHub's runners
(`.github/workflows/models-tatoeba.yml`) on Tatoeba's export of
2026-10-10, which the recipe records: 71 languages with at least 1,500
sentences after the licence check and de-duplication, 321,026 sentences
trained on and 35,500 held out, and the
trained model right on 92.19% of them against 83.91% for
`langdetect-183.bin`, a pass. The model and `training-sentences.txt` go
out as release `models-tatoeba-2026-10-10-1`, and the export's archives,
which Tatoeba replaces every week, as release `data-tatoeba-2026-10-10`.
The README has the record, the split, the thresholds and which way the
gate leans.

The web language detector recipe (`langdetect_web`, #429) trains on the
same sentences and on windows of at most 300 characters from FineWeb-2
and FineWeb (English) documents, 100 languages in all, read from three
ranges of each of 204 Parquet files at fixed revisions
(`fineweb-files.txt` and `fineweb-ranges.txt` list them with their
digests). Windows from tatoeba.org and windows holding a Tatoeba
sentence of 20 characters or more from the other side are passed over. It is gated twice against
`langdetect-183.bin`: on the 35,500 held-out sentences, 91.25% against
83.91%, and on 47,079 held-out windows, 96.46% against 93.45%, both a
pass, in run 38848321968 of `.github/workflows/models-fineweb.yml`.
Beside the Tatoeba model it is about a point worse on the sentences
(91.25% against 92.19%) and about six points better on the windows of
the languages that model knows (96.84% against 90.88%). It goes out as
release `models-tatoeba-fineweb-2026-10-10-1`, with the ids of the
sentences and the documents it was trained on.

## Differential fuzzing

The golden tests compare the ports with what OpenNLP wrote for a few
hundred hand picked inputs. `turbo-goldens fuzz` compares them on as
many generated texts as a machine has time for: it writes a seeded
corpus, runs the Java generator over it in the recipe container with the
run definitions of `Runs.java`, so every annotator that has a golden run
takes part, compares each golden with the Rust annotators through the
comparison code the golden tests use, checks the offset invariants of
every layer the Rust side wrote, and reduces each divergence to a
minimal input. Three pieces make it up.

### The corpus

`turbo_goldens::fuzz` (`analysis/goldens/src/fuzz/mod.rs`) is a
deterministic generator written in the crate, with its own SplitMix64
source (`analysis/goldens/src/fuzz/rng.rs`), so it depends on nothing
outside the workspace. `corpus(seed, count)` gives `count` inputs, and
input `index` of a seed is the same whatever `count` is and whatever came
before it (`input(seed, index)` makes one alone), so a failing input is
named by its seed and index: its id, `s<seed>-<index>` with five digits,
is its file name in every input set. The texts come from the grammar in
`analysis/goldens/src/fuzz/grammar.rs`: sentences of a small English
grammar with abbreviations, numbers in both notations, money, quantities,
URLs, quotes and brackets in them; lines of other scripts (CJK, Arabic
and Hebrew with bidi controls, Devanagari, Thai); emoji sequences with
ZWJ, variation selectors and skin tones; very long tokens; empty and
whitespace only lines; CR LF and lone CR line ends; and then zero width
and format characters, combining marks (in and out of canonical order),
NBSP and the other Unicode spaces dropped into the words. A text
is a document of a few lines of these.

### The Rust side

The crate `turbo-fuzz` (`analysis/fuzz`) holds, in `runs`, one `Run` for
every `<annotator>/<run>` directory of `analysis/goldens/data` the
differential covers: it opens the annotators as the crate's own golden
tests open them (the same models, fixtures, options and pipelines, the
golden's tokens and tags where a run tags golden tokens) and compares the
document with the golden through `matches_golden` or `matches_golden_key`
with the tolerance the golden test uses, reading the expectations the
golden tests read from `turbo_goldens::expect`: the JVM's spans over
surrogate halves, the whole character spans, the simple tokenizer's line
break spans, the view edit equivalence of the normalizers, the trailing
marks of the WordPiece spans. Those functions moved out of the tokenize,
embed and normalize golden tests into `turbo-goldens` so that the
differential and the tests cannot drift apart. `runs::RUNS` lists every
run and a test holds it to the directories on disk; the differential
leaves out the `train`, `formats`, `masc`, `ner`, `evaluate`,
`detokenize` and `bpe` directories (files, scores, token lists and
merges, not layers over a text) and `normalize/case-fold-tr` (the port
has no Turkish case fold).

Every comparison ends in one of four outcomes. A match. A documented
difference: a JVM behaviour the crate docs describe and test that the port
does not reproduce, recognized by a rule written in the run; a divergence
is anything else. A skip: a golden the Rust side cannot compare (a token
golden that cuts a surrogate pair, the lemma golden that does not read, a
run whose model is not downloaded). The documented differences are
exactly the ones the golden tests list:

- `langdetect`: the port declines a text with no evidence for any
  language, where the JVM ranks every language at the prior
  (`analysis/langdetect/tests/goldens.rs`); and where a bucket of the
  JVM's n-gram map that the JDK turned into a tree held n-grams with
  equal hash codes, which the JDK orders by identity hash code, the
  n-grams reached the model in an order the port cannot reproduce and
  the probabilities differ in their last bits, within
  1e-12 (`ContextGenerator::exact_order` in
  `analysis/langdetect/src/context.rs` says when).
- `tokenize/simple+keep-new-lines`: the two defects of `SimpleTokenizer`
  with `keepNewLines` (`analysis/tokenize/tests/golden.rs`,
  `simple_newlines`), the exception on a text that starts with a line
  break among them.
- `normalize`: for each normalizer whose section of `docs/normalizers.md`
  records a deliberate difference ("Differences from OpenNLP"), a rule in
  `analysis/fuzz/src/runs/normalize/mod.rs` recognizes that construct and no
  other: every region of the source the two sides change differently
  must be it (U+2E42 folded by `Quotes`, the five added bullets, the
  three added ellipsis forms, U+206A to U+206F deleted by `Invisible`, a
  word ending in combining marks joined by `Dehyphenate`, `y'all` and the
  added subjects of `EnglishContractions`, a symbol in running text for
  `SymbolWords`, an internationalized e-mail address for `BoundedUrls`,
  and for `ConfusableSkeleton` a JVM skeleton from the 17.0 data whose
  18.0 skeleton is the port's). The edits that are equivalent to the
  golden's though not the same rows are a match. `EmojiRemove` in its
  supplementary mode compares exactly: its documented differences are
  the sequence mode and the annotator.
- `subwords`: the four differences of `analysis/embed/tests/goldens.rs`
  and the special token strings. The run models how `WordpieceEncoder`
  maps its normalized text back to the source (controls dropped,
  whitespace and the no-break spaces made spaces, ideographs and
  punctuation isolated, `Character.toLowerCase` with final sigma, each
  stripped mark's end given to the character before it, and the whole
  run given to every character when canonical reordering crossed
  characters) and lays the port's pieces over that mapping; the golden is
  a documented difference only when its spans are exactly what the
  mapping gives.

`differential::compare_tree` walks a tree of goldens, opens each run
once, compares each golden and runs `turbo_goldens::invariants` on the
document, and renders a report: a table per run (matches, documented,
divergences, skips, broken invariants), every problem in full, then the
documented differences one line each. `reduce` is delta debugging (ddmin)
over the lines of a text and then over its characters, asking an oracle
whether a candidate still fails; the oracle gets a batch of candidates at
a time, so one container run tests a whole round.

### The command

```
cargo run -p turbo-goldens-tool --features fuzz -- fuzz --seed 1 --count 1000
```

writes the inputs of the corpus into `<out>/inputs/<set>/<id>.txt` for
each set of `--sets` (`en`, `edge`, `norm`, `num`, `money`, `quantity`,
`temporal`, `pii` and `sentence-fixture` by default: between them every covered run reads
each text), runs the generator in the container over that inputs
directory with `--skip-annotators` naming those annotators, writes the
goldens under `<out>/jvm/`, compares them, writes the report
to `<out>/report.txt` and prints it, and reduces up to `--reduce`
problems (10; `--no-reduce` none, `--reduce-documented` the documented
differences as well, to a minimal example of each). Each reduction runs
the container with `--only-runs <annotator>/<run>` over a round's
candidates, at most `--rounds` times (40), and writes the minimal text
as `<out>/candidates/<id>-<annotator>-<run>.txt` with a `.report` beside
it naming the run, the mismatch and the line that makes the input again:
`turbo-goldens fuzz --seed <seed> --count <index + 1> --sets <set>`.
`<out>` is `target/turbo-goldens/fuzz/s<seed>-<count>` unless `--out`
names a directory; `--models` is the models directory `fetch-models`
filled (`target/turbo-goldens/models` by default) and `--image` the
container image (the recipe's tag by default). `--reuse` compares and
reduces what an earlier run left under `<out>` without writing the
inputs or running the generator again, so a classification rule can be
worked on against a fixed tree. The command exits non zero when the
report has a problem.

A failing input is reproduced from its id alone: `s2-01234` is
`turbo_goldens::fuzz::input(2, 1234)`, and `turbo-goldens fuzz --seed 2
--count 1235 --sets edge` makes its golden again (the inputs before it
come along; every input of a seed is a fixed function of its index).
After a fix, the candidate file is checked in as an input of the set the
`.report` names, and `turbo-goldens generate` gives it a golden.

The intended use is a nightly run on a hardware host with Docker and the
models fetched, over a few thousand inputs with a fresh seed each night
(the date serves), the report kept and every candidate looked at. A
thousand inputs over the seven sets are a hundred thousand goldens: the
generator takes about two minutes for them in the container, and the
comparison about seven in a debug build, most of it in the MeCab
dictionaries loading.

### The offset invariants

`turbo_goldens::invariants` (`analysis/goldens/src/invariants/mod.rs`) checks
every span of every text anchored layer of a document, in the view the
spans are in: it is in bounds and ordered, both ends are on character
boundaries of the view's text, UTF-8 to UTF-16 to UTF-8 round trips
through the view's offset table and agrees with `turbo_unicode`'s own
conversion (written apart from it), code point offsets round trip too,
and the span maps to source bytes exactly through every normalizer
between the view and the source: the source span is on character
boundaries, mapping it forward covers the span, mapping that back gives
the same source span, and where the span touches no edit the source
bytes are the view's bytes. `check_document` finds every instance of the
shared layer types that have spans (sentences, tokens, subwords, chunks,
phrase chunks, entities, numbers, money, quantities, temporal mentions, PII
mentions, structure);
`check_spans` takes any span columns. A `Violation` names the layer, the
view, the span and what failed.

The differential runs them on every document the Rust side makes.
Without a container, `analysis/fuzz/tests/invariants.rs` runs every run
over the first forty inputs of seed 7 and every checked in golden input
but those of the sets it names as left out, each with its reason (the
stemmers' and the Hunspell runs' word lists); a set that is in neither its fed list nor its
left out list fails the test, and
`analysis/fuzz/tests/checked_in.rs` runs the differential itself over
the checked in goldens, which must show no divergence and no broken
invariant, only the documented differences and the known skips. Both
run in an ordinary `cargo test -p turbo-fuzz`; they skip the runs whose
published model is not downloaded unless
`TURBO_TEST_REQUIRE_OPENNLP_MODELS=1`, and the published MeCab
dictionaries unless `TURBO_TEST_MECAB_DICTIONARIES=1`.

### What the runs found

Three corpora were run while the differential was built: seed 1 with a
thousand inputs and seed 2 with three thousand over the seven default
sets (a hundred and three hundred thousand goldens), and seed 3 with
forty inputs of the `edge` set to exercise the reducer. Across them the
Rust side matched or explained every golden; no offset invariant broke.
What it took to get there:

- The lemmatizers lowercase with `turbo_unicode::java::to_lowercase`,
  which wrote final sigma under Unicode's Final_Sigma condition. The JDK's
  `String.toLowerCase` has its own: it walks the sigma's word, as its word
  break iterator for the root locale delimits it, passing over every
  character that is not cased, where Unicode skips only case-ignorable
  ones, and its list of cased characters is an older Unicode's. The
  reducer brought a 551 byte input down to "ΣΟΦΟΣขอบคุณǰ" in twelve
  container runs: Thai letters and then a cased letter make the sigma
  non-final on the JVM. `to_lowercase` follows the JDK's condition
  (`docs/normalizers.md`, "turbo-unicode"), with the cases where the
  two conditions differ checked against OpenJDK 21 in
  `analysis/unicode/tests/java.rs` and the reduced input in the
  lemmatizer's tests. The case fold normalizer lowercases through the same
  function, so it changed with it; its goldens still match.
- The view edit equivalence of the normalizer goldens compared the edits
  region by region, and a diff over repeated characters ("xxxẍ",
  "6/98/2635 4") can delete one occurrence in one region and keep another
  in the next, so four goldens of `nfc`, `nfkc` and `number` looked
  different though the texts were the same. The check widens a region
  over the following ones until the lists agree.
- The WordPiece goldens differ from the port on a third of the texts, in
  the ways `analysis/embed/tests/goldens.rs` documents, but telling those
  from anything else needed the JVM's offset mapping written out (the
  model in `analysis/fuzz/src/runs/subwords/mod.rs`); the rules of thumb the
  golden tests use for their few inputs (a span widened to its run, a
  trailing mark trimmed) left a few dozen of the first corpus unexplained
  until then. The model also records what `WordpieceEncoder` does that
  the golden tests had no input for: no-break spaces are whitespace to
  it, controls and format characters are dropped before the split, and a
  run's marks out of order give every piece of the run the whole run,
  punctuation included.

The documented differences are frequent in the corpora, as the generator
aims at them: in the seed 2 run, 2,338 WordPiece goldens with a stripped
mark in a span, 1,763 confusable skeletons (Unicode 18.0 confusables
against the JVM's), 2,381 simple tokenizer goldens with `keepNewLines`
(443 of them the exception), 303 bounded URL goldens (internationalized
e-mail addresses), 224 symbol word goldens (symbols in running text), 148
WordPiece goldens with final sigma, 100 language goldens. For the
normalizers with a "Differences from OpenNLP" section, the rule per
normalizer in `analysis/fuzz/src/runs/normalize/mod.rs` (listed above) names
the difference each mismatch is, and a region no rule recognizes is a
divergence. No OpenNLP bug was found that the crate docs did not already
record.

## Evaluation

`turbo-eval` (`analysis/eval/`) is the harness every change to the
chunker is to be measured with (#154). It holds the evaluation sets, the
pipeline, the baseline chunkers, the measures, the significance tests,
the record, the gate and the `turbo-eval` binary. The baseline records
are under `analysis/eval/records/<pipeline>/<set>.json`.

### The pipeline

`Pipeline` runs over one set (`set::LoadedSet`) in four timed stages:

1. Analysis: each document's text as a document in the source view,
   with `opennlp:sentences` (an OpenNLP maxent model, by default the
   English UD EWT model `opennlp-en-ud-ewt-sentence-1.3-2.5.4.bin`, read
   from the directory `TURBO_GOLDENS_MODELS` names or
   `target/turbo-goldens/models`, where `turbo-goldens fetch-models`
   downloads it; or one sentence per line),
   `turbo:structure` (plain text by default, Markdown, or none) and the
   embedder's `opennlp:subwords`. Every chunk span is a byte range of
   the set's own text, the offsets span judgements use.
2. Chunking: the chunker under test writes byte spans; a span given
   twice or past the text fails the run.
3. Embedding: one vector per chunk, from the chunk's text alone (with
   the document's title in front when `--titles prefix` is given), in
   the bundle's document role, and one per query, in the query role,
   with no chunking. The embedder (`embedder::TurboEmbedder`) tokenizes
   with the bundle's tokenizer, cuts a row as the bundle cuts it, runs
   the rows through a turbo session on the device named, mean pools,
   and scales each vector to unit length. A bundle that does not mean
   pool is refused. The reference embedder is all-MiniLM-L6-v2, the
   model turbo-bench measures; the bundle is a command line argument.
4. Retrieval: every query against every chunk vector, exact cosine with
   no index, summed in f32 in a fixed order so the scores are the same
   on every machine. A document scores its best chunk. The top 100
   documents and the top 100 chunks of each query are kept, ordered by
   the tie rules of `metrics` and `spans`. As in BEIR (its
   `ignore_identical_ids`, on by default), a document whose id is the
   query's own id is left out of both rankings before the cut; in
   ArguAna 1298 of the 1406 test queries are also documents.

The record's cost is the wall time of each stage, the tokens the
embedder was given (chunks, queries and any sentences a chunker
embedded, counted on the rows as run, after any cut), the chunk count,
and the chunk lengths in bytes of the source and in the embedder's
tokens, overhead included and before any cut. The record's analyzer
list names the sentence model by file name and SHA-256.

### Chunkers

Each chunker records every setting it ran with, defaults included.

- `document`: the whole document, its first sentence's start to its
  last sentence's end, as one chunk: the unit of a document-level
  baseline such as BM25 over whole documents.
- `fixed-tokens`: windows of N of the model's subword pieces, each
  starting N minus the overlap after the one before; the last ends at
  the last piece. N counts the text's pieces, so the binary refuses an N
  that with the template's special tokens is over the model's row.
- `sentence-windows`: consecutive windows of N sentences, no overlap.
- `opennlp-sandbox-semantic`: a port of the decision rule of the
  OpenNLP sandbox's `SemanticChunker` (the ai-pipestream/opennlp-sandbox
  fork, branch OPENNLP-1833-grpc-expansion). Each sentence is embedded
  alone; a chunk starts after each pair of neighbouring sentences whose
  cosine is below the threshold (a percentile of the document's own
  cosines when given, else a fixed cosine, 0.5 by default); chunks under
  the minimum sentences are merged into a neighbour, then chunks over
  the maximum are split. The cosine and the nearest rank percentile are
  computed as the Java computes them. Its parity with the JVM on golden
  inputs is not checked yet.
- `turbo-chunker`: turbo-chunker's `Chunker` over `StructureBoundaries`,
  optionally with `DriftBoundaries` weighed in (`--drift`) and the
  cohesion term (`CohesiveChunker`, `--cohesion`); the level 0 chunks
  are retrieved. Drift and cohesion read sentence vectors, which here
  are each sentence embedded alone rather than late pooled as
  `SentenceVectors` pools them; the record says so.

### BM25 retrieval

`--retrieval bm25` ranks the chunks by BM25 over their terms instead of
by the embedder's vectors, and embeds nothing: the embedder only splits
the text into its pieces for the analysis and counts chunk tokens for
the cost. The index is turbo-chunker's `Bm25PostingIndex`, one row a
chunk, with Lucene's `k1` 1.2 and `b` 0.75 unless `--k1` and `--b` say
otherwise. A chunk's terms (`--terms folded`, the default) are its UAX
#29 word tokens that hold a letter or a digit, lowercased: the folded
column turbo-chunker's term statistics index. `--terms english` also
drops a trailing `'s`, leaves out Lucene's 33 English stop words and
applies Porter's stemmer, after Lucene's `EnglishAnalyzer`. A document
scores its best chunk, ties and the query's own document as in the
dense ranking, and a chunk with no query term is not ranked. The record
names the retrieval and its settings under `pipeline.retrieval`; a
dense record has no such key.

`--dump-terms <dir>` writes each set's chunk and query terms, and
`analysis/eval/reference/Bm25Reference.java` indexes the same terms with
Lucene 9.12's `BM25Similarity` and reports nDCG@10, as a reference for
the scores. On the three BEIR sets with whole documents the two agree to
within 0.4 points; Lucene stores a document's length in one byte, which
is where they part.

### Measures and the gate

- `metrics`: nDCG@10 with linear graded gain and a `log2(rank + 1)`
  discount, and recall at 1, 5, 10 and 100, at document level. Hits
  with the same score are ordered by document id, descending, as
  `trec_eval` orders them. Queries without a relevant document are left
  out of the mean, as `trec_eval` leaves them out; a query with relevant
  documents and no ranking scores 0 and counts.
- `spans`: for the top k chunks of a query, as UTF-8 byte ranges of
  their documents, the precision and recall of relevant bytes covered,
  and the fraction of relevant spans that lie wholly inside one chunk.
- `significance`: a paired bootstrap of the mean per query difference
  (10,000 resamples, 95 percent percentile interval) and a paired sign
  flip permutation p value, both drawn from turbo-jvm's `JavaRandom`
  with a fixed seed, so the same records give the same bits.
- `record`: one JSON record per set and pipeline, written with the
  pretty printer turbo-bench uses. Its provenance block (commit,
  `pushed_to`, machine, device, bundle) reuses the core's
  `turbo::record` types. Every field is required, an unknown field is
  an error, and the record carries each query's nDCG@10 so two records
  can be compared query by query.
- `gate`: candidate records against baseline records, one per set. It
  passes when at least two sets gain 2.0 nDCG@10 points or more with
  the low end of the 95 percent bootstrap interval above zero, and no
  set's mean difference is below -0.5 points. It reports each
  difference with its interval and p value beside it.

### Model gates

`model_gate` is the gate a trained model passes before it ships (#401):
the model and the reference, the OpenNLP model for the same task, are
scored on the same held-out samples, each sample as counts (true
positives, predictions and references for an F-measure; correct and
total tokens for an accuracy), and a set's score is that of its summed
counts, as OpenNLP's `FMeasure` and `Mean` add them up. A paired
bootstrap draws the samples with replacement, 10,000 times from
`JavaRandom` seeded with 154 (the draws of `significance::bootstrap`),
and scores both models on each draw. The gate is a non-inferiority
test: the model passes when the low end of the 95 percent interval of
the difference, model minus reference, is at least -0.5 points
(`MARGIN_POINTS`, the chunker gate's floor), so a loss of more than half
a point is ruled out at that confidence. A small held-out set, whose
interval is wide, fails rather than passing for lack of evidence, and a
set of fewer than 30 units (`MIN_UNITS`) is refused. The difference is
that of the two pooled scores, each from the set's summed counts, not a
mean of per sample differences. Each model's own interval is reported
beside the difference. The F-measure is 0 where precision and recall are
both 0 (OpenNLP's is -1).

For an F-measure the bootstrap resamples whole units (the sentences or
documents a recipe scores) and scores each draw as OpenNLP's `FMeasure`
scores a set: it sums the drawn units' true positives, predictions and
references and computes precision, recall and F from those sums. It
never averages per unit F values, which would weigh a sentence of one
name as much as one of ten, would not be the score OpenNLP's evaluators
report, and do not exist for a sentence with neither a reference nor a
predicted span. The tasks scored this way are the sentence detector and
tokenizer (untyped spans), and the name finder and chunker (typed spans,
`TokenNameFinderEvaluator` and `ChunkerEvaluator`). A gate of typed spans
scores only the types its record lists, on both sides: held-out spans of
other types are left out of the references (as `NameSampleTypeFilter`
leaves them out of a name sample) and predicted spans of other types out
of the predictions; a held-out type the models name otherwise is renamed
first. `turbo-train-recipes`' `spans` module makes a unit's counts from
its evaluator's `FMeasure`.

`ModelGateRecord` is the record kept beside the model card: the task
and measure, the held-out set (its name, the corpus digest, the SHA-256
of the samples as scored, the held-out sentences the recipe's reader
skipped, and how the units were scored: spans; typed spans with the
types scored, the held-out types renamed, the held-out input tags the
reference is given another tag for (a chunker's part-of-speech tags;
empty for a model that reads no tags), the spans scored and those left
out; or tags with the tokens, the tokens left out and the mapping of the
reference's tags), both
models by id and SHA-256, the resamples and seed, the result, and both
models' counts per sample. `check` refuses empty names, digests that
are not 64 lower case hex, resamples or a seed other than the gate's, a
scoring that is not the task's, a tag scoring whose tokens less those
left out are not what each model's units count, a lemmatizer's record
with a mapping, and a typed span scoring with no type, a type listed
twice, a held-out type renamed twice, renamed to a type it does not
score or renamed though it is scored as written, a held-out input tag
given to the reference as another tag twice or as itself, or spans
scored that are not the references each model's units count, then
recomputes the result from the counts. It is strict JSON, as
the evaluation record is, and is never replaced once written.

The tasks are the sentence detector and the tokenizer, scored by
F-measure, and the part-of-speech tagger and the lemmatizer, scored by
accuracy as `POSEvaluator` and `LemmatizerEvaluator` count it (#415). A
lemmatizer's lemmas are recorded as a tag scoring with no mapping: the
tokens left out are those whose gold lemma class (the edit script from
the token to its lemma) gives the lemma while no class of the reference
does, and `check` refuses a lemmatizer's record with tag mappings.

### The sets

The sets (decision 10) are described one file each under
`analysis/eval/src/sets/`: name, published files with their URL and
SHA-256, licence, and the unit of the set's span judgements. A set is
fetched into `<cache>/<name>/`, where the cache root is the `--cache`
path or `$TURBO_EVAL_CACHE` (no default); each file is written to a
temporary name and renamed only when its SHA-256 matches the recorded
one. A set with an empty licence is neither fetched nor loaded, and a
file with no recorded digest is downloaded, its digest printed, and
refused. The readers under `analysis/eval/src/read/`, one per format,
give the typed `LoadedSet` (documents, queries, document judgements,
span judgements as UTF-8 byte offsets into the text as loaded), with
errors that name the file and line. `LoadedSet.sha256` is the SHA-256 of
a `sha256sum` style manifest of the files read, sorted by path.

| Set | Source | Licence | Spans | Digests |
|---|---|---|---|---|
| `scifact`, `nfcorpus`, `arguana` | BEIR layout, from MTEB's plain copies at a recorded revision | CC BY-SA 4.0 (BEIR dataset cards) | none | recorded |
| `legalbench-rag-mini` | a shared folder, placed by hand | MIT for the benchmark; the documents under their sources' terms | code points, converted after Python's newline translation; the mini subset is chosen with a port of CPython's `random` | none (no stable file URLs) |
| `conteb-covid-qa` | Parquet | none for the article text, so refused (below) | the judged chunk | not recorded yet |
| `conteb-squad` | Parquet, ConTEB's SQuAD set (`illuin-conteb/squad-conteb-eval`, revision 8f81cbd), validation split | CC BY-SA 4.0 (SQuAD over Wikipedia; downloaded at use, never bundled) | the judged chunk | recorded |
| `gutenqa` | Parquet | MIT (dataset card) | the quoted passage, located in its chunk as written or, failing that, with quotation marks dropped and whitespace runs made one space in both texts | recorded |
| `hicbench` | no copy that can be fetched | none stated, so refused (below) | quoted sentences, located in the document | none |
| `qasper` | gzip tar, test split | CC BY 4.0 | quoted evidence, located by paragraph or text | recorded |

ConTEB's and GutenQA's tables are read from the published Parquet files
by the crate's own reader (`read::parquet`): flat tables of string and
integer columns, PLAIN and dictionary pages in data page versions 1 and
2, nulls, and the SNAPPY (a decompressor in the crate), GZIP (`flate2`,
already in the build), ZSTD (`turbo-zstd`, in the workspace) and
UNCOMPRESSED codecs. pyarrow writes SNAPPY by default; FineWeb-2's text
column is ZSTD-compressed, as the models-fineweb run showed reading
`data/afr_Latn/train/000_00000.parquet`. Other encodings and codecs
(LZ4 and BROTLI among them) are refused with an error naming them, and
a column is read by name, so a reader reads only the columns it uses.
Each file with a recorded digest is checked on the bytes the reader
parses, so it is read and hashed once. A file too large to fetch whole
can be read in part: its footer's length from its last 8 bytes, its
schema and first row groups from the first bytes of its footer
(`Leading`), and those row groups from the file's first bytes
(`Parquet::open_head`), which is how the language detector's web recipe
reads FineWeb-2 and FineWeb. When the first bytes of the footer are all
of it, every row group must be read from them. Dropping quotation marks
in GutenQA's loose match can join words (“Yes”“No” matches "YesNo",
"its" matches "it's"), which is unlikely
to misplace a GutenQA passage. Chunked sets are joined into
documents with a line break between chunks. Evidence a reader cannot
place in its document is listed in `LoadedSet.unplaced` rather than
dropped silently. A BEIR judgement naming a document the corpus does
not have (ArguAna's test split has 5) is kept and listed in
`LoadedSet.absent`: it counts as relevant and is never retrieved, as
`trec_eval` scores it, and `run` prints how many there were.

Two sets stay refused (checked 2026-10-15; the set files carry the
links):

- `conteb-covid-qa`: ConTEB's card states no licence and reserves all
  rights to the original document authors, and none of the
  illuin-conteb datasets carries a licence. The source, deepset's
  COVID-QA, is Apache 2.0 for its annotations, but its contexts are the
  full text of 147 CORD-19 articles, and CORD-19 passes each article on
  under the article's own licence, some of which restrict
  redistribution or commercial use. Its replacement is
  ConTEB's SQuAD set, `conteb-squad` (`illuin-conteb/squad-conteb-eval`,
  revision 8f81cbd, validation split, same tables plus an `offset`
  column): the card states no licence either, but its source states
  one: SQuAD (`rajpurkar/squad`) is CC BY-SA 4.0, and its passages are
  Wikipedia text under the same licence, which ConTEB's chunks
  reproduce. It takes the same reader, and its data is downloaded at use
  and never bundled.
- `hicbench`: HiChunk's repository links no copy of the data, and the
  copy the paper names (`Youtu-RAG/HiCBench` on the hub) cannot be
  opened; neither states a licence. Its documents come from OHR-Bench
  (CC BY 4.0), but its questions and evidence are its own.

### The binary

```bash
cargo run --release -p turbo-eval -- fetch --sets scifact,qasper --cache <set-cache-dir>
cargo run --release -p turbo-eval -- run --bundle <bundle-dir> --sets scifact,qasper \
    --cache <set-cache-dir> --out <record-dir> --chunker sentence-windows --window 5
cargo run --release -p turbo-eval -- gate <candidate-dir> <baseline-dir>
```

`fetch` downloads each named set into the cache and checks it; `run`
reads them from there. Both take the cache root from `--cache`, or from
`$TURBO_EVAL_CACHE` when it is not given.

`--device cuda` needs the binary built with the core's cuda backend:
`cargo run --release -p turbo-eval --features turbo/cuda -- run ...`,
with `TURBO_CUDA_ROOT` set when nvcc is not under `/usr/local/cuda`
(`TURBO_CUDA_ROOT=/usr` for a distribution's toolkit; docs/cuda.md).
Without it the device is not found.

`run` writes `<record-dir>/<set>.json` for each set, so one directory
holds one pipeline's records, and never replaces a record. As
turbo-bench does, it refuses a working tree with changes (new files
under `analysis/eval/records/` aside), a commit on no branch of origin,
and a binary built from another commit, and checks the tree again after
the run. `gate` prints one line per set and the verdict, and exits with
status 1 when the gate fails and 2 on an error. `turbo-eval` with no
arguments prints every option.

`cargo test -p turbo-eval` checks the measures against values worked
out by hand in the tests, the tie and empty query rules, the
reproducibility of both tests, the permutation p value where the exact
answer is known, the record's round trip and its refusal of unknown
and missing fields, each chunker's cuts on small texts worked out by
hand, and whole runs over a hand made set with a stub embedder behind
the same trait (rankings, document scores from the best chunk, span
hits, cost, and the records `run` writes). One test runs the real
session path on the small BERT bundle in `testdata/tiny-bert-bundle`;
no test runs all-MiniLM-L6-v2, which is not in the tree. Each reader
is checked on a small sample made for the tests
(`analysis/eval/tests/data/`): typed values, offset conversion on
non-ASCII text, and errors with file and line (or row, for Parquet).
The Parquet reader is checked on small files pyarrow wrote (SNAPPY and
dictionary pages in version 1 pages, GZIP in version 2 pages, a
dictionary that fills up and falls back to PLAIN, several row groups,
nulls, PLAIN and uncompressed, and ZSTD; the script is in
`analysis/eval/tests/parquet.rs`), on files a writer in the tests
makes, for damaged values, a ZSTD page made by the `zstd` tool and a
codec it refuses, and on hand made pages and footers for damaged
sizes, counts and levels. None needs the network.

## Tests

`cargo test -p turbo-analysis -p turbo-analysis-layers -p turbo-chunker -p turbo-structure -p turbo-analysis-embed -p turbo-jvm -p turbo-goldens`
runs them on any machine:

- `analysis/core/tests/pipeline.rs` runs a pipeline of toy annotators
  over the shared layers: ordering by data, a normalized view whose
  tokens land on exact source bytes, a batch reaching a batching
  annotator in one call, failures, panics and skips per document, gold
  and predicted layers side by side and never read across, one type
  per id, and a pipeline from configuration.
- `analysis/core/tests/empty_spans.rs` maps empty spans at insertions
  and deletions, at both ends of the text and after the whole text is
  deleted, single offsets through a chain of two views, and every span
  of every alignment four `replace` calls can make over a three-byte
  text, checking that none panics.
- The unit tests of `analysis/core/src/bytes.rs` and
  `analysis/core/src/markup.rs` hold the byte view's checks (edits out
  of order or bounds, unchanged stretches that differ, lengths past
  `u32`) and the markup contract (`<p>Fish &amp; chips</p>`, with the
  "&" mapping to the five bytes of the entity, and every broken run
  refused). `cargo test -p turbo-encoding` adds the decoders, the
  detector against chardetng's tests, the mojibake repair against
  ftfy's tests and a windows-1252 HTML file through a markup view to
  its bytes; `analysis/encoding/README.md` lists them.
- `analysis/chunker/tests/chunker.rs` chunks hand-made layers: a cut
  where the topic shifts instead of where the budget fills, sections
  that are never crossed, and a paragraph break's token counted inside
  a chunk. The chunker's unit tests hold the optimizer against filling
  each chunk to the budget and against trying every cut, check the
  integer size penalty, the chunk term, that of equal cuts the earliest
  wins and that a huge penalty never lets a chunk past the maximum, and
  read the gaps a structure makes: each gap's strongest break, the
  units by their sentences, block edges inside sentences, which gaps
  units hold, and sections listed out of order.
- `analysis/chunker/tests/levels.rs` builds the levels from Markdown
  through the structure detector, the boundary costs and the chunker,
  and from hand-made chunks and sections: every unit's span read back as
  text, text before the first heading under the document, nested
  sections three deep, a chunk across a nested section's edge under the
  section around both, a section widened by a sentence past its end (a
  nested one the same however its text is chunked), empty sections and
  chunks, a document with no structure and one with no chunks, term
  counts at every level against the view's own, determinism, and
  overlapping chunks refused. The layer's unit tests hold its checks.
- `analysis/chunker/tests/limits.rs` holds each hard limit over
  hand-made layers: a table row, a code block and a quote that fit are
  never cut even where the costs say to, each one over the budget is cut
  and the chunk says why, a unit that fits inside one that does not is
  held, units that share sentences and chain over the budget are cut,
  a section's edge is a cut even inside a unit, and a sentence over the
  budget stands alone and keeps its reason, at a section start too. It
  refuses chunks outside their parents or not at a level above them. It
  checks that every chunk is the exact bytes of its
  sentences in text of one to four byte characters and, through a
  normalized view, in the source, and the costs `StructureBoundaries`
  writes.
- `analysis/chunker/tests/terms.rs` runs the term steps on hand-made
  tokens, structure and chunks, with the terms read from a case folded
  view: the vocabulary, a row per chunk, block and document with its
  lengths and df, BM25 scores against a plain textbook BM25 written in
  the test for four settings of k1 and b, Lucene's idf, an idf from a
  caller's corpus statistics against the document's own, ranking ties,
  full case folding that changes bytes, a token a unit cuts, units in
  another view and a view too far from the tokens. Two scores are
  worked by hand from Lucene 9's formula in the test's comments. It
  also holds a query term given twice, units with no terms left out of
  the local N and mean, a mean length of zero, a view whose edit crosses
  a token edge (refused) beside one whose edits do not, and the
  counter's refusals: tokens out of order and a layer of several units
  with no spans. The unit tests hold
  the interning, the sparse rows and the layer's checks.
- `analysis/structure/tests/fixtures.rs` holds hand-labelled documents,
  each one's whole structure written out block by block with the text
  each span covers: Markdown with nested sections, setext and ATX
  headings in one hierarchy, fenced code holding heading, list and table
  lines, unclosed fences, indented code, nested and mixed lists with
  tabs and lazy lines, what may interrupt a paragraph, items that start
  blank, tables with escaped pipes and pipes in code, quotes, CR LF and
  lone CR breaks, Unicode text and whitespace, empty and whitespace-only
  documents, front matter, and plain text: headings in capitals over
  their paragraphs, a letter's closing and signature, the `-- ` line and
  a year that is no list item.
- `analysis/structure/tests/exhaustive.rs` runs every document of up to
  three lines from 32 tricky lines, in both syntaxes, with each of the
  three line breaks and with and without a leading byte order mark, and
  holds each layer to the document's checks, to trimmed spans and to
  giving the same layer twice; it also runs ten thousand nested quotes,
  a thousand nested list items, twenty thousand nested list items under
  two hundred thousand blank lines (which must parse in under ten
  seconds; visiting every item on every blank line takes over a minute), a
  table row of 400 unclosed backtick runs and a long document.
- `analysis/structure/tests/pipeline.rs` runs the annotator ahead of the
  chunker, whose chunks then stay inside the sections it found, and
  builds it through the provider.
- `analysis/embed/tests/levels.rs` holds every local and contextual
  vector of the levels to the same unit pooled by hand on the small
  BERT, in one window and in parts, with and without structure, with
  empty units, and checks that the two kinds are named instances, that
  both under one name are refused, determinism, and the provider.
- `analysis/embed/tests/subwords.rs` runs the subwords annotator on the
  small BERT's WordPiece bundle, a byte-level BPE bundle and BGE-M3's
  Unigram, over the tokenizer test texts, and checks that the document
  takes every layer and that the ids are the tokenizer's. The WordPiece
  pieces are compared with upstream `tokenizers`, id for id and span for
  span. It also covers pieces of one character sharing its span, a text
  many times `max_seq` covered to its end, the overhead, pieces given
  their word tokens (a BPE piece over two tokens, punctuation in no
  token), tokens that cannot be aligned, and the provider. The crate's
  unit tests hold `align` on hand-made spans.
- `analysis/embed/tests/goldens.rs` compares the small BERT's WordPiece
  pieces with the `subwords` goldens of OpenNLP's `WordpieceEncoder`
  (["Against OpenNLP's WordPiece encoder"](#against-opennlps-wordpiece-encoder)):
  it checks that the vocabulary fixture is the bundle's, compares every
  golden through `matches_golden`, and has a test for each difference.
- `analysis/embed/tests/vectors.rs` runs the vector steps on the small
  BERT on the CPU. The row is the tokenizer's own row for each prompt
  role, and its overhead the subwords annotator's; a static model's has
  none. A range over a whole row is the row's mean-pooled vector bit
  for bit, and normalizing it on the host is the core's normalization
  bit for bit. Every sentence and chunk vector (sentences alone, chunks
  alone, both) is held to the same range run alone in a session of its
  own, the parts of a split item combined by hand: in one window,
  across several windows with runs smaller than a window's ranges, for
  a sentence that crosses a window's end, for chunks longer than a
  window, for a chunk over text no sentence covers, and for three
  documents in one batch against each alone. Sentence vectors are the
  same bits with or without the chunks beside them, and a hierarchy of
  chunks short and long adds no window to the sentences'. Items with
  no pieces get the zero vector and are listed in the layer's `empty`,
  and empty layers stay empty. It also checks provenance (the layer
  that lays out the windows included), the provider, subwords of
  another model and sentences in another view, malformed input to
  `embed`, a model pooled by its first token, and rows with no room for
  text. `analysis/embed/src/window/tests.rs`'s unit tests hold the policy on hand-made spans,
  pieces of no bytes included; `analysis/embed/src/encoder/tests.rs`'s hold the result's shape
  and the refusal of a backend that pools no ranges. One test runs the steps on the device
  `TURBO_TEST_DEVICE` names (a backend name or a device index), at the
  precision `TURBO_TEST_PRECISION` names, and holds the vectors to the
  CPU's within the tolerance the core gives that dtype; without the
  variable it says it was skipped. On a CUDA machine:
  `TURBO_TEST_DEVICE=cuda cargo test -p turbo-analysis-embed --features turbo/cuda --test vectors on_the_device`.
- `analysis/core/tests/offsets.rs` converts offsets in both directions
  at every character boundary of text with one to four byte
  characters, refuses offsets inside a surrogate pair and past the end,
  and checks a text of about ten thousand bytes against counts made
  with `encode_utf16` and `chars`, by point and by cursor.
- `analysis/jvm/tests/jvm_vectors.rs` compares `turbo-jvm` with the
  results a JVM wrote, bit for bit.
- `analysis/goldens/tests/goldens.rs` reads a hand written golden file
  (`tests/fixtures/zurich.golden`, over a short input beside it) into
  typed fields, as it does a second with options and a normalized view
  (`tests/fixtures/ligature.golden`), writes each back byte for byte,
  and compares documents built to match them, layer by layer.
  Deliberate differences (a span shifted by a byte, a changed tag, a
  value just outside the tolerance, a missing row, a golden layer with
  the wrong columns, another language code, tags without probabilities,
  a view's text or edits) each fail with a report naming the item, its
  text and both values, and a malformed file of each kind is refused
  with its line. The unit tests hold the escapes, the FIPS 180-4
  test vectors and Java's `Double.toString`.
- `analysis/goldens/tests/load.rs` loads the same fixture as test
  inputs, runs a table lookup tagger on the golden tokens and compares
  its tags with the golden's POS layer (and finds a changed entry). It
  also checks that every loaded layer matches its own golden, that
  golden tokens get the sentence they are in, that two
  names give two instances, and that a missing anchor, a text that is
  not the input, a token straddling two sentences, an entity token
  range past the tokens and a block with
  the wrong columns are each refused.
- `analysis/goldens/tests/jvm_error.rs` reads, writes and compares a
  golden with an error record (`tests/fixtures/newline-first.golden`):
  the message's escapes, a layer block or any other line after the
  record refused with its line, layers beside it refused by the writer,
  and `Mismatch::JvmError` from a comparison.
- `analysis/goldens/tests/utf16_spans.rs` reads and writes a golden
  whose tokens cut a surrogate pair (`tests/fixtures/smile.golden`),
  checks the UTF-16 layer's shape and spans, shows one Rust token over
  the character as a count mismatch against the golden's two rows, and
  compares `Utf16Spans` for tokens and sentences, a span and a cut flag
  that differ each reported with the token's text.
- `analysis/goldens/tests/shapes_from_other_crates.rs` defines a layer
  type in the test (`turbo:word-types`), reads a golden holding it with
  `read_with` (which `read` refuses), and compares a document with it.
- `analysis/goldens/tests/checked_in.rs` reads every checked in
  tokenizer, sentence detector and detokenizer golden with
  `segmentation_shapes()`, checks its input and writes it back byte for
  byte, and reads every dump, checking that its header is its golden's
  and that the sentence detector's `acceptable` records are on exactly
  the points whose best outcome is a split. `dump.rs`'s unit tests hold
  the dump reader on hand written dumps, malformed ones included.
- `analysis/goldens/src/fuzz/mod.rs` and `analysis/goldens/src/invariants/mod.rs`
  hold their own unit tests: the corpus is deterministic and an input
  does not depend on the count, the texts are varied, and no text holds
  a noncharacter or one of a sample of code points assigned after
  Unicode 15.0, the container's JDK's version; the invariants pass on well formed spans in
  text of every character width and in a normalized view, and catch a
  span cutting a character or leaving the text, a view whose alignment
  lies, and a broken span in a named instance of any layer type.
- `analysis/fuzz/tests/invariants.rs` runs every run of the
  differential over forty generated inputs and every checked in golden
  input but the sets it names as left out, and holds every layer to the
  offset invariants;
  `analysis/fuzz/tests/checked_in.rs` runs the differential over the
  checked in goldens, which must show no divergence
  (["Differential fuzzing"](#differential-fuzzing)). The unit tests of
  `turbo-fuzz` hold the run catalog to the directories on disk, the
  reducer on hand made oracles and the WordPiece mapping on texts with
  final sigma, controls, no-break spaces, ideographs, punctuation and
  reordered marks.
- The tests of `turbo-tokenize` and `turbo-sentdetect` read their
  goldens and dumps from `analysis/goldens/data` through
  `turbo_goldens` (`docs/segmentation.md`, "Golden files"). The tests of
  `turbo-postag` load each `postag` golden's sentences and tokens with
  `load_document`, tag them, and compare the tags with
  `matches_golden_key` (`analysis/postag/README.md`); the tests of
  `turbo-lemmatizer` do the same with the tokens and tags of each
  `lemmatize` and `lemmatize-dictionary` golden
  (`analysis/lemmatizer/README.md`), and those of `turbo-phrase-chunker`
  with the tokens and tags of each `chunk` golden
  (`analysis/phrase-chunker/README.md`).
- A doctest on `Ix` checks that an index into the wrong layer does not
  compile.

## Not built yet

- The annotators themselves: normalizers, sentence detection,
  tokenization, and the boundary signals besides structure and drift
  (lexical cohesion, coreference, discourse markers).
- A comparison of the sentence vectors with OpenNLP's
  `SentenceVectorsDL` on golden inputs, which needs the JVM container,
  and a measurement of documents per second through the vector steps.
  The tests build their layers by hand or with toy annotators.
- The rest of the evaluation harness: baseline records for
  `legalbench-rag-mini`, which is placed by hand, a published copy and
  licence for HiCBench, and the chunker's records on it. It is also where the BM25 only baseline
  is to be reported, from the term statistics above, and what each
  level and each kind of level vector adds.
- A gRPC surface, to be shared with the OpenNLP standard.
- A reader for spaCy's JSON pattern files, building `Tok` patterns from
  them.

### Using linguistic terms for BM25

`turbo_chunker::terms::NormalizedTermIndexer { form: TermForm::Folded }`
reads `turbo_analysis_layers::Terms` and `Tokens`, and writes `Vocabulary`
and `IndexedTerms`. Put it after `turbo_stem::TermAnnotator`, then use the
existing `TermCounter<L>` and `Query` APIs. Select `Surface`, `Folded`,
`Stem` or `Lemma`; a requested stem or lemma column must exist. Query
terms must use the same profile and form. Empty or punctuation-only
forms receive no retrieval ID.

`IndexedTerms` has layer ID `turbo:indexed-terms`; the old Rust name
`turbo_chunker::terms::Terms` is an alias. This changes the retrieval
layer's key from `turbo:terms`, which remains the linguistic text layer.
Consumers of textual layer keys must update the retrieval key. The raw
`TermIndexer` continues to index token text from a view and can coexist
with linguistic terms.

## Trainable dense classifiers

[Dense classifiers and feature selection](dense-classifier.md) describes Adam
training over typed vectors, temperature calibration, and MI/Chi2 selection
before document categorizer training.
