# turbo-goldens: making the golden layer files

The golden files in `analysis/goldens/data` record what OpenNLP 3.x
computes on the JVM for fixed inputs and models. This directory holds
everything that makes them:

- `java/`: the generator, a Maven project that depends only on the OpenNLP
  runtime, the OpenNLP formats module and the `numeric` module of the OpenNLP addons. It carries
  its own SLF4J provider (`CapturingLogging`), which keeps every logger off
  except the one a golden captures while it runs the annotator (the QN
  trainer reports its iterations only through debug logging). It runs the
  annotator runs listed below over their input sets and writes one golden
  per run and input; for the `num` set it runs `NumberScan` of the addons
  module, and for the `money` set `MoneyAnnotator` with
  `CursorMoneyExtractor` and `MoneyConversionAnnotator` with `EcbFxRates`,
  for the `quantity` set `QuantityAnnotator` with
  `CursorQuantityExtractor`, and for the `temporal` set `TemporalAnnotator`
  with `CursorTemporalExtractor` and `DocumentDateAnnotator`, and for the
  `pii` set `PiiAnnotator` with each extractor of `PiiPacks` (see below). With `--dumps` it also writes a decision point dump
  beside each golden of a run that asks a maxent model (the sentence
  detector, the tokenizer, the part-of-speech tagger, the maxent
  lemmatizer, the chunker and the language detector): every context the annotator gave
  the model and every outcome probability `eval` returned. `PosAccuracy` is a second main class, run by
  hand, that scores a part-of-speech model on a CoNLL-U file
  (`analysis/postag/README.md`).
- `container/Dockerfile`: the container the generator runs in. It starts
  from `eclipse-temurin:21-jdk` pinned by digest, clones the OpenNLP fork
  at a pinned commit, builds the runtime and the formats module with the
  repository's `mvnw`,
  clones `opennlp-addons` at a pinned commit and installs its `numeric`
  module, and builds the generator.
- `src/`: the driver, the `turbo-goldens` binary. `src/recipe.rs` holds the
  base image digest, the OpenNLP and addons commits, the published models
  with their URLs and SHA-512 hashes, the checked in fixtures with their
  SHA-512 hashes, and the input sets. `src/fuzz.rs` is the `fuzz`
  command, which is why the crate depends on `turbo-goldens` (the corpus
  generator) and `turbo-fuzz` (the Rust side of the comparison); the
  other commands use neither.

Both formats are described in `docs/analysis.md` ("Generating golden
files" and "Decision point dumps").

## Commands

Run from the repository's root. Each needs Docker except `fetch-models`.

```
cargo run -p turbo-goldens-tool -- image
cargo run -p turbo-goldens-tool -- fetch-models <dir> [--local]
cargo run -p turbo-goldens-tool -- generate [--out <dir>] [--models <dir>]
cargo run -p turbo-goldens-tool -- verify [--models <dir>]
cargo run -p turbo-goldens-tool --features fuzz -- fuzz [--seed <n>] [--count <n>] [--sets <a,b>] [--out <dir>] [--models <dir>] [--image <tag>] [--reduce <n>] [--no-reduce] [--reduce-documented] [--rounds <n>] [--reuse]
```

- `image` builds the container image, tags it with the OpenNLP commit and
  prints its id.
- `fetch-models <dir>` downloads the published models into `<dir>` with `curl`
  and checks each one's SHA-512 against the value published beside it on
  downloads.apache.org. A model already there with the right hash is kept.
  With `--local` it also downloads the OpenNLP 1.5 English models
  (`LOCAL_MODELS` in `src/recipe.rs`) from opennlp.sourceforge.net, which
  publishes no hash beside them: the recipe has the SHA-512 of the files
  whose SHA-256 the sequence crate's README lists. Their training corpora
  are LDC corpora or undocumented, so the `ci` workflow does not fetch
  them (the `models` workflow fetches `en-pos-maxent.bin` and
  `en-chunker.bin`, only to score them in the MASC release's gates);
  `generate`, `verify` and `fuzz` always do, since the generator runs
  them.
  It also fetches LibreOffice's `en_US`, `de_DE_frami` and `hu_HU` Hunspell
  dictionaries (`HUNSPELL_FILES`) into `<dir>/hunspell/`, each checked
  against the SHA-512 in the recipe.
  It also fetches the MeCab dictionaries (`DICTIONARIES` in `src/recipe.rs`)
  as archives, checks each archive's SHA-512 (neither project publishes
  one; these are the hashes OpenNLP's dictionary catalog test resource
  lists), and unpacks it with `tar` into a directory of the archive's
  name. For IPADIC, which is EUC-JP, it also writes a UTF-8 copy of the
  files a dictionary reader reads into `<dir>.utf8` with `iconv`, then
  replaces `U+2015` with `U+2014`, because Java's `EUC-JP` decoder reads
  0xA1BD as `U+2014` where `iconv` reads `U+2015`. The generator checks
  that each file of the copy is the text Java decodes from the original.
  Unpacking and the copy go through a `.part` directory renamed when
  complete; a directory already there is kept.
- `generate` fetches the models (into `target/turbo-goldens/models` unless
  `--models` names a directory), checks that `analysis/goldens/data/fixtures`
  holds exactly the recipe's fixtures with their SHA-512, runs the container
  with `--network none`, the inputs, the models, the fixtures and the rate
  snapshot's directory (at `/rates`) mounted read only, and replaces the
  annotator directories in `analysis/goldens/data` (or `<dir>`) with what
  it wrote. The generator is always given `--dumps`
  (`GENERATOR_ARGS` in `src/recipe.rs`), so the goldens and their dumps are
  made and installed together; before installing, every sentence, token,
  part-of-speech, maxent lemmatizer, chunk and language golden whose header names a model must have a dump beside it
  whose header parses, except a golden with an error record (the annotator
  threw on that input), which must have none. A golden that names no model
  (a tokenizer or sentence detector without one, a normalizer) has no dump,
  and neither has a detokenizer, number, money, quantity, temporal, PII,
  unigram or lattice golden.
- `verify` generates into a scratch directory and compares every golden
  and dump byte for byte with the checked in ones. It lists each file that
  differs, is missing or is extra, and exits non zero if there is any. It
  also checks the checked in dumps by the same rule, and that each one's
  header names the golden's input and model; it does not read the
  decision points.
- `fuzz` generates `<count>` texts from `<seed>`, runs the generator over
  them in the container, compares every golden with the Rust annotators
  and reduces each divergence to a minimal input; see "Fuzzing against
  the Rust ports" below.

## Runs

The golden path is `<annotator>/<model>/<set>/<id>.golden`. `<model>` is
the first model file's name without `.bin` (or `.xml` for a dictionary,
`.txt` for a vocabulary).
A run with options has `+` and their names appended, so it does not share
a directory with the same model run without them, and its options are
`config` records. A run without a model names itself there instead.
`Runs.java` lists every run:

| Annotator | `<model>` | Input sets | `config` |
|---|---|---|---|
| `sentdetect` (`SentenceDetectorME`) | `opennlp-en-ud-ewt-sentence-1.3-2.5.4` | `en`, `multi`, `edge` | |
| | `opennlp-de-ud-gsd-sentence-1.3-2.5.4`, `opennlp-pt-ud-gsd-sentence-1.3-2.5.4` | `edge` | |
| | `en-abbrev`, `en-plain`, `en-plain-rust`, `en-plain-qn-rust`, `en-plain-perceptron-rust` (fixtures), each also `+legacy` | `sentence-fixture` | `whitespace legacy` |
| `sentdetect` (`NewlineSentenceDetector`) | `newline`, `newline+legacy` | `edge` | `whitespace legacy` |
| `tokenize` (`TokenizerME`) | `opennlp-en-ud-ewt-tokens-1.3-2.5.4` | `en`, `multi`, `edge`, `surrogate` | |
| | the `de`, `es`, `fr` and `pt` tokenizer models | `edge`, `surrogate` | |
| | `tokenizer-abbreviations` (fixture) | `abbreviations`, `edge` | |
| | `opennlp-en-ud-ewt-tokens-1.3-2.5.4+english-dictionary` | `abbreviations`, `edge` | |
| | each of the five models `+alphanumeric` | `edge`, `alphanumeric` | `useAlphaNumericOptimization true`, `alphaNumericPattern <pattern>` |
| `tokenize` (`SimpleTokenizer`, `WhitespaceTokenizer`) | `simple`, `whitespace`, each also `+keep-new-lines`, `+legacy` and `+keep-new-lines+legacy` | `edge` | `keepNewLines true`, `whitespace legacy` |
| `tokenize` (UAX #29 `WordTokenizer`) | `uax29` | `edge` | |
| `detokenize` (`DictionaryDetokenizer`) | `en-detokenizer`, `fr-detokenizer`, `latin-detokenizer`, `pt-detokenizer`, `quotes-detokenizer` (fixtures) | `detokenizer`, the inputs whose id starts with the dictionary's name | `splitMarker <SPLIT>` |
| `postag` (`POSTaggerME`) | `opennlp-en-ud-ewt-pos-1.3-2.5.4` | `en`, `multi`, `edge` | |
| | `opennlp-en-ud-ewt-pos-1.3-2.5.4+penn` | `en` | `format PENN` |
| | `opennlp-en-ud-ewt-pos-1.3-2.5.4+beam1`, `+beam10` | `en` | `beamSize` |
| | `opennlp-de-ud-gsd-pos-1.3-2.5.4` | `edge` | |
| | `pos-fixture`, `pos-fixture-rust` (fixtures) | `en`, `edge` | |
| `lemmatize` (`LemmatizerME`) | `opennlp-en-ud-ewt-lemmas-1.3-2.5.4` | `en`, `multi`, `edge` | |
| | `opennlp-de-ud-gsd-lemmas-1.3-2.5.4` | `edge` | |
| | `lemma-fixture-rust` (fixture) | `en`, `edge` | |
| `lemmatize-dictionary` (`DictionaryLemmatizer`) | `lemmatizer-dictionary` (fixture) | `en`, `multi`, `edge` | |
| `chunk` (`ChunkerME`) | `chunker-fixture` (fixture) | `en`, `multi`, `edge` | |
| | `chunker-fixture+beam1` | `en` | `beamSize 1` |
| | `chunker-fixture-rust` (fixture) | `en`, `multi`, `edge` | |
| `langdetect` (`LanguageDetectorME`) | `langdetect-183`, `langdetect-fixture-rust` (fixture) | `en`, `multi` | |
| `normalize` (each normalizer) | the normalizer's name | `norm` | see below |
| `numbers` (`NumberScan` of the addons `numeric` module) | `latin-us`, `latin-eu` | `num` | `notation` |
| `money`, `money-convert` (`MoneyAnnotator`, `MoneyConversionAnnotator`) | `latin-us`, `latin-eu` | `money` | `notation`; for conversions also `target`, `as_of`, `snapshot` |
| `quantity` (`QuantityAnnotator`) | `latin-us`, `latin-eu` | `quantity` | `notation` |
| `temporal` (`TemporalAnnotator`, `DocumentDateAnnotator`) | `dateline`, `fixed` | `temporal` | `reference` |
| `pii` (`PiiAnnotator` over a `PiiPacks` extractor) | `payment`, `contact`, `network`, `secrets`, `crypto`, `us-identity`, `eu-identity`, `ca-identity`, `device`, `all-structured` | `pii` | `pack` |
| `subwords` (`WordpieceEncoder`) | `all-minilm-l6-v2-vocab` (fixture) | `en`, `multi`, `edge`, `surrogate`, `subwords` | `lowerCase true`, `maxWordCodePoints 100` |
| `bpe` (`BPETokenizerTrainer`, `BPETokenizer`) | `merges-40`, `merges-400` | `bpe` | `languageCode en`, `numMerges` |
| `train` (the data indexers and `GISTrainer`) | `<source>-<indexer>-cutoff<n>`, with `-unsorted` when sorting is off, then `-threads2`, `-smoothing`, `-smoothing0.5`, `-gaussian` or `-gaussian0.5` when the trainer is given that: nineteen runs over the sources `sentdetect`, `tokenize`, `realvalue` and `postag` | `en`, `multi`, `edge`; `sentdetect` and `postag` also `train` | `source`, `DataIndexer`, `Cutoff`, `sort`, `Iterations`, and when set `Threads`, `Smoothing`, `SmoothingObservation`, `GaussianSmoothing`, `GaussianSmoothingSigma` |
| `train` (`QNTrainer`) | the same, with `-qn` appended, then `-l1`, `-l2` or `-plain` when only one cost, or neither, is on: eight runs over `sentdetect`, `realvalue` and `postag` | `en`, `multi`, `edge`; one `sentdetect` run also `train`, one only `train` | the same with `Algorithm`, `L1Cost`, `L2Cost`, `NumOfUpdates`, `MaxFctEval`, `Threads` |
| `train` (`PerceptronTrainer`) | the same, with `-perceptron` appended, then `-raw` without averaging, `-skipped` with `UseSkippedAveraging`, `-decrease` with a `StepSizeDecrease`: five runs over `sentdetect`, `realvalue` and `postag` | `en`, `multi`, `edge`; one `sentdetect` run also `train`, one only `train` | `source`, `DataIndexer`, `Cutoff`, `sort`, `Algorithm`, `Iterations`, `UseAverage`, `UseSkippedAveraging`, `StepSizeDecrease`, `Tolerance` |
| `train` (`SimplePerceptronSequenceTrainer`) | the same, with `-perceptron-sequence` appended, then `-raw` without averaging: three runs over sequences of five events of `sentdetect` and `postag` | `en`, `multi`, `edge`; one `sentdetect` run also `train` | `source`, `DataIndexer`, `Cutoff`, `sort`, `Algorithm`, `Iterations`, `UseAverage` |
| `train-events` (`SDEventStream`, `TokSpanEventStream` with factory options) | `sentdetect-dictionary-pt`, `sentdetect-th`, `sentdetect-conllu`, `tokenize-dictionary-alnum-de`, `tokenize-conllu` | `en`, `multi`, `edge`, `train`, `abbreviations`, `alphanumeric`, `formats` by run | `run`, `source`, `language`, `format`, and when given `abbreviations`, `useTokenEnd`, `eosCharacters`, `useAlphaNumericOptimization`, `alphaNumericPattern`, `sentencesPerSample` |
| `train-events` (`POSSampleEventStream`, `LemmaSampleEventStream`) | `postag-wordtag`, `postag-wordtag-tagdict`, `postag-conllu-u`, `postag-conllu-x`, `lemma-text`, `lemma-conllu-u` | `tagged`, `formats` by run; the `wordtag`, `lemma` or `conllu` inputs by source and format | `run`, `source`, `language`, `format`, and when given `tagset`, `tagDictionary` |
| `namefind` (`NameFinderEventStream`, `NameFinderME.train`) | `train-bio`, `train-bilou`, `train-type`, `train-perceptron`, `train-qn` | `tagged`, `formats`: the inputs whose id starts with `name` | `run`, `language`, `codec`, and when given `type`, `algorithm` |
| `namefind` (`NameFinderME`) | `ner-fixture-rust` (fixture) | the same | `run find` |
| `train-events` (`ChunkerEventStream`, `ChunkerME.train`) | `chunker-defaults`, `chunker-perceptron`, `chunker-qn`, `chunker-perceptron-sequence` | `tagged`, `formats`; the `chunk` inputs | `run`, `source`, `language`, and every training parameter the run puts (`Algorithm`, `Iterations`, `Cutoff`, `TrainerType`, `BeamSize`) |
| `train-langdetect` (`LanguageDetectorME.train`) | `default`, `cutoff1` | `langdetect`, and the `formats` inputs whose id starts with `language-` | `run`, `Algorithm`, `Cutoff`, `DataIndexer`, `Iterations` |
| `unigram` (`UnigramSegmenter`) | `zh-unigram-fixture` (fixture), also `+legacy` | `zh`, `multi`, `edge`, `surrogate` | `whitespace legacy` |
| | `jieba-dict-subset` (fixture), also `+legacy` | `zh`, `multi`, `edge` | `whitespace legacy` |
| | `zh-unigram-modes` (fixture), also `+legacy` | `zh` | `whitespace legacy` |
| `lattice` (`LatticeTokenizer`) | `mecab-fixture-ja` (fixture), also `+legacy` | `ja`, `multi`, `edge`, `surrogate` | `whitespace legacy` |
| | `mecab-fixture-ko` (fixture) | `ko`, `multi`, `edge` | |
| | `mecab-fixture-partial` (fixture) | `ja`, `edge` | |
| | `mecab-ipadic-2.7.0-20070801` | `ja`, `multi`, `edge`, `surrogate` | |
| | `mecab-ko-dic-2.1.1-20180720` | `ko`, `multi`, `edge`, `surrogate` | |
| `formats` (the native sample streams) | `name`, `wordtag`, `token`, `doccat`, each also `+legacy`; `lemma`, `chunk`, `sentence`, `language` | `formats`, the inputs whose id starts with the format | `whitespace legacy` |
| `formats` (the conversion streams) | `name-to-token`, `name-to-sentence`, `pos-to-token`, `pos-to-sentence`, with `en-detokenizer` (fixture) | `formats`, the `name` and `wordtag` inputs | `chunkSize 2` for the sentence conversions |
| `evaluate` (the evaluators) | `pos`, `lemma`, `name`, `chunk`, `token`, `sentence` (also `+legacy`), `doccat`, `language`, `cross-validation` | `formats`, the `pairs-` inputs of the evaluator's format | `whitespace legacy` |
| `formats` (the CoNLL-U streams) | `conllu-pos-u`, `conllu-pos-x`, `conllu-lemma-u`, `conllu-lemma-x`, `conllu-sentence`, `conllu-token` (also `+legacy`) | `formats`, the `conllu` inputs | `tagset u` or `x`; `sentencesPerSample 2`; `whitespace legacy` |
| `formats` (the CoNLL streams) | `conll02-nld`, `conll02-spa`, `conll02-spa-per-loc`, `conll03-eng`, `conll03-deu`, `conll03-eng-org-misc`, `conllx-pos`, `conllx-token` and `conllx-sentence` (with `en-detokenizer`, fixture) | `formats`, the `conll02`, `conll03` and `conllx` inputs | `lang`, `types`; `chunkSize 2` for `conllx-sentence` |
| `masc` (the MASC readers) | `documents` (`MascDocument.parseDocument`), `pos`, `token`, `sentence`, `name` (the sample streams) | `formats`, the `masc-` inputs | `sentencesPerSample 2` for `sentence` |
| `ner` (the named entity corpus streams) | `germeval-outer`, `germeval-inner`, `germeval-outer-loc-misc`, `bionlp`, `bionlp-cell-line`, `bionlp-dna-cell-type`, `evalita`, `evalita-per-gpe`, `census`, `muc6-whitespace` (also `+legacy`), `muc6-simple`, `brat-whitespace` (also `+legacy`), `brat-simple`, `brat-recursive-person-location`, `brat-config` | `ner`, the inputs whose id starts with the corpus's prefix (`germeval-`, `bionlp-`, `evalita-`, `census-`, `muc-`, `brat-`) | `types`, `layer`, `lang it`, `tokenizer`, `recursive true`, `nameTypes`, `whitespace legacy` |
| `parse` (`ParseSampleStream`, `Parse.parseParse`) | `treebank`, `treebank+function-tags`; `en-head-rules`, `en-head-rules+possessives`, `es-head-rules`, `es-regex-head-rules` (fixtures) | `parse` | `useFunctionTags true`; `rules`, `fixPossesives true` |
| `head-rules` (`HeadRules`, `AncoraSpanishHeadRules`) | `en`, `es` | `head-rules` | |
| `gazetteer` (the addons `geo` module) | `naturalearth-populated-places` (the bundled table), `gazetteer-geonames`, `gazetteer-overture`, `gazetteer-user`, `overlay` (fixtures); `load-geonames`, `load-overture`, `load-user`, `load-suppressions` | `gazetteer`: the `names`, `ids` and `regions` inputs for the first five, the `table-<kind>-` inputs for the load runs | `source sites` for the user table |
| `place-profiles` (`PlaceProfiles` of the addons `geo` module) | `place-profiles` | `place-profiles` | `profilesCommit` |
| `string-pattern` (`StringPattern`, `FeatureGeneratorUtil.tokenFeature`) | `utf16-units`, `code-points` | `sequence` | `mode`; `branchCommit` for `code-points` |
| `features` (`GeneratorFactory`, `DefaultPOSContextGenerator`) | `pos-default`, `ner-default`, `everything`, `nesting` (fixtures, `featuregen/`), `default-pos-context`, `default-pos-context+names-dictionary`; `errors` | `sequence` (the `sentences` input); `errors` on `descriptor-errors` | `adaptiveData` for the descriptors |
| `codec` (`BioCodec`, `BilouCodec`) | `bio`, `bilou` | `codec`, the input of the codec's name | `compatibleOutcomes` |
| `beam` (`BeamSearch` as `POSTaggerME`, `ChunkerME` and `NameFinderME` wire it) | `pos-fixture`, `pos-fixture+default-context`, `chunker-fixture`, `ner-bio-fixture`, `ner-bilou-fixture`, `ner-qn-fixture`, `ner-qn-fixture+fresh-sums` (fixtures); `opennlp-en-ud-ewt-pos-1.3-2.5.4`; `en-pos-maxent`, `en-pos-perceptron`, `en-chunker`, `en-ner-person`, `en-ner-location` (OpenNLP 1.5) | `sequence` (the `sentences` input) | `kind` |
| `tag-dictionary` (`POSDictionary`) | `tags`, `tags-cases`, `tags-exact`, `pos-fixture` (fixtures); `opennlp-en-ud-ewt-pos-1.3-2.5.4`, `en-pos-maxent`, `en-pos-perceptron` | `sequence` (the `sentences` input) | |
| `tag-format` (`POSTagFormatMapper`) | `pos-fixture` (fixture), `opennlp-en-ud-ewt-pos-1.3-2.5.4`, `en-pos-maxent`, `en-pos-perceptron`; `guesses` | `sequence` (the `sentences` input); `guesses` on `tag-format` | |
| `namefind` (`DictionaryNameFinder`, `RegexNameFinder`, `NameFinderME.dropOverlappingSpans`) | `generated`, `dictionary`, `regex-builtin`, `regex-factory`, `regex-user`, `merge`, with the fixtures in `namefind/` | `namefind` | |
| `stem` (`PorterStemmer`, `SnowballStemmer`, the light and minimal stemmers) | `porter`, `snowball-<language>`, `snowball-<language>-repeat2`, `<language>-light`, `<language>-minimal`, and the Norwegian ones by variety | `stem`: `edge`, the stemmer's language and `generated-<language>` | `stemmer` |
| `hunspell` (`HunspellDictionary`, `HunspellStemmer` of the `OPENNLP-1927-hunspell-compatibility` branch) | `en_US`, `de_DE_frami`, `hu_HU` (downloaded); `fixture-<name>` for each fixture in `hunspell/`, and `fixture-unsupported-directives+partial` | `hunspell`: `edge` and the run's own; `stem`: the Snowball sample of the dictionary's language | `mode`, `branchCommit` |
| `stopword` (`StopwordLists`, `StopwordFilter`, `StopwordFilteringTokenizer`) | `lists`, `codes`, `words`, `filter`, `letters` | `stopword`: `codes` for `codes`, `words` for the others | |
| `glossary` (`AhoCorasickGlossaryMatcher`, `TermAnalyzingGlossaryMatcher`, `CompositeGlossaryMatcher`, `CsvGlossaryReader`, `TbxGlossaryReader`) | `exact`, `ignore-case`, `whitespace+ignore-case`, `analyzed`, `composite` (the fixture `glossary-terms.tsv`); `csv`, `tsv-header`, `tbx-en`, `tbx-de` | `glossary`: the `text-` inputs for the matchers, the `csv-`, `tsv-` and `tbx-` inputs for the readers | `ignoreCase`, `normalizer`, `analyzer`; `delimiter`, `skipHeader`; `languageTag` |
| `svm-doccat` (`DocumentCategorizerSVM.train`, `categorize`) | `topics-linear-<weighting>-<selection>` for the four weightings (`binary`, `tf`, `tfidf`, `lognorm`) and five selections (`none`, `ig`, `chi2`, `tf`, `df`); `topics-rbf-tfidf`, `topics-poly-tf`, `topics-sigmoid-binary`, `topics-nu-linear-tfidf`, `topics-nu-rbf-lognorm`, `topics-linear-tfidf-noscale`, `topics-rbf-tfidf-scale-1`, `topics-linear-lognorm-ngram`, `topics-linear-binary-c10`, `pair-linear-tfidf`, `pair-rbf-binary` | `svm-doccat`; the run's corpus in `svm-doccat-train` | `run`, `corpus`, `language`, `seed`, `weighting`, `selection`, `max-features`, `svm-type`, `kernel`, `degree`, `gamma`, `coef0`, `cost`, `scaling`, `generators` |

- `+english-dictionary` is the English model rebuilt through
  `TokenizerModel(MaxentModel, Map, TokenizerFactory)` with the
  abbreviation dictionary `english-dictionary.xml`, which is its second
  `model` record. No published model has an abbreviation dictionary.
- `+alphanumeric` is the model rebuilt the same way with the alphanumeric
  optimization on and its own pattern; the `alphaNumericPattern` record is
  the pattern the tokenizer uses (`Factory.DEFAULT_ALPHANUMERIC` when the
  factory has none, which no published model lacks).
- The legacy runs set OpenNLP's `WhitespaceMode` for the call and put it
  back afterwards.
- A detokenizer input holds one token per line, and its golden holds the
  tokens, the operation `detokenize` gave each and the text it returned
  with and without the split marker (`docs/analysis.md`).
- A `bpe` run trains `BPETokenizerTrainer` on the input alone with
  `numMerges` merges and writes the merges (`turbo:bpe-merges`) and the
  tokens `BPETokenizer` gives for the same text with them
  (`docs/analysis.md`, docs/bpe-trainer.md).
- The sentence detector's dumps also record, for each point whose best
  outcome is a split, what `isAcceptableBreak` returned
  (`docs/analysis.md`, "Decision point dumps").
- A `train` run (`TrainGolden`) makes events from the input with
  `SDEventStream` (`sentdetect`, the settings of `SentenceDetectorME.train`
  with a default factory), `TokSpanEventStream` (`tokenize`, those of
  `TokenizerME.train`) or a real valued source (`realvalue`: one event per
  line, outcome `long` or `short` by the number of fields, each distinct
  field a context whose value is its count, read back through
  `RealValueFileEventStream.parseEvent`) or a part-of-speech shaped source
  (`postag`: one event per whitespace separated field of each line, the
  outcome `L` and the field's length in UTF-16 units, so a text has many
  outcomes, and the context `bias`, `w=` the field, `p=` and `n=` its
  neighbours, `pre=` and `suf=` its first and last two UTF-16 units, which
  may be half a surrogate pair); records the events, the CRC-32C
  event checksum and what the named `DataIndexer` made of them with the
  run's `Cutoff` and `sort` (contexts and predicate labels with each
  backslash doubled and each lone surrogate, which the tokenizer's features
  hold, as the six characters `\uXXXX`); then trains a GIS model on the index with
  `Iterations` iterations, the run's `Threads` (one, or two) and its
  smoothing (none, `Smoothing` with `SmoothingObservation`, or
  `GaussianSmoothing` with `GaussianSmoothingSigma`) and records every iteration's
  log likelihood and a hash of its parameters, the parameters after the
  first and the last iteration, and the hash of the model
  `GenericModelWriter` writes (`docs/analysis.md`, "Training goldens").
  An index with no event left makes the indexer throw
  `InsufficientTrainingDataException`; the golden then has the events and
  one `turbo:indexer-error` row with the exception's class and message in
  place of the indexer's layers. With more than one thread the JVM adds
  the chunks' log likelihoods in the order its threads finish; two chunks
  give the same sum either way, so a two thread run is reproducible, and
  no run uses more.
- A `train-events` run (`EventGolden`) makes the same two kinds of events
  with the factory options `TrainGolden` leaves at their defaults: a
  `SentenceDetectorFactory` or `TokenizerFactory` built as the trainer
  tools build them from a language, the English abbreviation dictionary
  fixture or none, and explicit or absent end characters or alphanumeric
  pattern, over the input read as one sample per line or as CoNLL-U
  (`ConlluSentenceSampleStream` with ten sentences a sample,
  `ConlluTokenSampleStream`). It records the events and the checksum,
  `turbo:stream-errors` (one row per exception the sample stream threw,
  with the events made before it; reading goes on), `turbo:manifest` and
  `turbo:package-entries` (the manifest without `Timestamp` and the zip
  entries of a model file built from the factory around a one-predicate
  model), `turbo:dictionary` (the file's dictionary entry) and, on the
  `train` input of the dictionary run, `turbo:dictionary-cases`
  (`Dictionary.serialize` of dictionaries built in the generator;
  `docs/analysis.md`, "Training goldens").
  `analysis/train-components/tests/goldens.rs` compares with them.
- A `train-events` run of `TaggerEventGolden` makes the events of
  `POSSampleEventStream` over the context generator a `POSTaggerFactory`
  with no descriptor makes (the default feature generators), without and
  with the tag dictionary fixture `tags.tagdict`, and of
  `LemmaSampleEventStream`, over the input read with
  `WordTagSampleStream`, `LemmaSampleStream`, `ConlluPOSSampleStream` or
  `ConlluLemmaSampleStream` in a tag set. It records the same layers,
  the model file being a `POSModel` or `LemmatizerModel` around a
  placeholder model whose outcomes are the dictionary's tags,
  `turbo:tagdict` (the file's `tags.tagdict` entry) and, on the `tagged`
  inputs of the dictionary run, `turbo:tagdict-cases`
  (`POSDictionary.serialize` of dictionaries built in the generator).
  A predicate of these sources holds a token, which the sample streams
  never let hold a space; the generator stops if one does, so the space
  separated `context` field is never ambiguous.
- A `train-events` run of `ChunkerTrainGolden` is `ChunkerME.train` with
  `new ChunkerFactory()` over the input read with `ChunkSampleStream`, as
  the trainer tool reads it: the events of `ChunkerEventStream` over the
  factory's `DefaultChunkerContextGenerator`, then the model trained. The
  runs are the training parameters: `TrainingParameters.defaultParams()`
  (`chunker-defaults`: `MAXENT`, 100 iterations, cutoff 5), the
  perceptron with cutoff 1 and a beam size of 5 (`chunker-perceptron`),
  `MAXENT_QN` with cutoff 1 (`chunker-qn`), and the sequence perceptron
  (`chunker-perceptron-sequence`), which `ChunkerME.train` refuses. It
  records the events, `turbo:stream-errors` and the checksum as the
  tagger's runs do, then either `turbo:train-error` (the class and
  message of what training threw) or the model: `turbo:model-entry`
  (the algorithm, and the `chunker.model` entry's length and SHA-256),
  `turbo:model-outcomes` and `turbo:model-predicates` (the labels in the
  model's order), `turbo:model-parameters` (every parameter with its
  predicate and outcome index), and `turbo:manifest` and
  `turbo:package-entries` of the model file.
  `analysis/train-components/tests/chunker.rs` compares with them.
- A `train-langdetect` run (`LanguageTrainGolden`) trains a language
  detector as `LanguageDetectorME.train` does, on the samples
  `LanguageDetectorSampleStream` reads from the input, with the
  parameters the command line trainer uses when given none (`MAXENT`, 100
  iterations) and the run's cutoff (5 for `default`, 1 for `cutoff1`).
  It records the events in `turbo:language-events` (outcome and number
  of predicates, a row each) and `turbo:language-predicates` (a row per
  predicate, since an n-gram can hold a space or a tab), then the
  checksum, the index and the GIS training as a `train` run does, and the
  manifest without `Timestamp` and the zip entries of the model
  `LanguageDetectorME.train` returns, after checking that its weights and
  its `langdetect.model` entry are those of the monitored training.
  `analysis/train-components/tests/langdetect.rs` compares with them.
- `en-plain-rust.bin` is the `.bin` the Rust trainer writes when it trains
  `en-plain.bin`'s events again (`analysis/train/tests/roundtrip.rs`), and
  `en-plain-qn-rust.bin` and `en-plain-perceptron-rust.bin` the ones it
  writes when it trains them with `MAXENT_QN` and `PERCEPTRON` (cutoff 1,
  the defaults; the `train` goldens `sentdetect-twopass-cutoff1-qn` and
  `sentdetect-twopass-cutoff1-unsorted-perceptron` are the JVM's trainings
  of the same). The JVM loads all three with `SentenceModel` like the other
  fixtures, so their goldens show what the JVM makes of a model file
  written here.
  `pos-fixture-rust.bin`, `lemma-fixture-rust.bin` and
  `chunker-fixture-rust.bin` are the `.bin` files the Rust trainer writes
  from `inputs/tagged/wordtag.txt` (with `tags.tagdict`),
  `inputs/tagged/lemmas.txt` and `inputs/tagged/chunks.txt`
  (`analysis/train-components/tests/fixtures.rs`); the JVM loads them with
  `POSModel`, `LemmatizerModel` and `ChunkerModel` and tags, lemmatizes or
  chunks with them.
  `langdetect-fixture-rust.bin` is the one it writes from
  `inputs/langdetect/samples.txt` with the default parameters (the same
  test); the JVM loads it with `LanguageDetectorModel` and detects with
  it, and `analysis/langdetect/tests/fixture.rs` compares with those
  goldens.
- A `namefind` training run (`NameEventGolden`) reads the input with
  `NameSampleDataStream` and makes the events of `NameFinderEventStream`
  over the context generator and codec of a `TokenNameFinderFactory` with
  the default descriptor, with or without a type given to every name; it
  records the events, the checksum and `turbo:stream-errors` as a
  `train-events` run does. It then trains with `NameFinderME.train` from
  a fresh read of the input with empty parameters (or only `Algorithm`),
  and records `turbo:name-training` (the exception that ended training,
  or an empty row), `turbo:name-parameters` (the parameters afterwards,
  with their Java classes: the defaults `NameFinderME.train` puts in and
  the trainer's `sort`) and, for a trained model, `turbo:manifest`,
  `turbo:package-entries`, `turbo:name-model` (the SHA-256 and length of
  `nameFinder.model`) and `turbo:name-weights` (every predicate, outcome
  and weight as `GenericModelReader` reads the entry back).
  `ner-fixture-rust.bin` is the `.bin` the Rust trainer writes from
  `inputs/tagged/names.txt`; `NameFinderFixtureGolden` loads it with
  `TokenNameFinderModel` and records `turbo:name-found` and
  `turbo:name-probs`, the names `find` returns and `probs()` for each
  sample, clearing the adaptive data where a sample follows an empty
  line.
- A `postag` run cuts the input with the English (or, for the German
  model, German) sentence and tokenizer models, which are its second and
  third `model` records, and tags each sentence with `tag`; its golden has
  the sentences, the tokens and `opennlp:pos` aligned to the tokens, with
  `probs()` as the probability column. `+penn` asks the tagger for
  `POSTagFormat.PENN`; `+beam<n>` rebuilds the model through
  `POSModel(String, MaxentModel, int, Map, POSTaggerFactory)` with that
  beam size and the model's own descriptor and tag dictionary. The dumps
  record every context the beam search asked for, by wrapping the context
  generator (`RecordingPOSTaggerFactory`), at the start of the token it is
  about.
- A `lemmatize` run cuts and tags the input with the sentence, tokenizer
  and part-of-speech models of its language (its second, third and fourth
  `model` records) and lemmatizes each sentence with `lemmatize`; its
  golden has the sentences, tokens and tags, `opennlp:lemmas` and
  `turbo:confidence<opennlp:lemmas>` from `probs()`, and its dumps are
  made like the tagger's (`RecordingLemmatizerFactory`). A
  `lemmatize-dictionary` run does the same with `DictionaryLemmatizer`
  over the fixture dictionary and the English pipeline, without a
  confidence layer; it asks no model, so it has its own annotator
  directory and no dumps.
- A `chunk` run cuts the input with the English sentence and tokenizer
  models and tags it with the English part-of-speech model asked for
  `POSTagFormat.PENN` (its second, third and fourth `model` records), since
  the fixture chunker was trained on Penn Treebank tags, and chunks each
  sentence with `chunk`; its golden has the sentences, tokens and tags,
  `opennlp:chunk-tags` with `probs()` as the probability column, and
  `opennlp:chunks`, the phrases `ChunkSample.phrasesAsSpanList` reads off
  the tags (what `chunkAsSpans` returns), each spanning its tokens' text,
  with its token range and kind. `+beam<n>` rebuilds the model through
  `ChunkerModel(String, MaxentModel, int, Map, ChunkerFactory)` with that
  beam size. The dumps are made like the tagger's
  (`RecordingChunkerFactory`). The fixture is a perceptron model, so the
  generator depends on `opennlp-ml-perceptron`, which the runtime lists as
  optional.

The fixtures (`analysis/goldens/data/fixtures`) are checked in, mounted at
`/fixtures` and listed with their SHA-512 in `src/recipe.rs`; the data
directory's README says where each comes from.

## WordPiece goldens

`subwords/all-minilm-l6-v2-vocab/<set>/<id>.golden` is what OpenNLP's
`WordpieceEncoder` returned for the input, built from the fixture
`all-minilm-l6-v2-vocab.txt` (the uncased BERT vocabulary of
`testdata/all-minilm-l6-v2/tokenizer.json`, one piece per line, the line
number being the id) with lower casing and accent stripping on and a
limit of 100 code points per word: the BERT normalization and the greedy
longest-match segmentation. The layer is `opennlp:subwords`, one row per
piece with its UTF-8 byte span and its id. `encode` also returns `[CLS]`
first and `[SEP]` last with empty spans at the ends of the text; the
generator checks them and leaves them out, since they are not pieces of
the text. The encoder asks no model, so there are no dumps.

OpenNLP's `WordpieceTokenizer` is not what runs: at the commit in `src/recipe.rs` it
does no normalization, and its `tokenizePos` throws, so it has no spans.
`WordpieceEncoder` is the class with the BERT normalization, ids and
spans.

The `subwords` inputs are written for the project to reach the corners
of the BERT normalization: accents precomposed, combining and out of
canonical order, final sigma, the dotted capital I, Hangul, kana and CJK
(and the CJK Extension E ideographs where the reference BERT and
upstream `tokenizers` disagree), format characters, punctuation, spacing
marks, pieces with no entry, special token strings, compatibility
characters and words at the 100 code point limit.

## Unigram goldens

`unigram/<lexicon>/<set>/<id>.golden` is what OpenNLP's
`opennlp.tools.tokenize.lattice.UnigramSegmenter` returned from
`tokenizePos` for the input, with the frequency lexicon `<lexicon>.txt`
of the fixtures (one word, its count and an optional tag per line, read
as UTF-8). The layer is `opennlp:tokens`, one row per word with its UTF-8
byte span; a word is always whole code points, so there is never a
`turbo:utf16<opennlp:tokens>` layer, and the generator stops if one
would be needed. `StringUtil.isWhitespace` decides both where a lexicon
line's word ends and where the text is split, so a `+legacy` run loads
its lexicon and segments with the legacy mode active. The segmenter asks
no model, so there are no dumps; its goldens are under their own
annotator directory so that the rule that every `tokenize` golden naming
a model has a dump still holds.

## Lattice goldens

`lattice/<dictionary>/<set>/<id>.golden` is what OpenNLP's
`opennlp.tools.tokenize.lattice.LatticeTokenizer` returned from `analyze`
for the input, with the MeCab dictionary directory `<dictionary>` (a
fixture directory, or a dictionary `fetch-models` unpacked). The layers
are `opennlp:tokens`, one row per morpheme with its UTF-8 byte span, and
`turbo:morphemes`, aligned to it: `unknown` (1 for a morpheme that
unknown word handling made) and `features`, the entry's feature columns
joined with commas. A morpheme is always whole code points, so there is
never a `turbo:utf16<opennlp:tokens>` layer, and the generator stops if
one would be needed. Where `analyze` threw (a dictionary whose `unk.def`
has no `DEFAULT` template), the golden has the error record and no
layers.

A dictionary is a directory, so its `model` record is the directory's
name and the SHA-512 of a manifest of the files `MecabDictionary.load`
reads, in the order it reads them (`matrix.def`, the `*.csv` files
sorted by name, `char.def`, `unk.def`): for each, its name, a tab, the
SHA-512 of its bytes in lowercase hex and a line feed. IPADIC is loaded
in EUC-JP, as OpenNLP users load it, and its goldens have a second
`model` record for the UTF-8 copy, which the Rust port reads.
`StringUtil.isWhitespace` splits both the dictionary's lines and the
text, so a `+legacy` run loads the dictionary and segments with the
legacy mode active. There are no dumps.

The fixture dictionaries are small and written for this project
(`analysis/goldens/data/README.md`). IPADIC is under the license in its
`COPYING` file (the NAIST notice and the ICOT terms, which allow
redistribution with the notice) and mecab-ko-dic under the Apache
License 2.0; neither is checked in, and only the goldens made with them
are. Those goldens hold the feature columns of the entries chosen, so
`analysis/goldens/data/README.md` carries IPADIC's notice and names
mecab-ko-dic's license.

## Format goldens

`formats/<format>/formats/<id>.golden` is what one of OpenNLP's sample
streams read from `inputs/formats/<id>.txt`, a file in the format the id
starts with, read as the command line tools read it
(`PlainTextByLineStream` over UTF-8). `FormatGolden` lists the streams:
`NameSampleDataStream`, `WordTagSampleStream`, `LemmaSampleStream`,
`ChunkSampleStream` (of `opennlp.tools.chunker`), `TokenSampleStream`,
`SentenceSampleStream`, `DocumentSampleStream` and
`LanguageDetectorSampleStream`, the four that split lines with
`WhitespaceTokenizer` also in the legacy whitespace mode, and the
conversion streams `NameToTokenSampleStream`, `POSToTokenSampleStream`,
`NameToSentenceSampleStream` and `POSToSentenceSampleStream` with the
fixture `en-detokenizer.xml` (the `model` record). The sentence
conversions run with a chunk size of 2, so the inputs show the sample a
full chunk drops. The CoNLL-U streams `ConlluPOSSampleStream` and
`ConlluLemmaSampleStream` (with each tag set), `ConlluSentenceSampleStream`
(two sentences a sample) and `ConlluTokenSampleStream` (in both
whitespace modes) read the `conllu` inputs through a `ConlluStream`. `ConllGolden` runs `Conll02NameSampleStream` (Dutch and
Spanish) and `Conll03NameSampleStream` (English and German), each with
every name type and one run of each with some left out, over the
`conll02` and `conll03` inputs of both languages, and
`ConllXPOSSampleStream` over the `conllx` inputs, also through
`POSToTokenSampleStream` and `POSToSentenceSampleStream` (chunk size 2)
as `ConllXTokenSampleStreamFactory` and
`ConllXSentenceSampleStreamFactory` use them. Every stream reads lines of
UTF-8 text. CoNLL-2000 is the chunker's native format, read by
`ChunkSampleStream` (the `chunk` inputs).

The stream is read until it returns null; an exception from `read` is
recorded and reading goes on. Every layer is anchored to the document:
`turbo:samples` (each sample's `toString`), `turbo:read-errors` (the
number of samples before each exception, its class and message), and by
sample type `turbo:sample-tokens` (sample index and token) with
`turbo:sample-tags`, `turbo:sample-lemmas` and `turbo:sample-chunks`
aligned to it, `turbo:names` and `turbo:phrases` (sample, token span,
type), `turbo:phrase-errors` (what `getPhrasesAsSpanList` threw),
`turbo:sample-texts` and `turbo:sample-spans` (a token or sentence
sample's own text and the UTF-8 byte span of each token or sentence in
it) and `turbo:sample-labels` (a category or language). The streams ask
no model, so there are no dumps. `analysis/formats/tests/opennlp.rs`
compares the Rust readers with these goldens.

## MASC goldens

`masc/<run>/formats/<id>.golden` is what OpenNLP's MASC classes read from
the corpus bundled in `inputs/formats/<id>.txt` (the `masc-` inputs).
A bundle is one UTF-8 text holding the files of a corpus directory: a
line `@@ file <path>` starts a file whose content runs to the next line
`@@ end` (the line end before `@@ end` is not part of the file), `@@ hex
<path>` does the same with the bytes written in hexadecimal for content
that is not UTF-8, `@@ dir <path>` makes an empty directory, and other
lines outside a file are empty or start with `#`. `MascGolden` writes
the bundle to a temporary directory and reads it there. Each directory of
a bundle holds at most one `.hdr` file and one subdirectory, so the order
in which `MascDocumentStream` finds the documents does not depend on the
order the file system lists a directory in.

The `documents` run parses each document as the stream's constructor does
(`checkAnnotations`, called by reflection, then `parseDocument`) and
records, through reflection where the classes keep it private:
`turbo:masc-documents` (per header, its path in the corpus, the class and
message of what the parse threw with the corpus directory written
`<corpus>` and without the stack trace `addPennTags` appends to its
message, `abort` 1 when the throw ends the stream's constructor rather
than skipping the document, and whether it has Penn tags and named
entities), `turbo:masc-sentences` (each sentence's region in UTF-16 units,
the ids of its quarks, whether it was tokenized), `turbo:masc-tokens`
(each Penn token's UTF-16 span, id, tag, base, which of the two are null
and the ids of its quarks), `turbo:masc-names` (each named entity's token
span and type) and `turbo:masc-stream` or `turbo:masc-stream-errors` (the
documents the stream loaded, or what its constructor threw).

The sample runs read the corpus with `MascDocumentStream` and
`MascPOSSampleStream`, `MascTokenSampleStream`,
`MascSentenceSampleStream` (two sentences a sample) or
`MascNamedEntitySampleStream` until `read` returns null or throws
`NullPointerException` (the stream has run past its last document, and
every later read throws the same), then call `reset` and read again
(`pass` 1). Other exceptions are recorded and reading goes on. The layers
are those of the format goldens, with the pass added to `turbo:samples`
and `turbo:read-errors`, `turbo:open-errors` for what the constructors
threw, and `nulls` 1 in `turbo:names` for a name without a type.
`analysis/formats/tests/masc.rs` compares the Rust readers with these
goldens.

## NER goldens

`ner/<run>/ner/<id>.golden` is what one of OpenNLP's named entity corpus
streams read from `inputs/ner/<id>.txt`. `NerGolden` makes each stream
as its factory does: `GermEval2014NameSampleStream`,
`BioNLP2004NameSampleStream` and `EvalitaNameSampleStream` with the name
types of the `types` record (and the GermEval layer of `layer`), and
`NameFinderCensus90NameStream`, each over the file read as UTF-8.

The `muc-` and `brat-` inputs are corpus directories bundled into one
text, as the MASC inputs are. The MUC runs read the files whose name
ends with `.sgm` in any case with `MucNameSampleStream`, as
`Muc6NameSampleStreamFactory` does; the factory splits words with a
tokenizer model, and the runs use the whitespace and the simple
tokenizer in its place. The message of an unknown MUC name type lists
the expected types in an order that changes from one JVM run to the
next, so the golden writes them in the order of OpenNLP's source. The
brat runs read the corpus with `BratDocumentStream` and
`BratNameSampleStream` as `BratNameSampleStreamFactory` does without a
sentence or tokenizer model: sentences split at line ends, and the
whitespace or the simple tokenizer. The simple tokenizer cuts a
character outside the Basic Multilingual Plane into its two surrogates,
which a golden cannot hold, so `brat-simple` does not read `brat-utf16`.
Messages naming a document write the corpus directory as `<corpus>`.
The `brat-config` run parses each `.conf` file of a bundle's top
directory with `AnnotationConfiguration.parse` and writes
`turbo:brat-config` (each type and its class) and
`turbo:brat-config-errors` (what the parse threw).

The streams are read until `read` returns null; an exception is recorded
and reading goes on. The layers are `turbo:samples`,
`turbo:read-errors`, `turbo:sample-tokens` and `turbo:names`, as the
format goldens write them. `analysis/formats/tests/ner.rs` compares the
Rust readers with these goldens.

## Parse goldens

`parse/<run>/parse/<id>.golden` is what OpenNLP's Penn Treebank reader
made of `inputs/parse/<id>.txt`, one bracketed parse per line, read with
`PlainTextByLineStream` over UTF-8. `ParserGolden` runs
`ParseSampleStream` (`treebank`, and `treebank+function-tags` with
`Parse.useFunctionTags(true)`, set back to false afterwards), and
`Parse.parseParse` with a rule class as the gap labeler followed by
`updateHeads` (`en-head-rules` and `es-head-rules`, OpenNLP's English and
AnCora rule files, and `es-regex-head-rules`, AnCora rules written for this
project whose tags use more of Java's regular expressions;
`en-head-rules+possessives` calls `pruneParse` and `fixPossesives` before
`updateHeads`, as `AbstractParserEventStream` does). A line that throws is
recorded with the number of parses before it, and reading goes on. The
layers are `turbo:samples` (`toStringPennTreebank` of each parse),
`turbo:read-errors`, `turbo:parse-texts` (each parse's text),
`turbo:parse-nodes` (every node in preorder with its depth, type, UTF-8
span, its head's span, whether it is its own head and its head index),
`turbo:pos-samples` and `turbo:chunk-samples` (what `PosSampleStream` and
`ChunkSampleStream` make of each parse).

`head-rules/<en|es>/head-rules/<id>.golden` is what
`opennlp.tools.parser.lang.en.HeadRules` or `AncoraSpanishHeadRules` read
from the rule file `inputs/head-rules/<id>.txt`: `turbo:head-rules`, the
lines `serialize` wrote, or `turbo:rules-errors`, what the reader threw.
`analysis/parser/tests/goldens.rs` compares the Rust reader, writer and
rule classes with both.

## Gazetteer goldens

`gazetteer/<run>/gazetteer/<id>.golden` is what a gazetteer of the addons
`geo` module (built from the same addons commit as the `numeric` module)
answered for each line of `inputs/gazetteer/<id>.txt`, the input split at
line feeds: `lookup` for a `names` input, `byId` for an `ids` input (a
source and a record id per line, split at the first tab) and `byRegion`
for a `regions` input. `GazetteerGolden` asks the bundled gazetteer
(`BundledGazetteer.getInstance()`, whose resource's SHA-512 is the
golden's model record), the GeoNames, Overture and user gazetteers over
the fixture tables, and an `OverlayGazetteer` with the bundled gazetteer
as its base, the user table as additions and the fixture suppression
rules. The layers are `turbo:gazetteer-sources` (sorted, since the
module's sets have no order), `turbo:gazetteer-queries` (per line, how
many entries came back, or the exception), `turbo:gazetteer-candidates`
(the entries of each line in order, as indices into the next layer),
`turbo:gazetteer-entries` (each entry found, once, with every field),
`turbo:gazetteer-alternates`, `turbo:gazetteer-containment` and
`turbo:gazetteer-attributes` (sorted by key). The `load-<kind>` runs load
each `table-<kind>-` input as a table of that kind and write
`turbo:gazetteer-load` (whether it loaded, or the exception) and
`turbo:gazetteer-suppressions` (the rules of a suppression table).
`analysis/geo/tests/jvm_goldens.rs` asks the Rust gazetteers the same and
compares every cell.

The OpenNLP commit of the recipe has no `StringUtil.stripByteOrderMark`, which
two readers of the module call; the Dockerfile replaces both calls with
the expression the method returns before building the module.

## Place profile goldens

`place-profiles/place-profiles/place-profiles/<id>.golden` is what
`PlaceProfiles.load` made of the table `inputs/place-profiles/<id>.txt`.
The class is not in the addons commit the rest of the `geo` module is built
from, so the Dockerfile takes `PlaceProfiles.java` from
`PLACE_PROFILES_COMMIT` (`src/recipe.rs`, on the addons'
`ready/geo-place-profiles` branch) and compiles it into the module; the
golden's `profilesCommit` record names that commit. The layers are
`turbo:profiles-load` (whether it loaded, or the exception),
`turbo:profiles-metrics`, `turbo:profiles-places` (the ids sorted by
`String.compareTo`, since the class keeps them in a map),
`turbo:profiles-values` (the standardized profiles, read from the class's
private map by reflection), `turbo:profiles-similarity` (every ordered pair)
and `turbo:profiles-neighbors` (`mostSimilar` of every place with the
counts 1, 2 and the number of places). `analysis/geo/tests/profile_goldens.rs`
rebuilds every layer and compares the doubles bit for bit.

## Evaluator goldens

`evaluate/<evaluator>/formats/<id>.golden` is what one of OpenNLP's
evaluators made of `inputs/formats/<id>.txt`, a `pairs-` file of
reference samples each followed by a prediction, read with its format's
stream. `EvalGolden` answers each reference with a stand-in component, a
`java.lang.reflect.Proxy` of the component's interface (`POSTagger`,
`Lemmatizer`, `TokenNameFinder`, `Chunker`, `Tokenizer`,
`SentenceDetector`, `DocumentCategorizer`, `LanguageDetector`) that
returns the prediction's tags, lemmas, names, chunk tags, spans, category
or language, and hears the verdict with a proxy evaluation monitor. The
layers are `turbo:verdicts` (per pair, correct, misclassified, or the
class and message of what `evaluateSample` threw), `turbo:fmeasure` or
`turbo:accuracy` (the scores after each pair), `turbo:span-scores` (the
static `FMeasure.precision` and `recall` of each pair of name or token
spans) and, for `cross-validation`, `turbo:folds` (the training and test
places of every fold of `CrossValidationPartitioner` for 1 to 4
partitions). `DetokenizerEvaluator` is in `opennlp-tools`, which the
image does not build, so it has no golden. `analysis/formats/tests/evaluate.rs`
compares the Rust evaluators with these goldens.

## Normalizer goldens

`normalize/<name>/norm/<id>.golden` is what the OpenNLP normalizer `<name>`
did to `inputs/norm/<id>.txt`: a `turbo:view` layer with one row per edit,
the UTF-8 byte span of the source it replaces and the text it writes. There
is no model, so the normalizer's name takes the model's place in the path.
A normalizer run with a non-default option also has a `config` header
record (`case-fold-tr` has `config	locale	tr`).

`NormalizerGolden` lists the normalizers. One that implements
`OffsetAwareNormalizer` is run through `normalizeAligned` and `Edit.fromAlignment`
reads its edits from the alignment: characters that share an original range
are one edit, characters with ranges of their own are one edit each, an
original range no output character covers is a deletion, and a block that
writes its own text back is not an edit. The others are run through
`normalize` and `Edit.fromDiff` reconstructs the edits by a diff of source
and result on code points, so for them the alignment is reconstructed and
the `annotator` line says `normalize` where the others say `normalizeAligned`.
The diff drops runs of fewer than three equal code points inside a changed
region, because they are coincidences (a URL replaced by a space next to a
space), which makes it join changes that lie one or two characters apart.

The `norm` inputs are written for the project and use characters of the
Unicode 15.0 repertoire only, which is what JDK 21 implements.

## Number goldens

`numbers/<notation>/num/<id>.golden` is what number scanning did to
`inputs/num/<id>.txt`, with `<notation>` `latin-us` (`,` groups, `.` is the
decimal mark) or `latin-eu` (the reverse). It has a `turbo:numbers` layer
with one row per number mention: `start` and `end` (UTF-8 byte offsets of
the mention, from the sign to the end of the scale marker), `numeral_end`
(where the digits stop and a `k` or `million` begins; equal to `end` when
there is none), `negative`, and the value as `coefficient` (the unscaled
value without sign), `scale` (digits after the decimal point) and `plain`
(`toPlainString()`, for a reader of the file). The golden has a
`config notation` record. As with the normalizers, there is no model, so
the notation takes the model's place in the path.

The scan is `NumberScan.parse` in the `numeric` module of
`ai-pipestream/opennlp-addons` (branch `ready/numeric`, commit
`33e95a4956f1535cdc074056b29246fe88b3c81a`, pinned in `src/recipe.rs`). `NumberGolden` runs the
loop of that module's `CursorQuantityExtractor.extract` and `matchAt`
without the exclusion of money spans and without units, using only public
`NumberScan` methods: at each offset an optional `-`, an ASCII digit that
does not continue an earlier number, a boundary before the mention (the
sign boundary after a `-`), then `parse` with scale markers.

The OpenNLP fork at the pinned commit also contains
`opennlp.tools.extraction.NumberScan` and `NumberNotation`, older than the
module's (the fork's `NumberScan` allows only an ASCII space before a scale
word). The generator's pom lists the module before the runtime so that its
classes come first on the class path, and the generator stops if `NumberScan`
was not loaded from the module's jar.

The `num` inputs are written for the project and use characters of the
Unicode 15.0 repertoire only. Only `NumberGolden` runs on them.

## Money goldens

`money/<notation>/money/<id>.golden` is what `MoneyAnnotator` over
`CursorMoneyExtractor` found in `inputs/money/<id>.txt`, in `latin-us` or
`latin-eu`. `money-convert/<notation>/money/<id>.golden` is the same
mentions restated in EUR by `MoneyConversionAnnotator` with `EcbFxRates`
as of 2026-03-15. Both have one row per mention with `start`, `end`
(UTF-8 byte offsets), `coefficient`, `negative`, `scale`, `plain` and
`currency`, in the layer `turbo:money`. The
`config` records are `notation`, and for the conversions `target`,
`as_of` and `snapshot`. A mention without a usable rate has no row in the
converted layer. An amount that comes out with a negative scale, such as
`2E+1`, is written with scale 0.

The rates are `analysis/goldens/data/rates/ecb-snapshot.csv`, a small
hand made file in the format of the ECB's `eurofxref-hist.csv` with
invented values. The generator reads `rates/ecb-snapshot.csv` beside its
inputs directory; the driver mounts the data directory's `rates` at
`/rates` in the container. 2026-03-15 is a Sunday, so the Friday row
applies; the file has no CNY quote on that Friday and never one for RUB.

The `money` inputs are written for the project. Only the two money
annotators run on them, and they have no dumps.

## Quantity goldens

`quantity/<notation>/quantity/<id>.golden` is what `QuantityAnnotator`
over `CursorQuantityExtractor`, with its default units, found in
`inputs/quantity/<id>.txt`, in `latin-us` or `latin-eu`. Each has one row
per mention with `start`, `end` (UTF-8 byte offsets), `coefficient`,
`negative`, `scale`, `plain` and `unit`, in the layer `turbo:quantities`,
and a `config notation` record. `negative` follows the value's sign, and a
value with a negative scale is written with scale 0, as for money.

The `quantity` inputs are written for the project. Only the quantity
annotator runs on them, and it has no dumps.

## Temporal goldens

`temporal/<mode>/temporal/<id>.golden` is what `TemporalAnnotator` over
`CursorTemporalExtractor`, followed by `DocumentDateAnnotator`, found in
`inputs/temporal/<id>.txt`. In the `dateline` mode the annotator takes the
reference date of relative mentions from the text's first absolute day,
and a text without one has no relative mentions; in the `fixed` mode the
reference date is 2026-03-15. The `config reference` record is `dateline`
or `2026-03-15`; the mode takes the model's place in the path.

Each golden has two layers. `turbo:temporal` has one row per mention with
`start`, `end` (UTF-8 byte offsets), `value` (`TemporalExpression.value()`:
`2026-07-14`, `2026-W29`, `2026-07`, `2024-Q3` or `2026`), and
`granularity` and `origin`, the names of the enum constants.
`turbo:document-date` is anchored on the document and has no row or one:
`date`, the chosen `LocalDate`, and `mention`, the index in
`turbo:temporal` of the mention the date annotation spans.

The `temporal` inputs are written for the project. Only the temporal
annotators run on them, and they have no dumps.

## SVM document categorizer goldens

These runs are the parity test of `turbo-doccat`
(`analysis/doccat/tests/goldens.rs`, described in `docs/doccat.md`). Each
run trains a model with `DocumentCategorizerSVM.train` on one corpus of
`inputs/svm-doccat-train` (one sample per line: the category, a space and
the text), with its own term weighting, feature selection, scaling, SVM
type and kernel and probability estimates on, seeding LIBSVM's random
numbers first so the model is the same on every run of the generator. The
`config` records name all of it.

`svm-doccat/<run>/svm-doccat-train/<corpus>.golden` holds the model as the
two files `turbo-doccat` reads, one line per row: `turbo:svm-model`, the
LIBSVM model as `svm.svm_save_model` writes it, and `turbo:svm-sidecar`, the
vocabulary, inverse document frequencies, feature ranges, weighting,
scaling, generators and categories (`SvmDoccatExport.java`).

`svm-doccat/<run>/svm-doccat/<id>.golden` holds, for one document,
`opennlp:tokens` (the `WhitespaceTokenizer` tokens the categorizer is
given) and, anchored on the document, `turbo:svm-vector` (the feature
vector `categorize` hands LIBSVM), `turbo:svm-decisions` (the
`svm_predict_values` decision value of each class pair, in LIBSVM's
order), `turbo:svm-vote` (the category the pairwise vote picks) and
`turbo:categories` (every category with its probability, best first).

The corpora and documents are written for the project. Only these runs read
them, and they have no dumps. The goldens are not checked in; the parity
test skips, naming the `generate` command, while
`analysis/goldens/data/svm-doccat` is absent.

## PII goldens

`pii/<pack>/pii/<id>.golden` is what `PiiAnnotator` found in
`inputs/pii/<id>.txt` with the extractor the `PiiPacks` factory of that
pack builds (`payment()`, `usIdentity()`, `allStructured()` and so on).
The header names `opennlp.tools.pii.PiiPacks` and the factory method, and
the `config pack` record is the pack's directory name, which takes the
model's place in the path.

Each golden has one layer, `turbo:pii`, with one row per mention in the
order `PiiAnnotator` writes them: `start`, `end` (UTF-8 byte offsets),
`type` (`PiiMention.type()`: `email`, `card`, `us-ssn` and so on) and
`normalized`, the value as the extractor normalized it.

The `pii` inputs are written for the project and hold no real personal
data: documented test IBANs and card numbers, `example.com` addresses,
fictional phone ranges, documentation IP ranges and made up crypto
addresses. Only the PII annotator runs on them, and it has no dumps.
`analysis/pii/tests/goldens.rs` compares them with `turbo-pii`.

## Fuzzing against the Rust ports

`turbo-goldens fuzz` compares the JVM and the Rust ports on generated
texts. `docs/analysis.md` ("Differential fuzzing") describes the corpus,
the comparison, the offset invariants and what the runs found; this
section is about running it.

```
cargo run -p turbo-goldens-tool -- fetch-models target/turbo-goldens/models
cargo run -p turbo-goldens-tool -- image
cargo run -p turbo-goldens-tool --features fuzz -- fuzz --seed 1 --count 1000
```

The command:

1. writes the first `<count>` texts of the seed's corpus
   (`turbo_goldens::fuzz::corpus`) as `<out>/inputs/<set>/s<seed>-<index>.txt`
   for each set named by `--sets`, `en`, `edge`, `norm`, `num`, `money`,
   `quantity`, `temporal`, `pii` and `sentence-fixture` by default, so that every run the
   differential covers reads each text; `<out>` is
   `target/turbo-goldens/fuzz/s<seed>-<count>` unless `--out` names a
   directory; the inputs and the `jvm` directory of an earlier run there
   are replaced;
2. runs the generator in the container (`--image`, the recipe's tag by
   default) over that inputs directory with the models, the fixtures and
   the rate snapshot mounted as `generate` mounts them, and with
   `--skip-annotators` set to `turbo_fuzz::runs::NOT_COVERED` (the training,
   format reader, evaluator, detokenizer, BPE, parse, sequence, name finder,
   stemmer, stopword and glossary annotators),
   the annotators the differential does not cover; the goldens go under
   `<out>/jvm/`;
3. compares every golden with the Rust side (`turbo_fuzz::compare_tree`),
   checks the offset invariants of every layer the Rust side wrote,
   prints the report and writes it to `<out>/report.txt`;
4. reduces up to `--reduce` problems (10; `--no-reduce` none,
   `--reduce-documented` the documented differences as well), each by
   delta debugging over the text's lines and then its characters, with
   the container as the oracle: each round writes the candidates as an
   input set and runs the generator with `--only-runs <annotator>/<model>`
   over them, at most `--rounds` times (40). The minimal text is written
   as `<out>/candidates/<id>-<annotator>-<model>.txt` with a `.report`
   beside it naming the run, the mismatch, how many bytes and rounds the
   reduction took and the command that makes the input again.

It exits non zero when the report has a problem. `--reuse` compares and
reduces what an earlier run left under `<out>` without writing the
inputs or running the generator again (the input count is what the
directory holds, whatever `--count` says); it is how a classification
rule in `analysis/fuzz/src/runs` is worked on against a fixed tree.

Delta debugging keeps any candidate on which the run still has a
problem, not necessarily the problem the input started with, so the
minimal text can show another divergence than the input it came from;
the `.report` names the mismatch of the minimal text. When a fix for
that one lands, reduce the original input again.

The command is behind the crate's `fuzz` feature: `image`,
`fetch-models`, `generate` and `verify` use the standard library alone,
and the feature brings in `turbo-goldens` and `turbo-fuzz`, with every
analysis crate behind them.

To make one input again, `turbo_goldens::fuzz::input(seed, index)` gives
the text of `s<seed>-<index>`, and
`turbo-goldens fuzz --seed <seed> --count <index + 1> --sets <set>`
its golden (the inputs before it come along, and every input of a seed is
a fixed function of its index). After a fix, the candidate file is
checked in under `analysis/goldens/data/inputs/<set>/` and `generate`
gives it a golden.

The intended use is a nightly run on a hardware host with Docker and the
models fetched, a few thousand inputs with a fresh seed each night (the
date serves), the report kept and every candidate looked at. A thousand
inputs over the seven default sets are a hundred thousand goldens and
take about ten minutes: two for the generator, the rest for the
comparison in a debug build.

The generator's two options behind this are general: `--skip-annotators
<a>,<b>` leaves out every run whose annotator directory is named, and
`--only-runs <annotator>/<model>,...` keeps only the runs named. They
follow the five positional arguments, in any order with `--dumps`, and
the inputs directory may hold any subset of the sets.

## Sequence, name finder, stemmer, stopword and glossary goldens

These runs are the parity tests of `turbo-sequence`, `turbo-namefind`,
`turbo-stem`, `turbo-hunspell`, `turbo-stopword` and `turbo-glossary`; each
crate's README describes its inputs, fixtures and layers, and the Java class of
each run says what it records. Their layers are anchored to the document, but
for the glossary matchers' matches and analyzed terms, which are spans of the
text; a string that can hold half of a surrogate pair is written as
`TrainGolden.lossless` writes it.

The `code-points` run of `string-pattern` runs the code point
`StringPattern` of the fork's `ready/string-pattern-unicode` branch
(`STRING_PATTERN_COMMIT` in `src/recipe.rs`). The image compiles that commit's
`StringPattern`, `FeatureGeneratorUtil` and `ParamChecks` against the
generator's OpenNLP build into `/string-pattern/string-pattern.jar`, and the
entrypoint names the jar in the system property
`turbo.goldens.stringPatternJar`; the run loads the three classes from it in a
class loader that looks there first, beside the generator's own. Run outside
the container, the generator needs that property.

The `hunspell` runs work the same way with the Hunspell classes of the fork's
`OPENNLP-1927-hunspell-compatibility` branch (`HUNSPELL_COMMIT` in
`src/recipe.rs`): the image compiles that commit's `HunspellStemmer`,
`HunspellDictionary`, `HunspellAffixTable`, `HunspellCompoundRule`,
`HunspellConversion` and `AffixCondition`, with `java/hunspell/HunspellBridge.java`,
into `/hunspell/hunspell.jar` beside a file holding the commit, and the
entrypoint names the jar in `turbo.goldens.hunspellJar`. The runs check the
commit and load every class of the package from the jar first. The
dictionaries come from `fetch-models`, which downloads `HUNSPELL_FILES` of
the recipe into `<models>/hunspell/` and checks the SHA-512 the recipe holds
for each (the repository publishes none); `analysis/hunspell/README.md` lists
their licences. None is checked in.

The fixture models `pos-fixture.bin`, `chunker-fixture.bin` and
`ner-{bio,bilou,qn}-fixture.bin` are trained by `FixtureModels.java` from
`analysis/goldens/data/train-fixtures/`. It is not one of the runs; it is run
by hand in the image, and the models it writes have a new manifest timestamp,
so the recipe's SHA-512 of each and every golden made from it change:

```
docker run --rm --network none --entrypoint java \
    -v "$PWD/analysis/goldens/data/train-fixtures:/train:ro" \
    -v "$PWD/analysis/goldens/data/fixtures:/fixtures" \
    <image> -cp /generator/target/turbo-goldens-java.jar \
    ai.pipestream.turbo.goldens.FixtureModels /train /fixtures
```

`ner-bio-fixture.bin` stores its cluster resources through
`golden.StoredClusters`, whose serializers its manifest names, so OpenNLP loads
it only with that class on the class path and its package registered with
`ExtensionLoader`; the generator does both.

## How the checked in goldens and dumps were made

The goldens and dumps checked in now were made in the container, with
`turbo-goldens image` and then `turbo-goldens generate`, and `verify`
reproduces every one of them byte for byte. The `jdk` line of each names
the container's JDK, Eclipse Temurin 21.0.12.1.

Before that, they were made by running the generator directly on Ubuntu's
OpenJDK 21.0.12.1. The container run gave the same bytes except the `jdk`
header line. The generator can still be run that way while working on it,
with the OpenNLP fork built at the commit pinned in `src/recipe.rs`
(commit `cdf33c653d5d61d5af7d37ffee189a83bd8392ea` of the
`kristian-3.x-features` branch, Maven version `3.0.0-SNAPSHOT`; the commit
pinned before it does not have `Dehyphenation`, `BoundedUrl`,
`EnglishContraction` and `SymbolJoiner`), and the `numeric` module of
`opennlp-addons` installed after it at its commit in `src/recipe.rs` (the
`numbers` goldens do not record that commit in their header). Its output
then differs from the checked in files in the `jdk` line:

```
git init opennlp && cd opennlp
git remote add origin https://github.com/ai-pipestream/opennlp.git
git fetch --depth 1 origin cdf33c653d5d61d5af7d37ffee189a83bd8392ea
git checkout FETCH_HEAD
./mvnw -q -DskipTests -DskipITs -Dforbiddenapis.skip=true -pl opennlp-core/opennlp-runtime,opennlp-core/opennlp-formats,opennlp-core/opennlp-ml/opennlp-ml-libsvm -am install
cd ..
git init opennlp-addons && cd opennlp-addons
git remote add origin https://github.com/ai-pipestream/opennlp-addons.git
GIT_LFS_SKIP_SMUDGE=1 git fetch --depth 1 origin 33e95a4956f1535cdc074056b29246fe88b3c81a
GIT_LFS_SKIP_SMUDGE=1 git checkout FETCH_HEAD
../opennlp/mvnw -q -DskipTests -pl numeric -am install
cd <repository>/analysis/goldens-tool/java
mvn -q package
cargo run -p turbo-goldens-tool -- fetch-models <models dir> --local
java -Dturbo.goldens.stringPatternJar=<string-pattern.jar> -Dturbo.goldens.hunspellJar=<hunspell.jar> -jar target/turbo-goldens-java.jar ../../goldens/data/inputs <models dir> ../../goldens/data/fixtures \
    <empty out dir> cdf33c653d5d61d5af7d37ffee189a83bd8392ea --dumps
```

The `money` and `money-convert` goldens come from the same run: the
generator finds `rates/ecb-snapshot.csv` beside the inputs directory it is
given.

Moving the pin from `92d486d05151ef00b6d145c405046aa3a33b5a48` changed the
`opennlp` header line of every golden and, in the `langdetect` goldens, the
lower ranked languages' confidences and in places their order; the sentence
and token goldens are otherwise identical. In the dumps it changed the
`opennlp` header line of every one and, in three `langdetect` dumps
(`en/emoji`, `en/url` and `multi/mixed`), the n-gram contexts and with them
every outcome probability; the sentence and token dumps are otherwise
identical.

## Why the OpenNLP build skips two checks

`-DskipITs` skips the integration tests, which download models from the
network. `-Dforbiddenapis.skip=true` skips the forbidden API check, which
fails at the earlier pinned commit on `NameSample` and `Parse` in `opennlp-api`
(their use of `java.util.regex`); the check reads the compiled classes
without changing them, so skipping it changes nothing the generator runs,
including the `formats` goldens, which read `NameSample`s.
