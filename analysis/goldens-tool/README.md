# turbo-goldens: making the golden layer files

The golden files in `analysis/goldens/data` record what OpenNLP 3.x
computes on the JVM for fixed inputs and models. This directory holds
everything that makes them:

- `java/`: the generator, a Maven project that depends only on the OpenNLP
  runtime, the OpenNLP formats module and the `numeric` module of the OpenNLP addons. It runs the
  annotator runs listed below over their input sets and writes one golden
  per run and input; for the `num` set it runs `NumberScan` of the addons
  module, and for the `money` set `MoneyAnnotator` with
  `CursorMoneyExtractor` and `MoneyConversionAnnotator` with `EcbFxRates`,
  and for the `quantity` set `QuantityAnnotator` with
  `CursorQuantityExtractor` (see below). With `--dumps` it also writes a decision point dump
  beside each golden of a run that asks a maxent model (the sentence
  detector, the tokenizer and the language detector): every context the
  annotator gave the model and every outcome probability `eval` returned.
- `container/Dockerfile`: the container the generator runs in. It starts
  from `eclipse-temurin:21-jdk` pinned by digest, clones the OpenNLP fork
  at a pinned commit, builds the runtime and the formats module with the
  repository's `mvnw`,
  clones `opennlp-addons` at a pinned commit and installs its `numeric`
  module, and builds the generator.
- `src/`: the driver, the `turbo-goldens` binary. `src/recipe.rs` holds the
  base image digest, the OpenNLP and addons commits, the published models
  with their URLs and SHA-512 hashes, the checked in fixtures with their
  SHA-512 hashes, and the input sets.

Both formats are described in `docs/analysis.md` ("Generating golden
files" and "Decision point dumps").

## Commands

Run from the repository's root. Each needs Docker except `fetch-models`.

```
cargo run -p turbo-goldens-tool -- image
cargo run -p turbo-goldens-tool -- fetch-models <dir>
cargo run -p turbo-goldens-tool -- generate [--out <dir>] [--models <dir>]
cargo run -p turbo-goldens-tool -- verify [--models <dir>]
```

- `image` builds the container image, tags it with the OpenNLP commit and
  prints its id.
- `fetch-models <dir>` downloads the published models into `<dir>` with `curl`
  and checks each one's SHA-512 against the value published beside it on
  downloads.apache.org. A model already there with the right hash is kept.
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
  made and installed together; before installing, every sentence, token and
  language golden whose header names a model must have a dump beside it
  whose header parses, except a golden with an error record (the annotator
  threw on that input), which must have none. A golden that names no model
  (a tokenizer or sentence detector without one, a normalizer) has no dump,
  and neither has a detokenizer, number, money, quantity, unigram or
  lattice golden.
- `verify` generates into a scratch directory and compares every golden
  and dump byte for byte with the checked in ones. It lists each file that
  differs, is missing or is extra, and exits non zero if there is any. It
  also checks the checked in dumps by the same rule, and that each one's
  header names the golden's input and model; it does not read the
  decision points.

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
| | `en-abbrev`, `en-plain` (fixtures), each also `+legacy` | `sentence-fixture` | `whitespace legacy` |
| `sentdetect` (`NewlineSentenceDetector`) | `newline`, `newline+legacy` | `edge` | `whitespace legacy` |
| `tokenize` (`TokenizerME`) | `opennlp-en-ud-ewt-tokens-1.3-2.5.4` | `en`, `multi`, `edge`, `surrogate` | |
| | the `de`, `es`, `fr` and `pt` tokenizer models | `edge`, `surrogate` | |
| | `tokenizer-abbreviations` (fixture) | `abbreviations`, `edge` | |
| | `opennlp-en-ud-ewt-tokens-1.3-2.5.4+english-dictionary` | `abbreviations`, `edge` | |
| | each of the five models `+alphanumeric` | `edge`, `alphanumeric` | `useAlphaNumericOptimization true`, `alphaNumericPattern <pattern>` |
| `tokenize` (`SimpleTokenizer`, `WhitespaceTokenizer`) | `simple`, `whitespace`, each also `+keep-new-lines`, `+legacy` and `+keep-new-lines+legacy` | `edge` | `keepNewLines true`, `whitespace legacy` |
| `tokenize` (UAX #29 `WordTokenizer`) | `uax29` | `edge` | |
| `detokenize` (`DictionaryDetokenizer`) | `en-detokenizer`, `fr-detokenizer`, `latin-detokenizer`, `pt-detokenizer`, `quotes-detokenizer` (fixtures) | `detokenizer`, the inputs whose id starts with the dictionary's name | `splitMarker <SPLIT>` |
| `langdetect` (`LanguageDetectorME`) | `langdetect-183` | `en`, `multi` | |
| `normalize` (each normalizer) | the normalizer's name | `norm` | see below |
| `numbers` (`NumberScan` of the addons `numeric` module) | `latin-us`, `latin-eu` | `num` | `notation` |
| `money`, `money-convert` (`MoneyAnnotator`, `MoneyConversionAnnotator`) | `latin-us`, `latin-eu` | `money` | `notation`; for conversions also `target`, `as_of`, `snapshot` |
| `quantity` (`QuantityAnnotator`) | `latin-us`, `latin-eu` | `quantity` | `notation` |
| `subwords` (`WordpieceEncoder`) | `all-minilm-l6-v2-vocab` (fixture) | `en`, `multi`, `edge`, `surrogate`, `subwords` | `lowerCase true`, `maxWordCodePoints 100` |
| `bpe` (`BPETokenizerTrainer`, `BPETokenizer`) | `merges-40`, `merges-400` | `bpe` | `languageCode en`, `numMerges` |
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
full chunk drops.

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
./mvnw -q -DskipTests -DskipITs -Dforbiddenapis.skip=true -pl opennlp-core/opennlp-runtime,opennlp-core/opennlp-formats -am install
cd ..
git init opennlp-addons && cd opennlp-addons
git remote add origin https://github.com/ai-pipestream/opennlp-addons.git
GIT_LFS_SKIP_SMUDGE=1 git fetch --depth 1 origin 33e95a4956f1535cdc074056b29246fe88b3c81a
GIT_LFS_SKIP_SMUDGE=1 git checkout FETCH_HEAD
../opennlp/mvnw -q -DskipTests -pl numeric -am install
cd <repository>/analysis/goldens-tool/java
mvn -q package
cargo run -p turbo-goldens-tool -- fetch-models <models dir>
java -jar target/turbo-goldens-java.jar ../../goldens/data/inputs <models dir> ../../goldens/data/fixtures \
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
