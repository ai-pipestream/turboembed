# turbo-goldens: making the golden layer files

The golden files in `analysis/goldens/data` record what OpenNLP 3.x
computes on the JVM for fixed inputs and models. This directory holds
everything that makes them:

- `java/`: the generator, a Maven project that depends only on the OpenNLP
  runtime. It runs the annotator runs listed below over their input sets
  and writes one golden per run and input. With `--dumps` it also writes a
  decision point dump beside each golden of a run that asks a maxent model
  (the sentence detector, the tokenizer and the language detector): every
  context the annotator gave the model and every outcome probability
  `eval` returned.
- `container/Dockerfile`: the container the generator runs in. It starts
  from `eclipse-temurin:21-jdk` pinned by digest, clones the OpenNLP fork
  at a pinned commit, builds the runtime with the repository's `mvnw` and
  builds the generator.
- `src/`: the driver, the `turbo-goldens` binary. `src/recipe.rs` holds the
  base image digest, the OpenNLP commit, the published models with their
  URLs and SHA-512 hashes, the checked in fixtures with their SHA-512
  hashes, and the input sets.

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
- `generate` fetches the models (into `target/turbo-goldens/models` unless
  `--models` names a directory), checks that `analysis/goldens/data/fixtures`
  holds exactly the recipe's fixtures with their SHA-512, runs the container
  with `--network none`, the inputs, the models and the fixtures mounted read
  only, and replaces the annotator directories in `analysis/goldens/data`
  (or `<dir>`) with what it wrote. The generator is always given `--dumps`
  (`GENERATOR_ARGS` in `src/recipe.rs`), so the goldens and their dumps are
  made and installed together; before installing, every sentence, token and
  language golden whose header names a model must have a dump beside it
  whose header parses, except a golden with an error record (the annotator
  threw on that input), which must have none. A golden that names no model
  (a tokenizer or sentence detector without one, a normalizer) has no dump,
  and neither has a detokenizer golden.
- `verify` generates into a scratch directory and compares every golden
  and dump byte for byte with the checked in ones. It lists each file that
  differs, is missing or is extra, and exits non zero if there is any. It
  also checks the checked in dumps by the same rule, and that each one's
  header names the golden's input and model; it does not read the
  decision points.

## Runs

The golden path is `<annotator>/<model>/<set>/<id>.golden`. `<model>` is
the first model file's name without `.bin` (or `.xml` for a dictionary).
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
- The sentence detector's dumps also record, for each point whose best
  outcome is a split, what `isAcceptableBreak` returned
  (`docs/analysis.md`, "Decision point dumps").

The fixtures (`analysis/goldens/data/fixtures`) are checked in, mounted at
`/fixtures` and listed with their SHA-512 in `src/recipe.rs`; the data
directory's README says where each comes from.

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
`EnglishContraction` and `SymbolJoiner`). Its output then differs from the
checked in files in the `jdk` line:

```
git init opennlp && cd opennlp
git remote add origin https://github.com/ai-pipestream/opennlp.git
git fetch --depth 1 origin cdf33c653d5d61d5af7d37ffee189a83bd8392ea
git checkout FETCH_HEAD
./mvnw -q -DskipTests -DskipITs -Dforbiddenapis.skip=true -pl opennlp-core/opennlp-runtime -am install
cd <repository>/analysis/goldens-tool/java
mvn -q package
cargo run -p turbo-goldens-tool -- fetch-models <models dir>
java -jar target/turbo-goldens-java.jar ../../goldens/data/inputs <models dir> ../../goldens/data/fixtures \
    <empty out dir> cdf33c653d5d61d5af7d37ffee189a83bd8392ea --dumps
```

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
(their use of `java.util.regex`); the generator uses neither class, and
the check reads the compiled classes without changing them.
