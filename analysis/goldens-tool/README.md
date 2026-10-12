# turbo-goldens: making the golden layer files

The golden files in `analysis/goldens/data` record what OpenNLP 3.x
computes on the JVM for fixed inputs and models. This directory holds
everything that makes them:

- `java/`: the generator, a Maven project that depends only on the OpenNLP
  runtime and the `numeric` module of the OpenNLP addons. It runs `SentenceDetectorME.sentPosDetect`,
  `TokenizerME.tokenizePos` and `LanguageDetectorME.predictLanguages` over
  every input of the `en` and `multi` sets and writes one golden per
  annotator, model and input. For the `norm` set it runs each normalizer of
  `opennlp.tools.util.normalizer` (see below) and writes one `turbo:view`
  golden per normalizer and input. For the `num` set it runs `NumberScan`
  of the addons module (see below) and writes one `turbo:numbers` golden
  per notation and input. For the `money` set it runs `MoneyAnnotator` with
  `CursorMoneyExtractor`, and `MoneyConversionAnnotator` with `EcbFxRates`
  (see below), and writes `turbo:money` goldens for both
  per notation and input. With `--dumps` it also writes a
  decision point dump beside each sentence, token and language golden:
  every context the annotator gave its maxent model and every outcome
  probability `eval` returned.
- `container/Dockerfile`: the container the generator runs in. It starts
  from `eclipse-temurin:21-jdk` pinned by digest, clones the OpenNLP fork
  at a pinned commit, builds the runtime with the repository's `mvnw`,
  clones `opennlp-addons` at a pinned commit and installs its `numeric`
  module, and builds the generator.
- `src/`: the driver, the `turbo-goldens` binary. `src/recipe.rs` holds the
  base image digest, the OpenNLP and addons commits, the models with their URLs and
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
- `fetch-models <dir>` downloads the three models into `<dir>` with `curl`
  and checks each one's SHA-512 against the value published beside it on
  downloads.apache.org. A model already there with the right hash is kept.
- `generate` fetches the models (into `target/turbo-goldens/models` unless
  `--models` names a directory), runs the container with `--network none`,
  the inputs and the models mounted read only, and replaces the annotator
  directories in `analysis/goldens/data` (or `<dir>`) with what it wrote.
  The generator is always given `--dumps` (`GENERATOR_ARGS` in
  `src/recipe.rs`), so the goldens and their dumps are made and installed
  together; before installing, every sentence, token and language golden
  must have a dump beside it whose header parses, except a golden with an
  error record (the annotator threw on that input), which must have none.
  A normalizer, number or money golden has no dump.
- `verify` generates into a scratch directory and compares every golden
  and dump byte for byte with the checked in ones. It lists each file that
  differs, is missing or is extra, and exits non zero if there is any. It
  also checks the checked in dumps by the same rule, and that each one's
  header names the golden's input and model; it does not read the
  decision points.

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

## How the checked in goldens and dumps were made

The goldens and dumps checked in now were produced by running the generator
directly, not in the container, on JDK 21 (the `jdk` line of each golden
names it), with these commands. The OpenNLP fork is cloned at the commit
pinned in `src/recipe.rs` (commit `cdf33c653d5d61d5af7d37ffee189a83bd8392ea`
of the `kristian-3.x-features` branch, Maven version `3.0.0-SNAPSHOT`),
because the commit pinned before it does not have `Dehyphenation`,
`BoundedUrl`, `EnglishContraction` and `SymbolJoiner`. The `numeric` module of
`opennlp-addons` is cloned at its pinned commit and installed after it (the
`numbers` goldens do not record this commit in their header; it is the one in
`src/recipe.rs`):

```
git init opennlp && cd opennlp
git remote add origin https://github.com/ai-pipestream/opennlp.git
git fetch --depth 1 origin cdf33c653d5d61d5af7d37ffee189a83bd8392ea
git checkout FETCH_HEAD
./mvnw -q -DskipTests -DskipITs -Dforbiddenapis.skip=true -pl opennlp-core/opennlp-runtime -am install
cd ..
git init opennlp-addons && cd opennlp-addons
git remote add origin https://github.com/ai-pipestream/opennlp-addons.git
GIT_LFS_SKIP_SMUDGE=1 git fetch --depth 1 origin 33e95a4956f1535cdc074056b29246fe88b3c81a
GIT_LFS_SKIP_SMUDGE=1 git checkout FETCH_HEAD
../opennlp/mvnw -q -DskipTests -pl numeric -am install
cd <repository>/analysis/goldens-tool/java
mvn -q package
cargo run -p turbo-goldens-tool -- fetch-models <models dir>   # std only; or curl each URL in src/recipe.rs
java -jar target/turbo-goldens-java.jar ../../goldens/data/inputs <models dir> <empty out dir> \
    cdf33c653d5d61d5af7d37ffee189a83bd8392ea --dumps
```

The `money` and `money-convert` goldens come from the same run: the generator finds
`rates/ecb-snapshot.csv` beside the inputs directory it is given.

then `sentdetect`, `tokenize`, `langdetect`, `normalize`, `numbers`, `money`
and `money-convert` of the output directory were copied over those of `analysis/goldens/data`. (The models were
fetched with `curl` from the URLs in `src/recipe.rs` and their SHA-512
compared, as `fetch-models` does.) The container recipe and the `image`,
`generate` and `verify` commands have not been run: they are to be proven on
a project machine with Docker, where `verify` shows whether the container's
JDK gives the same bytes.

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
