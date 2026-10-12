# turbo-goldens: making the golden layer files

The golden files in `analysis/goldens/data` record what OpenNLP 3.x
computes on the JVM for fixed inputs and models. This directory holds
everything that makes them:

- `java/`: the generator, a Maven project that depends only on the OpenNLP
  runtime. It runs `SentenceDetectorME.sentPosDetect`,
  `TokenizerME.tokenizePos` and `LanguageDetectorME.predictLanguages` over
  every input of the `en` and `multi` sets and writes one golden per
  annotator, model and input. With `--dumps` it also writes a decision
  point dump beside each of those goldens: every context the annotator
  gave its maxent model and every outcome probability `eval` returned. For
  the `norm` set it runs each normalizer of `opennlp.tools.util.normalizer`
  (see below) and writes one `turbo:view` golden per normalizer and input.
- `container/Dockerfile`: the container the generator runs in. It starts
  from `eclipse-temurin:21-jdk` pinned by digest, clones the OpenNLP fork
  at a pinned commit, builds the runtime with the repository's `mvnw` and
  builds the generator.
- `src/`: the driver, the `turbo-goldens` binary. `src/recipe.rs` holds the
  base image digest, the OpenNLP commit, the models with their URLs and
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
  together; before installing, every golden must have a dump beside it
  whose header parses.
- `verify` generates into a scratch directory and compares every golden
  and dump byte for byte with the checked in ones. It lists each file that
  differs, is missing or is extra, and exits non zero if there is any. It
  also checks that every checked in golden has a dump beside it whose
  header parses and names the golden's input and model; it does not read
  the decision points.

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

The goldens and dumps checked in now were produced by running the generator
directly, not in the container, on JDK 21 (the `jdk` line of each golden
names it), with these commands. The OpenNLP fork is cloned at the commit
pinned in `src/recipe.rs` (commit `cdf33c653d5d61d5af7d37ffee189a83bd8392ea`
of the `kristian-3.x-features` branch, Maven version `3.0.0-SNAPSHOT`),
because the commit pinned before it does not have `Dehyphenation`,
`BoundedUrl`, `EnglishContraction` and `SymbolJoiner`:

```
git init opennlp && cd opennlp
git remote add origin https://github.com/ai-pipestream/opennlp.git
git fetch --depth 1 origin cdf33c653d5d61d5af7d37ffee189a83bd8392ea
git checkout FETCH_HEAD
./mvnw -q -DskipTests -DskipITs -Dforbiddenapis.skip=true -pl opennlp-core/opennlp-runtime -am install
cd <repository>/analysis/goldens-tool/java
mvn -q package
cargo run -p turbo-goldens-tool -- fetch-models <models dir>   # std only; or curl each URL in src/recipe.rs
java -jar target/turbo-goldens-java.jar ../../goldens/data/inputs <models dir> <empty out dir> \
    cdf33c653d5d61d5af7d37ffee189a83bd8392ea --dumps
```

then `sentdetect`, `tokenize`, `langdetect` and `normalize` of the output
directory were copied over those of `analysis/goldens/data`. (The models were
fetched with `curl` from the URLs in `src/recipe.rs` and their SHA-512
compared, as `fetch-models` does.) The container recipe and the `image`,
`generate` and `verify` commands have not been run: they are to be proven on
a project machine with Docker, where `verify` shows whether the container's
JDK gives the same bytes.

Moving the pin from `92d486d05151ef00b6d145c405046aa3a33b5a48` changed the
`opennlp` header line of every golden and, in the `langdetect` goldens, the
lower ranked languages' confidences and in places their order; the sentence
and token goldens are otherwise identical.

## Why the OpenNLP build skips two checks

`-DskipITs` skips the integration tests, which download models from the
network. `-Dforbiddenapis.skip=true` skips the forbidden API check, which
fails at the earlier pinned commit on `NameSample` and `Parse` in `opennlp-api`
(their use of `java.util.regex`); the generator uses neither class, and
the check reads the compiled classes without changing them.
