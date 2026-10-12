# turbo-goldens: making the golden layer files

The golden files in `analysis/goldens/data` record what OpenNLP 3.x
computes on the JVM for fixed inputs and models. This directory holds
everything that makes them:

- `java/`: the generator, a Maven project that depends only on the OpenNLP
  runtime. It runs `SentenceDetectorME.sentPosDetect`,
  `TokenizerME.tokenizePos` and `LanguageDetectorME.predictLanguages` over
  every input and writes one golden per annotator, model and input. With
  `--dumps` it also writes a decision point dump beside each golden: every
  context the annotator gave its maxent model and every outcome
  probability `eval` returned.
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

## How the checked in goldens and dumps were made

The goldens and dumps checked in now were produced by running the generator
directly, not in the container: OpenNLP was built from the pinned commit
with `./mvnw -q -DskipTests -DskipITs -Dforbiddenapis.skip=true -pl
opennlp-core/opennlp-runtime -am install`, the generator with `mvn -q
package` in `java/`, the models fetched with `turbo-goldens fetch-models`,
and then

```
java -jar java/target/turbo-goldens-java.jar \
    ../goldens/data/inputs <models dir> <empty out dir> 92d486d05151ef00b6d145c405046aa3a33b5a48 --dumps
```

with the output copied into `analysis/goldens/data`. The dumps were made
in a separate run of this command with the same JDK, OpenNLP build and
models; the goldens that run wrote are byte for byte the checked in ones.
The `jdk` line of each golden and dump names the JDK that ran it. The
container recipe and the `image`, `generate` and `verify` commands have
not been run yet: they are to be proven on a project machine with Docker,
where `verify` shows whether the container's JDK gives the same bytes.

## Why the OpenNLP build skips two checks

`-DskipITs` skips the integration tests, which download models from the
network. `-Dforbiddenapis.skip=true` skips the forbidden API check, which
fails at the pinned commit on `NameSample` and `Parse` in `opennlp-api`
(their use of `java.util.regex`); the generator uses neither class, and
the check reads the compiled classes without changing them.
