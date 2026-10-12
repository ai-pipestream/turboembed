# turbo-goldens: making the golden layer files

The golden files in `analysis/goldens/data` record what OpenNLP 3.x
computes on the JVM for fixed inputs and models. This directory holds
everything that makes them:

- `java/`: the generator, a Maven project that depends only on the OpenNLP
  runtime. It runs `SentenceDetectorME.sentPosDetect`,
  `TokenizerME.tokenizePos` and `LanguageDetectorME.predictLanguages` over
  every input of the `en` and `multi` sets and writes one golden per
  annotator, model and input. For the `norm` set it runs each normalizer of
  `opennlp.tools.util.normalizer` (see below) and writes one `turbo:view`
  golden per normalizer and input. With `--dumps` it also writes a
  decision point dump beside each sentence, token and language golden:
  every context the annotator gave its maxent model and every outcome
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
  together; before installing, every sentence, token and language golden
  must have a dump beside it whose header parses.
- `verify` generates into a scratch directory and compares every golden
  and dump byte for byte with the checked in ones. It lists each file that
  differs, is missing or is extra, and exits non zero if there is any. It
  also checks that every checked in sentence, token and language golden
  has a dump beside it whose header parses and names the golden's input
  and model; it does not read the decision points.

  every input of the `en` and `multi` sets and writes one golden per
  annotator, model and input. For the `norm` set it runs each normalizer of
  `opennlp.tools.util.normalizer` (see below) and writes one `turbo:view`
  golden per normalizer and input. With `--dumps` it also writes a
  decision point dump beside each sentence, token and language golden:
  every context the annotator gave its maxent model and every outcome
  probability `eval` returned.

## Why the OpenNLP build skips two checks

`-DskipITs` skips the integration tests, which download models from the
network. `-Dforbiddenapis.skip=true` skips the forbidden API check, which
fails at the earlier pinned commit on `NameSample` and `Parse` in `opennlp-api`
(their use of `java.util.regex`); the generator uses neither class, and
the check reads the compiled classes without changing them.
