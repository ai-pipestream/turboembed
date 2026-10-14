# turbo-maxent test fixtures

Files made by Java, for comparing the Rust readers with the JVM. The Java
programs that made them are in `src/`. The JVM is used to make the files
only; nothing calls it at test time. They are produced by compiling and running these
programs by hand with `javac` against the OpenNLP fork commit given under Provenance.
`models/tokenizer.bin` is also one of the golden generator's fixtures, copied unchanged to
`analysis/goldens/data/fixtures/tokenizer-abbreviations.bin`.

## What is here

| Path | Made by | Read by | Contents |
|---|---|---|---|
| `inflate/*.deflate`, `inflate/index.txt` | `MakeInflateFixtures.java` | `tests/inflate.rs` | Raw deflate streams (no zlib header) from `java.util.zip.Deflater` at levels 0 to 9, plus Huffman-only, filtered, and sync-flushed streams. The index gives each stream's input, settings, size and CRC-32. |
| `zip/*.zip`, `zip/index.txt` | `MakeZipFixtures.java` | `tests/zip.rs` | Archives from `java.util.zip.ZipOutputStream`: deflated entries with data descriptors, an archive comment, a UTF-8 entry name, an empty entry, stored entries, and entries with an extended timestamp extra field. |
| `models/gis.model`, `qn.model`, `perceptron.model` | `MakeModelFixtures.java` | `tests/fixture_models.rs` | Models trained by OpenNLP's GIS, QN and perceptron trainers on nine events, written by `GenericModelWriter`. Some predicates hold U+0000, an accented letter and a character outside the BMP. |
| `models/tokenizer.bin` | `MakeModelFixtures.java` | `tests/fixture_models.rs` | A TokenizerME model file serialized by OpenNLP (`TokenizerModel.serialize`), trained on six sentences, with a case-insensitive abbreviation dictionary. |
| `models/expected.txt` | `MakeModelFixtures.java` | `tests/fixture_models.rs` | For each model, its outcomes and the probabilities `MaxentModel.eval` gave for each context, printed with `Double.toString`. |
| `text/tricky.properties` | written by hand | `tests/text.rs` | A properties file with every separator, escape, comment and continuation rule, ISO-8859-1 bytes, CR LF and lone CR line ends, and two quirks of the JDK line reader. |
| `text/tricky.properties.expected` | `MakeTextFixtures.java` | `tests/text.rs` | What `java.util.Properties.load(InputStream)` read from it. |
| `text/tricky.dictionary` | written by hand | `tests/text.rs` | A dictionary XML with entities, character references, CDATA, comments inside a token, a namespace prefix, attribute whitespace normalization, a `token` outside an `entry`, and entries that differ only in case. |
| `text/tricky.dictionary.expected` | `MakeTextFixtures.java` | `tests/text.rs` | What OpenNLP's `Dictionary` and `DictionaryEntryPersistor` read from it, and the result of 19 `Dictionary.contains` probes. |

Escaping in the `expected` files: every UTF-16 code unit outside U+0021..U+007E, and the
backslash, is written as `\uXXXX`; a character outside the BMP is two escapes.

The inputs of the inflate and zip fixtures are not checked in. `Inputs.java` makes them
with a 64-bit linear congruential generator, and `tests/inflate.rs` rebuilds the same
bytes with the same generator and checks their size and CRC-32 against the index.

`RealModelValues.java` prints the values in `tests/models.rs` for the OpenNLP 2.5.4 UD
Spanish and French tokenizer models, which are not checked in (about 540 KB each):

| Model | SHA-256 |
|---|---|
| `opennlp-es-ud-gsd-tokens-1.3-2.5.4.bin` | `309de611ff7a8c4227c57086838d4783bcede8520026d7ad416c85a513567ce7` |
| `opennlp-fr-ud-gsd-tokens-1.3-2.5.4.bin` | `a06dbc9b72600ba0886f64a632636e136e5fc225955913c4c1a926a9333bc14a` |

`turbo-goldens fetch-models` downloads them, and `tests/models.rs` reads them from
`TURBO_GOLDENS_MODELS` (or `target/turbo-goldens/models`) through
`turbo_goldens::published_model`; without them it skips, unless
`TURBO_TEST_REQUIRE_OPENNLP_MODELS=1`.

## Provenance

- OpenNLP: the 3.0.0-SNAPSHOT jars built from the OpenNLP checkout at commit
  `cdf33c653d5d61d5af7d37ffee189a83bd8392ea`. The class path is the runtime, API and ML
  (commons, maxent, perceptron, bayes) jars of that build plus `slf4j-api` 2.0.19.
- JDK: OpenJDK 21.0.12.1.

## Running it

From `analysis/maxent/tests/fixtures`, with `$CP` set to the class path above (jar files
separated by `:`) and `$CLASSES` an empty directory:

```
javac -d "$CLASSES" -cp "$CP" src/*.java
java -cp "$CLASSES:$CP" MakeInflateFixtures .
java -cp "$CLASSES:$CP" MakeZipFixtures .
java -cp "$CLASSES:$CP" MakeModelFixtures .
java -cp "$CLASSES:$CP" MakeTextFixtures .
java -Dstdout.encoding=UTF-8 -cp "$CLASSES:$CP" RealModelValues "$MODELS/opennlp-es-ud-gsd-tokens-1.3-2.5.4.bin"
java -Dstdout.encoding=UTF-8 -cp "$CLASSES:$CP" RealModelValues "$MODELS/opennlp-fr-ud-gsd-tokens-1.3-2.5.4.bin"
```

Training is deterministic, so the models and expected values come out the same on a
rerun; `tokenizer.bin` differs only in its manifest timestamp and the zip entry times.
`RealModelValues` prints Rust source lines; paste them into the case tables of
`tests/models.rs`.
