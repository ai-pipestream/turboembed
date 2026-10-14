# turbo-train

Training OpenNLP 3.x maxent models in Rust, the event side: what
`opennlp.tools.ml` does between a stream of events and a trainer, ported
so that the JVM and this crate agree on every label, count, index and
checksum. Sample types (sentences, tokens, tags) and the event streams
that make events from them belong to `analysis/formats`; this crate
starts at events.

| Module | OpenNLP |
|---|---|
| `event` | `Event`, `FileEventStream`, `RealValueFileEventStream` (`parseEvent`, `parseContexts`), `ChecksumEventStream` |
| `java_string` | `JavaString` (from `turbo-maxent`): a Java string that may hold a lone surrogate, in modified UTF-8 |
| `index` | `OnePassDataIndexer`, `TwoPassDataIndexer`, `OnePassRealValueDataIndexer`, `ComparableEvent`, `DataIndexerFactory` |
| `params` | `TrainingParameters`, `Parameters`, `AlgorithmType`, `TrainerFactory.TrainerType`, `getTrainerType`, `isValid` |
| `monitor` | `TrainingProgressMonitor`, `DefaultTrainingProgressMonitor`, `TrainingMeasure`, `StopCriteria`, `LogLikelihoodThresholdBreached`, `IterDeltaAccuracyUnderTolerance`, `TrainingConfiguration` |
| `gis` | `GISTrainer`, `GISModel`'s data structures, `UniformPrior`, `AbstractEventTrainer.train` |
| `writer` | `GenericModelWriter`, `BinaryGISModelWriter`, `GISModelWriter.persist`, `ComparablePredicate`, `ModelParameterChunker` |
| `package` | `BaseModel.serialize`, the manifest `BaseModel` builds and `PropertiesSerializer` writes, `SentenceModel`'s entries |
| `context` | `BasicContextGenerator` |
| `crc32c` | `java.util.zip.CRC32C` |

Numbers are formatted and parsed as Java does (`turbo_jvm::text`),
predicate labels sort by `String.compareTo`, `float` stays `f32`, and
parameters keep Java's case-insensitive keys and typed values.

## Predicates and outcomes are Java strings

The tokenizer's context generator names the character before and after
a candidate split as one `char`, so for an emoji the predicate holds half
a surrogate pair, which no Rust `String` can. A predicate is therefore a
`JavaString`: modified UTF-8, the encoding model files store predicates
in and `turbo-maxent` keys them by. An outcome is a `JavaString` too: a
lemma class is an edit script over the UTF-16 units of a word, so a
token or lemma with a character outside the Basic Multilingual Plane
gives a class holding half of its surrogate pair, and the JVM trains on
it, numbers it by its units and writes it with `writeUTF`. The indexers
and the GIS model hold outcome labels as `JavaString`s and the writer
writes them as the bytes they are; `Event::java_string` and the event
checksum write the class as `Event.toString` and `getBytes(UTF_8)` do
(a lone surrogate becomes `?`).

## JVM reference

`analysis/goldens/data/train/` holds, for events the JVM made from the
golden input texts with `SDEventStream`, `TokSpanEventStream`, a real
valued source and a part-of-speech shaped source with many outcomes, what
each indexer produced with each cutoff and sorting, the event checksum,
and the GIS trainer's log likelihood and parameters after every iteration,
with one or two threads, without smoothing and with each smoothing
(docs/analysis.md, "Training goldens"; `analysis/goldens-tool`,
`TrainGolden.java`). `tests/goldens.rs` reads
the events back and compares the indexers with them; the run
`sentdetect-twopass-cutoff1/train/sentences` is the training of the
fixture model `en-plain.bin`, whose manifest's `Training-Eventhash` is
that run's checksum.

## GIS trainer, writer and model file

`gis::train_events` is `TrainerFactory.getEventTrainer(MAXENT).train(events)`:
checksum, index, train; `gis::train` takes an index; `gis::train_model`
also takes an observer called with the parameters after every iteration.
The trainer keeps Java's arithmetic: `float` counts and correction
constant, `StrictMath.log` and `exp`, the uniform prior, one event at a
time in the JVM's order, and with `Threads` above one the JVM's chunking of
the events and order of merging the chunks' sums. `writer::write_gis`
produces the `.model` bytes, `package::ModelPackage` the `.bin` around
them, and `package::sentence_model_package` the sentence detector's
manifest entries.

Parity: `tests/goldens.rs` trains every training golden's index with the
run's `Threads` and smoothing settings and compares the log likelihood,
the correct and total counts, and the SHA-256 of every parameter after
every iteration, every parameter after the first and the last iteration
bit for bit, the end of training message, and the SHA-256 and length of
the written model with the JVM's. `tests/roundtrip.rs` trains the two
fixture models again from their events and parameters:
`analysis/maxent/tests/fixtures/models/gis.model` is reproduced byte for
byte, and so are the `sent.model` entry and the manifest of
`analysis/goldens/data/fixtures/en-plain.bin`; the `.bin` written around
them is the checked in fixture `analysis/goldens/data/fixtures/en-plain-rust.bin`,
byte for byte, and it opens with `turbo-maxent` and `turbo-sentdetect` and
splits the training text as the fixture does, probabilities included. The
golden generator loads that fixture in the JVM (`SentenceModel`,
`SentenceDetectorME`) and runs it beside `en-plain.bin` on the sentence
fixture inputs (`analysis/goldens/data/sentdetect/en-plain-rust`);
`analysis/sentdetect/tests/golden.rs` checks that the Rust detector gives
the spans and probabilities the JVM recorded with it, and that those are
the ones recorded with `en-plain.bin`. The archive around the entries is
not the JVM's bytes:

The `.bin` written here stores its entries uncompressed, the manifest
first, with the time of the manifest's `Timestamp` in UTC; the JVM
deflates them in `HashMap` order with the local time. Both readers take
both. Timestamps are an argument, never read from a clock, so a model
file is reproducible.

## OpenNLP behaviour worth knowing

- `FileEventStream` throws `NoSuchElementException` on an empty line, so
  an event file cannot have one (`Error::EmptyEventLine`).
- The indexers number an event's outcome before dropping the event for
  having no predicate at or above the cutoff, so the outcome labels can
  name an outcome no indexed event has. The GIS trainer then has an
  outcome with no observations.
- `ComparableEvent` compares and `OnePassRealValueDataIndexer` keeps the
  values of an event's full context, read at the positions of the
  predicates that survived the cutoff. After a drop they are the wrong
  values, and the kept array is longer than the context.
- `TwoPassDataIndexer` spills events with `DataOutputStream.writeUTF`,
  so an outcome or predicate over 65535 bytes of modified UTF-8 throws
  `UTFDataFormatException` there and nowhere else
  (`Error::StringTooLong`).
- `RealValueFileEventStream.parseContexts` reads the value after the last
  `=`; a value that is not a number is logged and counted as 1 with the
  `=` left in the predicate, and a negative one throws.
- `Event.toString`, which the event checksum hashes, writes values with
  `Float.toString`, and `getBytes(UTF_8)` turns a lone surrogate into `?`.
- `TrainingParameters.serialize` goes through `Properties.store`, which
  casts every value to `String`: a parameter put as an `int`, `double` or
  `boolean` makes it throw `ClassCastException` (`Error::SerializeType`).
- `TrainingParameters.getParameters(namespace)` matches the prefix with
  case although the map's keys are case-insensitive.
- `String.CASE_INSENSITIVE_ORDER` is computed as JDK 21 does, by code
  points; JDK 17 and earlier compared UTF-16 units, which differs only for
  keys with characters outside the BMP.
- `GISTrainer` refuses `Smoothing` with `GaussianSmoothing`
  (`Error::BothSmoothings`) and `Threads` under one (`Error::Threads`).
- With `Threads` above one `GISTrainer` sums the log likelihood and the
  expectations per chunk and then across chunks, so a model trained with
  two threads differs in the last bits from one trained with one. The
  chunking (`n / threads` events per chunk, the first `n % threads`
  chunks one longer) and the merge order are ported. The JVM adds the
  chunks' log likelihoods in the order its threads finish; with two
  chunks the sum is the same either way, so the two thread goldens are
  reproducible, and with more it may differ in the last bits from run to
  run in the JVM itself.
- `GISTrainer` stops when the log likelihood falls ("Model Diverging")
  without telling the monitor, so `finishedTraining` then reports the
  requested iteration count with no stop criterion.
- `ModelParameterChunker.writeUTF` cuts a string over 65535 bytes into
  chunks measured in standard UTF-8 but written in modified UTF-8: a chunk
  of characters outside the BMP can exceed 65535 bytes there and the write
  fails (`Error::StringTooLong`), and the UTF-8 encoder stops at a lone
  surrogate, so such a string is cut there and the rest is lost.
- `BaseModel` writes `OpenNLP-Version` from `Version.currentVersion()`,
  `3.0.0-SNAPSHOT` for the fork this crate ports; `package::OPENNLP_VERSION`
  holds it.

Not ported: the abbreviation dictionary entry of a sentence or tokenizer model
(`abbreviations.dictionary`, `DictionarySerializer`) has no writer here
yet; `ModelPackage::add_entry` takes its bytes. Nor are the QN, perceptron
and naive Bayes trainers.
