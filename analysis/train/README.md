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
| `java_string` | A Java string that may hold a lone surrogate, in modified UTF-8 |
| `index` | `OnePassDataIndexer`, `TwoPassDataIndexer`, `OnePassRealValueDataIndexer`, `ComparableEvent`, `DataIndexerFactory` |
| `params` | `TrainingParameters`, `Parameters`, `AlgorithmType`, `TrainerFactory.TrainerType`, `getTrainerType`, `isValid` |
| `monitor` | `TrainingProgressMonitor`, `DefaultTrainingProgressMonitor`, `TrainingMeasure`, `StopCriteria`, `LogLikelihoodThresholdBreached`, `IterDeltaAccuracyUnderTolerance`, `TrainingConfiguration` |
| `context` | `BasicContextGenerator` |
| `crc32c` | `java.util.zip.CRC32C` |

Numbers are formatted and parsed as Java does (`turbo_jvm::text`),
predicate labels sort by `String.compareTo`, `float` stays `f32`, and
parameters keep Java's case-insensitive keys and typed values.

## Predicates are Java strings

The tokenizer's context generator names the character before and after
a candidate split as one `char`, so for an emoji the predicate holds half
a surrogate pair, which no Rust `String` can. A predicate is therefore a
`JavaString`: modified UTF-8, the encoding model files store predicates
in and `turbo-maxent` keys them by. Outcomes are `String`s, as they are
labels from the samples.

## JVM reference

`analysis/goldens/data/train/` holds, for events the JVM made from the
golden input texts with `SDEventStream`, `TokSpanEventStream` and a real
valued source, what each indexer produced with each cutoff and sorting,
the event checksum, and the GIS trainer's log likelihood and parameters
after every iteration (docs/analysis.md, "Training goldens";
`analysis/goldens-tool`, `TrainGolden.java`). `tests/goldens.rs` reads
the events back and compares the indexers with them; the run
`sentdetect-twopass-cutoff1/train/sentences` is the training of the
fixture model `en-plain.bin`, whose manifest's `Training-Eventhash` is
that run's checksum.

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

Not ported: `Float.parseFloat` and `Double.parseDouble` read hexadecimal
literals (`0x1p3`), which `turbo_jvm::text` refuses; a real valued feature
written that way is counted as 1 here where OpenNLP would read it.
