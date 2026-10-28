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
| `qn` | `QNTrainer`, `QNMinimizer`, `LineSearch`, `NegLogLikelihood`, `ParallelNegLogLikelihood`, `QNModel`'s data structures |
| `perceptron` | `PerceptronTrainer`, `SimplePerceptronSequenceTrainer`, `SequenceStream.updateContext`, `PerceptronModel`'s data structures and `eval` |
| `naive_bayes` | `NaiveBayesTrainer`, `NaiveBayesModel`'s data structures and Lidstone-smoothed log-probability evaluation |
| `liblinear` | the OpenNLP addons' `LiblinearTrainer`, `LiblinearModel` and `LiblinearModelSerializer`, with liblinear-java 2.44's `Linear.train` for `L2R_LR`, `L2R_LR_DUAL` and `L1R_LR`, `saveModel` and `loadModel` |
| `trainer` | `TrainerFactory.getEventTrainer` and `getEventModelSequenceTrainer`: the trainer `Algorithm` names |
| `writer` | `GenericModelWriter`, `BinaryGISModelWriter`, `BinaryQNModelWriter`, `BinaryPerceptronModelWriter`, `BinaryNaiveBayesModelWriter`, `GISModelWriter.persist`, `QNModelWriter.persist`, `PerceptronModelWriter.persist`, `NaiveBayesModelWriter.persist`, `ComparablePredicate`, `ModelParameterChunker` |
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
and the GIS, quasi-Newton and perceptron models hold outcome labels as
`JavaString`s, and the writer writes each label's modified UTF-8 as it
is. `Event::java_string` and the event checksum write the class as
`Event.toString` and `getBytes(UTF_8)` do (a lone surrogate becomes `?`).

No golden holds a model file the JVM trained on such a class. What
checks the written labels is the unit tests of their bytes (a lone high
surrogate as `ED A0 BD` after the `writeUTF` length, for each of the
three writers), the `train-events` goldens, whose lemma events and event
checksums are the JVM's, and the JVM reading the Rust lemmatizer fixture
(`analysis/goldens/data/fixtures/lemma-fixture-rust.bin`), whose
decision dumps list its two such classes.

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

## QN trainer

`qn::train_events` and `qn::train` are `TrainerFactory.getEventTrainer` for
`Algorithm=MAXENT_QN`, and `trainer::train_events` picks the GIS or the QN
trainer by that parameter. `qn::train_model` runs `QNMinimizer` from the
origin with an observer called after every iteration with the iteration's
value, change rate, step and evaluation count and the point the minimizer
is at; `qn::model_parameters` turns a point into the parameters a run
stopping there writes, and `qn::training_accuracy` is the accuracy the
JVM's `ModelEvaluator` computes. The port keeps the L-BFGS two loop
recursion with its `NumOfUpdates` corrections, the backtracking line search
and the constrained one of `L1Cost`, the L2 penalty of `L2Cost`, the four
stop tests (`QnStop`), the rescaling by `sqrt(1 + L2Cost)` when both costs
are on, `StrictMath` in the log sum of exponentials, and with `Threads`
above one the JVM's chunks of events and their order of summation.
`writer::write_qn` produces the `.model` bytes (`QNModelWriter.persist`:
the type `QN`, then what a GIS model has without its correction fields).

Parity: `QNTrainer` reports nothing to a progress monitor, so the goldens
of the QN runs (`QnTrainGolden`, directories ending in `-qn`, `-qn-l1`,
`-qn-l2` or `-qn-plain`) record what `QNMinimizer` logs at debug level,
captured by the generator's own SLF4J provider: per iteration the objective
value, its change rate and the training accuracy, and the test that stopped
it; the parameters after each iteration come from a run of that many
iterations, as for GIS. `tests/goldens.rs` trains every QN golden's index
and compares all of it, every parameter bit for bit after the first and the
last iteration and by digest in between, and the written model's bytes.
The runs stop by the change rate or the evaluation limit; none reaches the
gradient norm or the minimum step first, so those two tests are checked
against the Java code only. The runs cover L1 only, L2 only, both costs
and neither, two and three threads, discarded corrections, and the
part-of-speech shaped source with many outcomes.
`tests/roundtrip.rs` reproduces `analysis/maxent/tests/fixtures/models/qn.model`
byte for byte and trains the QN sentence model of `en-plain.bin`'s events
(`MAXENT_QN`, cutoff 1, the defaults): its `sent.model` bytes are the
ones the JVM's `QNTrainer` wrote for the same events and parameters (the
golden `sentdetect-twopass-cutoff1-qn/train/sentences` records their
SHA-256), the `.bin` written around them is the checked in fixture
`analysis/goldens/data/fixtures/en-plain-qn-rust.bin` byte for byte, and
it splits the training text into 210 sentences. The golden generator
loads that fixture in the JVM with `SentenceModel` and runs it on the
sentence fixture inputs (`analysis/goldens/data/sentdetect/en-plain-qn-rust`);
`analysis/sentdetect/tests/golden.rs` checks that the Rust detector gives
the spans and probabilities the JVM recorded with it.

## Perceptron trainers

`perceptron::train_events` and `perceptron::train` are
`TrainerFactory.getEventTrainer` for `Algorithm=PERCEPTRON`, and
`perceptron::train_sequences` is `getEventModelSequenceTrainer` for
`PERCEPTRON_SEQUENCE`; `trainer::train_events` and
`trainer::train_sequences` dispatch on the parameter and return a
`TrainedModel`. `PerceptronParams` reads `Iterations`, `UseAverage`,
`UseSkippedAveraging`, `StepSizeDecrease` and `Tolerance` with the
trainer's checks (`Error::Tolerance`, `Error::StepSizeDecrease`).
`perceptron::train_model` runs `PerceptronTrainer.trainModel` with an
observer called after every iteration with the correct and total counts,
and the parameters a run stopping there would return (the mean so far with
averaging); the progress monitor of the `TrainingConfiguration` hears what
the JVM's hears, and the stop criterion is `IterDeltaAccuracyUnderTolerance`
unless the configuration has one. The sequence trainer takes a
`SequenceSource`, the `SequenceStream` of the JVM: its sequences, and
`update_context`, which tags a sequence's events with the current model;
`perceptron::train_sequence_model` has the same observer. A
`PerceptronModel` keeps a parameter for every predicate and outcome;
`PerceptronModel::eval` is `PerceptronModel.eval` with its normalization
(sums divided by the largest absolute sum, then exponentiated with
`StrictMath.exp` and normalized). `writer::write_perceptron` produces the
`.model` bytes (`PerceptronModelWriter.persist`: the type `Perceptron`,
then the body of a QN model without the zero parameters and without the
predicates that have none left).

Parity: the goldens of the perceptron runs (`PerceptronTrainGolden`,
directories ending in `-perceptron`, `-perceptron-raw-decrease`,
`-perceptron-skipped`, `-perceptron-sequence` and
`-perceptron-sequence-raw`) record what the monitor heard (iterations 1 to
9 and every tenth, then the end of training), the parameter digest after
every iteration from a run of that many iterations, every parameter after
the first and the last, and the written model or the writer's failure.
The sequence runs cut the source's events into sequences of five and tag
each event by its own context; the test implements the same stream.
`tests/goldens.rs` compares all of it, as for GIS and QN.
The runs cover averaging, none and skipped averaging, a step decrease,
the tolerance stop, real valued events, and the part-of-speech shaped
source with many outcomes for both trainers.
`tests/roundtrip.rs` reproduces `analysis/maxent/tests/fixtures/models/perceptron.model`
byte for byte and trains the perceptron sentence model of `en-plain.bin`'s
events (`PERCEPTRON`, cutoff 1, the defaults): its `sent.model` bytes are
the ones the JVM's `PerceptronTrainer` wrote for the same events and
parameters (the golden `sentdetect-twopass-cutoff1-unsorted-perceptron/train/sentences`
records their SHA-256), the `.bin` written around them is the checked in
fixture `analysis/goldens/data/fixtures/en-plain-perceptron-rust.bin` byte
for byte, and it splits the training text into 212 sentences. The golden
generator loads that fixture in the JVM with `SentenceModel` and runs it
on the sentence fixture inputs (`analysis/goldens/data/sentdetect/en-plain-perceptron-rust`);
`analysis/sentdetect/tests/golden.rs` checks that the Rust detector gives
the spans and probabilities the JVM recorded with it.

## liblinear trainer

`liblinear::train_events` is the liblinear add-on of the OpenNLP addons:
`new LiblinearTrainer(params)`, `init` and `train`, with the logistic
regression solvers of liblinear-java ported in the module, the model's
`eval`, and its serialized form (`liblinear::format`). Only the solvers
whose models the add-on can score are ported. `docs/liblinear.md` lists
the add-on's quirks, the tolerances and the measured agreement with the
JVM; `tests/liblinear_goldens.rs` compares with the `train-liblinear`
goldens.

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
- `QNTrainer` never calls its `TrainingProgressMonitor` and reads no
  `StopCriteria`; the only account of its iterations is `QNMinimizer`'s
  debug log, and its training accuracy is computed only when that log is
  on. The Rust trainer takes no `TrainingConfiguration` and reports through
  the observer of `qn::train_model` instead.
- `QNMinimizer.UpdateInfo` discards the oldest correction by shifting array
  references (`S[i] = S[i + 1]`), so once `NumOfUpdates` corrections are
  stored the newest slot and the one before it are the same array, and each
  further discard adds a slot to the alias group. Writing the newest
  correction then overwrites the older ones, while their `rho` values stay;
  the two loop recursion runs on copies of the latest correction. The
  aliasing is reproduced (`qn::UpdateInfo` keeps slot indices), since the
  parameters depend on it from iteration `NumOfUpdates + 1` on.
- `ParallelNegLogLikelihood` sums each thread's share of the value and of
  the gradient and then adds the shares in thread order, so a run with
  several threads differs from one thread's in the last bits but is
  reproducible, unlike `GISTrainer`'s. The last chunk takes the remainder
  of the events (`GISTrainer` spreads it over the first chunks).
- `QNModel.eval(String[], double[])` adds into the array it is given
  without clearing it, so a caller that reuses an output array gets sums
  accumulated across calls; `GISModel` and `PerceptronModel` clear theirs.
  Training is unaffected: `ModelEvaluator` evaluates on a fresh array per
  event, and `turbo-maxent` clears the array, so a model trained here
  scores as the JVM scores it on a fresh array.
- The line search halves the step until the Armijo condition holds. When
  the objective is NaN it never does and the JVM loops without end; here
  the loop stops with `Error::LineSearch` once the step has reached zero.
- `OnePassRealValueDataIndexer` keeps each event's values array as it is,
  null for an event made without values, and the trainers read
  `values[ei][ci]` for every event once the indexer has values:
  `GISTrainer`, `NegLogLikelihood.valueAt` and `PerceptronTrainer` throw
  `NullPointerException`. The index is built (and matches the JVM's), and
  the trainers refuse it with `Error::MissingValues` where the JVM would
  throw: after the parameter checks and the outcome check.
- `AbstractEventTrainer.train(events)` validates before it indexes:
  `Cutoff` and `Iterations` must parse, and `QNTrainer` reads and checks
  its own parameters (`Threads`, the costs, `NumOfUpdates`, `MaxFctEval`)
  then too, so a bad parameter is reported whatever the events are.
  `GISTrainer` reads `Threads` and `PerceptronTrainer` its tolerance and
  step decrease in `doTrain`, after the outcome check.
- `PerceptronTrainer` tells its progress monitor about iterations 1 to 9
  and every tenth only; the observer of `perceptron::train_model` hears
  every iteration. With `Iterations` at zero or below no iteration runs,
  and with averaging the parameters are the sums divided by the number of
  times they were summed, `0.0 / 0`: a model of NaN parameters, kept as
  the JVM has it, and written as it is, since NaN is not zero to the
  writer. Its stop test compares the accuracy with each of the
  three before it, so it needs three iterations at least; the tolerance
  is the default `0.00001` and an iteration counts an event once per time
  it was seen.
- `UseSkippedAveraging` only forces `UseAverage` on: the test that would
  skip an iteration's sum (`useSkippedAveraging && ... || useAverage`) is
  true whenever averaging is on, so every iteration is summed. The result
  is the mean of the parameters after each iteration.
- `StepSizeDecrease` is documented as a percent and checked against 0 to
  100, but applied as a fraction (`stepsize *= 1 - decrease`); a value
  above 1 makes the step negative. It is applied before the first
  iteration too, so no iteration uses a step of 1.
- `PerceptronModelWriter` drops every zero parameter and every predicate
  without one left, and then fails on no predicates
  (`computeOutcomePatterns` reads the first; `ArrayIndexOutOfBoundsException`):
  a model whose parameters are all zero cannot be written
  (`Error::NoParameters`). The sequence trainer leaves one when the zero
  model, which tags every event with the first outcome, is right about
  every event, as it is for an input whose events all have one outcome
  (golden `sentdetect-onepass-cutoff1-unsorted-perceptron-sequence-raw/en/whitespace`,
  three events with outcome `s`).
- `SimplePerceptronSequenceTrainer` does not check for one outcome, indexes
  with `OnePassDataIndexer` unsorted (so no merging) and tells no monitor
  anything. Its averaging is lazy: a parameter's weight is its value
  truncated to an `int` times the sequences it held, in `int` arithmetic,
  and the final division is by `iterations * sequences` as a `double`.
  After the iterations `trainingStats` counts the events tagged right with
  the index's outcome list at each event's position in the stream, which
  reads past the list when the cutoff dropped an event, so training then
  throws and returns no model (`Error::SequenceEventsDropped`).
- The tokenizer's event stream over the golden inputs has one outcome (no
  text carries a split mark), so no trainer golden trains on it.

- `NaiveBayesTrainer` defaults to `SORT = false` and sums raw counts (or
  real values when `events` is a `OnePassRealValueDataIndexer`) per
  `(predicate, outcome)`. `NaiveBayesModel` computes empirical outcome priors
  and evaluates contexts with Lidstone smoothing (`delta = 0.05`, and
  `delta` times the number of predicates in the denominator) using
  `LogProbabilities` (`StrictMath.log` and `StrictMath.exp`).

The abbreviation dictionary entry of a sentence or tokenizer model
(`abbreviations.dictionary`, `DictionarySerializer`) is written by
`turbo-train-components::dictionary`; `ModelPackage::add_entry` takes its bytes.
