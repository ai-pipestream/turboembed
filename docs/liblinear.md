# liblinear trainer

`turbo_train::liblinear` (`analysis/train/src/liblinear/mod.rs`) is the
liblinear trainer of the OpenNLP addons, `liblinear-addon` on their
`main` branch (`LiblinearTrainer`, `LiblinearModel`,
`LiblinearModelSerializer`), with the solvers it uses from liblinear-java
2.44 (`de.bwaldvogel:liblinear`, a Java port of LIBLINEAR's
`linear.cpp`) ported into the crate. A model trained here has the JVM's
header, outcome labels and predicate map, and weights and probabilities
that agree with the JVM's to within the tolerances below.

It is a port and not a binding or another crate: the workspace takes no
new dependency for it, and parity with the JVM needs liblinear-java's
order of floating point operations, its `java.util.Random` shuffles and
its text model format, which a general linear model library does not
reproduce.

## Use

| Item | OpenNLP add-on |
|---|---|
| `liblinear::train_events(events, params)` | `new LiblinearTrainer(params)`, `init(params, ...)`, `train(events)`: the checksum, the indexer (sorting on, the default cutoff when none is given), then training |
| `liblinear::train(index, params)` | `AbstractEventTrainer.train(DataIndexer)` with the add-on's `doTrain` |
| `liblinear::train_model(index, &LiblinearParams)` | `doTrain` alone |
| `LiblinearParams::from_training_parameters` | the constructor's reads: `solverType`, `c`, `eps`, `p`, `bias` |
| `LiblinearModel::eval`, `eval_modified_utf8`, `best_outcome` | `LiblinearModel.eval`, `getBestOutcome` |
| `format::write`, `format::read` | `LiblinearModel.serialize`, `new LiblinearModel(InputStream)` |
| `format::save_model`, `format::load_model` | `Linear.saveModel`, `Linear.loadModel` |

`train_events` returns the model and a `TrainingReport` with the event
checksum, as the GIS trainer's does. `LiblinearKey` names the parameters.

## Solvers

Only the solvers whose models the add-on can score are ported, since
`LiblinearModel.eval` calls `Linear.predictProbability`, which throws for
every other solver:

- `L2R_LR`: the primal trust region Newton method with a preconditioned
  conjugate gradient (`analysis/train/src/liblinear/newton.rs`).
- `L2R_LR_DUAL`: dual coordinate descent
  (`analysis/train/src/liblinear/dual.rs`). After 300 outer iterations
  without converging, `train_one` hands the solution to the Newton method
  with a tenth of the tolerance, as liblinear-java does.
- `L1R_LR`: newGLMNET (`analysis/train/src/liblinear/l1.rs`). The add-on's
  sample parameters file (`LiblinearParams.txt`) uses it.

A model with more than two classes is trained one class against the
rest, in liblinear's class order, as `Linear.train` does. Any other
`SolverType` name is refused before training (`Error::NotScorable`); a
name that is no `SolverType` is refused with the add-on's message.

## What the port keeps

- Every sum in the order the Java loops add it; BLAS `ddot` five products
  at a time and `dnrm2` from the last element.
- One `java.util.Random` per `Parameter`, seeded with zero, which the
  dual and the L1 solver shuffle with, shared by the classes of a
  multi-class model in turn. liblinear-java's L1 shuffle draws
  `nextInt(QP_active_size - j)`, which is not the C code's draw; the port
  keeps the Java one.
- `Math.max` and `Math.min` with Java's NaN and signed zero rules.
- The text model as `saveModel` writes it: `%.17g` in `Locale.ENGLISH`
  for the bias and every nonzero weight, `0` for a zero weight, each
  weight followed by a space. Java's `%.17g` pads the shortest digits that
  read back to 17 significant digits; where a double lies exactly halfway
  between two shortest candidates that both read back, Java takes the
  even one, and for the eight smallest subnormals it writes two digits.
  The writer here does the same; it matched the JVM on 199,972 random
  doubles, on every double whose bits are 1 to 300,000, on 100,000 more
  subnormals and on the powers of two and their neighbours.
- The serialized predicate map in `HashMap<String, Integer>` iteration
  order (`turbo_jvm::hash_map`).

## Add-on quirks

- `TrainerFactory` cannot create the trainer: it has no constructor
  without arguments and its classes are in the default package. Its
  constructor reads the parameters without handing them to
  `AbstractTrainer`, so `train` fails with a `NullPointerException` unless
  `init` is called first. `train_events` is the working sequence.
- The parameter defaults are the constructor's: no solver (refused), and
  `c`, `eps`, `p` and `bias` zero. `c` and `eps` must be given; the
  `Parameter` checks run at training, after the index is built.
- liblinear refuses an instance whose feature indexes do not ascend, and
  the add-on passes each event's predicate ids in the event's order with
  duplicates kept. With the indexer's sorting, an event trains only when
  its predicates are in `String.compareTo` order with none twice
  (`Error::Unsorted`, which displays liblinear's message and the event).
- Each feature's value is the number of times the unique event was seen.
  With `bias` zero or more, every instance gets a last feature of that
  value; `eval` gives every known predicate the value one and never adds
  the bias feature.
- `eval` returns the probabilities in liblinear's class order (by first
  occurrence in the indexed events), which the add-on reads as outcome
  order. The two agree when the index is sorted and every outcome kept an
  event.

The model reader (`format::read`) takes only what `serialize` writes:
the text model's records in `saveModel`'s order with single spaces, the
weight count of each line, no bytes after the predicate map, no predicate
twice, ids from zero, and only logistic regression models.

## Math functions and tolerances

liblinear-java calls `Math.exp` and `Math.log`. On the JDK 21 HotSpot of
the goldens (x86-64), `Math.log` matched a correctly rounded logarithm on
all 199,972 arguments sampled, and the port uses `turbo_jvm::math::log`;
glibc's `ln` differs from it on 62 of them. HotSpot's `Math.exp` is not
correctly rounded and is not portable here; the port uses `f64::exp`,
which differs from it in the last bit on 472 of the 199,972. A last bit
difference can grow through the solver's iterations, so the weights are
compared to a tolerance.

Against a JVM oracle on 21 trainings of events written for the purpose
(two, three and many outcomes, each solver, tight and loose tolerances):

- 2,827 of 4,232 weights the same to the bit, the largest difference
  3.4e-13; six of the 17 models have the JVM's serialized bytes exactly;
- 10,134 of 17,800 probabilities the same to the bit, the largest
  difference 6.4e-14; the best outcome the same on all 4,100 contexts;
- the four refusals (unknown solver, `c` and `eps` of zero, an SVM
  solver) with the JVM's messages.

With `f64::ln` in place of the correctly rounded logarithm, 2,407 of the
4,232 weights are the same to the bit, and on the goldens below 124 text
models in place of 128 are the JVM's exactly. The correctly rounded
logarithm costs about 0.8 microseconds a call, and the dual solver takes
one at every step of its inner Newton method, so it dominates there: in
a release build, `L2R_LR_DUAL` on the part-of-speech shaped events of
`analysis/goldens/data/inputs/train/sentences.txt` (13 outcomes, `c` 100,
`eps` 1e-8) trains in about 1.2 s, against about 0.08 s with `f64::ln`.
The 21 oracle trainings take about 2.3 s, against about 0.6 s.

## Goldens

`LiblinearTrainGolden.java` in `analysis/goldens-tool` writes
`train-liblinear/<run>/<set>/<id>.golden`: the events given to the
trainer, then the exception it threw, or the text model, the SHA-256 and
length of the serialized model, and `eval` and `getBestOutcome` on the
contexts of the first 40 events, as given and reversed. The image builds
the add-on's three classes from the addons commit `recipe.rs` names into
a jar of their own, since a class in the default package cannot be
imported. The runs:

| Run | Events | Solver, `c`, `eps`, `bias` |
|---|---|---|
| `sentdetect-sample` | sentence detector, sorted | `L1R_LR`, 1, 0.001, 0 (the add-on's sample file, `p` 0.1) |
| `postag-l2` | part-of-speech shaped, sorted | `L2R_LR`, 1, 0.01, 1 |
| `postag-dual` | the same, `OnePass`, cutoff 0 | `L2R_LR_DUAL`, 0.5, 0.1, -1 |
| `sentdetect-dual-newton` | sentence detector, sorted, on `train` | `L2R_LR_DUAL`, 10000, 1e-10, 1: the dual solver stops at 300 iterations and the Newton method continues |
| `realvalue-l1` | real valued, sorted | `L1R_LR`, 2, 0.01, 1 |
| `sentdetect-l2` | sentence detector, sorted, cutoff 2 | `L2R_LR`, 0.1, 0.001, -1 |
| `sentdetect-raw` | sentence detector as the stream makes them | `L2R_LR`, 1, 0.01, 0: refused, the predicates do not ascend |

`analysis/train/tests/liblinear_goldens.rs` trains every golden's events
with its run's parameters and checks the refusals' messages, the text
model's header exactly and every weight within 1e-9. It rebuilds the
JVM's model from its text with the index's labels and predicate map and
checks that the bytes written from it have the JVM's SHA-256, that its
probabilities are within 1e-12 of the JVM's with the same best outcome,
and that the model trained here scores within 1e-9. It prints how many
weights and probabilities are the same to the bit and the largest
differences. Without the goldens it says so and passes; they are written
with `cargo run -p turbo-goldens-tool -- generate`.

On goldens of these runs made outside the image (the same generator and
add-on commit on an Ubuntu build of JDK 21.0.12, x86-64), the test gave:
285 goldens, 152 refusals with the JVM's messages, 133 models of which
128 have the JVM's text model exactly; 95,788 of 112,188 weights the same
to the bit, the largest difference 3.3e-14; the JVM's model scored here
34,652 of 34,772 probabilities the same to the bit, the largest
difference 1.1e-16, with every one of the 4,736 best outcomes the JVM's;
the model trained here 33,455 of 34,772, the largest difference 3.8e-15,
with the same best outcomes.

The fuzz differential does not cover the trainer (`train-liblinear` is in
its not covered list in `analysis/fuzz/src/runs/catalog.rs`): its goldens
are model files and scores, not layers of a text.
