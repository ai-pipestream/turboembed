# liblinear trainer

This is documentation of the remaining classical training implementation,
scheduled for removal in #1532. Classical archive loading is disabled by #1511.

`turbo_train::liblinear` (`analysis/train/src/liblinear/mod.rs`) implements the
LIBLINEAR trainer (`LiblinearTrainer`, `LiblinearModel`,
`LiblinearModelSerializer`), with the solvers from `liblinear-java`
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

| Item | Behaviour |
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
- The serialized predicate map in stored predicate vector order
  (`format::serialize` writes `model.predicates` in slice order).

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

## Math functions and historical tolerances

liblinear-java calls `Math.exp` and `Math.log`. The active port uses
`f64::ln` and `f64::exp`.

During the historical JVM parity comparison (retired under #1511 and #1531),
HotSpot's `Math.exp` and `Math.log` were compared against fdlibm's
logarithm, platform `f64::ln`, and a correctly rounded logarithm. On the
historical goldens in a release build (training and checking all 285, the
median of three runs):

| Logarithm (historical) | Time | Text models identical (of 133) | Weights identical (of 112,188) |
|---|---|---|---|
| fdlibm logarithm | 0.30 s | 83 | 83,020 |
| platform `f64::ln` (glibc, active) | 0.33 s | 124 | 95,752 |
| correctly rounded logarithm | 0.56 s | 128 | 95,788 |

## Historical comparison status

The JVM golden comparison test was removed under #1511. The previous solver
counts and timing table above describe historical runs, not current acceptance
coverage (`analysis/train/src/liblinear/tests.rs` exercises the unit tests).
Classical archive readers are disabled; no JVM output or downloaded model is
required to validate this change. Removal of the remaining classical training
stack is tracked in #1532.
