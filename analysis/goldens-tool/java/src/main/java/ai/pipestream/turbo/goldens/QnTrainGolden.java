package ai.pipestream.turbo.goldens;

import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Set;
import java.util.TreeMap;
import java.util.regex.Matcher;
import java.util.regex.Pattern;
import opennlp.tools.ml.AbstractEventTrainer;
import opennlp.tools.ml.EventTrainer;
import opennlp.tools.ml.TrainerFactory;
import opennlp.tools.ml.maxent.quasinewton.QNMinimizer;
import opennlp.tools.ml.maxent.quasinewton.QNTrainer;
import opennlp.tools.ml.model.AbstractDataIndexer;
import opennlp.tools.ml.model.AbstractModel;
import opennlp.tools.ml.model.Context;
import opennlp.tools.ml.model.DataIndexer;
import opennlp.tools.ml.model.DataIndexerFactory;
import opennlp.tools.ml.model.Event;
import opennlp.tools.ml.model.GenericModelWriter;
import opennlp.tools.ml.model.MaxentModel;
import opennlp.tools.util.InsufficientTrainingDataException;
import opennlp.tools.util.ObjectStreamUtils;
import opennlp.tools.util.Parameters;
import opennlp.tools.util.TrainingParameters;

/**
 * The golden of OpenNLP's L-BFGS maxent trainer ({@link QNTrainer}, {@code Algorithm=MAXENT_QN}):
 * the events of a {@link TrainGolden.Source}, and what the trainer makes of their index after
 * every iteration. The index itself is what {@link TrainGolden} records for the same source,
 * indexer, cutoff and sorting, so only the events are repeated here.
 *
 * <p>{@link QNMinimizer} reports nothing to a progress monitor. It logs, at debug level, one line
 * per iteration with the objective value after the line search, the function's change rate and
 * the training accuracy of the new point, and a line naming the criterion that stopped it; the
 * generator's {@link CapturingLogging} records those lines, and the number of lines is the number
 * of iterations run. The parameters after iteration k are those of a run of k iterations, which
 * is the same computation cut short (the minimizer's loop does not read the iteration limit), so
 * each run of k iterations is trained again to record them.</p>
 *
 * <p>The layers, all anchored on the document:</p>
 * <ul>
 *   <li>{@code turbo:events} and {@code turbo:indexer-error}: as in {@link TrainGolden}.</li>
 *   <li>{@code turbo:qn-training}: one row: how many {@code iterations} the minimizer ran, the
 *   {@code stop} that ended it ({@code change-rate}, {@code gradient-norm}, {@code step-size},
 *   {@code fct-eval}, or {@code iterations} when it ran the limit), and an empty {@code message}.
 *   When the index has one outcome the trainer refuses it: zero iterations, {@code stop} is
 *   {@code refused}, the message is the exception's {@code toString()}, and the layers below are
 *   left out.</li>
 *   <li>{@code turbo:qn-iterations}: one row per iteration: its number, the objective
 *   {@code value} at the new point (the negative log likelihood with the L2 term, and with the L1
 *   term when L1 is on), the function change {@code rate}, the training {@code accuracy} of the
 *   new point, and {@code parameters}, the SHA-256 over the model's parameters after the iteration
 *   as {@link TrainGolden} digests them (so after the elastic net rescaling the minimizer applies
 *   at the end when both costs are on).</li>
 *   <li>{@code turbo:qn-parameters}: every parameter after the first and after the last
 *   iteration: {@code iteration}, {@code predicate} id, {@code outcome} id and {@code value}.</li>
 *   <li>{@code turbo:qn-model}: one row, the SHA-256 and the {@code length} of the bytes
 *   {@link GenericModelWriter} writes for the final model.</li>
 * </ul>
 *
 * <p>The {@code config} records name the source, the indexer settings, {@code Algorithm},
 * {@code Iterations}, {@code L1Cost}, {@code L2Cost}, {@code NumOfUpdates}, {@code MaxFctEval} and
 * {@code Threads}. The directory is {@code <source>-<indexer>-cutoff<n>[-unsorted]-qn} with
 * {@code -l1}, {@code -l2} or {@code -plain} appended when only one cost, or neither, is above
 * zero.</p>
 */
final class QnTrainGolden implements AnnotatorGolden {

  private static final String TRAINER_CLASS = QNTrainer.class.getName();

  /** The debug line of one iteration: {@code iter: \tvalue\trate\taccuracy}. */
  private static final Pattern ITERATION = Pattern.compile("(\\d+): \t(\\S+)\t(\\S+)\t(\\S+)");

  /** The debug lines that name why the minimizer stopped, by their start. */
  private static final Map<String, String> STOPS = Map.of(
      "Function change rate is smaller than the threshold", "change-rate",
      "Relative L2-norm of the gradient is smaller than the threshold", "gradient-norm",
      "Step size is smaller than the minimum step size", "step-size",
      "Maximum number of function evaluations has exceeded the threshold", "fct-eval");

  private final TrainGolden.Source source;
  private final String indexer;
  private final int cutoff;
  private final boolean sort;
  private final int iterations;
  private final double l1Cost;
  private final double l2Cost;
  private final int updates;
  private final int maxFctEval;
  private final int threads;
  private final Set<String> inputSets;

  QnTrainGolden(TrainGolden.Source source, String indexer, int cutoff, boolean sort, int iterations,
      double l1Cost, double l2Cost, int updates, int maxFctEval, int threads, Set<String> inputSets) {
    this.source = source;
    this.indexer = indexer;
    this.cutoff = cutoff;
    this.sort = sort;
    this.iterations = iterations;
    this.l1Cost = l1Cost;
    this.l2Cost = l2Cost;
    this.updates = updates;
    this.maxFctEval = maxFctEval;
    this.threads = threads;
    this.inputSets = Set.copyOf(inputSets);
  }

  @Override
  public String annotator() {
    return "train";
  }

  @Override
  public List<ModelRef> models() {
    return List.of();
  }

  @Override
  public String modelDir() {
    String costs = l1Cost > 0 && l2Cost > 0 ? "" : l1Cost > 0 ? "-l1" : l2Cost > 0 ? "-l2" : "-plain";
    return source.name + "-" + indexer.toLowerCase(Locale.ROOT) + "-cutoff" + cutoff + (sort ? "" : "-unsorted")
        + "-qn" + costs;
  }

  @Override
  public String annotatorClass() {
    return TRAINER_CLASS;
  }

  @Override
  public String method() {
    return "train";
  }

  @Override
  public Map<String, String> config() {
    Map<String, String> c = new TreeMap<>();
    c.put("source", source.name);
    c.put(AbstractEventTrainer.DATA_INDEXER_PARAM, indexer);
    c.put(Parameters.CUTOFF_PARAM, Integer.toString(cutoff));
    c.put(AbstractDataIndexer.SORT_PARAM, Boolean.toString(sort));
    c.put(Parameters.ALGORITHM_PARAM, QNTrainer.MAXENT_QN_VALUE);
    c.put(Parameters.ITERATIONS_PARAM, Integer.toString(iterations));
    c.put(QNTrainer.L1COST_PARAM, Double.toString(l1Cost));
    c.put(QNTrainer.L2COST_PARAM, Double.toString(l2Cost));
    c.put(QNTrainer.M_PARAM, Integer.toString(updates));
    c.put(QNTrainer.MAX_FCT_EVAL_PARAM, Integer.toString(maxFctEval));
    c.put(QNTrainer.THREADS_PARAM, Integer.toString(threads));
    return c;
  }

  @Override
  public Set<String> inputSets() {
    return inputSets;
  }

  @Override
  public List<Layer> layers(Input input) throws AnnotatorException {
    List<Event> events;
    try {
      events = TrainGolden.events(source, input);
    } catch (IOException e) {
      throw new IllegalStateException(input.inputId() + ": the event stream failed", e);
    }
    List<Layer> layers = new ArrayList<>();
    layers.add(TrainGolden.eventsLayer(events));

    TrainingParameters params = parameters(iterations);
    DataIndexer<TrainingParameters> index = DataIndexerFactory.getDataIndexer(params, new HashMap<>());
    try {
      index.index(ObjectStreamUtils.createObjectStream(events));
    } catch (InsufficientTrainingDataException e) {
      layers.add(new Layer("turbo:indexer-error", "document",
          new Column.Str("class", new String[] {e.getClass().getName()}),
          new Column.Str("message", new String[] {e.getMessage()})));
      return layers;
    } catch (IOException e) {
      throw new IllegalStateException(input.inputId() + ": the indexer failed", e);
    }
    layers.addAll(training(index));
    return layers;
  }

  /** The training parameters of a run of {@code iterations} iterations. */
  private TrainingParameters parameters(int iterations) {
    TrainingParameters params = new TrainingParameters();
    params.put(Parameters.ALGORITHM_PARAM, QNTrainer.MAXENT_QN_VALUE);
    params.put(Parameters.ITERATIONS_PARAM, iterations);
    params.put(Parameters.CUTOFF_PARAM, cutoff);
    params.put(AbstractEventTrainer.DATA_INDEXER_PARAM, indexer);
    params.put(AbstractDataIndexer.SORT_PARAM, sort);
    params.put(QNTrainer.L1COST_PARAM, l1Cost);
    params.put(QNTrainer.L2COST_PARAM, l2Cost);
    params.put(QNTrainer.M_PARAM, updates);
    params.put(QNTrainer.MAX_FCT_EVAL_PARAM, maxFctEval);
    params.put(QNTrainer.THREADS_PARAM, threads);
    return params;
  }

  /** What the minimizer logged about one run: its iterations and why it stopped. */
  private record Run(AbstractModel model, List<double[]> iterations, String stop) {}

  /** The QN layers: see the class comment. */
  private List<Layer> training(DataIndexer<TrainingParameters> index) {
    List<Layer> layers = new ArrayList<>();
    Run full;
    try {
      full = train(index, parameters(iterations));
    } catch (InsufficientTrainingDataException e) {
      layers.add(new Layer("turbo:qn-training", "document", new Column.U32("iterations", new long[] {0}),
          new Column.Str("stop", new String[] {"refused"}), new Column.Str("message", new String[] {e.toString()})));
      return layers;
    }
    int ran = full.iterations.size();
    layers.add(new Layer("turbo:qn-training", "document", new Column.U32("iterations", new long[] {ran}),
        new Column.Str("stop", new String[] {full.stop}), new Column.Str("message", new String[] {""})));

    // The parameters after each iteration k are those of a run of k iterations.
    Context[][] after = new Context[ran][];
    for (int k = 1; k < ran; k++) {
      Run partial;
      try {
        partial = train(index, parameters(k));
      } catch (InsufficientTrainingDataException e) {
        throw new IllegalStateException(e);
      }
      if (partial.iterations.size() != k) {
        throw new IllegalStateException("a run of " + k + " iterations logged " + partial.iterations.size());
      }
      after[k - 1] = TrainGolden.contexts(partial.model);
    }
    after[ran - 1] = TrainGolden.contexts(full.model);

    long[] iteration = new long[ran];
    double[] value = new double[ran];
    double[] rate = new double[ran];
    double[] accuracy = new double[ran];
    String[] digests = new String[ran];
    for (int i = 0; i < ran; i++) {
      double[] row = full.iterations.get(i);
      iteration[i] = i + 1;
      value[i] = row[0];
      rate[i] = row[1];
      accuracy[i] = row[2];
      digests[i] = TrainGolden.digest(after[i]);
    }
    layers.add(new Layer("turbo:qn-iterations", "document", new Column.U32("iteration", iteration),
        new Column.F64("value", value), new Column.F64("rate", rate), new Column.F64("accuracy", accuracy),
        new Column.Str("parameters", digests)));
    layers.add(TrainGolden.parameters("turbo:qn-parameters", after, ran));

    byte[] bytes = TrainGolden.modelBytes(full.model);
    layers.add(new Layer("turbo:qn-model", "document", new Column.Str("sha256", new String[] {Sha.sha256(bytes)}),
        new Column.U32("length", new long[] {bytes.length})));
    return layers;
  }

  /** Trains with the minimizer's debug lines captured and read. */
  private static Run train(DataIndexer<TrainingParameters> index, TrainingParameters params)
      throws InsufficientTrainingDataException {
    EventTrainer<TrainingParameters> trainer = TrainerFactory.getEventTrainer(params, new HashMap<>());
    MaxentModel[] model = new MaxentModel[1];
    InsufficientTrainingDataException[] refused = new InsufficientTrainingDataException[1];
    List<String> lines = CapturingLogging.capture(QNMinimizer.class.getName(), () -> {
      try {
        model[0] = trainer.train(index);
      } catch (InsufficientTrainingDataException e) {
        refused[0] = e;
      } catch (IOException e) {
        throw new IllegalStateException("the trainer failed", e);
      }
    });
    if (refused[0] != null) {
      throw refused[0];
    }
    List<double[]> iterations = new ArrayList<>();
    String stop = "iterations";
    for (String line : lines) {
      Matcher m = ITERATION.matcher(line);
      if (m.matches()) {
        if (Integer.parseInt(m.group(1)) != iterations.size() + 1) {
          throw new IllegalStateException("iteration line out of order: " + line);
        }
        iterations.add(new double[] {
            Double.parseDouble(m.group(2)), Double.parseDouble(m.group(3)), Double.parseDouble(m.group(4))});
        continue;
      }
      for (Map.Entry<String, String> s : STOPS.entrySet()) {
        if (line.startsWith(s.getKey())) {
          if (!stop.equals("iterations")) {
            throw new IllegalStateException("two stop lines: " + stop + " and " + line);
          }
          stop = s.getValue();
        }
      }
    }
    if (iterations.isEmpty()) {
      throw new IllegalStateException("the minimizer logged no iteration; is debug logging captured?");
    }
    return new Run((AbstractModel) model[0], iterations, stop);
  }

  /** Every run: the costs, update counts, thread counts and sources the goldens cover. */
  static List<QnTrainGolden> all(Set<String> texts, Set<String> sentences) {
    Set<String> both = new java.util.HashSet<>(texts);
    both.addAll(sentences);
    List<QnTrainGolden> runs = new ArrayList<>();
    // The trainer's defaults: elastic net with both costs 0.1, 15 updates, one thread.
    runs.add(new QnTrainGolden(TrainGolden.Source.SENTDETECT, "TwoPass", 5, true, 100,
        QNTrainer.L1COST_DEFAULT, QNTrainer.L2COST_DEFAULT, QNTrainer.M_DEFAULT, QNTrainer.MAX_FCT_EVAL_DEFAULT,
        1, texts));
    // L2 only, few updates (the oldest are discarded early), two threads.
    runs.add(new QnTrainGolden(TrainGolden.Source.SENTDETECT, "TwoPass", 1, true, 100, 0, 0.1, 3,
        QNTrainer.MAX_FCT_EVAL_DEFAULT, 2, both));
    // L1 only (the constrained line search), three threads. The tokenizer's events have one
    // outcome on every input text (no text has split marks), so this run stays with the sentences.
    runs.add(new QnTrainGolden(TrainGolden.Source.SENTDETECT, "OnePass", 1, true, 100, 0.1, 0, QNTrainer.M_DEFAULT,
        QNTrainer.MAX_FCT_EVAL_DEFAULT, 3, texts));
    // No regularization on real valued events, and a limit on function evaluations low enough to
    // stop the two inputs with more than one outcome.
    runs.add(new QnTrainGolden(TrainGolden.Source.REALVALUE, "OnePassRealValue", 1, true, 100, 0, 0, 5, 10, 1,
        texts));
    // The same without the limit, on an unsorted index with its values, two threads.
    runs.add(new QnTrainGolden(TrainGolden.Source.REALVALUE, "OnePassRealValue", 3, false, 100, 0, 0, 5,
        QNTrainer.MAX_FCT_EVAL_DEFAULT, 2, texts));
    // Real valued events indexed without their values, both costs, two threads, fewer iterations.
    runs.add(new QnTrainGolden(TrainGolden.Source.REALVALUE, "TwoPass", 1, true, 30, QNTrainer.L1COST_DEFAULT,
        QNTrainer.L2COST_DEFAULT, QNTrainer.M_DEFAULT, QNTrainer.MAX_FCT_EVAL_DEFAULT, 2, texts));
    // The defaults on the fixture model's sentences with cutoff 1: the training of the fixture
    // en-plain-qn-rust.bin (analysis/train/tests/roundtrip.rs), whose model bytes this run records.
    runs.add(new QnTrainGolden(TrainGolden.Source.SENTDETECT, "TwoPass", 1, true, 100,
        QNTrainer.L1COST_DEFAULT, QNTrainer.L2COST_DEFAULT, QNTrainer.M_DEFAULT, QNTrainer.MAX_FCT_EVAL_DEFAULT,
        1, sentences));
    // The defaults on the part-of-speech shaped events, which have many outcomes.
    runs.add(new QnTrainGolden(TrainGolden.Source.POSTAG, "TwoPass", 1, true, 100,
        QNTrainer.L1COST_DEFAULT, QNTrainer.L2COST_DEFAULT, QNTrainer.M_DEFAULT, QNTrainer.MAX_FCT_EVAL_DEFAULT,
        1, texts));
    return runs;
  }
}
