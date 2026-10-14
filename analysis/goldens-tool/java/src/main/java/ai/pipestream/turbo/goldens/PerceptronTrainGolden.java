package ai.pipestream.turbo.goldens;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Set;
import java.util.TreeMap;
import opennlp.tools.ml.AbstractEventTrainer;
import opennlp.tools.ml.EventModelSequenceTrainer;
import opennlp.tools.ml.EventTrainer;
import opennlp.tools.ml.TrainerFactory;
import opennlp.tools.ml.model.AbstractDataIndexer;
import opennlp.tools.ml.model.AbstractModel;
import opennlp.tools.ml.model.Context;
import opennlp.tools.ml.model.DataIndexer;
import opennlp.tools.ml.model.DataIndexerFactory;
import opennlp.tools.ml.model.Event;
import opennlp.tools.ml.model.GenericModelWriter;
import opennlp.tools.ml.model.MaxentModel;
import opennlp.tools.ml.model.Sequence;
import opennlp.tools.ml.model.SequenceStream;
import opennlp.tools.ml.perceptron.PerceptronTrainer;
import opennlp.tools.ml.perceptron.SimplePerceptronSequenceTrainer;
import opennlp.tools.monitoring.StopCriteria;
import opennlp.tools.monitoring.TrainingMeasure;
import opennlp.tools.monitoring.TrainingProgressMonitor;
import opennlp.tools.util.InsufficientTrainingDataException;
import opennlp.tools.util.ObjectStreamUtils;
import opennlp.tools.util.Parameters;
import opennlp.tools.util.TrainingConfiguration;
import opennlp.tools.util.TrainingParameters;

/**
 * The goldens of OpenNLP's perceptron trainers: {@link PerceptronTrainer}
 * ({@code Algorithm=PERCEPTRON}) over the index of a {@link TrainGolden.Source}'s events, and
 * {@link SimplePerceptronSequenceTrainer} ({@code PERCEPTRON_SEQUENCE}) over the same events cut
 * into sequences. As for the QN goldens, only the events are repeated here; the index is what
 * {@link TrainGolden} records for the same source, indexer, cutoff and sorting.
 *
 * <p>The perceptron trainer tells its progress monitor about iterations 1 to 9 and every tenth,
 * and about the end of training; the parameters after iteration k are those of a run of k
 * iterations (the mean over the first k with {@code UseAverage}), so each run of k iterations is
 * trained again to record them. The sequence trainer tells the monitor nothing; its iterations
 * are recorded the same way, with the accuracy it logs read from its own counting here.</p>
 *
 * <p>The sequence stream ({@link EventSequences}) cuts the source's events into sequences of
 * {@link #SEQUENCE_LENGTH} consecutive events (the last one shorter), and tags a sequence by
 * evaluating the model on each event's own context and taking {@code getBestOutcome}; the Rust
 * side's test implements the same stream.</p>
 *
 * <p>The layers, all anchored on the document:</p>
 * <ul>
 *   <li>{@code turbo:events} and {@code turbo:indexer-error}: as in {@link TrainGolden}.</li>
 *   <li>{@code turbo:perceptron-training}: one row: how many {@code iterations} ran and the
 *   {@code message} the monitor recorded last (the stop criterion's or the finished message; for
 *   the sequence trainer, which has no monitor, empty). With one outcome the event trainer
 *   refuses the index: zero iterations and the exception's {@code toString()}, and the layers
 *   below are left out. The sequence trainer fails the same way, after its iterations, when the
 *   cutoff dropped an event (its {@code trainingStats} reads past the indexer's outcome list).</li>
 *   <li>{@code turbo:perceptron-monitor}: one row per iteration the monitor heard of:
 *   {@code iteration}, {@code correct}, {@code events} and the {@code accuracy}; absent for the
 *   sequence trainer.</li>
 *   <li>{@code turbo:perceptron-iterations}: one row per iteration run: its number and
 *   {@code parameters}, the SHA-256 over the model's parameters after it as {@link TrainGolden}
 *   digests them.</li>
 *   <li>{@code turbo:perceptron-parameters}: every parameter after the first and after the last
 *   iteration: {@code iteration}, {@code predicate} id, {@code outcome} id and {@code value}.</li>
 *   <li>{@code turbo:perceptron-model}: one row, the SHA-256 and the {@code length} of the bytes
 *   {@link GenericModelWriter} writes for the final model, or an empty digest and zero length when
 *   the writer throws (it does when no parameter is left after dropping the zeros), with the
 *   exception's class in {@code error}.</li>
 * </ul>
 *
 * <p>The {@code config} records name the source, the indexer settings, {@code Algorithm},
 * {@code Iterations}, {@code UseAverage}, {@code UseSkippedAveraging}, {@code StepSizeDecrease}
 * and {@code Tolerance}. The directory is
 * {@code <source>-<indexer>-cutoff<n>[-unsorted]-perceptron[-raw][-skipped][-decrease]} for the
 * event trainer ({@code -raw} without averaging) and {@code ...-perceptron-sequence[-raw]} for
 * the sequence trainer.</p>
 */
final class PerceptronTrainGolden implements AnnotatorGolden {

  /** How many consecutive events make one sequence. */
  static final int SEQUENCE_LENGTH = 5;

  private static final String USE_AVERAGE = "UseAverage";
  private static final String USE_SKIPPED_AVERAGING = "UseSkippedAveraging";
  private static final String STEP_SIZE_DECREASE = "StepSizeDecrease";
  private static final String TOLERANCE = "Tolerance";

  private final TrainGolden.Source source;
  private final String indexer;
  private final int cutoff;
  private final boolean sort;
  private final int iterations;
  private final boolean sequence;
  private final boolean useAverage;
  private final boolean useSkippedAveraging;
  private final double stepSizeDecrease;
  private final double tolerance;
  private final Set<String> inputSets;

  PerceptronTrainGolden(TrainGolden.Source source, String indexer, int cutoff, boolean sort, int iterations,
      boolean sequence, boolean useAverage, boolean useSkippedAveraging, double stepSizeDecrease,
      double tolerance, Set<String> inputSets) {
    this.source = source;
    this.indexer = indexer;
    this.cutoff = cutoff;
    this.sort = sort;
    this.iterations = iterations;
    this.sequence = sequence;
    this.useAverage = useAverage;
    this.useSkippedAveraging = useSkippedAveraging;
    this.stepSizeDecrease = stepSizeDecrease;
    this.tolerance = tolerance;
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
    StringBuilder b = new StringBuilder(source.name).append('-').append(indexer.toLowerCase(Locale.ROOT))
        .append("-cutoff").append(cutoff).append(sort ? "" : "-unsorted").append("-perceptron");
    if (sequence) {
      b.append("-sequence");
    }
    if (!useAverage && !useSkippedAveraging) {
      b.append("-raw");
    }
    if (useSkippedAveraging) {
      b.append("-skipped");
    }
    if (stepSizeDecrease > 0) {
      b.append("-decrease");
    }
    return b.toString();
  }

  @Override
  public String annotatorClass() {
    return sequence ? SimplePerceptronSequenceTrainer.class.getName() : PerceptronTrainer.class.getName();
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
    c.put(Parameters.ALGORITHM_PARAM, algorithm());
    c.put(Parameters.ITERATIONS_PARAM, Integer.toString(iterations));
    c.put(USE_AVERAGE, Boolean.toString(useAverage));
    if (!sequence) {
      c.put(USE_SKIPPED_AVERAGING, Boolean.toString(useSkippedAveraging));
      c.put(STEP_SIZE_DECREASE, Double.toString(stepSizeDecrease));
      c.put(TOLERANCE, Double.toString(tolerance));
    }
    return c;
  }

  private String algorithm() {
    return sequence ? SimplePerceptronSequenceTrainer.PERCEPTRON_SEQUENCE_VALUE : PerceptronTrainer.PERCEPTRON_VALUE;
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
    layers.addAll(sequence ? sequenceTraining(events) : training(index));
    return layers;
  }

  /** The training parameters of a run of {@code iterations} iterations. */
  private TrainingParameters parameters(int iterations) {
    TrainingParameters params = new TrainingParameters();
    params.put(Parameters.ALGORITHM_PARAM, algorithm());
    params.put(Parameters.ITERATIONS_PARAM, iterations);
    params.put(Parameters.CUTOFF_PARAM, cutoff);
    params.put(AbstractEventTrainer.DATA_INDEXER_PARAM, indexer);
    params.put(AbstractDataIndexer.SORT_PARAM, sort);
    params.put(USE_AVERAGE, useAverage);
    if (!sequence) {
      params.put(USE_SKIPPED_AVERAGING, useSkippedAveraging);
      params.put(STEP_SIZE_DECREASE, stepSizeDecrease);
      params.put(TOLERANCE, tolerance);
    }
    return params;
  }

  /** Records what the perceptron trainer reports: some iterations, and the end of training. */
  private static final class Recorder implements TrainingProgressMonitor {
    final List<long[]> rows = new ArrayList<>();
    final List<Double> accuracies = new ArrayList<>();
    String last;
    boolean finished;

    @Override
    public void finishedIteration(int iteration, int numberCorrectEvents, int totalEvents, TrainingMeasure measure,
        double measureValue) {
      if (measure != TrainingMeasure.ACCURACY) {
        throw new IllegalStateException("iteration " + iteration + " reported " + measure);
      }
      rows.add(new long[] {iteration, numberCorrectEvents, totalEvents});
      accuracies.add(measureValue);
    }

    @Override
    public void finishedTraining(int iterations, StopCriteria<?> stopCriteria) {
      last = stopCriteria != null ? stopCriteria.getMessageIfSatisfied()
          : String.format(StopCriteria.FINISHED, iterations);
      finished = true;
    }

    @Override
    public boolean isTrainingFinished() {
      return finished;
    }

    @Override
    public void display(boolean clear) {}
  }

  /** The event trainer's layers: see the class comment. */
  private List<Layer> training(DataIndexer<TrainingParameters> index) {
    List<Layer> layers = new ArrayList<>();
    Recorder full = new Recorder();
    AbstractModel model;
    try {
      model = train(index, parameters(iterations), full);
    } catch (InsufficientTrainingDataException e) {
      layers.add(new Layer("turbo:perceptron-training", "document", new Column.U32("iterations", new long[] {0}),
          new Column.Str("message", new String[] {e.toString()})));
      return layers;
    }
    // The monitor hears of iterations 1 to 9 and every tenth, so the count
    // of iterations run is found by training again with the limit lowered
    // until the monitor's record changes: a run of k iterations ends with
    // the stop message when it stopped before k.
    int ran = iterationsRun(index, full);
    layers.add(new Layer("turbo:perceptron-training", "document", new Column.U32("iterations", new long[] {ran}),
        new Column.Str("message", new String[] {full.last})));
    layers.add(monitorLayer(full));
    layers.addAll(iterationLayers(k -> {
      try {
        return TrainGolden.contexts(train(index, parameters(k), new Recorder()));
      } catch (InsufficientTrainingDataException e) {
        throw new IllegalStateException(e);
      }
    }, TrainGolden.contexts(model), ran));
    layers.add(modelLayer(model));
    return layers;
  }

  /**
   * How many iterations the full run made. A run that stopped early recorded the stop criterion's
   * message; the iteration it stopped at is the smallest k whose run of k iterations records that
   * message too, since a run of k iterations is the full run cut short.
   */
  private int iterationsRun(DataIndexer<TrainingParameters> index, Recorder full) {
    if (!full.last.startsWith("Stopping")) {
      return iterations;
    }
    int lo = 1;
    int hi = iterations;
    while (lo < hi) {
      int mid = (lo + hi) / 2;
      Recorder r = new Recorder();
      try {
        train(index, parameters(mid), r);
      } catch (InsufficientTrainingDataException e) {
        throw new IllegalStateException(e);
      }
      if (r.last.startsWith("Stopping")) {
        hi = mid;
      } else {
        lo = mid + 1;
      }
    }
    return lo;
  }

  private static Layer monitorLayer(Recorder r) {
    int n = r.rows.size();
    long[] iteration = new long[n];
    long[] correct = new long[n];
    long[] events = new long[n];
    double[] accuracy = new double[n];
    for (int i = 0; i < n; i++) {
      iteration[i] = r.rows.get(i)[0];
      correct[i] = r.rows.get(i)[1];
      events[i] = r.rows.get(i)[2];
      accuracy[i] = r.accuracies.get(i);
    }
    return new Layer("turbo:perceptron-monitor", "document", new Column.U32("iteration", iteration),
        new Column.U32("correct", correct), new Column.U32("events", events), new Column.F64("accuracy", accuracy));
  }

  /** A run of k iterations. */
  private interface Partial {
    Context[] train(int k);
  }

  /** {@code turbo:perceptron-iterations} and {@code turbo:perceptron-parameters} from runs of 1 to ran iterations. */
  private static List<Layer> iterationLayers(Partial partial, Context[] last, int ran) {
    Context[][] after = new Context[ran][];
    for (int k = 1; k < ran; k++) {
      after[k - 1] = partial.train(k);
    }
    after[ran - 1] = last;
    long[] iteration = new long[ran];
    String[] digests = new String[ran];
    for (int i = 0; i < ran; i++) {
      iteration[i] = i + 1;
      digests[i] = TrainGolden.digest(after[i]);
    }
    List<Layer> layers = new ArrayList<>();
    layers.add(new Layer("turbo:perceptron-iterations", "document", new Column.U32("iteration", iteration),
        new Column.Str("parameters", digests)));
    layers.add(TrainGolden.parameters("turbo:perceptron-parameters", after, ran));
    return layers;
  }

  /** {@code turbo:perceptron-model}: the written bytes, or the writer's exception. */
  private static Layer modelLayer(AbstractModel model) {
    String sha;
    long length;
    String error = "";
    try {
      byte[] bytes = TrainGolden.modelBytes(model);
      sha = Sha.sha256(bytes);
      length = bytes.length;
    } catch (IllegalStateException e) {
      // modelBytes wraps IOException; the writer's own failure comes through unwrapped.
      throw e;
    } catch (RuntimeException e) {
      sha = "";
      length = 0;
      error = e.getClass().getName();
    }
    return new Layer("turbo:perceptron-model", "document", new Column.Str("sha256", new String[] {sha}),
        new Column.U32("length", new long[] {length}), new Column.Str("error", new String[] {error}));
  }

  private static AbstractModel train(DataIndexer<TrainingParameters> index, TrainingParameters params,
      Recorder recorder) throws InsufficientTrainingDataException {
    EventTrainer<TrainingParameters> trainer = TrainerFactory.getEventTrainer(params, new HashMap<>(),
        new TrainingConfiguration(recorder, null));
    MaxentModel model;
    try {
      model = trainer.train(index);
    } catch (InsufficientTrainingDataException e) {
      throw e;
    } catch (IOException e) {
      throw new IllegalStateException("the trainer failed", e);
    }
    return (AbstractModel) model;
  }

  /** The sequence trainer's layers: see the class comment. */
  private List<Layer> sequenceTraining(List<Event> events) {
    List<Layer> layers = new ArrayList<>();
    AbstractModel model;
    try {
      model = trainSequences(events, iterations);
    } catch (ArrayIndexOutOfBoundsException e) {
      // trainingStats, after the iterations, reads the indexer's outcome
      // list at each event's position in the stream, past its end when the
      // cutoff dropped an event. No model comes back.
      layers.add(new Layer("turbo:perceptron-training", "document", new Column.U32("iterations", new long[] {0}),
          new Column.Str("message", new String[] {e.toString()})));
      return layers;
    }
    layers.add(new Layer("turbo:perceptron-training", "document",
        new Column.U32("iterations", new long[] {iterations}), new Column.Str("message", new String[] {""})));
    layers.addAll(iterationLayers(k -> TrainGolden.contexts(trainSequences(events, k)),
        TrainGolden.contexts(model), iterations));
    layers.add(modelLayer(model));
    return layers;
  }

  private AbstractModel trainSequences(List<Event> events, int iterations) {
    EventModelSequenceTrainer<Event, TrainingParameters> trainer =
        TrainerFactory.getEventModelSequenceTrainer(parameters(iterations), new HashMap<>());
    try {
      return (AbstractModel) trainer.train(new EventSequences(events));
    } catch (IOException e) {
      throw new IllegalStateException("the sequence trainer failed", e);
    }
  }

  /**
   * The events cut into sequences of {@link #SEQUENCE_LENGTH}; tagging evaluates the model on each
   * event's context and takes the best outcome.
   */
  static final class EventSequences implements SequenceStream<Event> {
    private final List<Sequence<Event>> sequences = new ArrayList<>();
    private int next;

    EventSequences(List<Event> events) {
      for (int start = 0; start < events.size(); start += SEQUENCE_LENGTH) {
        List<Event> part = events.subList(start, Math.min(start + SEQUENCE_LENGTH, events.size()));
        sequences.add(new Sequence<>(part.toArray(new Event[0]), null));
      }
    }

    @Override
    public Event[] updateContext(Sequence<Event> sequence, MaxentModel model) {
      Event[] events = sequence.getEvents();
      Event[] tagged = new Event[events.length];
      for (int i = 0; i < events.length; i++) {
        double[] probs = events[i].getValues() == null ? model.eval(events[i].getContext())
            : model.eval(events[i].getContext(), events[i].getValues());
        tagged[i] = new Event(model.getBestOutcome(probs), events[i].getContext(), events[i].getValues());
      }
      return tagged;
    }

    @Override
    public Sequence<Event> read() {
      return next < sequences.size() ? sequences.get(next++) : null;
    }

    @Override
    public void reset() {
      next = 0;
    }

    @Override
    public void close() {}
  }

  /** Every run. */
  static List<PerceptronTrainGolden> all(Set<String> texts, Set<String> sentences) {
    Set<String> both = new java.util.HashSet<>(texts);
    both.addAll(sentences);
    List<PerceptronTrainGolden> runs = new ArrayList<>();
    // The trainer's defaults: averaging, no step decrease, the default tolerance.
    runs.add(new PerceptronTrainGolden(TrainGolden.Source.SENTDETECT, "OnePass", 1, false, 100, false, true, false,
        0, PerceptronTrainer.TOLERANCE_DEFAULT, both));
    // No averaging, a step decrease, a tolerance that stops early.
    runs.add(new PerceptronTrainGolden(TrainGolden.Source.SENTDETECT, "TwoPass", 5, false, 100, false, false, false,
        0.05, 0.01, texts));
    // Skipped averaging on real valued events kept with their values.
    runs.add(new PerceptronTrainGolden(TrainGolden.Source.REALVALUE, "OnePassRealValue", 1, false, 50, false, false,
        true, 0, PerceptronTrainer.TOLERANCE_DEFAULT, texts));
    // The sequence trainer, averaged and raw.
    runs.add(new PerceptronTrainGolden(TrainGolden.Source.SENTDETECT, "OnePass", 5, false, 20, true, true, false, 0,
        0, both));
    runs.add(new PerceptronTrainGolden(TrainGolden.Source.SENTDETECT, "OnePass", 1, false, 10, true, false, false, 0,
        0, texts));
    return runs;
  }
}
