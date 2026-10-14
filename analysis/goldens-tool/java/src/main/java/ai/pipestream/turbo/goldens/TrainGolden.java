package ai.pipestream.turbo.goldens;

import java.io.ByteArrayInputStream;
import java.io.ByteArrayOutputStream;
import java.io.DataOutputStream;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.charset.StandardCharsets;
import java.security.MessageDigest;
import java.security.NoSuchAlgorithmException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Set;
import java.util.TreeMap;
import opennlp.tools.ml.AbstractEventTrainer;
import opennlp.tools.ml.EventTrainer;
import opennlp.tools.ml.TrainerFactory;
import opennlp.tools.ml.model.AbstractDataIndexer;
import opennlp.tools.ml.model.AbstractModel;
import opennlp.tools.ml.model.ChecksumEventStream;
import opennlp.tools.ml.model.Context;
import opennlp.tools.ml.model.DataIndexer;
import opennlp.tools.ml.model.DataIndexerFactory;
import opennlp.tools.ml.model.Event;
import opennlp.tools.ml.model.GenericModelWriter;
import opennlp.tools.ml.model.MaxentModel;
import opennlp.tools.ml.model.RealValueFileEventStream;
import opennlp.tools.monitoring.DefaultTrainingProgressMonitor;
import opennlp.tools.monitoring.StopCriteria;
import opennlp.tools.monitoring.TrainingMeasure;
import opennlp.tools.monitoring.TrainingProgressMonitor;
import opennlp.tools.sentdetect.SDEventStream;
import opennlp.tools.sentdetect.SentenceDetectorFactory;
import opennlp.tools.sentdetect.SentenceSampleStream;
import opennlp.tools.tokenize.TokSpanEventStream;
import opennlp.tools.tokenize.TokenSampleStream;
import opennlp.tools.tokenize.TokenizerFactory;
import opennlp.tools.util.InsufficientTrainingDataException;
import opennlp.tools.util.ObjectStream;
import opennlp.tools.util.ObjectStreamUtils;
import opennlp.tools.util.Parameters;
import opennlp.tools.util.PlainTextByLineStream;
import opennlp.tools.util.StringUtil;
import opennlp.tools.util.TrainingConfiguration;
import opennlp.tools.util.TrainingParameters;

/**
 * The golden of OpenNLP's training stack below the sample types: the events an event stream makes
 * from the input text, what a data indexer makes of them, and what the GIS trainer makes of the
 * index after every iteration. The Rust port of the indexers and of the trainer is compared with
 * these; the event streams themselves are not ported here, so their events are recorded in full.
 *
 * <p>The events come from one of three sources ({@link Source}):</p>
 * <ul>
 *   <li>{@code sentdetect}: {@link SDEventStream} over the input read as sentence samples, one
 *   sentence per line ({@link SentenceSampleStream} over {@link PlainTextByLineStream}), with the
 *   context generator and scanner of {@code new SentenceDetectorFactory("en", false, null, null)}.
 *   This is what {@code SentenceDetectorME.train} runs.</li>
 *   <li>{@code tokenize}: {@link TokSpanEventStream} over the input read as token samples
 *   ({@link TokenSampleStream}, {@code <SPLIT>} marks), with the settings of
 *   {@code new TokenizerFactory("en", null, false, null)}, as {@code TokenizerME.train} runs.</li>
 *   <li>{@code realvalue}: one event per line with at least one field. The outcome is {@code long}
 *   when the line has eight or more whitespace separated fields and {@code short} otherwise; the
 *   contexts are the distinct fields in order of first occurrence, each with the number of its
 *   occurrences as its value. The line is written as {@code outcome field=count ...} and read
 *   back with {@link RealValueFileEventStream#parseEvent}, so the recorded event is what that
 *   parser produced.</li>
 * </ul>
 *
 * <p>The layers, all anchored on the document:</p>
 * <ul>
 *   <li>{@code turbo:events}: one row per event in stream order: {@code outcome}, {@code context}
 *   (the contexts joined with single spaces, as {@code FileEventStream.toLine} writes them) and
 *   {@code values} (the values as {@code Float.toString}, joined with single spaces; empty when
 *   the event has none). The tokenizer's contexts hold lone surrogates, which a {@code str}
 *   column cannot carry, so this column and the predicate labels are written through
 *   {@link #lossless}: a backslash doubled, a lone surrogate as the six characters
 *   {@code \\uXXXX}.</li>
 *   <li>{@code turbo:indexer-error}: one row, the class and message of the exception the indexer
 *   threw; the golden then ends here.</li>
 *   <li>{@code turbo:event-checksum}: one row, the CRC-32C {@link ChecksumEventStream} computed
 *   over the events while the indexer read them ({@code Training-Eventhash}).</li>
 *   <li>{@code turbo:indexer-outcomes}: the indexer's outcome labels, by outcome id.</li>
 *   <li>{@code turbo:indexer-predicates}: the predicate labels and their counts, by predicate
 *   id.</li>
 *   <li>{@code turbo:indexer-events}: one row per indexed (unique) event: its {@code outcome} id,
 *   its {@code context} (predicate ids joined with single spaces), its {@code values} (as in
 *   {@code turbo:events}; empty when the indexer keeps none or the event has none) and how often
 *   it was {@code seen}.</li>
 *   <li>{@code turbo:gis-training}: one row: how many {@code iterations} the GIS trainer ran and
 *   the {@code message} its progress monitor recorded last (the stop criterion's, or the finished
 *   message). When the index has one outcome, so that the trainer refuses it, the row has zero
 *   iterations and the exception's {@code toString()} as the message, and the layers below are
 *   left out.</li>
 *   <li>{@code turbo:gis-iterations}: one row per iteration run: its number, the log
 *   {@code likelihood} the monitor was given, the {@code correct} events of the {@code events}
 *   counted, and {@code parameters}, the SHA-256 of the model parameters after the iteration (the
 *   parameters of every predicate in order, each as the 8 bytes of its IEEE 754 bits, big endian).
 *   The parameters after iteration k are those of a training run of k iterations, which is the
 *   same computation cut short.</li>
 *   <li>{@code turbo:gis-parameters}: every parameter after the first iteration and after the last:
 *   {@code iteration}, {@code predicate} id, {@code outcome} id and {@code value}.</li>
 *   <li>{@code turbo:gis-model}: one row, the SHA-256 and the {@code length} of the bytes
 *   {@link GenericModelWriter} writes for the final model.</li>
 * </ul>
 *
 * <p>The indexer is run through {@link DataIndexerFactory} with the {@code DataIndexer},
 * {@code Cutoff} and {@code sort} parameters of the run, which the {@code config} records name
 * with the {@code source} and the {@code Iterations}; the directory is
 * {@code <source>-<indexer>-cutoff<n>}, with {@code -unsorted} when sorting is off. An index with
 * no event left after the cutoff makes the indexer throw
 * {@link InsufficientTrainingDataException}; the golden then has the events and, in place of the
 * indexer's layers, one {@code turbo:indexer-error} row with the exception's class and message,
 * so the events the Rust indexer must refuse are on record. The trainer runs with one thread and
 * otherwise OpenNLP's defaults.</p>
 */
final class TrainGolden implements AnnotatorGolden {

  /** Where the events come from. */
  enum Source {
    SENTDETECT("sentdetect"),
    TOKENIZE("tokenize"),
    REALVALUE("realvalue");

    final String name;

    Source(String name) {
      this.name = name;
    }
  }

  /** The language the factories are made for; it only names the model. */
  private static final String LANGUAGE = "en";
  /** How many fields make a {@code realvalue} line {@code long}. */
  private static final int LONG_LINE = 8;
  private static final String TRAINER_CLASS = "opennlp.tools.ml.maxent.GISTrainer";

  private final Source source;
  private final String indexer;
  private final int cutoff;
  private final boolean sort;
  private final int iterations;
  private final Set<String> inputSets;

  TrainGolden(Source source, String indexer, int cutoff, boolean sort, int iterations, Set<String> inputSets) {
    this.source = source;
    this.indexer = indexer;
    this.cutoff = cutoff;
    this.sort = sort;
    this.iterations = iterations;
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
    return source.name + "-" + indexer.toLowerCase(Locale.ROOT) + "-cutoff" + cutoff + (sort ? "" : "-unsorted");
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
    c.put(Parameters.ITERATIONS_PARAM, Integer.toString(iterations));
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
      events = events(source, input);
    } catch (IOException e) {
      throw new IllegalStateException(input.inputId() + ": the event stream failed", e);
    }
    List<Layer> layers = new ArrayList<>();
    layers.add(eventsLayer(events));

    TrainingParameters params = parameters(iterations);
    ChecksumEventStream checked = new ChecksumEventStream(ObjectStreamUtils.createObjectStream(events));
    DataIndexer<TrainingParameters> index = DataIndexerFactory.getDataIndexer(params, new HashMap<>());
    try {
      index.index(checked);
    } catch (InsufficientTrainingDataException e) {
      layers.add(new Layer("turbo:indexer-error", "document",
          new Column.Str("class", new String[] {e.getClass().getName()}),
          new Column.Str("message", new String[] {e.getMessage()})));
      return layers;
    } catch (IOException e) {
      throw new IllegalStateException(input.inputId() + ": the indexer failed", e);
    }
    layers.add(new Layer("turbo:event-checksum", "document",
        new Column.U32("crc32c", new long[] {checked.calculateChecksum()})));
    layers.add(new Layer("turbo:indexer-outcomes", "document", new Column.Str("label", index.getOutcomeLabels())));
    String[] predicates = index.getPredLabels().clone();
    for (int i = 0; i < predicates.length; i++) {
      predicates[i] = lossless(predicates[i]);
    }
    layers.add(new Layer("turbo:indexer-predicates", "document", new Column.Str("label", predicates),
        new Column.U32("count", longs(index.getPredCounts()))));
    layers.add(indexedEvents(index));
    layers.addAll(training(index, params));
    return layers;
  }

  /** The events of the source for the input, in stream order. */
  static List<Event> events(Source source, Input input) throws IOException {
    ObjectStream<String> lines = new PlainTextByLineStream(() -> new ByteArrayInputStream(input.bytes()),
        StandardCharsets.UTF_8);
    ObjectStream<Event> stream = switch (source) {
      case SENTDETECT -> {
        SentenceDetectorFactory f = new SentenceDetectorFactory(LANGUAGE, false, null, null);
        yield new SDEventStream(new SentenceSampleStream(lines), f.getSDContextGenerator(),
            f.getEndOfSentenceScanner());
      }
      case TOKENIZE -> {
        TokenizerFactory f = new TokenizerFactory(LANGUAGE, null, false, null);
        yield new TokSpanEventStream(new TokenSampleStream(lines), f.isUseAlphaNumericOptimization(),
            f.getAlphaNumericPattern(), f.getContextGenerator());
      }
      case REALVALUE -> realValueEvents(lines);
    };
    List<Event> events = new ArrayList<>();
    Event e;
    while ((e = stream.read()) != null) {
      events.add(e);
    }
    return events;
  }

  /** The {@code realvalue} events: see the class comment. */
  private static ObjectStream<Event> realValueEvents(ObjectStream<String> lines) throws IOException {
    List<Event> events = new ArrayList<>();
    String line;
    while ((line = lines.read()) != null) {
      String[] fields = StringUtil.splitOnUnicodeWhitespace(line);
      if (fields.length == 0) {
        continue;
      }
      Map<String, Integer> counts = new LinkedHashMap<>();
      for (String f : fields) {
        counts.merge(f, 1, Integer::sum);
      }
      StringBuilder b = new StringBuilder(fields.length >= LONG_LINE ? "long" : "short");
      for (Map.Entry<String, Integer> c : counts.entrySet()) {
        b.append(' ').append(c.getKey()).append('=').append(Float.toString(c.getValue()));
      }
      events.add(RealValueFileEventStream.parseEvent(b.toString()));
    }
    return ObjectStreamUtils.createObjectStream(events);
  }

  static Layer eventsLayer(List<Event> events) {
    String[] outcomes = new String[events.size()];
    String[] contexts = new String[events.size()];
    String[] values = new String[events.size()];
    for (int i = 0; i < outcomes.length; i++) {
      Event e = events.get(i);
      outcomes[i] = e.getOutcome();
      StringBuilder b = new StringBuilder();
      for (String c : e.getContext()) {
        if (b.length() > 0) {
          b.append(' ');
        }
        b.append(lossless(c));
      }
      contexts[i] = b.toString();
      values[i] = values(e.getValues());
    }
    return new Layer("turbo:events", "document", new Column.Str("outcome", outcomes),
        new Column.Str("context", contexts), new Column.Str("values", values));
  }

  private static Layer indexedEvents(DataIndexer<TrainingParameters> index) {
    int[][] contexts = index.getContexts();
    int n = contexts.length;
    long[] outcomes = longs(index.getOutcomeList());
    long[] seen = longs(index.getNumTimesEventsSeen());
    String[] context = new String[n];
    String[] values = new String[n];
    float[][] indexValues = index.getValues();
    for (int i = 0; i < n; i++) {
      StringBuilder b = new StringBuilder();
      for (int j = 0; j < contexts[i].length; j++) {
        if (j > 0) {
          b.append(' ');
        }
        b.append(contexts[i][j]);
      }
      context[i] = b.toString();
      values[i] = values(indexValues == null ? null : indexValues[i]);
    }
    return new Layer("turbo:indexer-events", "document", new Column.U32("outcome", outcomes),
        new Column.Str("context", context), new Column.Str("values", values), new Column.U32("seen", seen));
  }

  /**
   * The string with each backslash doubled and each lone surrogate written as {@code \\u} and four
   * lower case hex digits, so a {@code str} column, which holds UTF-8, can carry a predicate the
   * tokenizer cut at a UTF-16 unit. A paired surrogate is left as it is.
   */
  static String lossless(String s) {
    StringBuilder b = new StringBuilder(s.length());
    for (int i = 0; i < s.length(); i++) {
      char c = s.charAt(i);
      if (c == '\\') {
        b.append("\\\\");
      } else if (Character.isSurrogate(c) && !paired(s, i)) {
        b.append(String.format(Locale.ROOT, "\\u%04x", (int) c));
      } else {
        b.append(c);
      }
    }
    return b.toString();
  }

  /** Whether the surrogate at {@code i} forms a pair with its neighbour. */
  private static boolean paired(String s, int i) {
    char c = s.charAt(i);
    if (Character.isHighSurrogate(c)) {
      return i + 1 < s.length() && Character.isLowSurrogate(s.charAt(i + 1));
    }
    return i > 0 && Character.isHighSurrogate(s.charAt(i - 1));
  }

  /** The values as {@code Float.toString}, space separated; empty for none. */
  private static String values(float[] values) {
    if (values == null) {
      return "";
    }
    StringBuilder b = new StringBuilder();
    for (int i = 0; i < values.length; i++) {
      if (i > 0) {
        b.append(' ');
      }
      b.append(Float.toString(values[i]));
    }
    return b.toString();
  }

  private static long[] longs(int[] values) {
    long[] out = new long[values.length];
    for (int i = 0; i < values.length; i++) {
      out[i] = values[i];
    }
    return out;
  }

  /** The training parameters of a run of {@code iterations} iterations. */
  private TrainingParameters parameters(int iterations) {
    TrainingParameters params = new TrainingParameters();
    params.put(Parameters.ALGORITHM_PARAM, "MAXENT");
    params.put(Parameters.ITERATIONS_PARAM, iterations);
    params.put(Parameters.CUTOFF_PARAM, cutoff);
    params.put(AbstractEventTrainer.DATA_INDEXER_PARAM, indexer);
    params.put(AbstractDataIndexer.SORT_PARAM, sort);
    return params;
  }

  /** Records every iteration the trainer reports. */
  private static final class Recorder implements TrainingProgressMonitor {
    final List<long[]> counts = new ArrayList<>();
    final List<Double> likelihoods = new ArrayList<>();
    String last;
    boolean finished;

    @Override
    public void finishedIteration(int iteration, int numberCorrectEvents, int totalEvents, TrainingMeasure measure,
        double measureValue) {
      if (iteration != counts.size() + 1 || measure != TrainingMeasure.LOG_LIKELIHOOD) {
        throw new IllegalStateException("iteration " + iteration + " reported " + measure + " after "
            + counts.size() + " iterations");
      }
      counts.add(new long[] {numberCorrectEvents, totalEvents});
      likelihoods.add(measureValue);
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

  /** The GIS layers: see the class comment. */
  private List<Layer> training(DataIndexer<TrainingParameters> index, TrainingParameters params) {
    List<Layer> layers = new ArrayList<>();
    Recorder full = new Recorder();
    AbstractModel model;
    try {
      model = train(index, params, full);
    } catch (InsufficientTrainingDataException e) {
      layers.add(new Layer("turbo:gis-training", "document", new Column.U32("iterations", new long[] {0}),
          new Column.Str("message", new String[] {e.toString()})));
      return layers;
    }
    int ran = full.counts.size();
    layers.add(new Layer("turbo:gis-training", "document", new Column.U32("iterations", new long[] {ran}),
        new Column.Str("message", new String[] {full.last})));

    // The parameters after each iteration k are those of a run of k iterations.
    Context[][] after = new Context[ran][];
    for (int k = 1; k < ran; k++) {
      Recorder partial = new Recorder();
      AbstractModel m;
      try {
        m = train(index, parameters(k), partial);
      } catch (InsufficientTrainingDataException e) {
        throw new IllegalStateException(e);
      }
      if (partial.counts.size() != k) {
        throw new IllegalStateException("a run of " + k + " iterations reported " + partial.counts.size());
      }
      after[k - 1] = contexts(m);
    }
    after[ran - 1] = contexts(model);

    long[] iteration = new long[ran];
    double[] likelihood = new double[ran];
    long[] correct = new long[ran];
    long[] events = new long[ran];
    String[] digests = new String[ran];
    for (int i = 0; i < ran; i++) {
      iteration[i] = i + 1;
      likelihood[i] = full.likelihoods.get(i);
      correct[i] = full.counts.get(i)[0];
      events[i] = full.counts.get(i)[1];
      digests[i] = digest(after[i]);
    }
    layers.add(new Layer("turbo:gis-iterations", "document", new Column.U32("iteration", iteration),
        new Column.F64("likelihood", likelihood), new Column.U32("correct", correct),
        new Column.U32("events", events), new Column.Str("parameters", digests)));
    layers.add(parameters("turbo:gis-parameters", after, ran));

    byte[] bytes = modelBytes(model);
    layers.add(new Layer("turbo:gis-model", "document", new Column.Str("sha256", new String[] {Sha.sha256(bytes)}),
        new Column.U32("length", new long[] {bytes.length})));
    return layers;
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

  /** The model's parameters by predicate id. */
  static Context[] contexts(AbstractModel model) {
    return (Context[]) model.getDataStructures()[0];
  }

  /** The SHA-256 over every parameter's bits, predicates in order. */
  static String digest(Context[] contexts) {
    MessageDigest md;
    try {
      md = MessageDigest.getInstance("SHA-256");
    } catch (NoSuchAlgorithmException e) {
      throw new IllegalStateException(e);
    }
    ByteBuffer buf = ByteBuffer.allocate(8);
    for (Context c : contexts) {
      for (double p : c.getParameters()) {
        buf.clear();
        buf.putDouble(p);
        md.update(buf.array());
      }
    }
    return Sha.hex(md.digest());
  }

  /** The {@code id} layer: every parameter after the first and after the last iteration. */
  static Layer parameters(String id, Context[][] after, int ran) {
    int[] iterations = ran > 1 ? new int[] {1, ran} : new int[] {1};
    int rows = 0;
    for (int it : iterations) {
      for (Context c : after[it - 1]) {
        rows += c.getParameters().length;
      }
    }
    long[] iteration = new long[rows];
    long[] predicate = new long[rows];
    long[] outcome = new long[rows];
    double[] value = new double[rows];
    int r = 0;
    for (int it : iterations) {
      Context[] contexts = after[it - 1];
      for (int p = 0; p < contexts.length; p++) {
        int[] outcomes = contexts[p].getOutcomes();
        double[] values = contexts[p].getParameters();
        for (int o = 0; o < values.length; o++) {
          iteration[r] = it;
          predicate[r] = p;
          outcome[r] = outcomes[o];
          value[r] = values[o];
          r++;
        }
      }
    }
    return new Layer(id, "document", new Column.U32("iteration", iteration),
        new Column.U32("predicate", predicate), new Column.U32("outcome", outcome), new Column.F64("value", value));
  }

  /** What {@link GenericModelWriter} writes for the model. */
  static byte[] modelBytes(AbstractModel model) {
    ByteArrayOutputStream out = new ByteArrayOutputStream();
    try (DataOutputStream dos = new DataOutputStream(out)) {
      new GenericModelWriter(model, dos).persist();
    } catch (IOException e) {
      throw new IllegalStateException("the model writer failed", e);
    }
    return out.toByteArray();
  }

  /** Every run: the sources, indexers, cutoffs and sorting the goldens cover. */
  static List<TrainGolden> all(Set<String> texts, Set<String> sentences) {
    Set<String> both = new java.util.HashSet<>(texts);
    both.addAll(sentences);
    List<TrainGolden> runs = new ArrayList<>();
    // The sentence detector's events: the default cutoff, the cutoff the
    // fixture model en-plain.bin was trained with (TwoPass, 1, 100
    // iterations), the OnePass indexer, and an unsorted index.
    runs.add(new TrainGolden(Source.SENTDETECT, "TwoPass", 5, true, 100, texts));
    runs.add(new TrainGolden(Source.SENTDETECT, "TwoPass", 1, true, 100, both));
    runs.add(new TrainGolden(Source.SENTDETECT, "OnePass", 1, true, 100, both));
    runs.add(new TrainGolden(Source.SENTDETECT, "TwoPass", 0, false, 100, both));
    // The tokenizer's events.
    runs.add(new TrainGolden(Source.TOKENIZE, "TwoPass", 5, true, 100, texts));
    runs.add(new TrainGolden(Source.TOKENIZE, "OnePass", 2, false, 100, texts));
    // Real valued events: the indexer that keeps the values, with and without
    // sorting, and the two that sort by them and then drop them.
    runs.add(new TrainGolden(Source.REALVALUE, "OnePassRealValue", 1, true, 100, texts));
    runs.add(new TrainGolden(Source.REALVALUE, "OnePassRealValue", 3, false, 100, texts));
    runs.add(new TrainGolden(Source.REALVALUE, "TwoPass", 1, true, 100, texts));
    runs.add(new TrainGolden(Source.REALVALUE, "OnePass", 2, true, 100, texts));
    return runs;
  }
}
