package ai.pipestream.turbo.goldens;

import java.io.ByteArrayInputStream;
import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.StringReader;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Properties;
import java.util.Set;
import java.util.TreeMap;
import java.util.zip.ZipEntry;
import java.util.zip.ZipInputStream;
import opennlp.tools.langdetect.LanguageDetectorEventStream;
import opennlp.tools.langdetect.LanguageDetectorFactory;
import opennlp.tools.langdetect.LanguageDetectorME;
import opennlp.tools.langdetect.LanguageDetectorModel;
import opennlp.tools.langdetect.LanguageDetectorSampleStream;
import opennlp.tools.langdetect.LanguageSample;
import opennlp.tools.ml.AbstractEventTrainer;
import opennlp.tools.ml.model.AbstractModel;
import opennlp.tools.ml.model.ChecksumEventStream;
import opennlp.tools.ml.model.Context;
import opennlp.tools.ml.model.DataIndexer;
import opennlp.tools.ml.model.DataIndexerFactory;
import opennlp.tools.ml.model.Event;
import opennlp.tools.util.InsufficientTrainingDataException;
import opennlp.tools.util.ObjectStream;
import opennlp.tools.util.ObjectStreamUtils;
import opennlp.tools.util.Parameters;
import opennlp.tools.util.PlainTextByLineStream;
import opennlp.tools.util.TrainingParameters;
import opennlp.tools.util.model.ModelUtil;

/**
 * The language detector's training as {@code LanguageDetectorME.train} runs it: the events of
 * {@link LanguageDetectorEventStream}, what the indexer and the GIS trainer make of them, and the
 * model file. The golden directory is {@code train-langdetect/<run>}.
 *
 * <p>The samples are read from the input with {@link LanguageDetectorSampleStream} over
 * {@link PlainTextByLineStream} (a language code, a tab and the text per line), and the events
 * made with the context generator of {@code new LanguageDetectorFactory()}. The parameters are
 * those the command line trainer uses when it is given none,
 * {@link ModelUtil#createDefaultTrainingParameters()} ({@code MAXENT}, 100 iterations, cutoff 5),
 * with the run's cutoff; {@code LanguageDetectorME.train} puts {@code DataIndexer=OnePass} into
 * them, and the indexer and the trainer here run with the parameters it hands the trainer. The
 * {@code config} records name the {@code run} and those parameters.</p>
 *
 * <p>Layers, all anchored on the document:</p>
 * <ul>
 *   <li>{@code turbo:language-events}: one row per event in stream order, its {@code outcome}
 *   (the sample's language) and how many {@code predicates} it has.</li>
 *   <li>{@code turbo:language-predicates}: one row per predicate of each event, in the event's
 *   order: the {@code event} number and the {@code predicate}. An n-gram can hold a space or a
 *   tab, so the predicates of an event are not joined into one cell. The outcome and the
 *   predicates are written as {@link TrainGolden#lossless} writes them.</li>
 *   <li>{@code turbo:event-checksum}, {@code turbo:indexer-error}, {@code turbo:indexer-outcomes},
 *   {@code turbo:indexer-predicates}, {@code turbo:indexer-events}, {@code turbo:gis-training},
 *   {@code turbo:gis-iterations}, {@code turbo:gis-parameters} and {@code turbo:gis-model} as
 *   {@link TrainGolden} writes them, the indexer and the trainer run with the parameters above.</li>
 *   <li>{@code turbo:manifest} and {@code turbo:package-entries}: the manifest without
 *   {@code Timestamp} and the zip entries (name and length) of the model file
 *   {@code LanguageDetectorME.train} returns, serialized. The generator checks that the model it
 *   trained has the parameters of the last iteration above and that its {@code langdetect.model}
 *   entry is the bytes {@code turbo:gis-model} describes.</li>
 * </ul>
 */
final class LanguageTrainGolden implements AnnotatorGolden {

  static final String ANNOTATOR = "train-langdetect";
  /** The start of the ids of the language format inputs of the {@code formats} set. */
  private static final String FORMAT_PREFIX = "language-";

  private final String run;
  private final int cutoff;

  private LanguageTrainGolden(String run, int cutoff) {
    this.run = run;
    this.cutoff = cutoff;
  }

  /** The runs: the parameters langdetect-183 is reproduced with, and every predicate kept. */
  static List<LanguageTrainGolden> all() {
    return List.of(new LanguageTrainGolden("default", 5), new LanguageTrainGolden("cutoff1", 1));
  }

  @Override
  public String annotator() {
    return ANNOTATOR;
  }

  @Override
  public List<ModelRef> models() {
    return List.of();
  }

  @Override
  public String modelDir() {
    return run;
  }

  @Override
  public String annotatorClass() {
    return LanguageDetectorME.class.getName();
  }

  @Override
  public String method() {
    return "train";
  }

  @Override
  public Map<String, String> config() {
    Map<String, String> c = new TreeMap<>();
    c.put("run", run);
    for (Map.Entry<String, Object> e : trainerParameters(100).getObjectSettings().entrySet()) {
      c.put(e.getKey(), String.valueOf(e.getValue()));
    }
    return c;
  }

  @Override
  public Set<String> inputSets() {
    return Set.of(Runs.LANGDETECT, Runs.FORMATS);
  }

  @Override
  public boolean accepts(Input input) {
    return input.set().equals(Runs.LANGDETECT)
        || (input.set().equals(Runs.FORMATS) && input.id().startsWith(FORMAT_PREFIX));
  }

  /** The parameters {@code LanguageDetectorME.train} is called with, for {@code iterations}. */
  private TrainingParameters parameters(int iterations) {
    TrainingParameters p = ModelUtil.createDefaultTrainingParameters();
    p.put(Parameters.ITERATIONS_PARAM, iterations);
    p.put(Parameters.CUTOFF_PARAM, cutoff);
    return p;
  }

  /** The parameters it hands the trainer: {@code DataIndexer=OnePass} put if absent. */
  private TrainingParameters trainerParameters(int iterations) {
    TrainingParameters p = parameters(iterations);
    p.putIfAbsent(AbstractEventTrainer.DATA_INDEXER_PARAM, AbstractEventTrainer.DATA_INDEXER_ONE_PASS_VALUE);
    return p;
  }

  private static ObjectStream<LanguageSample> samples(Input input) throws IOException {
    return new LanguageDetectorSampleStream(
        new PlainTextByLineStream(() -> new ByteArrayInputStream(input.bytes()), StandardCharsets.UTF_8));
  }

  @Override
  public List<Layer> layers(Input input) throws AnnotatorException {
    List<Event> events = new ArrayList<>();
    try (ObjectStream<Event> stream = new LanguageDetectorEventStream(samples(input),
        new LanguageDetectorFactory().getContextGenerator())) {
      Event e;
      while ((e = stream.read()) != null) {
        events.add(e);
      }
    } catch (IOException e) {
      throw new IllegalStateException(input.inputId() + ": the event stream failed", e);
    }
    List<Layer> layers = new ArrayList<>();
    layers.addAll(eventLayers(events));

    int iterations = 100;
    TrainingParameters params = trainerParameters(iterations);
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
      predicates[i] = TrainGolden.lossless(predicates[i]);
    }
    long[] counts = new long[predicates.length];
    for (int i = 0; i < counts.length; i++) {
      counts[i] = index.getPredCounts()[i];
    }
    layers.add(new Layer("turbo:indexer-predicates", "document", new Column.Str("label", predicates),
        new Column.U32("count", counts)));
    layers.add(indexedEvents(index));

    TrainGolden.Recorder full = new TrainGolden.Recorder();
    AbstractModel model;
    try {
      model = TrainGolden.train(index, params, full);
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
      TrainGolden.Recorder partial = new TrainGolden.Recorder();
      AbstractModel m;
      try {
        m = TrainGolden.train(index, trainerParameters(k), partial);
      } catch (InsufficientTrainingDataException e) {
        throw new IllegalStateException(e);
      }
      if (partial.counts.size() != k) {
        throw new IllegalStateException("a run of " + k + " iterations reported " + partial.counts.size());
      }
      after[k - 1] = TrainGolden.contexts(m);
    }
    after[ran - 1] = TrainGolden.contexts(model);

    long[] iteration = new long[ran];
    double[] likelihood = new double[ran];
    long[] correct = new long[ran];
    long[] total = new long[ran];
    String[] digests = new String[ran];
    for (int i = 0; i < ran; i++) {
      iteration[i] = i + 1;
      likelihood[i] = full.likelihoods.get(i);
      correct[i] = full.counts.get(i)[0];
      total[i] = full.counts.get(i)[1];
      digests[i] = TrainGolden.digest(after[i]);
    }
    layers.add(new Layer("turbo:gis-iterations", "document", new Column.U32("iteration", iteration),
        new Column.F64("likelihood", likelihood), new Column.U32("correct", correct),
        new Column.U32("events", total), new Column.Str("parameters", digests)));
    layers.add(TrainGolden.parameters("turbo:gis-parameters", after, ran));
    byte[] bytes = TrainGolden.modelBytes(model);
    layers.add(new Layer("turbo:gis-model", "document", new Column.Str("sha256", new String[] {Sha.sha256(bytes)}),
        new Column.U32("length", new long[] {bytes.length})));

    try {
      layers.addAll(packageLayers(input, digests[ran - 1], bytes));
    } catch (IOException e) {
      throw new IllegalStateException(input.inputId() + ": the model file failed", e);
    }
    return layers;
  }

  /** The events and their predicates: see the class comment. */
  private static List<Layer> eventLayers(List<Event> events) {
    String[] outcomes = new String[events.size()];
    long[] sizes = new long[events.size()];
    List<Long> owner = new ArrayList<>();
    List<String> predicates = new ArrayList<>();
    for (int i = 0; i < outcomes.length; i++) {
      Event e = events.get(i);
      if (e.getValues() != null) {
        throw new IllegalStateException("event " + i + " has values");
      }
      outcomes[i] = TrainGolden.lossless(e.getOutcome());
      sizes[i] = e.getContext().length;
      for (String p : e.getContext()) {
        owner.add((long) i);
        predicates.add(TrainGolden.lossless(p));
      }
    }
    return List.of(
        new Layer("turbo:language-events", "document", new Column.Str("outcome", outcomes),
            new Column.U32("predicates", sizes)),
        new Layer("turbo:language-predicates", "document",
            new Column.U32("event", owner.stream().mapToLong(Long::longValue).toArray()),
            new Column.Str("predicate", predicates.toArray(new String[0]))));
  }

  /** {@code turbo:indexer-events} as {@link TrainGolden} writes it, for events without values. */
  private static Layer indexedEvents(DataIndexer<TrainingParameters> index) {
    int[][] contexts = index.getContexts();
    int n = contexts.length;
    long[] outcomes = new long[n];
    long[] seen = new long[n];
    String[] context = new String[n];
    String[] values = new String[n];
    for (int i = 0; i < n; i++) {
      outcomes[i] = index.getOutcomeList()[i];
      seen[i] = index.getNumTimesEventsSeen()[i];
      StringBuilder b = new StringBuilder();
      for (int j = 0; j < contexts[i].length; j++) {
        if (j > 0) {
          b.append(' ');
        }
        b.append(contexts[i][j]);
      }
      context[i] = b.toString();
      values[i] = "";
    }
    if (index.getValues() != null) {
      throw new IllegalStateException("the indexer kept values");
    }
    return new Layer("turbo:indexer-events", "document", new Column.U32("outcome", outcomes),
        new Column.Str("context", context), new Column.Str("values", values), new Column.U32("seen", seen));
  }

  /**
   * The manifest and the zip entries of the model {@code LanguageDetectorME.train} returns, after
   * checking that its parameters hash to {@code digest} and its model entry is {@code modelBytes}.
   */
  private List<Layer> packageLayers(Input input, String digest, byte[] modelBytes) throws IOException {
    LanguageDetectorModel model = LanguageDetectorME.train(samples(input), parameters(100),
        new LanguageDetectorFactory());
    String trained = TrainGolden.digest(TrainGolden.contexts((AbstractModel) model.getMaxentModel()));
    if (!trained.equals(digest)) {
      throw new IllegalStateException(input.inputId() + ": LanguageDetectorME.train made parameters " + trained
          + ", the monitored training " + digest);
    }
    ByteArrayOutputStream out = new ByteArrayOutputStream();
    model.serialize(out);

    List<String> names = new ArrayList<>();
    List<Long> lengths = new ArrayList<>();
    String manifest = null;
    byte[] entry = null;
    try (ZipInputStream zip = new ZipInputStream(new ByteArrayInputStream(out.toByteArray()))) {
      ZipEntry e;
      while ((e = zip.getNextEntry()) != null) {
        byte[] content = zip.readAllBytes();
        names.add(e.getName());
        lengths.add((long) content.length);
        if (e.getName().equals("manifest.properties")) {
          manifest = new String(content, StandardCharsets.UTF_8);
        } else if (e.getName().equals("langdetect.model")) {
          entry = content;
        }
      }
    }
    if (manifest == null || entry == null) {
      throw new IllegalStateException(input.inputId() + ": the model file lacks its manifest or model");
    }
    if (!java.util.Arrays.equals(entry, modelBytes)) {
      throw new IllegalStateException(input.inputId() + ": langdetect.model is not the monitored model's bytes");
    }
    Properties properties = new Properties();
    properties.load(new StringReader(manifest));
    TreeMap<String, String> sorted = new TreeMap<>();
    for (String key : properties.stringPropertyNames()) {
      if (!key.equals("Timestamp")) {
        sorted.put(key, properties.getProperty(key));
      }
    }
    return List.of(
        new Layer("turbo:manifest", "document", new Column.Str("key", sorted.keySet().toArray(new String[0])),
            new Column.Str("value", sorted.values().toArray(new String[0]))),
        new Layer("turbo:package-entries", "document", new Column.Str("name", names.toArray(new String[0])),
            new Column.U32("length", lengths.stream().mapToLong(Long::longValue).toArray())));
  }
}
