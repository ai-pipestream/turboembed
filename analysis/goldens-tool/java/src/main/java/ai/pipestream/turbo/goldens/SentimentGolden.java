package ai.pipestream.turbo.goldens;

import java.io.ByteArrayInputStream;
import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.StringReader;
import java.nio.charset.StandardCharsets;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Map;
import java.util.Properties;
import java.util.Set;
import java.util.TreeMap;
import java.util.zip.ZipEntry;
import java.util.zip.ZipInputStream;
import opennlp.tools.ml.model.Event;
import opennlp.tools.sentiment.EmojiSentimentFactory;
import opennlp.tools.sentiment.SentimentEventStream;
import opennlp.tools.sentiment.SentimentFactory;
import opennlp.tools.sentiment.SentimentME;
import opennlp.tools.sentiment.SentimentModel;
import opennlp.tools.sentiment.SentimentSample;
import opennlp.tools.sentiment.SentimentSampleStream;
import opennlp.tools.tokenize.WhitespaceTokenizer;
import opennlp.tools.util.ObjectStream;
import opennlp.tools.util.Parameters;
import opennlp.tools.util.PlainTextByLineStream;
import opennlp.tools.util.Span;
import opennlp.tools.util.TrainingParameters;
import opennlp.tools.util.model.ModelUtil;

/**
 * The sentiment detector, {@code opennlp.tools.sentiment}: models trained with
 * {@link SentimentME#train} on the corpus of the {@code sentiment-train} set, with
 * {@link SentimentFactory} or {@link EmojiSentimentFactory}, and {@code probabilities} on the
 * documents of the {@code sentiment} set. The golden directory is {@code sentiment/<run>}.
 *
 * <p>Each run trains one model with its factory and training parameters. The corpus is read with
 * {@link SentimentSampleStream} over {@link PlainTextByLineStream}: one sample per line, the
 * sentiment and then the tokens, split by {@link WhitespaceTokenizer}. The model is trained once,
 * on the first golden that needs it, and serialized; the documents are scored by the model read
 * back from those bytes, as a user of the file would load it.</p>
 *
 * <p>The golden of the corpus ({@link Train}, method {@code train}) holds, anchored on the
 * document:</p>
 * <ul>
 *   <li>{@code turbo:sentiment-events}: one row per event of {@link SentimentEventStream}, its
 *   {@code outcome} and how many {@code predicates} it has.</li>
 *   <li>{@code turbo:sentiment-predicates}: one row per predicate of each event, in the event's
 *   order: the {@code event} number and the {@code predicate}, as {@link TrainGolden#lossless}
 *   writes it.</li>
 *   <li>{@code turbo:sentiment-model}: the {@code sha256} and {@code length} of the
 *   {@code sentiment.model} entry, the classifier's bytes.</li>
 *   <li>{@code turbo:manifest} and {@code turbo:package-entries}: the manifest without
 *   {@code Timestamp} and the zip entries (name and length) of the serialized model.</li>
 * </ul>
 *
 * <p>The golden of a document (method {@code probabilities}) holds {@code opennlp:tokens} (the
 * {@link WhitespaceTokenizer} tokens the detector is given) and, anchored on the document:</p>
 * <ul>
 *   <li>{@code turbo:sentiment-context}: the {@code feature}s the factory's context generator
 *   gives for the tokens, one per row in order, as {@link TrainGolden#lossless} writes them.</li>
 *   <li>{@code turbo:categories}: every outcome with its probability, best first
 *   ({@code Double.compare}, ties in outcome order); the first is {@code getBestSentiment}'s.</li>
 * </ul>
 */
final class SentimentGolden implements AnnotatorGolden {

  static final String ANNOTATOR = "sentiment";
  /** The documents scored. */
  static final String INPUT_SET = "sentiment";
  /** The corpus the models are trained on. */
  static final String TRAIN_SET = "sentiment-train";
  /** The corpus's id in {@link #TRAIN_SET}. */
  static final String CORPUS = "feedback";
  /** The language code the models are trained with. */
  static final String LANGUAGE = "en";

  /**
   * One model: its factory, and its algorithm, iterations and cutoff, or {@code null} for the
   * parameters {@link ModelUtil#createDefaultTrainingParameters()} gives.
   */
  record Run(String name, boolean emoji, String algorithm, int iterations, int cutoff) {

    SentimentFactory factory() {
      return emoji ? new EmojiSentimentFactory() : new SentimentFactory();
    }

    TrainingParameters parameters() {
      if (algorithm == null) {
        return ModelUtil.createDefaultTrainingParameters();
      }
      TrainingParameters p = new TrainingParameters();
      p.put(Parameters.ALGORITHM_PARAM, algorithm);
      p.put(Parameters.ITERATIONS_PARAM, iterations);
      p.put(Parameters.CUTOFF_PARAM, cutoff);
      return p;
    }

    /** The {@code config} records: the run, the corpus, the factory and the parameters. */
    Map<String, String> config() {
      Map<String, String> c = new TreeMap<>();
      c.put("run", name);
      c.put("corpus", TRAIN_SET + "/" + CORPUS);
      c.put("language", LANGUAGE);
      c.put("factory", factory().getClass().getName());
      for (Map.Entry<String, Object> e : parameters().getObjectSettings().entrySet()) {
        c.put(e.getKey(), String.valueOf(e.getValue()));
      }
      return c;
    }
  }

  /**
   * The runs: each factory with GIS and every predicate kept, then the emoji factory with the
   * command line trainer's defaults (cutoff 5), with L-BFGS and with the perceptron.
   */
  static List<Run> runs() {
    return List.of(
        new Run("tokens-maxent", false, "MAXENT", 100, 1),
        new Run("emoji-maxent", true, "MAXENT", 100, 1),
        new Run("emoji-default", true, null, 0, 0),
        new Run("emoji-qn", true, "MAXENT_QN", 100, 1),
        new Run("emoji-perceptron", true, "PERCEPTRON", 100, 1));
  }

  /** For every run, the golden of its corpus and the golden of the documents, over one model. */
  static List<AnnotatorGolden> all(Path inputs) {
    List<AnnotatorGolden> out = new ArrayList<>();
    for (Run run : runs()) {
      Trained trained = new Trained(run, inputs.resolve(TRAIN_SET).resolve(CORPUS + ".txt"));
      out.add(new Train(trained));
      out.add(new SentimentGolden(trained));
    }
    return out;
  }

  /** A run's model, trained on first use. */
  static final class Trained {

    final Run run;
    private final Path corpus;
    private Input input;
    private byte[] file;
    private SentimentModel model;
    private SentimentME detector;

    Trained(Run run, Path corpus) {
      this.run = run;
      this.corpus = corpus;
    }

    private ObjectStream<SentimentSample> samples() throws IOException {
      return new SentimentSampleStream(new PlainTextByLineStream(
          () -> new ByteArrayInputStream(input.bytes()), StandardCharsets.UTF_8));
    }

    private synchronized void train() {
      if (file != null) {
        return;
      }
      try {
        input = Input.read(corpus, TRAIN_SET, CORPUS);
      } catch (IOException e) {
        throw new IllegalStateException(corpus + ": the training corpus could not be read", e);
      }
      try (ObjectStream<SentimentSample> samples = samples()) {
        SentimentModel model = SentimentME.train(LANGUAGE, samples, run.parameters(), run.factory());
        ByteArrayOutputStream out = new ByteArrayOutputStream();
        model.serialize(out);
        file = out.toByteArray();
        this.model = new SentimentModel(new ByteArrayInputStream(file));
        detector = new SentimentME(this.model);
      } catch (IOException e) {
        throw new IllegalStateException(input.inputId() + ": training failed", e);
      }
    }

    /** The events of {@link SentimentEventStream} over the corpus. */
    List<Event> events() {
      train();
      List<Event> events = new ArrayList<>();
      try (ObjectStream<Event> stream = new SentimentEventStream(samples(),
          run.factory().createContextGenerator())) {
        Event e;
        while ((e = stream.read()) != null) {
          events.add(e);
        }
      } catch (IOException e) {
        throw new IllegalStateException(input.inputId() + ": the event stream failed", e);
      }
      return events;
    }

    byte[] file() {
      train();
      return file;
    }

    SentimentME detector() {
      train();
      return detector;
    }

    /** The label of outcome {@code i} of the classifier read back from the file. */
    String outcome(int i) {
      train();
      return model.getMaxentModel().getOutcome(i);
    }
  }

  private final Trained trained;

  private SentimentGolden(Trained trained) {
    this.trained = trained;
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
    return trained.run.name();
  }

  @Override
  public String annotatorClass() {
    return SentimentME.class.getName();
  }

  @Override
  public String method() {
    return "probabilities";
  }

  @Override
  public Map<String, String> config() {
    return trained.run.config();
  }

  @Override
  public Set<String> inputSets() {
    return Set.of(INPUT_SET);
  }

  @Override
  public List<Layer> layers(Input input) throws AnnotatorException {
    Span[] spans = WhitespaceTokenizer.INSTANCE.tokenizePos(input.text());
    String[] tokens = Span.spansToStrings(spans, input.text());
    List<Layer> layers = new ArrayList<>(SpanLayers.of("opennlp:tokens", input.offsets(), spans));

    String[] context = trained.run.factory().createContextGenerator().getContext(tokens).clone();
    for (int i = 0; i < context.length; i++) {
      context[i] = TrainGolden.lossless(context[i]);
    }
    layers.add(new Layer("turbo:sentiment-context", "document", new Column.Str("feature", context)));

    SentimentME detector = trained.detector();
    double[] probabilities = AnnotatorException.call(() -> detector.probabilities(tokens));
    Integer[] order = new Integer[probabilities.length];
    for (int i = 0; i < order.length; i++) {
      order[i] = i;
    }
    // A stable sort, so tied outcomes stay in outcome order.
    Arrays.sort(order, (a, b) -> Double.compare(probabilities[b], probabilities[a]));
    String[] category = new String[order.length];
    double[] score = new double[order.length];
    for (int i = 0; i < order.length; i++) {
      category[i] = trained.outcome(order[i]);
      score[i] = probabilities[order[i]];
    }
    if (order.length > 0 && !category[0].equals(detector.getBestSentiment(probabilities))) {
      throw new IllegalStateException(input.inputId() + ": the first outcome is not getBestSentiment's");
    }
    layers.add(new Layer("turbo:categories", "document", new Column.Str("category", category),
        new Column.F64("score", score)));
    return layers;
  }

  /** The golden of the corpus: the events, the classifier's bytes and the model file. */
  static final class Train implements AnnotatorGolden {

    private final Trained trained;

    Train(Trained trained) {
      this.trained = trained;
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
      return trained.run.name();
    }

    @Override
    public String annotatorClass() {
      return SentimentME.class.getName();
    }

    @Override
    public String method() {
      return "train";
    }

    @Override
    public Map<String, String> config() {
      return trained.run.config();
    }

    @Override
    public Set<String> inputSets() {
      return Set.of(TRAIN_SET);
    }

    @Override
    public boolean accepts(Input input) {
      return input.set().equals(TRAIN_SET) && input.id().equals(CORPUS);
    }

    @Override
    public List<Layer> layers(Input input) {
      List<Layer> layers = new ArrayList<>(eventLayers(trained.events()));
      try {
        layers.addAll(packageLayers(input, trained.file()));
      } catch (IOException e) {
        throw new IllegalStateException(input.inputId() + ": the model file could not be read", e);
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
        outcomes[i] = TrainGolden.lossless(e.getOutcome());
        sizes[i] = e.getContext().length;
        for (String p : e.getContext()) {
          owner.add((long) i);
          predicates.add(TrainGolden.lossless(p));
        }
      }
      return List.of(
          new Layer("turbo:sentiment-events", "document", new Column.Str("outcome", outcomes),
              new Column.U32("predicates", sizes)),
          new Layer("turbo:sentiment-predicates", "document",
              new Column.U32("event", owner.stream().mapToLong(Long::longValue).toArray()),
              new Column.Str("predicate", predicates.toArray(new String[0]))));
    }

    /** The classifier's digest, the manifest and the zip entries of the serialized model. */
    private static List<Layer> packageLayers(Input input, byte[] file) throws IOException {
      List<String> names = new ArrayList<>();
      List<Long> lengths = new ArrayList<>();
      String manifest = null;
      byte[] classifier = null;
      try (ZipInputStream zip = new ZipInputStream(new ByteArrayInputStream(file))) {
        ZipEntry e;
        while ((e = zip.getNextEntry()) != null) {
          byte[] content = zip.readAllBytes();
          names.add(e.getName());
          lengths.add((long) content.length);
          if (e.getName().equals("manifest.properties")) {
            manifest = new String(content, StandardCharsets.UTF_8);
          } else if (e.getName().equals("sentiment.model")) {
            classifier = content;
          }
        }
      }
      if (manifest == null || classifier == null) {
        throw new IllegalStateException(input.inputId() + ": the model file lacks its manifest or classifier");
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
          new Layer("turbo:sentiment-model", "document",
              new Column.Str("sha256", new String[] {Sha.sha256(classifier)}),
              new Column.U32("length", new long[] {classifier.length})),
          new Layer("turbo:manifest", "document", new Column.Str("key", sorted.keySet().toArray(new String[0])),
              new Column.Str("value", sorted.values().toArray(new String[0]))),
          new Layer("turbo:package-entries", "document", new Column.Str("name", names.toArray(new String[0])),
              new Column.U32("length", lengths.stream().mapToLong(Long::longValue).toArray())));
    }
  }
}
