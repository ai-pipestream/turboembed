package ai.pipestream.turbo.goldens;

import java.io.ByteArrayInputStream;
import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.StringReader;
import java.nio.ByteBuffer;
import java.nio.charset.CharacterCodingException;
import java.nio.charset.CodingErrorAction;
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
import opennlp.tools.doccat.BagOfWordsFeatureGenerator;
import opennlp.tools.doccat.DoccatFactory;
import opennlp.tools.doccat.DoccatModel;
import opennlp.tools.doccat.DocumentCategorizerEventStream;
import opennlp.tools.doccat.DocumentCategorizerME;
import opennlp.tools.doccat.DocumentSample;
import opennlp.tools.doccat.DocumentSampleStream;
import opennlp.tools.doccat.EmojiFeatureGenerator;
import opennlp.tools.doccat.FeatureGenerator;
import opennlp.tools.doccat.NGramFeatureGenerator;
import opennlp.tools.ml.model.Event;
import opennlp.tools.tokenize.WhitespaceTokenizer;
import opennlp.tools.util.ObjectStream;
import opennlp.tools.util.ObjectStreamUtils;
import opennlp.tools.util.Parameters;
import opennlp.tools.util.PlainTextByLineStream;
import opennlp.tools.util.Span;
import opennlp.tools.util.TrainingParameters;
import opennlp.tools.util.model.ModelUtil;

/**
 * The maxent document categorizer, {@code opennlp.tools.doccat}: models trained with
 * {@link DocumentCategorizerME#train} with {@link BagOfWordsFeatureGenerator},
 * {@link NGramFeatureGenerator} and {@link EmojiFeatureGenerator}, and {@code categorize} on the
 * documents of the {@code doccat} set. The golden directory is {@code doccat/<run>}.
 *
 * <p>Each run trains one model on the SMS Spam Collection (Almeida, G&oacute;mez Hidalgo and
 * Yamakami, 2011; CC BY 4.0), which {@code fetch-models} downloads into the models directory
 * ({@link Runs#SMS_SPAM_CORPUS}) and which is never checked in, followed by the {@code emoji}
 * corpus of the {@code doccat-train} set, written for this project, since the SMS corpus has no
 * emoji. Both are read with {@link DocumentSampleStream} over {@link PlainTextByLineStream}: one
 * sample per line, the category and then the tokens, split by {@link WhitespaceTokenizer}. The
 * downloaded corpus is the golden's {@code model} record, with its SHA-512.</p>
 *
 * <p>Every run uses the generators' public constructors ({@code NGramFeatureGenerator()} is
 * bigrams), the ones {@link DoccatFactory} builds again from the model file's manifest, which
 * records only their class names. The model is trained once, on the first golden that needs it,
 * and serialized; the documents are categorized by the model read back from those bytes, as a
 * user of the file would load it.</p>
 *
 * <p>The golden of the {@code emoji} corpus ({@link Train}, method {@code train}) holds, anchored
 * on the document:</p>
 * <ul>
 *   <li>{@code turbo:doccat-samples}: one row per category of the training samples, in order of
 *   first appearance, with its {@code samples} count.</li>
 *   <li>{@code turbo:doccat-model}: the {@code sha256} and {@code length} of the
 *   {@code doccat.model} entry, the classifier's bytes.</li>
 *   <li>{@code turbo:manifest} and {@code turbo:package-entries}: the manifest without
 *   {@code Timestamp} and the zip entries (name and length) of the serialized model.</li>
 * </ul>
 *
 * <p>The golden of a document (method {@code categorize}) holds {@code opennlp:tokens} (the
 * {@link WhitespaceTokenizer} tokens the categorizer is given) and, anchored on the document:</p>
 * <ul>
 *   <li>{@code turbo:doccat-features}: the {@code feature}s the model's generators give for the
 *   tokens, one per row in order, as {@link DocumentCategorizerEventStream} builds the context and
 *   as {@link TrainGolden#lossless} writes them.</li>
 *   <li>{@code turbo:categories}: every category with its probability, best first
 *   ({@code Double.compare}, ties in category order); the first is {@code getBestCategory}'s.</li>
 * </ul>
 */
final class DoccatGolden implements AnnotatorGolden {

  static final String ANNOTATOR = "doccat";
  /** The documents categorized. */
  static final String INPUT_SET = "doccat";
  /** The corpus written for this project that is trained on after the downloaded one. */
  static final String TRAIN_SET = "doccat-train";
  /** The corpus's id in {@link #TRAIN_SET}. */
  static final String CORPUS = "emoji";
  /** The language code the models are trained with. */
  static final String LANGUAGE = "en";

  /**
   * One model: its algorithm, iterations and cutoff, or {@code null} for the parameters
   * {@link ModelUtil#createDefaultTrainingParameters()} gives.
   */
  record Run(String name, String algorithm, int iterations, int cutoff) {

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

    /** The {@code config} records: the run, the corpora, the generators and the parameters. */
    Map<String, String> config() {
      Map<String, String> c = new TreeMap<>();
      c.put("run", name);
      c.put("corpus", Runs.SMS_SPAM_CORPUS + "," + TRAIN_SET + "/" + CORPUS);
      c.put("language", LANGUAGE);
      StringBuilder names = new StringBuilder();
      for (FeatureGenerator g : generators()) {
        names.append(names.isEmpty() ? "" : ",").append(g.getClass().getName());
      }
      c.put("generators", names.toString());
      for (Map.Entry<String, Object> e : parameters().getObjectSettings().entrySet()) {
        c.put(e.getKey(), String.valueOf(e.getValue()));
      }
      return c;
    }
  }

  /** The feature generators of every run, in context order. */
  static FeatureGenerator[] generators() {
    try {
      return new FeatureGenerator[] {
          new BagOfWordsFeatureGenerator(), new NGramFeatureGenerator(), new EmojiFeatureGenerator()};
    } catch (IOException e) {
      throw new IllegalStateException("the default n-gram generator was refused", e);
    }
  }

  /**
   * The runs: GIS with the trainer's defaults (cutoff 5) and with every predicate kept, then
   * L-BFGS and the perceptron.
   */
  static List<Run> runs() {
    return List.of(
        new Run("sms-default", null, 0, 0),
        new Run("sms-maxent", "MAXENT", 100, 1),
        new Run("sms-qn", "MAXENT_QN", 100, 1),
        new Run("sms-perceptron", "PERCEPTRON", 100, 1));
  }

  /** For every run, the golden of its corpus and the golden of the documents, over one model. */
  static List<AnnotatorGolden> all(Path inputs, Path models) throws IOException {
    ModelFile sms = ModelFile.read(models.resolve(Runs.SMS_SPAM_CORPUS));
    List<AnnotatorGolden> out = new ArrayList<>();
    for (Run run : runs()) {
      Trained trained = new Trained(run, sms, inputs.resolve(TRAIN_SET).resolve(CORPUS + ".txt"));
      out.add(new Train(trained));
      out.add(new DoccatGolden(trained));
    }
    return out;
  }

  /** A run's model, trained on first use. */
  static final class Trained {

    final Run run;
    final ModelFile sms;
    private final Path corpus;
    private Input input;
    private byte[] file;
    private DoccatModel model;
    private DocumentCategorizerME categorizer;

    Trained(Run run, ModelFile sms, Path corpus) {
      this.run = run;
      this.sms = sms;
      this.corpus = corpus;
    }

    /** The downloaded corpus's lines, then the checked in corpus's. */
    @SuppressWarnings("unchecked")
    private ObjectStream<DocumentSample> samples() throws IOException {
      ObjectStream<String> lines = ObjectStreamUtils.concatenateObjectStream(
          new PlainTextByLineStream(() -> new ByteArrayInputStream(sms.bytes()), StandardCharsets.UTF_8),
          new PlainTextByLineStream(() -> new ByteArrayInputStream(input.bytes()), StandardCharsets.UTF_8));
      return new DocumentSampleStream(lines);
    }

    private synchronized void train() {
      if (file != null) {
        return;
      }
      try {
        StandardCharsets.UTF_8.newDecoder().onMalformedInput(CodingErrorAction.REPORT)
            .onUnmappableCharacter(CodingErrorAction.REPORT).decode(ByteBuffer.wrap(sms.bytes()));
      } catch (CharacterCodingException e) {
        throw new IllegalStateException(Runs.SMS_SPAM_CORPUS + ": not valid UTF-8", e);
      }
      try {
        input = Input.read(corpus, TRAIN_SET, CORPUS);
      } catch (IOException e) {
        throw new IllegalStateException(corpus + ": the training corpus could not be read", e);
      }
      try (ObjectStream<DocumentSample> samples = samples()) {
        DoccatModel trained = DocumentCategorizerME.train(LANGUAGE, samples, run.parameters(),
            new DoccatFactory(generators()));
        ByteArrayOutputStream out = new ByteArrayOutputStream();
        trained.serialize(out);
        file = out.toByteArray();
        model = new DoccatModel(new ByteArrayInputStream(file));
        categorizer = new DocumentCategorizerME(model);
      } catch (IOException e) {
        throw new IllegalStateException(input.inputId() + ": training failed", e);
      }
    }

    /** Each category of the training samples with its count, in order of first appearance. */
    Map<String, Long> categoryCounts() {
      train();
      Map<String, Long> counts = new java.util.LinkedHashMap<>();
      try (ObjectStream<DocumentSample> stream = samples()) {
        DocumentSample s;
        while ((s = stream.read()) != null) {
          counts.merge(s.getCategory(), 1L, Long::sum);
        }
      } catch (IOException e) {
        throw new IllegalStateException(input.inputId() + ": the samples could not be read", e);
      }
      return counts;
    }

    /** The context the model's generators give for the tokens, from a one sample event stream. */
    String[] context(String[] tokens) {
      train();
      DocumentSample sample = new DocumentSample("?", tokens);
      try (ObjectStream<Event> events = new DocumentCategorizerEventStream(
          ObjectStreamUtils.createObjectStream(sample), model.getFactory().getFeatureGenerators())) {
        Event e = events.read();
        if (e == null || events.read() != null) {
          throw new IllegalStateException("one sample gave other than one event");
        }
        return e.getContext();
      } catch (IOException e) {
        throw new IllegalStateException("the event stream failed", e);
      }
    }

    byte[] file() {
      train();
      return file;
    }

    DocumentCategorizerME categorizer() {
      train();
      return categorizer;
    }
  }

  private final Trained trained;

  private DoccatGolden(Trained trained) {
    this.trained = trained;
  }

  @Override
  public String annotator() {
    return ANNOTATOR;
  }

  @Override
  public List<ModelRef> models() {
    return List.of(trained.sms.ref());
  }

  @Override
  public String modelDir() {
    return trained.run.name();
  }

  @Override
  public String annotatorClass() {
    return DocumentCategorizerME.class.getName();
  }

  @Override
  public String method() {
    return "categorize";
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

    String[] context = trained.context(tokens).clone();
    for (int i = 0; i < context.length; i++) {
      context[i] = TrainGolden.lossless(context[i]);
    }
    layers.add(new Layer("turbo:doccat-features", "document", new Column.Str("feature", context)));

    DocumentCategorizerME categorizer = trained.categorizer();
    double[] probabilities = AnnotatorException.call(() -> categorizer.categorize(tokens));
    Integer[] order = new Integer[probabilities.length];
    for (int i = 0; i < order.length; i++) {
      order[i] = i;
    }
    // A stable sort, so tied categories stay in category order.
    Arrays.sort(order, (a, b) -> Double.compare(probabilities[b], probabilities[a]));
    String[] category = new String[order.length];
    double[] score = new double[order.length];
    for (int i = 0; i < order.length; i++) {
      category[i] = categorizer.getCategory(order[i]);
      score[i] = probabilities[order[i]];
    }
    if (order.length > 0 && !category[0].equals(categorizer.getBestCategory(probabilities))) {
      throw new IllegalStateException(input.inputId() + ": the first category is not getBestCategory's");
    }
    layers.add(new Layer("turbo:categories", "document", new Column.Str("category", category),
        new Column.F64("score", score)));
    return layers;
  }

  /** The golden of the checked in corpus: the samples, the classifier's bytes and the model file. */
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
      return List.of(trained.sms.ref());
    }

    @Override
    public String modelDir() {
      return trained.run.name();
    }

    @Override
    public String annotatorClass() {
      return DocumentCategorizerME.class.getName();
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
      Map<String, Long> counts = trained.categoryCounts();
      List<Layer> layers = new ArrayList<>();
      layers.add(new Layer("turbo:doccat-samples", "document",
          new Column.Str("category", counts.keySet().stream().map(TrainGolden::lossless).toArray(String[]::new)),
          new Column.U32("samples", counts.values().stream().mapToLong(Long::longValue).toArray())));
      try {
        layers.addAll(packageLayers(input, trained.file()));
      } catch (IOException e) {
        throw new IllegalStateException(input.inputId() + ": the model file could not be read", e);
      }
      return layers;
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
          } else if (e.getName().equals("doccat.model")) {
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
          new Layer("turbo:doccat-model", "document",
              new Column.Str("sha256", new String[] {Sha.sha256(classifier)}),
              new Column.U32("length", new long[] {classifier.length})),
          new Layer("turbo:manifest", "document", new Column.Str("key", sorted.keySet().toArray(new String[0])),
              new Column.Str("value", sorted.values().toArray(new String[0]))),
          new Layer("turbo:package-entries", "document", new Column.Str("name", names.toArray(new String[0])),
              new Column.U32("length", lengths.stream().mapToLong(Long::longValue).toArray())));
    }
  }
}
