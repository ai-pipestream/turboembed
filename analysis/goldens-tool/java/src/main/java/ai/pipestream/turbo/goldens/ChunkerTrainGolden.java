package ai.pipestream.turbo.goldens;

import java.io.ByteArrayInputStream;
import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.StringReader;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Properties;
import java.util.Set;
import java.util.TreeMap;
import java.util.zip.ZipEntry;
import java.util.zip.ZipInputStream;
import opennlp.tools.chunker.ChunkSample;
import opennlp.tools.chunker.ChunkSampleStream;
import opennlp.tools.chunker.ChunkerEventStream;
import opennlp.tools.chunker.ChunkerFactory;
import opennlp.tools.chunker.ChunkerME;
import opennlp.tools.chunker.ChunkerModel;
import opennlp.tools.ml.model.AbstractModel;
import opennlp.tools.ml.model.ChecksumEventStream;
import opennlp.tools.ml.model.Context;
import opennlp.tools.ml.model.Event;
import opennlp.tools.util.ObjectStream;
import opennlp.tools.util.ObjectStreamUtils;
import opennlp.tools.util.PlainTextByLineStream;
import opennlp.tools.util.TrainingParameters;

/**
 * {@code ChunkerME.train} with {@code new ChunkerFactory()}: the events of
 * {@link ChunkerEventStream} over the factory's {@code DefaultChunkerContextGenerator}, and the
 * model the training returns, more runs of the golden directory {@code train-events}.
 *
 * <p>Each run is a set of training parameters: OpenNLP's defaults
 * ({@code TrainingParameters.defaultParams()}: {@code MAXENT}, 100 iterations, cutoff 5), the
 * perceptron with cutoff 1 and a beam size of 5, the L-BFGS trainer with cutoff 1, and the
 * sequence perceptron, which {@code ChunkerME.train} refuses. The samples are read from the
 * input with {@link ChunkSampleStream} over {@link PlainTextByLineStream} (UTF-8), as the
 * trainer tool reads them, and trained on as read. The {@code config} records name the run, the
 * source and language, and every training parameter as the run puts it.</p>
 *
 * <p>Layers: {@code turbo:events}, {@code turbo:stream-errors} and {@code turbo:event-checksum}
 * as {@link TaggerEventGolden} writes them; then either {@code turbo:train-error}, the class and
 * message of what {@code ChunkerME.train} threw, or the trained model:
 * {@code turbo:model-entry} (the {@code chunker.model} entry's algorithm, as the {@code Algorithm}
 * parameter names it, its length and SHA-256),
 * {@code turbo:model-outcomes} (the outcome labels in the model's order),
 * {@code turbo:model-predicates} (the predicate labels in the model's order, which is the
 * order its {@code pmap} iterates and the writer reads), {@code turbo:model-parameters} (every
 * parameter: its predicate's index, its outcome's index and its value), and
 * {@code turbo:manifest} and {@code turbo:package-entries} of the serialized model, as
 * {@link TaggerEventGolden} writes them (the manifest without {@code Timestamp}). Labels are
 * written as {@code lossless}.</p>
 */
final class ChunkerTrainGolden implements AnnotatorGolden {

  /** The start of the ids of the inputs read, of the formats and the tagged sets. */
  private static final String INPUT_PREFIX = "chunk";

  /** The reads a stream may take before the generator gives it up as not ending. */
  private static final int MAX_READS = 100_000;

  private final String run;
  /** The training parameters as put, in the order put; each value a string or an integer. */
  private final Map<String, Object> params;
  private final Set<String> inputSets;

  private ChunkerTrainGolden(String run, Map<String, Object> params, Set<String> inputSets) {
    this.run = run;
    this.params = params;
    this.inputSets = Set.copyOf(inputSets);
  }

  @Override
  public String annotator() {
    return EventGolden.ANNOTATOR;
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
    return "opennlp.tools.chunker.ChunkerME";
  }

  @Override
  public String method() {
    return "train";
  }

  @Override
  public Map<String, String> config() {
    Map<String, String> c = new TreeMap<>();
    c.put("run", run);
    c.put("source", "chunk");
    c.put("language", "en");
    for (Map.Entry<String, Object> e : params.entrySet()) {
      c.put(e.getKey(), String.valueOf(e.getValue()));
    }
    return c;
  }

  @Override
  public Set<String> inputSets() {
    return inputSets;
  }

  @Override
  public boolean accepts(Input input) {
    return inputSets.contains(input.set()) && input.id().startsWith(INPUT_PREFIX);
  }

  /** The parameters as {@code TrainingParameters}, integers put as integers. */
  private TrainingParameters trainingParameters() {
    TrainingParameters p = new TrainingParameters();
    for (Map.Entry<String, Object> e : params.entrySet()) {
      if (e.getValue() instanceof Integer i) {
        p.put(e.getKey(), i);
      } else {
        p.put(e.getKey(), (String) e.getValue());
      }
    }
    return p;
  }

  @Override
  public List<Layer> layers(Input input) throws AnnotatorException {
    List<ChunkSample> samples = new ArrayList<>();
    List<Event> events = new ArrayList<>();
    FormatGolden.Columns errors = new FormatGolden.Columns();
    ChunkerFactory factory = new ChunkerFactory();
    try (ObjectStream<ChunkSample> stream = new ChunkSampleStream(new PlainTextByLineStream(
        () -> new ByteArrayInputStream(input.bytes()), StandardCharsets.UTF_8))) {
      for (int reads = 0; ; reads++) {
        if (reads > MAX_READS) {
          throw new IllegalStateException(input.inputId() + ": the stream does not end");
        }
        ChunkSample s;
        try {
          s = stream.read();
        } catch (IOException | RuntimeException ex) {
          errors.u32("after", events.size()).str("class", ex.getClass().getName())
              .str("message", ex.getMessage() == null ? "" : ex.getMessage());
          continue;
        }
        if (s == null) {
          break;
        }
        samples.add(s);
        try (ObjectStream<Event> es = new ChunkerEventStream(ObjectStreamUtils.createObjectStream(s),
            factory.getContextGenerator())) {
          for (Event e = es.read(); e != null; e = es.read()) {
            events.add(e);
          }
        }
      }
    } catch (IOException e) {
      throw new IllegalStateException(input.inputId() + ": the stream failed", e);
    }

    List<Layer> layers = new ArrayList<>();
    layers.add(eventsLayer(input, events));
    layers.add(errors.layer("turbo:stream-errors", "document", "after", "class", "message"));
    ChecksumEventStream checked = new ChecksumEventStream(ObjectStreamUtils.createObjectStream(events));
    try {
      while (checked.read() != null) {
        // The checksum is over every event.
      }
    } catch (IOException e) {
      throw new IllegalStateException("the checksum stream failed", e);
    }
    layers.add(new Layer("turbo:event-checksum", "document",
        new Column.U32("crc32c", new long[] {checked.calculateChecksum()})));

    ChunkerModel model;
    try {
      model = ChunkerME.train("en", ObjectStreamUtils.createObjectStream(samples), trainingParameters(), factory);
    } catch (IOException | RuntimeException ex) {
      layers.add(new Layer("turbo:train-error", "document",
          new Column.Str("class", new String[] {ex.getClass().getName()}),
          new Column.Str("message", new String[] {ex.getMessage() == null ? "" : ex.getMessage()})));
      return layers;
    }
    try {
      layers.addAll(modelLayers(model));
    } catch (IOException e) {
      throw new IllegalStateException("the model file failed", e);
    }
    return layers;
  }

  /** The events as {@link TaggerEventGolden} writes them; a predicate holding a space stops the generator. */
  private static Layer eventsLayer(Input input, List<Event> events) {
    String[] outcomes = new String[events.size()];
    String[] contexts = new String[events.size()];
    String[] values = new String[events.size()];
    Arrays.fill(values, "");
    for (int i = 0; i < outcomes.length; i++) {
      Event e = events.get(i);
      outcomes[i] = TrainGolden.lossless(e.getOutcome());
      StringBuilder b = new StringBuilder();
      for (String c : e.getContext()) {
        if (c.indexOf(' ') >= 0) {
          throw new IllegalStateException(input.inputId() + ": the predicate " + c + " holds a space");
        }
        if (b.length() > 0) {
          b.append(' ');
        }
        b.append(TrainGolden.lossless(c));
      }
      contexts[i] = b.toString();
    }
    return new Layer("turbo:events", "document", new Column.Str("outcome", outcomes),
        new Column.Str("context", contexts), new Column.Str("values", values));
  }

  /** The trained model's entry, labels and parameters, and the file it serializes to. */
  private static List<Layer> modelLayers(ChunkerModel model) throws IOException {
    AbstractModel classifier = model.getArtifact("chunker.model");
    Object[] data = classifier.getDataStructures();
    @SuppressWarnings("unchecked")
    Map<String, Context> pmap = (Map<String, Context>) data[1];
    String[] outcomeNames = (String[]) data[2];

    List<String> predicates = new ArrayList<>();
    List<Long> predicate = new ArrayList<>();
    List<Long> outcome = new ArrayList<>();
    List<Double> value = new ArrayList<>();
    int p = 0;
    for (Map.Entry<String, Context> e : new LinkedHashMap<>(pmap).entrySet()) {
      predicates.add(TrainGolden.lossless(e.getKey()));
      int[] outcomes = e.getValue().getOutcomes();
      double[] parameters = e.getValue().getParameters();
      for (int o = 0; o < parameters.length; o++) {
        predicate.add((long) p);
        outcome.add((long) outcomes[o]);
        value.add(parameters[o]);
      }
      p++;
    }

    ByteArrayOutputStream bytes = new ByteArrayOutputStream();
    model.serialize(bytes);
    List<String> names = new ArrayList<>();
    List<Long> lengths = new ArrayList<>();
    String manifest = null;
    byte[] entry = null;
    try (ZipInputStream zip = new ZipInputStream(new ByteArrayInputStream(bytes.toByteArray()))) {
      ZipEntry z;
      while ((z = zip.getNextEntry()) != null) {
        byte[] content = zip.readAllBytes();
        names.add(z.getName());
        lengths.add((long) content.length);
        if (z.getName().equals("manifest.properties")) {
          manifest = new String(content, StandardCharsets.UTF_8);
        } else if (z.getName().equals("chunker.model")) {
          entry = content;
        }
      }
    }
    if (manifest == null || entry == null) {
      throw new IllegalStateException("the model file has no manifest or no chunker.model");
    }
    Properties properties = new Properties();
    properties.load(new StringReader(manifest));
    TreeMap<String, String> sorted = new TreeMap<>();
    for (String key : properties.stringPropertyNames()) {
      if (!key.equals("Timestamp")) {
        sorted.put(key, properties.getProperty(key));
      }
    }

    List<Layer> layers = new ArrayList<>();
    layers.add(new Layer("turbo:model-entry", "document",
        new Column.Str("type", new String[] {classifier.getModelType().getAlgorithmType()}),
        new Column.U32("length", new long[] {entry.length}),
        new Column.Str("sha256", new String[] {Sha.sha256(entry)})));
    layers.add(new Layer("turbo:model-outcomes", "document",
        new Column.Str("label", Arrays.stream(outcomeNames).map(TrainGolden::lossless).toArray(String[]::new))));
    layers.add(new Layer("turbo:model-predicates", "document",
        new Column.Str("label", predicates.toArray(new String[0]))));
    layers.add(new Layer("turbo:model-parameters", "document",
        new Column.U32("predicate", predicate.stream().mapToLong(Long::longValue).toArray()),
        new Column.U32("outcome", outcome.stream().mapToLong(Long::longValue).toArray()),
        new Column.F64("value", value.stream().mapToDouble(Double::doubleValue).toArray())));
    layers.add(new Layer("turbo:manifest", "document",
        new Column.Str("key", sorted.keySet().toArray(new String[0])),
        new Column.Str("value", sorted.values().toArray(new String[0]))));
    layers.add(new Layer("turbo:package-entries", "document",
        new Column.Str("name", names.toArray(new String[0])),
        new Column.U32("length", lengths.stream().mapToLong(Long::longValue).toArray())));
    return layers;
  }

  /** The parameters put in this order. */
  private static Map<String, Object> params(Object... keysAndValues) {
    Map<String, Object> m = new LinkedHashMap<>();
    for (int i = 0; i < keysAndValues.length; i += 2) {
      m.put((String) keysAndValues[i], keysAndValues[i + 1]);
    }
    return m;
  }

  /** Every run, over the chunk inputs of {@code inputSets}. */
  static List<ChunkerTrainGolden> all(Set<String> inputSets) {
    List<ChunkerTrainGolden> runs = new ArrayList<>();
    // TrainingParameters.defaultParams(), as the trainer tool uses without a parameters file.
    Map<String, Object> defaults = new LinkedHashMap<>(TrainingParameters.defaultParams().getObjectSettings());
    runs.add(new ChunkerTrainGolden("chunker-defaults", defaults, inputSets));
    runs.add(new ChunkerTrainGolden("chunker-perceptron",
        params("Algorithm", "PERCEPTRON", "Iterations", 100, "Cutoff", 1, "BeamSize", 5), inputSets));
    runs.add(new ChunkerTrainGolden("chunker-qn", params("Algorithm", "MAXENT_QN", "Cutoff", 1), inputSets));
    runs.add(new ChunkerTrainGolden("chunker-perceptron-sequence",
        params("Algorithm", "PERCEPTRON_SEQUENCE", "Iterations", 10, "Cutoff", 1), inputSets));
    return runs;
  }
}
