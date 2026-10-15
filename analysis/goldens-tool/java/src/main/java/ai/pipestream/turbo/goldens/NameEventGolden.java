package ai.pipestream.turbo.goldens;

import java.io.ByteArrayInputStream;
import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.StringReader;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.IdentityHashMap;
import java.util.List;
import java.util.Map;
import java.util.Properties;
import java.util.Set;
import java.util.TreeMap;
import java.util.function.Supplier;
import java.util.zip.ZipEntry;
import java.util.zip.ZipInputStream;
import opennlp.tools.ml.model.AbstractModel;
import opennlp.tools.ml.model.BinaryFileDataReader;
import opennlp.tools.ml.model.ChecksumEventStream;
import opennlp.tools.ml.model.Context;
import opennlp.tools.ml.model.Event;
import opennlp.tools.ml.model.GenericModelReader;
import opennlp.tools.namefind.BilouCodec;
import opennlp.tools.namefind.BioCodec;
import opennlp.tools.namefind.NameFinderEventStream;
import opennlp.tools.namefind.NameFinderME;
import opennlp.tools.namefind.NameSample;
import opennlp.tools.namefind.NameSampleDataStream;
import opennlp.tools.namefind.TokenNameFinderFactory;
import opennlp.tools.namefind.TokenNameFinderModel;
import opennlp.tools.util.InputStreamFactory;
import opennlp.tools.util.ObjectStream;
import opennlp.tools.util.ObjectStreamUtils;
import opennlp.tools.util.PlainTextByLineStream;
import opennlp.tools.util.SequenceCodec;
import opennlp.tools.util.TrainingParameters;

/**
 * The name finder's training: the events of {@link NameFinderEventStream} as
 * {@code NameFinderME.train} makes them, and the model {@code NameFinderME.train} trains from the
 * same input, with the parameters a run gives and the defaults it puts in.
 *
 * <p>Each run is a codec, a type (or none) and an algorithm (or none). The factory is built as the
 * trainer tool builds it with no descriptor, {@code new TokenNameFinderFactory(null, Map.of(),
 * codec)}, so the default feature generators are loaded when the context generator is first made.
 * The samples are the input read with {@link NameSampleDataStream}. The parameters start empty,
 * as the trainer tool's are without a parameters file, and hold {@code Algorithm} when the run
 * names one; {@code NameFinderME.train} puts {@code MAXENT}, cutoff 0 and 300 iterations in where
 * absent. The {@code config} records name the run and every option: {@code run}, {@code language},
 * {@code codec}, and {@code type} and {@code algorithm} when given.</p>
 *
 * <p>Layers, all anchored on the document:</p>
 * <ul>
 *   <li>{@code turbo:events}, {@code turbo:stream-errors} and {@code turbo:event-checksum}: the
 *   events read one at a time, as {@code TaggerEventGolden} writes them; an exception from the
 *   sample stream or the event stream (a codec writing outside the sentence) is recorded with the
 *   number of events read before it, and reading goes on with the next sample.</li>
 *   <li>{@code turbo:name-training}: what {@code NameFinderME.train} over the input threw, its
 *   class and message, or one row of empty strings when it trained a model. The training reads
 *   the samples again from the input, so the first exception the event stream records stops it.</li>
 *   <li>{@code turbo:name-parameters}: the parameters after training, with the defaults put in:
 *   each key in the order the parameters keep them, the value's class and its string.</li>
 *   <li>For a trained model: {@code turbo:manifest} (without {@code Timestamp}) and
 *   {@code turbo:package-entries} of the serialized {@code TokenNameFinderModel};
 *   {@code turbo:name-model}: the SHA-256 and length of the {@code nameFinder.model} entry; and
 *   {@code turbo:name-weights}: every parameter of the model read back from that entry, its
 *   predicate and outcome labels (written as {@code lossless}) and its value.</li>
 * </ul>
 */
final class NameEventGolden implements AnnotatorGolden {

  /** The golden directory of the name finder's runs. */
  static final String ANNOTATOR = "namefind";

  enum Codec {
    BIO("bio", BioCodec::new),
    BILOU("bilou", BilouCodec::new);

    final String name;
    final Supplier<SequenceCodec<String>> make;

    Codec(String name, Supplier<SequenceCodec<String>> make) {
      this.name = name;
      this.make = make;
    }
  }

  /** The reads a stream may take before the generator gives it up as not ending. */
  private static final int MAX_READS = 100_000;

  private final String run;
  private final Codec codec;
  /** {@code null} for the samples' own types. */
  private final String type;
  /** {@code null} for {@code NameFinderME.train}'s default. */
  private final String algorithm;
  private final Set<String> inputSets;

  private NameEventGolden(String run, Codec codec, String type, String algorithm, Set<String> inputSets) {
    this.run = run;
    this.codec = codec;
    this.type = type;
    this.algorithm = algorithm;
    this.inputSets = Set.copyOf(inputSets);
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
    return "opennlp.tools.namefind.NameFinderEventStream";
  }

  @Override
  public String method() {
    return "read";
  }

  @Override
  public Map<String, String> config() {
    Map<String, String> c = new TreeMap<>();
    c.put("run", run);
    c.put("language", "en");
    c.put("codec", codec.name);
    if (type != null) {
      c.put("type", type);
    }
    if (algorithm != null) {
      c.put("algorithm", algorithm);
    }
    return c;
  }

  @Override
  public Set<String> inputSets() {
    return inputSets;
  }

  @Override
  public boolean accepts(Input input) {
    return inputSets.contains(input.set()) && input.id().startsWith("name");
  }

  /** The factory the trainer tool builds with no descriptor, a fresh one each time. */
  private TokenNameFinderFactory factory() {
    return new TokenNameFinderFactory(null, Map.of(), codec.make.get());
  }

  private static ObjectStream<NameSample> samples(Input input) throws IOException {
    InputStreamFactory in = () -> new ByteArrayInputStream(input.bytes());
    return new NameSampleDataStream(new PlainTextByLineStream(in, StandardCharsets.UTF_8));
  }

  @Override
  public List<Layer> layers(Input input) throws AnnotatorException {
    List<Event> events = new ArrayList<>();
    FormatGolden.Columns errors = new FormatGolden.Columns();
    TokenNameFinderFactory f = factory();
    try (ObjectStream<Event> stream = new NameFinderEventStream(samples(input), type,
        f.createContextGenerator(), f.createSequenceCodec())) {
      for (int reads = 0; ; reads++) {
        if (reads > MAX_READS) {
          throw new IllegalStateException(input.inputId() + ": the stream does not end");
        }
        Event e;
        try {
          e = stream.read();
        } catch (IOException | RuntimeException ex) {
          errors.u32("after", events.size()).str("class", ex.getClass().getName())
              .str("message", ex.getMessage() == null ? "" : ex.getMessage());
          continue;
        }
        if (e == null) {
          break;
        }
        events.add(e);
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
    try {
      layers.addAll(training(input));
    } catch (IOException e) {
      throw new IllegalStateException(input.inputId() + ": the model file failed", e);
    }
    return layers;
  }

  /** The events, the predicates joined with a space, everything written as {@code lossless}. */
  private static Layer eventsLayer(Input input, List<Event> events) {
    String[] outcomes = new String[events.size()];
    String[] contexts = new String[events.size()];
    String[] values = new String[events.size()];
    java.util.Arrays.fill(values, "");
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

  /** {@code NameFinderME.train} over the input, and what it made or threw. */
  private List<Layer> training(Input input) throws IOException {
    TrainingParameters params = new TrainingParameters();
    if (algorithm != null) {
      params.put("Algorithm", algorithm);
    }
    TokenNameFinderModel model = null;
    String errorClass = "";
    String errorMessage = "";
    try (ObjectStream<NameSample> samples = samples(input)) {
      model = NameFinderME.train("en", type, samples, params, factory());
    } catch (IOException | RuntimeException e) {
      errorClass = e.getClass().getName();
      errorMessage = e.getMessage() == null ? "" : e.getMessage();
    }
    List<Layer> layers = new ArrayList<>();
    layers.add(new Layer("turbo:name-training", "document", new Column.Str("class", new String[] {errorClass}),
        new Column.Str("message", new String[] {errorMessage})));

    List<String> keys = new ArrayList<>();
    List<String> classes = new ArrayList<>();
    List<String> values = new ArrayList<>();
    for (Map.Entry<String, Object> e : params.getObjectSettings().entrySet()) {
      keys.add(e.getKey());
      classes.add(e.getValue().getClass().getName());
      values.add(String.valueOf(e.getValue()));
    }
    layers.add(new Layer("turbo:name-parameters", "document", new Column.Str("key", keys.toArray(new String[0])),
        new Column.Str("class", classes.toArray(new String[0])),
        new Column.Str("value", values.toArray(new String[0]))));
    if (model != null) {
      layers.addAll(modelLayers(model));
    }
    return layers;
  }

  /** The layers of a trained model's file: see the class comment. */
  private static List<Layer> modelLayers(TokenNameFinderModel model) throws IOException {
    ByteArrayOutputStream bytes = new ByteArrayOutputStream();
    model.serialize(bytes);
    List<String> names = new ArrayList<>();
    List<Long> lengths = new ArrayList<>();
    String manifest = null;
    byte[] classifier = null;
    try (ZipInputStream zip = new ZipInputStream(new ByteArrayInputStream(bytes.toByteArray()))) {
      ZipEntry entry;
      while ((entry = zip.getNextEntry()) != null) {
        byte[] content = zip.readAllBytes();
        names.add(entry.getName());
        lengths.add((long) content.length);
        if (entry.getName().equals("manifest.properties")) {
          manifest = new String(content, StandardCharsets.UTF_8);
        } else if (entry.getName().equals("nameFinder.model")) {
          classifier = content;
        }
      }
    }
    if (manifest == null || classifier == null) {
      throw new IllegalStateException("the model file has no manifest or no classifier");
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
    layers.add(new Layer("turbo:manifest", "document",
        new Column.Str("key", sorted.keySet().toArray(new String[0])),
        new Column.Str("value", sorted.values().toArray(new String[0]))));
    layers.add(new Layer("turbo:package-entries", "document",
        new Column.Str("name", names.toArray(new String[0])),
        new Column.U32("length", lengths.stream().mapToLong(Long::longValue).toArray())));
    layers.add(new Layer("turbo:name-model", "document",
        new Column.Str("sha256", new String[] {Sha.sha256(classifier)}),
        new Column.U32("length", new long[] {classifier.length})));
    layers.add(weights(classifier));
    return layers;
  }

  /**
   * Every parameter of the classifier read back from its entry, by predicate in the order the
   * file holds them, each with its outcome. The model keeps its predicates in a map to their
   * parameters, so a predicate's label is found by the identity of its parameters.
   */
  private static Layer weights(byte[] classifier) throws IOException {
    AbstractModel m = new GenericModelReader(
        new BinaryFileDataReader(new ByteArrayInputStream(classifier))).getModel();
    Object[] data = m.getDataStructures();
    Context[] params = (Context[]) data[0];
    @SuppressWarnings("unchecked")
    Map<String, Context> pmap = (Map<String, Context>) data[1];
    String[] outcomeNames = (String[]) data[2];
    IdentityHashMap<Context, String> labels = new IdentityHashMap<>();
    for (Map.Entry<String, Context> e : pmap.entrySet()) {
      labels.put(e.getValue(), e.getKey());
    }
    List<String> predicates = new ArrayList<>();
    List<String> outcomes = new ArrayList<>();
    List<Double> values = new ArrayList<>();
    for (Context c : params) {
      String label = labels.get(c);
      if (label == null) {
        throw new IllegalStateException("a predicate's parameters have no label");
      }
      int[] os = c.getOutcomes();
      double[] vs = c.getParameters();
      for (int i = 0; i < vs.length; i++) {
        predicates.add(TrainGolden.lossless(label));
        outcomes.add(TrainGolden.lossless(outcomeNames[os[i]]));
        values.add(vs[i]);
      }
    }
    return new Layer("turbo:name-weights", "document",
        new Column.Str("predicate", predicates.toArray(new String[0])),
        new Column.Str("outcome", outcomes.toArray(new String[0])),
        new Column.F64("value", values.stream().mapToDouble(Double::doubleValue).toArray()));
  }

  /**
   * Every training run, and the run of {@code NameFinderME} over the fixture the Rust trainer
   * wrote from {@code inputs/tagged/names.txt}.
   */
  static List<AnnotatorGolden> all(java.nio.file.Path rustFixture) throws IOException {
    Set<String> sets = Set.of(Runs.TAGGED, Runs.FORMATS);
    List<AnnotatorGolden> runs = new ArrayList<>();
    // NameFinderME.train's defaults (GIS, cutoff 0, 300 iterations) with each codec, with the
    // type the stream gives every name, and with the other event trainers.
    runs.add(new NameEventGolden("train-bio", Codec.BIO, null, null, sets));
    runs.add(new NameEventGolden("train-bilou", Codec.BILOU, null, null, sets));
    runs.add(new NameEventGolden("train-type", Codec.BIO, "person", null, sets));
    runs.add(new NameEventGolden("train-perceptron", Codec.BIO, null, "PERCEPTRON", sets));
    runs.add(new NameEventGolden("train-qn", Codec.BILOU, null, "MAXENT_QN", sets));
    runs.add(NameFinderFixtureGolden.load(rustFixture, sets));
    return runs;
  }
}
