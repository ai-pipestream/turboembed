package ai.pipestream.turbo.goldens;

import java.io.ByteArrayInputStream;
import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.StringReader;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.Properties;
import java.util.Set;
import java.util.TreeMap;
import java.util.TreeSet;
import java.util.zip.ZipEntry;
import java.util.zip.ZipInputStream;
import opennlp.tools.formats.conllu.ConlluLemmaSampleStream;
import opennlp.tools.formats.conllu.ConlluPOSSampleStream;
import opennlp.tools.formats.conllu.ConlluStream;
import opennlp.tools.formats.conllu.ConlluTagset;
import opennlp.tools.lemmatizer.LemmaSample;
import opennlp.tools.lemmatizer.LemmaSampleEventStream;
import opennlp.tools.lemmatizer.LemmaSampleStream;
import opennlp.tools.lemmatizer.LemmatizerFactory;
import opennlp.tools.lemmatizer.LemmatizerModel;
import opennlp.tools.ml.maxent.GISModel;
import opennlp.tools.ml.model.ChecksumEventStream;
import opennlp.tools.ml.model.Context;
import opennlp.tools.ml.model.Event;
import opennlp.tools.ml.model.MaxentModel;
import opennlp.tools.postag.POSDictionary;
import opennlp.tools.postag.POSModel;
import opennlp.tools.postag.POSSample;
import opennlp.tools.postag.POSSampleEventStream;
import opennlp.tools.postag.POSTaggerFactory;
import opennlp.tools.postag.WordTagSampleStream;
import opennlp.tools.util.InputStreamFactory;
import opennlp.tools.util.ObjectStream;
import opennlp.tools.util.ObjectStreamUtils;
import opennlp.tools.util.PlainTextByLineStream;
import opennlp.tools.util.model.BaseModel;

/**
 * The events of {@link POSSampleEventStream} and {@link LemmaSampleEventStream} as
 * {@code POSTaggerME.train} and {@code LemmatizerME.train} make them, and what the model file
 * around a trained model holds: more runs of the golden directory {@code train-events}.
 *
 * <p>Each run is a source and a sample format. The tagger's factory is built as the trainer tool
 * builds it, {@code new POSTaggerFactory(null, Map.of(), tagDictionary)}: no descriptor, so the
 * default feature generators are loaded when the context generator is first asked for, and a tag
 * dictionary read from a fixture file or none. The lemmatizer's factory has no settings. The
 * samples are read from the input with {@link WordTagSampleStream} or {@link LemmaSampleStream}
 * ({@code text}), or from CoNLL-U with {@link ConlluPOSSampleStream} or
 * {@link ConlluLemmaSampleStream} in a tag set ({@code conllu}). The {@code config} records name
 * the run and every option: {@code run}, {@code source}, {@code language}, {@code format},
 * {@code tagset} for CoNLL-U and {@code tagDictionary} (the fixture file) when given.</p>
 *
 * <p>Layers: {@code turbo:events}, {@code turbo:stream-errors}, {@code turbo:event-checksum},
 * {@code turbo:manifest} and {@code turbo:package-entries} as {@link EventGolden} writes them (the
 * file is {@code new POSModel} or {@code new LemmatizerModel} around a placeholder maxent model
 * whose outcomes are the tag dictionary's tags, since the model refuses a dictionary tag it has
 * no outcome for); {@code turbo:tagdict}: with a tag dictionary, its {@code tags.tagdict} entry as
 * text, written as {@code lossless}; {@code turbo:tagdict-cases}: on the {@code tagged} input of
 * the dictionary run only, {@code POSDictionary.serialize} of dictionaries built in the generator:
 * the fixture as read, a case-sensitive one with the same word in three cases, a case-insensitive
 * one with those words (which fold to one entry, the last tags winning), words and tags with the
 * characters the XML serializer escapes and an entry with no tags, an empty dictionary, forty
 * entries (the hash map's table doubles twice), and words whose lowercase is not ASCII.</p>
 */
final class TaggerEventGolden implements AnnotatorGolden {

  enum Source {
    POSTAG("postag", "opennlp.tools.postag.POSSampleEventStream", "wordtag"),
    LEMMATIZE("lemmatize", "opennlp.tools.lemmatizer.LemmaSampleEventStream", "lemma");

    final String name;
    final String streamClass;
    /** The start of the ids of the text inputs this source reads. */
    final String inputPrefix;

    Source(String name, String streamClass, String inputPrefix) {
      this.name = name;
      this.streamClass = streamClass;
      this.inputPrefix = inputPrefix;
    }
  }

  /** The reads a stream may take before the generator gives it up as not ending. */
  private static final int MAX_READS = 100_000;

  private final String run;
  private final Source source;
  private final EventGolden.Format format;
  private final String language;
  /** {@code null} for a text format. */
  private final ConlluTagset tagset;
  private final Path tagDictionaryFile;
  /** {@code null} for no tag dictionary. */
  private final POSDictionary tagDictionary;
  private final Set<String> inputSets;
  private final boolean dictionaryCases;

  private TaggerEventGolden(String run, Source source, EventGolden.Format format, String language,
      ConlluTagset tagset, Path tagDictionaryFile, Set<String> inputSets, boolean dictionaryCases)
      throws IOException {
    this.run = run;
    this.source = source;
    this.format = format;
    this.language = language;
    this.tagset = tagset;
    this.tagDictionaryFile = tagDictionaryFile;
    this.tagDictionary = tagDictionaryFile == null ? null : load(tagDictionaryFile);
    this.inputSets = Set.copyOf(inputSets);
    this.dictionaryCases = dictionaryCases;
  }

  static TaggerEventGolden tags(String run, EventGolden.Format format, String language, ConlluTagset tagset,
      Path tagDictionary, Set<String> inputSets, boolean dictionaryCases) throws IOException {
    return new TaggerEventGolden(run, Source.POSTAG, format, language, tagset, tagDictionary, inputSets,
        dictionaryCases);
  }

  static TaggerEventGolden lemmas(String run, EventGolden.Format format, String language, ConlluTagset tagset,
      Set<String> inputSets) throws IOException {
    return new TaggerEventGolden(run, Source.LEMMATIZE, format, language, tagset, null, inputSets, false);
  }

  /** {@code POSDictionary.create(InputStream)} of the fixture file. */
  private static POSDictionary load(Path file) throws IOException {
    try (InputStream in = Files.newInputStream(file)) {
      return POSDictionary.create(in);
    }
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
    return source.streamClass;
  }

  @Override
  public String method() {
    return "read";
  }

  @Override
  public Map<String, String> config() {
    Map<String, String> c = new TreeMap<>();
    c.put("run", run);
    c.put("source", source.name);
    c.put("language", language);
    c.put("format", format.name);
    if (tagset != null) {
      c.put("tagset", tagset.name().toLowerCase(java.util.Locale.ROOT));
    }
    if (tagDictionaryFile != null) {
      c.put("tagDictionary", tagDictionaryFile.getFileName().toString());
    }
    return c;
  }

  @Override
  public Set<String> inputSets() {
    return inputSets;
  }

  @Override
  public boolean accepts(Input input) {
    if (!inputSets.contains(input.set())) {
      return false;
    }
    if (format == EventGolden.Format.CONLLU) {
      return input.id().startsWith("conllu-");
    }
    return input.id().startsWith(source.inputPrefix);
  }

  /** The factory {@code POSTaggerTrainerTool} builds: no descriptor, no resources, the dictionary. */
  private POSTaggerFactory posFactory() {
    return new POSTaggerFactory(null, Map.of(), tagDictionary);
  }

  /** The event stream over the input. */
  private ObjectStream<Event> open(Input input) throws IOException {
    InputStreamFactory in = () -> new ByteArrayInputStream(input.bytes());
    switch (source) {
      case POSTAG: {
        ObjectStream<POSSample> samples = format == EventGolden.Format.CONLLU
            ? new ConlluPOSSampleStream(new ConlluStream(in), tagset)
            : new WordTagSampleStream(new PlainTextByLineStream(in, StandardCharsets.UTF_8));
        return new POSSampleEventStream(samples, posFactory().getPOSContextGenerator());
      }
      case LEMMATIZE: {
        ObjectStream<LemmaSample> samples = format == EventGolden.Format.CONLLU
            ? new ConlluLemmaSampleStream(new ConlluStream(in), tagset)
            : new LemmaSampleStream(new PlainTextByLineStream(in, StandardCharsets.UTF_8));
        return new LemmaSampleEventStream(samples, new LemmatizerFactory().getContextGenerator());
      }
      default:
        throw new IllegalStateException(source.name());
    }
  }

  @Override
  public List<Layer> layers(Input input) throws AnnotatorException {
    List<Event> events = new ArrayList<>();
    FormatGolden.Columns errors = new FormatGolden.Columns();
    try (ObjectStream<Event> stream = open(input)) {
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
      layers.addAll(packageLayers());
      if (dictionaryCases && input.set().equals(Runs.TAGGED)) {
        layers.add(dictionaryCases());
      }
    } catch (IOException e) {
      throw new IllegalStateException("the model file failed", e);
    }
    return layers;
  }

  /**
   * The events as {@link EventGolden} writes them, the predicates joined with a space. A predicate
   * of these sources holds a token, which the sample streams never let hold a space; the generator
   * stops if one does, so a golden is never ambiguous. The outcome is written as {@code lossless}
   * too: a lemma class is an edit script over UTF-16 units, so a token or lemma with a character
   * outside the Basic Multilingual Plane can give a class holding half of a surrogate pair.
   */
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

  /**
   * A maxent model of one predicate, enough for a model file to hold. Its outcomes are the tag
   * dictionary's tags, sorted, since {@code POSModel} refuses a dictionary with a tag the model
   * has no outcome for; without a dictionary, two universal tags.
   */
  private MaxentModel placeholderModel() {
    TreeSet<String> outcomes = new TreeSet<>();
    if (tagDictionary != null) {
      for (String word : tagDictionary) {
        outcomes.addAll(List.of(tagDictionary.getTags(word)));
      }
    }
    if (outcomes.size() < 2) {
      outcomes.addAll(List.of("NOUN", "VERB"));
    }
    Context[] params = {new Context(new int[] {0, 1}, new double[] {0.25, -0.25})};
    return new GISModel(params, new String[] {"p"}, outcomes.toArray(new String[0]));
  }

  /** The manifest, the zip entries and the tag dictionary entry of a model file built from the factory. */
  private List<Layer> packageLayers() throws IOException {
    Map<String, String> info = new java.util.HashMap<>();
    info.put("Training-Eventhash", "1");
    info.put("TrainerType", "Event");
    BaseModel model = switch (source) {
      case POSTAG -> new POSModel(language, placeholderModel(), info, posFactory());
      case LEMMATIZE -> new LemmatizerModel(language, placeholderModel(), info, new LemmatizerFactory());
    };
    ByteArrayOutputStream bytes = new ByteArrayOutputStream();
    model.serialize(bytes);

    List<String> names = new ArrayList<>();
    List<Long> lengths = new ArrayList<>();
    String manifest = null;
    String dictionaryXml = null;
    try (ZipInputStream zip = new ZipInputStream(new ByteArrayInputStream(bytes.toByteArray()))) {
      ZipEntry entry;
      while ((entry = zip.getNextEntry()) != null) {
        byte[] content = zip.readAllBytes();
        names.add(entry.getName());
        lengths.add((long) content.length);
        if (entry.getName().equals("manifest.properties")) {
          manifest = new String(content, StandardCharsets.UTF_8);
        } else if (entry.getName().equals("tags.tagdict")) {
          dictionaryXml = new String(content, StandardCharsets.UTF_8);
        }
      }
    }
    if (manifest == null) {
      throw new IllegalStateException("the model file has no manifest");
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
    if (dictionaryXml != null) {
      layers.add(new Layer("turbo:tagdict", "document",
          new Column.Str("xml", new String[] {TrainGolden.lossless(dictionaryXml)})));
    }
    return layers;
  }

  /** {@code POSDictionary.serialize} of the dictionaries the class comment lists. */
  private Layer dictionaryCases() throws IOException {
    List<String> names = new ArrayList<>();
    List<String> xmls = new ArrayList<>();
    names.add("fixture");
    xmls.add(serialize(tagDictionary));

    POSDictionary caseSensitive = new POSDictionary(true);
    caseSensitive.put("Run", "NN", "VB");
    caseSensitive.put("run", "VBP");
    caseSensitive.put("RUN", "NNP");
    names.add("case-sensitive");
    xmls.add(serialize(caseSensitive));

    POSDictionary collapsed = new POSDictionary(false);
    collapsed.put("Run", "NN", "VB");
    collapsed.put("run", "VBP");
    collapsed.put("RUN", "NNP");
    collapsed.put("Paris", "NNP");
    names.add("collapsed");
    xmls.add(serialize(collapsed));

    POSDictionary special = new POSDictionary(true);
    special.put("a&b<c>d\"e'f", "X");
    special.put("tab\there", "Y");
    special.put("cr\rhere", "Z");
    special.put("😀.", "SYM");
    special.put("q", "A\"B", "C<D&E");
    special.put("none");
    names.add("special-characters");
    xmls.add(serialize(special));

    names.add("empty");
    xmls.add(serialize(new POSDictionary(false)));

    POSDictionary forty = new POSDictionary(false);
    for (int i = 1; i <= 40; i++) {
      forty.put("w" + i, "T");
    }
    names.add("forty");
    xmls.add(serialize(forty));

    POSDictionary unicode = new POSDictionary(false);
    unicode.put("İstanbul", "NNP");
    unicode.put("ΣΟΦΟΣ", "NNP");
    unicode.put("Straße", "NN");
    unicode.put("ǅemal", "NNP");
    names.add("unicode-case");
    xmls.add(serialize(unicode));

    return new Layer("turbo:tagdict-cases", "document",
        new Column.Str("name", names.toArray(new String[0])),
        new Column.Str("xml", xmls.toArray(new String[0])));
  }

  private static String serialize(POSDictionary d) throws IOException {
    ByteArrayOutputStream out = new ByteArrayOutputStream();
    d.serialize(out);
    return TrainGolden.lossless(out.toString(StandardCharsets.UTF_8));
  }

  /** Every run, with the tag dictionary fixture. */
  static List<TaggerEventGolden> all(Path tagDictionary) throws IOException {
    Set<String> texts = Set.of(Runs.TAGGED, Runs.FORMATS);
    Set<String> formats = Set.of(Runs.FORMATS);
    List<TaggerEventGolden> runs = new ArrayList<>();
    // The tagger over word and tag lines, as the fixture models were trained: without and with
    // the tag dictionary; the dictionary serialization cases ride on the dictionary run.
    runs.add(tags("postag-wordtag", EventGolden.Format.TEXT, "en", null, null, texts, false));
    runs.add(tags("postag-wordtag-tagdict", EventGolden.Format.TEXT, "en", null, tagDictionary, texts, true));
    // The tagger over CoNLL-U in both tag sets, as the published models were trained (universal).
    runs.add(tags("postag-conllu-u", EventGolden.Format.CONLLU, "es", ConlluTagset.U, null, formats, false));
    runs.add(tags("postag-conllu-x", EventGolden.Format.CONLLU, "es", ConlluTagset.X, null, formats, false));
    // The lemmatizer over word, tag and lemma lines and over CoNLL-U.
    runs.add(lemmas("lemma-text", EventGolden.Format.TEXT, "en", null, texts));
    runs.add(lemmas("lemma-conllu-u", EventGolden.Format.CONLLU, "es", ConlluTagset.U, formats));
    return runs;
  }
}
