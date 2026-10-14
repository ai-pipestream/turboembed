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
import java.util.Locale;
import java.util.Map;
import java.util.Properties;
import java.util.Set;
import java.util.TreeMap;
import java.util.zip.ZipEntry;
import java.util.zip.ZipInputStream;
import opennlp.tools.dictionary.Dictionary;
import opennlp.tools.formats.conllu.ConlluSentenceSampleStream;
import opennlp.tools.formats.conllu.ConlluStream;
import opennlp.tools.formats.conllu.ConlluTokenSampleStream;
import opennlp.tools.ml.maxent.GISModel;
import opennlp.tools.ml.model.ChecksumEventStream;
import opennlp.tools.ml.model.Context;
import opennlp.tools.ml.model.Event;
import opennlp.tools.ml.model.MaxentModel;
import opennlp.tools.sentdetect.SDEventStream;
import opennlp.tools.sentdetect.SentenceDetectorFactory;
import opennlp.tools.sentdetect.SentenceModel;
import opennlp.tools.sentdetect.SentenceSample;
import opennlp.tools.sentdetect.SentenceSampleStream;
import opennlp.tools.tokenize.TokSpanEventStream;
import opennlp.tools.tokenize.TokenSample;
import opennlp.tools.tokenize.TokenSampleStream;
import opennlp.tools.tokenize.TokenizerFactory;
import opennlp.tools.tokenize.TokenizerModel;
import opennlp.tools.util.InputStreamFactory;
import opennlp.tools.util.ObjectStream;
import opennlp.tools.util.ObjectStreamUtils;
import opennlp.tools.util.PlainTextByLineStream;
import opennlp.tools.util.StringList;
import opennlp.tools.util.model.BaseModel;

/**
 * The events of {@link SDEventStream} and {@link TokSpanEventStream} with the factory options
 * {@link TrainGolden} leaves at their defaults, and what the model file around a trained model
 * holds: the golden directory {@code train-events}.
 *
 * <p>Each run is a factory and a sample format. The factories are built as the trainer tools build
 * them, from a language code, a dictionary file or none, and explicit or absent end characters or
 * alphanumeric pattern. The samples are read from the input as one sentence or token sample per
 * line ({@code text}), or from CoNLL-U with {@link ConlluSentenceSampleStream} (ten sentences a
 * sample) and {@link ConlluTokenSampleStream} ({@code conllu}). The {@code config} records name
 * the run and every option: {@code run}, {@code source}, {@code language}, {@code format},
 * {@code abbreviations} (the fixture file), and for the sentence detector {@code useTokenEnd} and
 * {@code eosCharacters} when given, for the tokenizer {@code useAlphaNumericOptimization} and
 * {@code alphaNumericPattern} when given, and {@code sentencesPerSample} for CoNLL-U sentences.</p>
 *
 * <p>Layers:</p>
 * <ul>
 *   <li>{@code turbo:events}: the outcome and the contexts of every event, written as
 *   {@link TrainGolden#lossless}, in stream order;</li>
 *   <li>{@code turbo:stream-errors}: when the sample stream throws, one row per exception with
 *   the number of events made before it, its class and its message; reading goes on with the
 *   next sample, as the format goldens do;</li>
 *   <li>{@code turbo:event-checksum}: the CRC-32C of {@code ChecksumEventStream} over the
 *   events;</li>
 *   <li>{@code turbo:manifest}: the manifest entries of a model file built from the factory,
 *   sorted by key, without {@code Timestamp} (the file is {@code new SentenceModel} or
 *   {@code new TokenizerModel} around a one-predicate maxent model, serialized);</li>
 *   <li>{@code turbo:package-entries}: that file's zip entries in order, with their lengths;</li>
 *   <li>{@code turbo:dictionary}: with a dictionary, its {@code abbreviations.dictionary} entry
 *   as text, written as {@code lossless};</li>
 *   <li>{@code turbo:dictionary-cases}: on the {@code train} input of the dictionary run only,
 *   {@code Dictionary.serialize} of dictionaries built in the generator: the English fixture,
 *   tokens with the characters the XML serializer escapes (a control character other than tab,
 *   line feed and carriage return makes the serializer throw, so none is among them), an empty
 *   dictionary, a case-sensitive
 *   one, forty entries (the hash set's table doubles twice), entries of several tokens, and
 *   entries that differ only in case.</li>
 * </ul>
 */
final class EventGolden implements AnnotatorGolden {

  enum Source {
    SENTDETECT("sentdetect", "opennlp.tools.sentdetect.SDEventStream"),
    TOKENIZE("tokenize", "opennlp.tools.tokenize.TokSpanEventStream");

    final String name;
    final String streamClass;

    Source(String name, String streamClass) {
      this.name = name;
      this.streamClass = streamClass;
    }
  }

  enum Format {
    TEXT("text"),
    CONLLU("conllu");

    final String name;

    Format(String name) {
      this.name = name;
    }
  }

  static final String ANNOTATOR = "train-events";
  static final int SENTENCES_PER_SAMPLE = 10;
  /** The reads a stream may take before the generator gives it up as not ending. */
  private static final int MAX_READS = 100_000;

  private final String run;
  private final Source source;
  private final Format format;
  private final String language;
  private final Path dictionaryFile;
  /** {@code null} for no dictionary. */
  private final Dictionary dictionary;
  private final Boolean useTokenEnd;
  private final String eosCharacters;
  private final Boolean useAlphaNumericOptimization;
  private final String alphaNumericPattern;
  private final Set<String> inputSets;
  private final boolean dictionaryCases;

  private EventGolden(String run, Source source, Format format, String language, Path dictionaryFile,
      Boolean useTokenEnd, String eosCharacters, Boolean useAlphaNumericOptimization,
      String alphaNumericPattern, Set<String> inputSets, boolean dictionaryCases) throws IOException {
    this.run = run;
    this.source = source;
    this.format = format;
    this.language = language;
    this.dictionaryFile = dictionaryFile;
    this.dictionary = dictionaryFile == null ? null : load(dictionaryFile);
    this.useTokenEnd = useTokenEnd;
    this.eosCharacters = eosCharacters;
    this.useAlphaNumericOptimization = useAlphaNumericOptimization;
    this.alphaNumericPattern = alphaNumericPattern;
    this.inputSets = Set.copyOf(inputSets);
    this.dictionaryCases = dictionaryCases;
  }

  static EventGolden sentences(String run, Format format, String language, Path dictionary,
      boolean useTokenEnd, String eosCharacters, Set<String> inputSets, boolean dictionaryCases)
      throws IOException {
    return new EventGolden(run, Source.SENTDETECT, format, language, dictionary, useTokenEnd, eosCharacters,
        null, null, inputSets, dictionaryCases);
  }

  static EventGolden tokens(String run, Format format, String language, Path dictionary,
      boolean useAlphaNumericOptimization, String alphaNumericPattern, Set<String> inputSets) throws IOException {
    return new EventGolden(run, Source.TOKENIZE, format, language, dictionary, null, null,
        useAlphaNumericOptimization, alphaNumericPattern, inputSets, false);
  }

  /** {@code new Dictionary(InputStream)} of the fixture file. */
  private static Dictionary load(Path file) throws IOException {
    try (InputStream in = Files.newInputStream(file)) {
      return new Dictionary(in);
    }
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
    if (dictionaryFile != null) {
      c.put("abbreviations", dictionaryFile.getFileName().toString());
    }
    if (useTokenEnd != null) {
      c.put("useTokenEnd", Boolean.toString(useTokenEnd));
    }
    if (eosCharacters != null) {
      c.put("eosCharacters", TrainGolden.lossless(eosCharacters));
    }
    if (useAlphaNumericOptimization != null) {
      c.put("useAlphaNumericOptimization", Boolean.toString(useAlphaNumericOptimization));
    }
    if (alphaNumericPattern != null) {
      c.put("alphaNumericPattern", alphaNumericPattern);
    }
    if (source == Source.SENTDETECT && format == Format.CONLLU) {
      c.put("sentencesPerSample", Integer.toString(SENTENCES_PER_SAMPLE));
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
    return format == Format.TEXT || input.id().startsWith("conllu-");
  }

  private SentenceDetectorFactory sentenceFactory() {
    char[] eos = eosCharacters == null ? null : eosCharacters.toCharArray();
    return new SentenceDetectorFactory(language, useTokenEnd, dictionary, eos);
  }

  private TokenizerFactory tokenizerFactory() {
    java.util.regex.Pattern pattern = alphaNumericPattern == null ? null
        : java.util.regex.Pattern.compile(alphaNumericPattern);
    return new TokenizerFactory(language, dictionary, useAlphaNumericOptimization, pattern);
  }

  /** The event stream over the input. */
  private ObjectStream<Event> open(Input input) throws IOException {
    InputStreamFactory in = () -> new ByteArrayInputStream(input.bytes());
    switch (source) {
      case SENTDETECT: {
        ObjectStream<SentenceSample> samples = format == Format.CONLLU
            ? new ConlluSentenceSampleStream(new ConlluStream(in), SENTENCES_PER_SAMPLE)
            : new SentenceSampleStream(new PlainTextByLineStream(in, StandardCharsets.UTF_8));
        SentenceDetectorFactory f = sentenceFactory();
        return new SDEventStream(samples, f.getSDContextGenerator(), f.getEndOfSentenceScanner());
      }
      case TOKENIZE: {
        ObjectStream<TokenSample> samples = format == Format.CONLLU
            ? new ConlluTokenSampleStream(new ConlluStream(in))
            : new TokenSampleStream(new PlainTextByLineStream(in, StandardCharsets.UTF_8));
        TokenizerFactory f = tokenizerFactory();
        return new TokSpanEventStream(samples, f.isUseAlphaNumericOptimization(), f.getAlphaNumericPattern(),
            f.getContextGenerator());
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
    layers.add(eventsLayer(events));
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
      if (dictionaryCases && input.set().equals(Runs.TRAIN)) {
        layers.add(dictionaryCases());
      }
    } catch (IOException e) {
      throw new IllegalStateException("the model file failed", e);
    }
    return layers;
  }

  private static Layer eventsLayer(List<Event> events) {
    String[] outcomes = new String[events.size()];
    String[] contexts = new String[events.size()];
    String[] values = new String[events.size()];
    java.util.Arrays.fill(values, "");
    for (int i = 0; i < outcomes.length; i++) {
      Event e = events.get(i);
      outcomes[i] = e.getOutcome();
      StringBuilder b = new StringBuilder();
      for (String c : e.getContext()) {
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
   * A maxent model of one predicate and the component's two outcomes, enough for a model file to
   * hold: the model constructors check the outcomes are the component's.
   */
  private MaxentModel placeholderModel() {
    Context[] params = {new Context(new int[] {0, 1}, new double[] {0.25, -0.25})};
    String[] outcomes = source == Source.SENTDETECT ? new String[] {"n", "s"} : new String[] {"F", "T"};
    return new GISModel(params, new String[] {"p"}, outcomes);
  }

  /** The manifest, the zip entries and the dictionary entry of a model file built from the factory. */
  private List<Layer> packageLayers() throws IOException {
    Map<String, String> info = new java.util.HashMap<>();
    info.put("Training-Eventhash", "1");
    info.put("TrainerType", "Event");
    BaseModel model = switch (source) {
      case SENTDETECT -> new SentenceModel(language, placeholderModel(), info, sentenceFactory());
      case TOKENIZE -> new TokenizerModel(placeholderModel(), info, tokenizerFactory());
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
        } else if (entry.getName().equals("abbreviations.dictionary")) {
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
      layers.add(new Layer("turbo:dictionary", "document",
          new Column.Str("xml", new String[] {TrainGolden.lossless(dictionaryXml)})));
    }
    return layers;
  }

  /** {@code Dictionary.serialize} of the dictionaries the class comment lists. */
  private Layer dictionaryCases() throws IOException {
    List<String> names = new ArrayList<>();
    List<String> xmls = new ArrayList<>();
    names.add("english");
    xmls.add(serialize(dictionary));

    Dictionary special = new Dictionary(false);
    special.put(new StringList("a&b<c>d\"e'f"));
    special.put(new StringList("cr\rhere"));
    special.put(new StringList("tab\there"));
    special.put(new StringList("😀."));
    names.add("special-characters");
    xmls.add(serialize(special));

    names.add("empty");
    xmls.add(serialize(new Dictionary(false)));

    Dictionary caseSensitive = new Dictionary(true);
    caseSensitive.put(new StringList("Dr."));
    caseSensitive.put(new StringList("MR."));
    caseSensitive.put(new StringList("u.s."));
    names.add("case-sensitive");
    xmls.add(serialize(caseSensitive));

    Dictionary forty = new Dictionary(false);
    for (int i = 1; i <= 40; i++) {
      forty.put(new StringList("w" + i + "."));
    }
    names.add("forty");
    xmls.add(serialize(forty));

    Dictionary multi = new Dictionary(false);
    multi.put(new StringList("et", "al."));
    multi.put(new StringList("e.", "g."));
    multi.put(new StringList("a", "b", "c"));
    multi.put(new StringList("Dr."));
    names.add("multi-token");
    xmls.add(serialize(multi));

    Dictionary collapsed = new Dictionary(false);
    collapsed.put(new StringList("Dr."));
    collapsed.put(new StringList("DR."));
    collapsed.put(new StringList("dr."));
    collapsed.put(new StringList("Mr."));
    names.add("collapsed");
    xmls.add(serialize(collapsed));

    return new Layer("turbo:dictionary-cases", "document",
        new Column.Str("name", names.toArray(new String[0])),
        new Column.Str("xml", xmls.toArray(new String[0])));
  }

  private static String serialize(Dictionary d) throws IOException {
    ByteArrayOutputStream out = new ByteArrayOutputStream();
    d.serialize(out);
    return TrainGolden.lossless(out.toString(StandardCharsets.UTF_8));
  }

  /** Every run, with the English abbreviation dictionary fixture. */
  static List<EventGolden> all(Path englishDictionary) throws IOException {
    Set<String> textsAndTrain = Set.of(Runs.EN, Runs.MULTI, Runs.EDGE, Runs.TRAIN);
    Set<String> formats = Set.of(Runs.FORMATS);
    List<EventGolden> runs = new ArrayList<>();
    // The sentence detector with a dictionary and the Portuguese end characters, which the
    // language supplies; the dictionary serialization cases ride on its train input.
    runs.add(sentences("sentdetect-dictionary-pt", Format.TEXT, "pt", englishDictionary, true, null,
        textsAndTrain, true));
    // Thai: the language supplies a space and a line feed as end characters.
    runs.add(sentences("sentdetect-th", Format.TEXT, "th", null, false, null, Set.of(Runs.MULTI, Runs.EDGE),
        false));
    // CoNLL-U sentences, ten a sample, with explicit end characters.
    runs.add(sentences("sentdetect-conllu", Format.CONLLU, "en", null, true, ".!?", formats, false));
    // The tokenizer with a dictionary, the alphanumeric optimization and the German pattern.
    runs.add(tokens("tokenize-dictionary-alnum-de", Format.TEXT, "de", englishDictionary, true, null,
        Set.of(Runs.EN, Runs.MULTI, Runs.EDGE, Runs.ABBREVIATIONS, Runs.ALPHANUMERIC)));
    // CoNLL-U tokens with an explicit pattern.
    runs.add(tokens("tokenize-conllu", Format.CONLLU, "en", null, false, "^[A-Za-z0-9]+$", formats));
    return runs;
  }
}
