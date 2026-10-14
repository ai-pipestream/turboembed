package ai.pipestream.turbo.goldens;

import java.io.ByteArrayInputStream;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;
import opennlp.tools.formats.Conll02NameSampleStream;
import opennlp.tools.formats.Conll03NameSampleStream;
import opennlp.tools.formats.ConllXPOSSampleStream;
import opennlp.tools.formats.convert.POSToSentenceSampleStream;
import opennlp.tools.formats.convert.POSToTokenSampleStream;
import opennlp.tools.tokenize.DetokenizationDictionary;
import opennlp.tools.tokenize.Detokenizer;
import opennlp.tools.tokenize.DictionaryDetokenizer;
import opennlp.tools.util.InputStreamFactory;
import opennlp.tools.util.ObjectStream;
import opennlp.tools.util.PlainTextByLineStream;

/**
 * The golden of one of OpenNLP's CoNLL streams over a file in its format: the CoNLL-2002 and
 * CoNLL-2003 name streams in each language, also with some name types left out, and the CoNLL-X
 * part-of-speech stream with the token and sentence streams its factories make from it. Its
 * inputs are the {@code formats} set's files whose id starts with {@code conll02-},
 * {@code conll03-} or {@code conllx-}; each language's run reads the files of the other language
 * too. Every stream reads lines of UTF-8 text ({@code PlainTextByLineStream}), the encoding the
 * factories use for CoNLL-2003 and CoNLL-X; for CoNLL-2002 they use ISO-8859-1, which is a matter
 * of decoding the file.
 *
 * <p>The stream is read as {@link FormatGolden} reads one, and the golden has the same layers:
 * {@code turbo:samples} and {@code turbo:read-errors}, and {@code turbo:sample-tokens} with
 * {@code turbo:names} or {@code turbo:sample-tags}, or {@code turbo:sample-texts} and
 * {@code turbo:sample-spans}. CoNLL-2000 has no run of its own: it is the chunker's native format,
 * which {@link FormatGolden} reads.</p>
 */
final class ConllGolden implements AnnotatorGolden {

  /** The input set the formats are read from. */
  static final String SET = Runs.FORMATS;

  /** The chunk size of the sentence stream: small, so that the inputs show what a full chunk does. */
  static final int CHUNK_SIZE = 2;

  /** The name types of the runs that keep them all, as the factories' {@code types} parameter. */
  static final String ALL_TYPES = "per,org,loc,misc";

  /** The stopping point for a stream that throws without moving on, which would otherwise loop. */
  private static final int MAX_READS = 100_000;

  /** The streams, each with the prefix of the input ids it reads, its language and name types. */
  enum Kind {
    CONLL02_NLD("conll02-nld", "conll02", Conll02NameSampleStream.class, "nld", ALL_TYPES),
    CONLL02_SPA("conll02-spa", "conll02", Conll02NameSampleStream.class, "spa", ALL_TYPES),
    CONLL02_SPA_PER_LOC("conll02-spa-per-loc", "conll02", Conll02NameSampleStream.class, "spa", "per,loc"),
    CONLL03_ENG("conll03-eng", "conll03", Conll03NameSampleStream.class, "eng", ALL_TYPES),
    CONLL03_DEU("conll03-deu", "conll03", Conll03NameSampleStream.class, "deu", ALL_TYPES),
    CONLL03_ENG_ORG_MISC("conll03-eng-org-misc", "conll03", Conll03NameSampleStream.class, "eng", "org,misc"),
    CONLLX_POS("conllx-pos", "conllx", ConllXPOSSampleStream.class, null, null),
    CONLLX_TOKEN("conllx-token", "conllx", POSToTokenSampleStream.class, null, null),
    CONLLX_SENTENCE("conllx-sentence", "conllx", POSToSentenceSampleStream.class, null, null);

    final String dir;
    final String inputPrefix;
    final Class<?> streamClass;
    final String lang;
    final String types;

    Kind(String dir, String inputPrefix, Class<?> streamClass, String lang, String types) {
      this.dir = dir;
      this.inputPrefix = inputPrefix;
      this.streamClass = streamClass;
      this.lang = lang;
      this.types = types;
    }

    boolean detokenizes() {
      return this == CONLLX_TOKEN || this == CONLLX_SENTENCE;
    }
  }

  private final Kind kind;
  private final List<ModelRef> models;
  private final Detokenizer detokenizer;

  private ConllGolden(Kind kind, List<ModelRef> models, Detokenizer detokenizer) {
    this.kind = kind;
    this.models = List.copyOf(models);
    this.detokenizer = detokenizer;
  }

  /** Every run; the CoNLL-X token and sentence streams with the detokenizer dictionary {@code dictionary}. */
  static List<ConllGolden> all(Path dictionary) throws IOException {
    ModelFile d = ModelFile.read(dictionary);
    Detokenizer detokenizer = new DictionaryDetokenizer(new DetokenizationDictionary(d.stream()));
    List<ConllGolden> runs = new ArrayList<>();
    for (Kind k : Kind.values()) {
      runs.add(k.detokenizes() ? new ConllGolden(k, List.of(d.ref()), detokenizer) : new ConllGolden(k, List.of(), null));
    }
    return runs;
  }

  @Override
  public String annotator() {
    return "formats";
  }

  @Override
  public List<ModelRef> models() {
    return models;
  }

  @Override
  public String annotatorClass() {
    return kind.streamClass.getName();
  }

  @Override
  public String method() {
    return "read";
  }

  @Override
  public Map<String, String> config() {
    Map<String, String> c = new LinkedHashMap<>();
    if (kind.lang != null) {
      c.put("lang", kind.lang);
      c.put("types", kind.types);
    }
    if (kind == Kind.CONLLX_SENTENCE) {
      c.put("chunkSize", Integer.toString(CHUNK_SIZE));
    }
    return c;
  }

  @Override
  public Set<String> inputSets() {
    return Set.of(SET);
  }

  @Override
  public boolean accepts(Input input) {
    return input.set().equals(SET) && input.id().startsWith(kind.inputPrefix + "-");
  }

  @Override
  public String modelDir() {
    return kind.dir;
  }

  /** The name types of the run as the factories make them from their {@code types} parameter. */
  private int types() {
    int t = 0;
    if (kind.types.contains("per")) {
      t |= Conll02NameSampleStream.GENERATE_PERSON_ENTITIES;
    }
    if (kind.types.contains("org")) {
      t |= Conll02NameSampleStream.GENERATE_ORGANIZATION_ENTITIES;
    }
    if (kind.types.contains("loc")) {
      t |= Conll02NameSampleStream.GENERATE_LOCATION_ENTITIES;
    }
    if (kind.types.contains("misc")) {
      t |= Conll02NameSampleStream.GENERATE_MISC_ENTITIES;
    }
    return t;
  }

  private ObjectStream<?> open(Input input) throws IOException {
    InputStreamFactory in = () -> new ByteArrayInputStream(input.bytes());
    ObjectStream<String> lines = new PlainTextByLineStream(in, StandardCharsets.UTF_8);
    return switch (kind) {
      case CONLL02_NLD, CONLL02_SPA, CONLL02_SPA_PER_LOC -> new Conll02NameSampleStream(
          kind.lang.equals("nld") ? Conll02NameSampleStream.LANGUAGE.NLD : Conll02NameSampleStream.LANGUAGE.SPA,
          lines, types());
      case CONLL03_ENG, CONLL03_DEU, CONLL03_ENG_ORG_MISC -> new Conll03NameSampleStream(
          kind.lang.equals("eng") ? Conll03NameSampleStream.LANGUAGE.EN : Conll03NameSampleStream.LANGUAGE.DE,
          lines, types());
      case CONLLX_POS -> new ConllXPOSSampleStream(lines);
      case CONLLX_TOKEN -> new POSToTokenSampleStream(detokenizer, new ConllXPOSSampleStream(lines));
      case CONLLX_SENTENCE -> new POSToSentenceSampleStream(detokenizer, new ConllXPOSSampleStream(lines), CHUNK_SIZE);
    };
  }

  @Override
  public List<Layer> layers(Input input) {
    List<Object> samples = new ArrayList<>();
    FormatGolden.Columns errors = new FormatGolden.Columns();
    try (ObjectStream<?> stream = open(input)) {
      for (int reads = 0; ; reads++) {
        if (reads > MAX_READS) {
          throw new IllegalStateException(input.inputId() + ": the stream does not end");
        }
        Object sample;
        try {
          sample = stream.read();
        } catch (IOException | RuntimeException e) {
          errors.u32("after", samples.size()).str("class", e.getClass().getName())
              .str("message", FormatGolden.message(e));
          continue;
        }
        if (sample == null) {
          break;
        }
        samples.add(sample);
      }
    } catch (IOException e) {
      throw new IllegalStateException(input.inputId() + ": the line stream failed", e);
    }

    List<Layer> layers = new ArrayList<>();
    FormatGolden.Columns written = new FormatGolden.Columns();
    for (Object s : samples) {
      written.str("written", s.toString());
    }
    layers.add(written.layer("turbo:samples", "document", "written"));
    layers.add(errors.layer("turbo:read-errors", "document", "after", "class", "message"));
    switch (kind) {
      case CONLL02_NLD, CONLL02_SPA, CONLL02_SPA_PER_LOC, CONLL03_ENG, CONLL03_DEU, CONLL03_ENG_ORG_MISC ->
          FormatGolden.names(samples, layers);
      case CONLLX_POS -> FormatGolden.tagged(samples, layers);
      case CONLLX_TOKEN -> FormatGolden.tokenSamples(input, samples, layers);
      case CONLLX_SENTENCE -> FormatGolden.sentenceSamples(input, samples, layers);
    }
    return layers;
  }
}
