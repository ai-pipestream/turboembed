package ai.pipestream.turbo.goldens;

import java.io.ByteArrayInputStream;
import java.io.File;
import java.io.IOException;
import java.lang.reflect.Field;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Comparator;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.TreeMap;
import java.util.stream.Collectors;
import opennlp.tools.formats.BioNLP2004NameSampleStream;
import opennlp.tools.formats.EvalitaNameSampleStream;
import opennlp.tools.formats.GermEval2014NameSampleStream;
import opennlp.tools.formats.DirectorySampleStream;
import opennlp.tools.formats.NameFinderCensus90NameStream;
import opennlp.tools.formats.brat.AnnotationConfiguration;
import opennlp.tools.formats.brat.BratDocument;
import opennlp.tools.formats.brat.BratDocumentStream;
import opennlp.tools.formats.brat.BratNameSampleStream;
import opennlp.tools.formats.convert.FileToStringSampleStream;
import opennlp.tools.formats.muc.MucNameContentHandler;
import opennlp.tools.formats.muc.MucNameSampleStream;
import opennlp.tools.sentdetect.NewlineSentenceDetector;
import opennlp.tools.tokenize.SimpleTokenizer;
import opennlp.tools.tokenize.Tokenizer;
import opennlp.tools.tokenize.WhitespaceTokenizer;
import opennlp.tools.util.InputStreamFactory;
import opennlp.tools.util.ObjectStream;
import opennlp.tools.util.PlainTextByLineStream;
import opennlp.tools.util.StringList;
import opennlp.tools.util.StringUtil;
import opennlp.tools.util.WhitespaceMode;

/**
 * The golden of one of OpenNLP's named entity corpus streams over a file of the {@code ner} input
 * set: {@code GermEval2014NameSampleStream} (each layer, and one run with some name types left
 * out), {@code BioNLP2004NameSampleStream} and {@code EvalitaNameSampleStream} (every name type,
 * and runs with some left out), and {@code NameFinderCensus90NameStream}. Each run reads the
 * inputs whose id starts with its corpus's prefix ({@code germeval-}, {@code bionlp-},
 * {@code evalita-}, {@code census-}), as UTF-8, the encoding the factories use (and the census
 * dictionary tool by default). The name types are made from the factories' {@code types}
 * parameter, which the {@code types} config record gives, as the factories make them.
 *
 * <p>The MUC runs read the inputs whose id starts with {@code muc-}, each a bundle of a corpus
 * directory (the format is described in {@link MascGolden}), written to a temporary directory and
 * read as {@code Muc6NameSampleStreamFactory} reads one: a {@code DirectorySampleStream} of the
 * files whose lower-cased name ends with {@code .sgm}, not recursive, each read as UTF-8 by a
 * {@code FileToStringSampleStream}, then {@code MucNameSampleStream}. The factory splits the text
 * with a maxent tokenizer model; the runs use {@code WhitespaceTokenizer} (also in the legacy
 * whitespace mode) and {@code SimpleTokenizer} ({@code tokenizer} config record). The message of
 * an unknown name type ends with the set of the expected types, which {@code Set.of} iterates in
 * an order that changes from one JVM run to the next; the golden writes that set in the order
 * {@code MucNameContentHandler} lists the types in its source, so it is the same in every run.</p>
 *
 * <p>The brat runs read the inputs whose id starts with {@code brat-}, bundles too, as
 * {@code BratNameSampleStreamFactory} reads a corpus directory: its {@code annotation.conf}, a
 * {@code BratDocumentStream} (recursive in one run), a {@code NewlineSentenceDetector} (the
 * factory's choice without a sentence model) and the whitespace tokenizer (its choice without a
 * tokenizer, also in the legacy whitespace mode) or the simple one, and in one run the name types
 * {@code Person , Location} split and trimmed as the factory splits its {@code nameTypes}. The
 * simple tokenizer cuts a character outside the Basic Multilingual Plane into its two surrogates,
 * which a golden cannot hold, so its run does not read {@code brat-utf16}. The
 * messages name a document by its absolute path, which the golden writes with the corpus
 * directory as {@code <corpus>}. The {@code brat-config} run parses each {@code .conf} file of a
 * bundle's top directory with {@code AnnotationConfiguration.parse}, in name order, and writes
 * {@code turbo:brat-config} (per file, each type and its class, by type) and
 * {@code turbo:brat-config-errors} (per file that throws, the class and message).</p>
 *
 * <p>The stream is read until {@code read} returns null; an exception is recorded and reading goes
 * on. The layers, anchored to the document: {@code turbo:samples} ({@code toString} of each
 * sample), {@code turbo:read-errors} (per exception, the number of samples read before it, class
 * and message), and for the name streams {@code turbo:sample-tokens} and {@code turbo:names} as
 * {@link FormatGolden} writes them.</p>
 */
final class NerGolden implements AnnotatorGolden {

  /** The input set the corpora are read from. */
  static final String SET = Runs.NER;

  /** The stopping point for a stream that throws without moving on, which would otherwise loop. */
  private static final int MAX_READS = 100_000;

  /** The streams, each with the prefix of the input ids it reads and its parameters. */
  enum Kind {
    GERMEVAL_OUTER("germeval-outer", "germeval", GermEval2014NameSampleStream.class, "per,loc,org,misc", "outer"),
    GERMEVAL_INNER("germeval-inner", "germeval", GermEval2014NameSampleStream.class, "per,loc,org,misc", "inner"),
    GERMEVAL_LOC_MISC("germeval-outer-loc-misc", "germeval", GermEval2014NameSampleStream.class, "loc,misc",
        "outer"),
    BIONLP("bionlp", "bionlp", BioNLP2004NameSampleStream.class, "DNA,protein,cell_type,cell_line,RNA", null),
    BIONLP_CELL_LINE("bionlp-cell-line", "bionlp", BioNLP2004NameSampleStream.class, "cell_line", null),
    BIONLP_DNA_CELL_TYPE("bionlp-dna-cell-type", "bionlp", BioNLP2004NameSampleStream.class, "DNA,cell_type",
        null),
    EVALITA("evalita", "evalita", EvalitaNameSampleStream.class, "per,org,loc,gpe", null),
    EVALITA_PER_GPE("evalita-per-gpe", "evalita", EvalitaNameSampleStream.class, "per,gpe", null),
    CENSUS("census", "census", NameFinderCensus90NameStream.class, null, null),
    MUC_WHITESPACE("muc6-whitespace", "muc", MucNameSampleStream.class, null, null),
    MUC_SIMPLE("muc6-simple", "muc", MucNameSampleStream.class, null, null),
    BRAT_WHITESPACE("brat-whitespace", "brat", BratNameSampleStream.class, null, null),
    BRAT_SIMPLE("brat-simple", "brat", BratNameSampleStream.class, null, null),
    BRAT_RECURSIVE_TYPES("brat-recursive-person-location", "brat", BratNameSampleStream.class, null, null),
    BRAT_CONFIG("brat-config", "brat", AnnotationConfiguration.class, null, null);

    final String dir;
    final String inputPrefix;
    final Class<?> streamClass;
    final String types;
    final String layer;

    Kind(String dir, String inputPrefix, Class<?> streamClass, String types, String layer) {
      this.dir = dir;
      this.inputPrefix = inputPrefix;
      this.streamClass = streamClass;
      this.types = types;
      this.layer = layer;
    }

    /** Whether the inputs are bundles of a corpus directory. */
    boolean bundle() {
      return inputPrefix.equals("muc") || inputPrefix.equals("brat");
    }
  }

  /** The brat input with a character outside the Basic Multilingual Plane. */
  static final String BRAT_UTF16 = "brat-utf16";

  /** The name types of the brat run that keeps some, as its {@code nameTypes} parameter. */
  static final String BRAT_TYPES = "Person , Location";

  /** The expected MUC name types in the order of {@code MucNameContentHandler}'s source. */
  static final String MUC_TYPES = "[PERSON, ORGANIZATION, LOCATION, DATE, TIME, MONEY, PERCENT]";

  private final Kind kind;
  private final WhitespaceMode whitespace;

  private NerGolden(Kind kind, WhitespaceMode whitespace) {
    this.kind = kind;
    this.whitespace = whitespace;
  }

  /** Every run: each stream, and the MUC and brat streams with the whitespace tokenizer in the legacy mode too. */
  static List<NerGolden> all() {
    List<NerGolden> runs = new ArrayList<>();
    for (Kind k : Kind.values()) {
      runs.add(new NerGolden(k, WhitespaceMode.UNICODE));
    }
    runs.add(new NerGolden(Kind.MUC_WHITESPACE, WhitespaceMode.LEGACY));
    runs.add(new NerGolden(Kind.BRAT_WHITESPACE, WhitespaceMode.LEGACY));
    return runs;
  }

  @Override
  public String annotator() {
    return "ner";
  }

  @Override
  public List<ModelRef> models() {
    return List.of();
  }

  @Override
  public String annotatorClass() {
    return kind.streamClass.getName();
  }

  @Override
  public String method() {
    return kind == Kind.BRAT_CONFIG ? "parse" : "read";
  }

  @Override
  public Map<String, String> config() {
    Map<String, String> c = new LinkedHashMap<>();
    if (kind.streamClass == EvalitaNameSampleStream.class) {
      c.put("lang", "it");
    }
    if (kind.types != null) {
      c.put("types", kind.types);
    }
    if (kind.layer != null) {
      c.put("layer", kind.layer);
    }
    if (kind == Kind.MUC_WHITESPACE || kind == Kind.BRAT_WHITESPACE || kind == Kind.BRAT_RECURSIVE_TYPES) {
      c.put("tokenizer", "whitespace");
    } else if (kind == Kind.MUC_SIMPLE || kind == Kind.BRAT_SIMPLE) {
      c.put("tokenizer", "simple");
    }
    if (kind == Kind.BRAT_RECURSIVE_TYPES) {
      c.put("recursive", "true");
      c.put("nameTypes", BRAT_TYPES);
    }
    c.putAll(Whitespace.config(whitespace));
    return c;
  }

  @Override
  public Set<String> inputSets() {
    return Set.of(SET);
  }

  @Override
  public boolean accepts(Input input) {
    // Java's simple tokenizer cuts a surrogate pair in two, which a golden cannot hold
    if (kind == Kind.BRAT_SIMPLE && input.id().equals(BRAT_UTF16)) {
      return false;
    }
    return input.set().equals(SET) && input.id().startsWith(kind.inputPrefix + "-");
  }

  @Override
  public String modelDir() {
    return kind.dir + Whitespace.dirSuffix(whitespace);
  }

  /** The GermEval types as {@code GermEval2014NameSampleStreamFactory} makes them. */
  private int germEvalTypes() {
    int t = 0;
    if (kind.types.contains("per")) {
      t |= GermEval2014NameSampleStream.GENERATE_PERSON_ENTITIES;
    }
    if (kind.types.contains("org")) {
      t |= GermEval2014NameSampleStream.GENERATE_ORGANIZATION_ENTITIES;
    }
    if (kind.types.contains("loc")) {
      t |= GermEval2014NameSampleStream.GENERATE_LOCATION_ENTITIES;
    }
    if (kind.types.contains("misc")) {
      t |= GermEval2014NameSampleStream.GENERATE_MISC_ENTITIES;
    }
    return t;
  }

  /** The BioNLP types as {@code BioNLP2004NameSampleStreamFactory} makes them. */
  private int bioNlpTypes() {
    int t = 0;
    if (kind.types.contains("DNA")) {
      t |= BioNLP2004NameSampleStream.GENERATE_DNA_ENTITIES;
    }
    if (kind.types.contains("protein")) {
      t |= BioNLP2004NameSampleStream.GENERATE_PROTEIN_ENTITIES;
    }
    if (kind.types.contains("cell_type")) {
      t |= BioNLP2004NameSampleStream.GENERATE_CELLTYPE_ENTITIES;
    }
    if (kind.types.contains("cell_line")) {
      t |= BioNLP2004NameSampleStream.GENERATE_CELLLINE_ENTITIES;
    }
    if (kind.types.contains("RNA")) {
      t |= BioNLP2004NameSampleStream.GENERATE_RNA_ENTITIES;
    }
    return t;
  }

  /** The Evalita types as {@code EvalitaNameSampleStreamFactory} makes them. */
  private int evalitaTypes() {
    int t = 0;
    if (kind.types.contains("per")) {
      t |= EvalitaNameSampleStream.GENERATE_PERSON_ENTITIES;
    }
    if (kind.types.contains("org")) {
      t |= EvalitaNameSampleStream.GENERATE_ORGANIZATION_ENTITIES;
    }
    if (kind.types.contains("loc")) {
      t |= EvalitaNameSampleStream.GENERATE_LOCATION_ENTITIES;
    }
    if (kind.types.contains("gpe")) {
      t |= EvalitaNameSampleStream.GENERATE_GPE_ENTITIES;
    }
    return t;
  }

  /** The stream {@code Muc6NameSampleStreamFactory} makes over the directory {@code root}. */
  private ObjectStream<?> openMuc(Path root) {
    Tokenizer tokenizer = kind == Kind.MUC_SIMPLE ? SimpleTokenizer.INSTANCE : WhitespaceTokenizer.INSTANCE;
    ObjectStream<String> documents = new FileToStringSampleStream(new DirectorySampleStream(root.toFile(),
        file -> StringUtil.toLowerCase(file.getName()).endsWith(".sgm"), false), StandardCharsets.UTF_8);
    // the constructor is protected, as the factory is in the stream's package
    return new MucNameSampleStream(tokenizer, documents) {};
  }

  /** {@code message} with the expected MUC types written in a fixed order. */
  private static String mucMessage(String message) {
    try {
      Field f = MucNameContentHandler.class.getDeclaredField("EXPECTED_TYPES");
      f.setAccessible(true);
      Set<?> types = (Set<?>) f.get(null);
      if (!types.equals(Set.of("PERSON", "ORGANIZATION", "LOCATION", "DATE", "TIME", "MONEY", "PERCENT"))) {
        throw new IllegalStateException("unexpected MUC types " + types);
      }
      return message.replace(types.toString(), MUC_TYPES);
    } catch (ReflectiveOperationException e) {
      throw new IllegalStateException(e);
    }
  }

  /** The stream {@code BratNameSampleStreamFactory} makes over the directory {@code root}. */
  private ObjectStream<?> openBrat(Path root) throws IOException {
    AnnotationConfiguration config = AnnotationConfiguration.parse(root.resolve("annotation.conf").toFile());
    ObjectStream<BratDocument> documents = new BratDocumentStream(config, root.toFile(),
        kind == Kind.BRAT_RECURSIVE_TYPES, null);
    Tokenizer tokenizer = kind == Kind.BRAT_SIMPLE ? SimpleTokenizer.INSTANCE : WhitespaceTokenizer.INSTANCE;
    Set<String> nameTypes = null;
    if (kind == Kind.BRAT_RECURSIVE_TYPES) {
      String[] names = StringUtil.split(BRAT_TYPES, ',');
      if (names.length > 0) {
        nameTypes = Arrays.stream(names).map(String::trim).collect(Collectors.toSet());
      }
    }
    return new BratNameSampleStream(new NewlineSentenceDetector(), tokenizer, documents, nameTypes);
  }

  /** The layers of the {@code brat-config} run over the directory {@code root}. */
  private static List<Layer> bratConfig(Path root) {
    FormatGolden.Columns types = new FormatGolden.Columns();
    FormatGolden.Columns errors = new FormatGolden.Columns();
    File[] files = root.toFile().listFiles((dir, name) -> name.endsWith(".conf"));
    Arrays.sort(files, Comparator.comparing(File::getName));
    for (File f : files) {
      try {
        AnnotationConfiguration config = AnnotationConfiguration.parse(f);
        Field map = AnnotationConfiguration.class.getDeclaredField("typeToClassMap");
        map.setAccessible(true);
        Map<?, ?> classes = new TreeMap<>((Map<?, ?>) map.get(config));
        for (Map.Entry<?, ?> e : classes.entrySet()) {
          types.str("file", f.getName()).str("type", (String) e.getKey()).str("class", (String) e.getValue());
        }
      } catch (ReflectiveOperationException e) {
        throw new IllegalStateException(e);
      } catch (IOException | RuntimeException e) {
        errors.str("file", f.getName()).str("class", e.getClass().getName()).str("message", FormatGolden.message(e));
      }
    }
    return List.of(types.layer("turbo:brat-config", "document", "file", "type", "class"),
        errors.layer("turbo:brat-config-errors", "document", "file", "class", "message"));
  }

  private ObjectStream<?> open(Input input, Path root) throws IOException {
    InputStreamFactory in = () -> new ByteArrayInputStream(input.bytes());
    return switch (kind) {
      case MUC_WHITESPACE, MUC_SIMPLE -> openMuc(root);
      case BRAT_WHITESPACE, BRAT_SIMPLE, BRAT_RECURSIVE_TYPES -> openBrat(root);
      case BRAT_CONFIG -> throw new IllegalStateException("the configuration run reads no stream");
      case GERMEVAL_OUTER, GERMEVAL_INNER, GERMEVAL_LOC_MISC -> new GermEval2014NameSampleStream(
          new PlainTextByLineStream(in, StandardCharsets.UTF_8), germEvalTypes(),
          kind.layer.equals("inner") ? GermEval2014NameSampleStream.NerLayer.INNER
              : GermEval2014NameSampleStream.NerLayer.OUTER);
      case BIONLP, BIONLP_CELL_LINE, BIONLP_DNA_CELL_TYPE -> new BioNLP2004NameSampleStream(in, bioNlpTypes());
      case EVALITA, EVALITA_PER_GPE -> new EvalitaNameSampleStream(EvalitaNameSampleStream.LANGUAGE.IT,
          new PlainTextByLineStream(in, StandardCharsets.UTF_8), evalitaTypes());
      case CENSUS -> new NameFinderCensus90NameStream(in, StandardCharsets.UTF_8);
    };
  }

  @Override
  public List<Layer> layers(Input input) throws AnnotatorException {
    if (!kind.bundle()) {
      return read(input, null);
    }
    Path root = null;
    try {
      root = Files.createTempDirectory("ner-golden");
      MascGolden.writeBundle(input, root);
      Path dir = root;
      return Whitespace.in(whitespace, () -> kind == Kind.BRAT_CONFIG ? bratConfig(dir) : read(input, dir));
    } catch (IOException e) {
      throw new IllegalStateException(input.inputId() + ": the corpus could not be written", e);
    } finally {
      if (root != null) {
        MascGolden.delete(root);
      }
    }
  }

  private List<Layer> read(Input input, Path root) {
    List<Object> samples = new ArrayList<>();
    FormatGolden.Columns errors = new FormatGolden.Columns();
    try (ObjectStream<?> stream = open(input, root)) {
      for (int reads = 0; ; reads++) {
        if (reads > MAX_READS) {
          throw new IllegalStateException(input.inputId() + ": the stream does not end");
        }
        Object sample;
        try {
          sample = stream.read();
        } catch (IOException | RuntimeException e) {
          String message = FormatGolden.message(e);
          errors.u32("after", samples.size()).str("class", e.getClass().getName())
              .str("message", root == null ? message : mucMessage(message).replace(root.toString(), "<corpus>"));
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
    if (kind == Kind.CENSUS) {
      FormatGolden.Columns names = new FormatGolden.Columns();
      for (int i = 0; i < samples.size(); i++) {
        for (String t : (StringList) samples.get(i)) {
          names.u32("sample", i).str("text", t);
        }
      }
      layers.add(names.layer("turbo:sample-tokens", "document", "sample", "text"));
    } else {
      FormatGolden.names(samples, layers);
    }
    return layers;
  }
}
