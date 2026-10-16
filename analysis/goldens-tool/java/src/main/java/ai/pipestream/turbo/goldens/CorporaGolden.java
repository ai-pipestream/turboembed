package ai.pipestream.turbo.goldens;

import java.io.ByteArrayInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.lang.reflect.Constructor;
import java.lang.reflect.Field;
import java.lang.reflect.InvocationTargetException;
import java.lang.reflect.Method;
import java.nio.charset.StandardCharsets;
import java.nio.file.DirectoryStream;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;
import opennlp.tools.formats.TwentyNewsgroupSampleStream;
import opennlp.tools.formats.irishsentencebank.IrishSentenceBankDocument;
import opennlp.tools.formats.leipzig.LeipzigLanguageSampleStream;
import opennlp.tools.formats.letsmt.DetokenizeSentenceSampleStream;
import opennlp.tools.formats.letsmt.LetsmtDocument;
import opennlp.tools.formats.moses.MosesSentenceSampleStream;
import opennlp.tools.formats.nkjp.NKJPSegmentationDocument;
import opennlp.tools.formats.nkjp.NKJPTextDocument;
import opennlp.tools.tokenize.DetokenizationDictionary;
import opennlp.tools.tokenize.DictionaryDetokenizer;
import opennlp.tools.tokenize.SimpleTokenizer;
import opennlp.tools.tokenize.Tokenizer;
import opennlp.tools.tokenize.WhitespaceTokenizer;
import opennlp.tools.util.ObjectStream;
import opennlp.tools.util.PlainTextByLineStream;
import opennlp.tools.util.WhitespaceMode;

/**
 * The golden of one of OpenNLP's sentence, language or document corpus streams over a file of the
 * {@code corpora} input set, made as its factory makes it: {@code MosesSentenceSampleStream} over
 * the file's lines read as UTF-8 ({@code moses-} inputs); {@code LetsmtSentenceStream} over the
 * parsed document, alone and behind {@code DetokenizeSentenceSampleStream} with the dictionary of
 * {@link #DETOKENIZER} ({@code letsmt-}); {@code NKJPSentenceSampleStream} over the
 * {@code ann_segmentation.xml} and {@code text.xml} of a bundle ({@code nkjp-});
 * {@code IrishSentenceBankSentenceStream} and {@code IrishSentenceBankTokenSampleStream}
 * ({@code irish-}); {@code LeipzigLanguageSampleStream} behind {@code SampleShuffleStream} and
 * {@code SampleSkipStream} over a bundled directory, with two sets of parameters
 * ({@code leipzig-}); and {@code TwentyNewsgroupSampleStream} over a bundled directory with the
 * whitespace tokenizer (also in the legacy whitespace mode) and the simple tokenizer
 * ({@code newsgroups-}). The bundles are described in {@link MascGolden}. The streams and the
 * document parsers whose constructors or methods are package-private are reached by reflection.
 *
 * <p>What the factory does before the stream reads (parsing a document, and for Leipzig reading
 * every file, as {@code SampleShuffleStream} buffers all the samples) is recorded as a read
 * error before any sample when it throws. The stream is then read until {@code read} returns
 * null; an exception is recorded and reading goes on.</p>
 *
 * <p>The 20 Newsgroups stream keeps its files in a {@code HashMap} keyed by path, so the order of
 * its samples depends on the corpus directory's path: the runs write the corpus to
 * {@link #NEWSGROUPS_DIR} under {@code java.io.tmpdir}, and give the path as the {@code dataDir}
 * config record. They also write {@code turbo:listing}, the files as the directories list them
 * (the order the map is filled in), and {@code turbo:files}, the order the map gives them, each
 * as {@code <group>/<file>}. The Leipzig message that names a file writes the corpus directory as
 * {@code <corpus>}.</p>
 *
 * <p>The layers, anchored to the document: {@code turbo:samples} ({@code toString} of each
 * sample) and {@code turbo:read-errors} (per exception, the number of samples read before it,
 * class and message); then for sentence and token samples {@code turbo:sample-texts} and
 * {@code turbo:sample-spans} (UTF-8 byte offsets), for language samples
 * {@code turbo:sample-labels} and {@code turbo:sample-texts}, and for document samples
 * {@code turbo:sample-labels} and {@code turbo:sample-tokens}, as {@link FormatGolden} writes
 * them.</p>
 */
final class CorporaGolden implements AnnotatorGolden {

  /** The input set the corpora are read from. */
  static final String INPUT_SET = Runs.CORPORA;

  /** The annotator directory. */
  static final String ANNOTATOR = "corpora";

  /** The detokenizer dictionary of the detokenized LetsMT run: each token and its operation. */
  static final String DETOKENIZER = ". MOVE_LEFT , MOVE_LEFT ) MOVE_LEFT ( MOVE_RIGHT \" RIGHT_LEFT_MATCHING";

  /** The directory under {@code java.io.tmpdir} the 20 Newsgroups corpora are written to. */
  static final String NEWSGROUPS_DIR = "turbo-20newsgroup";

  /** The stopping point for a stream that throws without moving on, which would otherwise loop. */
  private static final int MAX_READS = 100_000;

  /** What a run reads. */
  enum Kind {
    MOSES("moses", "moses", "opennlp.tools.formats.moses.MosesSentenceSampleStream"),
    LETSMT("letsmt", "letsmt", "opennlp.tools.formats.letsmt.LetsmtSentenceStream"),
    LETSMT_DETOKENIZED("letsmt-detokenized", "letsmt",
        "opennlp.tools.formats.letsmt.DetokenizeSentenceSampleStream"),
    NKJP("nkjp", "nkjp", "opennlp.tools.formats.nkjp.NKJPSentenceSampleStream"),
    IRISH_SENTENCES("irish-sentences", "irish",
        "opennlp.tools.formats.irishsentencebank.IrishSentenceBankSentenceStream"),
    IRISH_TOKENS("irish-tokens", "irish",
        "opennlp.tools.formats.irishsentencebank.IrishSentenceBankTokenSampleStream"),
    LEIPZIG_2X3("leipzig-2x3", "leipzig", "opennlp.tools.formats.leipzig.LeipzigLanguageSampleStream"),
    LEIPZIG_1X4_SKIP2("leipzig-1x4-skip2", "leipzig",
        "opennlp.tools.formats.leipzig.LeipzigLanguageSampleStream"),
    NEWSGROUPS_WHITESPACE("20newsgroup-whitespace", "newsgroups",
        "opennlp.tools.formats.TwentyNewsgroupSampleStream"),
    NEWSGROUPS_SIMPLE("20newsgroup-simple", "newsgroups", "opennlp.tools.formats.TwentyNewsgroupSampleStream");

    final String dir;
    final String inputPrefix;
    final String streamClass;

    Kind(String dir, String inputPrefix, String streamClass) {
      this.dir = dir;
      this.inputPrefix = inputPrefix;
      this.streamClass = streamClass;
    }

    /** Whether the inputs are bundles of a corpus directory. */
    boolean bundle() {
      return inputPrefix.equals("nkjp") || inputPrefix.equals("leipzig") || inputPrefix.equals("newsgroups");
    }

    /** {@code sentencesPerSample}, {@code samplesPerLanguage} and {@code samplesToSkip} of a Leipzig run. */
    int[] leipzig() {
      return this == LEIPZIG_2X3 ? new int[] {2, 3, 0} : new int[] {1, 4, 2};
    }
  }

  private final Kind kind;
  private final WhitespaceMode whitespace;

  private CorporaGolden(Kind kind, WhitespaceMode whitespace) {
    this.kind = kind;
    this.whitespace = whitespace;
  }

  /** Every run, and the 20 Newsgroups whitespace tokenizer in the legacy whitespace mode too. */
  static List<CorporaGolden> all() {
    List<CorporaGolden> runs = new ArrayList<>();
    for (Kind k : Kind.values()) {
      runs.add(new CorporaGolden(k, WhitespaceMode.UNICODE));
    }
    runs.add(new CorporaGolden(Kind.NEWSGROUPS_WHITESPACE, WhitespaceMode.LEGACY));
    return runs;
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
  public String annotatorClass() {
    return kind.streamClass;
  }

  @Override
  public String method() {
    return "read";
  }

  @Override
  public Map<String, String> config() {
    Map<String, String> c = new LinkedHashMap<>();
    if (kind == Kind.LETSMT_DETOKENIZED) {
      c.put("detokenizer", DETOKENIZER);
    }
    if (kind == Kind.LEIPZIG_2X3 || kind == Kind.LEIPZIG_1X4_SKIP2) {
      int[] p = kind.leipzig();
      c.put("sentencesPerSample", Integer.toString(p[0]));
      c.put("samplesPerLanguage", Integer.toString(p[1]));
      c.put("samplesToSkip", Integer.toString(p[2]));
    }
    if (kind == Kind.NEWSGROUPS_WHITESPACE || kind == Kind.NEWSGROUPS_SIMPLE) {
      c.put("tokenizer", kind == Kind.NEWSGROUPS_SIMPLE ? "simple" : "whitespace");
      c.put("dataDir", newsgroupsDir().toString());
    }
    c.putAll(Whitespace.config(whitespace));
    return c;
  }

  @Override
  public Set<String> inputSets() {
    return Set.of(INPUT_SET);
  }

  @Override
  public boolean accepts(Input input) {
    return input.set().equals(INPUT_SET) && input.id().startsWith(kind.inputPrefix + "-");
  }

  @Override
  public String modelDir() {
    return kind.dir + Whitespace.dirSuffix(whitespace);
  }

  private static Path newsgroupsDir() {
    return Path.of(System.getProperty("java.io.tmpdir"), NEWSGROUPS_DIR);
  }

  /** The detokenizer of {@link #DETOKENIZER}. */
  private static DictionaryDetokenizer detokenizer() {
    String[] parts = DETOKENIZER.split(" ");
    String[] tokens = new String[parts.length / 2];
    DetokenizationDictionary.Operation[] ops = new DetokenizationDictionary.Operation[parts.length / 2];
    for (int i = 0; i < tokens.length; i++) {
      tokens[i] = parts[2 * i];
      ops[i] = DetokenizationDictionary.Operation.valueOf(parts[2 * i + 1]);
    }
    return new DictionaryDetokenizer(new DetokenizationDictionary(tokens, ops));
  }

  /** A package-private constructor of {@code className}, made accessible. */
  private static Object construct(String className, Class<?>[] types, Object... args) throws IOException {
    try {
      Constructor<?> c = Class.forName(className).getDeclaredConstructor(types);
      c.setAccessible(true);
      return c.newInstance(args);
    } catch (InvocationTargetException e) {
      throw rethrow(e);
    } catch (ReflectiveOperationException e) {
      throw new IllegalStateException(e);
    }
  }

  /** The cause of a reflective call's exception, thrown as it was. */
  private static IOException rethrow(InvocationTargetException e) {
    Throwable cause = e.getCause();
    if (cause instanceof IOException io) {
      return io;
    }
    if (cause instanceof RuntimeException r) {
      throw r;
    }
    if (cause instanceof Error err) {
      throw err;
    }
    throw new IllegalStateException(cause);
  }

  /** {@code LetsmtDocument.parse(InputStream)}, which is package-private. */
  private static LetsmtDocument parseLetsmt(InputStream in) throws IOException {
    try {
      Method m = LetsmtDocument.class.getDeclaredMethod("parse", InputStream.class);
      m.setAccessible(true);
      return (LetsmtDocument) m.invoke(null, in);
    } catch (InvocationTargetException e) {
      throw rethrow(e);
    } catch (ReflectiveOperationException e) {
      throw new IllegalStateException(e);
    }
  }

  /** The stream the run's factory makes; {@code root} is the bundle's directory. */
  @SuppressWarnings("unchecked")
  private ObjectStream<Object> open(Input input, Path root) throws IOException {
    InputStream in = new ByteArrayInputStream(input.bytes());
    return (ObjectStream<Object>) switch (kind) {
      case MOSES -> new MosesSentenceSampleStream(new PlainTextByLineStream(() -> in, StandardCharsets.UTF_8));
      case LETSMT -> construct(kind.streamClass, new Class<?>[] {LetsmtDocument.class}, parseLetsmt(in));
      case LETSMT_DETOKENIZED -> new DetokenizeSentenceSampleStream(detokenizer(),
          (ObjectStream<opennlp.tools.sentdetect.SentenceSample>) construct(Kind.LETSMT.streamClass,
              new Class<?>[] {LetsmtDocument.class}, parseLetsmt(in)));
      case NKJP -> {
        NKJPSegmentationDocument segments;
        NKJPTextDocument text;
        try (InputStream s = Files.newInputStream(root.resolve("ann_segmentation.xml"));
            InputStream t = Files.newInputStream(root.resolve("text.xml"))) {
          segments = NKJPSegmentationDocument.parse(s);
          text = NKJPTextDocument.parse(t);
        }
        yield construct(kind.streamClass, new Class<?>[] {NKJPSegmentationDocument.class, NKJPTextDocument.class},
            segments, text);
      }
      case IRISH_SENTENCES, IRISH_TOKENS -> construct(kind.streamClass,
          new Class<?>[] {IrishSentenceBankDocument.class}, IrishSentenceBankDocument.parse(in));
      case LEIPZIG_2X3, LEIPZIG_1X4_SKIP2 -> {
        int[] p = kind.leipzig();
        Object leipzig = new LeipzigLanguageSampleStream(root.toFile(), p[0], p[1] + p[2]);
        Object shuffled = construct("opennlp.tools.formats.leipzig.SampleShuffleStream",
            new Class<?>[] {ObjectStream.class}, leipzig);
        yield construct("opennlp.tools.formats.leipzig.SampleSkipStream",
            new Class<?>[] {ObjectStream.class, int.class}, shuffled, p[2]);
      }
      case NEWSGROUPS_WHITESPACE, NEWSGROUPS_SIMPLE -> {
        Tokenizer tokenizer = kind == Kind.NEWSGROUPS_SIMPLE ? SimpleTokenizer.INSTANCE : WhitespaceTokenizer.INSTANCE;
        yield construct(kind.streamClass, new Class<?>[] {Tokenizer.class, Path.class}, tokenizer, root);
      }
    };
  }

  @Override
  public List<Layer> layers(Input input) throws AnnotatorException {
    if (!kind.bundle()) {
      return read(input, null);
    }
    boolean newsgroups = kind.inputPrefix.equals("newsgroups");
    Path root = null;
    try {
      if (newsgroups) {
        root = newsgroupsDir();
        if (Files.exists(root)) {
          MascGolden.delete(root);
        }
        Files.createDirectories(root);
      } else {
        root = Files.createTempDirectory("corpora-golden");
      }
      MascGolden.writeBundle(input, root);
      Path dir = root;
      return Whitespace.in(whitespace, () -> read(input, dir));
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
    List<Layer> layers = new ArrayList<>();
    ObjectStream<Object> stream = null;
    try {
      stream = open(input, root);
    } catch (IOException | RuntimeException e) {
      errors.u32("after", 0).str("class", e.getClass().getName()).str("message", message(e, root));
    }
    if (stream != null) {
      if (kind.inputPrefix.equals("newsgroups")) {
        newsgroupFiles(stream, root, layers);
      }
      for (int reads = 0; ; reads++) {
        if (reads > MAX_READS) {
          throw new IllegalStateException(input.inputId() + ": the stream does not end");
        }
        Object sample;
        try {
          sample = stream.read();
        } catch (IOException | RuntimeException e) {
          errors.u32("after", samples.size()).str("class", e.getClass().getName()).str("message", message(e, root));
          continue;
        }
        if (sample == null) {
          break;
        }
        samples.add(sample);
      }
    }

    FormatGolden.Columns written = new FormatGolden.Columns();
    for (Object s : samples) {
      written.str("written", s.toString());
    }
    layers.add(0, written.layer("turbo:samples", "document", "written"));
    layers.add(1, errors.layer("turbo:read-errors", "document", "after", "class", "message"));
    switch (kind) {
      case MOSES, LETSMT, LETSMT_DETOKENIZED, NKJP, IRISH_SENTENCES ->
          FormatGolden.sentenceSamples(input, samples, layers);
      case IRISH_TOKENS -> FormatGolden.tokenSamples(input, samples, layers);
      case LEIPZIG_2X3, LEIPZIG_1X4_SKIP2 -> FormatGolden.languages(samples, layers);
      case NEWSGROUPS_WHITESPACE, NEWSGROUPS_SIMPLE -> FormatGolden.documents(samples, layers);
    }
    return layers;
  }

  /** {@code message} with the corpus directory written {@code <corpus>}. */
  private static String message(Exception e, Path root) {
    String message = FormatGolden.message(e);
    return root == null ? message : message.replace(root.toString(), "<corpus>");
  }

  /**
   * The {@code turbo:listing} and {@code turbo:files} layers of a 20 Newsgroups run: the files as
   * the stream's constructor lists them, and as its map gives them.
   */
  private static void newsgroupFiles(ObjectStream<Object> stream, Path root, List<Layer> layers) {
    FormatGolden.Columns listing = new FormatGolden.Columns();
    try (DirectoryStream<Path> dirs = Files.newDirectoryStream(root, Files::isDirectory)) {
      for (Path dir : dirs) {
        try (DirectoryStream<Path> files = Files.newDirectoryStream(dir)) {
          for (Path file : files) {
            listing.str("file", root.relativize(file).toString());
          }
        }
      }
    } catch (IOException e) {
      throw new IllegalStateException("could not list " + root, e);
    }
    FormatGolden.Columns files = new FormatGolden.Columns();
    try {
      Field f = TwentyNewsgroupSampleStream.class.getDeclaredField("catFileMap");
      f.setAccessible(true);
      for (Object key : ((Map<?, ?>) f.get(stream)).keySet()) {
        files.str("file", root.relativize((Path) key).toString());
      }
    } catch (ReflectiveOperationException e) {
      throw new IllegalStateException(e);
    }
    layers.add(listing.layer("turbo:listing", "document", "file"));
    layers.add(files.layer("turbo:files", "document", "file"));
  }
}
