package ai.pipestream.turbo.goldens;

import java.io.ByteArrayInputStream;
import java.io.IOException;
import java.lang.reflect.InvocationHandler;
import java.lang.reflect.Method;
import java.lang.reflect.Proxy;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.Set;
import opennlp.tools.chunker.ChunkSample;
import opennlp.tools.chunker.ChunkSampleStream;
import opennlp.tools.chunker.Chunker;
import opennlp.tools.chunker.ChunkerEvaluationMonitor;
import opennlp.tools.chunker.ChunkerEvaluator;
import opennlp.tools.doccat.DoccatEvaluationMonitor;
import opennlp.tools.doccat.DocumentCategorizer;
import opennlp.tools.doccat.DocumentCategorizerEvaluator;
import opennlp.tools.doccat.DocumentSample;
import opennlp.tools.doccat.DocumentSampleStream;
import opennlp.tools.langdetect.Language;
import opennlp.tools.langdetect.LanguageDetector;
import opennlp.tools.langdetect.LanguageDetectorEvaluationMonitor;
import opennlp.tools.langdetect.LanguageDetectorEvaluator;
import opennlp.tools.langdetect.LanguageDetectorSampleStream;
import opennlp.tools.langdetect.LanguageSample;
import opennlp.tools.lemmatizer.LemmaSample;
import opennlp.tools.lemmatizer.LemmaSampleStream;
import opennlp.tools.lemmatizer.Lemmatizer;
import opennlp.tools.lemmatizer.LemmatizerEvaluationMonitor;
import opennlp.tools.lemmatizer.LemmatizerEvaluator;
import opennlp.tools.namefind.NameSample;
import opennlp.tools.namefind.NameSampleDataStream;
import opennlp.tools.namefind.TokenNameFinder;
import opennlp.tools.namefind.TokenNameFinderEvaluationMonitor;
import opennlp.tools.namefind.TokenNameFinderEvaluator;
import opennlp.tools.postag.POSEvaluator;
import opennlp.tools.postag.POSSample;
import opennlp.tools.postag.POSTagger;
import opennlp.tools.postag.POSTaggerEvaluationMonitor;
import opennlp.tools.postag.WordTagSampleStream;
import opennlp.tools.sentdetect.SentenceDetector;
import opennlp.tools.sentdetect.SentenceDetectorEvaluationMonitor;
import opennlp.tools.sentdetect.SentenceDetectorEvaluator;
import opennlp.tools.sentdetect.SentenceSample;
import opennlp.tools.sentdetect.SentenceSampleStream;
import opennlp.tools.tokenize.TokenSample;
import opennlp.tools.tokenize.TokenSampleStream;
import opennlp.tools.tokenize.Tokenizer;
import opennlp.tools.tokenize.TokenizerEvaluationMonitor;
import opennlp.tools.tokenize.TokenizerEvaluator;
import opennlp.tools.util.ObjectStream;
import opennlp.tools.util.PlainTextByLineStream;
import opennlp.tools.util.Span;
import opennlp.tools.util.WhitespaceMode;
import opennlp.tools.util.eval.CrossValidationPartitioner;
import opennlp.tools.util.eval.Evaluator;
import opennlp.tools.util.eval.FMeasure;

/**
 * The golden of one of OpenNLP's evaluators over a file of sample pairs: the inputs of the
 * {@code formats} set whose id starts with {@code pairs-} and a format name, read with that
 * format's stream, hold a reference sample and then the prediction for it, pair after pair. A
 * stand-in component (a {@link Proxy} of the component's interface) answers each reference with
 * its prediction: the tags, lemmas, names, chunk tags, token spans, sentence spans, category or
 * language of the sample after it. A component's prediction is applied to the reference as it
 * stands, so a predicted token or sentence span is an offset of the prediction's own text.
 *
 * <p>The layers, all anchored to the document:</p>
 *
 * <ul>
 *   <li>{@code turbo:verdicts}: per pair, {@code correct} or {@code misclassified} as the
 *       evaluation monitor heard it, or {@code error} with the class and message of what
 *       {@code evaluateSample} threw.</li>
 *   <li>{@code turbo:fmeasure}: per pair, after it, the evaluator's precision, recall and
 *       F-measure, for the evaluators that keep an {@code FMeasure}.</li>
 *   <li>{@code turbo:accuracy}: per pair, after it, the accuracy and the count of tokens or
 *       documents, for the evaluators that keep a mean.</li>
 *   <li>{@code turbo:span-scores}: per pair of name or token samples, the static
 *       {@code FMeasure.precision} and {@code FMeasure.recall} of the pair's spans.</li>
 *   <li>{@code turbo:folds}: for the {@code cross-validation} run, the place of every sample
 *       of each fold's training and test streams, for 1 to 4 partitions over all the samples
 *       of the file.</li>
 * </ul>
 */
final class EvalGolden implements AnnotatorGolden {

  /** The input set the pairs are read from. */
  static final String SET = Runs.FORMATS;

  /** The prefix of the pair inputs' ids, before the format name. */
  static final String PAIRS = "pairs-";

  /** The largest number of partitions the cross-validation run lists. */
  static final int MAX_PARTITIONS = 4;

  enum Kind {
    POS("pos", "wordtag", POSEvaluator.class, false),
    LEMMA("lemma", "lemma", LemmatizerEvaluator.class, false),
    NAME("name", "name", TokenNameFinderEvaluator.class, false),
    CHUNK("chunk", "chunk", ChunkerEvaluator.class, false),
    TOKEN("token", "token", TokenizerEvaluator.class, false),
    SENTENCE("sentence", "sentence", SentenceDetectorEvaluator.class, true),
    DOCCAT("doccat", "doccat", DocumentCategorizerEvaluator.class, false),
    LANGUAGE("language", "language", LanguageDetectorEvaluator.class, false),
    CROSS_VALIDATION("cross-validation", "wordtag", CrossValidationPartitioner.class, false);

    final String dir;
    final String format;
    final Class<?> evaluatorClass;
    /** Whether the evaluator tests for whitespace, so it is also run in the legacy mode. */
    final boolean testsWhitespace;

    Kind(String dir, String format, Class<?> evaluatorClass, boolean testsWhitespace) {
      this.dir = dir;
      this.format = format;
      this.evaluatorClass = evaluatorClass;
      this.testsWhitespace = testsWhitespace;
    }

    boolean keepsFMeasure() {
      return this == NAME || this == CHUNK || this == TOKEN || this == SENTENCE;
    }
  }

  private final Kind kind;
  private final WhitespaceMode whitespace;

  private EvalGolden(Kind kind, WhitespaceMode whitespace) {
    this.kind = kind;
    this.whitespace = whitespace;
  }

  /** Every run: each evaluator, in the legacy whitespace mode too when it tests for whitespace. */
  static List<EvalGolden> all() {
    List<EvalGolden> runs = new ArrayList<>();
    for (Kind k : Kind.values()) {
      runs.add(new EvalGolden(k, WhitespaceMode.UNICODE));
      if (k.testsWhitespace) {
        runs.add(new EvalGolden(k, WhitespaceMode.LEGACY));
      }
    }
    return runs;
  }

  @Override
  public String annotator() {
    return "evaluate";
  }

  @Override
  public List<ModelRef> models() {
    return List.of();
  }

  @Override
  public String annotatorClass() {
    return kind.evaluatorClass.getName();
  }

  @Override
  public String method() {
    return kind == Kind.CROSS_VALIDATION ? "next" : "evaluateSample";
  }

  @Override
  public Map<String, String> config() {
    return Whitespace.config(whitespace);
  }

  @Override
  public Set<String> inputSets() {
    return Set.of(SET);
  }

  @Override
  public boolean accepts(Input input) {
    return input.set().equals(SET) && input.id().startsWith(PAIRS + kind.format + "-");
  }

  @Override
  public String modelDir() {
    return kind.dir + Whitespace.dirSuffix(whitespace);
  }

  @Override
  public List<Layer> layers(Input input) throws AnnotatorException {
    return Whitespace.in(whitespace, () -> run(input));
  }

  private ObjectStream<?> open(Input input) throws IOException {
    ObjectStream<String> lines =
        new PlainTextByLineStream(() -> new ByteArrayInputStream(input.bytes()), StandardCharsets.UTF_8);
    return switch (kind.format) {
      case "wordtag" -> new WordTagSampleStream(lines);
      case "lemma" -> new LemmaSampleStream(lines);
      case "name" -> new NameSampleDataStream(lines);
      case "chunk" -> new ChunkSampleStream(lines);
      case "token" -> new TokenSampleStream(lines);
      case "sentence" -> new SentenceSampleStream(lines);
      case "doccat" -> new DocumentSampleStream(lines);
      case "language" -> new LanguageDetectorSampleStream(lines);
      default -> throw new IllegalStateException(kind.format);
    };
  }

  private List<Object> samples(Input input) {
    List<Object> samples = new ArrayList<>();
    try (ObjectStream<?> stream = open(input)) {
      Object sample;
      while ((sample = stream.read()) != null) {
        samples.add(sample);
      }
    } catch (IOException e) {
      throw new IllegalStateException(input.inputId() + ": the pairs do not read", e);
    }
    return samples;
  }

  private List<Layer> run(Input input) {
    List<Object> samples = samples(input);
    if (kind == Kind.CROSS_VALIDATION) {
      return List.of(folds(input, samples.size()));
    }
    if (samples.size() % 2 != 0) {
      throw new IllegalStateException(input.inputId() + ": an odd number of samples");
    }
    Object[] prediction = new Object[1];
    String[] heard = new String[1];
    Evaluator<?> evaluator = evaluator(prediction, heard);

    FormatGolden.Columns verdicts = new FormatGolden.Columns();
    FormatGolden.Columns spanScores = new FormatGolden.Columns();
    List<Long> after = new ArrayList<>();
    List<double[]> scores = new ArrayList<>();
    List<Long> counts = new ArrayList<>();
    List<double[]> spanValues = new ArrayList<>();
    for (int pair = 0; pair < samples.size() / 2; pair++) {
      Object reference = samples.get(2 * pair);
      prediction[0] = samples.get(2 * pair + 1);
      heard[0] = null;
      try {
        evaluate(evaluator, reference);
        verdicts.u32("sample", pair).str("verdict", heard[0]).str("class", "").str("message", "");
      } catch (RuntimeException e) {
        String message = e.getMessage() == null ? "" : e.getMessage();
        verdicts.u32("sample", pair).str("verdict", "error").str("class", e.getClass().getName())
            .str("message", message);
      }
      after.add((long) pair);
      if (kind.keepsFMeasure()) {
        FMeasure f = fmeasure(evaluator);
        scores.add(new double[] {f.getPrecisionScore(), f.getRecallScore(), f.getFMeasure()});
      } else {
        scores.add(new double[] {accuracy(evaluator)});
        counts.add(count(evaluator));
      }
      Span[][] spans = spans(reference, prediction[0]);
      if (spans != null) {
        spanScores.u32("sample", pair);
        spanValues.add(new double[] {FMeasure.precision(spans[0], spans[1]), FMeasure.recall(spans[0], spans[1])});
      }
    }

    List<Layer> layers = new ArrayList<>();
    layers.add(verdicts.layer("turbo:verdicts", "document", "sample", "verdict", "class", "message"));
    long[] afterValues = after.stream().mapToLong(Long::longValue).toArray();
    if (kind.keepsFMeasure()) {
      layers.add(new Layer("turbo:fmeasure", "document", new Column.U32("after", afterValues),
          f64("precision", scores, 0), f64("recall", scores, 1), f64("fmeasure", scores, 2)));
    } else {
      layers.add(new Layer("turbo:accuracy", "document", new Column.U32("after", afterValues),
          f64("accuracy", scores, 0),
          new Column.U32("count", counts.stream().mapToLong(Long::longValue).toArray())));
    }
    if (kind == Kind.NAME || kind == Kind.TOKEN) {
      long[] pairs = new long[spanValues.size()];
      for (int i = 0; i < pairs.length; i++) {
        pairs[i] = i;
      }
      layers.add(new Layer("turbo:span-scores", "document", new Column.U32("sample", pairs),
          f64("precision", spanValues, 0), f64("recall", spanValues, 1)));
    }
    return layers;
  }

  private static Column f64(String name, List<double[]> rows, int column) {
    double[] values = new double[rows.size()];
    for (int i = 0; i < values.length; i++) {
      values[i] = rows.get(i)[column];
    }
    return new Column.F64(name, values);
  }

  @SuppressWarnings("unchecked")
  private static void evaluate(Evaluator<?> evaluator, Object reference) {
    ((Evaluator<Object>) evaluator).evaluateSample(reference);
  }

  private Span[][] spans(Object reference, Object prediction) {
    return switch (kind) {
      case NAME -> new Span[][] {((NameSample) reference).getNames(), ((NameSample) prediction).getNames()};
      case TOKEN -> new Span[][] {((TokenSample) reference).getTokenSpans(), ((TokenSample) prediction).getTokenSpans()};
      default -> null;
    };
  }

  private FMeasure fmeasure(Evaluator<?> evaluator) {
    return switch (kind) {
      case NAME -> ((TokenNameFinderEvaluator) evaluator).getFMeasure();
      case CHUNK -> ((ChunkerEvaluator) evaluator).getFMeasure();
      case TOKEN -> ((TokenizerEvaluator) evaluator).getFMeasure();
      case SENTENCE -> ((SentenceDetectorEvaluator) evaluator).getFMeasure();
      default -> throw new IllegalStateException(kind.name());
    };
  }

  private double accuracy(Evaluator<?> evaluator) {
    return switch (kind) {
      case POS -> ((POSEvaluator) evaluator).getWordAccuracy();
      case LEMMA -> ((LemmatizerEvaluator) evaluator).getWordAccuracy();
      case DOCCAT -> ((DocumentCategorizerEvaluator) evaluator).getAccuracy();
      case LANGUAGE -> ((LanguageDetectorEvaluator) evaluator).getAccuracy();
      default -> throw new IllegalStateException(kind.name());
    };
  }

  private long count(Evaluator<?> evaluator) {
    return switch (kind) {
      case POS -> ((POSEvaluator) evaluator).getWordCount();
      case LEMMA -> ((LemmatizerEvaluator) evaluator).getWordCount();
      case DOCCAT -> ((DocumentCategorizerEvaluator) evaluator).getDocumentCount();
      case LANGUAGE -> ((LanguageDetectorEvaluator) evaluator).getDocumentCount();
      default -> throw new IllegalStateException(kind.name());
    };
  }

  /** The evaluator over a stand-in component that answers with {@code prediction[0]}, its monitor writing to {@code heard[0]}. */
  private Evaluator<?> evaluator(Object[] prediction, String[] heard) {
    return switch (kind) {
      case POS -> new POSEvaluator(stand(POSTagger.class, prediction), monitor(POSTaggerEvaluationMonitor.class, heard));
      case LEMMA ->
          new LemmatizerEvaluator(stand(Lemmatizer.class, prediction), monitor(LemmatizerEvaluationMonitor.class, heard));
      case NAME -> new TokenNameFinderEvaluator(stand(TokenNameFinder.class, prediction),
          monitor(TokenNameFinderEvaluationMonitor.class, heard));
      case CHUNK -> new ChunkerEvaluator(stand(Chunker.class, prediction), monitor(ChunkerEvaluationMonitor.class, heard));
      case TOKEN ->
          new TokenizerEvaluator(stand(Tokenizer.class, prediction), monitor(TokenizerEvaluationMonitor.class, heard));
      case SENTENCE -> new SentenceDetectorEvaluator(stand(SentenceDetector.class, prediction),
          monitor(SentenceDetectorEvaluationMonitor.class, heard));
      case DOCCAT -> new DocumentCategorizerEvaluator(stand(DocumentCategorizer.class, prediction),
          monitor(DoccatEvaluationMonitor.class, heard));
      case LANGUAGE -> new LanguageDetectorEvaluator(stand(LanguageDetector.class, prediction),
          monitor(LanguageDetectorEvaluationMonitor.class, heard));
      case CROSS_VALIDATION -> throw new IllegalStateException("no evaluator");
    };
  }

  /** A component of interface {@code type} that answers every call it is evaluated with from {@code prediction[0]}. */
  private static <T> T stand(Class<T> type, Object[] prediction) {
    InvocationHandler handler = (proxy, method, args) -> {
      Object p = prediction[0];
      return switch (method.getName()) {
        case "tag" -> ((POSSample) p).getTags();
        case "lemmatize" -> answer(method, ((LemmaSample) p).getLemmas());
        case "find" -> ((NameSample) p).getNames();
        case "clearAdaptiveData" -> null;
        case "chunk" -> answer(method, ((ChunkSample) p).getPreds());
        case "tokenizePos" -> ((TokenSample) p).getTokenSpans();
        case "sentPosDetect" -> ((SentenceSample) p).getSentences();
        case "categorize" -> answer(method, new double[0]);
        case "getBestCategory" -> ((DocumentSample) p).getCategory();
        case "predictLanguage" -> new Language(((LanguageSample) p).language().getLang());
        default -> objectMethod(proxy, method, args);
      };
    };
    return type.cast(Proxy.newProxyInstance(type.getClassLoader(), new Class<?>[] {type}, handler));
  }

  /** {@code value} when {@code method} returns its type: the overload the evaluator calls. */
  private static Object answer(Method method, Object value) {
    if (!method.getReturnType().isInstance(value)) {
      throw new UnsupportedOperationException(method.toString());
    }
    return value;
  }

  private static Object objectMethod(Object proxy, Method method, Object[] args) {
    return switch (method.getName()) {
      case "toString" -> "stand-in";
      case "hashCode" -> System.identityHashCode(proxy);
      case "equals" -> proxy == args[0];
      default -> throw new UnsupportedOperationException(method.toString());
    };
  }

  /** An evaluation monitor of interface {@code type} that writes what it heard to {@code heard[0]}. */
  private static <T> T monitor(Class<T> type, String[] heard) {
    InvocationHandler handler = (proxy, method, args) -> switch (method.getName()) {
      case "correctlyClassified" -> {
        heard[0] = "correct";
        yield null;
      }
      case "misclassified" -> {
        heard[0] = "misclassified";
        yield null;
      }
      default -> objectMethod(proxy, method, args);
    };
    return type.cast(Proxy.newProxyInstance(type.getClassLoader(), new Class<?>[] {type}, handler));
  }

  private static Layer folds(Input input, int count) {
    FormatGolden.Columns c = new FormatGolden.Columns();
    List<Integer> places = new ArrayList<>();
    for (int i = 0; i < count; i++) {
      places.add(i);
    }
    try {
      for (int partitions = 1; partitions <= MAX_PARTITIONS; partitions++) {
        CrossValidationPartitioner<Integer> partitioner = new CrossValidationPartitioner<>(places, partitions);
        int fold = 0;
        while (partitioner.hasNext()) {
          CrossValidationPartitioner.TrainingSampleStream<Integer> training = partitioner.next();
          Integer place;
          while ((place = training.read()) != null) {
            c.u32("partitions", partitions).u32("fold", fold).str("part", "train").u32("sample", place);
          }
          ObjectStream<Integer> test = training.getTestSampleStream();
          while ((place = test.read()) != null) {
            c.u32("partitions", partitions).u32("fold", fold).str("part", "test").u32("sample", place);
          }
          fold++;
        }
      }
    } catch (IOException e) {
      throw new IllegalStateException(input.inputId() + ": the partitions do not read", e);
    }
    return c.layer("turbo:folds", "document", "partitions", "fold", "part", "sample");
  }
}
