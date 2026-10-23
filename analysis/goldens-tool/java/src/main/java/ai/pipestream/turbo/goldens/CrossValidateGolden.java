package ai.pipestream.turbo.goldens;

import java.io.ByteArrayInputStream;
import java.io.IOException;
import java.lang.reflect.Field;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.TreeMap;
import opennlp.tools.langdetect.LanguageDetectorCrossValidator;
import opennlp.tools.langdetect.LanguageDetectorEvaluationMonitor;
import opennlp.tools.langdetect.LanguageDetectorEvaluator;
import opennlp.tools.langdetect.LanguageDetectorEventStream;
import opennlp.tools.langdetect.LanguageDetectorFactory;
import opennlp.tools.langdetect.LanguageDetectorME;
import opennlp.tools.langdetect.LanguageDetectorModel;
import opennlp.tools.langdetect.LanguageDetectorSampleStream;
import opennlp.tools.langdetect.LanguageSample;
import opennlp.tools.ml.model.ChecksumEventStream;
import opennlp.tools.ml.model.Event;
import opennlp.tools.namefind.NameFinderEventStream;
import opennlp.tools.namefind.NameFinderME;
import opennlp.tools.namefind.NameSample;
import opennlp.tools.namefind.NameSampleDataStream;
import opennlp.tools.namefind.TokenNameFinderCrossValidator;
import opennlp.tools.namefind.TokenNameFinderEvaluationMonitor;
import opennlp.tools.namefind.TokenNameFinderEvaluator;
import opennlp.tools.namefind.TokenNameFinderFactory;
import opennlp.tools.namefind.TokenNameFinderModel;
import opennlp.tools.util.ObjectStream;
import opennlp.tools.util.ObjectStreamUtils;
import opennlp.tools.util.Parameters;
import opennlp.tools.util.PlainTextByLineStream;
import opennlp.tools.util.Span;
import opennlp.tools.util.TrainingParameters;
import opennlp.tools.util.eval.CrossValidationPartitioner;
import opennlp.tools.util.eval.FMeasure;
import opennlp.tools.util.eval.Mean;
import opennlp.tools.util.model.ModelUtil;

/**
 * The name finder's and the language detector's cross-validation, fold by fold:
 * {@code TokenNameFinderCrossValidator} and {@code LanguageDetectorCrossValidator} with 3 folds.
 * The golden directory is {@code cross-validate/<run>}.
 *
 * <p>The cross-validators keep their folds to themselves, so the generator runs each fold as they
 * do, with the same classes: the samples (for the name finder grouped into documents as
 * {@code NameToDocumentSampleStream} groups them, a new one at each sample that clears adaptive
 * data) split by {@link CrossValidationPartitioner}, the component trained on the training side
 * with {@code NameFinderME.train} or {@code LanguageDetectorME.train} and evaluated on the test side
 * with {@link TokenNameFinderEvaluator} or {@link LanguageDetectorEvaluator}. Then it runs the
 * cross-validator itself over the same samples and throws when its merged figures are not the
 * folds' merged, and checks every fold model's {@code Training-Eventhash} against the checksum of
 * the training events it reads. The counts of an {@link FMeasure} are private and read by
 * reflection.</p>
 *
 * <p>Runs: {@code namefind-bio} and {@code namefind-bilou} over {@code tagged/names} with the
 * factory {@code new TokenNameFinderFactory(null, Map.of(), codec)}, language {@code en}, no type
 * and empty parameters ({@code NameFinderME.train}'s defaults); {@code langdetect-default} and
 * {@code langdetect-cutoff1} over {@code langdetect/samples} with
 * {@link ModelUtil#createDefaultTrainingParameters()} and the run's cutoff. The {@code config}
 * records name the run, the folds and those options.</p>
 *
 * <p>Layers, all anchored on the document. Name finder runs: {@code turbo:name-cv-folds}, a row
 * per fold (documents and samples on each side, training events, their {@code crc32c}, the fold's
 * {@code selected}, {@code target} and {@code true-positives} counts, its precision, recall and
 * F-measure, and the test samples the monitors heard as correct); {@code turbo:name-cv-names}, a
 * row per predicted name (fold, test sample within the fold, span and type); and
 * {@code turbo:name-cv-merged}, the cross-validator's {@code getFMeasure()}. Language detector
 * runs: {@code turbo:language-cv-folds} (samples on each side, events, {@code crc32c}, correct,
 * the evaluator's document count and accuracy); {@code turbo:language-cv-predictions}, the
 * language predicted for each test sample; and {@code turbo:language-cv-merged}, the
 * cross-validator's {@code getDocumentCount()} and {@code getDocumentAccuracy()}.</p>
 */
final class CrossValidateGolden implements AnnotatorGolden {

  static final String ANNOTATOR = "cross-validate";
  static final int FOLDS = 3;

  private final String run;
  /** {@code null} for a language detector run. */
  private final NameEventGolden.Codec codec;
  private final int cutoff;

  private CrossValidateGolden(String run, NameEventGolden.Codec codec, int cutoff) {
    this.run = run;
    this.codec = codec;
    this.cutoff = cutoff;
  }

  static List<AnnotatorGolden> all() {
    return List.of(
        new CrossValidateGolden("namefind-bio", NameEventGolden.Codec.BIO, 0),
        new CrossValidateGolden("namefind-bilou", NameEventGolden.Codec.BILOU, 0),
        new CrossValidateGolden("langdetect-default", null, 5),
        new CrossValidateGolden("langdetect-cutoff1", null, 1));
  }

  private boolean names() {
    return codec != null;
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
    return names() ? TokenNameFinderCrossValidator.class.getName() : LanguageDetectorCrossValidator.class.getName();
  }

  @Override
  public String method() {
    return "evaluate";
  }

  @Override
  public Map<String, String> config() {
    Map<String, String> c = new TreeMap<>();
    c.put("run", run);
    c.put("folds", String.valueOf(FOLDS));
    if (names()) {
      c.put("language", "en");
      c.put("codec", codec.name);
    } else {
      for (Map.Entry<String, Object> e : languageParameters().getObjectSettings().entrySet()) {
        c.put(e.getKey(), String.valueOf(e.getValue()));
      }
    }
    return c;
  }

  @Override
  public Set<String> inputSets() {
    return Set.of(names() ? Runs.TAGGED : Runs.LANGDETECT);
  }

  @Override
  public boolean accepts(Input input) {
    return inputSets().contains(input.set()) && input.id().equals(names() ? "names" : "samples");
  }

  @Override
  public List<Layer> layers(Input input) throws AnnotatorException {
    try {
      return names() ? nameLayers(input) : languageLayers(input);
    } catch (IOException | ReflectiveOperationException e) {
      throw new IllegalStateException(input.inputId() + ": cross-validation failed", e);
    }
  }

  private TrainingParameters languageParameters() {
    TrainingParameters p = ModelUtil.createDefaultTrainingParameters();
    p.put(Parameters.CUTOFF_PARAM, cutoff);
    return p;
  }

  private static <T> List<T> drain(ObjectStream<T> stream) throws IOException {
    List<T> out = new ArrayList<>();
    T t;
    while ((t = stream.read()) != null) {
      out.add(t);
    }
    return out;
  }

  /** Reads every event and returns the count and the CRC32C. */
  private static long[] checksum(ObjectStream<Event> events) throws IOException {
    ChecksumEventStream checked = new ChecksumEventStream(events);
    long n = 0;
    while (checked.read() != null) {
      n++;
    }
    return new long[] {n, checked.calculateChecksum()};
  }

  private static void checkHash(String what, String hash, long crc) {
    if (!String.valueOf(crc).equals(hash)) {
      throw new IllegalStateException(what + ": Training-Eventhash " + hash + " is not the events' " + crc);
    }
  }

  private static long count(FMeasure m, String field) throws ReflectiveOperationException {
    Field f = FMeasure.class.getDeclaredField(field);
    f.setAccessible(true);
    return f.getLong(m);
  }

  private static long[] counts(FMeasure m) throws ReflectiveOperationException {
    return new long[] {count(m, "selected"), count(m, "target"), count(m, "truePositive")};
  }

  private static <T> List<T> flatten(List<List<T>> documents) {
    List<T> out = new ArrayList<>();
    documents.forEach(out::addAll);
    return out;
  }

  private List<Layer> nameLayers(Input input) throws IOException, ReflectiveOperationException {
    List<NameSample> samples = drain(new NameSampleDataStream(new PlainTextByLineStream(
        () -> new ByteArrayInputStream(input.bytes()), StandardCharsets.UTF_8)));
    List<List<NameSample>> documents = new ArrayList<>();
    for (NameSample s : samples) {
      if (documents.isEmpty() || s.isClearAdaptiveDataSet()) {
        documents.add(new ArrayList<>());
      }
      documents.get(documents.size() - 1).add(s);
    }
    TokenNameFinderFactory factory = new TokenNameFinderFactory(null, Map.of(), codec.make.get());
    FormatGolden.Columns folds = new FormatGolden.Columns(Set.of("training-documents", "training-samples",
        "test-documents", "test-samples", "events", "crc32c", "selected", "target", "true-positives", "correct"));
    List<Double> precision = new ArrayList<>();
    List<Double> recall = new ArrayList<>();
    List<Double> fmeasure = new ArrayList<>();
    FormatGolden.Columns found = new FormatGolden.Columns();
    FMeasure merged = new FMeasure();
    CrossValidationPartitioner<List<NameSample>> partitioner = new CrossValidationPartitioner<>(documents, FOLDS);
    for (int fold = 0; partitioner.hasNext(); fold++) {
      CrossValidationPartitioner.TrainingSampleStream<List<NameSample>> training = partitioner.next();
      List<List<NameSample>> trainDocs = drain(training);
      List<List<NameSample>> testDocs = drain(training.getTestSampleStream());
      List<NameSample> train = flatten(trainDocs);
      List<NameSample> test = flatten(testDocs);
      TokenNameFinderFactory events = new TokenNameFinderFactory(null, Map.of(), codec.make.get());
      long[] sum = checksum(new NameFinderEventStream(ObjectStreamUtils.createObjectStream(train), null,
          events.createContextGenerator(), events.createSequenceCodec()));
      TokenNameFinderModel model = NameFinderME.train("en", null, ObjectStreamUtils.createObjectStream(train),
          new TrainingParameters(), factory);
      checkHash(run + " fold " + fold, model.getManifestProperty("Training-Eventhash"), sum[1]);
      List<NameSample> predicted = new ArrayList<>();
      long[] correct = {0};
      TokenNameFinderEvaluator evaluator = new TokenNameFinderEvaluator(new NameFinderME(model),
          new TokenNameFinderEvaluationMonitor() {
            @Override
            public void correctlyClassified(NameSample reference, NameSample prediction) {
              predicted.add(prediction);
              correct[0]++;
            }

            @Override
            public void misclassified(NameSample reference, NameSample prediction) {
              predicted.add(prediction);
            }
          });
      evaluator.evaluate(ObjectStreamUtils.createObjectStream(test));
      FMeasure f = evaluator.getFMeasure();
      merged.mergeInto(f);
      long[] c = counts(f);
      folds.u32("fold", fold).u32("training-documents", trainDocs.size()).u32("training-samples", train.size())
          .u32("test-documents", testDocs.size()).u32("test-samples", test.size()).u32("events", sum[0])
          .u32("crc32c", sum[1]).u32("selected", c[0]).u32("target", c[1]).u32("true-positives", c[2])
          .u32("correct", correct[0]);
      precision.add(f.getPrecisionScore());
      recall.add(f.getRecallScore());
      fmeasure.add(f.getFMeasure());
      for (int i = 0; i < predicted.size(); i++) {
        for (Span s : predicted.get(i).getNames()) {
          found.u32("fold", fold).u32("sample", i).u32("start", s.getStart()).u32("end", s.getEnd())
              .str("type", s.getType());
        }
      }
    }
    TokenNameFinderCrossValidator cv = new TokenNameFinderCrossValidator("en", null, new TrainingParameters(),
        new TokenNameFinderFactory(null, Map.of(), codec.make.get()));
    cv.evaluate(ObjectStreamUtils.createObjectStream(samples), FOLDS);
    FMeasure real = cv.getFMeasure();
    if (!java.util.Arrays.equals(counts(real), counts(merged))) {
      throw new IllegalStateException(run + ": the cross-validator's counts " + real + " are not the folds' " + merged);
    }
    Layer foldLayer = folds.layer("turbo:name-cv-folds", "document", "fold", "training-documents",
        "training-samples", "test-documents", "test-samples", "events", "crc32c", "selected", "target",
        "true-positives", "correct");
    List<Column> columns = new ArrayList<>(foldLayer.columns());
    columns.add(new Column.F64("precision", precision.stream().mapToDouble(Double::doubleValue).toArray()));
    columns.add(new Column.F64("recall", recall.stream().mapToDouble(Double::doubleValue).toArray()));
    columns.add(new Column.F64("f-measure", fmeasure.stream().mapToDouble(Double::doubleValue).toArray()));
    long[] m = counts(real);
    return List.of(new Layer(foldLayer.id(), foldLayer.anchor(), columns),
        found.layer("turbo:name-cv-names", "document", "fold", "sample", "start", "end", "type"),
        new Layer("turbo:name-cv-merged", "document",
            new Column.U32("selected", new long[] {m[0]}), new Column.U32("target", new long[] {m[1]}),
            new Column.U32("true-positives", new long[] {m[2]}),
            new Column.F64("precision", new double[] {real.getPrecisionScore()}),
            new Column.F64("recall", new double[] {real.getRecallScore()}),
            new Column.F64("f-measure", new double[] {real.getFMeasure()})));
  }

  private List<Layer> languageLayers(Input input) throws IOException {
    List<LanguageSample> samples = drain(new LanguageDetectorSampleStream(new PlainTextByLineStream(
        () -> new ByteArrayInputStream(input.bytes()), StandardCharsets.UTF_8)));
    LanguageDetectorFactory factory = new LanguageDetectorFactory();
    FormatGolden.Columns folds = new FormatGolden.Columns(Set.of("training-samples", "test-samples", "events",
        "crc32c", "correct", "documents"));
    List<Double> accuracy = new ArrayList<>();
    FormatGolden.Columns predictions = new FormatGolden.Columns();
    Mean merged = new Mean();
    CrossValidationPartitioner<LanguageSample> partitioner = new CrossValidationPartitioner<>(samples, FOLDS);
    for (int fold = 0; partitioner.hasNext(); fold++) {
      CrossValidationPartitioner.TrainingSampleStream<LanguageSample> training = partitioner.next();
      List<LanguageSample> train = drain(training);
      List<LanguageSample> test = drain(training.getTestSampleStream());
      long[] sum = checksum(new LanguageDetectorEventStream(ObjectStreamUtils.createObjectStream(train),
          new LanguageDetectorFactory().getContextGenerator()));
      LanguageDetectorModel model = LanguageDetectorME.train(ObjectStreamUtils.createObjectStream(train),
          languageParameters(), factory);
      checkHash(run + " fold " + fold, model.getManifestProperty("Training-Eventhash"), sum[1]);
      List<String> predicted = new ArrayList<>();
      long[] correct = {0};
      LanguageDetectorEvaluator evaluator = new LanguageDetectorEvaluator(new LanguageDetectorME(model),
          new LanguageDetectorEvaluationMonitor() {
            @Override
            public void correctlyClassified(LanguageSample reference, LanguageSample prediction) {
              predicted.add(prediction.language().getLang());
              correct[0]++;
            }

            @Override
            public void misclassified(LanguageSample reference, LanguageSample prediction) {
              predicted.add(prediction.language().getLang());
            }
          });
      evaluator.evaluate(ObjectStreamUtils.createObjectStream(test));
      merged.add(evaluator.getAccuracy(), evaluator.getDocumentCount());
      folds.u32("fold", fold).u32("training-samples", train.size()).u32("test-samples", test.size())
          .u32("events", sum[0]).u32("crc32c", sum[1]).u32("correct", correct[0])
          .u32("documents", evaluator.getDocumentCount());
      accuracy.add(evaluator.getAccuracy());
      for (int i = 0; i < predicted.size(); i++) {
        predictions.u32("fold", fold).u32("sample", i).str("language", predicted.get(i));
      }
    }
    LanguageDetectorCrossValidator cv = new LanguageDetectorCrossValidator(languageParameters(),
        new LanguageDetectorFactory());
    cv.evaluate(ObjectStreamUtils.createObjectStream(samples), FOLDS);
    if (Double.doubleToRawLongBits(cv.getDocumentAccuracy()) != Double.doubleToRawLongBits(merged.mean())
        || cv.getDocumentCount() != merged.count()) {
      throw new IllegalStateException(run + ": the cross-validator's accuracy " + cv.getDocumentAccuracy()
          + " over " + cv.getDocumentCount() + " is not the folds' " + merged.mean() + " over " + merged.count());
    }
    Layer foldLayer = folds.layer("turbo:language-cv-folds", "document", "fold", "training-samples",
        "test-samples", "events", "crc32c", "correct", "documents");
    List<Column> columns = new ArrayList<>(foldLayer.columns());
    columns.add(new Column.F64("accuracy", accuracy.stream().mapToDouble(Double::doubleValue).toArray()));
    return List.of(new Layer(foldLayer.id(), foldLayer.anchor(), columns),
        predictions.layer("turbo:language-cv-predictions", "document", "fold", "sample", "language"),
        new Layer("turbo:language-cv-merged", "document",
            new Column.U32("documents", new long[] {cv.getDocumentCount()}),
            new Column.F64("accuracy", new double[] {cv.getDocumentAccuracy()})));
  }
}
