package ai.pipestream.turbo.goldens;

import de.hhn.mi.configuration.KernelType;
import de.hhn.mi.configuration.SvmConfigurationImpl;
import de.hhn.mi.configuration.SvmType;
import de.hhn.mi.domain.SvmDocument;
import de.hhn.mi.domain.SvmFeature;
import java.io.ByteArrayInputStream;
import java.io.IOException;
import java.lang.reflect.InvocationTargetException;
import java.lang.reflect.Method;
import java.nio.charset.StandardCharsets;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.TreeMap;
import libsvm.svm;
import libsvm.svm_model;
import libsvm.svm_node;
import opennlp.tools.doccat.BagOfWordsFeatureGenerator;
import opennlp.tools.doccat.DocumentSample;
import opennlp.tools.doccat.DocumentSampleStream;
import opennlp.tools.doccat.FeatureGenerator;
import opennlp.tools.doccat.NGramFeatureGenerator;
import opennlp.tools.ml.libsvm.doccat.DocumentCategorizerSVM;
import opennlp.tools.ml.libsvm.doccat.FeatureSelectionStrategy;
import opennlp.tools.ml.libsvm.doccat.SvmDoccatConfiguration;
import opennlp.tools.ml.libsvm.doccat.SvmDoccatModel;
import opennlp.tools.ml.libsvm.doccat.TermWeightingStrategy;
import opennlp.tools.tokenize.WhitespaceTokenizer;
import opennlp.tools.util.InvalidFormatException;
import opennlp.tools.util.ObjectStream;
import opennlp.tools.util.PlainTextByLineStream;
import opennlp.tools.util.Span;

/**
 * The SVM document categorizer of {@code opennlp-ml-libsvm}: models trained with
 * {@link DocumentCategorizerSVM#train} on the corpora of the {@code svm-doccat-train} set, exported
 * as the files turbo-doccat reads, and {@code categorize} on the documents of the
 * {@code svm-doccat} set. The golden directory is {@code svm-doccat/<run>}.
 *
 * <p>Each run trains one model on one corpus (one sample per line, the category, a space and
 * the text, read with {@link DocumentSampleStream}) with its own term weighting, feature
 * selection, scaling, SVM type and kernel, and probability estimates on. LIBSVM's random numbers
 * (the probability cross validation) are seeded before training, so the model is the same on
 * every generation. The model is trained once, on the first golden that needs it.</p>
 *
 * <p>The golden of the run's corpus ({@link Train}, method {@code train}) holds the model, both
 * layers anchored on the document with one {@code line} per row:</p>
 * <ul>
 *   <li>{@code turbo:svm-model}: the LIBSVM model as {@code svm.svm_save_model} writes it.</li>
 *   <li>{@code turbo:svm-sidecar}: the rest of the model ({@link SvmDoccatExport#sidecar}).</li>
 * </ul>
 *
 * <p>The golden of a document (method {@code categorize}) holds, after {@code opennlp:tokens}
 * (the {@link WhitespaceTokenizer} tokens the categorizer is given), layers anchored on the
 * document:</p>
 * <ul>
 *   <li>{@code turbo:svm-vector}: the feature vector {@code categorize} hands LIBSVM, its
 *   {@code index} and {@code value} per row.</li>
 *   <li>{@code turbo:svm-decisions}: {@code svm_predict_values} on that vector, one row per class
 *   pair in LIBSVM's order: the {@code first} and {@code second} category and the decision
 *   {@code value}.</li>
 *   <li>{@code turbo:svm-vote}: the {@code category} the pairwise vote picks.</li>
 *   <li>{@code turbo:categories}: every category with the probability {@code categorize} gives
 *   it, best first ({@code Double.compare}, ties in category order).</li>
 * </ul>
 */
final class SvmDoccatGolden implements AnnotatorGolden {

  static final String ANNOTATOR = "svm-doccat";
  /** The documents categorized. */
  static final String INPUT_SET = "svm-doccat";
  /** The corpora the models are trained on. */
  static final String TRAIN_SET = "svm-doccat-train";
  /** The seed of LIBSVM's random numbers before each training. */
  static final long SEED = 1;
  /** The language code the models are trained with. */
  static final String LANGUAGE = "eng";

  /** One model: what it is trained on and with. */
  record Run(String name, String corpus, TermWeightingStrategy weighting, FeatureSelectionStrategy selection,
      int maxFeatures, SvmType type, KernelType kernel, int degree, double gamma, double coef0, double cost,
      boolean scale, double lower, double upper, boolean ngram) {

    /** A C-SVC run with the linear kernel, scaled to [0, 1] and every feature kept. */
    static Run linear(String name, String corpus, TermWeightingStrategy w) {
      return new Run(name, corpus, w, FeatureSelectionStrategy.NONE, -1, SvmType.C_SVC, KernelType.LINEAR, 3, 0, 0,
          1, true, 0, 1, false);
    }

    Run selection(FeatureSelectionStrategy s, int max) {
      return new Run(name, corpus, weighting, s, max, type, kernel, degree, gamma, coef0, cost, scale, lower, upper,
          ngram);
    }

    Run kernel(SvmType t, KernelType k, int d, double g, double c0) {
      return new Run(name, corpus, weighting, selection, maxFeatures, t, k, d, g, c0, cost, scale, lower, upper,
          ngram);
    }

    Run cost(double c) {
      return new Run(name, corpus, weighting, selection, maxFeatures, type, kernel, degree, gamma, coef0, c, scale,
          lower, upper, ngram);
    }

    Run scaling(boolean on, double lo, double hi) {
      return new Run(name, corpus, weighting, selection, maxFeatures, type, kernel, degree, gamma, coef0, cost, on,
          lo, hi, ngram);
    }

    Run withNgrams() {
      return new Run(name, corpus, weighting, selection, maxFeatures, type, kernel, degree, gamma, coef0, cost,
          scale, lower, upper, true);
    }

    /** The generators the model is trained and run with. */
    FeatureGenerator[] generators() {
      if (!ngram) {
        return new FeatureGenerator[] {new BagOfWordsFeatureGenerator()};
      }
      try {
        return new FeatureGenerator[] {new BagOfWordsFeatureGenerator(), new NGramFeatureGenerator(2, 3)};
      } catch (InvalidFormatException e) {
        throw new IllegalStateException("2 to 3 is an n-gram range", e);
      }
    }

    SvmDoccatConfiguration configuration() {
      SvmConfigurationImpl.Builder svm = new SvmConfigurationImpl.Builder();
      svm.setSvmType(type).setKernelType(kernel).setDegree(degree).setGamma(gamma).setCoef0(coef0).setCost(cost)
          .setProbability(true);
      SvmDoccatConfiguration.Builder b = new SvmDoccatConfiguration.Builder().setSvmConfiguration(svm.build())
          .setTermWeightingStrategy(weighting).setFeatureSelectionStrategy(selection).setMaxFeatures(maxFeatures)
          .setScaleFeatures(scale);
      if (scale) {
        b.setScaleRange(lower, upper);
      }
      return b.build();
    }

    /** The {@code config} records: the run and everything it is trained with. */
    Map<String, String> config() {
      Map<String, String> c = new TreeMap<>();
      c.put("run", name);
      c.put("corpus", TRAIN_SET + "/" + corpus);
      c.put("language", LANGUAGE);
      c.put("seed", Long.toString(SEED));
      c.put("weighting", weighting.name());
      c.put("selection", selection.name());
      c.put("max-features", Integer.toString(maxFeatures));
      c.put("svm-type", type.name());
      c.put("kernel", kernel.name());
      c.put("degree", Integer.toString(degree));
      c.put("gamma", Double.toString(gamma));
      c.put("coef0", Double.toString(coef0));
      c.put("cost", Double.toString(cost));
      c.put("scaling", scale ? Double.toString(lower) + " " + Double.toString(upper) : "none");
      c.put("generators", ngram ? "bag-of-words ngram(2,3)" : "bag-of-words");
      return c;
    }
  }

  /** The runs: every weighting with every selection, then each kernel, SVM type and option. */
  static List<Run> runs() {
    List<Run> runs = new ArrayList<>();
    String[][] weightings = {{"binary", "BINARY"}, {"tf", "TERM_FREQUENCY"}, {"tfidf", "TF_IDF"},
        {"lognorm", "LOG_NORMALIZED_TF"}};
    String[][] selections = {{"none", "NONE"}, {"ig", "INFORMATION_GAIN"}, {"chi2", "CHI_SQUARE"},
        {"tf", "TERM_FREQUENCY"}, {"df", "DOCUMENT_FREQUENCY"}};
    for (String[] w : weightings) {
      for (String[] s : selections) {
        FeatureSelectionStrategy strategy = FeatureSelectionStrategy.valueOf(s[1]);
        runs.add(Run.linear("topics-linear-" + w[0] + "-" + s[0], "topics", TermWeightingStrategy.valueOf(w[1]))
            .selection(strategy, strategy == FeatureSelectionStrategy.NONE ? -1 : 40));
      }
    }
    runs.add(Run.linear("topics-rbf-tfidf", "topics", TermWeightingStrategy.TF_IDF)
        .kernel(SvmType.C_SVC, KernelType.RBF, 3, 0, 0));
    runs.add(Run.linear("topics-poly-tf", "topics", TermWeightingStrategy.TERM_FREQUENCY)
        .kernel(SvmType.C_SVC, KernelType.POLYNOMIAL, 3, 0.5, 1));
    runs.add(Run.linear("topics-sigmoid-binary", "topics", TermWeightingStrategy.BINARY)
        .kernel(SvmType.C_SVC, KernelType.SIGMOID, 3, 0.05, 0));
    runs.add(Run.linear("topics-sigmoid-tfidf", "topics", TermWeightingStrategy.TF_IDF)
        .kernel(SvmType.C_SVC, KernelType.SIGMOID, 3, 0.05, -0.1));
    runs.add(Run.linear("topics-nu-linear-tfidf", "topics", TermWeightingStrategy.TF_IDF)
        .kernel(SvmType.NU_SVC, KernelType.LINEAR, 3, 0, 0));
    runs.add(Run.linear("topics-nu-rbf-lognorm", "topics", TermWeightingStrategy.LOG_NORMALIZED_TF)
        .kernel(SvmType.NU_SVC, KernelType.RBF, 3, 0, 0));
    runs.add(Run.linear("topics-linear-tfidf-noscale", "topics", TermWeightingStrategy.TF_IDF)
        .scaling(false, 0, 1));
    runs.add(Run.linear("topics-rbf-tfidf-scale-1", "topics", TermWeightingStrategy.TF_IDF)
        .kernel(SvmType.C_SVC, KernelType.RBF, 3, 0, 0).scaling(true, -1, 1)
        .selection(FeatureSelectionStrategy.CHI_SQUARE, 60));
    runs.add(Run.linear("topics-linear-lognorm-ngram", "topics", TermWeightingStrategy.LOG_NORMALIZED_TF)
        .withNgrams());
    runs.add(Run.linear("topics-linear-binary-c10", "topics", TermWeightingStrategy.BINARY).cost(10)
        .selection(FeatureSelectionStrategy.INFORMATION_GAIN, 25));
    runs.add(Run.linear("pair-linear-tfidf", "pair", TermWeightingStrategy.TF_IDF));
    runs.add(Run.linear("pair-rbf-binary", "pair", TermWeightingStrategy.BINARY)
        .kernel(SvmType.C_SVC, KernelType.RBF, 3, 0, 0));
    return runs;
  }

  /** For every run, the golden of its corpus and the golden of the documents, over one model. */
  static List<AnnotatorGolden> all(Path inputs) {
    List<AnnotatorGolden> out = new ArrayList<>();
    for (Run run : runs()) {
      Trained trained = new Trained(run, inputs.resolve(TRAIN_SET).resolve(run.corpus() + ".txt"));
      out.add(new Train(trained));
      out.add(new SvmDoccatGolden(trained));
    }
    return out;
  }

  /** A run's model, trained on first use. */
  static final class Trained {

    final Run run;
    private final Path corpus;
    private SvmDoccatModel model;
    private DocumentCategorizerSVM categorizer;
    private svm_model svmModel;
    private Method toDocument;

    Trained(Run run, Path corpus) {
      this.run = run;
      this.corpus = corpus;
    }

    private synchronized void train() {
      if (model != null) {
        return;
      }
      Input input;
      try {
        input = Input.read(corpus, TRAIN_SET, run.corpus());
      } catch (IOException e) {
        throw new IllegalStateException(corpus + ": the training corpus could not be read", e);
      }
      FeatureGenerator[] generators = run.generators();
      try (ObjectStream<DocumentSample> samples = new DocumentSampleStream(new PlainTextByLineStream(
          () -> new ByteArrayInputStream(input.bytes()), StandardCharsets.UTF_8))) {
        svm.rand.setSeed(SEED);
        model = DocumentCategorizerSVM.train(LANGUAGE, samples, run.configuration(), generators);
      } catch (InvalidFormatException e) {
        throw new IllegalStateException(input.inputId() + ": not a document categorizer corpus", e);
      } catch (IOException e) {
        throw new IllegalStateException(input.inputId() + ": training failed", e);
      }
      categorizer = new DocumentCategorizerSVM(model, generators);
      svmModel = SvmDoccatExport.nativeModel(model);
      try {
        toDocument = DocumentCategorizerSVM.class.getDeclaredMethod("textToSvmDocument", String[].class, Map.class);
      } catch (NoSuchMethodException e) {
        throw new IllegalStateException("DocumentCategorizerSVM has no textToSvmDocument", e);
      }
      toDocument.setAccessible(true);
    }

    SvmDoccatModel model() {
      train();
      return model;
    }

    DocumentCategorizerSVM categorizer() {
      train();
      return categorizer;
    }

    svm_model svmModel() {
      train();
      return svmModel;
    }

    /** The document {@code categorize} hands LIBSVM for {@code tokens}. */
    SvmDocument document(String[] tokens) {
      train();
      try {
        return (SvmDocument) toDocument.invoke(categorizer, tokens, Collections.emptyMap());
      } catch (InvocationTargetException e) {
        throw SvmDoccatExport.unwrap(e);
      } catch (IllegalAccessException e) {
        throw new IllegalStateException(e);
      }
    }
  }

  private final Trained trained;

  private SvmDoccatGolden(Trained trained) {
    this.trained = trained;
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
    return trained.run.name();
  }

  @Override
  public String annotatorClass() {
    return DocumentCategorizerSVM.class.getName();
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
    DocumentCategorizerSVM categorizer = trained.categorizer();
    double[] probabilities = AnnotatorException.call(() -> categorizer.categorize(tokens));

    SvmDocument document = AnnotatorException.call(() -> trained.document(tokens));
    List<SvmFeature> features = document.getSvmFeatures();
    svm_node[] x = new svm_node[features.size()];
    long[] index = new long[x.length];
    double[] value = new double[x.length];
    for (int i = 0; i < x.length; i++) {
      x[i] = new svm_node();
      x[i].index = features.get(i).getIndex();
      x[i].value = features.get(i).getValue();
      index[i] = x[i].index;
      value[i] = x[i].value;
    }
    layers.add(new Layer("turbo:svm-vector", "document", new Column.U32("index", index),
        new Column.F64("value", value)));

    svm_model m = trained.svmModel();
    Map<Integer, String> names = trained.model().getIndexToCategory();
    int k = m.nr_class;
    double[] decisions = new double[k * (k - 1) / 2];
    double vote = svm.svm_predict_values(m, x, decisions);
    String[] first = new String[decisions.length];
    String[] second = new String[decisions.length];
    int p = 0;
    for (int i = 0; i < k; i++) {
      for (int j = i + 1; j < k; j++) {
        first[p] = names.get(m.label[i]);
        second[p] = names.get(m.label[j]);
        p++;
      }
    }
    layers.add(new Layer("turbo:svm-decisions", "document", new Column.Str("first", first),
        new Column.Str("second", second), new Column.F64("value", decisions)));
    layers.add(new Layer("turbo:svm-vote", "document",
        new Column.Str("category", new String[] {names.get((int) vote)})));

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

  /** The golden of a run's corpus: the model it trained, as the two files turbo-doccat reads. */
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
      return List.of();
    }

    @Override
    public String modelDir() {
      return trained.run.name();
    }

    @Override
    public String annotatorClass() {
      return DocumentCategorizerSVM.class.getName();
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
      return input.set().equals(TRAIN_SET) && input.id().equals(trained.run.corpus());
    }

    @Override
    public List<Layer> layers(Input input) {
      String model = SvmDoccatExport.libsvmText(trained.svmModel());
      String sidecar = SvmDoccatExport.sidecar(trained.model(), trained.run.generators());
      return List.of(new Layer("turbo:svm-model", "document", new Column.Str("line", lines(model))),
          new Layer("turbo:svm-sidecar", "document", new Column.Str("line", lines(sidecar))));
    }

    /** The lines of a file that ends each line with {@code \n}. */
    static String[] lines(String text) {
      if (!text.endsWith("\n")) {
        throw new IllegalStateException("an exported file does not end its last line");
      }
      List<String> out = new ArrayList<>();
      int start = 0;
      for (int i = text.indexOf('\n'); i >= 0; i = text.indexOf('\n', start)) {
        out.add(text.substring(start, i));
        start = i + 1;
      }
      return out.toArray(new String[0]);
    }
  }
}
