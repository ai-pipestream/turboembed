package ai.pipestream.turbo.goldens;

import java.io.IOException;
import java.nio.file.Path;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.TreeMap;
import opennlp.tools.langdetect.Language;
import opennlp.tools.langdetect.LanguageDetectorConfig;
import opennlp.tools.langdetect.LanguageDetectorME;
import opennlp.tools.langdetect.LanguageDetectorModel;
import opennlp.tools.langdetect.ProbingLanguageDetectionResult;

/**
 * The language detector's golden: {@code opennlp:language} with one row per language
 * {@link LanguageDetectorME#predictLanguages(CharSequence)} or
 * {@link LanguageDetectorME#probingPredictLanguages(CharSequence, LanguageDetectorConfig)} returns,
 * in its order, best first, and for a probing run {@code turbo:probing-length} with the number of
 * Unicode code points processed ({@link ProbingLanguageDetectionResult#getLength()}).
 */
final class LanguageGolden implements AnnotatorGolden {

  /**
   * The smaller-chunk probing configuration that exercises multi-chunk accumulation, early
   * stopping ({@code seenEnough}), and the {@code maxLength} cap on the {@code en}, {@code multi},
   * and {@code langdetect} inputs.
   */
  static final LanguageDetectorConfig PROBING_CHUNK_50 = chunk50Config();

  private static LanguageDetectorConfig chunk50Config() {
    LanguageDetectorConfig c = new LanguageDetectorConfig();
    c.setMaxLength(250);
    c.setChunkSize(50);
    c.setMinConsecImprovements(2);
    c.setMinDiff(0.20);
    return c;
  }

  private final ModelRef model;
  private final LanguageDetectorModel loaded;
  private final LanguageDetectorME detector;
  private final String modelDir;
  private final LanguageDetectorConfig probing;
  private final Map<String, String> config;
  private final Set<String> inputSets;

  private LanguageGolden(ModelRef model, LanguageDetectorModel loaded, String modelDir,
      LanguageDetectorConfig probing, Map<String, String> config, Set<String> inputSets) {
    this.model = model;
    this.loaded = loaded;
    this.detector = new LanguageDetectorME(loaded);
    this.modelDir = modelDir;
    this.probing = probing;
    this.config = Map.copyOf(config);
    this.inputSets = Set.copyOf(inputSets);
  }

  /** Loads the language detector model from {@code file}, hashing exactly the bytes loaded. */
  static LanguageGolden load(Path file, Set<String> inputSets) throws IOException {
    ModelFile f = ModelFile.read(file);
    return new LanguageGolden(f.ref(), new LanguageDetectorModel(f.stream()), f.ref().baseName(), null, Map.of(),
        inputSets);
  }

  /**
   * Runs {@link LanguageDetectorME#probingPredictLanguages(CharSequence, LanguageDetectorConfig)}
   * with {@link LanguageDetectorConfig#DEFAULT_LANGUAGE_DETECTOR_CONFIG} ({@code +probing}).
   */
  static LanguageGolden probing(Path file, Set<String> inputSets) throws IOException {
    return withProbing(file, "probing", LanguageDetectorConfig.DEFAULT_LANGUAGE_DETECTOR_CONFIG, inputSets);
  }

  /**
   * Runs {@link LanguageDetectorME#probingPredictLanguages(CharSequence, LanguageDetectorConfig)}
   * with {@link #PROBING_CHUNK_50} ({@code +probing-chunk50}).
   */
  static LanguageGolden probingChunk50(Path file, Set<String> inputSets) throws IOException {
    return withProbing(file, "probing-chunk50", PROBING_CHUNK_50, inputSets);
  }

  private static LanguageGolden withProbing(Path file, String suffix, LanguageDetectorConfig probing,
      Set<String> inputSets) throws IOException {
    ModelFile f = ModelFile.read(file);
    Map<String, String> c = new TreeMap<>();
    c.put("chunkSize", Integer.toString(probing.getChunkSize()));
    c.put("maxLength", Integer.toString(probing.getMaxLength()));
    c.put("minConsecImprovements", Integer.toString(probing.getMinConsecImprovements()));
    c.put("minDiff", Double.toString(probing.getMinDiff()));
    return new LanguageGolden(f.ref(), new LanguageDetectorModel(f.stream()), f.ref().baseName() + "+" + suffix,
        probing, c, inputSets);
  }

  /** The loaded model, which {@link LanguageDump} builds its recording model from. */
  LanguageDetectorModel loaded() {
    return loaded;
  }

  /** Whether this run calls {@code probingPredictLanguages} rather than {@code predictLanguages}. */
  boolean isProbing() {
    return probing != null;
  }

  @Override
  public String annotator() {
    return "langdetect";
  }

  @Override
  public List<ModelRef> models() {
    return List.of(model);
  }

  @Override
  public String modelDir() {
    return modelDir;
  }

  @Override
  public String annotatorClass() {
    return LanguageDetectorME.class.getName();
  }

  @Override
  public String method() {
    return probing == null ? "predictLanguages" : "probingPredictLanguages";
  }

  @Override
  public Map<String, String> config() {
    return config;
  }

  @Override
  public Set<String> inputSets() {
    return inputSets;
  }

  @Override
  public List<Layer> layers(Input input) throws AnnotatorException {
    return layers(detector, input);
  }

  /** The layers {@code detector} produces for the input; {@link LanguageDump} runs it on its recording detector. */
  List<Layer> layers(LanguageDetectorME detector, Input input) throws AnnotatorException {
    if (probing == null) {
      Language[] languages = AnnotatorException.call(() -> detector.predictLanguages(input.text()));
      return List.of(languageLayer(languages));
    }
    ProbingLanguageDetectionResult result =
        AnnotatorException.call(() -> detector.probingPredictLanguages(input.text(), probing));
    return List.of(
        languageLayer(result.languages()),
        new Layer("turbo:probing-length", "document",
            new Column.U32("length", new long[] {result.length()})));
  }

  private static Layer languageLayer(Language[] languages) {
    String[] codes = new String[languages.length];
    double[] confidences = new double[languages.length];
    for (int i = 0; i < languages.length; i++) {
      codes[i] = languages[i].getLang();
      confidences[i] = languages[i].getConfidence();
    }
    return new Layer("opennlp:language", "document", new Column.Str("code", codes),
        new Column.F64("confidence", confidences));
  }
}
