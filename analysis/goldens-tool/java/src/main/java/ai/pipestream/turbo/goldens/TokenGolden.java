package ai.pipestream.turbo.goldens;

import java.io.IOException;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.regex.Pattern;
import opennlp.tools.dictionary.Dictionary;
import opennlp.tools.tokenize.TokenizerFactory;
import opennlp.tools.tokenize.TokenizerME;
import opennlp.tools.tokenize.TokenizerModel;
import opennlp.tools.tokenize.lang.Factory;
import opennlp.tools.util.Span;

/**
 * The tokenizer's golden: {@code opennlp:tokens} from {@link TokenizerME#tokenizePos(String)}
 * over the whole input, then {@code turbo:utf16<opennlp:tokens>} when a token starts or ends
 * inside a surrogate pair ({@link SpanLayers}), then its confidence layer from {@code getTokenProbabilities()}, which
 * must hold one value per token.
 *
 * <p>Besides a model file as it is, the tokenizer runs on two models rebuilt from one through
 * {@link TokenizerModel#TokenizerModel(opennlp.tools.ml.model.MaxentModel, Map, TokenizerFactory)}
 * with other factory settings, as an application would build them: with an abbreviation
 * dictionary read from an XML file, and with the alphanumeric optimization on. Neither is
 * possible with a published model as it is, since none has a dictionary or the optimization.</p>
 */
final class TokenGolden implements AnnotatorGolden {

  /** The {@code config} keys of a run with the alphanumeric optimization on. */
  static final String ALPHANUMERIC_KEY = "useAlphaNumericOptimization";
  static final String PATTERN_KEY = "alphaNumericPattern";

  private final List<ModelRef> models;
  private final TokenizerModel loaded;
  private final TokenizerME tokenizer;
  private final String modelDir;
  private final Map<String, String> config;
  private final Set<String> inputSets;

  private TokenGolden(List<ModelRef> models, TokenizerModel loaded, String modelDir, Map<String, String> config,
      Set<String> inputSets) {
    this.models = List.copyOf(models);
    this.loaded = loaded;
    this.tokenizer = new TokenizerME(loaded);
    this.modelDir = modelDir;
    this.config = Map.copyOf(config);
    this.inputSets = Set.copyOf(inputSets);
  }

  /** Loads the tokenizer model from {@code file}, hashing exactly the bytes loaded. */
  static TokenGolden load(Path file, Set<String> inputSets) throws IOException {
    ModelFile f = ModelFile.read(file);
    return new TokenGolden(List.of(f.ref()), new TokenizerModel(f.stream()), f.ref().baseName(), Map.of(),
        inputSets);
  }

  /**
   * The model of {@code file} rebuilt with the abbreviation dictionary of {@code dictionary} (an
   * OpenNLP dictionary XML file) and its own other settings. Both files are {@code model} records,
   * the model first, and the directory is {@code <model>+<dictionary>}.
   */
  static TokenGolden withAbbreviations(Path file, Path dictionary, Set<String> inputSets) throws IOException {
    ModelFile f = ModelFile.read(file);
    ModelFile d = ModelFile.read(dictionary);
    TokenizerModel base = new TokenizerModel(f.stream());
    TokenizerFactory factory = base.getFactory();
    TokenizerModel rebuilt = new TokenizerModel(base.getMaxentModel(), new HashMap<>(),
        new TokenizerFactory(base.getLanguage(), new Dictionary(d.stream()), factory.isUseAlphaNumericOptimization(),
            factory.getAlphaNumericPattern()));
    return new TokenGolden(List.of(f.ref(), d.ref()), rebuilt, f.ref().baseName() + "+" + d.ref().baseName(),
        Map.of(), inputSets);
  }

  /**
   * The model of {@code file} rebuilt with the alphanumeric optimization on and its own other
   * settings, its own pattern included. The directory is {@code <model>+alphanumeric}, and the
   * {@code config} records are the optimization and the pattern the tokenizer uses: the factory's,
   * or {@code Factory.DEFAULT_ALPHANUMERIC} when it has none, as {@code TokenizerME} falls back.
   */
  static TokenGolden withAlphaNumericOptimization(Path file, Set<String> inputSets) throws IOException {
    ModelFile f = ModelFile.read(file);
    TokenizerModel base = new TokenizerModel(f.stream());
    TokenizerModel rebuilt = new TokenizerModel(base.getMaxentModel(), new HashMap<>(),
        new TokenizerFactory(base.getLanguage(), base.getAbbreviations(), true,
            base.getFactory().getAlphaNumericPattern()));
    if (!rebuilt.getFactory().isUseAlphaNumericOptimization()) {
      throw new IllegalStateException(f.ref().fileName() + ": the alphanumeric optimization did not turn on");
    }
    Pattern pattern = rebuilt.getFactory().getAlphaNumericPattern();
    String effective = (pattern == null ? Factory.DEFAULT_ALPHANUMERIC : pattern).pattern();
    return new TokenGolden(List.of(f.ref()), rebuilt, f.ref().baseName() + "+alphanumeric",
        Map.of(ALPHANUMERIC_KEY, "true", PATTERN_KEY, effective), inputSets);
  }

  /** The loaded model, which {@link TokenDump} builds its recording model from. */
  TokenizerModel loaded() {
    return loaded;
  }

  @Override
  public String annotator() {
    return "tokenize";
  }

  @Override
  public List<ModelRef> models() {
    return models;
  }

  @Override
  public String modelDir() {
    return modelDir;
  }

  @Override
  public String annotatorClass() {
    return TokenizerME.class.getName();
  }

  @Override
  public String method() {
    return "tokenizePos";
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
    return layers(tokenizer, input);
  }

  /** The layers {@code tokenizer} produces for the input; {@link TokenDump} runs it on its recording tokenizer. */
  @SuppressWarnings("removal")
  static List<Layer> layers(TokenizerME tokenizer, Input input) throws AnnotatorException {
    Span[] spans = AnnotatorException.call(() -> tokenizer.tokenizePos(input.text()));
    double[] probs = tokenizer.getTokenProbabilities();
    if (probs.length != spans.length) {
      throw new IllegalStateException("input " + input.inputId() + ": getTokenProbabilities returned "
          + probs.length + " values for " + spans.length + " tokens");
    }
    List<Layer> layers = new ArrayList<>(SpanLayers.of("opennlp:tokens", input.offsets(), spans));
    layers.add(new Layer("turbo:confidence<opennlp:tokens>", "aligned:opennlp:tokens", new Column.F64("value", probs)));
    return layers;
  }
}
