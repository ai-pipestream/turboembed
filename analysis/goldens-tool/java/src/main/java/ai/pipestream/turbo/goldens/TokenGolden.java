package ai.pipestream.turbo.goldens;

import java.io.ByteArrayInputStream;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import opennlp.tools.tokenize.TokenizerME;
import opennlp.tools.tokenize.TokenizerModel;
import opennlp.tools.util.Span;

/**
 * The tokenizer's golden: {@code opennlp:tokens} from {@link TokenizerME#tokenizePos(String)}
 * over the whole input, then {@code turbo:utf16<opennlp:tokens>} when a token starts or ends
 * inside a surrogate pair ({@link SpanLayers}), then its confidence layer from {@code getTokenProbabilities()}, which
 * must hold one value per token.
 */
final class TokenGolden implements AnnotatorGolden {

  private final ModelRef model;
  private final TokenizerModel loaded;
  private final TokenizerME tokenizer;

  private TokenGolden(ModelRef model, TokenizerModel loaded) {
    this.model = model;
    this.loaded = loaded;
    this.tokenizer = new TokenizerME(loaded);
  }

  /** Loads the tokenizer model from {@code file}, hashing exactly the bytes loaded. */
  static TokenGolden load(Path file) throws IOException {
    byte[] bytes = Files.readAllBytes(file);
    TokenizerModel m = new TokenizerModel(new ByteArrayInputStream(bytes));
    return new TokenGolden(new ModelRef(file.getFileName().toString(), Sha.sha512(bytes)), m);
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
    return List.of(model);
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
  public List<Layer> layers(Input input) {
    return layers(tokenizer, input);
  }

  /** The layers {@code tokenizer} produces for the input; {@link TokenDump} runs it on its recording tokenizer. */
  @SuppressWarnings("removal")
  static List<Layer> layers(TokenizerME tokenizer, Input input) {
    Span[] spans = tokenizer.tokenizePos(input.text());
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
