package ai.pipestream.turbo.goldens;

import java.io.ByteArrayInputStream;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import opennlp.tools.tokenize.TokenizerME;
import opennlp.tools.tokenize.TokenizerModel;
import opennlp.tools.util.Span;

/**
 * The tokenizer's golden: {@code opennlp:tokens} from {@link TokenizerME#tokenizePos(String)}
 * over the whole input, then its confidence layer from {@code getTokenProbabilities()}, which
 * must hold one value per token.
 */
final class TokenGolden implements AnnotatorGolden {

  private final ModelRef model;
  private final TokenizerME tokenizer;

  private TokenGolden(ModelRef model, TokenizerME tokenizer) {
    this.model = model;
    this.tokenizer = tokenizer;
  }

  /** Loads the tokenizer model from {@code file}, hashing exactly the bytes loaded. */
  static TokenGolden load(Path file) throws IOException {
    byte[] bytes = Files.readAllBytes(file);
    TokenizerModel m = new TokenizerModel(new ByteArrayInputStream(bytes));
    return new TokenGolden(new ModelRef(file.getFileName().toString(), Sha.sha512(bytes)), new TokenizerME(m));
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
  @SuppressWarnings("removal")
  public List<Layer> layers(Input input) {
    Span[] spans = tokenizer.tokenizePos(input.text());
    double[] probs = tokenizer.getTokenProbabilities();
    if (probs.length != spans.length) {
      throw new IllegalStateException("input " + input.inputId() + ": getTokenProbabilities returned "
          + probs.length + " values for " + spans.length + " tokens");
    }
    long[] starts = new long[spans.length];
    long[] ends = new long[spans.length];
    for (int i = 0; i < spans.length; i++) {
      Utf8Offsets.ByteSpan b = input.offsets().span(spans[i].getStart(), spans[i].getEnd());
      starts[i] = b.start();
      ends[i] = b.end();
    }
    return List.of(
        new Layer("opennlp:tokens", "text", new Column.U32("start", starts), new Column.U32("end", ends)),
        new Layer("turbo:confidence<opennlp:tokens>", "aligned:opennlp:tokens", new Column.F64("value", probs)));
  }
}
