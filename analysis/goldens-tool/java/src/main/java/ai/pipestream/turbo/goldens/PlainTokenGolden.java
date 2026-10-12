package ai.pipestream.turbo.goldens;

import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;
import opennlp.tools.tokenize.SimpleTokenizer;
import opennlp.tools.tokenize.Tokenizer;
import opennlp.tools.tokenize.WhitespaceTokenizer;
import opennlp.tools.util.Span;
import opennlp.tools.util.WhitespaceMode;

/**
 * The golden of a tokenizer without a model, {@link SimpleTokenizer} or {@link WhitespaceTokenizer}:
 * {@code opennlp:tokens} from {@code tokenizePos} over the whole input, then
 * {@code turbo:utf16<opennlp:tokens>} when a token starts or ends inside a surrogate pair
 * ({@link SpanLayers}). The simple tokenizer classifies UTF-16 units, so it cuts every character
 * outside the Basic Multilingual Plane in two.
 *
 * <p>Each runs in one whitespace mode, with line breaks kept as tokens or not. The directory is
 * the tokenizer's name ({@code simple} or {@code whitespace}), then {@code +keep-new-lines} and
 * {@code +legacy} when those apply; the {@code config} records are {@code keepNewLines true} and
 * {@code whitespace legacy}, each only when it applies.</p>
 */
final class PlainTokenGolden implements AnnotatorGolden {

  /** The {@code config} key of a run that keeps line breaks as tokens. */
  static final String KEEP_NEW_LINES_KEY = "keepNewLines";

  /** The two tokenizers. */
  enum Kind {
    SIMPLE("simple", SimpleTokenizer.class),
    WHITESPACE("whitespace", WhitespaceTokenizer.class);

    final String dir;
    final Class<? extends Tokenizer> type;

    Kind(String dir, Class<? extends Tokenizer> type) {
      this.dir = dir;
      this.type = type;
    }
  }

  private final Kind kind;
  private final boolean keepNewLines;
  private final WhitespaceMode whitespace;
  private final Set<String> inputSets;

  PlainTokenGolden(Kind kind, boolean keepNewLines, WhitespaceMode whitespace, Set<String> inputSets) {
    this.kind = kind;
    this.keepNewLines = keepNewLines;
    this.whitespace = whitespace;
    this.inputSets = Set.copyOf(inputSets);
  }

  @Override
  public String annotator() {
    return "tokenize";
  }

  @Override
  public List<ModelRef> models() {
    return List.of();
  }

  @Override
  public String modelDir() {
    return kind.dir + (keepNewLines ? "+keep-new-lines" : "") + Whitespace.dirSuffix(whitespace);
  }

  @Override
  public String annotatorClass() {
    return kind.type.getName();
  }

  @Override
  public String method() {
    return "tokenizePos";
  }

  @Override
  public Map<String, String> config() {
    Map<String, String> config = new HashMap<>(Whitespace.config(whitespace));
    if (keepNewLines) {
      config.put(KEEP_NEW_LINES_KEY, "true");
    }
    return config;
  }

  @Override
  public Set<String> inputSets() {
    return inputSets;
  }

  @Override
  public List<Layer> layers(Input input) throws AnnotatorException {
    Span[] spans = Whitespace.in(whitespace, () -> AnnotatorException.call(() -> tokenize(input.text())));
    return SpanLayers.of("opennlp:tokens", input.offsets(), spans);
  }

  /**
   * {@code SimpleTokenizer} has only its shared instance, so the line break setting is set for
   * the one call and put back; {@code WhitespaceTokenizer} makes an instance with it.
   */
  private Span[] tokenize(String text) {
    return switch (kind) {
      case SIMPLE -> {
        SimpleTokenizer.INSTANCE.setKeepNewLines(keepNewLines);
        try {
          yield SimpleTokenizer.INSTANCE.tokenizePos(text);
        } finally {
          SimpleTokenizer.INSTANCE.setKeepNewLines(false);
        }
      }
      case WHITESPACE -> WhitespaceTokenizer.newInstance(keepNewLines).tokenizePos(text);
    };
  }
}
