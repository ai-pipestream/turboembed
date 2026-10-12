package ai.pipestream.turbo.goldens;

import java.util.ArrayList;
import java.util.List;
import java.util.Set;
import opennlp.tools.tokenize.uax29.WordToken;
import opennlp.tools.tokenize.uax29.WordTokenizer;
import opennlp.tools.util.Span;

/**
 * The golden of OpenNLP's UAX #29 word tokenizer with its default maximum token length:
 * {@code opennlp:tokens} from {@link WordTokenizer#tokenizeTyped(CharSequence)}, then
 * {@code turbo:utf16<opennlp:tokens>} when a token cuts a surrogate pair ({@link SpanLayers}),
 * then {@code turbo:word-types}, aligned to the tokens, with the name of each token's
 * {@code WordType} in its {@code kind} column. It has no model, so its directory is
 * {@code uax29}.
 */
final class WordTokenGolden implements AnnotatorGolden {

  private final WordTokenizer tokenizer = new WordTokenizer();
  private final Set<String> inputSets;

  WordTokenGolden(Set<String> inputSets) {
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
    return "uax29";
  }

  @Override
  public String annotatorClass() {
    return WordTokenizer.class.getName();
  }

  @Override
  public String method() {
    return "tokenizeTyped";
  }

  @Override
  public Set<String> inputSets() {
    return inputSets;
  }

  @Override
  public List<Layer> layers(Input input) throws AnnotatorException {
    List<WordToken> tokens = AnnotatorException.call(() -> tokenizer.tokenizeTyped(input.text()));
    Span[] spans = new Span[tokens.size()];
    String[] kinds = new String[tokens.size()];
    for (int i = 0; i < spans.length; i++) {
      spans[i] = tokens.get(i).span();
      kinds[i] = tokens.get(i).type().name();
    }
    List<Layer> layers = new ArrayList<>(SpanLayers.of("opennlp:tokens", input.offsets(), spans));
    layers.add(new Layer("turbo:word-types", "aligned:opennlp:tokens", new Column.Str("kind", kinds)));
    return layers;
  }
}
