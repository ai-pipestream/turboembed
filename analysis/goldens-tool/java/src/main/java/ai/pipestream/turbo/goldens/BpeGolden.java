package ai.pipestream.turbo.goldens;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.Set;
import opennlp.tools.tokenize.BPEModel;
import opennlp.tools.tokenize.BPETokenizer;
import opennlp.tools.tokenize.BPETokenizer.SymbolPair;
import opennlp.tools.tokenize.BPETokenizerTrainer;
import opennlp.tools.util.Span;

/**
 * The golden of OpenNLP's {@link BPETokenizerTrainer} trained on the input text, and of the
 * {@link BPETokenizer} it makes run over the same text.
 *
 * <p>The corpus is the input text as one element; the trainer cuts it into words at whitespace,
 * so its lines are not kept apart. The first layer is {@code turbo:bpe-merges}, anchored on the
 * document: one row per merge in the order learned, with the left and right symbols as
 * {@code str} columns. A symbol is UTF-16 units and can be one half of a surrogate pair, which the
 * writer escapes as {@code \}{@code uXXXX}. Then {@code opennlp:tokens} from
 * {@code tokenizePos}, and {@code turbo:utf16<opennlp:tokens>} when a token cuts a character
 * ({@link SpanLayers}).</p>
 *
 * <p>The directory is {@code merges-<numMerges>}; {@code numMerges} and {@code languageCode} are
 * the {@code config} records.</p>
 */
final class BpeGolden implements AnnotatorGolden {

  /** The language code the model is trained for; the trainer only records it. */
  static final String LANGUAGE = "en";

  private final int numMerges;
  private final Set<String> inputSets;

  BpeGolden(int numMerges, Set<String> inputSets) {
    this.numMerges = numMerges;
    this.inputSets = Set.copyOf(inputSets);
  }

  @Override
  public String annotator() {
    return "bpe";
  }

  @Override
  public List<ModelRef> models() {
    return List.of();
  }

  @Override
  public String modelDir() {
    return "merges-" + numMerges;
  }

  @Override
  public String annotatorClass() {
    return BPETokenizerTrainer.class.getName();
  }

  @Override
  public String method() {
    return "train";
  }

  @Override
  public Map<String, String> config() {
    return Map.of("languageCode", LANGUAGE, "numMerges", Integer.toString(numMerges));
  }

  @Override
  public Set<String> inputSets() {
    return inputSets;
  }

  @Override
  public List<Layer> layers(Input input) throws AnnotatorException {
    String text = input.text();
    BPEModel model = AnnotatorException.call(() -> new BPETokenizerTrainer().train(List.of(text), numMerges,
        LANGUAGE));
    List<SymbolPair> merges = model.getMerges();
    String[] left = new String[merges.size()];
    String[] right = new String[merges.size()];
    for (int i = 0; i < left.length; i++) {
      left[i] = merges.get(i).left();
      right[i] = merges.get(i).right();
    }
    Span[] spans = AnnotatorException.call(() -> new BPETokenizer(model).tokenizePos(text));
    List<Layer> layers = new ArrayList<>();
    layers.add(new Layer("turbo:bpe-merges", "document", new Column.Str("left", left),
        new Column.Str("right", right)));
    layers.addAll(SpanLayers.of("opennlp:tokens", input.offsets(), spans));
    return layers;
  }
}
