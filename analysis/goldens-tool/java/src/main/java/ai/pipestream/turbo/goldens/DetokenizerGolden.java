package ai.pipestream.turbo.goldens;

import java.io.IOException;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.Set;
import opennlp.tools.tokenize.DetokenizationDictionary;
import opennlp.tools.tokenize.Detokenizer;
import opennlp.tools.tokenize.DictionaryDetokenizer;

/**
 * The golden of {@link DictionaryDetokenizer} with one dictionary. Its input is a list of tokens,
 * not a text: an input file holds one token per line, each line ended by a line feed, and an
 * empty file is an empty list. A file whose id starts with {@code <name>-} is run with the
 * dictionary {@code <name>-detokenizer.xml}, which is the golden's {@code model} record; the
 * directory is the dictionary's base name.
 *
 * <p>The layers are {@code opennlp:tokens}, each token's span in the input file, then
 * {@code turbo:detokenization}, aligned to the tokens, with the name of the
 * {@code DetokenizationOperation} {@code detokenize(String[])} gave each token in its
 * {@code operation} column, then {@code turbo:detokenized}, one document row with what
 * {@code detokenize(String[], String)} returned without a split marker ({@code text}) and with
 * {@link #SPLIT_MARKER} ({@code marked}). The marker is the {@code config} record
 * {@code splitMarker}.</p>
 */
final class DetokenizerGolden implements AnnotatorGolden {

  /** The split marker of the {@code marked} column. */
  static final String SPLIT_MARKER = "<SPLIT>";

  /** The suffix of a dictionary's file name after the name that input ids start with. */
  static final String DICTIONARY_SUFFIX = "-detokenizer.xml";

  private final ModelRef dictionary;
  private final String name;
  private final Detokenizer detokenizer;
  private final Set<String> inputSets;

  private DetokenizerGolden(ModelRef dictionary, String name, Detokenizer detokenizer, Set<String> inputSets) {
    this.dictionary = dictionary;
    this.name = name;
    this.detokenizer = detokenizer;
    this.inputSets = Set.copyOf(inputSets);
  }

  /** Loads the dictionary {@code <name>-detokenizer.xml}, hashing exactly the bytes loaded. */
  static DetokenizerGolden load(Path file, Set<String> inputSets) throws IOException {
    String fileName = file.getFileName().toString();
    if (!fileName.endsWith(DICTIONARY_SUFFIX)) {
      throw new IllegalArgumentException(file + ": a detokenizer dictionary is named <name>" + DICTIONARY_SUFFIX);
    }
    ModelFile f = ModelFile.read(file);
    Detokenizer detokenizer = new DictionaryDetokenizer(new DetokenizationDictionary(f.stream()));
    String name = fileName.substring(0, fileName.length() - DICTIONARY_SUFFIX.length());
    return new DetokenizerGolden(f.ref(), name, detokenizer, inputSets);
  }

  @Override
  public String annotator() {
    return "detokenize";
  }

  @Override
  public List<ModelRef> models() {
    return List.of(dictionary);
  }

  @Override
  public String annotatorClass() {
    return DictionaryDetokenizer.class.getName();
  }

  @Override
  public String method() {
    return "detokenize";
  }

  @Override
  public Map<String, String> config() {
    return Map.of("splitMarker", SPLIT_MARKER);
  }

  @Override
  public Set<String> inputSets() {
    return inputSets;
  }

  @Override
  public boolean accepts(Input input) {
    return inputSets.contains(input.set()) && input.id().startsWith(name + "-");
  }

  @Override
  public List<Layer> layers(Input input) throws AnnotatorException {
    String text = input.text();
    List<int[]> spans = lines(input.inputId(), text);
    String[] tokens = new String[spans.size()];
    long[] starts = new long[tokens.length];
    long[] ends = new long[tokens.length];
    for (int i = 0; i < tokens.length; i++) {
      int[] s = spans.get(i);
      tokens[i] = text.substring(s[0], s[1]);
      Utf8Offsets.ByteSpan b = input.offsets().span(s[0], s[1]);
      starts[i] = b.start();
      ends[i] = b.end();
    }
    Detokenizer.DetokenizationOperation[] ops = AnnotatorException.call(() -> detokenizer.detokenize(tokens));
    if (ops.length != tokens.length) {
      throw new IllegalStateException("input " + input.inputId() + ": " + ops.length + " operations for "
          + tokens.length + " tokens");
    }
    String[] names = new String[ops.length];
    for (int i = 0; i < ops.length; i++) {
      names[i] = ops[i].name();
    }
    String plain = AnnotatorException.call(() -> detokenizer.detokenize(tokens, null));
    String marked = AnnotatorException.call(() -> detokenizer.detokenize(tokens, SPLIT_MARKER));
    return List.of(
        new Layer("opennlp:tokens", "text", new Column.U32("start", starts), new Column.U32("end", ends)),
        new Layer("turbo:detokenization", "aligned:opennlp:tokens", new Column.Str("operation", names)),
        new Layer("turbo:detokenized", "document", new Column.Str("text", new String[] {plain}),
            new Column.Str("marked", new String[] {marked})));
  }

  /**
   * The UTF-16 span of each line of {@code text}, without its line feed. Every line must end with
   * one, so a token can be empty but never hold a line feed.
   */
  static List<int[]> lines(String inputId, String text) {
    List<int[]> out = new ArrayList<>();
    int start = 0;
    while (start < text.length()) {
      int end = text.indexOf('\n', start);
      if (end < 0) {
        throw new IllegalArgumentException("input " + inputId + ": the last token has no line feed after it");
      }
      out.add(new int[] {start, end});
      start = end + 1;
    }
    return out;
  }
}
