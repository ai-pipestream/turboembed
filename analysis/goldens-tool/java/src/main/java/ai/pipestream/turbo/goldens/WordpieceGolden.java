package ai.pipestream.turbo.goldens;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.charset.CharacterCodingException;
import java.nio.charset.CodingErrorAction;
import java.nio.charset.StandardCharsets;
import java.nio.file.Path;
import java.util.List;
import java.util.Map;
import java.util.Set;
import opennlp.tools.tokenize.SubwordPiece;
import opennlp.tools.tokenize.WordpieceEncoder;
import opennlp.tools.tokenize.WordpieceTokenizer;

/**
 * The golden of OpenNLP's {@link WordpieceEncoder} for an uncased BERT vocabulary: the BERT
 * normalization (control characters removed, whitespace made spaces, CJK ideographs isolated,
 * lower casing and accent stripping, punctuation isolated) and greedy longest-match WordPiece
 * segmentation, with an id and a source span per piece.
 *
 * <p>The vocabulary is a {@code vocab.txt}: one piece per line, each line ended by a line feed,
 * the line number being the id. Its file is the golden's {@code model} record and names the
 * directory.</p>
 *
 * <p>The layer is {@code opennlp:subwords}: one row per piece of the text, with its UTF-8 byte
 * span and its id. {@code encode} also returns {@code [CLS]} first and {@code [SEP]} last, with
 * empty spans at the two ends of the text; they are not pieces of the text, so they are checked
 * and left out, as the Rust layer leaves out a model's special tokens. A piece's UTF-16 span
 * always holds whole code points, so it converts to bytes exactly.</p>
 *
 * <p>{@link WordpieceTokenizer} is not the class run: it does no normalization and has no spans
 * ({@code tokenizePos} throws), where {@code WordpieceEncoder} has both.</p>
 */
final class WordpieceGolden implements AnnotatorGolden {

  /** The longest word, in normalized code points, that is segmented rather than made unknown. */
  static final int MAX_WORD_CODE_POINTS = 100;

  private final ModelRef vocabulary;
  private final WordpieceEncoder encoder;
  private final Set<String> inputSets;

  private WordpieceGolden(ModelRef vocabulary, WordpieceEncoder encoder, Set<String> inputSets) {
    this.vocabulary = vocabulary;
    this.encoder = encoder;
    this.inputSets = Set.copyOf(inputSets);
  }

  /** Loads the vocabulary from {@code file}, hashing exactly the bytes loaded. */
  static WordpieceGolden load(Path file, Set<String> inputSets) throws IOException {
    ModelFile f = ModelFile.read(file);
    List<String> pieces = lines(file, f.bytes());
    return new WordpieceGolden(f.ref(), new WordpieceEncoder(pieces, true, MAX_WORD_CODE_POINTS), inputSets);
  }

  /** The lines of a {@code vocab.txt}, which must be UTF-8 with a line feed after every line. */
  static List<String> lines(Path file, byte[] bytes) throws IOException {
    String text;
    try {
      text = StandardCharsets.UTF_8.newDecoder().onMalformedInput(CodingErrorAction.REPORT)
          .onUnmappableCharacter(CodingErrorAction.REPORT).decode(ByteBuffer.wrap(bytes)).toString();
    } catch (CharacterCodingException e) {
      throw new IOException(file + ": not valid UTF-8", e);
    }
    if (!text.endsWith("\n")) {
      throw new IOException(file + ": the last line has no line feed after it");
    }
    return List.of(text.substring(0, text.length() - 1).split("\n", -1));
  }

  @Override
  public String annotator() {
    return "subwords";
  }

  @Override
  public List<ModelRef> models() {
    return List.of(vocabulary);
  }

  @Override
  public String annotatorClass() {
    return WordpieceEncoder.class.getName();
  }

  @Override
  public String method() {
    return "encode";
  }

  @Override
  public Map<String, String> config() {
    return Map.of("lowerCase", "true", "maxWordCodePoints", Integer.toString(MAX_WORD_CODE_POINTS));
  }

  @Override
  public Set<String> inputSets() {
    return inputSets;
  }

  @Override
  public List<Layer> layers(Input input) throws AnnotatorException {
    String text = input.text();
    List<SubwordPiece> all = AnnotatorException.call(() -> encoder.encode(text));
    expect(input, all.get(0), WordpieceTokenizer.BERT_CLS_TOKEN, 0);
    expect(input, all.get(all.size() - 1), WordpieceTokenizer.BERT_SEP_TOKEN, text.length());
    List<SubwordPiece> pieces = all.subList(1, all.size() - 1);
    long[] starts = new long[pieces.size()];
    long[] ends = new long[pieces.size()];
    long[] ids = new long[pieces.size()];
    for (int i = 0; i < ids.length; i++) {
      SubwordPiece p = pieces.get(i);
      Utf8Offsets.ByteSpan b = input.offsets().span(p.start(), p.end());
      starts[i] = b.start();
      ends[i] = b.end();
      ids[i] = p.id();
    }
    return List.of(new Layer("opennlp:subwords", "text", new Column.U32("start", starts),
        new Column.U32("end", ends), new Column.U32("id", ids)));
  }

  /** Fails unless {@code p} is the special token {@code token} with an empty span at {@code at}. */
  private static void expect(Input input, SubwordPiece p, String token, int at) {
    if (!p.piece().equals(token) || p.start() != at || p.end() != at) {
      throw new IllegalStateException("input " + input.inputId() + ": expected " + token + " at " + at
          + ", found " + p);
    }
  }
}
