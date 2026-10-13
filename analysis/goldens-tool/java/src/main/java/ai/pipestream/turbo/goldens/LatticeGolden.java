package ai.pipestream.turbo.goldens;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.charset.CharacterCodingException;
import java.nio.charset.Charset;
import java.nio.charset.CodingErrorAction;
import java.nio.charset.StandardCharsets;
import java.nio.file.DirectoryStream;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Map;
import java.util.Set;
import opennlp.tools.tokenize.lattice.LatticeTokenizer;
import opennlp.tools.tokenize.lattice.MecabDictionary;
import opennlp.tools.tokenize.lattice.Morpheme;
import opennlp.tools.util.Span;
import opennlp.tools.util.WhitespaceMode;

/**
 * The golden of OpenNLP's {@link LatticeTokenizer}: the morphemes of the cheapest path through the
 * word lattice of a MeCab dictionary over each whitespace-free stretch of the text.
 *
 * <p>The dictionary is a directory, and its {@code model} record is the directory's name with the
 * SHA-512 of a manifest of the files {@link MecabDictionary#load} reads, in the order it reads
 * them ({@code matrix.def}, the {@code *.csv} files sorted by path, {@code char.def},
 * {@code unk.def}): for each, its name, a tab, the SHA-512 of its bytes in lowercase hex and a
 * line feed. The directory's name is the golden's model directory. {@code StringUtil.isWhitespace}
 * splits the dictionary's lines and the text, so a {@code +legacy} run loads the dictionary and
 * segments with the legacy mode active.</p>
 *
 * <p>A dictionary that is not UTF-8 (IPADIC is EUC-JP) is loaded in its own encoding, as OpenNLP
 * users load it, and has a second {@code model} record: a UTF-8 copy of its files, which the
 * generator checks decodes to exactly the same text, file by file. The Rust port reads UTF-8 only
 * and is compared on the copy.</p>
 *
 * <p>The layers are {@code opennlp:tokens}, one row per morpheme, and {@code turbo:morphemes},
 * aligned to it: {@code unknown} (1 for a morpheme unknown word handling made) and {@code
 * features}, the entry's feature columns joined with commas. A morpheme is always whole code
 * points, so there is never a {@code turbo:utf16<opennlp:tokens>} layer, and the generator stops
 * if one would be needed.</p>
 */
final class LatticeGolden implements AnnotatorGolden {

  /** The files of a dictionary directory other than the lexicon files, in the order read. */
  private static final String MATRIX = "matrix.def";
  private static final String CHAR_DEF = "char.def";
  private static final String UNK_DEF = "unk.def";

  private final List<ModelRef> models;
  private final LatticeTokenizer tokenizer;
  private final WhitespaceMode whitespace;
  private final Set<String> inputSets;

  private LatticeGolden(List<ModelRef> models, LatticeTokenizer tokenizer, WhitespaceMode whitespace,
      Set<String> inputSets) {
    this.models = List.copyOf(models);
    this.tokenizer = tokenizer;
    this.whitespace = whitespace;
    this.inputSets = Set.copyOf(inputSets);
  }

  /** Loads the UTF-8 dictionary directory {@code dir} in {@code whitespace} mode. */
  static LatticeGolden load(Path dir, WhitespaceMode whitespace, Set<String> inputSets) throws IOException {
    ModelRef ref = manifest(dir);
    MecabDictionary dictionary = loadIn(dir, StandardCharsets.UTF_8, whitespace);
    return new LatticeGolden(List.of(ref), new LatticeTokenizer(dictionary), whitespace, inputSets);
  }

  /**
   * Loads the dictionary directory {@code dir} in {@code charset}, after checking that {@code
   * utf8Copy} holds the same files and that each, read as UTF-8, is the same text as the original
   * read in {@code charset}. Both decoders refuse malformed input.
   */
  static LatticeGolden loadRecoded(Path dir, Charset charset, Path utf8Copy, WhitespaceMode whitespace,
      Set<String> inputSets) throws IOException {
    List<Path> original = files(dir);
    List<Path> copy = files(utf8Copy);
    List<String> names = original.stream().map(p -> p.getFileName().toString()).toList();
    if (!names.equals(copy.stream().map(p -> p.getFileName().toString()).toList())) {
      throw new IllegalStateException(utf8Copy + " does not hold the files of " + dir);
    }
    for (int i = 0; i < original.size(); i++) {
      String a = decode(Files.readAllBytes(original.get(i)), charset, original.get(i));
      String b = decode(Files.readAllBytes(copy.get(i)), StandardCharsets.UTF_8, copy.get(i));
      if (!a.equals(b)) {
        throw new IllegalStateException(copy.get(i) + " is not " + original.get(i) + " in UTF-8");
      }
    }
    List<ModelRef> refs = List.of(manifest(dir), manifest(utf8Copy));
    MecabDictionary dictionary = loadIn(dir, charset, whitespace);
    return new LatticeGolden(refs, new LatticeTokenizer(dictionary), whitespace, inputSets);
  }

  private static MecabDictionary loadIn(Path dir, Charset charset, WhitespaceMode whitespace) throws IOException {
    WhitespaceMode previous = WhitespaceMode.current();
    WhitespaceMode.setActive(whitespace);
    try {
      return MecabDictionary.load(dir, charset);
    } finally {
      WhitespaceMode.setActive(previous);
    }
  }

  private static String decode(byte[] bytes, Charset charset, Path file) throws IOException {
    try {
      return charset.newDecoder().onMalformedInput(CodingErrorAction.REPORT)
          .onUnmappableCharacter(CodingErrorAction.REPORT).decode(ByteBuffer.wrap(bytes)).toString();
    } catch (CharacterCodingException e) {
      throw new IOException(file + ": not " + charset, e);
    }
  }

  /** The files {@link MecabDictionary#load} reads from {@code dir}, in the order it reads them. */
  static List<Path> files(Path dir) throws IOException {
    List<Path> csv = new ArrayList<>();
    try (DirectoryStream<Path> stream = Files.newDirectoryStream(dir, "*.csv")) {
      for (Path p : stream) {
        csv.add(p);
      }
    }
    Collections.sort(csv);
    List<Path> all = new ArrayList<>();
    all.add(dir.resolve(MATRIX));
    all.addAll(csv);
    all.add(dir.resolve(CHAR_DEF));
    all.add(dir.resolve(UNK_DEF));
    return all;
  }

  /** The directory's name and the SHA-512 of the manifest of its files. */
  static ModelRef manifest(Path dir) throws IOException {
    StringBuilder m = new StringBuilder();
    for (Path f : files(dir)) {
      m.append(f.getFileName()).append('\t').append(Sha.sha512(Files.readAllBytes(f))).append('\n');
    }
    return new ModelRef(dir.getFileName().toString(), Sha.sha512(m.toString().getBytes(StandardCharsets.UTF_8)));
  }

  @Override
  public String annotator() {
    return "lattice";
  }

  @Override
  public List<ModelRef> models() {
    return models;
  }

  @Override
  public String modelDir() {
    return models.get(0).baseName() + Whitespace.dirSuffix(whitespace);
  }

  @Override
  public String annotatorClass() {
    return LatticeTokenizer.class.getName();
  }

  @Override
  public String method() {
    return "analyze";
  }

  @Override
  public Map<String, String> config() {
    return Whitespace.config(whitespace);
  }

  @Override
  public Set<String> inputSets() {
    return inputSets;
  }

  @Override
  public List<Layer> layers(Input input) throws AnnotatorException {
    List<Morpheme> morphemes =
        Whitespace.in(whitespace, () -> AnnotatorException.call(() -> tokenizer.analyze(input.text())));
    int n = morphemes.size();
    Span[] spans = new Span[n];
    long[] unknown = new long[n];
    String[] features = new String[n];
    for (int i = 0; i < n; i++) {
      Morpheme m = morphemes.get(i);
      spans[i] = m.span();
      unknown[i] = m.unknown() ? 1 : 0;
      features[i] = String.join(",", m.features());
    }
    List<Layer> layers = new ArrayList<>(SpanLayers.of("opennlp:tokens", input.offsets(), spans));
    if (layers.size() != 1) {
      throw new IllegalStateException("input " + input.inputId() + ": a morpheme cuts a character");
    }
    layers.add(new Layer("turbo:morphemes", "aligned:opennlp:tokens", new Column.U32("unknown", unknown),
        new Column.Str("features", features)));
    return layers;
  }
}
