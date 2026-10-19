package ai.pipestream.turbo.goldens;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.io.IOException;
import java.io.OutputStream;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.TreeMap;
import opennlp.tools.ml.maxent.GISModel;
import opennlp.tools.ml.model.Context;
import opennlp.tools.tokenize.TokenizerFactory;
import opennlp.tools.tokenize.TokenizerModel;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

/** {@link Main#generate}: what the generator writes for one input, with stub and real annotators. */
class GeneratorTest {

  /** An annotator whose run on any input does what {@code run} does. */
  private record Stub(Map<String, String> config, Run run) implements AnnotatorGolden {

    interface Run {
      List<Layer> layers(Input input) throws AnnotatorException;
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
    public String annotatorClass() {
      return "opennlp.tools.tokenize.SimpleTokenizer";
    }

    @Override
    public String method() {
      return "tokenizePos";
    }

    @Override
    public Set<String> inputSets() {
      return Set.of("test");
    }

    @Override
    public List<Layer> layers(Input input) throws AnnotatorException {
      return run.layers(input);
    }
  }

  /** A recorder of {@code golden} whose run does what {@code run} does, with no decision points. */
  private record StubDump(AnnotatorGolden golden, Stub.Run run) implements AnnotatorDump {
    @Override
    public Run run(Input input) throws AnnotatorException {
      return new Run(new String[] {"F", "T"}, List.of(), run.layers(input));
    }
  }

  private static Input input(String set, String id, String text) {
    String inputId = set + "/" + id;
    return new Input(set, id, text.getBytes(StandardCharsets.UTF_8), text, new Utf8Offsets(inputId, text));
  }

  private static Header header(Input input, AnnotatorGolden a) {
    return new Header("0123456789abcdef0123456789abcdef01234567", "3.0.0-PRE-SNAPSHOT",
        new Jdk("21.0.5", "21.0.5+11-LTS", "Eclipse Adoptium", "15.0"), input.inputId(), input.bytes().length,
        input.text().length(), Sha.sha256(input.bytes()), a.models(), a.annotatorClass(), a.method(), a.config());
  }

  private static Map<Path, byte[]> generate(Input input, List<AnnotatorGolden> annotators,
      List<AnnotatorDump> recorders) {
    Map<Path, byte[]> files = new TreeMap<>();
    Main.generate(input, annotators, recorders, a -> header(input, a), files);
    return files;
  }

  private static String text(byte[] bytes) {
    return new String(bytes, StandardCharsets.UTF_8);
  }

  /**
   * What OpenNLP's {@code SimpleTokenizer} with {@code keepNewLines} throws on a text that starts
   * with a line break, as the {@code newline-first.golden} fixture of the turbo-goldens reader
   * has it.
   */
  private static final String NEWLINE_FIRST_GOLDEN = """
      golden 1
      generator\tturbo-goldens-java\t1
      opennlp\t0123456789abcdef0123456789abcdef01234567\t3.0.0-PRE-SNAPSHOT
      jdk\t21.0.5\t21.0.5+11-LTS\tEclipse Adoptium
      unicode\t15.0
      input\ttest/newline-first\t11\t11\teb783ec46c71892387c17c46cdae5ee94d202b74f75b5a4fc60b0b9d119a11fe
      annotator\topennlp.tools.tokenize.SimpleTokenizer\ttokenizePos
      config\tkeepNewLines\ttrue
      error\tjava.lang.IllegalArgumentException\tstart index must be zero or greater: -1
      """;

  private static final Stub.Run THROWS = in -> AnnotatorException.call(() -> {
    throw new IllegalArgumentException("start index must be zero or greater: -1");
  });

  @Test
  void anAnnotatorThatThrowsGetsAnErrorRecordAndNoDump() {
    Input in = input("test", "newline-first", "\nIt rained.");
    Stub stub = new Stub(Map.of("keepNewLines", "true"), THROWS);
    Map<Path, byte[]> files = generate(in, List.of(stub), List.of(new StubDump(stub, THROWS)));
    assertEquals(List.of(Path.of("tokenize/none/test/newline-first.golden")), List.copyOf(files.keySet()));
    assertEquals(NEWLINE_FIRST_GOLDEN, text(files.values().iterator().next()));
  }

  @Test
  void theMessageIsEscapedAndANullMessageIsEmpty() {
    Input in = input("test", "x", "x");
    Stub tabs = new Stub(Map.of(), i -> AnnotatorException.call(() -> {
      throw new IllegalStateException("a\tb\nc\\");
    }));
    String golden = text(generate(in, List.of(tabs), List.of()).values().iterator().next());
    assertTrue(golden.endsWith("\nerror\tjava.lang.IllegalStateException\ta\\tb\\nc\\\\\n"), golden);
    Stub none = new Stub(Map.of(), i -> AnnotatorException.call(() -> {
      throw new NullPointerException();
    }));
    golden = text(generate(in, List.of(none), List.of()).values().iterator().next());
    assertTrue(golden.endsWith("\nannotator\topennlp.tools.tokenize.SimpleTokenizer\ttokenizePos\n"
        + "error\tjava.lang.NullPointerException\t\n"), golden);
  }

  @Test
  void anExceptionThatIsNotTheAnnotatorsStopsTheGenerator() {
    Input in = input("test", "x", "x");
    IllegalStateException own = new IllegalStateException("a check of the generator");
    Stub stub = new Stub(Map.of(), i -> {
      throw own;
    });
    assertSame(own, assertThrows(IllegalStateException.class, () -> generate(in, List.of(stub), List.of())));
  }

  @Test
  void aRecorderThatDoesNotThrowWhereThePlainAnnotatorThrewStopsTheGenerator() {
    Input in = input("test", "x", "x");
    Stub stub = new Stub(Map.of(), THROWS);
    StubDump fine = new StubDump(stub, i -> List.of(new Layer("opennlp:tokens", "text",
        new Column.U32("start", new long[] {0}), new Column.U32("end", new long[] {1}))));
    assertThrows(IllegalStateException.class, () -> generate(in, List.of(stub), List.of(fine)));
    StubDump other = new StubDump(stub, i -> AnnotatorException.call(() -> {
      throw new IllegalStateException("start index must be zero or greater: -1");
    }));
    assertThrows(IllegalStateException.class, () -> generate(in, List.of(stub), List.of(other)));
  }

  @Test
  void aTokenizerThatSplitsAnEmojiWritesItsUtf16SpansAndADump(@TempDir Path dir) throws IOException {
    // No predicate is known, so both outcomes have 0.5, and the first, T (split), wins: every
    // UTF-16 unit of a token that is not alphanumeric becomes a token of its own, the two halves
    // of the emoji too.
    GISModel weights = new GISModel(new Context[] {new Context(new int[] {0, 1}, new double[] {0.0, 0.0})},
        new String[] {"never"}, new String[] {"T", "F"});
    TokenizerModel model = new TokenizerModel(weights, Map.of(), new TokenizerFactory("en", null, false, null));
    Path file = dir.resolve("t.bin");
    try (OutputStream out = Files.newOutputStream(file)) {
      model.serialize(out);
    }
    TokenGolden tokens = TokenGolden.load(file, Set.of("test"));
    Input in = input("test", "smile", "x \ud83d\ude00!");
    Map<Path, byte[]> files = generate(in, List.of(tokens), List.of(TokenDump.of(tokens)));
    assertEquals(List.of(Path.of("tokenize/t/test/smile.dump"), Path.of("tokenize/t/test/smile.golden")),
        List.copyOf(files.keySet()));
    String golden = text(files.get(Path.of("tokenize/t/test/smile.golden")));
    // "x" is too short to be split; "😀!" is three UTF-16 units and three tokens.
    assertTrue(golden.contains("""
        layer\topennlp:tokens\ttext\t4\tstart:u32\tend:u32
        0\t1
        2\t6
        2\t6
        6\t7
        end\topennlp:tokens
        layer\tturbo:utf16<opennlp:tokens>\taligned:opennlp:tokens\t4\tstart16:u32\tend16:u32\tcut:u32
        0\t1\t0
        2\t3\t1
        3\t4\t1
        4\t5\t0
        end\tturbo:utf16<opennlp:tokens>
        layer\tturbo:confidence<opennlp:tokens>\taligned:opennlp:tokens\t4\tvalue:f64
        """), golden);
    assertFalse(golden.contains("\nerror\t"), golden);
  }
}
