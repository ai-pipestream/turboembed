package ai.pipestream.turbo.goldens;

import static org.junit.jupiter.api.Assertions.assertEquals;

import java.nio.charset.StandardCharsets;
import java.util.List;
import opennlp.tools.util.Span;
import org.junit.jupiter.api.Test;

class SpanLayersTest {

  /** "Smile 😀": six ASCII characters, then the emoji, two UTF-16 units and four UTF-8 bytes. */
  private static final String SMILE = "Smile 😀";

  /**
   * The golden of a tokenizer that splits the emoji between its two UTF-16 units, as the
   * {@code smile.golden} fixture of the turbo-goldens reader has it.
   */
  private static final String SMILE_GOLDEN = """
      golden 1
      generator\tturbo-goldens-java\t1
      opennlp\t0123456789abcdef0123456789abcdef01234567\t3.0.0-PRE-SNAPSHOT
      jdk\t21.0.5\t21.0.5+11-LTS\tEclipse Adoptium
      unicode\t15.0
      input\ttest/smile\t10\t8\tb7b55bcbe6c0c7d889485a656f2840a7d40808f36385d86e65d1ffb7798b82f4
      annotator\topennlp.tools.tokenize.SimpleTokenizer\ttokenizePos
      layer\topennlp:tokens\ttext\t3\tstart:u32\tend:u32
      0\t5
      6\t10
      6\t10
      end\topennlp:tokens
      layer\tturbo:utf16<opennlp:tokens>\taligned:opennlp:tokens\t3\tstart16:u32\tend16:u32\tcut:u32
      0\t5\t0
      6\t7\t1
      7\t8\t1
      end\tturbo:utf16<opennlp:tokens>
      """;

  @Test
  void tokensThatSplitASurrogatePairWriteTheReaderFixtureByteForByte() {
    byte[] bytes = SMILE.getBytes(StandardCharsets.UTF_8);
    Header h = new Header("0123456789abcdef0123456789abcdef01234567", "3.0.0-PRE-SNAPSHOT",
        new Jdk("21.0.5", "21.0.5+11-LTS", "Eclipse Adoptium", "15.0"), "test/smile", bytes.length, SMILE.length(),
        Sha.sha256(bytes), List.of(), "opennlp.tools.tokenize.SimpleTokenizer", "tokenizePos");
    List<Layer> layers = SpanLayers.of("opennlp:tokens", new Utf8Offsets("test/smile", SMILE),
        new Span[] {new Span(0, 5), new Span(6, 7), new Span(7, 8)});
    assertEquals(SMILE_GOLDEN, new String(GoldenWriter.write(h, layers), StandardCharsets.UTF_8));
  }

  @Test
  void spansThatCutNoCharacterGiveOnlyTheByteLayer() {
    List<Layer> layers = SpanLayers.of("opennlp:tokens", new Utf8Offsets("test/smile", SMILE),
        new Span[] {new Span(0, 5), new Span(6, 8)});
    assertEquals(1, layers.size());
    Layer tokens = layers.get(0);
    assertEquals("opennlp:tokens", tokens.id());
    assertEquals("text", tokens.anchor());
    assertEquals(10L, ((Column.U32) tokens.columns().get(1)).values()[1]);
  }

  @Test
  void aSentenceThatCutsACharacterGetsItsOwnUtf16Layer() {
    // "a😀b": a sentence ending after the high surrogate, another starting at the low one.
    List<Layer> layers = SpanLayers.of("opennlp:sentences", new Utf8Offsets("t/s", "a😀b"),
        new Span[] {new Span(0, 2), new Span(2, 4)});
    assertEquals(2, layers.size());
    Layer utf16 = layers.get(1);
    assertEquals("turbo:utf16<opennlp:sentences>", utf16.id());
    assertEquals("aligned:opennlp:sentences", utf16.anchor());
    assertEquals(List.of("start16", "end16", "cut"), utf16.columns().stream().map(Column::name).toList());
    assertEquals(List.of(0L, 5L), List.of(((Column.U32) layers.get(0).columns().get(0)).values()[0],
        ((Column.U32) layers.get(0).columns().get(1)).values()[0]));
    assertEquals(List.of(1L, 6L), List.of(((Column.U32) layers.get(0).columns().get(0)).values()[1],
        ((Column.U32) layers.get(0).columns().get(1)).values()[1]));
    long[] cut = ((Column.U32) utf16.columns().get(2)).values();
    assertEquals(List.of(1L, 1L), List.of(cut[0], cut[1]));
  }
}
