package ai.pipestream.turbo.goldens;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;

import java.nio.charset.StandardCharsets;
import java.util.List;
import java.util.Map;
import java.util.Set;
import org.junit.jupiter.api.Test;

/** The BPE trainer's golden on a corpus whose merges have no ties. */
class BpeGoldenTest {

  private static Input input(String text) {
    return new Input("s", "x", text.getBytes(StandardCharsets.UTF_8), text, new Utf8Offsets("s/x", text));
  }

  private static String[] str(Layer layer, int column) {
    return ((Column.Str) layer.columns().get(column)).values();
  }

  private static long[] u32(Layer layer, int column) {
    return ((Column.U32) layer.columns().get(column)).values();
  }

  @Test
  void theRunIsNamedByItsNumberOfMerges() {
    BpeGolden g = new BpeGolden(3, Set.of("s"));
    assertEquals("bpe", g.annotator());
    assertEquals("merges-3", g.modelDir());
    assertEquals("opennlp.tools.tokenize.BPETokenizerTrainer", g.annotatorClass());
    assertEquals(Map.of("languageCode", "en", "numMerges", "3"), g.config());
  }

  @Test
  void theMergesAreTheDocumentLayerAndTheTokensFollow() throws AnnotatorException {
    // ab</w> occurs four times, bc</w> three and ab twice: a b</w>, then b c</w>.
    List<Layer> layers = new BpeGolden(2, Set.of("s")).layers(input("ab ab ab ab abc abc bc"));
    assertEquals(2, layers.size());
    Layer merges = layers.get(0);
    assertEquals("turbo:bpe-merges", merges.id());
    assertEquals("document", merges.anchor());
    assertArrayEquals(new String[] {"a", "b"}, str(merges, 0));
    assertArrayEquals(new String[] {"b</w>", "c</w>"}, str(merges, 1));
    Layer tokens = layers.get(1);
    assertEquals("opennlp:tokens", tokens.id());
    // ab four times, then a bc twice, then bc.
    assertArrayEquals(new long[] {0, 3, 6, 9, 12, 13, 16, 17, 20}, u32(tokens, 0));
    assertArrayEquals(new long[] {2, 5, 8, 11, 13, 15, 17, 19, 22}, u32(tokens, 1));
  }
}
