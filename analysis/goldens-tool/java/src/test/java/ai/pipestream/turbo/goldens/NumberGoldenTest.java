package ai.pipestream.turbo.goldens;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.nio.charset.StandardCharsets;
import java.util.Arrays;
import java.util.List;
import org.junit.jupiter.api.Test;

/** {@link NumberGolden}: the mentions of the scan loop and the columns written for them. */
class NumberGoldenTest {

  private static Input input(String set, String text) {
    return new Input(set, "t", text.getBytes(StandardCharsets.UTF_8), text, new Utf8Offsets(set + "/t", text));
  }

  /** The layer of the annotator at {@code notation} (0 for US, 1 for EU) on {@code text}. */
  private static Layer layer(int notation, String text) {
    return ((NumberGolden) NumberGolden.all().get(notation)).layers(input("num", text)).get(0);
  }

  private static List<String> strings(Layer layer, String column) {
    return layer.columns().stream().filter(c -> c.name().equals(column))
        .map(c -> List.of(((Column.Str) c).values())).findFirst().orElseThrow();
  }

  private static List<Long> numbers(Layer layer, String column) {
    return layer.columns().stream().filter(c -> c.name().equals(column))
        .map(c -> Arrays.stream(((Column.U32) c).values()).boxed().toList()).findFirst().orElseThrow();
  }

  @Test
  void thereIsOneAnnotatorPerNotationAndItAcceptsOnlyTheNumberSet() {
    List<AnnotatorGolden> all = NumberGolden.all();
    assertEquals(List.of("latin-us", "latin-eu"), all.stream().map(AnnotatorGolden::modelDir).toList());
    assertTrue(all.get(0).accepts(input("num", "5")));
    assertFalse(all.get(0).accepts(input("en", "5")));
    assertFalse(annotatorWithDefaults().accepts(input("num", "5")));
    assertTrue(annotatorWithDefaults().accepts(input("en", "5")));
  }

  /** An annotator that keeps the default {@code accepts}, as the sentence, token and language ones do. */
  private static AnnotatorGolden annotatorWithDefaults() {
    return new AnnotatorGolden() {
      @Override
      public String annotator() {
        return "x";
      }

      @Override
      public List<ModelRef> models() {
        return List.of();
      }

      @Override
      public String annotatorClass() {
        return "x";
      }

      @Override
      public String method() {
        return "x";
      }

      @Override
      public List<Layer> layers(Input input) {
        return List.of();
      }
    };
  }

  @Test
  void theSameTextReadsDifferentlyInEachNotation() {
    assertEquals(List.of("1.234", "1234"), strings(layer(0, "1.234 1,234"), "plain"));
    assertEquals(List.of("1234", "1.234"), strings(layer(1, "1.234 1,234"), "plain"));
  }

  @Test
  void aScaleWordAfterOneWhitespaceCharacterEndsTheMentionButNotTheNumeral() {
    Layer us = layer(0, "a 5 million b");
    assertEquals(List.of("5000000"), strings(us, "plain"));
    assertEquals(List.of(2L), numbers(us, "start"));
    // "5", a two byte no-break space and "million".
    assertEquals(List.of(2L + 1 + 2 + 7), numbers(us, "end"));
    assertEquals(List.of(3L), numbers(us, "numeral_end"));
  }

  @Test
  void aSignNeedsABoundaryAndOffsetsAreUtf8Bytes() {
    Layer us = layer(0, "é -5 a-6 -0.50");
    assertEquals(List.of("-5", "6", "-0.50"), strings(us, "plain"));
    assertEquals(List.of("5", "6", "50"), strings(us, "coefficient"));
    assertEquals(List.of(3L, 8L, 10L), numbers(us, "start"));
    assertEquals(List.of(1L, 0L, 1L), numbers(us, "negative"));
    assertEquals(List.of(0L, 0L, 2L), numbers(us, "scale"));
  }
}
