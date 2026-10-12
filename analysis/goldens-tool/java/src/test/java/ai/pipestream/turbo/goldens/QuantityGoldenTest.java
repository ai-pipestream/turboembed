package ai.pipestream.turbo.goldens;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.nio.charset.StandardCharsets;
import java.util.Arrays;
import java.util.List;
import java.util.Map;
import java.util.Set;
import org.junit.jupiter.api.Test;

/** {@link QuantityGolden}: the mentions of each notation and the columns written. */
class QuantityGoldenTest {

  private static Input input(String set, String text) {
    return new Input(set, "t", text.getBytes(StandardCharsets.UTF_8), text, new Utf8Offsets(set + "/t", text));
  }

  /** The layer of the annotator at {@code index} of {@code all}: 0 is US, 1 is EU. */
  private static Layer layer(int index, String text) {
    return ((QuantityGolden) QuantityGolden.all().get(index)).layers(input("quantity", text)).get(0);
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
  void thereIsAnAnnotatorPerNotationThatAcceptsOnlyTheQuantitySet() {
    List<AnnotatorGolden> all = QuantityGolden.all();
    assertEquals(List.of("quantity", "quantity"), all.stream().map(AnnotatorGolden::annotator).toList());
    assertEquals(List.of("latin-us", "latin-eu"), all.stream().map(AnnotatorGolden::modelDir).toList());
    for (AnnotatorGolden a : all) {
      assertTrue(a.accepts(input("quantity", "5 kg")));
      assertFalse(a.accepts(input("money", "5 kg")));
      assertFalse(a.accepts(input("num", "5 kg")));
      assertFalse(a.accepts(input("en", "5 kg")));
      assertTrue(a.models().isEmpty());
    }
  }

  @Test
  void otherAnnotatorsDoNotAcceptTheQuantitySet() {
    AnnotatorGolden withDefaults = new AnnotatorGolden() {
      @Override
      public Set<String> inputSets() {
        return Set.of("en");
      }

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
    assertFalse(withDefaults.accepts(input("quantity", "5 kg")));
    assertTrue(withDefaults.accepts(input("en", "5 kg")));
  }

  @Test
  void configRecordsTheNotation() {
    List<AnnotatorGolden> all = QuantityGolden.all();
    assertEquals(Map.of("notation", "LATIN_US"), all.get(0).config());
    assertEquals(Map.of("notation", "LATIN_EU"), all.get(1).config());
  }

  @Test
  void theLayerIsTurboQuantitiesWithItsColumns() {
    Layer layer = layer(0, "5 kg");
    assertEquals("turbo:quantities", layer.id());
    assertEquals(List.of("start", "end", "coefficient", "negative", "scale", "plain", "unit"),
        layer.columns().stream().map(Column::name).toList());
  }

  @Test
  void theSameTextReadsDifferentlyInEachNotation() {
    assertEquals(List.of("1250", "1.250"), strings(layer(0, "1,250 GB and 1.250 GB"), "plain"));
    assertEquals(List.of("1.250", "1250"), strings(layer(1, "1,250 GB and 1.250 GB"), "plain"));
  }

  @Test
  void negativeFollowsTheValueNotTheText() {
    // A negated zero is zero, and a hyphen after a letter is no minus.
    Layer us = layer(0, "-0% and -5 kg and a-5 kg");
    assertEquals(List.of(0L, 1L, 0L), numbers(us, "negative"));
    assertEquals(List.of("0", "-5", "5"), strings(us, "plain"));
  }

  @Test
  void offsetsAreUtf8BytesAndPercentagesHaveThePercentUnit() {
    // "é" is two bytes, so the mentions start at bytes 3 and 13.
    Layer us = layer(0, "é -2.5km and 50 percent and $3m");
    assertEquals(List.of(3L, 14L), numbers(us, "start"));
    assertEquals(List.of(9L, 24L), numbers(us, "end"));
    assertEquals(List.of("25", "50"), strings(us, "coefficient"));
    assertEquals(List.of(1L, 0L), numbers(us, "negative"));
    assertEquals(List.of(1L, 0L), numbers(us, "scale"));
    assertEquals(List.of("-2.5", "50"), strings(us, "plain"));
    assertEquals(List.of("km", "%"), strings(us, "unit"));
  }
}
