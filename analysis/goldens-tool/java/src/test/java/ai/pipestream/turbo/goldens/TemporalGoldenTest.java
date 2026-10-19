package ai.pipestream.turbo.goldens;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.nio.charset.StandardCharsets;
import java.util.Arrays;
import java.util.List;
import java.util.Map;
import org.junit.jupiter.api.Test;

/** {@link TemporalGolden}: the two modes, the layers and their columns. */
class TemporalGoldenTest {

  private static Input input(String set, String text) {
    return new Input(set, "t", text.getBytes(StandardCharsets.UTF_8), text, new Utf8Offsets(set + "/t", text));
  }

  /** The layers of the run at {@code index} of {@code all}: 0 is dateline, 1 is fixed. */
  private static List<Layer> layers(int index, String text) {
    return ((TemporalGolden) TemporalGolden.all().get(index)).layers(input("temporal", text));
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
  void thereIsADatelineAndAFixedRunThatAcceptOnlyTheTemporalSet() {
    List<AnnotatorGolden> all = TemporalGolden.all();
    assertEquals(List.of("temporal", "temporal"), all.stream().map(AnnotatorGolden::annotator).toList());
    assertEquals(List.of("dateline", "fixed"), all.stream().map(AnnotatorGolden::modelDir).toList());
    for (AnnotatorGolden a : all) {
      assertTrue(a.accepts(input("temporal", "today")));
      assertFalse(a.accepts(input("money", "today")));
      assertFalse(a.accepts(input("en", "today")));
      assertTrue(a.models().isEmpty());
    }
  }

  @Test
  void configRecordsTheReference() {
    List<AnnotatorGolden> all = TemporalGolden.all();
    assertEquals(Map.of("reference", "dateline"), all.get(0).config());
    assertEquals(Map.of("reference", "2026-03-15"), all.get(1).config());
  }

  @Test
  void theLayersAreTheRustLayersWithTheirColumns() {
    List<Layer> layers = layers(0, "July 2026");
    assertEquals(List.of("turbo:temporal", "turbo:document-date"), layers.stream().map(Layer::id).toList());
    assertEquals("text", layers.get(0).anchor());
    assertEquals("document", layers.get(1).anchor());
    assertEquals(List.of("start", "end", "value", "granularity", "origin"),
        layers.get(0).columns().stream().map(Column::name).toList());
    assertEquals(List.of("date", "mention"), layers.get(1).columns().stream().map(Column::name).toList());
    // A month dates nothing.
    assertEquals(0, layers.get(1).rows());
  }

  @Test
  void theDatelineResolvesRelativesAndDatesTheDocument() {
    // "é" is two bytes, so the date starts at byte 15 and "yesterday" at byte 34.
    List<Layer> layers = layers(0, "Berlín, filed 14 July 2026, paid yesterday.");
    Layer temporal = layers.get(0);
    assertEquals(List.of("2026-07-14", "2026-07-13"), strings(temporal, "value"));
    assertEquals(List.of(15L, 34L), numbers(temporal, "start"));
    assertEquals(List.of(27L, 43L), numbers(temporal, "end"));
    assertEquals(List.of("DAY", "DAY"), strings(temporal, "granularity"));
    assertEquals(List.of("ABSOLUTE", "RELATIVE"), strings(temporal, "origin"));
    assertEquals(List.of("2026-07-14"), strings(layers.get(1), "date"));
    assertEquals(List.of(0L), numbers(layers.get(1), "mention"));
  }

  @Test
  void theFixedReferenceWinsAndTheDateIsTheFirstAbsoluteDay() {
    List<Layer> layers = layers(1, "Next week, not 2026-07-14: in 2 quarters.");
    assertEquals(List.of("2026-W12", "2026-07-14", "2026-Q3"), strings(layers.get(0), "value"));
    assertEquals(List.of("RELATIVE", "ABSOLUTE", "RELATIVE"), strings(layers.get(0), "origin"));
    assertEquals(List.of("2026-07-14"), strings(layers.get(1), "date"));
    assertEquals(List.of(1L), numbers(layers.get(1), "mention"));
  }
}
