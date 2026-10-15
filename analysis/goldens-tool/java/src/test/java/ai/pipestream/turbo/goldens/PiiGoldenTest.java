package ai.pipestream.turbo.goldens;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.nio.charset.StandardCharsets;
import java.util.Arrays;
import java.util.List;
import java.util.Map;
import org.junit.jupiter.api.Test;

/** {@link PiiGolden}: one run per pack, the mentions in order and UTF-8 offsets. */
class PiiGoldenTest {

  private static Input input(String set, String text) {
    return new Input(set, "t", text.getBytes(StandardCharsets.UTF_8), text, new Utf8Offsets(set + "/t", text));
  }

  private static Layer layer(String pack, String text) {
    PiiGolden golden = (PiiGolden) PiiGolden.all().stream().filter(a -> a.modelDir().equals(pack)).findFirst()
        .orElseThrow();
    return golden.layers(input("pii", text)).get(0);
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
  void thereIsOneRunPerPackThatAcceptsOnlyThePiiSet() {
    List<AnnotatorGolden> all = PiiGolden.all();
    assertEquals(List.of("payment", "contact", "network", "secrets", "crypto", "us-identity", "eu-identity",
        "ca-identity", "device", "all-structured"), all.stream().map(AnnotatorGolden::modelDir).toList());
    for (AnnotatorGolden a : all) {
      assertEquals("pii", a.annotator());
      assertEquals(Map.of("pack", a.modelDir()), a.config());
      assertTrue(a.accepts(input("pii", "x")));
      assertFalse(a.accepts(input("en", "x")));
    }
  }

  @Test
  void anIbanHoldsAPhoneAndBothAreWritten() {
    Layer contact = layer("all-structured", "IBAN DE89 3704 0044 0532 0130 00 received");
    assertEquals(List.of("iban", "phone"), strings(contact, "type"));
    assertEquals(List.of("DE89370400440532013000", "0532013000"), strings(contact, "normalized"));
    assertEquals(List.of(5L, 20L), numbers(contact, "start"));
    assertEquals(List.of(32L, 32L), numbers(contact, "end"));
  }

  @Test
  void offsetsAreUtf8Bytes() {
    // "é" is two bytes and the emoji four, so the address starts at byte 8.
    Layer contact = layer("contact", "é 😀 jane@Example.COM.");
    assertEquals(List.of(8L), numbers(contact, "start"));
    assertEquals(List.of(24L), numbers(contact, "end"));
    assertEquals(List.of("jane@example.com"), strings(contact, "normalized"));
  }

  @Test
  void aPackFindsOnlyItsKinds() {
    String text = "mail jane@example.com host 192.0.2.1 key AKIAIOSFODNN7EXAMPLE";
    assertEquals(List.of("ipv4"), strings(layer("network", text), "type"));
    assertEquals(List.of("aws-access-key"), strings(layer("secrets", text), "type"));
    assertEquals(List.of("email", "ipv4", "aws-access-key"), strings(layer("all-structured", text), "type"));
  }
}
