package ai.pipestream.turbo.goldens;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.nio.charset.StandardCharsets;
import java.util.Arrays;
import java.util.List;
import org.junit.jupiter.api.Test;

/** {@link PiiRewriteGolden}: one run per pack, masks in UTF-8 bytes, rewrites, labels and the audit. */
class PiiRewriteGoldenTest {

  private static Input input(String text) {
    return new Input("pii", "t", text.getBytes(StandardCharsets.UTF_8), text, new Utf8Offsets("pii/t", text));
  }

  private static Layer layer(String pack, String text, String id) {
    PiiRewriteGolden golden = (PiiRewriteGolden) PiiRewriteGolden.all().stream()
        .filter(a -> a.modelDir().equals(pack)).findFirst().orElseThrow();
    return golden.layers(input(text)).stream().filter(l -> l.id().equals(id)).findFirst().orElseThrow();
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
  void thereIsOneRunPerPackOnThePiiSet() {
    List<AnnotatorGolden> all = PiiRewriteGolden.all();
    assertEquals(PiiGolden.PACKS.stream().map(PiiGolden.Pack::dir).toList(),
        all.stream().map(AnnotatorGolden::modelDir).toList());
    for (AnnotatorGolden a : all) {
      assertEquals("pii-rewrite", a.annotator());
      assertEquals(a.modelDir(), a.config().get("pack"));
      assertEquals(PiiRewriteGolden.KEY, a.config().get("key"));
      assertTrue(a.accepts(input("x")));
    }
    assertEquals(List.of("turbo:view", "turbo:pii-mask-all", "turbo:pii-rewrites", "turbo:pii-labels",
        "turbo:pii-audit", "turbo:pii-audit-report"),
        ((PiiRewriteGolden) all.get(0)).layers(input("x")).stream().map(Layer::id).toList());
  }

  @Test
  void aMaskedCharacterIsOneMaskBytePerUtf8Byte() {
    // "é" is two bytes and the emoji four, so the card starts at byte 8; MaskPolicies keeps its
    // separators and last four digits.
    String text = "é 😀 4111 1111 1111 1111.";
    Layer view = layer("payment", text, "turbo:view");
    assertEquals(List.of(8L, 9L, 10L, 11L, 13L), numbers(view, "start").subList(0, 5));
    assertEquals(12, numbers(view, "start").size());
    assertEquals("é 😀 ###################.", strings(layer("payment", text, "turbo:pii-mask-all"), "text").get(0));
    // 𝕒 is two UTF-16 units and four bytes, é one unit and two bytes.
    assertEquals("ab######", PiiRewriteGolden.utf8Masked("ab𝕒é", "ab###", '#'));
    assertEquals("***", PiiRewriteGolden.utf8Masked("a*b", "***", '*'));
  }

  @Test
  void rewritesWriteTheirLabelsOrTheirRefusal() {
    String text = "mail jane@example.com or é jane@example.com";
    Layer rewrites = layer("contact", text, "turbo:pii-rewrites");
    assertEquals(List.of("pseudonymize", "hmac"), strings(rewrites, "operation"));
    assertEquals("mail EMAIL-1 or é EMAIL-1", strings(rewrites, "text").get(0));
    assertEquals(List.of("", ""), strings(rewrites, "error"));
    Layer labels = layer("contact", text, "turbo:pii-labels");
    assertEquals(List.of(5L, 28L, 5L, 28L), numbers(labels, "source_start"));
    assertEquals(List.of(5L, 19L, 5L, 34L), numbers(labels, "start"));
    String overlap = "IBAN DE89 3704 0044 0532 0130 00 received";
    Layer refused = layer("all-structured", overlap, "turbo:pii-rewrites");
    assertEquals(List.of("", ""), strings(refused, "text"));
    assertTrue(strings(refused, "error").get(0).startsWith("mentions must not overlap"));
    assertEquals(0, numbers(layer("all-structured", overlap, "turbo:pii-labels"), "start").size());
  }

  @Test
  void theAuditHoldsTokensNotValues() {
    String text = "jane@example.com, bob@example.com, jane@example.com";
    Layer audit = layer("contact", text, "turbo:pii-audit");
    assertEquals(List.of("email"), strings(audit, "type"));
    assertEquals(List.of(3L), numbers(audit, "mentions"));
    assertEquals(List.of(2L), numbers(audit, "distinct"));
    String report = strings(layer("contact", text, "turbo:pii-audit-report"), "text").get(0);
    assertTrue(report.startsWith("email: 3 mentions, 2 distinct [EMAIL-"), report);
    assertFalse(report.contains("jane") || report.contains("example"), report);
  }
}
