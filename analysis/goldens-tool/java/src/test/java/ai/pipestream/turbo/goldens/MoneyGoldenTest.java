package ai.pipestream.turbo.goldens;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Arrays;
import java.util.List;
import java.util.Map;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

/** {@link MoneyGolden}: the mentions of each notation, the conversions and the columns written. */
class MoneyGoldenTest {

  /** Rates in the format of the ECB's history file, newest first: Friday 13 March has no CNY quote. */
  private static final String RATES = String.join("\n",
      "Date,USD,GBP,CNY,",
      "2026-03-16,1.5,0.8,8,",
      "2026-03-13,1.25,0.5,N/A,",
      "2026-03-12,1.2,0.6,7,",
      "");

  @TempDir
  Path dir;

  private static Input input(String set, String text) {
    return new Input(set, "t", text.getBytes(StandardCharsets.UTF_8), text, new Utf8Offsets(set + "/t", text));
  }

  /** The annotators in the order of {@code all}: extraction US, EU, then conversion US, EU. */
  private List<AnnotatorGolden> all() throws IOException {
    Path csv = dir.resolve("rates.csv");
    Files.writeString(csv, RATES);
    return MoneyGolden.all(csv);
  }

  private Layer layer(int index, String text) throws IOException {
    return ((MoneyGolden) all().get(index)).layers(input("money", text)).get(0);
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
  void thereAreExtractionAndConversionAnnotatorsPerNotationThatAcceptOnlyTheMoneySet() throws IOException {
    List<AnnotatorGolden> all = all();
    assertEquals(List.of("money", "money", "money-convert", "money-convert"),
        all.stream().map(AnnotatorGolden::annotator).toList());
    assertEquals(List.of("latin-us", "latin-eu", "latin-us", "latin-eu"),
        all.stream().map(AnnotatorGolden::modelDir).toList());
    for (AnnotatorGolden a : all) {
      assertTrue(a.accepts(input("money", "$5")));
      assertFalse(a.accepts(input("num", "$5")));
      assertFalse(a.accepts(input("en", "$5")));
    }
  }

  @Test
  void otherAnnotatorsDoNotAcceptTheMoneySet() {
    AnnotatorGolden withDefaults = new AnnotatorGolden() {
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
    assertFalse(withDefaults.accepts(input("money", "$5")));
    assertTrue(withDefaults.accepts(input("en", "$5")));
  }

  @Test
  void configRecordsTheNotationAndForConversionsTheTargetDateAndSnapshot() throws IOException {
    List<AnnotatorGolden> all = all();
    assertEquals(Map.of("notation", "LATIN_US"), all.get(0).config());
    assertEquals(Map.of("notation", "LATIN_EU"), all.get(1).config());
    assertEquals(Map.of("notation", "LATIN_EU", "target", "EUR", "as_of", "2026-03-15",
        "snapshot", "rates/ecb-snapshot.csv"), all.get(3).config());
  }

  @Test
  void theSameTextReadsDifferentlyInEachNotation() throws IOException {
    assertEquals(List.of("1234.56"), strings(layer(0, "$1,234.56 and 1.234,56 EUR"), "plain"));
    assertEquals(List.of("1234.56"), strings(layer(1, "$1,234.56 and 1.234,56 EUR"), "plain"));
    assertEquals(List.of("USD"), strings(layer(0, "$1,234.56 and 1.234,56 EUR"), "currency"));
    assertEquals(List.of("EUR"), strings(layer(1, "$1,234.56 and 1.234,56 EUR"), "currency"));
  }

  @Test
  void negativeFollowsTheAmountNotTheText() throws IOException {
    // OpenNLP drops the minus before a currency code, and a negated zero is zero.
    Layer us = layer(0, "-USD 5 and -$0.00");
    assertEquals(List.of(0L, 0L), numbers(us, "negative"));
    assertEquals(List.of("5", "0.00"), strings(us, "plain"));
  }

  @Test
  void offsetsAreUtf8BytesAndTheSignIsPartOfTheMention() throws IOException {
    // "é" is two bytes, so the mentions start at bytes 3 and 11.
    Layer us = layer(0, "é -$5 and 7.50€");
    assertEquals(List.of(3L, 11L), numbers(us, "start"));
    assertEquals(List.of(6L, 18L), numbers(us, "end"));
    assertEquals(List.of("5", "750"), strings(us, "coefficient"));
    assertEquals(List.of(1L, 0L), numbers(us, "negative"));
    assertEquals(List.of(0L, 2L), numbers(us, "scale"));
    assertEquals(List.of("-5", "7.50"), strings(us, "plain"));
    assertEquals(List.of("USD", "EUR"), strings(us, "currency"));
  }

  @Test
  void conversionUsesTheLatestRowOnOrBeforeTheDateAndOmitsMentionsWithoutARate() throws IOException {
    // As of Sunday 15 March the Friday row applies: 1 EUR is 1.25 USD and 0.5 GBP, and it has no CNY.
    Layer converted = layer(2, "$5 and 100 CNY and 10 GBP and 3 EUR and 2 CHF");
    assertEquals(List.of("EUR", "EUR", "EUR"), strings(converted, "currency"));
    assertEquals(List.of("4", "20", "3"), strings(converted, "plain"));
    assertEquals(List.of(0L, 19L, 30L), numbers(converted, "start"));
    assertEquals(List.of(2L, 25L, 35L), numbers(converted, "end"));
  }
}
