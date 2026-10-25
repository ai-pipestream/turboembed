// JDK locale lowercase oracle. Run with the pinned goldens container's JDK.
import java.nio.charset.StandardCharsets;
import java.util.HexFormat;
import java.util.Locale;

public final class LocaleLowercase {
  private static String hex(String value) {
    return HexFormat.of().formatHex(value.getBytes(StandardCharsets.UTF_8));
  }
  private static void row(String text) {
    System.out.println(hex(text) + "\t" + hex(text.toLowerCase(Locale.forLanguageTag("tr")))
        + "\t" + hex(text.toLowerCase(Locale.forLanguageTag("az"))));
  }
  public static void main(String[] args) {
    for (String text : new String[] {
        "İstanbul'da IĞDIR otobüsünü bekliyorum.",
        "AZƏRBAYCAN RESPUBLİKASI", "IŞIQ VƏ İSTİLİK", "İyi akşamlar!", ""}) row(text);
    for (String first : new String[] {"I", "İ", "i", "ı", "A", "Σ", "ΟΣ"}) {
      for (String middle : new String[] {"", "\u0307", "\u0323", "\u0301", "\u0315", "\u0327",
          "\u0323\u0301", "\u0301\u0323", " ", "\u034f"}) {
        for (String last : new String[] {"", "\u0307", "\u0307\u0307", "I\u0307", "A", "Σ1a"}) {
          row(first + middle + last);
        }
      }
    }
  }
}
