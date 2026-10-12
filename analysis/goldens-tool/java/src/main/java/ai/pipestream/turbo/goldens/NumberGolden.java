package ai.pipestream.turbo.goldens;

import java.math.BigDecimal;
import java.security.CodeSource;
import java.util.ArrayList;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import opennlp.tools.extraction.NumberNotation;
import opennlp.tools.extraction.NumberScan;

/**
 * The golden of {@code NumberScan.parse} in one {@link NumberNotation}: a {@code turbo:numbers}
 * layer with one row per number mention in the input.
 *
 * <p>The scan loop is the one in {@code CursorQuantityExtractor.extract} and {@code matchAt} of
 * the OpenNLP addons {@code numeric} module, without the exclusion of money spans and without
 * units, and it calls only public methods of {@code NumberScan}. At each offset it takes an
 * optional {@code -}, requires an ASCII digit that does not continue an earlier number and a
 * boundary before the mention (the sign boundary when there is a {@code -}), and parses the
 * number with scale markers on. A mention is the span from the sign to the end of the scaled
 * number. {@code numeral_end} is the end of the same number parsed without scale markers, so
 * the row says where the digits stop and a {@code k} or {@code million} begins.</p>
 *
 * <p>The value is written as its sign, the unscaled value without sign ({@code coefficient}) and
 * the scale, which give {@code (-1)^negative * coefficient / 10^scale} exactly; {@code plain} is
 * {@code toPlainString()}, for a reader of the file. The path of a golden is
 * {@code numbers/<notation>/<set>/<id>.golden}: the notation takes the place of a model, which
 * the scan does not have.</p>
 */
final class NumberGolden implements AnnotatorGolden {

  /** The input set the number goldens are made from. */
  static final String INPUT_SET = "num";

  /** One mention: UTF-16 offsets, whether it has a sign, and its value with the sign applied. */
  private record Mention(int start, int end, int numeralEnd, boolean negative, BigDecimal value) {}

  private final NumberNotation notation;

  private NumberGolden(NumberNotation notation) {
    this.notation = notation;
  }

  /**
   * One annotator per notation, in the order of the enum. Fails when {@code NumberScan} was not
   * loaded from the {@code numeric} module's jar: the OpenNLP runtime of the fork has classes of
   * the same names with fewer features, and the goldens must come from the module's.
   */
  static List<AnnotatorGolden> all() {
    requireNumericModule();
    List<AnnotatorGolden> all = new ArrayList<>();
    for (NumberNotation n : NumberNotation.values()) {
      all.add(new NumberGolden(n));
    }
    return all;
  }

  private static void requireNumericModule() {
    CodeSource source = NumberScan.class.getProtectionDomain().getCodeSource();
    String location = source == null ? "" : source.getLocation().getPath();
    String file = location.substring(location.lastIndexOf('/') + 1);
    if (!file.startsWith("numeric-")) {
      throw new IllegalStateException("NumberScan was loaded from " + location
          + ", not from the numeric module's jar; the module must come before the OpenNLP runtime on the class path");
    }
  }

  @Override
  public String annotator() {
    return "numbers";
  }

  @Override
  public List<ModelRef> models() {
    return List.of();
  }

  /** The notation's name in lower case with a hyphen, such as {@code latin-us}. */
  @Override
  public String modelDir() {
    return notation.name().toLowerCase(Locale.ROOT).replace('_', '-');
  }

  @Override
  public String annotatorClass() {
    return NumberScan.class.getName();
  }

  @Override
  public String method() {
    return "parse";
  }

  @Override
  public Map<String, String> config() {
    return Map.of("notation", notation.name());
  }

  @Override
  public boolean accepts(Input input) {
    return input.set().equals(INPUT_SET);
  }

  @Override
  public List<Layer> layers(Input input) {
    List<Mention> mentions = scan(input);
    int n = mentions.size();
    long[] starts = new long[n];
    long[] ends = new long[n];
    long[] numeralEnds = new long[n];
    long[] negatives = new long[n];
    String[] coefficients = new String[n];
    long[] scales = new long[n];
    String[] plain = new String[n];
    for (int k = 0; k < n; k++) {
      Mention m = mentions.get(k);
      Utf8Offsets.ByteSpan span = input.offsets().span(m.start(), m.end());
      starts[k] = span.start();
      ends[k] = span.end();
      numeralEnds[k] = input.offsets().span(m.numeralEnd(), m.numeralEnd()).start();
      negatives[k] = m.negative() ? 1 : 0;
      if (m.value().scale() < 0) {
        throw new IllegalStateException(label(input) + ": the value " + m.value() + " has a negative scale");
      }
      coefficients[k] = m.value().unscaledValue().abs().toString();
      scales[k] = m.value().scale();
      plain[k] = m.value().toPlainString();
    }
    return List.of(new Layer("turbo:numbers", "text", new Column.U32("start", starts), new Column.U32("end", ends),
        new Column.U32("numeral_end", numeralEnds), new Column.U32("negative", negatives),
        new Column.Str("coefficient", coefficients), new Column.U32("scale", scales), new Column.Str("plain", plain)));
  }

  private List<Mention> scan(Input input) {
    String text = input.text();
    List<Mention> mentions = new ArrayList<>();
    int i = 0;
    while (i < text.length()) {
      int start = i;
      boolean negative = NumberScan.charAt(text, i) == '-';
      int j = negative ? i + 1 : i;
      if (NumberScan.isAsciiDigit(NumberScan.charAt(text, j))
          && !NumberScan.continuesNumber(text, j, notation)
          && (negative ? NumberScan.signBoundaryBefore(text, start) : NumberScan.boundaryBefore(text, start))) {
        NumberScan.Result scaled = NumberScan.parse(text, j, true, notation);
        if (scaled != null) {
          NumberScan.Result numeral = NumberScan.parse(text, j, false, notation);
          if (numeral == null) {
            throw new IllegalStateException(label(input) + ": parse with scale markers found a number at " + j
                + " but parse without them did not");
          }
          BigDecimal value = negative ? scaled.value().negate() : scaled.value();
          mentions.add(new Mention(start, scaled.end(), numeral.end(), negative, value));
          i = scaled.end();
          continue;
        }
      }
      i += Character.charCount(NumberScan.codePointAt(text, i));
    }
    return mentions;
  }

  private String label(Input input) {
    return "numbers/" + modelDir() + " on " + input.inputId();
  }
}
