package ai.pipestream.turbo.goldens;

import java.math.BigDecimal;
import java.util.ArrayList;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Set;
import opennlp.tools.document.Annotation;
import opennlp.tools.document.Document;
import opennlp.tools.document.DocumentAnalyzer;
import opennlp.tools.extraction.NumberNotation;
import opennlp.tools.quantity.CursorQuantityExtractor;
import opennlp.tools.quantity.Quantity;
import opennlp.tools.quantity.QuantityAnnotator;

/**
 * The goldens of {@code QuantityAnnotator} over {@code CursorQuantityExtractor} with its default
 * unit set, from the OpenNLP addons {@code numeric} module, in one {@link NumberNotation} each.
 *
 * <p>The layer is {@code turbo:quantities}, with one row per mention and the columns
 * {@code start} and {@code end} (UTF-8 byte offsets of the mention, from the minus sign to the
 * end of the unit), {@code coefficient} (the unscaled value without sign), {@code negative} (1
 * when the value is below zero, which a mention starting with {@code -} need not be),
 * {@code scale}, {@code plain} ({@code toPlainString()} of the value, for a reader of the file)
 * and {@code unit} (the unit token, or {@code %} for a percentage). A value with a negative scale
 * is written with scale 0. The value is {@code (-1)^negative * coefficient / 10^scale}. The path
 * of a golden is {@code quantity/<notation>/quantity/<id>.golden}; the notation takes the place
 * of a model, which the extractor does not have.</p>
 */
final class QuantityGolden implements AnnotatorGolden {

  /** The input set the quantity goldens are made from. */
  static final String INPUT_SET = "quantity";

  /** The name of the golden's annotator directory. */
  static final String ANNOTATOR = "quantity";

  /** The layer the goldens hold: the id of the Rust layer type. */
  static final String LAYER = "turbo:quantities";

  private final NumberNotation notation;

  private QuantityGolden(NumberNotation notation) {
    this.notation = notation;
  }

  /** One annotator per {@link NumberNotation}, in the order of its values. */
  static List<AnnotatorGolden> all() {
    List<AnnotatorGolden> all = new ArrayList<>();
    for (NumberNotation n : NumberNotation.values()) {
      all.add(new QuantityGolden(n));
    }
    return all;
  }

  @Override
  public String annotator() {
    return ANNOTATOR;
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
    return "opennlp.tools.quantity.CursorQuantityExtractor";
  }

  @Override
  public String method() {
    return "extract";
  }

  @Override
  public Map<String, String> config() {
    return Map.of("notation", notation.name());
  }

  @Override
  public Set<String> inputSets() {
    return Set.of(INPUT_SET);
  }

  @Override
  public List<Layer> layers(Input input) {
    Document document = DocumentAnalyzer.builder()
        .add(new QuantityAnnotator(new CursorQuantityExtractor(notation)))
        .build()
        .analyze(input.text());
    List<Annotation<Quantity>> mentions = document.get(QuantityAnnotator.QUANTITIES);
    int n = mentions.size();
    long[] starts = new long[n];
    long[] ends = new long[n];
    String[] coefficients = new String[n];
    long[] negatives = new long[n];
    long[] scales = new long[n];
    String[] plain = new String[n];
    String[] units = new String[n];
    for (int k = 0; k < n; k++) {
      Quantity q = mentions.get(k).value();
      Utf8Offsets.ByteSpan span = input.offsets().span(q.span().getStart(), q.span().getEnd());
      starts[k] = span.start();
      ends[k] = span.end();
      // scale is a u32 column, so a value such as 2E+1 would be written as 20. A value read from
      // text has a scale of zero or more; the rule is the money goldens'.
      BigDecimal value = q.value().scale() < 0 ? q.value().setScale(0) : q.value();
      coefficients[k] = value.unscaledValue().abs().toString();
      negatives[k] = value.signum() < 0 ? 1 : 0;
      scales[k] = value.scale();
      plain[k] = value.toPlainString();
      units[k] = q.unit();
    }
    return List.of(new Layer(LAYER, "text", new Column.U32("start", starts), new Column.U32("end", ends),
        new Column.Str("coefficient", coefficients), new Column.U32("negative", negatives),
        new Column.U32("scale", scales), new Column.Str("plain", plain), new Column.Str("unit", units)));
  }
}
