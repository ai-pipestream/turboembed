package ai.pipestream.turbo.goldens;

import java.io.IOException;
import java.math.BigDecimal;
import java.nio.file.Path;
import java.time.LocalDate;
import java.util.ArrayList;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Set;
import opennlp.tools.document.Annotation;
import opennlp.tools.document.Document;
import opennlp.tools.document.DocumentAnalyzer;
import opennlp.tools.extraction.NumberNotation;
import opennlp.tools.money.CursorMoneyExtractor;
import opennlp.tools.money.EcbFxRates;
import opennlp.tools.money.MoneyAmount;
import opennlp.tools.money.MoneyAnnotator;
import opennlp.tools.money.MoneyConversionAnnotator;

/**
 * The goldens of the money annotators of the OpenNLP addons {@code numeric} module, in one
 * {@link NumberNotation} each:
 *
 * <ul>
 *   <li>{@link Kind#EXTRACT}: {@code MoneyAnnotator} over {@code CursorMoneyExtractor}, written
 *   as a {@code turbo:money} layer with one row per mention;</li>
 *   <li>{@link Kind#CONVERT}: the same followed by {@code MoneyConversionAnnotator} into
 *   {@link #TARGET} as of {@link #AS_OF}, with the rates of the snapshot file, written as a
 *   {@code turbo:money} layer too, the layer the Rust conversion writes under its own key. A mention without a usable rate has no row, as in the
 *   annotator's own layer.</li>
 * </ul>
 *
 * <p>Both layers have the columns {@code start} and {@code end} (UTF-8 byte offsets of the
 * mention, from the minus sign or symbol to its last character), {@code coefficient} (the
 * unscaled value of the amount without sign), {@code negative} (1 when the amount is below
 * zero, which a mention starting with {@code -} need not be), {@code scale}, {@code plain} ({@code toPlainString()} of the amount, for a reader of
 * the file) and {@code currency} (the ISO 4217 code). An amount with a negative scale, such as
 * {@code 2E+1}, is written with scale 0. The amount is
 * {@code (-1)^negative * coefficient / 10^scale}. The path of a golden is
 * {@code <annotator>/<notation>/<set>/<id>.golden} with the annotator {@code money} or
 * {@code money-convert}; the notation takes the place of a model, which these annotators do not
 * have.</p>
 *
 * <p>The rate snapshot is a hand made file in the format of the ECB's {@code eurofxref-hist.csv},
 * with invented values (see {@code analysis/goldens/data/README.md}). The generator reads it from
 * {@code rates/ecb-snapshot.csv} beside the inputs directory, and the goldens record the path
 * relative to the data directory, the target and the as-of date in {@code config} records.</p>
 */
final class MoneyGolden implements AnnotatorGolden {

  /** The input set the money goldens are made from. */
  static final String INPUT_SET = "money";

  /** The directory beside the inputs directory that holds the snapshot, and the snapshot's name in it. */
  static final String SNAPSHOT_DIR = "rates";
  static final String SNAPSHOT_FILE = "ecb-snapshot.csv";

  /** The currency every conversion golden restates the amounts in. */
  static final String TARGET = "EUR";

  /** The date of the conversions: a Sunday, so the snapshot's Friday row is the one used. */
  static final LocalDate AS_OF = LocalDate.of(2026, 3, 15);

  /** What a golden holds. */
  enum Kind {
    EXTRACT("money", "turbo:money", "opennlp.tools.money.CursorMoneyExtractor", "extract"),
    CONVERT("money-convert", "turbo:money", "opennlp.tools.money.EcbFxRates", "convert");

    private final String annotator;
    private final String layer;
    private final String annotatorClass;
    private final String method;

    Kind(String annotator, String layer, String annotatorClass, String method) {
      this.annotator = annotator;
      this.layer = layer;
      this.annotatorClass = annotatorClass;
      this.method = method;
    }
  }

  private final Kind kind;
  private final NumberNotation notation;
  private final EcbFxRates rates;

  private MoneyGolden(Kind kind, NumberNotation notation, EcbFxRates rates) {
    this.kind = kind;
    this.notation = notation;
    this.rates = rates;
  }

  /**
   * The extraction annotators, then the conversion annotators, each in the order of the
   * {@link NumberNotation} values. The snapshot is read once, from the file named.
   */
  static List<AnnotatorGolden> all(Path snapshot) throws IOException {
    EcbFxRates rates = EcbFxRates.load(snapshot);
    List<AnnotatorGolden> all = new ArrayList<>();
    for (Kind kind : Kind.values()) {
      for (NumberNotation n : NumberNotation.values()) {
        all.add(new MoneyGolden(kind, n, rates));
      }
    }
    return all;
  }

  @Override
  public String annotator() {
    return kind.annotator;
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
    return kind.annotatorClass;
  }

  @Override
  public String method() {
    return kind.method;
  }

  /** The notation, and for conversions also the target, the as-of date and the snapshot's path. */
  @Override
  public Map<String, String> config() {
    if (kind == Kind.EXTRACT) {
      return Map.of("notation", notation.name());
    }
    return Map.of("notation", notation.name(), "target", TARGET, "as_of", AS_OF.toString(),
        "snapshot", SNAPSHOT_DIR + "/" + SNAPSHOT_FILE);
  }

  @Override
  public Set<String> inputSets() {
    return Set.of(INPUT_SET);
  }

  @Override
  public List<Layer> layers(Input input) {
    DocumentAnalyzer.Builder builder = DocumentAnalyzer.builder()
        .add(new MoneyAnnotator(new CursorMoneyExtractor(notation)));
    if (kind == Kind.CONVERT) {
      builder.add(new MoneyConversionAnnotator(rates, TARGET, AS_OF));
    }
    Document document = builder.build().analyze(input.text());
    List<Annotation<MoneyAmount>> mentions = document.get(
        kind == Kind.EXTRACT ? MoneyAnnotator.MONEY : MoneyConversionAnnotator.CONVERTED_MONEY);
    return List.of(layer(input, mentions));
  }

  private Layer layer(Input input, List<Annotation<MoneyAmount>> mentions) {
    int n = mentions.size();
    long[] starts = new long[n];
    long[] ends = new long[n];
    String[] coefficients = new String[n];
    long[] negatives = new long[n];
    long[] scales = new long[n];
    String[] plain = new String[n];
    String[] currencies = new String[n];
    for (int k = 0; k < n; k++) {
      MoneyAmount m = mentions.get(k).value();
      int start = m.span().getStart();
      Utf8Offsets.ByteSpan span = input.offsets().span(start, m.span().getEnd());
      starts[k] = span.start();
      ends[k] = span.end();
      // A quotient can come out as 2E+1, with a negative scale; scale is a u32 column, so it is
      // written as 20. The value is the same.
      BigDecimal amount = m.amount().scale() < 0 ? m.amount().setScale(0) : m.amount();
      coefficients[k] = amount.unscaledValue().abs().toString();
      negatives[k] = amount.signum() < 0 ? 1 : 0;
      scales[k] = amount.scale();
      plain[k] = amount.toPlainString();
      currencies[k] = m.currency();
    }
    return new Layer(kind.layer, "text", new Column.U32("start", starts), new Column.U32("end", ends),
        new Column.Str("coefficient", coefficients), new Column.U32("negative", negatives),
        new Column.U32("scale", scales), new Column.Str("plain", plain), new Column.Str("currency", currencies));
  }
}
