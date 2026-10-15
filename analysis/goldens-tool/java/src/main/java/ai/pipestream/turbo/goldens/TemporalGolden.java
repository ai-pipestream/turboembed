package ai.pipestream.turbo.goldens;

import java.time.LocalDate;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.Set;
import opennlp.tools.document.Annotation;
import opennlp.tools.document.Document;
import opennlp.tools.document.DocumentAnalyzer;
import opennlp.tools.temporal.CursorTemporalExtractor;
import opennlp.tools.temporal.DocumentDateAnnotator;
import opennlp.tools.temporal.TemporalAnnotator;
import opennlp.tools.temporal.TemporalExpression;

/**
 * The goldens of {@code TemporalAnnotator} over {@code CursorTemporalExtractor} followed by
 * {@code DocumentDateAnnotator}, from the OpenNLP addons {@code numeric} module, with the
 * reference date taken from each text's first absolute day ({@code dateline}) or fixed at
 * {@link #REFERENCE} ({@code fixed}).
 *
 * <p>Two layers are written. {@code turbo:temporal} has one row per mention with the columns
 * {@code start} and {@code end} (UTF-8 byte offsets of the mention), {@code value} (what
 * {@code TemporalExpression.value()} holds, such as {@code 2026-07-14}, {@code 2026-W29},
 * {@code 2026-07}, {@code 2024-Q3} or {@code 2026}), {@code granularity} and {@code origin} (the
 * enum constants' names). {@code turbo:document-date}, anchored on the document, has no row or
 * one: {@code date} ({@code LocalDate.toString()}) and {@code mention}, the index in
 * {@code turbo:temporal} of the mention whose span the date annotation has. The path of a golden
 * is {@code temporal/<mode>/temporal/<id>.golden}; the mode takes the place of a model, which the
 * extractor does not have.</p>
 */
final class TemporalGolden implements AnnotatorGolden {

  /** The input set the temporal goldens are made from. */
  static final String INPUT_SET = "temporal";

  /** The name of the golden's annotator directory. */
  static final String ANNOTATOR = "temporal";

  /** The layers the goldens hold: the ids of the Rust layer types. */
  static final String LAYER = "turbo:temporal";
  static final String DATE_LAYER = "turbo:document-date";

  /** The reference date of the {@code fixed} mode. */
  static final LocalDate REFERENCE = LocalDate.of(2026, 3, 15);

  /** The reference date of the run, or {@code null} to take it from the text. */
  private final LocalDate reference;

  private TemporalGolden(LocalDate reference) {
    this.reference = reference;
  }

  /** The {@code dateline} run, then the {@code fixed} run. */
  static List<AnnotatorGolden> all() {
    return List.of(new TemporalGolden(null), new TemporalGolden(REFERENCE));
  }

  @Override
  public String annotator() {
    return ANNOTATOR;
  }

  @Override
  public List<ModelRef> models() {
    return List.of();
  }

  /** {@code dateline} or {@code fixed}. */
  @Override
  public String modelDir() {
    return reference == null ? "dateline" : "fixed";
  }

  @Override
  public String annotatorClass() {
    return "opennlp.tools.temporal.CursorTemporalExtractor";
  }

  @Override
  public String method() {
    return "extract";
  }

  /** The mode, and for the fixed mode the reference date. */
  @Override
  public Map<String, String> config() {
    return reference == null ? Map.of("reference", "dateline") : Map.of("reference", reference.toString());
  }

  @Override
  public Set<String> inputSets() {
    return Set.of(INPUT_SET);
  }

  @Override
  public List<Layer> layers(Input input) {
    CursorTemporalExtractor extractor = new CursorTemporalExtractor();
    TemporalAnnotator temporal =
        reference == null ? new TemporalAnnotator(extractor) : new TemporalAnnotator(extractor, reference);
    Document document = DocumentAnalyzer.builder()
        .add(temporal)
        .add(new DocumentDateAnnotator())
        .build()
        .analyze(input.text());
    List<Annotation<TemporalExpression>> mentions = document.get(TemporalAnnotator.TEMPORALS);
    int n = mentions.size();
    long[] starts = new long[n];
    long[] ends = new long[n];
    String[] values = new String[n];
    String[] granularities = new String[n];
    String[] origins = new String[n];
    for (int k = 0; k < n; k++) {
      TemporalExpression t = mentions.get(k).value();
      Utf8Offsets.ByteSpan span = input.offsets().span(t.span().getStart(), t.span().getEnd());
      starts[k] = span.start();
      ends[k] = span.end();
      values[k] = t.value();
      granularities[k] = t.granularity().name();
      origins[k] = t.origin().name();
    }
    List<String> dates = new ArrayList<>();
    List<Long> indexes = new ArrayList<>();
    for (Annotation<LocalDate> date : document.get(DocumentDateAnnotator.DOCUMENT_DATE)) {
      int index = -1;
      for (int k = 0; k < n; k++) {
        if (mentions.get(k).span().equals(date.span())) {
          index = k;
          break;
        }
      }
      if (index < 0) {
        throw new IllegalStateException("the document date's span is no temporal mention's: " + date.span());
      }
      dates.add(date.value().toString());
      indexes.add((long) index);
    }
    return List.of(
        new Layer(LAYER, "text", new Column.U32("start", starts), new Column.U32("end", ends),
            new Column.Str("value", values), new Column.Str("granularity", granularities),
            new Column.Str("origin", origins)),
        new Layer(DATE_LAYER, "document", new Column.Str("date", dates.toArray(String[]::new)),
            new Column.U32("mention", indexes.stream().mapToLong(Long::longValue).toArray())));
  }
}
