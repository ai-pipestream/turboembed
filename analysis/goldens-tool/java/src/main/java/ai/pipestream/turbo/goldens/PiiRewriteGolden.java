package ai.pipestream.turbo.goldens;

import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;
import opennlp.tools.document.Annotation;
import opennlp.tools.document.Document;
import opennlp.tools.pii.HmacTokenizer;
import opennlp.tools.pii.MaskPolicies;
import opennlp.tools.pii.Masker;
import opennlp.tools.pii.PiiAnnotator;
import opennlp.tools.pii.PiiAuditReport;
import opennlp.tools.pii.PiiMention;
import opennlp.tools.pii.PiiRewrite;
import opennlp.tools.pii.Pseudonymizer;
import opennlp.tools.util.Span;

/**
 * What OpenNLP does with the mentions {@code PiiAnnotator} finds with each pack of
 * {@code PiiPacks}: masking, numbered and keyed labels, and the audit report, one run per pack
 * over the {@code pii} input set, under a fixed test key.
 *
 * <p>The path of a golden is {@code pii-rewrite/<pack>/pii/<id>.golden}. Its layers, in order:</p>
 * <ul>
 *   <li>{@code turbo:view}: the edits of {@code Masker.mask(document, PiiAnnotator.PII,
 *       MaskPolicies.byType())}, one row per character the mask changed, with UTF-8 byte offsets
 *       ({@code start}, {@code end}) and the mask it became ({@code with});</li>
 *   <li>{@code turbo:pii-mask-all}: one row, the text of {@code Masker.mask(document,
 *       PiiAnnotator.PII, '#')};</li>
 *   <li>{@code turbo:pii-rewrites}: a row for {@code new Pseudonymizer().rewrite(document)}
 *       ({@code pseudonymize}) and one for {@code new HmacTokenizer(key).rewrite(document)}
 *       ({@code hmac}): the new text, or the message of the exception the rewrite threw;</li>
 *   <li>{@code turbo:pii-labels}: per label of each rewrite that did not throw, the replaced
 *       mention's UTF-8 span in the input, the label's UTF-8 span in the new text and the
 *       label;</li>
 *   <li>{@code turbo:pii-audit}: per type of {@code PiiAuditReport.of(document, tokenizer)}, its
 *       mentions, its distinct values and its sample tokens joined with {@code ", "};</li>
 *   <li>{@code turbo:pii-audit-report}: one row, the report's {@code toString()}.</li>
 * </ul>
 *
 * <p>OpenNLP masks one UTF-16 unit with one mask character. The goldens write a masked character
 * as one mask byte per UTF-8 byte of the character, which is what the Rust port writes, so the
 * masked text keeps the input's byte length. Which characters are masked is OpenNLP's.</p>
 */
final class PiiRewriteGolden implements AnnotatorGolden {

  /** The name of the golden's annotator directory. */
  static final String ANNOTATOR = "pii-rewrite";

  /**
   * The test key: 32 ASCII bytes, the shortest key {@code HmacTokenizer} takes. It is written in
   * the {@code config key} record; it protects nothing.
   */
  static final String KEY = "0123456789abcdef0123456789abcdef";

  /** The mask {@code MaskPolicies.byType()} uses. */
  static final char MASK = '*';

  /** The mask of the run that masks every character of every mention. */
  static final char MASK_ALL = '#';

  private final PiiGolden.Pack pack;

  private PiiRewriteGolden(PiiGolden.Pack pack) {
    this.pack = pack;
  }

  /** One run per pack, in the order of {@link PiiGolden#PACKS}. */
  static List<AnnotatorGolden> all() {
    List<AnnotatorGolden> all = new ArrayList<>();
    for (PiiGolden.Pack p : PiiGolden.PACKS) {
      all.add(new PiiRewriteGolden(p));
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

  @Override
  public String modelDir() {
    return pack.dir();
  }

  @Override
  public String annotatorClass() {
    return "opennlp.tools.pii.PiiPacks";
  }

  @Override
  public String method() {
    return pack.method();
  }

  @Override
  public Map<String, String> config() {
    Map<String, String> config = new LinkedHashMap<>();
    config.put("pack", pack.dir());
    config.put("key", KEY);
    config.put("mask", String.valueOf(MASK));
    config.put("mask-all", String.valueOf(MASK_ALL));
    return config;
  }

  @Override
  public Set<String> inputSets() {
    return Set.of(PiiGolden.INPUT_SET);
  }

  @Override
  public List<Layer> layers(Input input) {
    Document document = new PiiAnnotator(pack.factory().get()).annotate(Document.of(input.text()));
    HmacTokenizer tokenizer = new HmacTokenizer(KEY.getBytes(StandardCharsets.US_ASCII));
    List<Layer> layers = new ArrayList<>();
    layers.add(maskEdits(input, Masker.mask(document, PiiAnnotator.PII, MaskPolicies.byType()), MASK));
    String all = Masker.mask(document, PiiAnnotator.PII, MASK_ALL);
    layers.add(new Layer("turbo:pii-mask-all", "document",
        new Column.Str("text", new String[] {utf8Masked(input.text(), all, MASK_ALL)})));

    String[] operations = {"pseudonymize", "hmac"};
    String[] texts = new String[2];
    String[] errors = new String[2];
    Labels labels = new Labels();
    List<Span> replaced = new ArrayList<>();
    for (Annotation<PiiMention> a : document.get(PiiAnnotator.PII)) {
      replaced.add(a.span());
    }
    replaced.sort(Comparator.comparingInt(Span::getStart));
    for (int k = 0; k < 2; k++) {
      try {
        PiiRewrite r = k == 0 ? new Pseudonymizer().rewrite(document) : tokenizer.rewrite(document);
        texts[k] = r.text();
        errors[k] = "";
        labels.add(operations[k], input, r, replaced);
      } catch (IllegalArgumentException e) {
        texts[k] = "";
        errors[k] = e.getMessage() == null ? "" : e.getMessage();
      }
    }
    layers.add(new Layer("turbo:pii-rewrites", "document", new Column.Str("operation", operations),
        new Column.Str("text", texts), new Column.Str("error", errors)));
    layers.add(labels.layer());

    PiiAuditReport report = PiiAuditReport.of(document, tokenizer);
    List<String> types = new ArrayList<>(report.types());
    int n = types.size();
    long[] mentions = new long[n];
    long[] distinct = new long[n];
    String[] samples = new String[n];
    for (int k = 0; k < n; k++) {
      String type = types.get(k);
      mentions[k] = report.counts().get(type);
      distinct[k] = report.distinctCounts().get(type);
      samples[k] = String.join(", ", report.samples(type));
    }
    layers.add(new Layer("turbo:pii-audit", "document", new Column.Str("type", types.toArray(new String[0])),
        new Column.U32("mentions", mentions), new Column.U32("distinct", distinct),
        new Column.Str("samples", samples)));
    layers.add(new Layer("turbo:pii-audit-report", "document",
        new Column.Str("text", new String[] {report.toString()})));
    return layers;
  }

  /**
   * The edits that turn the input into {@code masked}: one per code point that differs, with UTF-8
   * offsets and one mask byte per UTF-8 byte of the code point.
   */
  static Layer maskEdits(Input input, String masked, char mask) {
    String text = input.text();
    List<Long> starts = new ArrayList<>();
    List<Long> ends = new ArrayList<>();
    List<String> with = new ArrayList<>();
    for (int i = 0; i < text.length();) {
      int cp = text.codePointAt(i);
      int units = Character.charCount(cp);
      if (masked(text, masked, i, units, mask)) {
        Utf8Offsets.ByteSpan span = input.offsets().span(i, i + units);
        starts.add((long) span.start());
        ends.add((long) span.end());
        with.add(String.valueOf(mask).repeat(Utf8Offsets.utf8Length(cp)));
      }
      i += units;
    }
    return new Layer("turbo:view", "text",
        new Column.U32("start", starts.stream().mapToLong(Long::longValue).toArray()),
        new Column.U32("end", ends.stream().mapToLong(Long::longValue).toArray()),
        new Column.Str("with", with.toArray(new String[0])));
  }

  /** {@code masked} with each masked code point written as one mask per UTF-8 byte. */
  static String utf8Masked(String text, String masked, char mask) {
    StringBuilder out = new StringBuilder(text.length());
    for (int i = 0; i < text.length();) {
      int cp = text.codePointAt(i);
      int units = Character.charCount(cp);
      if (masked(text, masked, i, units, mask)) {
        out.append(String.valueOf(mask).repeat(Utf8Offsets.utf8Length(cp)));
      } else {
        out.appendCodePoint(cp);
      }
      i += units;
    }
    return out.toString();
  }

  /**
   * Whether the code point of {@code units} units at {@code i} was masked: its units in
   * {@code masked} differ from the input's. They are then all the mask, which is checked; a code
   * point that is the mask itself is unchanged either way.
   */
  private static boolean masked(String text, String masked, int i, int units, char mask) {
    if (masked.length() != text.length()) {
      throw new IllegalStateException("the mask changed the UTF-16 length");
    }
    if (text.regionMatches(i, masked, i, units)) {
      return false;
    }
    for (int u = 0; u < units; u++) {
      if (masked.charAt(i + u) != mask) {
        throw new IllegalStateException("a code point was partly masked at " + i);
      }
    }
    return true;
  }

  /** The rows of {@code turbo:pii-labels}. */
  private static final class Labels {
    private final List<String> operations = new ArrayList<>();
    private final List<Long> sourceStarts = new ArrayList<>();
    private final List<Long> sourceEnds = new ArrayList<>();
    private final List<Long> starts = new ArrayList<>();
    private final List<Long> ends = new ArrayList<>();
    private final List<String> labels = new ArrayList<>();

    /**
     * The labels of {@code r}, a rewrite of the input, in text order. {@code replaced} holds the
     * mentions it replaced, ordered by start as the rewrite orders them; they do not overlap, or
     * the rewrite would have thrown, so label k stands for mention k.
     */
    void add(String operation, Input input, PiiRewrite r, List<Span> replaced) {
      Utf8Offsets rewritten = new Utf8Offsets(input.inputId() + " " + operation, r.text());
      List<PiiMention> labelled = r.mentions();
      if (labelled.size() != replaced.size()) {
        throw new IllegalStateException(labelled.size() + " labels for " + replaced.size() + " mentions");
      }
      for (int k = 0; k < labelled.size(); k++) {
        PiiMention label = labelled.get(k);
        Utf8Offsets.ByteSpan inNew = rewritten.span(label.span().getStart(), label.span().getEnd());
        Utf8Offsets.ByteSpan inSource = input.offsets().span(replaced.get(k).getStart(), replaced.get(k).getEnd());
        operations.add(operation);
        sourceStarts.add((long) inSource.start());
        sourceEnds.add((long) inSource.end());
        starts.add((long) inNew.start());
        ends.add((long) inNew.end());
        labels.add(label.normalized());
      }
    }

    Layer layer() {
      return new Layer("turbo:pii-labels", "document", new Column.Str("operation", operations.toArray(new String[0])),
          new Column.U32("source_start", longs(sourceStarts)), new Column.U32("source_end", longs(sourceEnds)),
          new Column.U32("start", longs(starts)), new Column.U32("end", longs(ends)),
          new Column.Str("label", labels.toArray(new String[0])));
    }

    private static long[] longs(List<Long> values) {
      return values.stream().mapToLong(Long::longValue).toArray();
    }
  }
}
