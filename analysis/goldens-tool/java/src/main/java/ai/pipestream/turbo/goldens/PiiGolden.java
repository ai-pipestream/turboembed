package ai.pipestream.turbo.goldens;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.function.Supplier;
import opennlp.tools.document.Annotation;
import opennlp.tools.document.Document;
import opennlp.tools.pii.PiiAnnotator;
import opennlp.tools.pii.PiiExtractor;
import opennlp.tools.pii.PiiMention;
import opennlp.tools.pii.PiiPacks;

/**
 * The goldens of {@code PiiAnnotator} over each pack of {@code PiiPacks}, one run per pack.
 *
 * <p>The {@code turbo:pii} layer has one row per mention, in the annotator's order, with the
 * columns {@code start} and {@code end} (UTF-8 byte offsets of the mention), {@code type} (the
 * {@code PiiMention} type name, such as {@code email} or {@code aws-access-key}) and
 * {@code normalized} (the mention's normalized form). The path of a golden is
 * {@code pii/<pack>/pii/<id>.golden}, with the pack's name in kebab case ({@code payment},
 * {@code us-identity}, {@code all-structured}); the pack takes the place of a model, which the
 * extractors do not have.</p>
 */
final class PiiGolden implements AnnotatorGolden {

  /** The input set the PII goldens are made from. */
  static final String INPUT_SET = "pii";

  /** The name of the golden's annotator directory. */
  static final String ANNOTATOR = "pii";

  /** The layer the goldens hold: the id of the Rust layer type. */
  static final String LAYER = "turbo:pii";

  /** A pack: its directory name, its factory method's name and the factory. */
  record Pack(String dir, String method, Supplier<PiiExtractor> factory) {
  }

  /** The packs in the order of {@code PiiPacks}. */
  static final List<Pack> PACKS = List.of(
      new Pack("payment", "payment", PiiPacks::payment),
      new Pack("contact", "contact", PiiPacks::contact),
      new Pack("network", "network", PiiPacks::network),
      new Pack("secrets", "secrets", PiiPacks::secrets),
      new Pack("crypto", "crypto", PiiPacks::crypto),
      new Pack("us-identity", "usIdentity", PiiPacks::usIdentity),
      new Pack("eu-identity", "euIdentity", PiiPacks::euIdentity),
      new Pack("ca-identity", "caIdentity", PiiPacks::caIdentity),
      new Pack("device", "device", PiiPacks::device),
      new Pack("all-structured", "allStructured", PiiPacks::allStructured));

  private final Pack pack;

  private PiiGolden(Pack pack) {
    this.pack = pack;
  }

  /** One run per pack, in the order of {@link #PACKS}. */
  static List<AnnotatorGolden> all() {
    List<AnnotatorGolden> all = new ArrayList<>();
    for (Pack p : PACKS) {
      all.add(new PiiGolden(p));
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
    return Map.of("pack", pack.dir());
  }

  @Override
  public Set<String> inputSets() {
    return Set.of(INPUT_SET);
  }

  @Override
  public List<Layer> layers(Input input) {
    Document document = new PiiAnnotator(pack.factory().get()).annotate(Document.of(input.text()));
    List<Annotation<PiiMention>> mentions = document.get(PiiAnnotator.PII);
    int n = mentions.size();
    long[] starts = new long[n];
    long[] ends = new long[n];
    String[] types = new String[n];
    String[] normalized = new String[n];
    for (int k = 0; k < n; k++) {
      PiiMention m = mentions.get(k).value();
      Utf8Offsets.ByteSpan span = input.offsets().span(m.span().getStart(), m.span().getEnd());
      starts[k] = span.start();
      ends[k] = span.end();
      types[k] = m.type();
      normalized[k] = m.normalized();
    }
    return List.of(new Layer(LAYER, "text", new Column.U32("start", starts), new Column.U32("end", ends),
        new Column.Str("type", types), new Column.Str("normalized", normalized)));
  }
}
