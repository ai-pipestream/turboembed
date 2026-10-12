package ai.pipestream.turbo.goldens;

import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.function.Supplier;
import opennlp.tools.util.normalizer.AccentFoldCharSequenceNormalizer;
import opennlp.tools.util.normalizer.AlignedText;
import opennlp.tools.util.normalizer.BoundedUrlCharSequenceNormalizer;
import opennlp.tools.util.normalizer.BulletCharSequenceNormalizer;
import opennlp.tools.util.normalizer.CaseFoldCharSequenceNormalizer;
import opennlp.tools.util.normalizer.CharSequenceNormalizer;
import opennlp.tools.util.normalizer.ConfusableSkeletonCharSequenceNormalizer;
import opennlp.tools.util.normalizer.DashCharSequenceNormalizer;
import opennlp.tools.util.normalizer.DehyphenationCharSequenceNormalizer;
import opennlp.tools.util.normalizer.DigitCharSequenceNormalizer;
import opennlp.tools.util.normalizer.EllipsisCharSequenceNormalizer;
import opennlp.tools.util.normalizer.EmojiCharSequenceNormalizer;
import opennlp.tools.util.normalizer.EmojiToEmoticonCharSequenceNormalizer;
import opennlp.tools.util.normalizer.EmoticonToEmojiCharSequenceNormalizer;
import opennlp.tools.util.normalizer.EnglishContractionCharSequenceNormalizer;
import opennlp.tools.util.normalizer.FullCaseFoldCharSequenceNormalizer;
import opennlp.tools.util.normalizer.GermanUmlautCharSequenceNormalizer;
import opennlp.tools.util.normalizer.InvisibleCharSequenceNormalizer;
import opennlp.tools.util.normalizer.LineBreakPreservingWhitespaceCharSequenceNormalizer;
import opennlp.tools.util.normalizer.NfcCharSequenceNormalizer;
import opennlp.tools.util.normalizer.NfkcCharSequenceNormalizer;
import opennlp.tools.util.normalizer.NumberCharSequenceNormalizer;
import opennlp.tools.util.normalizer.OffsetAwareNormalizer;
import opennlp.tools.util.normalizer.ParagraphPreservingWhitespaceCharSequenceNormalizer;
import opennlp.tools.util.normalizer.QuoteCharSequenceNormalizer;
import opennlp.tools.util.normalizer.ShrinkCharSequenceNormalizer;
import opennlp.tools.util.normalizer.SocialMediaCharSequenceNormalizer;
import opennlp.tools.util.normalizer.SymbolJoinerCharSequenceNormalizer;
import opennlp.tools.util.normalizer.TextNormalizer;
import opennlp.tools.util.normalizer.UrlCharSequenceNormalizer;
import opennlp.tools.util.normalizer.WhitespaceCharSequenceNormalizer;

/**
 * The golden of one OpenNLP normalizer: a {@code turbo:view} layer with one row per edit the
 * normalizer made to the input.
 *
 * <p>A normalizer that implements {@link OffsetAwareNormalizer} is run through
 * {@code normalizeAligned} and the edits are read from the alignment it returns (see
 * {@link Edit#fromAlignment}); its plain {@code normalize} must give the same text, which is
 * checked. A normalizer without an aligned method is run through {@code normalize} and its edits
 * are reconstructed by a diff of source and result on code points (see {@link Edit#fromDiff}), so
 * for those the alignment is reconstructed, not reported by OpenNLP. In both cases the edits
 * applied to the source must give the normalized text, which is checked before anything is
 * written.</p>
 *
 * <p>The path of a golden is {@code normalize/<name>/<set>/<id>.golden}: the annotator is
 * {@code normalize} and the name of the normalizer takes the place of a model, which these
 * annotators do not have. A normalizer run with a non-default option also records it in a
 * {@code config} header record.</p>
 */
final class NormalizerGolden implements AnnotatorGolden {

  /** The input set the normalizer goldens are made from. */
  static final String INPUT_SET = "norm";

  private final String name;
  private final Class<?> type;
  private final Map<String, String> config;
  private final Supplier<CharSequenceNormalizer> factory;
  private final String method;

  private NormalizerGolden(String name, Class<?> type, Map<String, String> config,
      Supplier<CharSequenceNormalizer> factory) {
    this.name = name;
    this.type = type;
    this.config = config;
    this.factory = factory;
    this.method = OffsetAwareNormalizer.class.isAssignableFrom(type) ? "normalizeAligned" : "normalize";
  }

  private static NormalizerGolden of(String name, Class<?> type, Supplier<CharSequenceNormalizer> factory) {
    return new NormalizerGolden(name, type, Map.of(), factory);
  }

  /** Every normalizer the Rust crate has or will have a counterpart of, in the order of their names. */
  static List<AnnotatorGolden> all() {
    return List.<AnnotatorGolden>of(
        of("accent-fold", AccentFoldCharSequenceNormalizer.class, AccentFoldCharSequenceNormalizer::getInstance),
        of("bounded-url", BoundedUrlCharSequenceNormalizer.class, BoundedUrlCharSequenceNormalizer::getInstance),
        of("bullet", BulletCharSequenceNormalizer.class, BulletCharSequenceNormalizer::getInstance),
        of("case-fold", CaseFoldCharSequenceNormalizer.class, CaseFoldCharSequenceNormalizer::getInstance),
        new NormalizerGolden("case-fold-tr", CaseFoldCharSequenceNormalizer.class, Map.of("locale", "tr"),
            () -> CaseFoldCharSequenceNormalizer.getInstance(Locale.forLanguageTag("tr"))),
        of("confusable-skeleton", ConfusableSkeletonCharSequenceNormalizer.class,
            ConfusableSkeletonCharSequenceNormalizer::getInstance),
        of("dash", DashCharSequenceNormalizer.class, DashCharSequenceNormalizer::getInstance),
        new NormalizerGolden("default-chain", TextNormalizer.class, Map.of(), TextNormalizer::defaultChain),
        of("dehyphenation", DehyphenationCharSequenceNormalizer.class,
            DehyphenationCharSequenceNormalizer::getInstance),
        of("digit", DigitCharSequenceNormalizer.class, DigitCharSequenceNormalizer::getInstance),
        of("ellipsis", EllipsisCharSequenceNormalizer.class, EllipsisCharSequenceNormalizer::getInstance),
        of("emoji", EmojiCharSequenceNormalizer.class, EmojiCharSequenceNormalizer::getInstance),
        of("emoji-to-emoticon", EmojiToEmoticonCharSequenceNormalizer.class,
            EmojiToEmoticonCharSequenceNormalizer::getInstance),
        of("emoticon-to-emoji", EmoticonToEmojiCharSequenceNormalizer.class,
            EmoticonToEmojiCharSequenceNormalizer::getInstance),
        of("english-contraction", EnglishContractionCharSequenceNormalizer.class,
            EnglishContractionCharSequenceNormalizer::getInstance),
        of("full-case-fold", FullCaseFoldCharSequenceNormalizer.class, FullCaseFoldCharSequenceNormalizer::getInstance),
        of("german-umlaut", GermanUmlautCharSequenceNormalizer.class, GermanUmlautCharSequenceNormalizer::getInstance),
        of("invisible", InvisibleCharSequenceNormalizer.class, InvisibleCharSequenceNormalizer::getInstance),
        of("nfc", NfcCharSequenceNormalizer.class, NfcCharSequenceNormalizer::getInstance),
        of("nfkc", NfkcCharSequenceNormalizer.class, NfkcCharSequenceNormalizer::getInstance),
        of("number", NumberCharSequenceNormalizer.class, NumberCharSequenceNormalizer::getInstance),
        of("quote", QuoteCharSequenceNormalizer.class, QuoteCharSequenceNormalizer::getInstance),
        of("shrink", ShrinkCharSequenceNormalizer.class, ShrinkCharSequenceNormalizer::getInstance),
        of("social-media", SocialMediaCharSequenceNormalizer.class, SocialMediaCharSequenceNormalizer::getInstance),
        of("symbol-joiner", SymbolJoinerCharSequenceNormalizer.class, SymbolJoinerCharSequenceNormalizer::getInstance),
        of("url", UrlCharSequenceNormalizer.class, UrlCharSequenceNormalizer::getInstance),
        of("whitespace", WhitespaceCharSequenceNormalizer.class, WhitespaceCharSequenceNormalizer::getInstance),
        of("whitespace-keep-lines", LineBreakPreservingWhitespaceCharSequenceNormalizer.class,
            LineBreakPreservingWhitespaceCharSequenceNormalizer::getInstance),
        of("whitespace-keep-paragraphs", ParagraphPreservingWhitespaceCharSequenceNormalizer.class,
            ParagraphPreservingWhitespaceCharSequenceNormalizer::getInstance));
  }

  @Override
  public String annotator() {
    return "normalize";
  }

  @Override
  public List<ModelRef> models() {
    return List.of();
  }

  @Override
  public String modelDir() {
    return name;
  }

  @Override
  public String annotatorClass() {
    return type.getName();
  }

  @Override
  public String method() {
    return type == TextNormalizer.class ? "defaultChain().normalize" : method;
  }

  @Override
  public Map<String, String> config() {
    return config;
  }

  @Override
  public boolean accepts(Input input) {
    return input.set().equals(INPUT_SET);
  }

  @Override
  public List<Layer> layers(Input input) {
    String text = input.text();
    CharSequenceNormalizer normalizer = factory.get();
    String result = normalizer.normalize(text).toString();
    List<Edit> edits;
    if (normalizer instanceof OffsetAwareNormalizer aware) {
      AlignedText aligned = aware.normalizeAligned(text);
      if (!aligned.normalizedString().equals(result)) {
        throw new IllegalStateException(label(input) + ": normalizeAligned and normalize disagree");
      }
      edits = Edit.fromAlignment(text, aligned);
    } else {
      edits = Edit.fromDiff(text, result);
    }
    if (!Edit.apply(text, edits).equals(result)) {
      throw new IllegalStateException(label(input) + ": the edits do not reproduce the normalized text");
    }
    int n = edits.size();
    long[] starts = new long[n];
    long[] ends = new long[n];
    String[] with = new String[n];
    for (int i = 0; i < n; i++) {
      Edit e = edits.get(i);
      Utf8Offsets.ByteSpan b = input.offsets().span(e.start(), e.end());
      starts[i] = b.start();
      ends[i] = b.end();
      with[i] = e.with();
    }
    return List.of(new Layer("turbo:view", "text", new Column.U32("start", starts), new Column.U32("end", ends),
        new Column.Str("with", with)));
  }

  private String label(Input input) {
    return name + " on " + input.inputId();
  }
}
