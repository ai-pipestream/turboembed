package ai.pipestream.turbo.goldens;

import java.io.ByteArrayInputStream;
import java.io.IOException;
import java.lang.reflect.Constructor;
import java.lang.reflect.InvocationTargetException;
import java.lang.reflect.Method;
import java.util.ArrayList;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Set;
import opennlp.tools.document.Annotation;
import opennlp.tools.document.Document;
import opennlp.tools.document.LayerKey;
import opennlp.tools.document.Layers;
import opennlp.tools.formats.conllu.ConlluTagset;
import opennlp.tools.util.InputStreamFactory;
import opennlp.tools.util.ObjectStream;
import opennlp.tools.util.Span;

/**
 * The golden of {@code ConlluCorefDocumentStream} over a CoNLL-U file with CorefUD {@code Entity}
 * attributes: every document it read and every exception it threw.
 *
 * <p>The stream is in the {@code coref-formats} module of the OpenNLP addons' {@code ready/coref}
 * branch, with {@code CorefAnnotator} and {@code CorefMention} of its {@code coref} module, at
 * {@link #ADDONS_COMMIT}. The container compiles these three classes against the image's OpenNLP
 * into a jar of their own, which {@link BranchJar} loads.</p>
 *
 * <p>A run is one tag set. It reads the {@code conllu} set's files whose id starts with {@code
 * corefud-}, until {@code read} returns null; an exception is recorded with the number of
 * documents before it, and reading goes on. The layers, all anchored to the document, give every
 * span as UTF-8 byte offsets into its document's text:</p>
 *
 * <ul>
 *   <li>{@code turbo:read-errors}: per exception, {@code after}, {@code class} and {@code
 *       message}.</li>
 *   <li>{@code turbo:coref-texts}: per document, its text.</li>
 *   <li>{@code turbo:coref-sentences}: per sentence, the document's index, the span and the
 *       annotation's value.</li>
 *   <li>{@code turbo:coref-tokens}: per token, the document's index, the span and the token; and
 *       aligned to it {@code turbo:coref-tags}, the tag's span and value.</li>
 *   <li>{@code turbo:coref-mentions}: per annotation of the gold chain layer, in its order, the
 *       document's index, the span, and the {@code CorefMention}'s chain, kind and entity.</li>
 *   <li>{@code turbo:coref-speakers}: per annotation of the speaker layer, when the document has
 *       one, the document's index, the span and the label.</li>
 * </ul>
 */
final class CorefUdGolden implements AnnotatorGolden {

  /** The input set the stream reads. */
  static final String SET = Runs.CONLLU;

  /** The prefix of the input ids the stream reads. */
  static final String PREFIX = "corefud-";

  /** The system property naming the jar of the addons' classes, which the image's entrypoint sets. */
  static final String JAR_PROPERTY = "turbo.goldens.corefJar";

  /** The commit of the addons' ready/coref branch the jar must be built from. */
  static final String ADDONS_COMMIT = "be11df610e24a9a2a10f80b7199349fe5b3594a9";

  static final String STREAM = "opennlp.tools.coref.formats.ConlluCorefDocumentStream";
  static final String ANNOTATOR = "opennlp.tools.coref.CorefAnnotator";
  static final String MENTION = "opennlp.tools.coref.CorefMention";

  /** The stopping point for a stream that throws without moving on, which would otherwise loop. */
  private static final int MAX_READS = 100_000;

  private final ConlluTagset tagset;
  private final Branch branch;

  private CorefUdGolden(ConlluTagset tagset, Branch branch) {
    this.tagset = tagset;
    this.branch = branch;
  }

  /** The runs: one per tag set. */
  static List<CorefUdGolden> all() throws IOException {
    Branch branch = Branch.open();
    return List.of(new CorefUdGolden(ConlluTagset.U, branch), new CorefUdGolden(ConlluTagset.X, branch));
  }

  /** The stream's constructor, the layer keys and the mention's accessors, from the addons' jar. */
  private record Branch(Constructor<?> stream, LayerKey<?> goldChains, LayerKey<?> speakers, Method chain,
      Method kind, Method entity) {

    static Branch open() throws IOException {
      ClassLoader loader = BranchJar.open(JAR_PROPERTY, ADDONS_COMMIT, List.of(STREAM, ANNOTATOR, MENTION));
      try {
        Class<?> stream = Class.forName(STREAM, true, loader);
        Class<?> annotator = Class.forName(ANNOTATOR, true, loader);
        Class<?> mention = Class.forName(MENTION, true, loader);
        return new Branch(stream.getConstructor(InputStreamFactory.class, ConlluTagset.class),
            (LayerKey<?>) annotator.getField("GOLD_CHAINS").get(null),
            (LayerKey<?>) annotator.getField("SPEAKERS").get(null), mention.getMethod("chain"),
            mention.getMethod("kind"), mention.getMethod("entity"));
      } catch (ReflectiveOperationException e) {
        throw new IllegalStateException(STREAM, e);
      }
    }

    ObjectStream<?> open(InputStreamFactory in, ConlluTagset tagset) throws IOException {
      try {
        return (ObjectStream<?>) stream.newInstance(in, tagset);
      } catch (InvocationTargetException e) {
        if (e.getCause() instanceof IOException io) {
          throw io;
        }
        throw new IllegalStateException(STREAM, e.getCause());
      } catch (ReflectiveOperationException e) {
        throw new IllegalStateException(STREAM, e);
      }
    }

    Object call(Method m, Object target) {
      try {
        return m.invoke(target);
      } catch (ReflectiveOperationException e) {
        throw new IllegalStateException(m.getName(), e);
      }
    }
  }

  @Override
  public String annotator() {
    return "conllu";
  }

  @Override
  public List<ModelRef> models() {
    return List.of();
  }

  @Override
  public String modelDir() {
    return "corefud-" + tagset.name().toLowerCase(Locale.ROOT);
  }

  @Override
  public String annotatorClass() {
    return STREAM;
  }

  @Override
  public String method() {
    return "read";
  }

  @Override
  public Map<String, String> config() {
    return Map.of("tagset", tagset.name().toLowerCase(Locale.ROOT), "addonsCommit", ADDONS_COMMIT);
  }

  @Override
  public Set<String> inputSets() {
    return Set.of(SET);
  }

  @Override
  public boolean accepts(Input input) {
    return input.set().equals(SET) && input.id().startsWith(PREFIX);
  }

  @Override
  public List<Layer> layers(Input input) {
    List<Document> documents = new ArrayList<>();
    Table errors = new Table("turbo:read-errors", "after:u32", "class:str", "message:str");
    InputStreamFactory in = () -> new ByteArrayInputStream(input.bytes());
    try (ObjectStream<?> stream = branch.open(in, tagset)) {
      for (int reads = 0; ; reads++) {
        if (reads > MAX_READS) {
          throw new IllegalStateException(input.inputId() + ": the stream does not end");
        }
        Object document;
        try {
          document = stream.read();
        } catch (IOException | RuntimeException e) {
          errors.row(documents.size(), e.getClass().getName(), FormatGolden.message(e));
          continue;
        }
        if (document == null) {
          break;
        }
        documents.add((Document) document);
      }
    } catch (IOException e) {
      throw new IllegalStateException(input.inputId() + ": the stream failed", e);
    }

    Table texts = new Table("turbo:coref-texts", "text:str");
    Table sentences = new Table("turbo:coref-sentences", "document:u32", "start:u32", "end:u32", "text:str");
    Table tokens = new Table("turbo:coref-tokens", "document:u32", "start:u32", "end:u32", "text:str");
    Table tags = Table.aligned("turbo:coref-tags", "turbo:coref-tokens", "start:u32", "end:u32", "tag:str");
    Table mentions = new Table("turbo:coref-mentions", "document:u32", "start:u32", "end:u32", "chain:u32",
        "kind:str", "entity:str");
    Table speakers = new Table("turbo:coref-speakers", "document:u32", "start:u32", "end:u32", "label:str");
    for (int d = 0; d < documents.size(); d++) {
      Document document = documents.get(d);
      String text = document.text().toString();
      texts.row(text);
      Utf8Offsets offsets = new Utf8Offsets(input.inputId() + " document " + d, text);
      for (Annotation<String> a : document.get(Layers.SENTENCES)) {
        Utf8Offsets.ByteSpan b = bytes(offsets, a.span());
        sentences.row(d, b.start(), b.end(), a.value());
      }
      for (Annotation<String> a : document.get(Layers.TOKENS)) {
        Utf8Offsets.ByteSpan b = bytes(offsets, a.span());
        tokens.row(d, b.start(), b.end(), a.value());
      }
      for (Annotation<String> a : document.get(Layers.POS_TAGS)) {
        Utf8Offsets.ByteSpan b = bytes(offsets, a.span());
        tags.row(b.start(), b.end(), a.value());
      }
      for (Annotation<?> a : document.get(branch.goldChains())) {
        Utf8Offsets.ByteSpan b = bytes(offsets, a.span());
        Object m = a.value();
        mentions.row(d, b.start(), b.end(), (Integer) branch.call(branch.chain(), m),
            (String) branch.call(branch.kind(), m), Integer.toString((Integer) branch.call(branch.entity(), m)));
      }
      if (document.layers().contains(branch.speakers())) {
        for (Annotation<?> a : document.get(branch.speakers())) {
          Utf8Offsets.ByteSpan b = bytes(offsets, a.span());
          speakers.row(d, b.start(), b.end(), (String) a.value());
        }
      }
    }
    return List.of(errors.layer(), texts.layer(), sentences.layer(), tokens.layer(), tags.layer(),
        mentions.layer(), speakers.layer());
  }

  private static Utf8Offsets.ByteSpan bytes(Utf8Offsets offsets, Span span) {
    return offsets.span(span.getStart(), span.getEnd());
  }
}
