package ai.pipestream.turbo.goldens;

import java.io.BufferedInputStream;
import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.lang.reflect.Field;
import java.lang.reflect.InvocationTargetException;
import java.lang.reflect.Method;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.stream.Stream;
import opennlp.tools.formats.masc.MascDocument;
import opennlp.tools.formats.masc.MascDocumentStream;
import opennlp.tools.formats.masc.MascNamedEntitySampleStream;
import opennlp.tools.formats.masc.MascPOSSampleStream;
import opennlp.tools.formats.masc.MascSentence;
import opennlp.tools.formats.masc.MascSentenceSampleStream;
import opennlp.tools.formats.masc.MascToken;
import opennlp.tools.formats.masc.MascTokenSampleStream;
import opennlp.tools.formats.masc.MascWord;
import opennlp.tools.namefind.NameSample;
import opennlp.tools.postag.POSSample;
import opennlp.tools.sentdetect.SentenceSample;
import opennlp.tools.tokenize.TokenSample;
import opennlp.tools.util.ObjectStream;
import opennlp.tools.util.Span;

/**
 * The golden of OpenNLP's MASC readers over a small corpus: the inputs of the {@code formats}
 * set whose id starts with {@code masc-} are bundles of the files of a corpus directory, which
 * this class writes to a temporary directory before reading it.
 *
 * <p>A bundle is UTF-8 text. A line {@code @@ file <path>} starts a file whose content is the
 * text after that line up to the next line {@code \n@@ end} (the line end before {@code @@ end}
 * is not part of the file, so a file that ends with a line end has an empty line before it). A
 * line {@code @@ hex <path>} does the same with the bytes written as hexadecimal digits, for
 * content that is not valid UTF-8. A line {@code @@ dir <path>} makes an empty directory. Other
 * lines outside a file must be empty or start with {@code #}. Each directory of a bundle holds at
 * most one {@code .hdr} file and at most one subdirectory, so the order in which
 * {@code MascDocumentStream} finds the documents does not depend on the order the file system
 * lists a directory in.</p>
 *
 * <p>The {@code documents} run parses each document as {@code MascDocumentStream}'s constructor
 * does ({@code checkAnnotations}, called by reflection, then {@code MascDocument.parseDocument}),
 * and records what each document holds, read through reflection where the class keeps it
 * private. Its layers, all anchored to the document:</p>
 *
 * <ul>
 *   <li>{@code turbo:masc-documents}: per {@code .hdr} file in the order the stream finds them,
 *       its path in the corpus, the class and message of what parsing it threw (the corpus
 *       directory written {@code <corpus>}, and the stack trace that {@code addPennTags} puts in
 *       its message left out), {@code abort} 1 when the throw ends the stream's
 *       constructor rather than skipping the document, and whether it has Penn tags and named
 *       entities.</li>
 *   <li>{@code turbo:masc-sentences}: per sentence of each document, its region in UTF-16 units,
 *       the ids of its quarks and whether it was tokenized.</li>
 *   <li>{@code turbo:masc-tokens}: per Penn token of each sentence, its UTF-16 span, id, tag,
 *       base, which of tag and base are null (bit 0 and bit 1 of {@code nulls}) and the ids of its
 *       quarks.</li>
 *   <li>{@code turbo:masc-names}: per named entity of each sentence, its token span and type
 *       ({@code nulls} 1 for a null type).</li>
 *   <li>{@code turbo:masc-stream} and {@code turbo:masc-stream-errors}: the paths of the documents
 *       {@code MascDocumentStream} loaded, in order, or what its constructor threw.</li>
 * </ul>
 *
 * <p>The sample runs read the corpus with {@code MascDocumentStream} and one of the sample
 * streams, until {@code read} returns null or throws {@code NullPointerException} (the document
 * stream is used up and every later read throws the same), then call {@code reset} and read the
 * same way again ({@code pass} 1). The layers, all anchored to the document: {@code turbo:samples}
 * ({@code toString} of each sample), {@code turbo:read-errors} (per exception, the pass, the
 * number of samples read before it, class and message), {@code turbo:open-errors} (what the
 * constructors threw), and the samples' contents as {@link FormatGolden} writes them.</p>
 */
final class MascGolden implements AnnotatorGolden {

  /** The input set the bundles are in. */
  static final String SET = Runs.FORMATS;

  /** The prefix of the bundles' ids. */
  static final String PREFIX = "masc-";

  /** The sentences per sample of the sentence stream, small so the inputs show a full sample. */
  static final int SENTENCES_PER_SAMPLE = 2;

  private static final int MAX_READS = 100_000;

  private static final Set<String> U32 = Set.of("doc", "tokenized", "sentence", "id", "nulls", "abort", "penn",
      "ne", "pass");

  enum Kind {
    DOCUMENTS("documents", MascDocument.class, "parseDocument"),
    POS("pos", MascPOSSampleStream.class, "read"),
    TOKEN("token", MascTokenSampleStream.class, "read"),
    SENTENCE("sentence", MascSentenceSampleStream.class, "read"),
    NAME("name", MascNamedEntitySampleStream.class, "read");

    final String dir;
    final Class<?> streamClass;
    final String method;

    Kind(String dir, Class<?> streamClass, String method) {
      this.dir = dir;
      this.streamClass = streamClass;
      this.method = method;
    }
  }

  private final Kind kind;

  private MascGolden(Kind kind) {
    this.kind = kind;
  }

  static List<MascGolden> all() {
    List<MascGolden> runs = new ArrayList<>();
    for (Kind k : Kind.values()) {
      runs.add(new MascGolden(k));
    }
    return runs;
  }

  @Override
  public String annotator() {
    return "masc";
  }

  @Override
  public List<ModelRef> models() {
    return List.of();
  }

  @Override
  public String annotatorClass() {
    return kind.streamClass.getName();
  }

  @Override
  public String method() {
    return kind.method;
  }

  @Override
  public Map<String, String> config() {
    return kind == Kind.SENTENCE ? Map.of("sentencesPerSample", Integer.toString(SENTENCES_PER_SAMPLE)) : Map.of();
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
  public String modelDir() {
    return kind.dir;
  }

  @Override
  public List<Layer> layers(Input input) throws AnnotatorException {
    Path root = null;
    try {
      root = Files.createTempDirectory("masc-golden");
      writeBundle(input, root);
      checkLayout(input, root.toFile());
      return kind == Kind.DOCUMENTS ? documents(root) : samples(input, root);
    } catch (IOException e) {
      throw new IllegalStateException(input.inputId() + ": the corpus could not be written", e);
    } finally {
      if (root != null) {
        delete(root);
      }
    }
  }

  // ---- the bundle ----

  private static void writeBundle(Input input, Path root) throws IOException {
    String text = input.text();
    int pos = 0;
    while (pos < text.length()) {
      int eol = text.indexOf('\n', pos);
      String line = eol < 0 ? text.substring(pos) : text.substring(pos, eol);
      int next = eol < 0 ? text.length() : eol + 1;
      if (line.startsWith("@@ file ") || line.startsWith("@@ hex ")) {
        boolean hex = line.startsWith("@@ hex ");
        Path file = resolve(input, root, line.substring(hex ? 7 : 8));
        int end;
        int after;
        if (text.startsWith("@@ end", next)) {
          end = next;
          after = next + 6;
        } else {
          end = text.indexOf("\n@@ end", next);
          if (end < 0) {
            throw new IllegalArgumentException(input.inputId() + ": " + line + " has no @@ end");
          }
          after = end + 7;
        }
        if (after < text.length() && text.charAt(after) != '\n') {
          throw new IllegalArgumentException(input.inputId() + ": text after @@ end");
        }
        String content = text.substring(next, end);
        Files.createDirectories(file.getParent());
        Files.write(file, hex ? hexBytes(input, content) : content.getBytes(StandardCharsets.UTF_8));
        pos = Math.min(after + 1, text.length());
      } else if (line.startsWith("@@ dir ")) {
        Files.createDirectories(resolve(input, root, line.substring(7)));
        pos = next;
      } else if (line.isEmpty() || line.startsWith("#")) {
        pos = next;
      } else {
        throw new IllegalArgumentException(input.inputId() + ": a line outside a file: " + line);
      }
    }
  }

  private static Path resolve(Input input, Path root, String relative) {
    Path p = root.resolve(relative).normalize();
    if (relative.isEmpty() || relative.startsWith("/") || !p.startsWith(root) || p.equals(root)) {
      throw new IllegalArgumentException(input.inputId() + ": a path outside the corpus: " + relative);
    }
    return p;
  }

  private static byte[] hexBytes(Input input, String content) {
    String digits = content.replaceAll("\\s", "");
    if (digits.length() % 2 != 0) {
      throw new IllegalArgumentException(input.inputId() + ": an odd number of hex digits");
    }
    byte[] out = new byte[digits.length() / 2];
    for (int i = 0; i < out.length; i++) {
      out[i] = (byte) Integer.parseInt(digits.substring(2 * i, 2 * i + 2), 16);
    }
    return out;
  }

  /** Refuses a directory with two {@code .hdr} files or two subdirectories. */
  private static void checkLayout(Input input, File dir) {
    File[] files = dir.listFiles();
    int headers = 0;
    int dirs = 0;
    for (File f : files) {
      if (f.isDirectory()) {
        dirs++;
        checkLayout(input, f);
      } else if (f.getName().endsWith(".hdr")) {
        headers++;
      }
    }
    if (headers > 1 || dirs > 1) {
      throw new IllegalArgumentException(input.inputId() + ": " + dir + " has more than one .hdr file or directory");
    }
  }

  private static void delete(Path root) {
    try (Stream<Path> walk = Files.walk(root)) {
      for (Path p : walk.sorted(Comparator.reverseOrder()).toList()) {
        Files.delete(p);
      }
    } catch (IOException e) {
      throw new IllegalStateException("could not delete " + root, e);
    }
  }

  /** The prefix of the message {@code addPennTags} throws, which ends with the cause's stack trace. */
  private static final String PENN_FAILURE = "Could not attach POS tags to words. ";

  /**
   * A message with the corpus directory written {@code <corpus>}, and without the stack trace
   * {@code addPennTags} appends to its message (from the {@code [} after its prefix on), which
   * names the frames of this generator.
   */
  private static String relative(Path root, String message) {
    if (message == null) {
      return "";
    }
    if (message.startsWith(PENN_FAILURE)) {
      int trace = message.indexOf('[', PENN_FAILURE.length());
      if (trace >= 0) {
        message = message.substring(0, trace);
      }
    }
    return message.replace(root.toAbsolutePath().toString(), "<corpus>");
  }

  private static String relativePath(Path root, String path) {
    return root.toAbsolutePath().relativize(Path.of(path)).toString();
  }

  // ---- the documents run ----

  /** The {@code .hdr} files in the order the stream finds them: in each directory, then its subdirectory. */
  private static List<File> headers(File root) {
    List<File> out = new ArrayList<>();
    File dir = root;
    while (dir != null) {
      File sub = null;
      for (File f : dir.listFiles()) {
        if (f.isFile() && f.getAbsolutePath().endsWith(".hdr")) {
          out.add(f);
        } else if (f.isDirectory()) {
          sub = f;
        }
      }
      dir = sub;
    }
    return out;
  }

  private static Object field(Object o, String name) throws ReflectiveOperationException {
    Field f = o.getClass().getDeclaredField(name);
    f.setAccessible(true);
    return f.get(o);
  }

  private static String ids(List<MascWord> words) {
    StringBuilder b = new StringBuilder();
    for (MascWord w : words) {
      if (!b.isEmpty()) {
        b.append(' ');
      }
      b.append(w.getId());
    }
    return b.toString();
  }

  @SuppressWarnings("unchecked")
  private List<Layer> documents(Path root) throws IOException {
    FormatGolden.Columns docs = new FormatGolden.Columns(U32);
    FormatGolden.Columns sentences = new FormatGolden.Columns(U32);
    FormatGolden.Columns tokens = new FormatGolden.Columns(U32);
    FormatGolden.Columns names = new FormatGolden.Columns(U32);
    Path empty = Files.createTempDirectory("masc-empty");
    try {
      MascDocumentStream helper = new MascDocumentStream(empty.toFile());
      Method check = MascDocumentStream.class.getDeclaredMethod("checkAnnotations", String.class);
      check.setAccessible(true);
      int docIndex = 0;
      int sentenceIndex = 0;
      for (File hdr : headers(root.toFile())) {
        String path = hdr.getAbsolutePath();
        docs.str("path", relativePath(root, path));
        MascDocument doc = null;
        try {
          HashMap<String, File> group;
          try {
            group = (HashMap<String, File>) check.invoke(helper, path);
          } catch (InvocationTargetException e) {
            throw e.getCause() instanceof Exception x ? x : new IllegalStateException(e.getCause());
          }
          InputStream text = new BufferedInputStream(new FileInputStream(group.get("f.text")));
          InputStream seg = open(group, "f.seg");
          InputStream penn = open(group, "f.penn");
          InputStream s = open(group, "f.s");
          InputStream ne = open(group, "f.ne");
          try {
            doc = MascDocument.parseDocument(path, text, seg, penn, s, ne);
            docs.str("class", "").str("message", "").u32("abort", 0);
          } catch (IOException e) {
            docs.str("class", e.getClass().getName()).str("message", relative(root, e.getMessage())).u32("abort", 0);
          }
        } catch (Exception e) {
          docs.str("class", e.getClass().getName()).str("message", relative(root, e.getMessage())).u32("abort", 1);
        }
        docs.u32("penn", doc != null && doc.hasPennTags() ? 1 : 0).u32("ne", doc != null && doc.hasNamedEntities() ? 1 : 0);
        if (doc != null) {
          for (MascSentence sentence = doc.read(); sentence != null; sentence = doc.read()) {
            List<MascWord> words = (List<MascWord>) field(sentence, "words");
            List<MascToken> toks = (List<MascToken>) field(sentence, "sentenceTokens");
            sentences.u32("doc", docIndex).u32("start", sentence.getStart()).u32("end", sentence.getEnd())
                .str("quarks", ids(words)).u32("tokenized", toks != null ? 1 : 0);
            if (toks != null) {
              for (MascToken t : toks) {
                int nulls = (t.getPos() == null ? 1 : 0) | (t.getBase() == null ? 2 : 0);
                tokens.u32("sentence", sentenceIndex).u32("start", t.getStart()).u32("end", t.getEnd())
                    .u32("id", t.getTokenId()).str("pos", t.getPos()).str("base", t.getBase()).u32("nulls", nulls)
                    .str("quarks", ids(List.of(t.getQuarks())));
              }
            }
            for (Span n : sentence.getNamedEntities()) {
              names.u32("sentence", sentenceIndex).u32("start", n.getStart()).u32("end", n.getEnd())
                  .str("type", n.getType()).u32("nulls", n.getType() == null ? 1 : 0);
            }
            sentenceIndex++;
          }
          doc.reset();
        }
        docIndex++;
      }
    } catch (ReflectiveOperationException e) {
      throw new IllegalStateException("the MASC classes are not as this golden expects", e);
    } finally {
      delete(empty);
    }

    FormatGolden.Columns stream = new FormatGolden.Columns(U32);
    FormatGolden.Columns streamErrors = new FormatGolden.Columns(U32);
    try {
      MascDocumentStream s = new MascDocumentStream(root.toFile());
      for (MascDocument d = s.read(); d != null; d = s.read()) {
        stream.str("path", relativePath(root, (String) field(d, "pathToFile")));
      }
    } catch (IOException | RuntimeException e) {
      streamErrors.str("class", e.getClass().getName()).str("message", relative(root, e.getMessage()));
    } catch (ReflectiveOperationException e) {
      throw new IllegalStateException("the MASC classes are not as this golden expects", e);
    }

    List<Layer> layers = new ArrayList<>();
    layers.add(docs.layer("turbo:masc-documents", "document", "path", "class", "message", "abort", "penn", "ne"));
    layers.add(sentences.layer("turbo:masc-sentences", "document", "doc", "start", "end", "quarks", "tokenized"));
    layers.add(tokens.layer("turbo:masc-tokens", "document", "sentence", "start", "end", "id", "pos", "base",
        "nulls", "quarks"));
    layers.add(names.layer("turbo:masc-names", "document", "sentence", "start", "end", "type", "nulls"));
    layers.add(stream.layer("turbo:masc-stream", "document", "path"));
    layers.add(streamErrors.layer("turbo:masc-stream-errors", "document", "class", "message"));
    return layers;
  }

  private static InputStream open(Map<String, File> group, String key) throws IOException {
    return group.containsKey(key) ? new BufferedInputStream(new FileInputStream(group.get(key))) : null;
  }

  // ---- the sample runs ----

  private ObjectStream<?> open(Path root) throws IOException {
    MascDocumentStream docs = new MascDocumentStream(root.toFile());
    return switch (kind) {
      case POS -> new MascPOSSampleStream(docs);
      case TOKEN -> new MascTokenSampleStream(docs);
      case SENTENCE -> new MascSentenceSampleStream(docs, SENTENCES_PER_SAMPLE);
      case NAME -> new MascNamedEntitySampleStream(docs);
      case DOCUMENTS -> throw new IllegalStateException("not a sample run");
    };
  }

  private List<Layer> samples(Input input, Path root) {
    FormatGolden.Columns openErrors = new FormatGolden.Columns(U32);
    FormatGolden.Columns written = new FormatGolden.Columns(U32);
    FormatGolden.Columns errors = new FormatGolden.Columns(U32);
    List<Object> samples = new ArrayList<>();
    ObjectStream<?> stream = null;
    try {
      stream = open(root);
    } catch (IOException | RuntimeException e) {
      openErrors.str("class", e.getClass().getName()).str("message", relative(root, e.getMessage()));
    }
    if (stream != null) {
      for (int pass = 0; pass < 2; pass++) {
        if (pass == 1) {
          try {
            stream.reset();
          } catch (IOException e) {
            throw new IllegalStateException(input.inputId() + ": reset failed", e);
          }
        }
        for (int reads = 0; ; reads++) {
          if (reads > MAX_READS) {
            throw new IllegalStateException(input.inputId() + ": the stream does not end");
          }
          Object sample;
          try {
            sample = stream.read();
          } catch (IOException | RuntimeException e) {
            errors.u32("pass", pass).u32("after", samples.size()).str("class", e.getClass().getName())
                .str("message", relative(root, e.getMessage()));
            if (e instanceof NullPointerException) {
              break;
            }
            continue;
          }
          if (sample == null) {
            break;
          }
          samples.add(sample);
          written.u32("pass", pass).str("written", sample.toString());
        }
      }
    }

    List<Layer> layers = new ArrayList<>();
    layers.add(written.layer("turbo:samples", "document", "pass", "written"));
    layers.add(errors.layer("turbo:read-errors", "document", "pass", "after", "class", "message"));
    layers.add(openErrors.layer("turbo:open-errors", "document", "class", "message"));
    switch (kind) {
      case POS -> {
        FormatGolden.Columns toks = new FormatGolden.Columns(U32);
        FormatGolden.Columns tags = new FormatGolden.Columns(U32);
        for (int s = 0; s < samples.size(); s++) {
          POSSample p = (POSSample) samples.get(s);
          for (int i = 0; i < p.getSentence().length; i++) {
            toks.u32("sample", s).str("text", p.getSentence()[i]);
            tags.str("tag", p.getTags()[i]);
          }
        }
        layers.add(toks.layer("turbo:sample-tokens", "document", "sample", "text"));
        layers.add(tags.layer("turbo:sample-tags", "aligned:turbo:sample-tokens", "tag"));
      }
      case NAME -> {
        FormatGolden.Columns toks = new FormatGolden.Columns(U32);
        FormatGolden.Columns spans = new FormatGolden.Columns(U32);
        for (int s = 0; s < samples.size(); s++) {
          NameSample n = (NameSample) samples.get(s);
          for (String t : n.getSentence()) {
            toks.u32("sample", s).str("text", t);
          }
          for (Span span : n.getNames()) {
            spans.u32("sample", s).u32("start", span.getStart()).u32("end", span.getEnd()).str("type", span.getType())
                .u32("nulls", span.getType() == null ? 1 : 0);
          }
        }
        layers.add(toks.layer("turbo:sample-tokens", "document", "sample", "text"));
        layers.add(spans.layer("turbo:names", "document", "sample", "start", "end", "type", "nulls"));
      }
      case TOKEN, SENTENCE -> {
        FormatGolden.Columns texts = new FormatGolden.Columns(U32);
        FormatGolden.Columns spans = new FormatGolden.Columns(U32);
        for (int s = 0; s < samples.size(); s++) {
          String text;
          Span[] parts;
          if (samples.get(s) instanceof TokenSample t) {
            text = t.getText();
            parts = t.getTokenSpans();
          } else {
            SentenceSample d = (SentenceSample) samples.get(s);
            text = d.getDocument();
            parts = d.getSentences();
          }
          texts.str("text", text);
          Utf8Offsets offsets = new Utf8Offsets(input.inputId() + " sample " + s, text);
          for (Span span : parts) {
            Utf8Offsets.ByteSpan b = offsets.span(span.getStart(), span.getEnd());
            spans.u32("sample", s).u32("start", b.start()).u32("end", b.end());
          }
        }
        layers.add(texts.layer("turbo:sample-texts", "document", "text"));
        layers.add(spans.layer("turbo:sample-spans", "document", "sample", "start", "end"));
      }
      case DOCUMENTS -> throw new IllegalStateException("not a sample run");
    }
    return layers;
  }
}
