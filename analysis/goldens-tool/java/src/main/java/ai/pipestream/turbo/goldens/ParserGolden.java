package ai.pipestream.turbo.goldens;

import java.io.ByteArrayInputStream;
import java.io.IOException;
import java.io.StringReader;
import java.io.StringWriter;
import java.nio.charset.StandardCharsets;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.Set;

import opennlp.tools.chunker.ChunkSample;
import opennlp.tools.parser.ChunkSampleStream;
import opennlp.tools.parser.GapLabeler;
import opennlp.tools.parser.HeadRules;
import opennlp.tools.parser.Parse;
import opennlp.tools.parser.ParseSampleStream;
import opennlp.tools.parser.PosSampleStream;
import opennlp.tools.parser.lang.es.AncoraSpanishHeadRules;
import opennlp.tools.postag.POSSample;
import opennlp.tools.util.InputStreamFactory;
import opennlp.tools.util.ObjectStream;
import opennlp.tools.util.ObjectStreamUtils;
import opennlp.tools.util.PlainTextByLineStream;
import opennlp.tools.util.Span;

/**
 * The goldens of OpenNLP's Penn Treebank parse reader and writer and its head rules, over the
 * {@code parse} input set: each input holds one bracketed parse per line, read as the command
 * line tools read a file ({@code PlainTextByLineStream} over UTF-8), with blank lines skipped as
 * {@code ParseSampleStream} skips them.
 *
 * <p>The runs: {@code treebank} reads the lines with {@code ParseSampleStream};
 * {@code treebank+function-tags} does the same with {@code Parse.useFunctionTags(true)}, which is
 * set back to false afterwards. The head rule runs read each line with
 * {@code Parse.parseParse(line, rules)}, the rules being the gap labeler, then call
 * {@code updateHeads(rules)}: {@code en-head-rules} with OpenNLP's English rules,
 * {@code en-head-rules+possessives} with {@code pruneParse} and {@code fixPossesives} first, as
 * {@code AbstractParserEventStream} calls them, {@code es-head-rules} with OpenNLP's AnCora rules
 * and {@code es-regex-head-rules} with AnCora rules written for this project to reach the regular
 * expressions the rule tags can be. An exception on a line is recorded, with the number of parses
 * before it, and reading goes on.</p>
 *
 * <p>The layers, anchored to the document: {@code turbo:samples} ({@code toStringPennTreebank} of
 * each parse), {@code turbo:read-errors}, {@code turbo:parse-texts} (each parse's text),
 * {@code turbo:parse-nodes} (every node in preorder: its depth below the root, type, UTF-8 span in
 * the parse's text, the span of its head, whether it is its own head, and its head index),
 * {@code turbo:pos-samples} (what {@code PosSampleStream} makes of the parse: each token and tag)
 * and {@code turbo:chunk-samples} (what {@code ChunkSampleStream} makes of it: each token, tag and
 * chunk outcome).</p>
 */
final class ParserGolden implements AnnotatorGolden {

  /** The input set the parses are read from. */
  static final String SET = Runs.PARSE;

  /** The columns of the node layer that are {@code u32}. */
  private static final Set<String> NODE_U32 = Set.of("depth", "head-start", "head-end", "own-head", "head-index");

  enum Kind {
    TREEBANK,
    FUNCTION_TAGS,
    ENGLISH,
    ENGLISH_POSSESSIVES,
    SPANISH,
  }

  private final Kind kind;
  private final String dir;
  private final ModelFile rules;

  private ParserGolden(Kind kind, String dir, ModelFile rules) {
    this.kind = kind;
    this.dir = dir;
    this.rules = rules;
  }

  /** Every run, with the English rules, OpenNLP's AnCora rules and the AnCora rules written here. */
  static List<ParserGolden> all(Path english, Path spanish, Path spanishRegex) throws IOException {
    ModelFile en = ModelFile.read(english);
    ModelFile es = ModelFile.read(spanish);
    ModelFile esRegex = ModelFile.read(spanishRegex);
    return List.of(
        new ParserGolden(Kind.TREEBANK, "treebank", null),
        new ParserGolden(Kind.FUNCTION_TAGS, "treebank+function-tags", null),
        new ParserGolden(Kind.ENGLISH, en.ref().baseName(), en),
        new ParserGolden(Kind.ENGLISH_POSSESSIVES, en.ref().baseName() + "+possessives", en),
        new ParserGolden(Kind.SPANISH, es.ref().baseName(), es),
        new ParserGolden(Kind.SPANISH, esRegex.ref().baseName(), esRegex));
  }

  @Override
  public String annotator() {
    return "parse";
  }

  @Override
  public List<ModelRef> models() {
    return rules == null ? List.of() : List.of(rules.ref());
  }

  @Override
  public String modelDir() {
    return dir;
  }

  @Override
  public String annotatorClass() {
    return switch (kind) {
      case TREEBANK, FUNCTION_TAGS -> ParseSampleStream.class.getName();
      case ENGLISH, ENGLISH_POSSESSIVES, SPANISH -> Parse.class.getName();
    };
  }

  @Override
  public String method() {
    return switch (kind) {
      case TREEBANK, FUNCTION_TAGS -> "read";
      case ENGLISH, ENGLISH_POSSESSIVES, SPANISH -> "parseParse";
    };
  }

  @Override
  public Map<String, String> config() {
    return switch (kind) {
      case TREEBANK -> Map.of();
      case FUNCTION_TAGS -> Map.of("useFunctionTags", "true");
      case ENGLISH, SPANISH -> Map.of("rules", rulesClass().getName());
      case ENGLISH_POSSESSIVES -> Map.of("rules", rulesClass().getName(), "fixPossesives", "true");
    };
  }

  @Override
  public Set<String> inputSets() {
    return Set.of(SET);
  }

  private Class<?> rulesClass() {
    return kind == Kind.SPANISH ? AncoraSpanishHeadRules.class : opennlp.tools.parser.lang.en.HeadRules.class;
  }

  private HeadRules loadRules() throws IOException {
    String text = new String(rules.bytes(), StandardCharsets.UTF_8);
    return kind == Kind.SPANISH
        ? new AncoraSpanishHeadRules(new StringReader(text))
        : new opennlp.tools.parser.lang.en.HeadRules(new StringReader(text));
  }

  @Override
  public List<Layer> layers(Input input) {
    InputStreamFactory in = () -> new ByteArrayInputStream(input.bytes());
    List<Parse> parses = new ArrayList<>();
    FormatGolden.Columns errors = new FormatGolden.Columns();
    try (ObjectStream<String> lines = new PlainTextByLineStream(in, StandardCharsets.UTF_8)) {
      if (kind == Kind.TREEBANK || kind == Kind.FUNCTION_TAGS) {
        Parse.useFunctionTags(kind == Kind.FUNCTION_TAGS);
        try {
          readStream(new ParseSampleStream(lines), parses, errors);
        } finally {
          Parse.useFunctionTags(false);
        }
      } else {
        HeadRules headRules = loadRules();
        for (String line = lines.read(); line != null; line = lines.read()) {
          if (line.trim().isEmpty()) {
            continue;
          }
          try {
            Parse p = Parse.parseParse(line, (GapLabeler) headRules);
            if (kind == Kind.ENGLISH_POSSESSIVES) {
              Parse.pruneParse(p);
              Parse.fixPossesives(p);
            }
            p.updateHeads(headRules);
            parses.add(p);
          } catch (RuntimeException e) {
            errors.u32("after", parses.size()).str("class", e.getClass().getName())
                .str("message", FormatGolden.message(e));
          }
        }
      }
    } catch (IOException e) {
      throw new IllegalStateException(input.inputId() + ": the line stream failed", e);
    }
    return layers(input, parses, errors);
  }

  private static void readStream(ParseSampleStream stream, List<Parse> parses, FormatGolden.Columns errors)
      throws IOException {
    while (true) {
      Parse p;
      try {
        p = stream.read();
      } catch (RuntimeException e) {
        errors.u32("after", parses.size()).str("class", e.getClass().getName())
            .str("message", FormatGolden.message(e));
        continue;
      }
      if (p == null) {
        return;
      }
      parses.add(p);
    }
  }

  static List<Layer> layers(Input input, List<Parse> parses, FormatGolden.Columns errors) {
    FormatGolden.Columns written = new FormatGolden.Columns();
    FormatGolden.Columns texts = new FormatGolden.Columns();
    FormatGolden.Columns nodes = new FormatGolden.Columns(NODE_U32);
    FormatGolden.Columns pos = new FormatGolden.Columns();
    FormatGolden.Columns chunks = new FormatGolden.Columns();
    for (int i = 0; i < parses.size(); i++) {
      Parse p = parses.get(i);
      written.str("written", p.toStringPennTreebank());
      texts.u32("sample", i).str("text", p.getText());
      Utf8Offsets offsets = new Utf8Offsets(input.inputId() + " parse " + i, p.getText());
      nodes(p, 0, i, offsets, nodes);
      try {
        POSSample s = new PosSampleStream(ObjectStreamUtils.createObjectStream(p)).read();
        for (int t = 0; t < s.getSentence().length; t++) {
          pos.u32("sample", i).str("token", s.getSentence()[t]).str("tag", s.getTags()[t]);
        }
        ChunkSample c = new ChunkSampleStream(ObjectStreamUtils.createObjectStream(p)).read();
        for (int t = 0; t < c.getSentence().length; t++) {
          chunks.u32("sample", i).str("token", c.getSentence()[t]).str("tag", c.getTags()[t])
              .str("pred", c.getPreds()[t]);
        }
      } catch (IOException e) {
        throw new IllegalStateException("a stream over one parse failed", e);
      }
    }
    return List.of(
        written.layer("turbo:samples", "document", "written"),
        errors.layer("turbo:read-errors", "document", "after", "class", "message"),
        texts.layer("turbo:parse-texts", "document", "sample", "text"),
        nodes.layer("turbo:parse-nodes", "document", "sample", "depth", "type", "start", "end", "head-start",
            "head-end", "own-head", "head-index"),
        pos.layer("turbo:pos-samples", "document", "sample", "token", "tag"),
        chunks.layer("turbo:chunk-samples", "document", "sample", "token", "tag", "pred"));
  }

  private static void nodes(Parse node, int depth, int sample, Utf8Offsets offsets, FormatGolden.Columns out) {
    Span span = node.getSpan();
    Span head = node.getHead().getSpan();
    Utf8Offsets.ByteSpan bytes = offsets.span(span.getStart(), span.getEnd());
    Utf8Offsets.ByteSpan headBytes = offsets.span(head.getStart(), head.getEnd());
    if (node.getHeadIndex() < 0) {
      throw new IllegalStateException("a negative head index in parse " + sample);
    }
    out.u32("sample", sample).u32("depth", depth).str("type", node.getType()).u32("start", bytes.start())
        .u32("end", bytes.end()).u32("head-start", headBytes.start()).u32("head-end", headBytes.end())
        .u32("own-head", node.getHead() == node ? 1 : 0).u32("head-index", node.getHeadIndex());
    for (Parse child : node.getChildren()) {
      nodes(child, depth + 1, sample, offsets, out);
    }
  }

  /**
   * The goldens of OpenNLP's head rule readers and writers over the {@code head-rules} input set,
   * each input a rule file: {@code en} reads it with {@code opennlp.tools.parser.lang.en.HeadRules}
   * and {@code es} with {@code AncoraSpanishHeadRules}, and writes it back with {@code serialize}.
   * The layers are {@code turbo:head-rules}, one row per line written, and
   * {@code turbo:rules-errors}, the class and message of what the reader threw, which leaves the
   * first empty.
   */
  static final class Rules implements AnnotatorGolden {

    /** The input set the rule files are read from. */
    static final String SET = Runs.HEAD_RULES;

    private final boolean spanish;

    private Rules(boolean spanish) {
      this.spanish = spanish;
    }

    static List<Rules> all() {
      return List.of(new Rules(false), new Rules(true));
    }

    @Override
    public String annotator() {
      return "head-rules";
    }

    @Override
    public List<ModelRef> models() {
      return List.of();
    }

    @Override
    public String modelDir() {
      return spanish ? "es" : "en";
    }

    @Override
    public String annotatorClass() {
      return spanish ? AncoraSpanishHeadRules.class.getName() : opennlp.tools.parser.lang.en.HeadRules.class.getName();
    }

    @Override
    public String method() {
      return "serialize";
    }

    @Override
    public Set<String> inputSets() {
      return Set.of(SET);
    }

    @Override
    public List<Layer> layers(Input input) {
      FormatGolden.Columns lines = new FormatGolden.Columns();
      FormatGolden.Columns errors = new FormatGolden.Columns();
      StringWriter out = new StringWriter();
      try {
        if (spanish) {
          new AncoraSpanishHeadRules(new StringReader(input.text())).serialize(out);
        } else {
          new opennlp.tools.parser.lang.en.HeadRules(new StringReader(input.text())).serialize(out);
        }
      } catch (IOException | RuntimeException e) {
        errors.str("class", e.getClass().getName()).str("message", FormatGolden.message(e));
      }
      String written = out.toString();
      int start = 0;
      for (int end = written.indexOf('\n'); end >= 0; end = written.indexOf('\n', start)) {
        lines.str("line", written.substring(start, end));
        start = end + 1;
      }
      if (start != written.length()) {
        throw new IllegalStateException(input.inputId() + ": serialize did not end with a line feed");
      }
      return List.of(
          lines.layer("turbo:head-rules", "document", "line"),
          errors.layer("turbo:rules-errors", "document", "class", "message"));
    }
  }
}
