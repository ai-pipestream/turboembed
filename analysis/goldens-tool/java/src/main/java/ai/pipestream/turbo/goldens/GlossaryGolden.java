package ai.pipestream.turbo.goldens;

import java.io.ByteArrayInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;

import opennlp.tools.formats.glossary.CsvGlossaryReader;
import opennlp.tools.formats.glossary.GlossaryReader;
import opennlp.tools.formats.glossary.TbxGlossaryReader;
import opennlp.tools.glossary.AhoCorasickGlossaryMatcher;
import opennlp.tools.glossary.CompositeGlossaryMatcher;
import opennlp.tools.glossary.GlossaryEntry;
import opennlp.tools.glossary.GlossaryMatch;
import opennlp.tools.glossary.GlossaryMatcher;
import opennlp.tools.glossary.TermAnalyzingGlossaryMatcher;
import opennlp.tools.stemmer.PorterStemmer;
import opennlp.tools.util.StringUtil;
import opennlp.tools.util.normalizer.Term;
import opennlp.tools.util.normalizer.TermAnalyzer;
import opennlp.tools.util.normalizer.WhitespaceCharSequenceNormalizer;

/**
 * OpenNLP's glossary matchers and glossary readers, over the {@code glossary} input set.
 *
 * <p>The matcher runs read the {@code text-} inputs with the fixture glossary
 * ({@code glossary-terms.tsv}, read with {@code CsvGlossaryReader('\t', true)}) and write
 * {@code turbo:glossary-terms}: per match its UTF-8 byte span and {@code entry}, the row of the
 * fixture's first entry with the match's id and term. The runs:</p>
 * <ul>
 *   <li>{@code exact} and {@code ignore-case}: {@code AhoCorasickGlossaryMatcher} with
 *       {@code ignoreCase} false and true;</li>
 *   <li>{@code whitespace+ignore-case}: the same with {@code WhitespaceCharSequenceNormalizer}
 *       folding terms and text;</li>
 *   <li>{@code analyzed}: {@code TermAnalyzingGlossaryMatcher} with a {@code TermAnalyzer} that
 *       folds case and stems with {@code PorterStemmer}, over the entries whose terms give at
 *       least one token and no blank one (the matcher refuses the others);</li>
 *   <li>{@code composite}: {@code CompositeGlossaryMatcher} of the exact matcher over every entry,
 *       then the analyzed one.</li>
 * </ul>
 * <p>The analyzed and composite runs also write {@code turbo:analyzed-terms}, the analyzer's
 * tokens of the text with their normalized values, and {@code turbo:analyzed-entries}, the
 * normalized tokens of each entry the analyzed matcher was given.</p>
 *
 * <p>The reader runs write {@code turbo:glossary-entries}, the id and term of each entry in file
 * order, or the exception: {@code csv} ({@code CsvGlossaryReader()}) on the {@code csv-} inputs,
 * {@code tsv-header} ({@code CsvGlossaryReader('\t', true)}) on the {@code tsv-} inputs, and
 * {@code tbx-en} and {@code tbx-de} ({@code TbxGlossaryReader}) on the {@code tbx-} inputs.</p>
 */
final class GlossaryGolden implements AnnotatorGolden {

  static final String SET = Runs.GLOSSARY;

  enum Kind {
    EXACT("exact", "text-"),
    IGNORE_CASE("ignore-case", "text-"),
    WHITESPACE("whitespace+ignore-case", "text-"),
    ANALYZED("analyzed", "text-"),
    COMPOSITE("composite", "text-"),
    CSV("csv", "csv-"),
    TSV("tsv-header", "tsv-"),
    TBX_EN("tbx-en", "tbx-"),
    TBX_DE("tbx-de", "tbx-");

    final String dir;
    final String inputs;

    Kind(String dir, String inputs) {
      this.dir = dir;
      this.inputs = inputs;
    }
  }

  private final Kind kind;
  private final List<GlossaryEntry> entries;

  private GlossaryGolden(Kind kind, List<GlossaryEntry> entries) {
    this.kind = kind;
    this.entries = entries;
  }

  static List<GlossaryGolden> all(Path fixture) throws IOException {
    List<GlossaryEntry> entries;
    try (InputStream in = Files.newInputStream(fixture)) {
      entries = new CsvGlossaryReader('\t', true).read(in);
    }
    List<GlossaryGolden> out = new ArrayList<>();
    for (Kind k : Kind.values()) {
      out.add(new GlossaryGolden(k, entries));
    }
    return out;
  }

  @Override
  public String annotator() {
    return "glossary";
  }

  @Override
  public List<ModelRef> models() {
    return List.of();
  }

  @Override
  public String modelDir() {
    return kind.dir;
  }

  @Override
  public String annotatorClass() {
    return switch (kind) {
      case EXACT, IGNORE_CASE, WHITESPACE -> AhoCorasickGlossaryMatcher.class.getName();
      case ANALYZED -> TermAnalyzingGlossaryMatcher.class.getName();
      case COMPOSITE -> CompositeGlossaryMatcher.class.getName();
      case CSV, TSV -> CsvGlossaryReader.class.getName();
      case TBX_EN, TBX_DE -> TbxGlossaryReader.class.getName();
    };
  }

  @Override
  public String method() {
    return switch (kind) {
      case CSV, TSV, TBX_EN, TBX_DE -> "read";
      default -> "match";
    };
  }

  @Override
  public Map<String, String> config() {
    return switch (kind) {
      case EXACT -> Map.of("ignoreCase", "false");
      case IGNORE_CASE -> Map.of("ignoreCase", "true");
      case WHITESPACE -> Map.of("ignoreCase", "true", "normalizer", "WhitespaceCharSequenceNormalizer");
      case ANALYZED, COMPOSITE -> Map.of("analyzer", "caseFold stem(PorterStemmer)");
      case CSV -> Map.of("delimiter", ",", "skipHeader", "false");
      case TSV -> Map.of("delimiter", "\\t", "skipHeader", "true");
      case TBX_EN -> Map.of("languageTag", "en");
      case TBX_DE -> Map.of("languageTag", "de");
    };
  }

  @Override
  public Set<String> inputSets() {
    return Set.of(SET);
  }

  @Override
  public boolean accepts(Input input) {
    return input.set().equals(SET) && input.id().startsWith(kind.inputs);
  }

  @Override
  public List<Layer> layers(Input input) throws AnnotatorException {
    return switch (kind) {
      case CSV -> read(new CsvGlossaryReader(), input);
      case TSV -> read(new CsvGlossaryReader('\t', true), input);
      case TBX_EN -> read(new TbxGlossaryReader("en"), input);
      case TBX_DE -> read(new TbxGlossaryReader("de"), input);
      default -> match(input);
    };
  }

  private static List<Layer> read(GlossaryReader reader, Input input) throws AnnotatorException {
    List<GlossaryEntry> read;
    try {
      read = reader.read(new ByteArrayInputStream(input.bytes()));
    } catch (IOException e) {
      throw new AnnotatorException(e);
    } catch (RuntimeException e) {
      throw new AnnotatorException(e);
    }
    Table t = new Table("turbo:glossary-entries", "id:str", "term:str");
    for (GlossaryEntry e : read) {
      t.row(e.id(), e.term());
    }
    return List.of(t.layer());
  }

  private static TermAnalyzer analyzer() {
    return TermAnalyzer.builder().caseFold().stem(new PorterStemmer()).build();
  }

  /** The entries the term analyzing matcher accepts: at least one token, none blank. */
  private List<GlossaryEntry> analyzable(TermAnalyzer analyzer) {
    List<GlossaryEntry> out = new ArrayList<>();
    for (GlossaryEntry e : entries) {
      List<Term> terms = analyzer.analyze(e.term());
      boolean ok = !terms.isEmpty();
      for (Term t : terms) {
        ok &= !StringUtil.isBlank(t.normalized());
      }
      if (ok) {
        out.add(e);
      }
    }
    return out;
  }

  private List<Layer> match(Input input) throws AnnotatorException {
    Map<String, Integer> index = new HashMap<>();
    for (int i = entries.size() - 1; i >= 0; i--) {
      index.put(entries.get(i).id() + "\t" + entries.get(i).term(), i);
    }
    TermAnalyzer analyzer = analyzer();
    List<GlossaryEntry> analyzable = analyzable(analyzer);
    GlossaryMatcher matcher = switch (kind) {
      case EXACT -> new AhoCorasickGlossaryMatcher(entries, false);
      case IGNORE_CASE -> new AhoCorasickGlossaryMatcher(entries, true);
      case WHITESPACE -> new AhoCorasickGlossaryMatcher(entries, true,
          WhitespaceCharSequenceNormalizer.getInstance());
      case ANALYZED -> new TermAnalyzingGlossaryMatcher(analyzable, analyzer);
      case COMPOSITE -> new CompositeGlossaryMatcher(List.of(new AhoCorasickGlossaryMatcher(entries, false),
          new TermAnalyzingGlossaryMatcher(analyzable, analyzer)));
      default -> throw new IllegalStateException(kind.dir + " is not a matcher run");
    };
    List<GlossaryMatch> found = AnnotatorException.call(() -> matcher.match(input.text()));
    List<Layer> layers = new ArrayList<>();
    if (kind == Kind.ANALYZED || kind == Kind.COMPOSITE) {
      Table entryTokens = new Table("turbo:analyzed-entries", "entry:u32", "token:str");
      for (GlossaryEntry e : analyzable) {
        for (Term t : analyzer.analyze(e.term())) {
          entryTokens.row(index.get(e.id() + "\t" + e.term()), t.normalized());
        }
      }
      List<Term> terms = analyzer.analyze(input.text());
      long[] starts = new long[terms.size()];
      long[] ends = new long[terms.size()];
      String[] normalized = new String[terms.size()];
      for (int i = 0; i < terms.size(); i++) {
        Utf8Offsets.Enclosing b = bytes(input, terms.get(i).span().getStart(), terms.get(i).span().getEnd());
        starts[i] = b.start();
        ends[i] = b.end();
        normalized[i] = terms.get(i).normalized();
      }
      layers.add(new Layer("turbo:analyzed-terms", "text", new Column.U32("start", starts),
          new Column.U32("end", ends), new Column.Str("normalized", normalized)));
      layers.add(entryTokens.layer());
    }
    long[] starts = new long[found.size()];
    long[] ends = new long[found.size()];
    long[] entry = new long[found.size()];
    for (int i = 0; i < found.size(); i++) {
      GlossaryMatch m = found.get(i);
      Utf8Offsets.Enclosing b = bytes(input, m.span().getStart(), m.span().getEnd());
      starts[i] = b.start();
      ends[i] = b.end();
      Integer e = index.get(m.id() + "\t" + m.term());
      if (e == null) {
        throw new IllegalStateException("a match of no fixture entry: " + m);
      }
      entry[i] = e;
    }
    layers.add(new Layer("turbo:glossary-terms", "text", new Column.U32("start", starts),
        new Column.U32("end", ends), new Column.U32("entry", entry)));
    return layers;
  }

  /** The bytes of a UTF-16 span, which must not cut a character. */
  private static Utf8Offsets.Enclosing bytes(Input input, int start, int end) {
    Utf8Offsets.Enclosing b = input.offsets().enclosing(start, end);
    if (b.cut()) {
      throw new IllegalStateException(input.inputId() + ": a span cuts a character: " + start + ".." + end);
    }
    return b;
  }
}
