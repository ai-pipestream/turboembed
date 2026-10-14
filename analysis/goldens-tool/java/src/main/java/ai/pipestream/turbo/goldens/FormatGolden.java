package ai.pipestream.turbo.goldens;

import java.io.ByteArrayInputStream;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Set;
import opennlp.tools.chunker.ChunkSample;
import opennlp.tools.chunker.ChunkSampleStream;
import opennlp.tools.doccat.DocumentSample;
import opennlp.tools.doccat.DocumentSampleStream;
import opennlp.tools.formats.conllu.ConlluLemmaSampleStream;
import opennlp.tools.formats.conllu.ConlluPOSSampleStream;
import opennlp.tools.formats.conllu.ConlluSentenceSampleStream;
import opennlp.tools.formats.conllu.ConlluStream;
import opennlp.tools.formats.conllu.ConlluTagset;
import opennlp.tools.formats.conllu.ConlluTokenSampleStream;
import opennlp.tools.formats.convert.NameToSentenceSampleStream;
import opennlp.tools.formats.convert.NameToTokenSampleStream;
import opennlp.tools.formats.convert.POSToSentenceSampleStream;
import opennlp.tools.formats.convert.POSToTokenSampleStream;
import opennlp.tools.langdetect.LanguageDetectorSampleStream;
import opennlp.tools.langdetect.LanguageSample;
import opennlp.tools.lemmatizer.LemmaSample;
import opennlp.tools.lemmatizer.LemmaSampleStream;
import opennlp.tools.namefind.NameSample;
import opennlp.tools.namefind.NameSampleDataStream;
import opennlp.tools.postag.POSSample;
import opennlp.tools.postag.WordTagSampleStream;
import opennlp.tools.sentdetect.SentenceSample;
import opennlp.tools.sentdetect.SentenceSampleStream;
import opennlp.tools.tokenize.DetokenizationDictionary;
import opennlp.tools.tokenize.Detokenizer;
import opennlp.tools.tokenize.DictionaryDetokenizer;
import opennlp.tools.tokenize.TokenSample;
import opennlp.tools.tokenize.TokenSampleStream;
import opennlp.tools.util.InputStreamFactory;
import opennlp.tools.util.ObjectStream;
import opennlp.tools.util.PlainTextByLineStream;
import opennlp.tools.util.Span;
import opennlp.tools.util.WhitespaceMode;

/**
 * The golden of one of OpenNLP's sample streams over a file in its format: every sample the
 * stream returned and every exception it threw, read as the command line tools read a file
 * ({@code PlainTextByLineStream} over UTF-8). Its inputs are the {@code formats} set's files
 * whose id starts with the format's name and a hyphen ({@code name-edge.txt} for the name
 * format); a conversion stream reads the files of the format it converts from, and the CoNLL-U
 * streams read the files whose id starts with {@code conllu-}, each through a {@code ConlluStream}.
 *
 * <p>The stream is read until it returns null. An exception from {@code read} is recorded in
 * {@code turbo:read-errors} with the number of samples read before it, and reading goes on, as a
 * caller that skips a bad sample would. The layers, all anchored to the document:</p>
 *
 * <ul>
 *   <li>{@code turbo:samples}: per sample, what {@code toString} returned ({@code written}).</li>
 *   <li>{@code turbo:read-errors}: per exception, {@code after}, {@code class} and
 *       {@code message}.</li>
 *   <li>{@code turbo:sample-tokens}: per token of every sample, the sample's index and the
 *       token; with {@code turbo:sample-tags}, {@code turbo:sample-lemmas} and
 *       {@code turbo:sample-chunks} aligned to it for the samples that have them.</li>
 *   <li>{@code turbo:names} and {@code turbo:phrases}: per name or chunk phrase, the sample,
 *       the token span and the type; {@code turbo:phrase-errors} the exceptions
 *       {@code getPhrasesAsSpanList} threw, per sample.</li>
 *   <li>{@code turbo:sample-texts} and {@code turbo:sample-spans}: per token or sentence
 *       sample its own text, and per token or sentence its UTF-8 byte span in that text.</li>
 *   <li>{@code turbo:sample-labels}: per document category or language sample its category or
 *       language, and for a language sample its text in {@code turbo:sample-texts}.</li>
 * </ul>
 */
final class FormatGolden implements AnnotatorGolden {

  /** The input set the formats are read from. */
  static final String SET = Runs.FORMATS;

  /** The chunk size of the sentence conversions; small, so that the inputs show what a full chunk does. */
  static final int CHUNK_SIZE = 2;

  /** The sentences per sample of the CoNLL-U sentence stream, small for the same reason. */
  static final int SENTENCES_PER_SAMPLE = 2;

  /** The stopping point for a stream that throws without moving on, which would otherwise loop. */
  private static final int MAX_READS = 100_000;

  /** The streams, each with the prefix of the input ids it reads. */
  enum Kind {
    NAME("name", "name", NameSampleDataStream.class, true),
    WORDTAG("wordtag", "wordtag", WordTagSampleStream.class, true),
    LEMMA("lemma", "lemma", LemmaSampleStream.class, false),
    CHUNK("chunk", "chunk", ChunkSampleStream.class, false),
    TOKEN("token", "token", TokenSampleStream.class, true),
    SENTENCE("sentence", "sentence", SentenceSampleStream.class, false),
    DOCCAT("doccat", "doccat", DocumentSampleStream.class, true),
    LANGUAGE("language", "language", LanguageDetectorSampleStream.class, false),
    NAME_TO_TOKEN("name-to-token", "name", NameToTokenSampleStream.class, false),
    POS_TO_TOKEN("pos-to-token", "wordtag", POSToTokenSampleStream.class, false),
    NAME_TO_SENTENCE("name-to-sentence", "name", NameToSentenceSampleStream.class, false),
    POS_TO_SENTENCE("pos-to-sentence", "wordtag", POSToSentenceSampleStream.class, false),
    CONLLU_POS_U("conllu-pos-u", "conllu", ConlluPOSSampleStream.class, false),
    CONLLU_POS_X("conllu-pos-x", "conllu", ConlluPOSSampleStream.class, false),
    CONLLU_LEMMA_U("conllu-lemma-u", "conllu", ConlluLemmaSampleStream.class, false),
    CONLLU_LEMMA_X("conllu-lemma-x", "conllu", ConlluLemmaSampleStream.class, false),
    CONLLU_SENTENCE("conllu-sentence", "conllu", ConlluSentenceSampleStream.class, false),
    CONLLU_TOKEN("conllu-token", "conllu", ConlluTokenSampleStream.class, true);

    final String dir;
    final String inputPrefix;
    final Class<?> streamClass;
    /** Whether the stream splits lines with {@code WhitespaceTokenizer}, so it is also run in the legacy mode. */
    final boolean splitsWhitespace;

    Kind(String dir, String inputPrefix, Class<?> streamClass, boolean splitsWhitespace) {
      this.dir = dir;
      this.inputPrefix = inputPrefix;
      this.streamClass = streamClass;
      this.splitsWhitespace = splitsWhitespace;
    }

    boolean converts() {
      return this == NAME_TO_TOKEN || this == POS_TO_TOKEN || this == NAME_TO_SENTENCE || this == POS_TO_SENTENCE;
    }

    boolean chunks() {
      return this == NAME_TO_SENTENCE || this == POS_TO_SENTENCE;
    }

    /** The tag set of a CoNLL-U part-of-speech or lemma stream, null for the others. */
    ConlluTagset tagset() {
      return switch (this) {
        case CONLLU_POS_U, CONLLU_LEMMA_U -> ConlluTagset.U;
        case CONLLU_POS_X, CONLLU_LEMMA_X -> ConlluTagset.X;
        default -> null;
      };
    }
  }

  private final Kind kind;
  private final WhitespaceMode whitespace;
  private final List<ModelRef> models;
  private final Detokenizer detokenizer;

  private FormatGolden(Kind kind, WhitespaceMode whitespace, List<ModelRef> models, Detokenizer detokenizer) {
    this.kind = kind;
    this.whitespace = whitespace;
    this.models = List.copyOf(models);
    this.detokenizer = detokenizer;
  }

  /**
   * Every run: each native stream, in the legacy whitespace mode too when it splits on
   * whitespace, and each conversion stream with the detokenizer dictionary {@code dictionary}.
   */
  static List<FormatGolden> all(Path dictionary) throws IOException {
    ModelFile d = ModelFile.read(dictionary);
    Detokenizer detokenizer = new DictionaryDetokenizer(new DetokenizationDictionary(d.stream()));
    List<FormatGolden> runs = new ArrayList<>();
    for (Kind k : Kind.values()) {
      if (k.converts()) {
        runs.add(new FormatGolden(k, WhitespaceMode.UNICODE, List.of(d.ref()), detokenizer));
        continue;
      }
      runs.add(new FormatGolden(k, WhitespaceMode.UNICODE, List.of(), null));
      if (k.splitsWhitespace) {
        runs.add(new FormatGolden(k, WhitespaceMode.LEGACY, List.of(), null));
      }
    }
    return runs;
  }

  @Override
  public String annotator() {
    return "formats";
  }

  @Override
  public List<ModelRef> models() {
    return models;
  }

  @Override
  public String annotatorClass() {
    return kind.streamClass.getName();
  }

  @Override
  public String method() {
    return "read";
  }

  @Override
  public Map<String, String> config() {
    if (kind.chunks()) {
      return Map.of("chunkSize", Integer.toString(CHUNK_SIZE));
    }
    if (kind == Kind.CONLLU_SENTENCE) {
      return Map.of("sentencesPerSample", Integer.toString(SENTENCES_PER_SAMPLE));
    }
    if (kind.tagset() != null) {
      return Map.of("tagset", kind.tagset().name().toLowerCase(Locale.ROOT));
    }
    return Whitespace.config(whitespace);
  }

  @Override
  public Set<String> inputSets() {
    return Set.of(SET);
  }

  @Override
  public boolean accepts(Input input) {
    return input.set().equals(SET) && input.id().startsWith(kind.inputPrefix + "-");
  }

  @Override
  public String modelDir() {
    return kind.dir + Whitespace.dirSuffix(whitespace);
  }

  @Override
  public List<Layer> layers(Input input) throws AnnotatorException {
    return Whitespace.in(whitespace, () -> read(input));
  }

  private ObjectStream<?> open(Input input) throws IOException {
    InputStreamFactory in = () -> new ByteArrayInputStream(input.bytes());
    switch (kind) {
      case CONLLU_POS_U, CONLLU_POS_X:
        return new ConlluPOSSampleStream(new ConlluStream(in), kind.tagset());
      case CONLLU_LEMMA_U, CONLLU_LEMMA_X:
        return new ConlluLemmaSampleStream(new ConlluStream(in), kind.tagset());
      case CONLLU_SENTENCE:
        return new ConlluSentenceSampleStream(new ConlluStream(in), SENTENCES_PER_SAMPLE);
      case CONLLU_TOKEN:
        return new ConlluTokenSampleStream(new ConlluStream(in));
      default:
        break;
    }
    ObjectStream<String> lines = new PlainTextByLineStream(in, StandardCharsets.UTF_8);
    return switch (kind) {
      case NAME -> new NameSampleDataStream(lines);
      case WORDTAG -> new WordTagSampleStream(lines);
      case LEMMA -> new LemmaSampleStream(lines);
      case CHUNK -> new ChunkSampleStream(lines);
      case TOKEN -> new TokenSampleStream(lines);
      case SENTENCE -> new SentenceSampleStream(lines);
      case DOCCAT -> new DocumentSampleStream(lines);
      case LANGUAGE -> new LanguageDetectorSampleStream(lines);
      case NAME_TO_TOKEN -> new NameToTokenSampleStream(detokenizer, new NameSampleDataStream(lines));
      case POS_TO_TOKEN -> new POSToTokenSampleStream(detokenizer, new WordTagSampleStream(lines));
      case NAME_TO_SENTENCE ->
          new NameToSentenceSampleStream(detokenizer, new NameSampleDataStream(lines), CHUNK_SIZE);
      case POS_TO_SENTENCE -> new POSToSentenceSampleStream(detokenizer, new WordTagSampleStream(lines), CHUNK_SIZE);
      case CONLLU_POS_U, CONLLU_POS_X, CONLLU_LEMMA_U, CONLLU_LEMMA_X, CONLLU_SENTENCE, CONLLU_TOKEN ->
          throw new IllegalStateException("opened above");
    };
  }

  private List<Layer> read(Input input) {
    List<Object> samples = new ArrayList<>();
    Columns errors = new Columns();
    try (ObjectStream<?> stream = open(input)) {
      for (int reads = 0; ; reads++) {
        if (reads > MAX_READS) {
          throw new IllegalStateException(input.inputId() + ": the stream does not end");
        }
        Object sample;
        try {
          sample = stream.read();
        } catch (IOException | RuntimeException e) {
          errors.u32("after", samples.size()).str("class", e.getClass().getName()).str("message", message(e));
          continue;
        }
        if (sample == null) {
          break;
        }
        samples.add(sample);
      }
    } catch (IOException e) {
      throw new IllegalStateException(input.inputId() + ": the line stream failed", e);
    }

    List<Layer> layers = new ArrayList<>();
    Columns written = new Columns();
    for (Object s : samples) {
      written.str("written", s.toString());
    }
    layers.add(written.layer("turbo:samples", "document", "written"));
    layers.add(errors.layer("turbo:read-errors", "document", "after", "class", "message"));
    switch (kind) {
      case NAME -> names(samples, layers);
      case WORDTAG, CONLLU_POS_U, CONLLU_POS_X -> tagged(samples, layers);
      case LEMMA, CONLLU_LEMMA_U, CONLLU_LEMMA_X -> lemmas(samples, layers);
      case CHUNK -> chunks(samples, layers);
      case TOKEN, NAME_TO_TOKEN, POS_TO_TOKEN, CONLLU_TOKEN -> tokenSamples(input, samples, layers);
      case SENTENCE, NAME_TO_SENTENCE, POS_TO_SENTENCE, CONLLU_SENTENCE -> sentenceSamples(input, samples, layers);
      case DOCCAT -> documents(samples, layers);
      case LANGUAGE -> languages(samples, layers);
    }
    return layers;
  }

  static String message(Exception e) {
    return e.getMessage() == null ? "" : e.getMessage();
  }

  private static Layer tokens(List<String[]> perSample) {
    Columns c = new Columns();
    for (int s = 0; s < perSample.size(); s++) {
      for (String t : perSample.get(s)) {
        c.u32("sample", s).str("text", t);
      }
    }
    return c.layer("turbo:sample-tokens", "document", "sample", "text");
  }

  private static Layer aligned(String id, String column, List<String[]> perSample) {
    Columns c = new Columns();
    for (String[] values : perSample) {
      for (String v : values) {
        c.str(column, v);
      }
    }
    return c.layer(id, "aligned:turbo:sample-tokens", column);
  }

  private static Layer spans(String id, List<Span[]> perSample) {
    Columns c = new Columns();
    for (int s = 0; s < perSample.size(); s++) {
      for (Span span : perSample.get(s)) {
        c.u32("sample", s).u32("start", span.getStart()).u32("end", span.getEnd()).str("type", span.getType());
      }
    }
    return c.layer(id, "document", "sample", "start", "end", "type");
  }

  static void names(List<Object> samples, List<Layer> layers) {
    List<String[]> toks = new ArrayList<>();
    List<Span[]> names = new ArrayList<>();
    for (Object o : samples) {
      NameSample s = (NameSample) o;
      toks.add(s.getSentence());
      names.add(s.getNames());
    }
    layers.add(tokens(toks));
    layers.add(spans("turbo:names", names));
  }

  static void tagged(List<Object> samples, List<Layer> layers) {
    List<String[]> toks = new ArrayList<>();
    List<String[]> tags = new ArrayList<>();
    for (Object o : samples) {
      POSSample s = (POSSample) o;
      toks.add(s.getSentence());
      tags.add(s.getTags());
    }
    layers.add(tokens(toks));
    layers.add(aligned("turbo:sample-tags", "tag", tags));
  }

  private static void lemmas(List<Object> samples, List<Layer> layers) {
    List<String[]> toks = new ArrayList<>();
    List<String[]> tags = new ArrayList<>();
    List<String[]> lemmas = new ArrayList<>();
    for (Object o : samples) {
      LemmaSample s = (LemmaSample) o;
      toks.add(s.getTokens());
      tags.add(s.getTags());
      lemmas.add(s.getLemmas());
    }
    layers.add(tokens(toks));
    layers.add(aligned("turbo:sample-tags", "tag", tags));
    layers.add(aligned("turbo:sample-lemmas", "lemma", lemmas));
  }

  private static void chunks(List<Object> samples, List<Layer> layers) {
    List<String[]> toks = new ArrayList<>();
    List<String[]> tags = new ArrayList<>();
    List<String[]> preds = new ArrayList<>();
    List<Span[]> phrases = new ArrayList<>();
    Columns phraseErrors = new Columns();
    for (int i = 0; i < samples.size(); i++) {
      ChunkSample s = (ChunkSample) samples.get(i);
      toks.add(s.getSentence());
      tags.add(s.getTags());
      preds.add(s.getPreds());
      try {
        phrases.add(s.getPhrasesAsSpanList());
      } catch (RuntimeException e) {
        phrases.add(new Span[0]);
        phraseErrors.u32("sample", i).str("class", e.getClass().getName()).str("message", message(e));
      }
    }
    layers.add(tokens(toks));
    layers.add(aligned("turbo:sample-tags", "tag", tags));
    layers.add(aligned("turbo:sample-chunks", "chunk", preds));
    layers.add(spans("turbo:phrases", phrases));
    layers.add(phraseErrors.layer("turbo:phrase-errors", "document", "sample", "class", "message"));
  }

  private static void texts(List<String> texts, List<Span[]> spans, Input input, List<Layer> layers) {
    Columns t = new Columns();
    Columns c = new Columns();
    for (int s = 0; s < texts.size(); s++) {
      String text = texts.get(s);
      t.str("text", text);
      Utf8Offsets offsets = new Utf8Offsets(input.inputId() + " sample " + s, text);
      for (Span span : spans.get(s)) {
        Utf8Offsets.ByteSpan b = offsets.span(span.getStart(), span.getEnd());
        c.u32("sample", s).u32("start", b.start()).u32("end", b.end());
      }
    }
    layers.add(t.layer("turbo:sample-texts", "document", "text"));
    layers.add(c.layer("turbo:sample-spans", "document", "sample", "start", "end"));
  }

  static void tokenSamples(Input input, List<Object> samples, List<Layer> layers) {
    List<String> texts = new ArrayList<>();
    List<Span[]> spans = new ArrayList<>();
    for (Object o : samples) {
      TokenSample s = (TokenSample) o;
      texts.add(s.getText());
      spans.add(s.getTokenSpans());
    }
    texts(texts, spans, input, layers);
  }

  static void sentenceSamples(Input input, List<Object> samples, List<Layer> layers) {
    List<String> texts = new ArrayList<>();
    List<Span[]> spans = new ArrayList<>();
    for (Object o : samples) {
      SentenceSample s = (SentenceSample) o;
      texts.add(s.getDocument());
      spans.add(s.getSentences());
    }
    texts(texts, spans, input, layers);
  }

  private static void documents(List<Object> samples, List<Layer> layers) {
    Columns labels = new Columns();
    List<String[]> toks = new ArrayList<>();
    for (Object o : samples) {
      DocumentSample s = (DocumentSample) o;
      labels.str("label", s.getCategory());
      toks.add(s.getText());
    }
    layers.add(labels.layer("turbo:sample-labels", "document", "label"));
    layers.add(tokens(toks));
  }

  private static void languages(List<Object> samples, List<Layer> layers) {
    Columns labels = new Columns();
    Columns texts = new Columns();
    for (Object o : samples) {
      LanguageSample s = (LanguageSample) o;
      labels.str("label", s.language().getLang());
      texts.str("text", s.context().toString());
    }
    layers.add(labels.layer("turbo:sample-labels", "document", "label"));
    layers.add(texts.layer("turbo:sample-texts", "document", "text"));
  }

  /**
   * Builds the columns of a layer row by row: each call appends one value to the named column,
   * and {@link #layer} writes the columns in the order given, typed by what was appended. A
   * column given to {@code layer} with no values (a layer without rows) is a {@code str} column
   * unless it is named in {@link #U32_COLUMNS} or in the names given to the constructor.
   */
  static final class Columns {
    private static final Set<String> U32_COLUMNS = Set.of("after", "sample", "start", "end", "partitions", "fold");

    private final Set<String> u32Names;
    private final Map<String, List<Long>> u32 = new java.util.HashMap<>();
    private final Map<String, List<String>> str = new java.util.HashMap<>();

    Columns() {
      this(Set.of());
    }

    /** Columns where each of {@code u32Names} is also a {@code u32} column when it has no values. */
    Columns(Set<String> u32Names) {
      Set<String> all = new java.util.HashSet<>(U32_COLUMNS);
      all.addAll(u32Names);
      this.u32Names = Set.copyOf(all);
    }

    Columns u32(String name, long value) {
      u32.computeIfAbsent(name, k -> new ArrayList<>()).add(value);
      return this;
    }

    Columns str(String name, String value) {
      str.computeIfAbsent(name, k -> new ArrayList<>()).add(value == null ? "" : value);
      return this;
    }

    Layer layer(String id, String anchor, String... names) {
      List<Column> columns = new ArrayList<>();
      for (String n : names) {
        if (u32Names.contains(n)) {
          List<Long> v = u32.getOrDefault(n, List.of());
          columns.add(new Column.U32(n, v.stream().mapToLong(Long::longValue).toArray()));
        } else {
          List<String> v = str.getOrDefault(n, List.of());
          columns.add(new Column.Str(n, v.toArray(new String[0])));
        }
      }
      return new Layer(id, anchor, columns);
    }
  }
}
