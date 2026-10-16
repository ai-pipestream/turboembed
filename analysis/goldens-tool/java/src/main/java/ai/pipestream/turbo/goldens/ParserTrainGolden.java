package ai.pipestream.turbo.goldens;

import java.io.ByteArrayInputStream;
import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.StringReader;
import java.io.StringWriter;
import java.nio.charset.StandardCharsets;
import java.nio.file.Path;
import java.time.LocalDateTime;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Properties;
import java.util.Set;
import java.util.TreeMap;
import java.util.zip.ZipEntry;
import java.util.zip.ZipInputStream;
import java.util.zip.ZipOutputStream;
import opennlp.tools.cmdline.parser.ParserTool;
import opennlp.tools.dictionary.Dictionary;
import opennlp.tools.ml.model.Event;
import opennlp.tools.parser.AbstractBottomUpParser;
import opennlp.tools.parser.AbstractParserEventStream;
import opennlp.tools.parser.ChunkContextGenerator;
import opennlp.tools.parser.HeadRules;
import opennlp.tools.parser.Parse;
import opennlp.tools.parser.ParseSampleStream;
import opennlp.tools.parser.Parser;
import opennlp.tools.parser.ParserEvaluator;
import opennlp.tools.parser.ParserEventTypeEnum;
import opennlp.tools.parser.ParserFactory;
import opennlp.tools.parser.ParserModel;
import opennlp.tools.parser.ParserType;
import opennlp.tools.util.ObjectStream;
import opennlp.tools.util.ObjectStreamUtils;
import opennlp.tools.util.PlainTextByLineStream;
import opennlp.tools.util.Span;
import opennlp.tools.util.StringList;
import opennlp.tools.util.TrainingParameters;
import opennlp.tools.util.eval.FMeasure;

/**
 * The bottom-up parsers of {@code opennlp.tools.parser} ({@code chunking} and {@code treeinsert}):
 * the n-gram dictionary, the event streams, the context generators called directly, and models
 * trained on the {@code treebank} input of the {@code parse-treebank} set, parsing every input of
 * that set and scored by {@code ParserEvaluator}. The golden directory is {@code parse/<run>}, beside
 * the reader and head rule runs of {@link ParserGolden}; every run loads the English head rules
 * fixture.
 *
 * <p>The inputs hold one Penn Treebank parse per line, written for this project. A sample is a line
 * that is not blank, numbered from 0 in the order read, and read with {@code Parse.parseParse} as
 * {@link ParseSampleStream} reads it; every call starts from a fresh read of the line, since the
 * streams and the generators change the parse they are given. An exception is recorded with the
 * sample it was thrown on, and the generator goes on with the next sample.</p>
 *
 * <p>The runs:</p>
 * <ul>
 *   <li>{@code dictionary-cutoff-1}, {@code dictionary-cutoff-5}:
 *   {@code AbstractBottomUpParser.buildDictionary(samples, rules, cutoff)} over every sample of
 *   the input. {@code turbo:dictionary} holds one row per token of each n-gram: the n-gram's
 *   number and the token. The n-grams are sorted by their tokens ({@code String.compareTo}, token
 *   by token, a shorter list first when it is the start of a longer one).</li>
 *   <li>{@code events-chunking-build}, {@code events-chunking-build+dictionary},
 *   {@code events-chunking-check}, {@code events-treeinsert-build},
 *   {@code events-treeinsert-check}, {@code events-treeinsert-attach}, {@code events-chunk},
 *   {@code events-tag}: the events of {@code chunking.ParserEventStream} or
 *   {@code treeinsert.ParserEventStream} of the event type, each over one sample; the
 *   {@code +dictionary} run gives the stream the dictionary of {@code dictionary-cutoff-1} of the
 *   same input, as {@code chunking.Parser.train} gives it the dictionary it builds.
 *   {@code events-chunk} and {@code events-tag} are the {@code CHUNK} and {@code TAG} events
 *   {@code AbstractParserEventStream} makes, read through {@code chunking.ParserEventStream}.
 *   {@code turbo:events} holds one row per event: the sample, the outcome, the number of context
 *   strings and the context strings in order, joined by single spaces (a context string holding a
 *   space stops the generator). {@code turbo:sample-errors} holds the sample, class and message of
 *   each exception.</li>
 *   <li>{@code generators}: the context generators called directly on fixed chunk arrays. A
 *   sample's chunk array is what the event streams start from: the sample after
 *   {@code Parse.pruneParse} and {@code updateHeads(rules)}, its
 *   {@code AbstractParserEventStream.getInitialChunks}, then
 *   {@code AbstractBottomUpParser.collapsePunctuation} with the rules' punctuation tags. Each call
 *   is made in two states of the array: {@code unlabeled}, as made, and {@code complete}, with the
 *   label of every chunk set to {@code Parser.COMPLETE} ({@code c}). The calls, each a layer with
 *   the sample, the state, the call's arguments, the number of context strings and the context
 *   joined as in {@code turbo:events}:
 *   <ul>
 *     <li>{@code turbo:chunker-contexts}: {@code ChunkContextGenerator.getContext(i, words, tags,
 *     preds)} at every token, with the words, tags and chunk outcomes the {@code CHUNK} events are
 *     made from (no state; the column is {@code unlabeled}).</li>
 *     <li>{@code turbo:chunking-build} and {@code turbo:chunking-build-dictionary}:
 *     {@code chunking.BuildContextGenerator.getContext(chunks, index)} at every index, without
 *     and with the {@code dictionary-cutoff-1} dictionary of the input.</li>
 *     <li>{@code turbo:chunking-check}: {@code chunking.CheckContextGenerator.getContext(chunks,
 *     type, start, end)} for every start and every end at most two chunks after it, with each
 *     type of {@link #CHECK_TYPES}.</li>
 *     <li>{@code turbo:treeinsert-build}: {@code treeinsert.BuildContextGenerator.getContext(chunks,
 *     index)} at every index.</li>
 *     <li>{@code turbo:treeinsert-attach}: {@code treeinsert.AttachContextGenerator.getContext(chunks,
 *     index, frontier, frontier index)} at every index from 1, the frontier being the chunks
 *     before the index, nearest first, at every frontier index.</li>
 *     <li>{@code turbo:treeinsert-check}: {@code treeinsert.CheckContextGenerator.getContext(chunk,
 *     chunks, index, trim)} for every chunk that is not a part-of-speech tag, as the parent, at
 *     its own index, with {@code trim} false and true.</li>
 *   </ul>
 *   {@code turbo:chunks} lists each array: the sample, the index, the chunk's type, its UTF-8
 *   span in the sample's text ({@code turbo:parse-texts}), its head's span and its head index.
 *   {@code turbo:generator-errors} holds the layer, sample, state, index, class and message of
 *   each call that threw.</li>
 *   <li>{@code model-chunking}, {@code model-treeinsert}: {@code chunking.Parser.train} and
 *   {@code treeinsert.Parser.train} on the {@code treebank} input, read with
 *   {@link ParseSampleStream} over {@link PlainTextByLineStream} (UTF-8), which reads the lines
 *   again on each reset, with the parameters of {@link #trainingParameters()}. The golden of the
 *   {@code treebank} input holds the model file: the serialized model with every
 *   {@code Timestamp} manifest property removed and every zip entry, the nested tagger and
 *   chunker packages too, written again with a fixed time, so that the file is the same on each
 *   run ({@code turbo:model-file}, its SHA-256 and length; {@code turbo:model-hex}, its bytes
 *   as lowercase hex, 64 bytes a row; {@code turbo:model-entries}, the path, length and SHA-256 of each entry, a nested entry's path
 *   after its package's and a slash; {@code turbo:manifest}, each package's manifest). The goldens
 *   of every input hold, with the model read back from that file, for each beam size and advance
 *   percentage of {@link #SEARCHES} and each number of parses of {@link #PARSE_COUNTS}, what
 *   {@code ParserTool.parseLine} returns for the sample's tokens joined by spaces (the tokens of
 *   {@code getTokenNodes}, as {@code ParserEvaluator} makes the line): {@code turbo:parses} (the
 *   rank, {@code getProb} and {@code show}) and {@code turbo:parse-nodes} (every node in
 *   preorder: depth, type, UTF-8 span in the parse's text, the head's span and the head index),
 *   with {@code turbo:parse-errors}; and {@code turbo:evaluation}, the precision, recall and
 *   F-measure of {@code ParserEvaluator} over every sample of the input with each search and
 *   with {@code ParserFactory.create(model)}'s defaults (beam 20, advance 0.95).</li>
 * </ul>
 */
final class ParserTrainGolden implements AnnotatorGolden {

  static final String ANNOTATOR = "parse";
  /** The treebanks: the training sentences and the held out sentences. */
  static final String INPUT_SET = Runs.PARSE_TREEBANK;
  /** The input of {@link #INPUT_SET} the models are trained on. */
  static final String CORPUS = "treebank";
  static final String LANGUAGE = "en";

  /** The beam sizes and advance percentages the trained models parse with. */
  static final List<Search> SEARCHES = List.of(new Search(1, 0.5), new Search(3, 0.95));
  /** The numbers of parses asked for. */
  static final List<Integer> PARSE_COUNTS = List.of(1, 3);
  /** The constituent types the chunking check generator is asked about. */
  static final List<String> CHECK_TYPES = List.of("NP", "VP", "S");
  /** The iterations and cutoff of every model the parsers train. */
  static final int ITERATIONS = 100;
  static final int CUTOFF = 1;
  /** The parameter namespaces of the models the parsers train. */
  static final List<String> MODEL_NAMESPACES = List.of("build", "check", "attach", "tagger", "chunker");

  private static final int HEX_ROW_BYTES = 64;
  /** The time every zip entry of a normalized model file is written with. */
  private static final LocalDateTime ENTRY_TIME = LocalDateTime.of(2000, 1, 1, 0, 0);

  record Search(int beam, double advance) {}

  enum Kind {
    DICTIONARY,
    EVENTS,
    GENERATORS,
    MODEL,
  }

  private final Kind kind;
  private final String run;
  private final ModelFile rules;
  /** DICTIONARY: the cutoff. */
  private final int cutoff;
  /** EVENTS: the stream's parser type and event type, and whether it gets the dictionary. */
  private final ParserType streamType;
  private final ParserEventTypeEnum eventType;
  private final boolean withDictionary;
  /** MODEL: the trained model. */
  private final Trained trained;

  private ParserTrainGolden(Kind kind, String run, ModelFile rules, int cutoff, ParserType streamType,
      ParserEventTypeEnum eventType, boolean withDictionary, Trained trained) {
    this.kind = kind;
    this.run = run;
    this.rules = rules;
    this.cutoff = cutoff;
    this.streamType = streamType;
    this.eventType = eventType;
    this.withDictionary = withDictionary;
    this.trained = trained;
  }

  /** Every run, with the English head rules fixture and the inputs directory the corpus is read from. */
  static List<ParserTrainGolden> all(Path englishRules, Path inputs) throws IOException {
    ModelFile rules = ModelFile.read(englishRules);
    List<ParserTrainGolden> out = new ArrayList<>();
    for (int c : List.of(1, 5)) {
      out.add(new ParserTrainGolden(Kind.DICTIONARY, "dictionary-cutoff-" + c, rules, c, null, null, false, null));
    }
    out.add(events("events-chunking-build", rules, ParserType.CHUNKING, ParserEventTypeEnum.BUILD, false));
    out.add(events("events-chunking-build+dictionary", rules, ParserType.CHUNKING, ParserEventTypeEnum.BUILD, true));
    out.add(events("events-chunking-check", rules, ParserType.CHUNKING, ParserEventTypeEnum.CHECK, false));
    out.add(events("events-treeinsert-build", rules, ParserType.TREEINSERT, ParserEventTypeEnum.BUILD, false));
    out.add(events("events-treeinsert-check", rules, ParserType.TREEINSERT, ParserEventTypeEnum.CHECK, false));
    out.add(events("events-treeinsert-attach", rules, ParserType.TREEINSERT, ParserEventTypeEnum.ATTACH, false));
    out.add(events("events-chunk", rules, ParserType.CHUNKING, ParserEventTypeEnum.CHUNK, false));
    out.add(events("events-tag", rules, ParserType.CHUNKING, ParserEventTypeEnum.TAG, false));
    out.add(new ParserTrainGolden(Kind.GENERATORS, "generators", rules, 0, null, null, false, null));
    Path corpus = inputs.resolve(INPUT_SET).resolve(CORPUS + ".txt");
    for (ParserType type : List.of(ParserType.CHUNKING, ParserType.TREEINSERT)) {
      out.add(new ParserTrainGolden(Kind.MODEL, "model-" + type.name().toLowerCase(java.util.Locale.ROOT), rules, 0,
          type, null, false, new Trained(type, rules, corpus)));
    }
    return out;
  }

  private static ParserTrainGolden events(String run, ModelFile rules, ParserType type, ParserEventTypeEnum etype,
      boolean withDictionary) {
    return new ParserTrainGolden(Kind.EVENTS, run, rules, 0, type, etype, withDictionary, null);
  }

  @Override
  public String annotator() {
    return ANNOTATOR;
  }

  @Override
  public List<ModelRef> models() {
    return List.of(rules.ref());
  }

  @Override
  public String modelDir() {
    return run;
  }

  @Override
  public String annotatorClass() {
    return switch (kind) {
      case DICTIONARY -> AbstractBottomUpParser.class.getName();
      case EVENTS -> streamType == ParserType.CHUNKING
          ? opennlp.tools.parser.chunking.ParserEventStream.class.getName()
          : opennlp.tools.parser.treeinsert.ParserEventStream.class.getName();
      case GENERATORS -> "opennlp.tools.parser.AbstractContextGenerator";
      case MODEL -> streamType == ParserType.CHUNKING
          ? opennlp.tools.parser.chunking.Parser.class.getName()
          : opennlp.tools.parser.treeinsert.Parser.class.getName();
    };
  }

  @Override
  public String method() {
    return switch (kind) {
      case DICTIONARY -> "buildDictionary";
      case EVENTS -> "read";
      case GENERATORS -> "getContext";
      case MODEL -> "train";
    };
  }

  @Override
  public Map<String, String> config() {
    Map<String, String> c = new TreeMap<>();
    c.put("rules", opennlp.tools.parser.lang.en.HeadRules.class.getName());
    switch (kind) {
      case DICTIONARY -> c.put("cutoff", String.valueOf(cutoff));
      case EVENTS -> {
        c.put("etype", eventType.name());
        if (withDictionary) {
          c.put("dictionary-cutoff", "1");
        }
      }
      case GENERATORS -> {
        c.put("dictionary-cutoff", "1");
        c.put("check-types", String.join(",", CHECK_TYPES));
      }
      case MODEL -> {
        c.put("corpus", INPUT_SET + "/" + CORPUS);
        c.put("language", LANGUAGE);
        for (String ns : MODEL_NAMESPACES) {
          c.put(ns + ".Algorithm", "MAXENT");
          c.put(ns + ".Iterations", String.valueOf(ITERATIONS));
          c.put(ns + ".Cutoff", String.valueOf(CUTOFF));
        }
        c.put("dict.Cutoff", String.valueOf(CUTOFF));
        List<String> searches = new ArrayList<>();
        for (Search s : SEARCHES) {
          searches.add(s.beam() + "/" + s.advance());
        }
        c.put("searches", String.join(",", searches));
        c.put("parse-counts", PARSE_COUNTS.stream().map(String::valueOf).reduce((a, b) -> a + "," + b).orElse(""));
      }
    }
    return c;
  }

  @Override
  public Set<String> inputSets() {
    return Set.of(INPUT_SET);
  }

  /**
   * The training parameters of both parsers: {@code MAXENT}, {@link #ITERATIONS} and
   * {@link #CUTOFF} for each of {@code build}, {@code check}, {@code attach}, {@code tagger} and
   * {@code chunker}, and {@link #CUTOFF} for {@code dict}. A new object each time, since
   * {@code chunking.Parser.train} adds the tagger's beam size to the one it is given.
   */
  static TrainingParameters trainingParameters() {
    TrainingParameters p = new TrainingParameters();
    for (String ns : MODEL_NAMESPACES) {
      p.put(ns, "Algorithm", "MAXENT");
      p.put(ns, "Iterations", ITERATIONS);
      p.put(ns, "Cutoff", CUTOFF);
    }
    p.put("dict", "Cutoff", CUTOFF);
    return p;
  }

  private HeadRules headRules() {
    try {
      return new opennlp.tools.parser.lang.en.HeadRules(
          new StringReader(new String(rules.bytes(), StandardCharsets.UTF_8)));
    } catch (IOException e) {
      throw new IllegalStateException("the English head rules fixture could not be read", e);
    }
  }

  /** The lines of the input that are not blank, in order: the samples. */
  private static List<String> sampleLines(Input input) {
    List<String> lines = new ArrayList<>();
    try (ObjectStream<String> in = new PlainTextByLineStream(() -> new ByteArrayInputStream(input.bytes()),
        StandardCharsets.UTF_8)) {
      for (String line = in.read(); line != null; line = in.read()) {
        if (!line.trim().isEmpty()) {
          lines.add(line);
        }
      }
    } catch (IOException e) {
      throw new IllegalStateException(input.inputId() + ": the line stream failed", e);
    }
    return lines;
  }

  /** A stream of the samples that read without an exception, each read afresh. */
  private static ObjectStream<Parse> parses(List<String> lines) {
    List<Parse> out = new ArrayList<>();
    for (String line : lines) {
      try {
        out.add(Parse.parseParse(line));
      } catch (RuntimeException e) {
        // The events and generator goldens record this sample's exception.
      }
    }
    return ObjectStreamUtils.createObjectStream(out);
  }

  @Override
  public List<Layer> layers(Input input) {
    List<String> lines = sampleLines(input);
    return switch (kind) {
      case DICTIONARY -> dictionaryLayers(lines, cutoff);
      case EVENTS -> eventLayers(input, lines);
      case GENERATORS -> generatorLayers(input, lines);
      case MODEL -> modelLayers(input, lines);
    };
  }

  // The dictionary.

  private Dictionary dictionary(List<String> lines, int c) {
    try (ObjectStream<Parse> samples = parses(lines)) {
      return AbstractBottomUpParser.buildDictionary(samples, headRules(), c);
    } catch (IOException e) {
      throw new IllegalStateException("buildDictionary failed", e);
    }
  }

  private List<Layer> dictionaryLayers(List<String> lines, int c) {
    List<List<String>> ngrams = new ArrayList<>();
    for (StringList entry : dictionary(lines, c)) {
      List<String> tokens = new ArrayList<>();
      for (String t : entry) {
        tokens.add(t);
      }
      ngrams.add(tokens);
    }
    ngrams.sort(ParserTrainGolden::compareTokens);
    Rows rows = new Rows();
    for (int n = 0; n < ngrams.size(); n++) {
      for (String t : ngrams.get(n)) {
        rows.row().u32("ngram", n).str("token", t);
      }
    }
    return List.of(rows.layer("turbo:dictionary", "ngram", "token"));
  }

  private static int compareTokens(List<String> a, List<String> b) {
    for (int i = 0; i < Math.min(a.size(), b.size()); i++) {
      int c = a.get(i).compareTo(b.get(i));
      if (c != 0) {
        return c;
      }
    }
    return Integer.compare(a.size(), b.size());
  }

  // The event streams.

  private List<Layer> eventLayers(Input input, List<String> lines) {
    HeadRules headRules = headRules();
    Dictionary dict = withDictionary ? dictionary(lines, 1) : null;
    Rows events = new Rows();
    Rows errors = new Rows();
    for (int s = 0; s < lines.size(); s++) {
      List<Event> sampleEvents = new ArrayList<>();
      try {
        Parse p = Parse.parseParse(lines.get(s));
        ObjectStream<Parse> one = ObjectStreamUtils.createObjectStream(p);
        try (ObjectStream<Event> stream = streamType == ParserType.CHUNKING
            ? new opennlp.tools.parser.chunking.ParserEventStream(one, headRules, eventType, dict)
            : new opennlp.tools.parser.treeinsert.ParserEventStream(one, headRules, eventType, dict)) {
          for (Event e = stream.read(); e != null; e = stream.read()) {
            sampleEvents.add(e);
          }
        }
      } catch (IOException | RuntimeException e) {
        errors.row().u32("sample", s).str("class", e.getClass().getName()).str("message", FormatGolden.message(e));
        continue;
      }
      for (Event e : sampleEvents) {
        events.row().u32("sample", s).str("outcome", e.getOutcome()).u32("size", e.getContext().length)
            .str("context", joined(input, e.getContext()));
      }
    }
    return List.of(events.layer("turbo:events", "sample", "outcome", "size", "context"),
        errors.layer("turbo:sample-errors", "sample", "class", "message"));
  }

  /** The context strings joined by single spaces; one holding a space stops the generator. */
  private static String joined(Input input, String[] context) {
    for (String c : context) {
      if (c.indexOf(' ') >= 0) {
        throw new IllegalStateException(input.inputId() + ": the context string " + c + " holds a space");
      }
    }
    return String.join(" ", context);
  }

  // The context generators called directly.

  private static final List<String> STATES = List.of("unlabeled", "complete");

  /** The chunk array of a sample, as the event streams start from it. */
  private static Parse[] chunkArray(Parse p, HeadRules rules) {
    Parse.pruneParse(p);
    p.updateHeads(rules);
    return AbstractBottomUpParser.collapsePunctuation(AbstractParserEventStream.getInitialChunks(p),
        rules.getPunctuationTags());
  }

  private List<Layer> generatorLayers(Input input, List<String> lines) {
    HeadRules headRules = headRules();
    Set<String> punct = headRules.getPunctuationTags();
    Dictionary dict = dictionary(lines, 1);
    Rows texts = new Rows();
    Rows chunks = new Rows();
    Rows chunker = new Rows();
    Rows chunkingBuild = new Rows();
    Rows chunkingBuildDict = new Rows();
    Rows chunkingCheck = new Rows();
    Rows treeBuild = new Rows();
    Rows treeAttach = new Rows();
    Rows treeCheck = new Rows();
    Rows errors = new Rows();
    Rows readErrors = new Rows();
    for (int s = 0; s < lines.size(); s++) {
      Parse[] array;
      Parse[] initial;
      try {
        Parse p = Parse.parseParse(lines.get(s));
        array = chunkArray(p, headRules);
        initial = AbstractParserEventStream.getInitialChunks(p);
      } catch (RuntimeException e) {
        readErrors.row().u32("sample", s).str("class", e.getClass().getName()).str("message", FormatGolden.message(e));
        continue;
      }
      String text = array.length == 0 ? "" : array[0].getText();
      texts.row().u32("sample", s).str("text", text);
      Utf8Offsets offsets = new Utf8Offsets(input.inputId() + " sample " + s, text);
      for (int i = 0; i < array.length; i++) {
        Utf8Offsets.ByteSpan span = offsets.span(array[i].getSpan().getStart(), array[i].getSpan().getEnd());
        Span h = array[i].getHead().getSpan();
        Utf8Offsets.ByteSpan head = offsets.span(h.getStart(), h.getEnd());
        chunks.row().u32("sample", s).u32("index", i).str("type", array[i].getType()).u32("start", span.start())
            .u32("end", span.end()).u32("head-start", head.start()).u32("head-end", head.end())
            .u32("head-index", array[i].getHeadIndex());
      }

      // The chunker's words, tags and outcomes, as the CHUNK events make them.
      List<String> words = new ArrayList<>();
      List<String> tags = new ArrayList<>();
      List<String> preds = new ArrayList<>();
      for (Parse c : initial) {
        if (c.isPosTag()) {
          words.add(c.getCoveredText());
          tags.add(c.getType());
          preds.add(AbstractBottomUpParser.OTHER);
        } else {
          boolean start = true;
          for (Parse tok : c.getChildren()) {
            words.add(tok.getCoveredText());
            tags.add(tok.getType());
            preds.add((start ? AbstractBottomUpParser.START : AbstractBottomUpParser.CONT) + c.getType());
            start = false;
          }
        }
      }
      String[] w = words.toArray(new String[0]);
      String[] t = tags.toArray(new String[0]);
      String[] pr = preds.toArray(new String[0]);
      ChunkContextGenerator ccg = new ChunkContextGenerator();
      for (int i = 0; i < w.length; i++) {
        int at = i;
        call(input, errors, "turbo:chunker-contexts", s, "unlabeled", i,
            () -> ccg.getContext(at, w, t, pr), chunker.row().u32("sample", s).str("state", "unlabeled").u32("index", i));
      }

      for (String state : STATES) {
        if (state.equals("complete")) {
          for (Parse c : array) {
            c.setLabel(AbstractBottomUpParser.COMPLETE);
          }
        }
        opennlp.tools.parser.chunking.BuildContextGenerator cb = new opennlp.tools.parser.chunking.BuildContextGenerator();
        opennlp.tools.parser.chunking.BuildContextGenerator cbd =
            new opennlp.tools.parser.chunking.BuildContextGenerator(dict);
        opennlp.tools.parser.chunking.CheckContextGenerator ck = new opennlp.tools.parser.chunking.CheckContextGenerator();
        opennlp.tools.parser.treeinsert.BuildContextGenerator tb = new opennlp.tools.parser.treeinsert.BuildContextGenerator();
        opennlp.tools.parser.treeinsert.AttachContextGenerator ta =
            new opennlp.tools.parser.treeinsert.AttachContextGenerator(punct);
        opennlp.tools.parser.treeinsert.CheckContextGenerator tk =
            new opennlp.tools.parser.treeinsert.CheckContextGenerator(punct);
        for (int i = 0; i < array.length; i++) {
          int at = i;
          call(input, errors, "turbo:chunking-build", s, state, i, () -> cb.getContext(array, at),
              chunkingBuild.row().u32("sample", s).str("state", state).u32("index", i));
          call(input, errors, "turbo:chunking-build-dictionary", s, state, i, () -> cbd.getContext(array, at),
              chunkingBuildDict.row().u32("sample", s).str("state", state).u32("index", i));
          for (int end = i; end < array.length && end <= i + 2; end++) {
            int e = end;
            for (String type : CHECK_TYPES) {
              call(input, errors, "turbo:chunking-check", s, state, i, () -> ck.getContext(array, type, at, e),
                  chunkingCheck.row().u32("sample", s).str("state", state).str("type", type).u32("start", i)
                      .u32("end", e));
            }
          }
          call(input, errors, "turbo:treeinsert-build", s, state, i, () -> tb.getContext(array, at),
              treeBuild.row().u32("sample", s).str("state", state).u32("index", i));
          if (i > 0) {
            List<Parse> frontier = new ArrayList<>();
            for (int f = i - 1; f >= 0; f--) {
              frontier.add(array[f]);
            }
            for (int f = 0; f < frontier.size(); f++) {
              int rfi = f;
              call(input, errors, "turbo:treeinsert-attach", s, state, i,
                  () -> ta.getContext(array, at, frontier, rfi),
                  treeAttach.row().u32("sample", s).str("state", state).u32("index", i).u32("frontier-index", f));
            }
          }
          if (!array[i].isPosTag()) {
            for (boolean trim : new boolean[] {false, true}) {
              call(input, errors, "turbo:treeinsert-check", s, state, i, () -> tk.getContext(array[at], array, at, trim),
                  treeCheck.row().u32("sample", s).str("state", state).u32("index", i).u32("trim", trim ? 1 : 0));
            }
          }
        }
      }
    }
    return List.of(
        texts.layer("turbo:parse-texts", "sample", "text"),
        readErrors.layer("turbo:sample-errors", "sample", "class", "message"),
        chunks.layer("turbo:chunks", "sample", "index", "type", "start", "end", "head-start", "head-end", "head-index"),
        chunker.layer("turbo:chunker-contexts", "sample", "state", "index", "size", "context"),
        chunkingBuild.layer("turbo:chunking-build", "sample", "state", "index", "size", "context"),
        chunkingBuildDict.layer("turbo:chunking-build-dictionary", "sample", "state", "index", "size", "context"),
        chunkingCheck.layer("turbo:chunking-check", "sample", "state", "type", "start", "end", "size", "context"),
        treeBuild.layer("turbo:treeinsert-build", "sample", "state", "index", "size", "context"),
        treeAttach.layer("turbo:treeinsert-attach", "sample", "state", "index", "frontier-index", "size", "context"),
        treeCheck.layer("turbo:treeinsert-check", "sample", "state", "index", "trim", "size", "context"),
        errors.layer("turbo:generator-errors", "layer", "sample", "state", "index", "class", "message"));
  }

  /** A context generator call. */
  private interface Call {
    String[] context();
  }

  /**
   * Makes the call and completes the row begun for it with the context; a call that throws
   * drops the row and is recorded in {@code errors}.
   */
  private static void call(Input input, Rows errors, String layer, int sample, String state, int index, Call call,
      Rows.Row row) {
    String[] context;
    try {
      context = call.context();
    } catch (RuntimeException e) {
      row.drop();
      errors.row().str("layer", layer).u32("sample", sample).str("state", state).u32("index", index)
          .str("class", e.getClass().getName()).str("message", FormatGolden.message(e));
      return;
    }
    row.u32("size", context.length).str("context", joined(input, context));
  }

  // The trained models.

  /** A parser model, trained on first use. */
  static final class Trained {

    private final ParserType type;
    private final ModelFile rules;
    private final Path corpus;
    private byte[] file;
    private ParserModel model;

    Trained(ParserType type, ModelFile rules, Path corpus) {
      this.type = type;
      this.rules = rules;
      this.corpus = corpus;
    }

    synchronized void train() {
      if (file != null) {
        return;
      }
      Input input;
      try {
        input = Input.read(corpus, INPUT_SET, CORPUS);
      } catch (IOException e) {
        throw new IllegalStateException(corpus + ": the treebank could not be read", e);
      }
      try (ObjectStream<Parse> samples = new ParseSampleStream(new PlainTextByLineStream(
          () -> new ByteArrayInputStream(input.bytes()), StandardCharsets.UTF_8))) {
        HeadRules headRules = new opennlp.tools.parser.lang.en.HeadRules(
            new StringReader(new String(rules.bytes(), StandardCharsets.UTF_8)));
        ParserModel trained = type == ParserType.CHUNKING
            ? opennlp.tools.parser.chunking.Parser.train(LANGUAGE, samples, headRules, trainingParameters())
            : opennlp.tools.parser.treeinsert.Parser.train(LANGUAGE, samples, headRules, trainingParameters());
        ByteArrayOutputStream out = new ByteArrayOutputStream();
        trained.serialize(out);
        file = normalized(out.toByteArray());
        model = new ParserModel(new ByteArrayInputStream(file));
      } catch (IOException e) {
        throw new IllegalStateException(input.inputId() + ": training failed", e);
      }
    }
  }

  /**
   * The zip package written again: the same entries in the same order, a nested package
   * normalized the same way, {@code Timestamp} left out of {@code manifest.properties} (written
   * as {@code Properties.store} writes it, without its date comment), and every entry given
   * {@link #ENTRY_TIME}.
   */
  static byte[] normalized(byte[] zip) throws IOException {
    ByteArrayOutputStream bytes = new ByteArrayOutputStream();
    try (ZipInputStream in = new ZipInputStream(new ByteArrayInputStream(zip));
        ZipOutputStream out = new ZipOutputStream(bytes)) {
      ZipEntry z;
      while ((z = in.getNextEntry()) != null) {
        byte[] content = in.readAllBytes();
        if (z.getName().equals("manifest.properties")) {
          content = manifest(content);
        } else if (isZip(content)) {
          content = normalized(content);
        }
        ZipEntry e = new ZipEntry(z.getName());
        e.setTimeLocal(ENTRY_TIME);
        out.putNextEntry(e);
        out.write(content);
        out.closeEntry();
      }
    }
    return bytes.toByteArray();
  }

  private static boolean isZip(byte[] b) {
    return b.length >= 4 && b[0] == 'P' && b[1] == 'K' && b[2] == 3 && b[3] == 4;
  }

  private static byte[] manifest(byte[] content) throws IOException {
    Properties p = new Properties();
    p.load(new ByteArrayInputStream(content));
    p.remove("Timestamp");
    StringWriter w = new StringWriter();
    p.store(w, null);
    String s = w.toString();
    if (!s.startsWith("#")) {
      throw new IllegalStateException("Properties.store wrote no date comment");
    }
    s = s.substring(s.indexOf('\n') + 1);
    return s.getBytes(StandardCharsets.ISO_8859_1);
  }

  private List<Layer> modelLayers(Input input, List<String> lines) {
    trained.train();
    List<Layer> layers = new ArrayList<>();
    if (input.id().equals(CORPUS)) {
      layers.addAll(fileLayers(trained.file));
    }
    Rows texts = new Rows();
    Rows parses = new Rows();
    Rows nodes = new Rows();
    Rows errors = new Rows();
    List<Parse> references = new ArrayList<>();
    List<String> parseLines = new ArrayList<>();
    for (int s = 0; s < lines.size(); s++) {
      Parse reference;
      try {
        reference = Parse.parseParse(lines.get(s));
      } catch (RuntimeException e) {
        errors.row().u32("sample", s).u32("beam", 0).f64("advance", 0).u32("k", 0)
            .str("class", e.getClass().getName()).str("message", FormatGolden.message(e));
        parseLines.add(null);
        continue;
      }
      references.add(reference);
      List<String> tokens = new ArrayList<>();
      for (Parse token : reference.getTokenNodes()) {
        tokens.add(token.getSpan().getCoveredText(reference.getText()).toString());
      }
      String line = String.join(" ", tokens);
      parseLines.add(line);
      texts.row().u32("sample", s).str("text", line);
    }
    for (Search search : SEARCHES) {
      Parser parser = ParserFactory.create(trained.model, search.beam(), search.advance());
      for (int k : PARSE_COUNTS) {
        for (int s = 0; s < parseLines.size(); s++) {
          String line = parseLines.get(s);
          if (line == null) {
            continue;
          }
          Parse[] result;
          try {
            result = ParserTool.parseLine(line, parser, k);
          } catch (RuntimeException e) {
            errors.row().u32("sample", s).u32("beam", search.beam()).f64("advance", search.advance()).u32("k", k)
                .str("class", e.getClass().getName()).str("message", FormatGolden.message(e));
            continue;
          }
          for (int r = 0; r < result.length; r++) {
            Parse p = result[r];
            StringBuffer shown = new StringBuffer();
            p.show(shown);
            parses.row().u32("sample", s).u32("beam", search.beam()).f64("advance", search.advance()).u32("k", k)
                .u32("rank", r).f64("prob", p.getProb()).str("show", shown.toString());
            Utf8Offsets offsets = new Utf8Offsets(input.inputId() + " sample " + s, p.getText());
            nodes(p, 0, offsets, nodes, s, search, k, r);
          }
        }
      }
    }
    Rows evaluation = new Rows();
    List<Search> evaluated = new ArrayList<>(SEARCHES);
    evaluated.add(new Search(AbstractBottomUpParser.defaultBeamSize, AbstractBottomUpParser.defaultAdvancePercentage));
    for (Search search : evaluated) {
      ParserEvaluator evaluator = new ParserEvaluator(
          ParserFactory.create(trained.model, search.beam(), search.advance()));
      try {
        evaluator.evaluate(ObjectStreamUtils.createObjectStream(freshCopies(lines)));
      } catch (IOException | RuntimeException e) {
        errors.row().u32("sample", lines.size()).u32("beam", search.beam()).f64("advance", search.advance())
            .u32("k", 0).str("class", e.getClass().getName()).str("message", FormatGolden.message(e));
        continue;
      }
      FMeasure f = evaluator.getFMeasure();
      evaluation.row().u32("beam", search.beam()).f64("advance", search.advance())
          .f64("precision", f.getPrecisionScore()).f64("recall", f.getRecallScore()).f64("f-measure", f.getFMeasure());
    }
    layers.add(texts.layer("turbo:parse-texts", "sample", "text"));
    layers.add(parses.layer("turbo:parses", "sample", "beam", "advance", "k", "rank", "prob", "show"));
    layers.add(nodes.layer("turbo:parse-nodes", "sample", "beam", "advance", "k", "rank", "depth", "type", "start",
        "end", "head-start", "head-end", "head-index"));
    layers.add(errors.layer("turbo:parse-errors", "sample", "beam", "advance", "k", "class", "message"));
    layers.add(evaluation.layer("turbo:evaluation", "beam", "advance", "precision", "recall", "f-measure"));
    return layers;
  }

  /** The samples that read, each read again, so that the evaluator gets parses nothing has changed. */
  private static List<Parse> freshCopies(List<String> lines) {
    List<Parse> out = new ArrayList<>();
    for (String line : lines) {
      try {
        out.add(Parse.parseParse(line));
      } catch (RuntimeException e) {
        // Recorded with the parses.
      }
    }
    return out;
  }

  private static void nodes(Parse node, int depth, Utf8Offsets offsets, Rows out, int sample, Search search, int k,
      int rank) {
    Span span = node.getSpan();
    Span head = node.getHead().getSpan();
    Utf8Offsets.ByteSpan bytes = offsets.span(span.getStart(), span.getEnd());
    Utf8Offsets.ByteSpan headBytes = offsets.span(head.getStart(), head.getEnd());
    out.row().u32("sample", sample).u32("beam", search.beam()).f64("advance", search.advance()).u32("k", k)
        .u32("rank", rank).u32("depth", depth).str("type", node.getType()).u32("start", bytes.start())
        .u32("end", bytes.end()).u32("head-start", headBytes.start()).u32("head-end", headBytes.end())
        .u32("head-index", node.getHeadIndex());
    for (Parse child : node.getChildren()) {
      nodes(child, depth + 1, offsets, out, sample, search, k, rank);
    }
  }

  /** The normalized model file: its bytes, its entries and its manifests. */
  private static List<Layer> fileLayers(byte[] file) {
    Rows hex = new Rows();
    for (int i = 0; i < file.length; i += HEX_ROW_BYTES) {
      StringBuilder b = new StringBuilder();
      for (int j = i; j < Math.min(file.length, i + HEX_ROW_BYTES); j++) {
        b.append(String.format(java.util.Locale.ROOT, "%02x", file[j] & 0xff));
      }
      hex.row().str("hex", b.toString());
    }
    Rows entries = new Rows();
    Rows manifest = new Rows();
    try {
      entries(file, "", entries, manifest);
    } catch (IOException e) {
      throw new IllegalStateException("the normalized model file could not be read", e);
    }
    return List.of(
        new Layer("turbo:model-file", "document", new Column.Str("sha256", new String[] {Sha.sha256(file)}),
            new Column.U32("length", new long[] {file.length})),
        hex.layer("turbo:model-hex", "hex"),
        entries.layer("turbo:model-entries", "path", "length", "sha256"),
        manifest.layer("turbo:manifest", "package", "key", "value"));
  }

  private static void entries(byte[] zip, String prefix, Rows entries, Rows manifest) throws IOException {
    try (ZipInputStream in = new ZipInputStream(new ByteArrayInputStream(zip))) {
      ZipEntry z;
      while ((z = in.getNextEntry()) != null) {
        byte[] content = in.readAllBytes();
        String path = prefix + z.getName();
        entries.row().str("path", path).u32("length", content.length).str("sha256", Sha.sha256(content));
        if (z.getName().equals("manifest.properties")) {
          Properties p = new Properties();
          p.load(new ByteArrayInputStream(content));
          for (String key : new TreeMap<>(Collections.unmodifiableMap(toMap(p))).keySet()) {
            manifest.row().str("package", prefix.isEmpty() ? "" : prefix.substring(0, prefix.length() - 1))
                .str("key", key).str("value", p.getProperty(key));
          }
        } else if (isZip(content)) {
          entries(content, path + "/", entries, manifest);
        }
      }
    }
  }

  private static Map<String, String> toMap(Properties p) {
    Map<String, String> m = new HashMap<>();
    for (String k : p.stringPropertyNames()) {
      m.put(k, p.getProperty(k));
    }
    return m;
  }

  /**
   * The rows of a layer anchored to the document, built one row at a time: each row sets every
   * column once, as {@code u32}, {@code f64} or {@code str}, and a row may be dropped before the
   * next is begun.
   */
  static final class Rows {
    private final List<Row> rows = new ArrayList<>();

    Row row() {
      Row r = new Row();
      rows.add(r);
      return r;
    }

    final class Row {
      private final Map<String, Object> cells = new LinkedHashMap<>();

      Row u32(String name, long v) {
        cells.put(name, v);
        return this;
      }

      Row f64(String name, double v) {
        cells.put(name, v);
        return this;
      }

      Row str(String name, String v) {
        cells.put(name, v == null ? "" : v);
        return this;
      }

      void drop() {
        rows.remove(this);
      }
    }

    /**
     * The layer with the named columns in order; a column's type is that of its values, and a
     * column of a layer without rows is {@code str} unless its name is in {@link #U32} or
     * {@link #F64}.
     */
    Layer layer(String id, String... names) {
      List<Column> columns = new ArrayList<>();
      for (String n : names) {
        Object first = rows.isEmpty() ? null : rows.get(0).cells.get(n);
        boolean u32 = first instanceof Long || (first == null && U32.contains(n));
        boolean f64 = first instanceof Double || (first == null && F64.contains(n));
        if (u32) {
          long[] v = new long[rows.size()];
          for (int i = 0; i < v.length; i++) {
            v[i] = (Long) cell(rows.get(i), id, n);
          }
          columns.add(new Column.U32(n, v));
        } else if (f64) {
          double[] v = new double[rows.size()];
          for (int i = 0; i < v.length; i++) {
            v[i] = (Double) cell(rows.get(i), id, n);
          }
          columns.add(new Column.F64(n, v));
        } else {
          String[] v = new String[rows.size()];
          for (int i = 0; i < v.length; i++) {
            v[i] = (String) cell(rows.get(i), id, n);
          }
          columns.add(new Column.Str(n, v));
        }
      }
      for (Row r : rows) {
        if (r.cells.size() != names.length) {
          throw new IllegalStateException(id + ": a row has the columns " + r.cells.keySet());
        }
      }
      return new Layer(id, "document", columns);
    }

    private static final Set<String> U32 = Set.of("sample", "index", "start", "end", "head-start", "head-end",
        "head-index", "ngram", "size", "frontier-index", "trim", "beam", "k", "rank", "depth", "length");
    private static final Set<String> F64 = Set.of("advance", "prob", "precision", "recall", "f-measure");

    private static Object cell(Row r, String id, String name) {
      Object v = r.cells.get(name);
      if (v == null) {
        throw new IllegalStateException(id + ": a row has no " + name);
      }
      return v;
    }
  }
}
