package ai.pipestream.turbo.goldens;

import java.io.ByteArrayInputStream;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;
import opennlp.tools.formats.ad.ADChunkSampleStream;
import opennlp.tools.formats.ad.ADNameSampleStream;
import opennlp.tools.formats.ad.ADPOSSampleStream;
import opennlp.tools.formats.ad.ADSentenceSampleStream;
import opennlp.tools.formats.convert.NameToTokenSampleStream;
import opennlp.tools.sentdetect.lang.Factory;
import opennlp.tools.tokenize.DetokenizationDictionary;
import opennlp.tools.tokenize.Detokenizer;
import opennlp.tools.tokenize.DictionaryDetokenizer;
import opennlp.tools.util.InputStreamFactory;
import opennlp.tools.util.ObjectStream;
import opennlp.tools.util.PlainTextByLineStream;

/**
 * The golden of one of OpenNLP's Arvores Deitadas streams (the Floresta Sinta(c)tica treebank)
 * over a file of the {@code ad} input set: {@code ADNameSampleStream} with hyphenated tokens split
 * and not, {@code ADPOSSampleStream} with its defaults and with multiword expressions expanded and
 * features included, {@code ADChunkSampleStream} whole and from sentence 1 to before sentence 3,
 * {@code ADSentenceSampleStream} without and with titles, and the token samples
 * {@code ADTokenSampleStreamFactory} makes, a {@code NameToTokenSampleStream} over the name stream
 * with the Portuguese detokenizer dictionary. Each stream is made as its factory makes it, over
 * lines of UTF-8 text ({@code PlainTextByLineStream}); the factories take the encoding as a
 * parameter, which is a matter of decoding the file.
 *
 * <p>The stream is read as {@link FormatGolden} reads one: until {@code read} returns null, with an
 * exception recorded and the reading going on. The sentence runs do not read
 * {@value #NULL_TEXT}, whose first sentence has no text: {@code ADSentenceSampleStream} throws
 * {@code NullPointerException} on every read after it and never ends. The layers, anchored to the
 * document, are {@code turbo:samples} and {@code turbo:read-errors}, then those of the sample kind
 * as {@link FormatGolden} writes them: {@code turbo:sample-tokens} with {@code turbo:names},
 * {@code turbo:sample-tags}, or {@code turbo:sample-tags}, {@code turbo:sample-chunks},
 * {@code turbo:phrases} and {@code turbo:phrase-errors}, or {@code turbo:sample-texts} and
 * {@code turbo:sample-spans}.</p>
 */
final class AdGolden implements AnnotatorGolden {

  /** The input set the files are read from. */
  static final String SET = Runs.AD;

  /** The input whose first sentence has no text, which the sentence runs do not read. */
  static final String NULL_TEXT = "ad-null-text";

  /** The first sentence and the sentence before which the ranged chunk run stops. */
  static final int CHUNK_START = 1;
  static final int CHUNK_END = 3;

  /** The stopping point for a stream that throws without moving on, which would otherwise loop. */
  private static final int MAX_READS = 100_000;

  /** The streams and their parameters. */
  enum Kind {
    NAMES("names", ADNameSampleStream.class),
    NAMES_NO_SPLIT("names-nosplit", ADNameSampleStream.class),
    POS("pos", ADPOSSampleStream.class),
    POS_EXPAND_FEATURES("pos-expand-features", ADPOSSampleStream.class),
    CHUNK("chunk", ADChunkSampleStream.class),
    CHUNK_RANGE("chunk-1-3", ADChunkSampleStream.class),
    SENTENCE("sentence", ADSentenceSampleStream.class),
    SENTENCE_TITLES("sentence-titles", ADSentenceSampleStream.class),
    TOKEN("token", NameToTokenSampleStream.class);

    final String dir;
    final Class<?> streamClass;

    Kind(String dir, Class<?> streamClass) {
      this.dir = dir;
      this.streamClass = streamClass;
    }
  }

  private final Kind kind;
  private final List<ModelRef> models;
  private final Detokenizer detokenizer;

  private AdGolden(Kind kind, List<ModelRef> models, Detokenizer detokenizer) {
    this.kind = kind;
    this.models = List.copyOf(models);
    this.detokenizer = detokenizer;
  }

  /** Every run; the token run with the detokenizer dictionary {@code dictionary}. */
  static List<AdGolden> all(Path dictionary) throws IOException {
    ModelFile d = ModelFile.read(dictionary);
    Detokenizer detokenizer = new DictionaryDetokenizer(new DetokenizationDictionary(d.stream()));
    List<AdGolden> runs = new ArrayList<>();
    for (Kind k : Kind.values()) {
      runs.add(k == Kind.TOKEN ? new AdGolden(k, List.of(d.ref()), detokenizer) : new AdGolden(k, List.of(), null));
    }
    return runs;
  }

  @Override
  public String annotator() {
    return "ad";
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
    Map<String, String> c = new LinkedHashMap<>();
    switch (kind) {
      case NAMES, TOKEN -> c.put("splitHyphenatedTokens", "true");
      case NAMES_NO_SPLIT -> c.put("splitHyphenatedTokens", "false");
      case POS -> {
        c.put("expandME", "false");
        c.put("includeFeatures", "false");
      }
      case POS_EXPAND_FEATURES -> {
        c.put("expandME", "true");
        c.put("includeFeatures", "true");
      }
      case CHUNK -> { }
      case CHUNK_RANGE -> {
        c.put("start", Integer.toString(CHUNK_START));
        c.put("end", Integer.toString(CHUNK_END));
      }
      case SENTENCE -> c.put("includeTitles", "false");
      case SENTENCE_TITLES -> c.put("includeTitles", "true");
    }
    return c;
  }

  @Override
  public Set<String> inputSets() {
    return Set.of(SET);
  }

  @Override
  public boolean accepts(Input input) {
    boolean sentences = kind == Kind.SENTENCE || kind == Kind.SENTENCE_TITLES;
    return input.set().equals(SET) && !(sentences && input.id().equals(NULL_TEXT));
  }

  @Override
  public String modelDir() {
    return kind.dir;
  }

  private ObjectStream<?> open(Input input) throws IOException {
    InputStreamFactory in = () -> new ByteArrayInputStream(input.bytes());
    ObjectStream<String> lines = new PlainTextByLineStream(in, StandardCharsets.UTF_8);
    return switch (kind) {
      case NAMES -> new ADNameSampleStream(lines, true);
      case NAMES_NO_SPLIT -> new ADNameSampleStream(lines, false);
      case POS -> new ADPOSSampleStream(lines, false, false);
      case POS_EXPAND_FEATURES -> new ADPOSSampleStream(lines, true, true);
      case CHUNK -> new ADChunkSampleStream(lines);
      case CHUNK_RANGE -> {
        // as the factory sets them
        ADChunkSampleStream s = new ADChunkSampleStream(lines);
        s.setStart(CHUNK_START);
        s.setEnd(CHUNK_END);
        yield s;
      }
      case SENTENCE -> new ADSentenceSampleStream(lines, false);
      case SENTENCE_TITLES -> new ADSentenceSampleStream(lines, true);
      case TOKEN -> new NameToTokenSampleStream(detokenizer, new ADNameSampleStream(lines, true));
    };
  }

  @Override
  public List<Layer> layers(Input input) {
    List<Object> samples = new ArrayList<>();
    FormatGolden.Columns errors = new FormatGolden.Columns();
    // ADSentenceSampleStream sorts Factory.ptEosCharacters in place and keeps
    // searching it while it reads. The Portuguese sentence detector runs of
    // EventGolden read the same array for their manifests, so its order is put
    // back once the stream is done.
    char[] eos = Factory.ptEosCharacters.clone();
    try (ObjectStream<?> stream = open(input)) {
      for (int reads = 0; ; reads++) {
        if (reads > MAX_READS) {
          throw new IllegalStateException(input.inputId() + ": the stream does not end");
        }
        Object sample;
        try {
          sample = stream.read();
        } catch (IOException | RuntimeException e) {
          errors.u32("after", samples.size()).str("class", e.getClass().getName())
              .str("message", FormatGolden.message(e));
          continue;
        }
        if (sample == null) {
          break;
        }
        samples.add(sample);
      }
    } catch (IOException e) {
      throw new IllegalStateException(input.inputId() + ": the line stream failed", e);
    } finally {
      System.arraycopy(eos, 0, Factory.ptEosCharacters, 0, eos.length);
    }

    List<Layer> layers = new ArrayList<>();
    FormatGolden.Columns written = new FormatGolden.Columns();
    for (Object s : samples) {
      written.str("written", s.toString());
    }
    layers.add(written.layer("turbo:samples", "document", "written"));
    layers.add(errors.layer("turbo:read-errors", "document", "after", "class", "message"));
    switch (kind) {
      case NAMES, NAMES_NO_SPLIT -> FormatGolden.names(samples, layers);
      case POS, POS_EXPAND_FEATURES -> FormatGolden.tagged(samples, layers);
      case CHUNK, CHUNK_RANGE -> FormatGolden.chunks(samples, layers);
      case SENTENCE, SENTENCE_TITLES -> FormatGolden.sentenceSamples(input, samples, layers);
      case TOKEN -> FormatGolden.tokenSamples(input, samples, layers);
    }
    return layers;
  }
}
