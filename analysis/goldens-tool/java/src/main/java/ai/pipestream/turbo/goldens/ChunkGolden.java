package ai.pipestream.turbo.goldens;

import java.io.IOException;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.Set;
import opennlp.tools.chunker.ChunkSample;
import opennlp.tools.chunker.ChunkerFactory;
import opennlp.tools.chunker.ChunkerME;
import opennlp.tools.chunker.ChunkerModel;
import opennlp.tools.ml.model.MaxentModel;
import opennlp.tools.postag.POSModel;
import opennlp.tools.postag.POSTagFormat;
import opennlp.tools.postag.POSTaggerME;
import opennlp.tools.util.Span;

/**
 * The chunker's golden: the sentences, tokens and part-of-speech tags of the models in front of
 * it ({@link PosGolden}'s layers, the tags in the format the pipeline asks the tagger for), then
 * {@code opennlp:chunk-tags} aligned to the tokens, the chunk tag
 * {@link ChunkerME#chunk(String[], String[])} gave each token of its sentence with the probability
 * {@link ChunkerME#probs()} reports for it, and {@code opennlp:chunks}, the phrases
 * {@link ChunkSample#phrasesAsSpanList} reads off those tags (what {@link ChunkerME#chunkAsSpans}
 * returns), each as a span of the text from its first token's start to its last token's end, its
 * token range in the input's tokens and its kind. The Rust port is run on the JVM's own tokens
 * and tags from the golden, so what is compared is the chunking alone.
 *
 * <p>The model records name the chunker model first (its base name is the run's directory),
 * then the sentence, token and part-of-speech models. Besides a model file as it is, the chunker
 * runs on a model rebuilt with another beam size through
 * {@link ChunkerModel#ChunkerModel(String, MaxentModel, int, Map, ChunkerFactory)}
 * ({@code +beam<n>}, config {@code beamSize}).</p>
 */
final class ChunkGolden implements AnnotatorGolden {

  /** The {@code config} key of a run with a model rebuilt with another beam size. */
  static final String BEAM_KEY = "beamSize";

  /** The classifier's entry in a chunker model ({@code ChunkerModel.CHUNKER_MODEL_ENTRY_NAME}). */
  static final String CLASSIFIER_ENTRY = "chunker.model";

  /**
   * The sentence detector, tokenizer and tagger that cut and tag an input for a chunker, with the
   * tag format the tagger is asked for.
   */
  record Pipeline(PosGolden.Pipeline cut, ModelRef posModel, POSTaggerME tagger, POSTagFormat format) {

    /** Loads the three models, hashing exactly the bytes loaded; the tagger returns tags in {@code format}. */
    static Pipeline load(Path sentenceModel, Path tokenModel, Path posModel, POSTagFormat format)
        throws IOException {
      ModelFile p = ModelFile.read(posModel);
      return new Pipeline(PosGolden.Pipeline.load(sentenceModel, tokenModel), p.ref(),
          new POSTaggerME(new POSModel(p.stream()), format), format);
    }

    /** The model records of a run over this pipeline, after the chunker's own. */
    List<ModelRef> models(ModelRef first) {
      return List.of(first, cut.sentenceModel(), cut.tokenModel(), posModel);
    }
  }

  /**
   * What one run over an input produced: the layers, and for the decision point dump each
   * sentence's token array as the chunker received it (the very objects, so a context request
   * can be told by identity which token it was about) with the UTF-16 start of each token.
   */
  record Run(List<Layer> layers, List<String[]> sentenceTokens, List<int[]> tokenStarts) {}

  private final List<ModelRef> models;
  private final ChunkerModel loaded;
  private final ChunkerME chunker;
  private final Pipeline pipeline;
  private final String modelDir;
  private final Map<String, String> config;
  private final Set<String> inputSets;

  private ChunkGolden(ModelRef model, ChunkerModel loaded, Pipeline pipeline, String modelDir,
      Map<String, String> config, Set<String> inputSets) {
    this.models = pipeline.models(model);
    this.loaded = loaded;
    this.chunker = new ChunkerME(loaded);
    this.pipeline = pipeline;
    this.modelDir = modelDir;
    this.config = Map.copyOf(config);
    this.inputSets = Set.copyOf(inputSets);
  }

  /** Loads the chunker model from {@code file} and runs it as it is. */
  static ChunkGolden load(Path file, Pipeline pipeline, Set<String> inputSets) throws IOException {
    ModelFile f = ModelFile.read(file);
    return new ChunkGolden(f.ref(), new ChunkerModel(f.stream()), pipeline, f.ref().baseName(), Map.of(), inputSets);
  }

  /**
   * The model of {@code file} rebuilt with a beam of {@code beamSize}, its own classifier and
   * manifest entries and the plain factory. The directory is {@code <model>+beam<n>} and the
   * {@code config} record is the beam size.
   */
  static ChunkGolden withBeamSize(Path file, Pipeline pipeline, int beamSize, Set<String> inputSets)
      throws IOException {
    ModelFile f = ModelFile.read(file);
    ChunkerModel base = new ChunkerModel(f.stream());
    RecordingModels.checkFactory(base.getFactory(), ChunkerFactory.class);
    ChunkerModel rebuilt = new ChunkerModel(base.getLanguage(), classifier(base), beamSize,
        RecordingModels.entriesFor(base), new ChunkerFactory());
    if (beamSize(rebuilt) != beamSize) {
      throw new IllegalStateException(f.ref().fileName() + ": the rebuilt model's BeamSize is " + beamSize(rebuilt));
    }
    return new ChunkGolden(f.ref(), rebuilt, pipeline, f.ref().baseName() + "+beam" + beamSize,
        Map.of(BEAM_KEY, Integer.toString(beamSize)), inputSets);
  }

  /** The model's classifier. */
  static MaxentModel classifier(ChunkerModel model) {
    return model.getArtifact(CLASSIFIER_ENTRY);
  }

  /** The beam size the chunker reads from the model's manifest, or the default without one. */
  static int beamSize(ChunkerModel model) {
    String s = model.getManifestProperty("BeamSize");
    return s == null ? ChunkerME.DEFAULT_BEAM_SIZE : Integer.parseInt(s);
  }

  /** The loaded model, which {@link ChunkDump} builds its recording model from. */
  ChunkerModel loaded() {
    return loaded;
  }

  /** The sentence detector, tokenizer and tagger in front of the chunker. */
  Pipeline pipeline() {
    return pipeline;
  }

  @Override
  public String annotator() {
    return "chunk";
  }

  @Override
  public List<ModelRef> models() {
    return models;
  }

  @Override
  public String modelDir() {
    return modelDir;
  }

  @Override
  public String annotatorClass() {
    return ChunkerME.class.getName();
  }

  @Override
  public String method() {
    return "chunk";
  }

  @Override
  public Map<String, String> config() {
    return config;
  }

  @Override
  public Set<String> inputSets() {
    return inputSets;
  }

  @Override
  public List<Layer> layers(Input input) throws AnnotatorException {
    return run(chunker, pipeline, input).layers();
  }

  /**
   * Cuts the input into sentences and tokens, tags each sentence and chunks it with
   * {@code chunker}; {@link ChunkDump} runs it with its recording chunker. The chunk tags are
   * {@code chunk}'s and the phrases are {@code phrasesAsSpanList} over them, as
   * {@code chunkAsSpans} would compute them, without running the search twice. The probabilities
   * {@code probs()} reports must be one per token.
   */
  static Run run(ChunkerME chunker, Pipeline pipeline, Input input) throws AnnotatorException {
    String text = input.text();
    PosGolden.Pipeline cut = pipeline.cut();
    Span[] sentences = cut.sentences().sentPosDetect(text);
    List<Span> tokenSpans = new ArrayList<>();
    List<String> tags = new ArrayList<>();
    List<Double> tagProbs = new ArrayList<>();
    List<String> chunkTags = new ArrayList<>();
    List<Double> probs = new ArrayList<>();
    List<Span> phraseSpans = new ArrayList<>();
    List<Long> phraseTokenStarts = new ArrayList<>();
    List<Long> phraseTokenEnds = new ArrayList<>();
    List<String> kinds = new ArrayList<>();
    List<String[]> sentenceTokens = new ArrayList<>();
    List<int[]> tokenStarts = new ArrayList<>();
    for (Span sentence : sentences) {
      String sentenceText = sentence.getCoveredText(text).toString();
      Span[] inSentence = cut.tokenizer().tokenizePos(sentenceText);
      String[] words = new String[inSentence.length];
      int[] starts = new int[inSentence.length];
      int[] ends = new int[inSentence.length];
      int base = tokenSpans.size();
      for (int i = 0; i < inSentence.length; i++) {
        // A new object per token, so a context request about it is told by identity.
        words[i] = new String(inSentence[i].getCoveredText(sentenceText).toString());
        starts[i] = sentence.getStart() + inSentence[i].getStart();
        ends[i] = sentence.getStart() + inSentence[i].getEnd();
        tokenSpans.add(new Span(starts[i], ends[i]));
      }
      String[] tagged = AnnotatorException.call(() -> pipeline.tagger().tag(words));
      double[] tagReported = pipeline.tagger().probs();
      String[] chunked = AnnotatorException.call(() -> chunker.chunk(words, tagged));
      double[] reported = chunker.probs();
      if (tagged.length != words.length || chunked.length != words.length || reported.length != words.length) {
        throw new IllegalStateException("input " + input.inputId() + ": " + tagged.length + " tags, "
            + chunked.length + " chunk tags and " + reported.length + " probabilities for " + words.length + " tokens");
      }
      for (int i = 0; i < words.length; i++) {
        if (tagged[i] == null || chunked[i] == null) {
          throw new IllegalStateException("input " + input.inputId() + ": no tag or chunk sequence for a sentence");
        }
        tags.add(tagged[i]);
        tagProbs.add(tagReported[i]);
        chunkTags.add(chunked[i]);
        probs.add(reported[i]);
      }
      Span[] phrases = AnnotatorException.call(() -> ChunkSample.phrasesAsSpanList(words, tagged, chunked));
      for (Span phrase : phrases) {
        if (phrase.getStart() >= phrase.getEnd()) {
          throw new IllegalStateException("input " + input.inputId() + ": a phrase without a token");
        }
        phraseSpans.add(new Span(starts[phrase.getStart()], ends[phrase.getEnd() - 1]));
        phraseTokenStarts.add((long) base + phrase.getStart());
        phraseTokenEnds.add((long) base + phrase.getEnd());
        kinds.add(phrase.getType());
      }
      sentenceTokens.add(words);
      tokenStarts.add(starts);
    }
    List<Layer> layers = new ArrayList<>(SpanLayers.of("opennlp:sentences", input.offsets(), sentences));
    layers.addAll(SpanLayers.of("opennlp:tokens", input.offsets(), tokenSpans.toArray(new Span[0])));
    layers.add(new Layer("opennlp:pos", "aligned:opennlp:tokens", new Column.Str("tag", tags.toArray(new String[0])),
        new Column.F64("prob", doubles(tagProbs))));
    layers.add(new Layer("opennlp:chunk-tags", "aligned:opennlp:tokens",
        new Column.Str("tag", chunkTags.toArray(new String[0])), new Column.F64("prob", doubles(probs))));
    // The phrase spans as a text layer, with the token range and kind columns added to the span
    // columns; when a phrase's token cuts a character, the UTF-16 layer follows as for tokens.
    List<Layer> phraseLayers = SpanLayers.of("opennlp:chunks", input.offsets(), phraseSpans.toArray(new Span[0]));
    Layer spans = phraseLayers.get(0);
    layers.add(new Layer("opennlp:chunks", "text", List.of(spans.columns().get(0), spans.columns().get(1),
        new Column.U32("token_start", longs(phraseTokenStarts)), new Column.U32("token_end", longs(phraseTokenEnds)),
        new Column.Str("kind", kinds.toArray(new String[0])))));
    layers.addAll(phraseLayers.subList(1, phraseLayers.size()));
    return new Run(layers, sentenceTokens, tokenStarts);
  }

  private static double[] doubles(List<Double> values) {
    double[] out = new double[values.size()];
    for (int i = 0; i < out.length; i++) {
      out[i] = values.get(i);
    }
    return out;
  }

  private static long[] longs(List<Long> values) {
    long[] out = new long[values.size()];
    for (int i = 0; i < out.length; i++) {
      out[i] = values.get(i);
    }
    return out;
  }

  /** The manifest entries a recording model is rebuilt with, for {@link ChunkDump}. */
  static Map<String, String> entriesFor(ChunkerModel loaded) {
    return RecordingModels.entriesFor(loaded);
  }
}
