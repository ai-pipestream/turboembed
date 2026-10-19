package ai.pipestream.turbo.goldens;

import java.io.IOException;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Set;
import opennlp.tools.ml.model.MaxentModel;
import opennlp.tools.postag.POSModel;
import opennlp.tools.postag.POSTagFormat;
import opennlp.tools.postag.POSTaggerFactory;
import opennlp.tools.postag.POSTaggerME;
import opennlp.tools.sentdetect.SentenceDetectorME;
import opennlp.tools.sentdetect.SentenceModel;
import opennlp.tools.tokenize.TokenizerME;
import opennlp.tools.tokenize.TokenizerModel;
import opennlp.tools.util.Span;

/**
 * The part-of-speech tagger's golden: the sentences of {@link SentenceDetectorME#sentPosDetect}
 * and the tokens {@link TokenizerME#tokenizePos} finds in each ({@code opennlp:sentences} and
 * {@code opennlp:tokens}, each with its {@code turbo:utf16} layer when a span cuts a character,
 * {@link SpanLayers}), then {@code opennlp:pos} aligned to the tokens: the tag
 * {@link POSTaggerME#tag(String[])} gave each token of its sentence and the probability
 * {@link POSTaggerME#probs()} reports for it. The tagger sees one sentence at a time, as the
 * sentence and token layers cut it, and the Rust port is run on the JVM's own tokens from the
 * golden, so what is compared is the tagging alone.
 *
 * <p>The model records name the part-of-speech model first (its base name is the run's
 * directory), then the sentence and token models. Besides a model file as it is, the tagger runs
 * on a model rebuilt with another beam size through
 * {@link POSModel#POSModel(String, MaxentModel, int, Map, POSTaggerFactory)}, as an application
 * trained with another {@code BeamSize} would have it ({@code +beam<n>}, config
 * {@code beamSize}), and on a model as it is but asked for another {@link POSTagFormat}
 * ({@code +<format>}, config {@code format}). The default format, UD, writes no record.</p>
 */
final class PosGolden implements AnnotatorGolden {

  /** The {@code config} key of a run asked for a tag format other than the default. */
  static final String FORMAT_KEY = "format";
  /** The {@code config} key of a run with a model rebuilt with another beam size. */
  static final String BEAM_KEY = "beamSize";

  /** The classifier's entry in a part-of-speech model ({@code POSModel.POS_MODEL_ENTRY_NAME}). */
  static final String CLASSIFIER_ENTRY = "pos.model";
  /** The descriptor's entry ({@code POSModel.GENERATOR_DESCRIPTOR_ENTRY_NAME}). */
  static final String DESCRIPTOR_ENTRY = "generator.featuregen";

  /** The sentence detector and tokenizer that cut an input into the sentences the tagger tags. */
  record Pipeline(ModelRef sentenceModel, SentenceDetectorME sentences, ModelRef tokenModel, TokenizerME tokenizer) {

    /** Loads both models, hashing exactly the bytes loaded. */
    static Pipeline load(Path sentenceModel, Path tokenModel) throws IOException {
      ModelFile s = ModelFile.read(sentenceModel);
      ModelFile t = ModelFile.read(tokenModel);
      return new Pipeline(s.ref(), new SentenceDetectorME(new SentenceModel(s.stream())), t.ref(),
          new TokenizerME(new TokenizerModel(t.stream())));
    }
  }

  /**
   * What one run over an input produced: the layers, and for the decision point dump each
   * sentence's token array as the tagger received it (the very objects, so a context request can
   * be told by identity which token it was about) with the UTF-16 start of each token in the
   * input.
   */
  record Run(List<Layer> layers, List<String[]> sentenceTokens, List<int[]> tokenStarts) {}

  private final List<ModelRef> models;
  private final POSModel loaded;
  private final POSTaggerME tagger;
  private final Pipeline pipeline;
  private final POSTagFormat format;
  private final String modelDir;
  private final Map<String, String> config;
  private final Set<String> inputSets;

  private PosGolden(ModelRef model, POSModel loaded, Pipeline pipeline, POSTagFormat format, String modelDir,
      Map<String, String> config, Set<String> inputSets) {
    this.models = List.of(model, pipeline.sentenceModel(), pipeline.tokenModel());
    this.loaded = loaded;
    this.tagger = new POSTaggerME(loaded, format);
    this.pipeline = pipeline;
    this.format = format;
    this.modelDir = modelDir;
    this.config = Map.copyOf(config);
    this.inputSets = Set.copyOf(inputSets);
  }

  /** Loads the part-of-speech model from {@code file} and runs it as it is, in the default format. */
  static PosGolden load(Path file, Pipeline pipeline, Set<String> inputSets) throws IOException {
    ModelFile f = ModelFile.read(file);
    return new PosGolden(f.ref(), new POSModel(f.stream()), pipeline, POSTagFormat.UD, f.ref().baseName(), Map.of(),
        inputSets);
  }

  /**
   * The model of {@code file} as it is, asked for {@code format}: the directory is
   * {@code <model>+<format>} in lowercase and the {@code config} record names the format.
   */
  static PosGolden withFormat(Path file, Pipeline pipeline, POSTagFormat format, Set<String> inputSets)
      throws IOException {
    if (format == POSTagFormat.UD) {
      throw new IllegalArgumentException("UD is the default format; load the model as it is");
    }
    ModelFile f = ModelFile.read(file);
    String name = format.name().toLowerCase(Locale.ROOT);
    return new PosGolden(f.ref(), new POSModel(f.stream()), pipeline, format, f.ref().baseName() + "+" + name,
        Map.of(FORMAT_KEY, format.name()), inputSets);
  }

  /**
   * The model of {@code file} rebuilt with a beam of {@code beamSize} and its own classifier,
   * descriptor, tag dictionary and manifest entries. The directory is {@code <model>+beam<n>} and
   * the {@code config} record is the beam size.
   */
  static PosGolden withBeamSize(Path file, Pipeline pipeline, int beamSize, Set<String> inputSets)
      throws IOException {
    ModelFile f = ModelFile.read(file);
    POSModel base = new POSModel(f.stream());
    POSModel rebuilt = new POSModel(base.getLanguage(), classifier(base), beamSize, RecordingModels.entriesFor(base),
        new POSTaggerFactory(descriptor(base), new HashMap<>(), base.getFactory().getTagDictionary()));
    if (beamSize(rebuilt) != beamSize) {
      throw new IllegalStateException(f.ref().fileName() + ": the rebuilt model's BeamSize is " + beamSize(rebuilt));
    }
    return new PosGolden(f.ref(), rebuilt, pipeline, POSTagFormat.UD, f.ref().baseName() + "+beam" + beamSize,
        Map.of(BEAM_KEY, Integer.toString(beamSize)), inputSets);
  }

  /** The model's classifier. */
  static MaxentModel classifier(POSModel model) {
    return model.getArtifact(CLASSIFIER_ENTRY);
  }

  /**
   * The model's feature generator descriptor. Every model OpenNLP 1.8 or later trained has one;
   * an older model is refused, since its tagger reads an n-gram dictionary instead, which the
   * rebuilt and recording models here do not carry over.
   */
  static byte[] descriptor(POSModel model) {
    byte[] bytes = model.getArtifact(DESCRIPTOR_ENTRY);
    if (bytes == null) {
      throw new IllegalStateException("the part-of-speech model has no descriptor (trained before OpenNLP 1.8)");
    }
    return bytes;
  }

  /** The beam size the tagger reads from the model's manifest, or the default without one. */
  static int beamSize(POSModel model) {
    String s = model.getManifestProperty("BeamSize");
    return s == null ? POSTaggerME.DEFAULT_BEAM_SIZE : Integer.parseInt(s);
  }

  /** The loaded model, which {@link PosDump} builds its recording model from. */
  POSModel loaded() {
    return loaded;
  }

  /** The format the tagger is asked for. */
  POSTagFormat format() {
    return format;
  }

  /** The sentence detector and tokenizer in front of the tagger. */
  Pipeline pipeline() {
    return pipeline;
  }

  @Override
  public String annotator() {
    return "postag";
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
    return POSTaggerME.class.getName();
  }

  @Override
  public String method() {
    return "tag";
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
    return run(tagger, pipeline, input).layers();
  }

  /**
   * Cuts the input into sentences and tokens and tags each sentence with {@code tagger};
   * {@link PosDump} runs it with its recording tagger. The probabilities {@code probs()} reports
   * must be one per token.
   */
  static Run run(POSTaggerME tagger, Pipeline pipeline, Input input) throws AnnotatorException {
    String text = input.text();
    Span[] sentences = pipeline.sentences().sentPosDetect(text);
    List<Span> tokenSpans = new ArrayList<>();
    List<String> tags = new ArrayList<>();
    List<Double> probs = new ArrayList<>();
    List<String[]> sentenceTokens = new ArrayList<>();
    List<int[]> tokenStarts = new ArrayList<>();
    for (Span sentence : sentences) {
      String sentenceText = sentence.getCoveredText(text).toString();
      Span[] inSentence = pipeline.tokenizer().tokenizePos(sentenceText);
      String[] words = new String[inSentence.length];
      int[] starts = new int[inSentence.length];
      for (int i = 0; i < inSentence.length; i++) {
        // A new object per token, so a context request about it is told by identity.
        words[i] = new String(inSentence[i].getCoveredText(sentenceText).toString());
        starts[i] = sentence.getStart() + inSentence[i].getStart();
        tokenSpans.add(new Span(starts[i], sentence.getStart() + inSentence[i].getEnd()));
      }
      String[] tagged = AnnotatorException.call(() -> tagger.tag(words));
      double[] reported = tagger.probs();
      if (tagged.length != words.length || reported.length != words.length) {
        throw new IllegalStateException("input " + input.inputId() + ": " + tagged.length + " tags and "
            + reported.length + " probabilities for " + words.length + " tokens");
      }
      for (int i = 0; i < words.length; i++) {
        if (tagged[i] == null) {
          throw new IllegalStateException("input " + input.inputId() + ": no tag sequence for a sentence");
        }
        tags.add(tagged[i]);
        probs.add(reported[i]);
      }
      sentenceTokens.add(words);
      tokenStarts.add(starts);
    }
    List<Layer> layers = new ArrayList<>(SpanLayers.of("opennlp:sentences", input.offsets(), sentences));
    layers.addAll(SpanLayers.of("opennlp:tokens", input.offsets(), tokenSpans.toArray(new Span[0])));
    double[] prob = new double[probs.size()];
    for (int i = 0; i < prob.length; i++) {
      prob[i] = probs.get(i);
    }
    layers.add(new Layer("opennlp:pos", "aligned:opennlp:tokens", new Column.Str("tag", tags.toArray(new String[0])),
        new Column.F64("prob", prob)));
    return new Run(layers, sentenceTokens, tokenStarts);
  }
}
