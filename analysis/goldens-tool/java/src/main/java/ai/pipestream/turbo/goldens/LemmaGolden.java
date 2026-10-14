package ai.pipestream.turbo.goldens;

import java.io.IOException;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.Set;
import opennlp.tools.lemmatizer.Lemmatizer;
import opennlp.tools.lemmatizer.LemmatizerME;
import opennlp.tools.lemmatizer.LemmatizerModel;
import opennlp.tools.ml.model.MaxentModel;
import opennlp.tools.postag.POSModel;
import opennlp.tools.postag.POSTaggerME;
import opennlp.tools.util.Span;

/**
 * The maxent lemmatizer's golden: the sentences, tokens and part-of-speech tags of the models of
 * the same language ({@link PosGolden}'s layers), then {@code opennlp:lemmas} aligned to the
 * tokens, the lemma {@link LemmatizerME#lemmatize(String[], String[])} gave each token of its
 * sentence, and {@code turbo:confidence<opennlp:lemmas>}, the probability
 * {@link LemmatizerME#probs()} reports for it. The Rust port is run on the JVM's own tokens and
 * tags from the golden, so what is compared is the lemmatization alone.
 *
 * <p>The model records name the lemmatizer model first (its base name is the run's directory),
 * then the sentence, token and part-of-speech models.</p>
 */
final class LemmaGolden implements AnnotatorGolden {

  /** The classifier's entry in a lemmatizer model ({@code LemmatizerModel.LEMMATIZER_MODEL_ENTRY_NAME}). */
  static final String CLASSIFIER_ENTRY = "lemmatizer.model";

  /** The sentence detector, tokenizer and tagger that cut and tag an input for a lemmatizer. */
  record Pipeline(PosGolden.Pipeline cut, ModelRef posModel, POSTaggerME tagger) {

    /** Loads the three models, hashing exactly the bytes loaded; the tagger returns UD tags. */
    static Pipeline load(Path sentenceModel, Path tokenModel, Path posModel) throws IOException {
      ModelFile p = ModelFile.read(posModel);
      return new Pipeline(PosGolden.Pipeline.load(sentenceModel, tokenModel), p.ref(),
          new POSTaggerME(new POSModel(p.stream())));
    }

    /** The model records of a run over this pipeline, after the lemmatizer's own. */
    List<ModelRef> models(ModelRef first) {
      return List.of(first, cut.sentenceModel(), cut.tokenModel(), posModel);
    }
  }

  /**
   * What one run over an input produced: the layers, and for the decision point dump each
   * sentence's token array as the lemmatizer received it (the very objects, so a context request
   * can be told by identity which token it was about) with the UTF-16 start of each token.
   */
  record Run(List<Layer> layers, List<String[]> sentenceTokens, List<int[]> tokenStarts) {}

  private final List<ModelRef> models;
  private final LemmatizerModel loaded;
  private final LemmatizerME lemmatizer;
  private final Pipeline pipeline;
  private final Set<String> inputSets;

  private LemmaGolden(ModelRef model, LemmatizerModel loaded, Pipeline pipeline, Set<String> inputSets) {
    this.models = pipeline.models(model);
    this.loaded = loaded;
    this.lemmatizer = new LemmatizerME(loaded);
    this.pipeline = pipeline;
    this.inputSets = Set.copyOf(inputSets);
  }

  /** Loads the lemmatizer model from {@code file} and runs it as it is. */
  static LemmaGolden load(Path file, Pipeline pipeline, Set<String> inputSets) throws IOException {
    ModelFile f = ModelFile.read(file);
    return new LemmaGolden(f.ref(), new LemmatizerModel(f.stream()), pipeline, inputSets);
  }

  /** The model's classifier. */
  static MaxentModel classifier(LemmatizerModel model) {
    return model.getArtifact(CLASSIFIER_ENTRY);
  }

  /** The beam size the lemmatizer reads from the model's manifest, or the default without one. */
  static int beamSize(LemmatizerModel model) {
    String s = model.getManifestProperty("BeamSize");
    return s == null ? LemmatizerME.DEFAULT_BEAM_SIZE : Integer.parseInt(s);
  }

  /** The loaded model, which {@link LemmaDump} builds its recording model from. */
  LemmatizerModel loaded() {
    return loaded;
  }

  /** The sentence detector, tokenizer and tagger in front of the lemmatizer. */
  Pipeline pipeline() {
    return pipeline;
  }

  @Override
  public String annotator() {
    return "lemmatize";
  }

  @Override
  public List<ModelRef> models() {
    return models;
  }

  @Override
  public String annotatorClass() {
    return LemmatizerME.class.getName();
  }

  @Override
  public String method() {
    return "lemmatize";
  }

  @Override
  public Set<String> inputSets() {
    return inputSets;
  }

  @Override
  public List<Layer> layers(Input input) throws AnnotatorException {
    return run(lemmatizer, pipeline, input, true).layers();
  }

  /**
   * Cuts the input into sentences and tokens, tags each sentence and lemmatizes it with
   * {@code lemmatizer}; {@link LemmaDump} runs it with its recording lemmatizer and
   * {@link DictionaryLemmaGolden} with its dictionary, without a confidence layer. The
   * probabilities {@code probs()} reports must be one per token.
   */
  static Run run(Lemmatizer lemmatizer, Pipeline pipeline, Input input, boolean confidence)
      throws AnnotatorException {
    String text = input.text();
    PosGolden.Pipeline cut = pipeline.cut();
    Span[] sentences = cut.sentences().sentPosDetect(text);
    List<Span> tokenSpans = new ArrayList<>();
    List<String> tags = new ArrayList<>();
    List<Double> tagProbs = new ArrayList<>();
    List<String> lemmas = new ArrayList<>();
    List<Double> probs = new ArrayList<>();
    List<String[]> sentenceTokens = new ArrayList<>();
    List<int[]> tokenStarts = new ArrayList<>();
    for (Span sentence : sentences) {
      String sentenceText = sentence.getCoveredText(text).toString();
      Span[] inSentence = cut.tokenizer().tokenizePos(sentenceText);
      String[] words = new String[inSentence.length];
      int[] starts = new int[inSentence.length];
      for (int i = 0; i < inSentence.length; i++) {
        // A new object per token, so a context request about it is told by identity.
        words[i] = new String(inSentence[i].getCoveredText(sentenceText).toString());
        starts[i] = sentence.getStart() + inSentence[i].getStart();
        tokenSpans.add(new Span(starts[i], sentence.getStart() + inSentence[i].getEnd()));
      }
      String[] tagged = AnnotatorException.call(() -> pipeline.tagger().tag(words));
      double[] tagReported = pipeline.tagger().probs();
      String[] lemmatized = AnnotatorException.call(() -> lemmatizer.lemmatize(words, tagged));
      double[] reported = confidence ? ((LemmatizerME) lemmatizer).probs() : new double[words.length];
      if (tagged.length != words.length || lemmatized.length != words.length || reported.length != words.length) {
        throw new IllegalStateException("input " + input.inputId() + ": " + tagged.length + " tags, "
            + lemmatized.length + " lemmas and " + reported.length + " probabilities for " + words.length + " tokens");
      }
      for (int i = 0; i < words.length; i++) {
        if (tagged[i] == null || lemmatized[i] == null) {
          throw new IllegalStateException("input " + input.inputId() + ": no tag or lemma sequence for a sentence");
        }
        tags.add(tagged[i]);
        tagProbs.add(tagReported[i]);
        lemmas.add(lemmatized[i]);
        probs.add(reported[i]);
      }
      sentenceTokens.add(words);
      tokenStarts.add(starts);
    }
    List<Layer> layers = new ArrayList<>(SpanLayers.of("opennlp:sentences", input.offsets(), sentences));
    layers.addAll(SpanLayers.of("opennlp:tokens", input.offsets(), tokenSpans.toArray(new Span[0])));
    layers.add(new Layer("opennlp:pos", "aligned:opennlp:tokens", new Column.Str("tag", tags.toArray(new String[0])),
        new Column.F64("prob", doubles(tagProbs))));
    layers.add(new Layer("opennlp:lemmas", "aligned:opennlp:tokens",
        new Column.Str("lemma", lemmas.toArray(new String[0]))));
    if (confidence) {
      layers.add(new Layer("turbo:confidence<opennlp:lemmas>", "aligned:opennlp:lemmas",
          new Column.F64("value", doubles(probs))));
    }
    return new Run(layers, sentenceTokens, tokenStarts);
  }

  private static double[] doubles(List<Double> values) {
    double[] out = new double[values.size()];
    for (int i = 0; i < out.length; i++) {
      out[i] = values.get(i);
    }
    return out;
  }

  /** The manifest entries a recording model is rebuilt with, for {@link LemmaDump}. */
  static Map<String, String> entriesFor(LemmatizerModel loaded) {
    return RecordingModels.entriesFor(loaded);
  }
}
