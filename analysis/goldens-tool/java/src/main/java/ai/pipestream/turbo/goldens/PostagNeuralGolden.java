package ai.pipestream.turbo.goldens;

import java.io.ByteArrayInputStream;
import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.TreeMap;
import java.util.function.Function;
import opennlp.tools.postag.POSSample;
import opennlp.tools.postag.WordTagSampleStream;
import opennlp.tools.postag.neural.BilstmPOSModel;
import opennlp.tools.postag.neural.BilstmPOSTagger;
import opennlp.tools.postag.neural.BilstmPOSTrainer;
import opennlp.tools.postag.neural.FeedforwardPOSModel;
import opennlp.tools.postag.neural.FeedforwardPOSTagger;
import opennlp.tools.postag.neural.FeedforwardPOSTrainer;
import opennlp.tools.postag.neural.NeuralAccess;
import opennlp.tools.util.ObjectStream;
import opennlp.tools.util.PlainTextByLineStream;
import opennlp.tools.util.Sequence;

/**
 * The neural part-of-speech taggers of the OpenNLP addons ({@code neural-postag}, built from
 * {@link #NEURAL_POSTAG_COMMIT}): {@link BilstmPOSTagger} over models in each of the
 * {@code ONLP-BLPT} formats and {@link FeedforwardPOSTagger} over {@code ONLP-FFPT-1} and
 * {@code ONLP-FFPT-2} models. The golden directory is {@code postag-neural/<run>}.
 *
 * <p>Each run trains one model with the addons' trainer on the word and tag lines of
 * {@code tagged/wordtag} ({@link WordTagSampleStream}; the trainers skip the sample with no
 * tokens), with one worker thread, serializes it, and tags with the model read back from those
 * bytes, as a user of the file would load it. The runs with vectors give the trainer the word
 * vectors {@link #VECTORS}: two components, the word's length in UTF-16 units times
 * {@code 0.1f}, and {@code 0.5f}.</p>
 *
 * <p>A run writes two goldens: one of {@code tagged/wordtag}, which tags the training sentences
 * and holds the model file, and one of {@code sequence/sentences}, the fixed token list (one
 * sentence per line, tokens separated by single spaces), which tags those. The layers, anchored
 * on the document:</p>
 * <ul>
 *   <li>{@code turbo:sentence-tokens}: each token with its sentence number.</li>
 *   <li>{@code turbo:neural-model} and {@code turbo:neural-model-bytes}, in the training golden
 *   only: the file's {@code magic} (its first {@code writeUTF} string), {@code length} and
 *   {@code sha256}, and its bytes as lowercase hex, 64 bytes a row from {@code offset}.</li>
 *   <li>{@code turbo:neural-tags}: the model's tags in index order.</li>
 *   <li>{@code turbo:neural-emissions}: for each token every tag's unnormalized score:
 *   {@code BilstmPOSModel.score} of the sentence, or for the feedforward tagger the scores of its
 *   left to right decoding, whose tag history is its own earlier tags.</li>
 *   <li>For a CRF model, {@code turbo:crf-viterbi} (the tag index {@code CrfScorer.viterbi} gives
 *   each token) and {@code turbo:crf-marginals} (every tag's posterior probability).</li>
 *   <li>For the feedforward tagger, {@code turbo:neural-context}: each of the twelve slots of
 *   {@code FeedforwardPOSContext.extract} with its {@code symbol} ({@code present} 0 for a
 *   position outside the sentence or no earlier tag, where the symbol is empty) and the
 *   embedding {@code row} it maps to; and for a model with vectors {@code turbo:neural-vector-rows}:
 *   the five rows of the vector window ({@code present} 0 where the word has no vector).</li>
 *   <li>{@code turbo:neural-tagged}: each token's tag and probability from
 *   {@code topKSequences}, whose tags are checked against {@code tag}, and
 *   {@code turbo:neural-sequences}: each sentence's sequence score.</li>
 * </ul>
 */
final class PostagNeuralGolden implements AnnotatorGolden {

  static final String ANNOTATOR = "postag-neural";
  /** The commit of the addons' ready/neural-postag-bilstm branch the image builds the module from. */
  static final String NEURAL_POSTAG_COMMIT = "88767f8d09ffd350bd045e532bdd6f9d76cd9dd8";
  /** The training corpus, {@code <set>/<id>} of the inputs. */
  static final String TRAIN_ID = "wordtag";
  /** The fixed token list. */
  static final String TOKENS_ID = "sentences";
  /** Bytes per row of {@code turbo:neural-model-bytes}. */
  static final int BYTES_PER_ROW = 64;
  /** The word vectors of the runs with vectors. */
  static final Function<CharSequence, float[]> VECTORS = w -> new float[] {w.length() * 0.1f, 0.5f};
  static final String VECTORS_TEXT = "length*0.1f,0.5f";

  /** One trained model: its name, the golden's directory, and whether it was given word vectors. */
  sealed interface Run permits Bilstm, Feedforward {

    String name();

    boolean vectors();

    /** The trainer's settings as their record writes them, for the {@code settings} record. */
    String settingsText();
  }

  /** A BiLSTM run: its CRF, encoder layers, vectors and adapter. */
  record Bilstm(String name, boolean crf, int layers, boolean vectors, boolean adapter) implements Run {

    @Override
    public String settingsText() {
      return settings().toString();
    }

    BilstmPOSTrainer.Settings settings() {
      return new BilstmPOSTrainer.Settings(8, 4, 4, 8, 5, 2, 0.02, 5.0, 0.1, 1, 12, 7L, 1, 0.1, 2,
          crf, layers, vectors ? 0.1 : 0.0, 0.1, 0.0, 0.0, adapter);
    }
  }

  /** A feedforward run: with or without vectors. */
  record Feedforward(String name, boolean vectors) implements Run {

    @Override
    public String settingsText() {
      return settings().toString();
    }

    FeedforwardPOSTrainer.Settings settings() {
      return new FeedforwardPOSTrainer.Settings(8, 12, 5, 4, 0.05, 1e-6, 0.1, 1, 1, 7L);
    }
  }

  /** The runs: every BiLSTM model format, and the feedforward tagger with and without vectors. */
  static List<Run> runs() {
    return List.of(
        new Bilstm("bilstm", false, 1, false, false),
        new Bilstm("bilstm-crf", true, 1, false, false),
        new Bilstm("bilstm-two-layer", false, 2, false, false),
        new Bilstm("bilstm-two-layer-crf", true, 2, false, false),
        new Bilstm("bilstm-vectors-crf", true, 1, true, false),
        new Bilstm("bilstm-adapter-crf", true, 1, true, true),
        new Feedforward("feedforward", false),
        new Feedforward("feedforward-vectors", true));
  }

  static List<AnnotatorGolden> all(Path inputs) {
    Path corpus = inputs.resolve(Runs.TAGGED).resolve(TRAIN_ID + ".txt");
    List<AnnotatorGolden> out = new ArrayList<>();
    for (Run run : runs()) {
      out.add(new PostagNeuralGolden(run, corpus));
    }
    return out;
  }

  private final Run run;
  private final Path corpus;
  private byte[] file;
  private BilstmPOSModel bilstm;
  private FeedforwardPOSModel feedforward;

  private PostagNeuralGolden(Run run, Path corpus) {
    this.run = run;
    this.corpus = corpus;
  }

  private static ObjectStream<POSSample> samples(byte[] bytes) throws IOException {
    return new WordTagSampleStream(new PlainTextByLineStream(() -> new ByteArrayInputStream(bytes),
        StandardCharsets.UTF_8));
  }

  /** Trains the model once, on the first golden that needs it, and reads it back from its bytes. */
  private synchronized void train() {
    if (file != null) {
      return;
    }
    try {
      Input input = Input.read(corpus, Runs.TAGGED, TRAIN_ID);
      ByteArrayOutputStream out = new ByteArrayOutputStream();
      try (ObjectStream<POSSample> samples = samples(input.bytes())) {
        if (run instanceof Bilstm b) {
          BilstmPOSModel m = b.vectors() ? BilstmPOSTrainer.train(samples, b.settings(), VECTORS)
              : BilstmPOSTrainer.train(samples, b.settings());
          m.serialize(out);
        } else {
          Feedforward f = (Feedforward) run;
          FeedforwardPOSModel m = f.vectors() ? FeedforwardPOSTrainer.train(samples, f.settings(), VECTORS)
              : FeedforwardPOSTrainer.train(samples, f.settings());
          m.serialize(out);
        }
      }
      byte[] bytes = out.toByteArray();
      if (run instanceof Bilstm) {
        bilstm = BilstmPOSModel.load(new ByteArrayInputStream(bytes));
      } else {
        feedforward = FeedforwardPOSModel.load(new ByteArrayInputStream(bytes));
      }
      file = bytes;
    } catch (IOException e) {
      throw new IllegalStateException(corpus + ": training " + run.name() + " failed", e);
    }
  }

  @Override
  public String annotator() {
    return ANNOTATOR;
  }

  @Override
  public List<ModelRef> models() {
    return List.of();
  }

  @Override
  public String modelDir() {
    return run.name();
  }

  @Override
  public String annotatorClass() {
    return (run instanceof Bilstm ? BilstmPOSTagger.class : FeedforwardPOSTagger.class).getName();
  }

  @Override
  public String method() {
    return "topKSequences";
  }

  @Override
  public Map<String, String> config() {
    Map<String, String> c = new TreeMap<>();
    c.put("run", run.name());
    c.put("corpus", Runs.TAGGED + "/" + TRAIN_ID);
    c.put("neuralPostagCommit", NEURAL_POSTAG_COMMIT);
    c.put("settings", run.settingsText());
    if (run.vectors()) {
      c.put("vectors", VECTORS_TEXT);
    }
    return c;
  }

  @Override
  public Set<String> inputSets() {
    return Set.of(Runs.TAGGED, Runs.SEQUENCE);
  }

  @Override
  public boolean accepts(Input input) {
    return (input.set().equals(Runs.TAGGED) && input.id().equals(TRAIN_ID))
        || (input.set().equals(Runs.SEQUENCE) && input.id().equals(TOKENS_ID));
  }

  @Override
  public List<Layer> layers(Input input) throws AnnotatorException {
    train();
    boolean training = input.set().equals(Runs.TAGGED);
    List<String[]> sentences = new ArrayList<>();
    if (training) {
      try (ObjectStream<POSSample> samples = samples(input.bytes())) {
        POSSample s;
        while ((s = samples.read()) != null) {
          if (s.getSentence().length > 0) {
            sentences.add(s.getSentence());
          }
        }
      } catch (IOException e) {
        throw new IllegalStateException(input.inputId() + ": the samples could not be read", e);
      }
    } else {
      sentences = Sentences.of(input);
    }

    List<Layer> layers = new ArrayList<>();
    layers.add(Sentences.tokens(sentences, false).layer());
    if (training) {
      layers.addAll(modelLayers());
    }
    String[] tags = run instanceof Bilstm ? bilstm.tags() : feedforward.tags();
    layers.add(new Layer("turbo:neural-tags", "document", new Column.Str("tag", tags)));

    Table emissions = new Table("turbo:neural-emissions", "sentence:u32", "index:u32", "tag:u32", "score:f64");
    Table viterbi = new Table("turbo:crf-viterbi", "sentence:u32", "index:u32", "tag:u32");
    Table marginals = new Table("turbo:crf-marginals", "sentence:u32", "index:u32", "tag:u32", "probability:f64");
    Table context = new Table("turbo:neural-context", "sentence:u32", "index:u32", "slot:u32", "present:u32",
        "symbol:str", "row:u32");
    Table vectorRows = new Table("turbo:neural-vector-rows", "sentence:u32", "index:u32", "slot:u32",
        "present:u32", "row:u32");
    Table tagged = new Table("turbo:neural-tagged", "sentence:u32", "index:u32", "tag:str", "probability:f64");
    Table sequences = new Table("turbo:neural-sequences", "sentence:u32", "score:f64");

    if (run instanceof Bilstm) {
      BilstmPOSTagger tagger = new BilstmPOSTagger(bilstm);
      for (int k = 0; k < sentences.size(); k++) {
        String[] tokens = sentences.get(k);
        double[][] scores = AnnotatorException.call(() -> bilstm.score(tokens));
        addScores(emissions, k, scores);
        if (bilstm.isCrf()) {
          int[] path = AnnotatorException.call(() -> NeuralAccess.viterbi(bilstm, scores));
          double[][] probs = AnnotatorException.call(() -> NeuralAccess.marginals(bilstm, scores));
          for (int i = 0; i < path.length; i++) {
            viterbi.row(k, i, path[i]);
          }
          addScores(marginals, k, probs);
          checkTags(input, k, tags, path, tagger.tag(tokens));
        }
        addSequence(input, tagged, sequences, k, tagger.tag(tokens),
            AnnotatorException.call(() -> tagger.topKSequences(tokens)));
      }
    } else {
      FeedforwardPOSTagger tagger = new FeedforwardPOSTagger(feedforward);
      for (int k = 0; k < sentences.size(); k++) {
        String[] tokens = sentences.get(k);
        String[] assigned = new String[tokens.length];
        int[] best = new int[tokens.length];
        double[][] scores = new double[tokens.length][];
        for (int i = 0; i < tokens.length; i++) {
          String[] symbols = NeuralAccess.context(tokens, i, i > 0 ? assigned[i - 1] : null,
              i > 1 ? assigned[i - 2] : null);
          int[] ids = NeuralAccess.featureIds(feedforward, symbols);
          for (int s = 0; s < symbols.length; s++) {
            context.row(k, i, s, symbols[s] == null ? 0 : 1, symbols[s] == null ? "" : symbols[s], ids[s]);
          }
          int[] rows = NeuralAccess.vectorRows(feedforward, tokens, i);
          if (rows != null) {
            for (int p = 0; p < rows.length; p++) {
              boolean present = rows[p] != NeuralAccess.NO_VECTOR;
              vectorRows.row(k, i, p, present ? 1 : 0, present ? rows[p] : 0);
            }
          }
          double[] row = AnnotatorException.call(() -> NeuralAccess.score(feedforward, ids, rows));
          scores[i] = row;
          for (int o = 1; o < row.length; o++) {
            if (row[o] > row[best[i]]) {
              best[i] = o;
            }
          }
          assigned[i] = tags[best[i]];
        }
        addScores(emissions, k, scores);
        checkTags(input, k, tags, best, tagger.tag(tokens));
        addSequence(input, tagged, sequences, k, tagger.tag(tokens),
            AnnotatorException.call(() -> tagger.topKSequences(tokens)));
      }
    }

    layers.add(emissions.layer());
    if (run instanceof Bilstm && bilstm.isCrf()) {
      layers.add(viterbi.layer());
      layers.add(marginals.layer());
    }
    if (run instanceof Feedforward) {
      layers.add(context.layer());
      if (feedforward.usesPretrainedVectors()) {
        layers.add(vectorRows.layer());
      }
    }
    layers.add(tagged.layer());
    layers.add(sequences.layer());
    return layers;
  }

  /** The model file's magic, length, SHA-256 and bytes. */
  private List<Layer> modelLayers() {
    int magicLength = ((file[0] & 0xff) << 8) | (file[1] & 0xff);
    String magic = new String(file, 2, magicLength, StandardCharsets.UTF_8);
    Table model = new Table("turbo:neural-model", "magic:str", "length:u32", "sha256:str");
    model.row(magic, (long) file.length, Sha.sha256(file));
    Table bytes = new Table("turbo:neural-model-bytes", "offset:u32", "hex:str");
    for (int from = 0; from < file.length; from += BYTES_PER_ROW) {
      bytes.row(from, Sha.hex(Arrays.copyOfRange(file, from, Math.min(file.length, from + BYTES_PER_ROW))));
    }
    return List.of(model.layer(), bytes.layer());
  }

  private static void addScores(Table table, int sentence, double[][] scores) {
    for (int i = 0; i < scores.length; i++) {
      for (int t = 0; t < scores[i].length; t++) {
        table.row(sentence, i, t, scores[i][t]);
      }
    }
  }

  /** Checks that the recorded path gives the tagger's own tags. */
  private static void checkTags(Input input, int sentence, String[] tags, int[] path, String[] tagged) {
    for (int i = 0; i < path.length; i++) {
      if (!tags[path[i]].equals(tagged[i])) {
        throw new IllegalStateException(input.inputId() + ": sentence " + sentence + " token " + i
            + ": the recorded path gives " + tags[path[i]] + ", the tagger " + tagged[i]);
      }
    }
  }

  private static void addSequence(Input input, Table tagged, Table sequences, int sentence, String[] tags,
      Sequence[] best) {
    if (best.length != 1) {
      throw new IllegalStateException(input.inputId() + ": topKSequences gave " + best.length + " sequences");
    }
    List<String> outcomes = best[0].getOutcomes();
    double[] probs = best[0].getProbs();
    if (!outcomes.equals(Arrays.asList(tags))) {
      throw new IllegalStateException(input.inputId() + ": sentence " + sentence
          + ": topKSequences and tag give other tags");
    }
    for (int i = 0; i < outcomes.size(); i++) {
      tagged.row(sentence, i, outcomes.get(i), probs[i]);
    }
    sequences.row(sentence, best[0].getScore());
  }
}
