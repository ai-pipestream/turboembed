package ai.pipestream.turbo.goldens;

import java.io.IOException;
import java.io.InputStream;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.Set;

import golden.StoredClusters;
import opennlp.tools.chunker.ChunkerME;
import opennlp.tools.chunker.ChunkerModel;
import opennlp.tools.dictionary.Dictionary;
import opennlp.tools.ml.BeamSearch;
import opennlp.tools.ml.model.MaxentModel;
import opennlp.tools.namefind.NameContextGenerator;
import opennlp.tools.namefind.NameFinderME;
import opennlp.tools.namefind.TokenNameFinderFactory;
import opennlp.tools.namefind.TokenNameFinderModel;
import opennlp.tools.postag.DefaultPOSContextGenerator;
import opennlp.tools.postag.POSModel;
import opennlp.tools.postag.POSTagFormat;
import opennlp.tools.postag.POSTaggerME;
import opennlp.tools.util.BeamSearchContextGenerator;
import opennlp.tools.util.Sequence;
import opennlp.tools.util.SequenceCodec;
import opennlp.tools.util.SequenceValidator;
import opennlp.tools.util.Span;
import opennlp.tools.util.TokenTag;
import opennlp.tools.util.featuregen.AdditionalContextFeatureGenerator;
import opennlp.tools.util.featuregen.WindowFeatureGenerator;

/**
 * The beam search of {@code POSTaggerME}, {@code ChunkerME} and {@code NameFinderME} over the
 * {@code sequence/sentences} input: for each sentence, every sequence of the final beam with its
 * outcomes, their probabilities and its score, and for every eighth sentence from the first every
 * context the search asked for.
 *
 * <p>The search is wired as each tagger wires it, but called through {@code bestSequences} with
 * the beam size as the count, so the whole final beam is recorded; each sentence's best sequence
 * is checked against what the tagger itself returns. The tagger runs the factory's context
 * generator with a cache as large as the beam and the tag dictionary's validator; with
 * {@code +default-context} it runs {@code DefaultPOSContextGenerator} with the names dictionary as
 * its n-gram dictionary instead, as a model from before OpenNLP 1.8 with one would. The chunker
 * runs on the best tags of the tagger model that is its second {@code model} record. The name
 * finder runs the factory's context generator with an additional context generator in a window of
 * eight tokens on each side and an empty additional context, and the sentences are one document:
 * the best outcomes are passed to {@code updateAdaptiveData} after each sentence. With
 * {@code +fresh-sums} it runs over {@link #freshSums}, and is not checked against
 * {@code NameFinderME}.</p>
 *
 * <p>The layers, anchored to the document: {@code turbo:sentence-tokens}; for the chunker
 * {@code turbo:sentence-tags}, aligned to the tokens; {@code turbo:beam-calls} (each context asked
 * for, numbered from 0, with its sentence and token index), {@code turbo:beam-call-priors} (the
 * outcomes before it) and {@code turbo:beam-call-features} ({@link TrainGolden#lossless});
 * {@code turbo:beam-sequences} (each sequence of the final beam, numbered from 0, with its sentence
 * and score) and {@code turbo:beam-outcomes} (its outcomes with their probabilities); and for the
 * name finder {@code turbo:beam-names}, the spans the codec decodes from the best sequence.</p>
 */
final class BeamGolden implements AnnotatorGolden {

  enum Kind {
    POS("pos"),
    POS_DEFAULT_CONTEXT("pos-default-context"),
    CHUNK("chunk"),
    NAMES("names"),
    NAMES_FRESH_SUMS("names-fresh-sums");

    final String name;

    Kind(String name) {
      this.name = name;
    }
  }

  private final Kind kind;
  private final ModelFile model;
  /** The tagger whose tags the chunker runs on, or the n-gram dictionary of the default context. */
  private final ModelFile second;

  private BeamGolden(Kind kind, ModelFile model, ModelFile second) {
    this.kind = kind;
    this.model = model;
    this.second = second;
  }

  /**
   * Every run: the fixture models and the published ones. The cluster serializers of
   * {@code ner-bio-fixture.bin} are registered first ({@link StoredClusters}).
   */
  static List<BeamGolden> all(Path models, Path fixtures) throws IOException {
    StoredClusters.register();
    ModelFile pos = ModelFile.read(fixtures.resolve(Runs.POS_FIXTURE));
    ModelFile names = ModelFile.read(fixtures.resolve(Runs.FEATUREGEN).resolve(FeatureGolden.NAMES_DICTIONARY));
    ModelFile qn = ModelFile.read(fixtures.resolve(Runs.NER_QN_FIXTURE));
    ModelFile maxent = ModelFile.read(models.resolve(Runs.POS_MODEL_EN_15_MAXENT));
    List<BeamGolden> out = new ArrayList<>();
    out.add(new BeamGolden(Kind.POS, pos, null));
    out.add(new BeamGolden(Kind.POS_DEFAULT_CONTEXT, pos, names));
    out.add(new BeamGolden(Kind.CHUNK, ModelFile.read(fixtures.resolve(Runs.CHUNKER_FIXTURE)), pos));
    out.add(new BeamGolden(Kind.NAMES, ModelFile.read(fixtures.resolve(Runs.NER_BIO_FIXTURE)), null));
    out.add(new BeamGolden(Kind.NAMES, ModelFile.read(fixtures.resolve(Runs.NER_BILOU_FIXTURE)), null));
    out.add(new BeamGolden(Kind.NAMES, qn, null));
    out.add(new BeamGolden(Kind.NAMES_FRESH_SUMS, qn, null));
    out.add(new BeamGolden(Kind.POS, ModelFile.read(models.resolve(Runs.POS_MODEL_EN)), null));
    out.add(new BeamGolden(Kind.POS, ModelFile.read(models.resolve(Runs.POS_MODEL_EN_15_PERCEPTRON)), null));
    out.add(new BeamGolden(Kind.POS, maxent, null));
    out.add(new BeamGolden(Kind.CHUNK, ModelFile.read(models.resolve(Runs.CHUNKER_MODEL_EN_15)), maxent));
    out.add(new BeamGolden(Kind.NAMES, ModelFile.read(models.resolve(Runs.PERSON_MODEL_EN_15)), null));
    out.add(new BeamGolden(Kind.NAMES, ModelFile.read(models.resolve(Runs.LOCATION_MODEL_EN_15)), null));
    return out;
  }

  @Override
  public String annotator() {
    return "beam";
  }

  @Override
  public List<ModelRef> models() {
    return second == null ? List.of(model.ref()) : List.of(model.ref(), second.ref());
  }

  @Override
  public String modelDir() {
    return switch (kind) {
      case POS_DEFAULT_CONTEXT -> model.ref().baseName() + "+default-context";
      case NAMES_FRESH_SUMS -> model.ref().baseName() + "+fresh-sums";
      default -> model.ref().baseName();
    };
  }

  @Override
  public String annotatorClass() {
    return BeamSearch.class.getName();
  }

  @Override
  public String method() {
    return "bestSequences";
  }

  @Override
  public Map<String, String> config() {
    return Map.of("kind", kind.name);
  }

  @Override
  public Set<String> inputSets() {
    return Set.of(Runs.SEQUENCE);
  }

  @Override
  public boolean accepts(Input input) {
    return input.set().equals(Runs.SEQUENCE) && input.id().equals("sentences");
  }

  /** The sentences whose contexts are recorded. */
  static boolean recorded(int sentence) {
    return sentence % 8 == 0;
  }

  @Override
  public List<Layer> layers(Input input) {
    List<String[]> sentences = Sentences.of(input);
    Layers out = new Layers();
    try {
      switch (kind) {
        case POS, POS_DEFAULT_CONTEXT -> pos(model, sentences, kind == Kind.POS_DEFAULT_CONTEXT ? dictionary(second) : null,
            out);
        case CHUNK -> chunk(sentences, pos(second, sentences, null, new Layers()), out);
        case NAMES, NAMES_FRESH_SUMS -> names(sentences, kind == Kind.NAMES_FRESH_SUMS, out);
      }
    } catch (IOException e) {
      throw new IllegalStateException(modelDir() + ": loading the model", e);
    }
    List<Layer> layers = new ArrayList<>();
    layers.add(Sentences.tokens(sentences, false).layer());
    if (kind == Kind.CHUNK) {
      layers.add(out.tags.layer());
    }
    layers.add(out.config.layer());
    layers.add(out.calls.layer());
    layers.add(out.priors.layer());
    layers.add(out.features.layer());
    layers.add(out.sequences.layer());
    layers.add(out.outcomes.layer());
    if (kind == Kind.NAMES || kind == Kind.NAMES_FRESH_SUMS) {
      layers.add(out.names.layer());
    }
    return layers;
  }

  /** The tables one run fills. */
  static final class Layers {
    final Table config = new Table("turbo:beam-config", "beam-size:u32", "codec:str");
    final Table tags = Table.aligned("turbo:sentence-tags", "turbo:sentence-tokens", "tag:str");
    final Table calls = new Table("turbo:beam-calls", "call:u32", "sentence:u32", "index:u32");
    final Table priors = new Table("turbo:beam-call-priors", "call:u32", "outcome:str");
    final Table features = new Table("turbo:beam-call-features", "call:u32", "feature:str");
    final Table sequences = new Table("turbo:beam-sequences", "sequence:u32", "sentence:u32", "score:f64");
    final Table outcomes = new Table("turbo:beam-outcomes", "sequence:u32", "outcome:str", "probability:f64");
    final Table names = new Table("turbo:beam-names", "sentence:u32", "start:u32", "end:u32", "type:str", "typed:u32");

    void sequences(int sentence, Sequence[] found) {
      for (Sequence s : found) {
        int n = sequences.size();
        sequences.row(n, sentence, s.getScore());
        List<String> o = s.getOutcomes();
        double[] p = s.getProbs();
        for (int i = 0; i < o.size(); i++) {
          outcomes.row(n, o.get(i), p[i]);
        }
      }
    }
  }

  /** Passes contexts on, recording each in the tables while {@link #sentence} is set. */
  static final class Recorder<T> implements BeamSearchContextGenerator<T> {
    final BeamSearchContextGenerator<T> inner;
    final Layers out;
    /** The sentence being recorded, or -1. */
    int sentence = -1;

    Recorder(BeamSearchContextGenerator<T> inner, Layers out) {
      this.inner = inner;
      this.out = out;
    }

    @Override
    public String[] getContext(int index, T[] sequence, String[] prior, Object[] additional) {
      String[] context = inner.getContext(index, sequence, prior, additional);
      if (sentence >= 0) {
        int call = out.calls.size();
        out.calls.row(call, sentence, index);
        for (String p : prior) {
          out.priors.row(call, p);
        }
        for (String f : context) {
          out.features.row(call, TrainGolden.lossless(f));
        }
      }
      return context;
    }
  }

  /**
   * The model, but evaluating every context into a new array. {@code QNModel.eval(context, probs)}
   * adds its sums to what {@code probs} holds, and {@code BeamSearch} passes the same array to every
   * call, so inside a beam search a QN model's sums start from the probabilities of the evaluation
   * before. Running the search over this wrapper as well shows that this is the only difference.
   */
  static MaxentModel freshSums(MaxentModel m) {
    return new MaxentModel() {
      @Override
      public double[] eval(String[] context) {
        return m.eval(context);
      }

      @Override
      public double[] eval(String[] context, double[] probs) {
        double[] fresh = m.eval(context);
        System.arraycopy(fresh, 0, probs, 0, fresh.length);
        return probs;
      }

      @Override
      public double[] eval(String[] context, float[] values) {
        return m.eval(context, values);
      }

      @Override
      public String getBestOutcome(double[] outcomes) {
        return m.getBestOutcome(outcomes);
      }

      @Override
      public String getAllOutcomes(double[] outcomes) {
        return m.getAllOutcomes(outcomes);
      }

      @Override
      public String getOutcome(int i) {
        return m.getOutcome(i);
      }

      @Override
      public int getIndex(String outcome) {
        return m.getIndex(outcome);
      }

      @Override
      public int getNumOutcomes() {
        return m.getNumOutcomes();
      }
    };
  }

  private static Dictionary dictionary(ModelFile file) throws IOException {
    try (InputStream in = file.stream()) {
      return new Dictionary(in);
    }
  }

  private static int beamSize(String manifestValue, int defaultSize) {
    return manifestValue != null ? Integer.parseInt(manifestValue) : defaultSize;
  }

  private static void check(boolean ok, String what) {
    if (!ok) {
      throw new IllegalStateException(what);
    }
  }

  /** The tagger's search over every sentence; returns the best tags of each. */
  private List<String[]> pos(ModelFile file, List<String[]> sentences, Dictionary ngram, Layers out)
      throws IOException {
    POSModel m;
    try (InputStream in = file.stream()) {
      m = new POSModel(in);
    }
    int beam = beamSize(m.getManifestProperty(BeamSearch.BEAM_SIZE_PARAMETER), POSTaggerME.DEFAULT_BEAM_SIZE);
    out.config.row(beam, "");
    BeamSearch search = new BeamSearch(beam, (MaxentModel) m.getArtifact("pos.model"));
    Recorder<String> cg = new Recorder<>(ngram == null ? m.getFactory().getPOSContextGenerator(beam)
        : new DefaultPOSContextGenerator(beam, ngram), out);
    SequenceValidator<String> validator = m.getFactory().getSequenceValidator();
    POSTaggerME tagger = ngram == null ? new POSTaggerME(m, POSTagFormat.CUSTOM) : null;
    List<String[]> best = new ArrayList<>();
    for (int k = 0; k < sentences.size(); k++) {
      String[] tokens = sentences.get(k).clone();
      cg.sentence = recorded(k) ? k : -1;
      Sequence[] found = search.bestSequences(beam, tokens, null, cg, validator);
      cg.sentence = -1;
      out.sequences(k, found);
      String[] tags = found[0].getOutcomes().toArray(new String[0]);
      if (tagger != null) {
        check(Arrays.equals(tagger.tag(tokens), tags), file.ref().fileName() + " sentence " + k);
      }
      best.add(tags);
    }
    return best;
  }

  /** The chunker's search over every sentence, with the given tags. */
  private void chunk(List<String[]> sentences, List<String[]> tags, Layers out) throws IOException {
    ChunkerModel m;
    try (InputStream in = model.stream()) {
      m = new ChunkerModel(in);
    }
    int beam = beamSize(m.getManifestProperty(BeamSearch.BEAM_SIZE_PARAMETER), ChunkerME.DEFAULT_BEAM_SIZE);
    out.config.row(beam, "");
    BeamSearch search = new BeamSearch(beam, (MaxentModel) m.getArtifact("chunker.model"));
    Recorder<TokenTag> cg = new Recorder<>(m.getFactory().getContextGenerator(), out);
    SequenceValidator<TokenTag> validator = m.getFactory().getSequenceValidator();
    ChunkerME chunker = new ChunkerME(m);
    for (int k = 0; k < sentences.size(); k++) {
      String[] tokens = sentences.get(k).clone();
      for (String t : tags.get(k)) {
        out.tags.row(t);
      }
      TokenTag[] tuples = TokenTag.create(tokens, tags.get(k));
      cg.sentence = recorded(k) ? k : -1;
      Sequence[] found = search.bestSequences(beam, tuples, new Object[] {}, cg, validator);
      cg.sentence = -1;
      out.sequences(k, found);
      check(Arrays.equals(chunker.chunk(tokens, tags.get(k)), found[0].getOutcomes().toArray(new String[0])),
          model.ref().fileName() + " sentence " + k);
    }
  }

  /** The name finder's search over every sentence, as one document. */
  private void names(List<String[]> sentences, boolean fresh, Layers out) throws IOException {
    TokenNameFinderModel m;
    try (InputStream in = model.stream()) {
      m = new TokenNameFinderModel(in);
    }
    int beam = beamSize(m.getManifestProperty(BeamSearch.BEAM_SIZE_PARAMETER), NameFinderME.DEFAULT_BEAM_SIZE);
    MaxentModel maxent = m.getArtifact("nameFinder.model");
    BeamSearch search = new BeamSearch(beam, fresh ? freshSums(maxent) : maxent);
    TokenNameFinderFactory factory = m.getFactory();
    SequenceCodec<String> codec = factory.createSequenceCodec();
    out.config.row(beam, codec.getClass().getName());
    SequenceValidator<String> validator = codec.createSequenceValidator();
    NameContextGenerator context = factory.createContextGenerator();
    AdditionalContextFeatureGenerator additional = new AdditionalContextFeatureGenerator();
    context.addFeatureGenerator(new WindowFeatureGenerator(additional, 8, 8));
    Recorder<String> cg = new Recorder<>(context, out);
    NameFinderME finder = new NameFinderME(m);
    String[][] none = new String[0][0];
    for (int k = 0; k < sentences.size(); k++) {
      String[] tokens = sentences.get(k).clone();
      additional.setCurrentContext(none);
      cg.sentence = recorded(k) ? k : -1;
      Sequence[] found = search.bestSequences(beam, tokens, none, cg, validator);
      cg.sentence = -1;
      out.sequences(k, found);
      List<String> best = found[0].getOutcomes();
      context.updateAdaptiveData(tokens, best.toArray(new String[0]));
      Span[] spans = codec.decode(best);
      for (Span s : spans) {
        out.names.row(k, s.getStart(), s.getEnd(), s.getType() == null ? "" : s.getType(),
            Table.flag(s.getType() != null));
      }
      if (fresh) {
        continue;
      }
      Span[] expected = finder.find(tokens);
      check(expected.length == spans.length, model.ref().fileName() + " sentence " + k);
      for (int i = 0; i < spans.length; i++) {
        check(expected[i].getStart() == spans[i].getStart() && expected[i].getEnd() == spans[i].getEnd()
            && Objects.equals(expected[i].getType(), spans[i].getType()), model.ref().fileName() + " sentence " + k);
      }
    }
  }
}
