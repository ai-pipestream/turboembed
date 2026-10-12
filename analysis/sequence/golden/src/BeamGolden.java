import java.io.IOException;
import java.io.InputStream;
import java.io.Writer;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Objects;

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
 * Runs the beam search of {@code POSTaggerME}, {@code ChunkerME} and {@code NameFinderME} over
 * {@code inputs/sentences.txt} and writes, for each sentence, the best sequences with their
 * scores and probabilities as raw double bits, and for some sentences every context the search
 * asked for.
 *
 * <p>The search is wired as each tagger wires it, but called through {@code bestSequences} with
 * the beam size as the count, so the other sequences in the final beam are recorded too. Each
 * sentence's best sequence is checked against what the tagger itself returns.
 *
 * <p>Arguments: the golden directory, the output directory, the OpenNLP commit, and optionally a
 * directory holding the public English models (see the crate README).
 */
public final class BeamGolden {

  private BeamGolden() {
  }

  /** The sentences whose contexts are written. */
  static boolean recorded(int sentence) {
    return sentence % 8 == 0;
  }

  /** Writes each context it passes on, while {@link #lines} is set. */
  static final class Recorder<T> implements BeamSearchContextGenerator<T> {
    final BeamSearchContextGenerator<T> inner;
    List<String> lines;

    Recorder(BeamSearchContextGenerator<T> inner) {
      this.inner = inner;
    }

    @Override
    public String[] getContext(int index, T[] sequence, String[] prior, Object[] additional) {
      String[] context = inner.getContext(index, sequence, prior, additional);
      if (lines != null) {
        lines.add("call: " + index + (prior.length == 0 ? "" : "\t" + Common.join(prior)));
        lines.add("features: " + Common.join(context));
      }
      return context;
    }
  }

  /**
   * The model, but evaluating every context into a new array. {@code QNModel.eval(context, probs)}
   * adds its sums to what {@code probs} holds, and BeamSearch passes the same array to every call,
   * so a QN model's sums start from the probabilities of the evaluation before. Running the search
   * over this wrapper as well shows that this is the only difference.
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

  static int beamSize(String manifestValue, int defaultSize) {
    return manifestValue != null ? Integer.parseInt(manifestValue) : defaultSize;
  }

  static void writeSequences(Writer w, Sequence[] sequences) throws IOException {
    for (Sequence s : sequences) {
      Common.line(w, "sequence: " + Common.bits(s.getScore()));
      Common.line(w, "outcomes: " + Common.join(s.getOutcomes()));
      StringBuilder probs = new StringBuilder();
      for (double p : s.getProbs()) {
        if (probs.length() > 0) {
          probs.append('\t');
        }
        probs.append(Common.bits(p));
      }
      Common.line(w, "probs: " + probs);
    }
  }

  static Writer start(Path out, String name, String commit, Path model, String kind, int beam)
      throws IOException {
    Writer w = Common.writer(out.resolve("beam-" + name + ".txt"));
    Common.header(w, "beam", commit);
    Common.line(w, "model: " + model.getFileName());
    Common.line(w, "sha256: " + Common.sha256(model));
    Common.line(w, "kind: " + kind);
    Common.line(w, "beam: " + beam);
    return w;
  }

  static void check(boolean ok, String what) {
    if (!ok) {
      throw new IllegalStateException(what);
    }
  }

  /**
   * POSTaggerME: the factory's context generator with a cache as large as the beam, and the tag
   * dictionary's validator. Returns the best tags of each sentence. With {@code ngram}, the
   * search uses {@code DefaultPOSContextGenerator} with that dictionary instead, as a model from
   * before OpenNLP 1.8 with an n-gram dictionary would.
   */
  static List<String[]> pos(Path out, String name, String commit, Path file, List<String[]> sentences,
      Dictionary ngram) throws IOException {
    POSModel model;
    try (InputStream in = Files.newInputStream(file)) {
      model = new POSModel(in);
    }
    int beam = beamSize(model.getManifestProperty(BeamSearch.BEAM_SIZE_PARAMETER),
        POSTaggerME.DEFAULT_BEAM_SIZE);
    MaxentModel maxent = model.getArtifact("pos.model");
    BeamSearch search = new BeamSearch(beam, maxent);
    Recorder<String> cg = new Recorder<>(ngram == null
        ? model.getFactory().getPOSContextGenerator(beam)
        : new DefaultPOSContextGenerator(beam, ngram));
    SequenceValidator<String> validator = model.getFactory().getSequenceValidator();
    POSTaggerME tagger = ngram == null ? new POSTaggerME(model, POSTagFormat.CUSTOM) : null;

    List<String[]> best = new ArrayList<>();
    try (Writer w = start(out, name, commit, file, ngram == null ? "pos" : "pos-default-context", beam)) {
      for (int k = 0; k < sentences.size(); k++) {
        String[] tokens = sentences.get(k).clone();
        Common.line(w, "sentence: " + k);
        Common.line(w, "tokens: " + Common.join(tokens));
        List<String> lines = new ArrayList<>();
        cg.lines = recorded(k) ? lines : null;
        Sequence[] sequences = search.bestSequences(beam, tokens, null, cg, validator);
        cg.lines = null;
        for (String l : lines) {
          Common.line(w, l);
        }
        writeSequences(w, sequences);
        String[] tags = sequences[0].getOutcomes().toArray(new String[0]);
        if (tagger != null) {
          check(Arrays.equals(tagger.tag(tokens), tags), name + " sentence " + k);
        }
        best.add(tags);
      }
    }
    return best;
  }

  /** ChunkerME, with the given tags. */
  static void chunk(Path out, String name, String commit, Path file, List<String[]> sentences,
      List<String[]> tags) throws IOException {
    ChunkerModel model;
    try (InputStream in = Files.newInputStream(file)) {
      model = new ChunkerModel(in);
    }
    int beam = beamSize(model.getManifestProperty(BeamSearch.BEAM_SIZE_PARAMETER),
        ChunkerME.DEFAULT_BEAM_SIZE);
    MaxentModel maxent = model.getArtifact("chunker.model");
    BeamSearch search = new BeamSearch(beam, maxent);
    Recorder<TokenTag> cg = new Recorder<>(model.getFactory().getContextGenerator());
    SequenceValidator<TokenTag> validator = model.getFactory().getSequenceValidator();
    ChunkerME chunker = new ChunkerME(model);

    try (Writer w = start(out, name, commit, file, "chunk", beam)) {
      for (int k = 0; k < sentences.size(); k++) {
        String[] tokens = sentences.get(k).clone();
        TokenTag[] tuples = TokenTag.create(tokens, tags.get(k));
        Common.line(w, "sentence: " + k);
        Common.line(w, "tokens: " + Common.join(tokens));
        Common.line(w, "tags: " + Common.join(tags.get(k)));
        List<String> lines = new ArrayList<>();
        cg.lines = recorded(k) ? lines : null;
        Sequence[] sequences = search.bestSequences(beam, tuples, new Object[] {}, cg, validator);
        cg.lines = null;
        for (String l : lines) {
          Common.line(w, l);
        }
        writeSequences(w, sequences);
        check(Arrays.equals(chunker.chunk(tokens, tags.get(k)),
            sequences[0].getOutcomes().toArray(new String[0])), name + " sentence " + k);
      }
    }
  }

  /**
   * NameFinderME: the factory's context generator, then an additional context generator in a
   * window of eight tokens on each side, with an empty additional context; the outcomes of the
   * best sequence are passed to {@code updateAdaptiveData} after each sentence. The sentences are
   * one document.
   */
  static void names(Path out, String name, String commit, Path file, List<String[]> sentences)
      throws IOException {
    names(out, name, commit, file, sentences, false);
  }

  /** With {@code fresh}, over {@link #freshSums}, and not checked against NameFinderME. */
  static void names(Path out, String name, String commit, Path file, List<String[]> sentences,
      boolean fresh) throws IOException {
    TokenNameFinderModel model;
    try (InputStream in = Files.newInputStream(file)) {
      model = new TokenNameFinderModel(in);
    }
    int beam = beamSize(model.getManifestProperty(BeamSearch.BEAM_SIZE_PARAMETER),
        NameFinderME.DEFAULT_BEAM_SIZE);
    MaxentModel maxent = model.getArtifact("nameFinder.model");
    BeamSearch search = new BeamSearch(beam, fresh ? freshSums(maxent) : maxent);
    TokenNameFinderFactory factory = model.getFactory();
    SequenceCodec<String> codec = factory.createSequenceCodec();
    SequenceValidator<String> validator = codec.createSequenceValidator();
    NameContextGenerator context = factory.createContextGenerator();
    AdditionalContextFeatureGenerator additional = new AdditionalContextFeatureGenerator();
    context.addFeatureGenerator(new WindowFeatureGenerator(additional, 8, 8));
    Recorder<String> cg = new Recorder<>(context);
    NameFinderME finder = new NameFinderME(model);
    String[][] none = new String[0][0];

    try (Writer w = start(out, name, commit, file, fresh ? "names-fresh-sums" : "names", beam)) {
      Common.line(w, "codec: " + codec.getClass().getName());
      for (int k = 0; k < sentences.size(); k++) {
        String[] tokens = sentences.get(k).clone();
        Common.line(w, "sentence: " + k);
        Common.line(w, "tokens: " + Common.join(tokens));
        additional.setCurrentContext(none);
        List<String> lines = new ArrayList<>();
        cg.lines = recorded(k) ? lines : null;
        Sequence[] sequences = search.bestSequences(beam, tokens, none, cg, validator);
        cg.lines = null;
        for (String l : lines) {
          Common.line(w, l);
        }
        writeSequences(w, sequences);
        List<String> best = sequences[0].getOutcomes();
        context.updateAdaptiveData(tokens, best.toArray(new String[0]));
        Span[] spans = codec.decode(best);
        StringBuilder b = new StringBuilder();
        for (Span s : spans) {
          if (b.length() > 0) {
            b.append('\t');
          }
          b.append(s.getStart()).append(' ').append(s.getEnd()).append(' ')
              .append(Common.escape(s.getType() == null ? "null" : s.getType()));
        }
        Common.line(w, "spans: " + b);
        if (fresh) {
          continue;
        }
        Span[] found = finder.find(tokens);
        check(found.length == spans.length, name + " sentence " + k);
        for (int i = 0; i < spans.length; i++) {
          check(found[i].getStart() == spans[i].getStart() && found[i].getEnd() == spans[i].getEnd()
              && Objects.equals(found[i].getType(), spans[i].getType()),
              name + " sentence " + k);
        }
      }
    }
  }

  public static void main(String[] args) throws IOException {
    Path golden = Path.of(args[0]);
    Path out = Path.of(args[1]);
    String commit = args[2];
    StoredClusters.register();
    List<String[]> sentences = Common.sentences(golden.resolve("inputs/sentences.txt"));
    Path models = golden.resolve("models");

    List<String[]> tags = pos(out, "pos-fixture", commit, models.resolve("pos-fixture.bin"),
        sentences, null);
    Dictionary ngram;
    try (InputStream in = Files.newInputStream(golden.resolve("resources/names.dictionary"))) {
      ngram = new Dictionary(in);
    }
    pos(out, "pos-fixture-default-context", commit, models.resolve("pos-fixture.bin"), sentences,
        ngram);
    chunk(out, "chunker-fixture", commit, models.resolve("chunker-fixture.bin"), sentences, tags);
    names(out, "ner-bio-fixture", commit, models.resolve("ner-bio-fixture.bin"), sentences);
    names(out, "ner-bilou-fixture", commit, models.resolve("ner-bilou-fixture.bin"), sentences);
    names(out, "ner-qn-fixture", commit, models.resolve("ner-qn-fixture.bin"), sentences);
    names(out, "ner-qn-fixture-fresh-sums", commit, models.resolve("ner-qn-fixture.bin"), sentences,
        true);

    if (args.length > 3) {
      Path real = Path.of(args[3]);
      pos(out, "en-ud-ewt-pos", commit, real.resolve("opennlp-en-ud-ewt-pos-1.3-2.5.4.bin"),
          sentences, null);
      pos(out, "en-pos-perceptron", commit, real.resolve("en-pos-perceptron.bin"), sentences, null);
      List<String[]> penn = pos(out, "en-pos-maxent", commit, real.resolve("en-pos-maxent.bin"),
          sentences, null);
      chunk(out, "en-chunker", commit, real.resolve("en-chunker.bin"), sentences, penn);
      names(out, "en-ner-person", commit, real.resolve("en-ner-person.bin"), sentences);
      names(out, "en-ner-location", commit, real.resolve("en-ner-location.bin"), sentences);
    }
  }
}
