// Writes the BiLSTM-CRF and feedforward neural POS model fixtures and test
// vectors of the turbo-postag crate from the pinned JDK 21 image and OpenNLP
// commit cdf33c653d5d61d5af7d37ffee189a83bd8392ea.
//
// Run from the repository root with the JDK 21 image:
//
//     docker run --rm --user "$(id -u):$(id -g)" \
//         -v "$PWD:/work" -v /work/worktrees/opennlp/uber:/opennlp:ro \
//         eclipse-temurin@sha256:3e3c176ffed168beb42c607be9bc1639b466cf00261a0fb04425562c9d0c5c2b \
//         sh -c 'javac -cp "/opennlp/opennlp-api/target/classes:/opennlp/opennlp-core/opennlp-runtime/target/classes:/opennlp/opennlp-extensions/opennlp-uima/target/dependency/*" -d /tmp /work/analysis/postag/java/NeuralPosVectors.java && java -cp "/tmp:/opennlp/opennlp-api/target/classes:/opennlp/opennlp-core/opennlp-runtime/target/classes:/opennlp/opennlp-extensions/opennlp-uima/target/dependency/*" opennlp.tools.postag.NeuralPosVectors /work/analysis/postag/tests/vectors'

package opennlp.tools.postag;

import java.io.ByteArrayOutputStream;
import java.io.PrintWriter;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.security.MessageDigest;
import java.util.ArrayList;
import java.util.HexFormat;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Random;
import opennlp.tools.util.CollectionObjectStream;
import opennlp.tools.util.Sequence;

public final class NeuralPosVectors {

  private static final String OPENNLP_COMMIT = "cdf33c653d5d61d5af7d37ffee189a83bd8392ea";

  private NeuralPosVectors() {}

  private static List<POSSample> trainingCorpus() {
    return List.of(
        new POSSample(
            new String[] {"The", "quick", "brown", "fox", "jumps", "over", "the", "lazy", "dog", "."},
            new String[] {"DET", "ADJ", "ADJ", "NOUN", "VERB", "ADP", "DET", "ADJ", "NOUN", "PUNCT"}),
        new POSSample(
            new String[] {"A", "fast", "fox", "runs", "quickly", "in", "Paris", "."},
            new String[] {"DET", "ADJ", "NOUN", "VERB", "ADV", "ADP", "PROPN", "PUNCT"}),
        new POSSample(
            new String[] {"Dogs", "and", "cats", "sleep", "quietly", "under", "25", "trees", "."},
            new String[] {"NOUN", "CCONJ", "NOUN", "VERB", "ADV", "ADP", "NUM", "NOUN", "PUNCT"}),
        new POSSample(
            new String[] {"Paris", "welcomes", "happy", "travelers", "at", "caf\u00E9", "tables", "."},
            new String[] {"PROPN", "VERB", "ADJ", "NOUN", "ADP", "NOUN", "NOUN", "PUNCT"}));
  }

  private static Map<String, float[]> pretrainedTable() {
    Map<String, float[]> map = new LinkedHashMap<>();
    map.put("the", new float[] {0.25f, -0.50f, 0.75f, 0.125f});
    map.put("fox", new float[] {0.80f, 0.10f, -0.30f, 0.40f});
    map.put("dog", new float[] {0.75f, 0.15f, -0.25f, 0.35f});
    map.put("jumps", new float[] {-0.40f, 0.90f, 0.20f, -0.10f});
    map.put("runs", new float[] {-0.35f, 0.85f, 0.15f, -0.05f});
    map.put("paris", new float[] {0.10f, -0.20f, 0.60f, 0.95f});
    map.put("london", new float[] {0.12f, -0.18f, 0.58f, 0.92f});
    return map;
  }

  private static List<String[]> evaluationSentences() {
    return List.of(
        new String[] {"The", "brown", "fox", "jumps", "quickly", "."},
        new String[] {"A", "curious", "cat", "runs", "in", "London", "."},
        new String[] {"42", "dogs", "sleep", "at", "caf\u00E9", "tables", "."},
        new String[] {"Paris"});
  }

  private static String hex(double x) {
    return String.format("%016x", Double.doubleToRawLongBits(x));
  }

  private static String joinHex(double[] xs) {
    StringBuilder sb = new StringBuilder();
    for (int i = 0; i < xs.length; i++) {
      if (i > 0) sb.append(',');
      sb.append(hex(xs[i]));
    }
    return sb.toString();
  }

  private static String joinMatrixHex(double[][] m) {
    StringBuilder sb = new StringBuilder();
    for (int i = 0; i < m.length; i++) {
      if (i > 0) sb.append(';');
      sb.append(joinHex(m[i]));
    }
    return sb.toString();
  }

  private static String joinInts(int[] xs) {
    StringBuilder sb = new StringBuilder();
    for (int i = 0; i < xs.length; i++) {
      if (i > 0) sb.append(',');
      sb.append(xs[i]);
    }
    return sb.toString();
  }

  private static String sha256Hex(byte[] bytes) throws Exception {
    MessageDigest md = MessageDigest.getInstance("SHA-256");
    return HexFormat.of().formatHex(md.digest(bytes));
  }

  private static void emitBilstmModel(
      PrintWriter w, String modelId, BilstmPOSModel model, Path binPath, List<String[]> sentences)
      throws Exception {
    ByteArrayOutputStream baos = new ByteArrayOutputStream();
    model.serialize(baos);
    byte[] bytes = baos.toByteArray();
    Files.write(binPath, bytes);
    w.printf("bilstm_meta\t%s\tsha256\t%s\t%d%n", modelId, sha256Hex(bytes), bytes.length);
    w.printf("bilstm_meta\t%s\ttags\t%s%n", modelId, String.join(",", model.tags()));
    w.printf("bilstm_meta\t%s\tis_crf\t%b%n", modelId, model.isCrf());

    if (model.isCrf()) {
      w.printf("bilstm_crf_params\t%s\tstart\t%s%n", modelId, joinHex(model.startWeights()));
      w.printf("bilstm_crf_params\t%s\tend\t%s%n", modelId, joinHex(model.endWeights()));
      w.printf(
          "bilstm_crf_params\t%s\ttransitions\t%s%n",
          modelId, joinMatrixHex(model.transitionWeights()));
    }

    CrfScorer crf = new CrfScorer(model.tags().length);
    for (int sIdx = 0; sIdx < sentences.size(); sIdx++) {
      String[] sent = sentences.get(sIdx);
      for (int t = 0; t < sent.length; t++) {
        double[] rep = model.wordRepresentation(sent[t]);
        w.printf("bilstm_rep\t%s\t%d\t%d\t%s%n", modelId, sIdx, t, joinHex(rep));
      }
      double[][] emissions = model.score(sent);
      w.printf("bilstm_emissions\t%s\t%d\t%s%n", modelId, sIdx, joinMatrixHex(emissions));
      if (model.isCrf()) {
        int[] viterbi =
            crf.viterbi(
                emissions, model.transitionWeights(), model.startWeights(), model.endWeights());
        double[][] marginals =
            crf.marginals(
                emissions, model.transitionWeights(), model.startWeights(), model.endWeights());
        w.printf("bilstm_viterbi\t%s\t%d\t%s%n", modelId, sIdx, joinInts(viterbi));
        w.printf("bilstm_marginals\t%s\t%d\t%s%n", modelId, sIdx, joinMatrixHex(marginals));
      }
    }

    BilstmPOSTagger tagger = new BilstmPOSTagger(model);
    for (int sIdx = 0; sIdx < sentences.size(); sIdx++) {
      String[] sent = sentences.get(sIdx);
      String[] tags = tagger.tag(sent);
      Sequence seq = tagger.topKSequences(sent)[0];
      w.printf(
          "bilstm_tagger\t%s\t%d\t%s\t%s\t%s\t%s%n",
          modelId,
          sIdx,
          String.join(",", tags),
          String.join(",", seq.getOutcomes()),
          joinHex(seq.getProbs()),
          hex(seq.getScore()));
    }
  }

  private static void emitFeedforwardModel(
      PrintWriter w,
      String modelId,
      FeedforwardPOSModel model,
      Path binPath,
      List<String[]> sentences)
      throws Exception {
    ByteArrayOutputStream baos = new ByteArrayOutputStream();
    model.serialize(baos);
    byte[] bytes = baos.toByteArray();
    Files.write(binPath, bytes);
    w.printf("ff_meta\t%s\tsha256\t%s\t%d%n", modelId, sha256Hex(bytes), bytes.length);
    w.printf("ff_meta\t%s\ttags\t%s%n", modelId, String.join(",", model.tags()));
    w.printf("ff_meta\t%s\tpretrained\t%b%n", modelId, model.usesPretrainedVectors());

    // First score every position without the contribution cache enabled.
    String[] modelTags = model.tags();
    for (int sIdx = 0; sIdx < sentences.size(); sIdx++) {
      String[] sent = sentences.get(sIdx);
      String prev = null;
      String prevPrev = null;
      for (int i = 0; i < sent.length; i++) {
        String[] ctx = FeedforwardPOSContext.extract(sent, i, prev, prevPrev);
        int[] fIds = model.featureIds(ctx);
        int[] pRows = model.pretrainedRows(sent, i);
        double[] scores = model.score(fIds, pRows);
        int best = 0;
        for (int o = 1; o < scores.length; o++) {
          if (scores[o] > scores[best]) {
            best = o;
          }
        }
        w.printf(
            "ff_step_uncached\t%s\t%d\t%d\t%s\t%s\t%s%n",
            modelId,
            sIdx,
            i,
            joinInts(fIds),
            pRows == null ? "-" : joinInts(pRows),
            joinHex(scores));
        prevPrev = prev;
        prev = modelTags[best];
      }
    }

    // Now construct FeedforwardPOSTagger (which calls model.enableScoringCache())
    // and verify cached scores and tagger output.
    FeedforwardPOSTagger tagger = new FeedforwardPOSTagger(model);
    for (int sIdx = 0; sIdx < sentences.size(); sIdx++) {
      String[] sent = sentences.get(sIdx);
      String prev = null;
      String prevPrev = null;
      for (int i = 0; i < sent.length; i++) {
        String[] ctx = FeedforwardPOSContext.extract(sent, i, prev, prevPrev);
        int[] fIds = model.featureIds(ctx);
        int[] pRows = model.pretrainedRows(sent, i);
        double[] scores = model.score(fIds, pRows);
        int best = 0;
        for (int o = 1; o < scores.length; o++) {
          if (scores[o] > scores[best]) {
            best = o;
          }
        }
        w.printf("ff_step_cached\t%s\t%d\t%d\t%s%n", modelId, sIdx, i, joinHex(scores));
        prevPrev = prev;
        prev = modelTags[best];
      }
      String[] tags = tagger.tag(sent);
      Sequence seq = tagger.topKSequences(sent)[0];
      w.printf(
          "ff_tagger\t%s\t%d\t%s\t%s\t%s\t%s%n",
          modelId,
          sIdx,
          String.join(",", tags),
          String.join(",", seq.getOutcomes()),
          joinHex(seq.getProbs()),
          hex(seq.getScore()));
    }
  }

  public static void main(String[] args) throws Exception {
    if (args.length != 1) {
      System.err.println("usage: java opennlp.tools.postag.NeuralPosVectors <out-dir>");
      System.exit(2);
    }
    Path outDir = Path.of(args[0]);
    Files.createDirectories(outDir);

    List<POSSample> corpus = trainingCorpus();
    Map<String, float[]> pretrained = pretrainedTable();
    List<String> pretrainedVocab = new ArrayList<>(pretrained.keySet());
    List<String[]> evalSentences = evaluationSentences();

    BilstmPOSTrainer.Settings bilstmCrfSettings =
        new BilstmPOSTrainer.Settings(
            6, 4, 3, 6, 3, 2, 0.05, 5.0, 0.1, 1, 12, 42L, 1, 0.1, 2, true, 2, 0.1, 0.1, 0.2, 0.0,
            true);
    BilstmPOSModel bilstmCrf =
        BilstmPOSTrainer.train(
            new CollectionObjectStream<>(corpus),
            bilstmCrfSettings,
            k -> pretrained.get(k.toString()),
            pretrainedVocab);

    BilstmPOSTrainer.Settings bilstmSoftmaxSettings =
        new BilstmPOSTrainer.Settings(
            6, 4, 3, 6, 3, 2, 0.05, 5.0, 0.0, 1, 12, 77L, 1, 0.0, 0, false, 1, 0.0, 0.0, 0.0, 0.0,
            false);
    BilstmPOSModel bilstmSoftmax =
        BilstmPOSTrainer.train(new CollectionObjectStream<>(corpus), bilstmSoftmaxSettings);

    FeedforwardPOSTrainer.Settings ffV1Settings =
        new FeedforwardPOSTrainer.Settings(6, 8, 3, 4, 0.05, 1e-5, 0.1, 1, 1, 42L);
    FeedforwardPOSModel ffV1 =
        FeedforwardPOSTrainer.train(new CollectionObjectStream<>(corpus), ffV1Settings);

    FeedforwardPOSTrainer.Settings ffV2Settings =
        new FeedforwardPOSTrainer.Settings(6, 8, 3, 4, 0.05, 1e-5, 0.0, 1, 1, 99L);
    FeedforwardPOSModel ffV2 =
        FeedforwardPOSTrainer.train(
            new CollectionObjectStream<>(corpus),
            ffV2Settings,
            k -> pretrained.get(k.toString()),
            pretrainedVocab);

    Path tsvFile = outDir.resolve("neural_postag.tsv");
    try (PrintWriter w =
        new PrintWriter(Files.newBufferedWriter(tsvFile, StandardCharsets.US_ASCII))) {
      w.printf(
          "# generator=NeuralPosVectors.java\topennlp.commit=%s\tjava.version=%s\tjava.vm.version=%s%n",
          OPENNLP_COMMIT,
          System.getProperty("java.version"),
          System.getProperty("java.vm.version"));

      for (long seed : new long[] {0L, 1L, 42L, 77L, 99L, -123456789L}) {
        Random rng = new Random(seed);
        double[] gs = new double[8];
        for (int i = 0; i < gs.length; i++) {
          gs[i] = rng.nextGaussian();
        }
        w.printf("rng_gaussian\t%d\t%s%n", seed, joinHex(gs));
      }

      emitBilstmModel(w, "bilstm_crf", bilstmCrf, outDir.resolve("bilstm_crf.bin"), evalSentences);
      emitBilstmModel(
          w, "bilstm_softmax", bilstmSoftmax, outDir.resolve("bilstm_softmax.bin"), evalSentences);
      emitFeedforwardModel(
          w, "feedforward_v1", ffV1, outDir.resolve("feedforward_v1.bin"), evalSentences);
      emitFeedforwardModel(
          w, "feedforward_v2", ffV2, outDir.resolve("feedforward_v2.bin"), evalSentences);
    }
  }
}
