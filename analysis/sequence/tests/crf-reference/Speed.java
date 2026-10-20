package opennlp.tools.postag;

import java.util.Arrays;

/** Runs the unmodified OpenNLP scorer over the Rust benchmark's scores. */
public final class Speed {
  private static volatile double sink;
  public static void main(String[] args) {
    int repeats = args.length == 0 ? 1000 : Integer.parseInt(args[0]);
    double[][] emissions = new double[32][9];
    double[][] transitions = new double[9][9];
    double[] start = new double[9];
    double[] end = new double[9];
    for (int t = 0; t < 32; t++) {
      for (int k = 0; k < 9; k++) emissions[t][k] = Math.sin((t * 9 + k + 1) * 0.37);
    }
    for (int k = 0; k < 9; k++) {
      start[k] = k * 0.03;
      end[k] = -k * 0.02;
      for (int j = 0; j < 9; j++) transitions[k][j] = Math.cos((k * 9 + j + 1) * 0.21) * 0.2;
    }
    CrfScorer scorer = new CrfScorer(9);
    System.out.println("path=" + Arrays.toString(scorer.viterbi(emissions, transitions, start, end)));
    System.out.println("marginal=" + Arrays.toString(scorer.marginals(emissions, transitions, start, end)[0]));
    for (int i = 0; i < 200; i++) {
      sink = scorer.viterbi(emissions, transitions, start, end)[0]
          + scorer.marginals(emissions, transitions, start, end)[0][0];
    }
    long time = System.nanoTime();
    for (int i = 0; i < repeats; i++) {
      sink = scorer.viterbi(emissions, transitions, start, end)[0]
          + scorer.marginals(emissions, transitions, start, end)[0][0];
    }
    System.out.printf(java.util.Locale.ROOT,"repeats=%d seconds=%.9f%n",repeats,(System.nanoTime()-time)/1e9);
  }
}
