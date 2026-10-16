package opennlp.tools.postag.neural;

/**
 * The package private parts of the addons' neural part-of-speech taggers that the goldens record:
 * the CRF parameters of a BiLSTM model, its Viterbi path and marginals, and the feedforward
 * tagger's context symbols, feature rows, vector rows and tag scores. This class sits in the
 * taggers' package in the generator's jar so it can call them; it computes nothing itself.
 * {@code PostagNeuralGolden} checks that what it records gives the taggers' own tags.
 */
public final class NeuralAccess {

  /** The row of {@link #vectorRows} for a position without a pretrained vector. */
  public static final int NO_VECTOR = FeedforwardPOSModel.NO_VECTOR;

  private NeuralAccess() {}

  /** The CRF transition weights, {@code [from][to]}. */
  public static double[][] transitions(BilstmPOSModel model) {
    return model.transitionWeights();
  }

  /** The CRF start weights, one per tag. */
  public static double[] start(BilstmPOSModel model) {
    return model.startWeights();
  }

  /** The CRF end weights, one per tag. */
  public static double[] end(BilstmPOSModel model) {
    return model.endWeights();
  }

  /** {@code CrfScorer.viterbi} over the model's CRF parameters, as {@code BilstmPOSTagger} calls it. */
  public static int[] viterbi(BilstmPOSModel model, double[][] emissions) {
    return new CrfScorer(model.tags().length).viterbi(emissions, model.transitionWeights(),
        model.startWeights(), model.endWeights());
  }

  /** {@code CrfScorer.marginals} over the model's CRF parameters. */
  public static double[][] marginals(BilstmPOSModel model, double[][] emissions) {
    return new CrfScorer(model.tags().length).marginals(emissions, model.transitionWeights(),
        model.startWeights(), model.endWeights());
  }

  /** {@code FeedforwardPOSContext.extract}: one symbol per slot, {@code null} outside the sentence. */
  public static String[] context(String[] sentence, int index, String previousTag,
      String beforePreviousTag) {
    return FeedforwardPOSContext.extract(sentence, index, previousTag, beforePreviousTag);
  }

  /** {@code FeedforwardPOSModel.featureIds}: the embedding row of each slot's symbol. */
  public static int[] featureIds(FeedforwardPOSModel model, String[] symbols) {
    return model.featureIds(symbols);
  }

  /** {@code FeedforwardPOSModel.pretrainedRows}, or {@code null} for a model without vectors. */
  public static int[] vectorRows(FeedforwardPOSModel model, String[] sentence, int index) {
    return model.pretrainedRows(sentence, index);
  }

  /** {@code FeedforwardPOSModel.score}: one unnormalized score per tag. */
  public static double[] score(FeedforwardPOSModel model, int[] features, int[] vectorRows) {
    return model.score(features, vectorRows);
  }
}
