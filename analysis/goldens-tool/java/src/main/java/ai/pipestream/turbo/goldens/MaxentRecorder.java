package ai.pipestream.turbo.goldens;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import opennlp.tools.ml.model.AbstractModel;
import opennlp.tools.ml.model.Context;
import opennlp.tools.ml.model.MaxentModel;

/**
 * A maxent model that forwards every call to a loaded model and records each {@code eval} call:
 * the contexts and feature values passed in and the probabilities returned. The arrays are
 * copied, because the annotators and the models reuse their buffers.
 *
 * <p>It extends {@link AbstractModel} rather than only implementing {@link MaxentModel} because
 * {@code TokenizerModel} and {@code LanguageDetectorModel} refuse a maxent model that is not an
 * {@code AbstractModel}. It is built over the loaded model's own parameter table, predicate map
 * and outcome names (the arrays and map themselves, not copies), so the methods
 * {@code AbstractModel} makes final ({@code getBestOutcome}, {@code getAllOutcomes},
 * {@code getOutcome}) read exactly what the loaded model's would; every other method forwards.</p>
 *
 * <p>A recording context generator reports each request it answers with
 * {@link #contextRequested}; the next {@code eval} must then receive the very array the generator
 * returned, and the recorded call carries the request. Without such a generator a recorded call
 * carries no request.</p>
 */
final class MaxentRecorder extends AbstractModel {

  private final AbstractModel delegate;
  private final List<Evaluation> evaluations = new ArrayList<>();
  private ContextRequest pendingRequest;
  private String[] pendingContext;

  private MaxentRecorder(AbstractModel delegate, Object[] data) {
    super((Context[]) data[0], null, pmap(data[1]), (String[]) data[2]);
    this.delegate = delegate;
    this.modelType = delegate.getModelType();
  }

  /** Wraps {@code model}, which must be an {@link AbstractModel} as every loaded OpenNLP model is. */
  static MaxentRecorder wrap(MaxentModel model) {
    if (!(model instanceof AbstractModel m)) {
      throw new IllegalArgumentException("the maxent model " + model.getClass().getName()
          + " is not an AbstractModel");
    }
    return new MaxentRecorder(m, m.getDataStructures());
  }

  @SuppressWarnings("unchecked")
  private static Map<String, Context> pmap(Object o) {
    return (Map<String, Context>) o;
  }

  @Override
  public double[] eval(String[] context) {
    return record(context, null, delegate.eval(context));
  }

  @Override
  public double[] eval(String[] context, double[] probs) {
    return record(context, null, delegate.eval(context, probs));
  }

  @Override
  public double[] eval(String[] context, float[] values) {
    return record(context, values, delegate.eval(context, values));
  }

  @Override
  public int getIndex(String outcome) {
    return delegate.getIndex(outcome);
  }

  @Override
  public int getNumOutcomes() {
    return delegate.getNumOutcomes();
  }

  /** The outcome labels in the model's outcome index order. */
  String[] outcomeLabels() {
    String[] labels = new String[getNumOutcomes()];
    for (int i = 0; i < labels.length; i++) {
      labels[i] = delegate.getOutcome(i);
    }
    return labels;
  }

  /**
   * Notes that a context generator answered {@code request} with {@code context}, which the next
   * {@code eval} call must receive. Fails when an earlier request has not been evaluated yet.
   */
  void contextRequested(ContextRequest request, String[] context) {
    if (pendingRequest != null) {
      throw new IllegalStateException("context asked for at index " + request.index()
          + " before the context for index " + pendingRequest.index() + " was evaluated");
    }
    pendingRequest = request;
    pendingContext = context;
  }

  private double[] record(String[] context, float[] values, double[] outcomes) {
    if (pendingRequest != null && context != pendingContext) {
      throw new IllegalStateException("eval received a context array other than the one the context"
          + " generator returned for index " + pendingRequest.index());
    }
    evaluations.add(new Evaluation(pendingRequest, context.clone(), values == null ? null : values.clone(),
        outcomes.clone()));
    pendingRequest = null;
    pendingContext = null;
    return outcomes;
  }

  /** Returns the calls recorded since the last call and forgets them. */
  List<Evaluation> take() {
    if (pendingRequest != null) {
      throw new IllegalStateException("the context for index " + pendingRequest.index() + " was never evaluated");
    }
    List<Evaluation> out = List.copyOf(evaluations);
    evaluations.clear();
    return out;
  }

  /**
   * Forgets every call recorded since the last {@link #take}, and a context requested but not
   * evaluated: the annotator threw, so the run has no dump.
   */
  void discard() {
    evaluations.clear();
    pendingRequest = null;
    pendingContext = null;
  }

  // Identity, as two recorders are never interchangeable; the model classes use these for their own
  // equality only.
  @Override
  public boolean equals(Object o) {
    return o == this;
  }

  @Override
  public int hashCode() {
    return System.identityHashCode(this);
  }
}
