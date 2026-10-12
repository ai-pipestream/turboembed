package ai.pipestream.turbo.goldens;

/**
 * One call an annotator made to its maxent model's {@code eval}: where in the input the decision
 * is, the contexts the model was given, their feature values when the call carried any, and the
 * probability of each outcome the model returned, in the model's outcome index order.
 *
 * @param position the UTF-8 byte offset into the input the decision is about, or 0 for a decision
 *     about the whole document
 * @param unit 0 when the UTF-16 index OpenNLP decided at starts a code point; 1 when it lies
 *     between the two halves of a surrogate pair, in which case {@code position} is the byte offset
 *     of that pair's code point
 * @param contexts the context strings, in the order the model received them
 * @param values the feature value of each context, or {@code null} when {@code eval} was called
 *     without values
 * @param outcomes the probability of each outcome, in the model's outcome index order
 */
record DecisionPoint(long position, int unit, String[] contexts, float[] values, double[] outcomes) {

  DecisionPoint {
    if (unit != 0 && unit != 1) {
      throw new IllegalArgumentException("unit must be 0 or 1: " + unit);
    }
    if (values != null && values.length != contexts.length) {
      throw new IllegalArgumentException(values.length + " values for " + contexts.length + " contexts");
    }
  }

  /** A decision at a UTF-16 index of the input, converted to its UTF-8 byte offset. */
  static DecisionPoint at(Utf8Offsets offsets, int utf16Index, Evaluation e) {
    Utf8Offsets.Position p = offsets.position(utf16Index);
    return new DecisionPoint(p.offset(), p.unit(), e.contexts(), e.values(), e.outcomes());
  }

  /** A decision about the whole document, at position 0. */
  static DecisionPoint document(Evaluation e) {
    return new DecisionPoint(0, 0, e.contexts(), e.values(), e.outcomes());
  }
}
