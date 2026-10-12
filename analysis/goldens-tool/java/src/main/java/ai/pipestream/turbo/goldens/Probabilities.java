package ai.pipestream.turbo.goldens;

/** Checks that two arrays of probabilities an annotator reported hold the same values, bit for bit. */
final class Probabilities {

  private Probabilities() {}

  /**
   * Fails when {@code reported}, from the annotator's probability accessor, differs from
   * {@code expected} in length or in the raw bits of any value.
   */
  static void checkSame(String inputId, String accessor, double[] expected, double[] reported) {
    if (expected.length != reported.length) {
      throw new IllegalStateException("input " + inputId + ": " + accessor + " returned " + reported.length
          + " values for " + expected.length + " spans");
    }
    for (int i = 0; i < expected.length; i++) {
      if (Double.doubleToRawLongBits(expected[i]) != Double.doubleToRawLongBits(reported[i])) {
        throw new IllegalStateException("input " + inputId + ": " + accessor + " value " + i + " is "
            + reported[i] + ", the span's probability is " + expected[i]);
      }
    }
  }
}
