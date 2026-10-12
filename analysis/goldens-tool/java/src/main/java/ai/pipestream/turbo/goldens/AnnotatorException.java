package ai.pipestream.turbo.goldens;

import java.util.function.Supplier;

/**
 * An exception the OpenNLP annotator itself threw on an input. The generator records it in the
 * golden's {@code error} record instead of failing, because what OpenNLP does on that input is
 * the thing to record. Only the call into the annotator is wrapped ({@link #call}); an exception
 * anywhere else (reading a file, loading a model, one of the generator's own checks) is not
 * this one and stops the generator.
 */
final class AnnotatorException extends Exception {

  private static final long serialVersionUID = 1L;

  AnnotatorException(RuntimeException cause) {
    super(cause);
  }

  /** The exception the annotator threw. */
  @Override
  public synchronized RuntimeException getCause() {
    return (RuntimeException) super.getCause();
  }

  /** The thrown exception's fully qualified class, as the {@code error} record writes it. */
  String exceptionClass() {
    return getCause().getClass().getName();
  }

  /** What the thrown exception's {@code getMessage()} returned, or empty when it returned null. */
  String exceptionMessage() {
    String m = getCause().getMessage();
    return m == null ? "" : m;
  }

  /** Calls into the annotator, turning a runtime exception it throws into this one. */
  static <T> T call(Supplier<T> annotator) throws AnnotatorException {
    try {
      return annotator.get();
    } catch (RuntimeException e) {
      throw new AnnotatorException(e);
    }
  }
}
