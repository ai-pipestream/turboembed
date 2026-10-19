package ai.pipestream.turbo.goldens;

import java.util.Map;
import opennlp.tools.util.WhitespaceMode;

/**
 * OpenNLP's whitespace mode for one call. {@link WhitespaceMode} is a setting of the whole
 * process that {@code StringUtil.isWhitespace} reads on every call, so an annotator run in the
 * legacy mode sets it for the call and puts the previous mode back afterwards.
 */
final class Whitespace {

  /** The {@code config} key of the mode; only the legacy mode, which is not the default, is written. */
  static final String CONFIG_KEY = "whitespace";

  private Whitespace() {}

  /** A call into an annotator, which may throw what the annotator threw. */
  interface Call<T> {
    T get() throws AnnotatorException;
  }

  /** Calls {@code annotator} with {@code mode} active. */
  static <T> T in(WhitespaceMode mode, Call<T> annotator) throws AnnotatorException {
    WhitespaceMode previous = WhitespaceMode.current();
    WhitespaceMode.setActive(mode);
    try {
      return annotator.get();
    } finally {
      WhitespaceMode.setActive(previous);
    }
  }

  /** The {@code config} record of a run in {@code mode}: none for the default, the unicode mode. */
  static Map<String, String> config(WhitespaceMode mode) {
    return mode == WhitespaceMode.UNICODE ? Map.of() : Map.of(CONFIG_KEY, name(mode));
  }

  /** The suffix of a run's directory in {@code mode}: none for the unicode mode, {@code +legacy} otherwise. */
  static String dirSuffix(WhitespaceMode mode) {
    return mode == WhitespaceMode.UNICODE ? "" : "+" + name(mode);
  }

  private static String name(WhitespaceMode mode) {
    return mode == WhitespaceMode.LEGACY ? "legacy" : "unicode";
  }
}
