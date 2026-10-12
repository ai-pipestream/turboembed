import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import opennlp.tools.util.Span;

/**
 * Builds the text of one golden file: the header lines, the escaped input and one line per span.
 *
 * <p>Every line ends with a line feed. See the README for the format.
 */
final class GoldenFile {

  private final StringBuilder out = new StringBuilder();
  private final String input;
  private final Utf8Offsets offsets;

  /**
   * Starts a file with the common header.
   *
   * @param component the component name written on the {@code component:} line
   * @param run the values shared by every file of this run (OpenNLP commit, JDK, whitespace mode)
   * @param input the exact input text
   */
  GoldenFile(String component, RunInfo run, String input) {
    this.input = input;
    this.offsets = new Utf8Offsets(input);
    line("component: " + component);
    line("opennlp: " + run.opennlpCommit());
    line("jdk: " + run.jdkVersion());
    line("whitespace: " + run.whitespace());
  }

  /** Adds a header line such as {@code model: name}. Call before {@link #input()}. */
  GoldenFile header(String key, String value) {
    return line(key + ": " + value);
  }

  /** Writes the {@code input:} line. */
  GoldenFile input() {
    return line("input: " + TextEscape.escape(input));
  }

  /**
   * Writes one span line.
   *
   * @param span the span in UTF-16 offsets
   * @param probability the span's probability, or {@code null} when the component has none
   * @param type the span's type name, or {@code null} when the component has none
   */
  GoldenFile span(Span span, Double probability, String type) {
    int s = span.getStart();
    int e = span.getEnd();
    StringBuilder sb = new StringBuilder();
    if (offsets.isBoundary(s) && offsets.isBoundary(e)) {
      sb.append("span: ").append(offsets.utf8(s)).append(' ').append(offsets.utf8(e)).append(' ');
    } else {
      sb.append("span-invalid: ");
    }
    sb.append(s).append(' ').append(e);
    if (probability != null) {
      sb.append(' ').append(Double.toString(probability));
    }
    if (type != null) {
      sb.append(' ').append(type);
    }
    return line(sb.toString());
  }

  /** Writes an {@code error:} line for a run that threw. */
  GoldenFile error(Throwable t) {
    return line("error: " + TextEscape.escape(t.getClass().getName() + ": " + t.getMessage()));
  }

  GoldenFile line(String text) {
    out.append(text).append('\n');
    return this;
  }

  void write(Path file) throws IOException {
    Files.createDirectories(file.getParent());
    Files.write(file, out.toString().getBytes(StandardCharsets.UTF_8));
  }
}
