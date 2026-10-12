package ai.pipestream.turbo.goldens;

import java.util.List;
import opennlp.tools.util.Span;

/**
 * The text anchored layer of the spans an annotator returned, and with it, when a span cuts a
 * character, the layer of their exact UTF-16 spans.
 *
 * <p>The tokenizer classifies UTF-16 units, so it can put the boundary between two tokens
 * between the halves of a surrogate pair. A UTF-8 byte offset cannot fall there, so the row of
 * such a span holds the enclosing character's bytes ({@link Utf8Offsets#enclosing}), and the two
 * halves of the character are two rows over the same bytes. The aligned layer
 * {@code turbo:utf16<L>} then keeps what the JVM returned: for every row, the UTF-16 span
 * ({@code start16}, {@code end16}) and {@code cut}, 1 for a row that starts or ends inside a
 * surrogate pair and 0 otherwise. It is written only when at least one row cuts a character, so
 * the goldens of every other input are what they were without it.</p>
 */
final class SpanLayers {

  private SpanLayers() {}

  /**
   * The layer {@code id}, anchored on the text, with one row per span, followed by
   * {@code turbo:utf16<id>} when any span cuts a character.
   */
  static List<Layer> of(String id, Utf8Offsets offsets, Span[] spans) {
    int n = spans.length;
    long[] starts = new long[n];
    long[] ends = new long[n];
    long[] starts16 = new long[n];
    long[] ends16 = new long[n];
    long[] cuts = new long[n];
    boolean anyCut = false;
    for (int i = 0; i < n; i++) {
      Utf8Offsets.Enclosing b = offsets.enclosing(spans[i].getStart(), spans[i].getEnd());
      starts[i] = b.start();
      ends[i] = b.end();
      starts16[i] = spans[i].getStart();
      ends16[i] = spans[i].getEnd();
      cuts[i] = b.cut() ? 1 : 0;
      anyCut |= b.cut();
    }
    Layer bytes = new Layer(id, "text", new Column.U32("start", starts), new Column.U32("end", ends));
    if (!anyCut) {
      return List.of(bytes);
    }
    return List.of(bytes, new Layer(utf16Id(id), "aligned:" + id, new Column.U32("start16", starts16),
        new Column.U32("end16", ends16), new Column.U32("cut", cuts)));
  }

  /** {@code turbo:utf16<id>}, the id of the UTF-16 spans of the layer {@code id}. */
  static String utf16Id(String id) {
    return "turbo:utf16<" + id + ">";
  }
}
