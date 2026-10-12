package ai.pipestream.turbo.goldens;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.nio.charset.StandardCharsets;
import org.junit.jupiter.api.Test;

class Utf8OffsetsTest {

  @Test
  void asciiOffsetsAreTheIndices() {
    Utf8Offsets o = new Utf8Offsets("t/a", "Hello world");
    assertEquals(new Utf8Offsets.ByteSpan(6, 11), o.span(6, 11));
    assertEquals(11, o.utf8Length());
  }

  @Test
  void offsetsCountEveryUtf8Length() {
    // e acute is 2 bytes, the CJK character 3, the emoji 4 bytes and two UTF-16 units.
    String text = "é 中 😀 x";
    Utf8Offsets o = new Utf8Offsets("t/b", text);
    byte[] bytes = text.getBytes(StandardCharsets.UTF_8);
    assertEquals(bytes.length, o.utf8Length());
    int emoji = text.indexOf('\ud83d');
    Utf8Offsets.ByteSpan b = o.span(emoji, emoji + 2);
    assertEquals(7, b.start());
    assertEquals(11, b.end());
    assertEquals("😀", new String(bytes, b.start(), b.end() - b.start(), StandardCharsets.UTF_8));
    Utf8Offsets.ByteSpan x = o.span(text.length() - 1, text.length());
    assertEquals("x", new String(bytes, x.start(), x.end() - x.start(), StandardCharsets.UTF_8));
  }

  @Test
  void anEndInsideASurrogatePairFailsWithTheInputAndTheSpan() {
    Utf8Offsets o = new Utf8Offsets("multi/mixed", "a😀b");
    IllegalArgumentException e = assertThrows(IllegalArgumentException.class, () -> o.span(0, 2));
    assertTrue(e.getMessage().contains("multi/mixed"), e.getMessage());
    assertTrue(e.getMessage().contains("[0, 2)"), e.getMessage());
    assertThrows(IllegalArgumentException.class, () -> o.span(2, 4));
  }

  @Test
  void aLoneSurrogateIsRefused() {
    assertThrows(IllegalArgumentException.class, () -> new Utf8Offsets("t/c", "a\ud83db"));
  }

  @Test
  void aSpanOutsideTheTextIsRefused() {
    Utf8Offsets o = new Utf8Offsets("t/d", "abc");
    assertThrows(IllegalArgumentException.class, () -> o.span(1, 4));
    assertThrows(IllegalArgumentException.class, () -> o.span(2, 1));
  }
}
