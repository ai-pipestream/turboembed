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
  void anEnclosingSpanWidensAnEndInsideASurrogatePairToTheCharacter() {
    // a (1 byte), the emoji (4 bytes, UTF-16 indices 1 and 2), b.
    Utf8Offsets o = new Utf8Offsets("t/e", "a\ud83d\ude00b");
    assertEquals(new Utf8Offsets.Enclosing(0, 1, false), o.enclosing(0, 1));
    assertEquals(new Utf8Offsets.Enclosing(1, 5, false), o.enclosing(1, 3));
    // The end between the halves rounds up to the byte after the emoji.
    assertEquals(new Utf8Offsets.Enclosing(0, 5, true), o.enclosing(0, 2));
    // The start between the halves rounds down to the emoji's first byte.
    assertEquals(new Utf8Offsets.Enclosing(1, 6, true), o.enclosing(2, 4));
    // Both ends between the halves: the empty span inside the pair is the whole character.
    assertEquals(new Utf8Offsets.Enclosing(1, 5, true), o.enclosing(2, 2));
    assertThrows(IllegalArgumentException.class, () -> o.enclosing(0, 5));
    assertThrows(IllegalArgumentException.class, () -> o.enclosing(3, 2));
    assertThrows(IllegalArgumentException.class, () -> o.enclosing(-1, 1));
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

  @Test
  void aPositionBetweenTheHalvesOfAPairIsItsCodePointWithUnitOne() {
    // a (1 byte), e acute (2 bytes), the emoji (4 bytes, UTF-16 indices 2 and 3), b.
    Utf8Offsets o = new Utf8Offsets("t/p", "aé😀b");
    assertEquals(new Utf8Offsets.Position(0, 0), o.position(0));
    assertEquals(new Utf8Offsets.Position(1, 0), o.position(1));
    assertEquals(new Utf8Offsets.Position(3, 0), o.position(2));
    assertEquals(new Utf8Offsets.Position(3, 1), o.position(3));
    assertEquals(new Utf8Offsets.Position(7, 0), o.position(4));
    assertEquals(new Utf8Offsets.Position(8, 0), o.position(5));
    assertThrows(IllegalArgumentException.class, () -> o.position(6));
    assertThrows(IllegalArgumentException.class, () -> o.position(-1));
  }
}
