import java.io.ByteArrayOutputStream;
import java.nio.charset.StandardCharsets;

/**
 * Deterministic test inputs. tests/inflate.rs builds the same bytes with
 * the same generator, so only the compressed streams are checked in.
 */
final class Inputs {
  private long state;

  Inputs(long seed) {
    state = seed;
  }

  /** A 64-bit linear congruential generator; the top 31 bits are used. */
  int next() {
    state = state * 6364136223846793005L + 1442695040888963407L;
    return (int) (state >>> 33);
  }

  static final String[] WORDS = {"the", "model", "reads", "a", "zip", "entry", "and", "inflates", "its",
      "bytes", "into", "memory", "while", "tokens", "sentences", "offsets", "stay", "exact", "."};

  static byte[] text(long seed, int size) {
    Inputs r = new Inputs(seed);
    StringBuilder sb = new StringBuilder();
    while (sb.length() < size) {
      sb.append(WORDS[r.next() % WORDS.length]).append(' ');
    }
    return sb.substring(0, size).getBytes(StandardCharsets.US_ASCII);
  }

  static byte[] random(long seed, int size) {
    Inputs r = new Inputs(seed);
    byte[] b = new byte[size];
    for (int i = 0; i < size; i++) {
      b[i] = (byte) r.next();
    }
    return b;
  }

  /** Forty runs of one letter each, of 1 to 300 bytes. */
  static byte[] runs(long seed) {
    Inputs r = new Inputs(seed);
    ByteArrayOutputStream out = new ByteArrayOutputStream();
    for (int i = 0; i < 40; i++) {
      int len = 1 + r.next() % 300;
      for (int j = 0; j < len; j++) {
        out.write('a' + i % 26);
      }
    }
    return out.toByteArray();
  }

  /**
   * A random block, 30000 bytes of letter runs, then the random block
   * again: the repeat can only be coded with a distance above 30000.
   */
  static byte[] far() {
    byte[] block = random(7, 1000);
    byte[] middle = new byte[30000];
    byte[] runs = runs(8);
    for (int i = 0; i < middle.length; i++) {
      middle[i] = runs[i % runs.length];
    }
    byte[] b = new byte[block.length * 2 + middle.length];
    System.arraycopy(block, 0, b, 0, block.length);
    System.arraycopy(middle, 0, b, block.length, middle.length);
    System.arraycopy(block, 0, b, block.length + middle.length, block.length);
    return b;
  }

  static byte[] shortText() {
    return "Hello, hello, hello! Deflate me, deflate me.".getBytes(StandardCharsets.US_ASCII);
  }
}
