import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.PrintWriter;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.zip.CRC32;
import java.util.zip.Deflater;

/**
 * Writes raw deflate streams (no zlib header) made by java.util.zip.Deflater
 * at every level, and an index of what each one must inflate to.
 */
public final class MakeInflateFixtures {
  public static void main(String[] args) throws IOException {
    Path dir = Path.of(args[0], "inflate");
    Files.createDirectories(dir);
    try (PrintWriter index = new PrintWriter(Files.newBufferedWriter(dir.resolve("index.txt"),
        StandardCharsets.US_ASCII))) {
      index.println("# file input level strategy flush size crc32");
      String[] names = {"empty", "short", "text", "runs", "random", "far"};
      for (String name : names) {
        byte[] input = input(name);
        for (int level = 0; level <= 9; level++) {
          if (name.equals("far") && level == 0) {
            continue; // 32 KB stored; nothing a level 0 stream of "text" does not cover
          }
          write(dir, index, name, input, level, Deflater.DEFAULT_STRATEGY, 0);
        }
      }
      byte[] text = input("text");
      write(dir, index, "text", text, 6, Deflater.HUFFMAN_ONLY, 0);
      write(dir, index, "text", text, 6, Deflater.FILTERED, 0);
      write(dir, index, "text", text, 6, Deflater.DEFAULT_STRATEGY, 1000);
      write(dir, index, "text", text, 0, Deflater.DEFAULT_STRATEGY, 1000);
    }
  }

  static byte[] input(String name) {
    return switch (name) {
      case "empty" -> new byte[0];
      case "short" -> Inputs.shortText();
      case "text" -> Inputs.text(1, 6000);
      case "runs" -> Inputs.runs(2);
      case "random" -> Inputs.random(3, 500);
      case "far" -> Inputs.far();
      default -> throw new IllegalArgumentException(name);
    };
  }

  /**
   * Deflates input; with flushEvery above zero, a SYNC_FLUSH after every
   * that many bytes, which ends each piece with an empty stored block.
   */
  static void write(Path dir, PrintWriter index, String name, byte[] input, int level, int strategy,
      int flushEvery) throws IOException {
    Deflater d = new Deflater(level, true);
    d.setStrategy(strategy);
    ByteArrayOutputStream out = new ByteArrayOutputStream();
    byte[] buf = new byte[65536];
    int step = flushEvery > 0 ? flushEvery : Math.max(input.length, 1);
    for (int off = 0; off < input.length; off += step) {
      d.setInput(input, off, Math.min(step, input.length - off));
      int n;
      while ((n = d.deflate(buf, 0, buf.length, flushEvery > 0 ? Deflater.SYNC_FLUSH : Deflater.NO_FLUSH)) > 0) {
        out.write(buf, 0, n);
      }
    }
    d.finish();
    while (!d.finished()) {
      int n = d.deflate(buf);
      out.write(buf, 0, n);
    }
    d.end();
    String strategyName = switch (strategy) {
      case Deflater.HUFFMAN_ONLY -> "huffman";
      case Deflater.FILTERED -> "filtered";
      default -> "default";
    };
    String file = name + "-" + level + "-" + strategyName + (flushEvery > 0 ? "-sync" : "") + ".deflate";
    Files.write(dir.resolve(file), out.toByteArray());
    CRC32 crc = new CRC32();
    crc.update(input);
    index.printf("%s %s %d %s %d %d %08x%n", file, name, level, strategyName, flushEvery, input.length,
        crc.getValue());
  }
}
