import java.io.IOException;
import java.io.OutputStream;
import java.io.PrintWriter;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.attribute.FileTime;
import java.util.zip.CRC32;
import java.util.zip.ZipEntry;
import java.util.zip.ZipOutputStream;

/**
 * Writes zip archives with java.util.zip.ZipOutputStream: one with deflated
 * entries (which ZipOutputStream follows with data descriptors) and an
 * archive comment, and one mixing stored and deflated entries. An index
 * lists every entry's method, size and CRC-32.
 */
public final class MakeZipFixtures {
  public static void main(String[] args) throws IOException {
    Path dir = Path.of(args[0], "zip");
    Files.createDirectories(dir);
    byte[] text = Inputs.text(4, 3000);
    byte[] random = Inputs.random(5, 700);
    byte[] small = "hello zip\n".getBytes(StandardCharsets.UTF_8);
    try (PrintWriter index = new PrintWriter(Files.newBufferedWriter(dir.resolve("index.txt"),
        StandardCharsets.UTF_8))) {
      index.println("# archive entry method size crc32");
      try (OutputStream os = Files.newOutputStream(dir.resolve("deflated.zip"));
           ZipOutputStream zip = new ZipOutputStream(os)) {
        zip.setComment("an archive comment");
        put(zip, index, "deflated.zip", "text.txt", text, ZipEntry.DEFLATED, false);
        put(zip, index, "deflated.zip", "dir/random.bin", random, ZipEntry.DEFLATED, true);
        put(zip, index, "deflated.zip", "empty.txt", new byte[0], ZipEntry.DEFLATED, false);
        put(zip, index, "deflated.zip", "näme.txt", small, ZipEntry.DEFLATED, false);
      }
      try (OutputStream os = Files.newOutputStream(dir.resolve("stored.zip"));
           ZipOutputStream zip = new ZipOutputStream(os)) {
        put(zip, index, "stored.zip", "small.txt", small, ZipEntry.STORED, false);
        put(zip, index, "stored.zip", "text.txt", text, ZipEntry.DEFLATED, false);
        put(zip, index, "stored.zip", "random.bin", random, ZipEntry.STORED, true);
      }
    }
  }

  /**
   * Adds an entry. With extendedTime the entry gets a modification time
   * beyond what DOS time holds, so ZipOutputStream writes an extra field.
   */
  static void put(ZipOutputStream zip, PrintWriter index, String archive, String name, byte[] data, int method,
      boolean extendedTime) throws IOException {
    ZipEntry e = new ZipEntry(name);
    e.setMethod(method);
    CRC32 crc = new CRC32();
    crc.update(data);
    if (method == ZipEntry.STORED) {
      e.setSize(data.length);
      e.setCompressedSize(data.length);
      e.setCrc(crc.getValue());
    }
    if (extendedTime) {
      e.setLastModifiedTime(FileTime.fromMillis(4102444800000L));
    }
    zip.putNextEntry(e);
    zip.write(data);
    zip.closeEntry();
    index.printf("%s %s %s %d %08x%n", archive, name, method == ZipEntry.STORED ? "stored" : "deflated",
        data.length, crc.getValue());
  }
}
