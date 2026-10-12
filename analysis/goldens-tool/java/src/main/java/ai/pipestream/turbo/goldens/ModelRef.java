package ai.pipestream.turbo.goldens;

/**
 * A file an annotator loaded, a model or a dictionary: its file name and the SHA-512 of its
 * bytes, lowercase hex.
 */
record ModelRef(String fileName, String sha512) {

  /** The file name extensions {@link #baseName()} removes: a model's and a dictionary's. */
  private static final String[] EXTENSIONS = {".bin", ".xml"};

  /** The file's name in a golden's path: the file name without {@code .bin} or {@code .xml}. */
  String baseName() {
    for (String e : EXTENSIONS) {
      if (fileName.endsWith(e)) {
        return fileName.substring(0, fileName.length() - e.length());
      }
    }
    return fileName;
  }
}
