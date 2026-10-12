package ai.pipestream.turbo.goldens;

/** A model file an annotator loaded: its file name and the SHA-512 of its bytes, lowercase hex. */
record ModelRef(String fileName, String sha512) {

  /** The model's name in a golden's path: the file name without {@code .bin}. */
  String baseName() {
    return fileName.endsWith(".bin") ? fileName.substring(0, fileName.length() - 4) : fileName;
  }
}
