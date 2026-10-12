import opennlp.tools.tokenize.TokenizerModel;

/** Header lines that identify the model a golden file was made with. */
final class ModelHeader {

  private ModelHeader() {
  }

  /** The file name and SHA-256, for any model. */
  static void write(GoldenFile file, ModelFile model) {
    file.header("model", model.fileName());
    file.header("model-sha256", model.sha256());
  }

  /**
   * The model lines plus the alphanumeric settings TokenizerME reads from the manifest. The pattern
   * is the manifest value as stored, escaped; it is empty when the manifest has none.
   */
  static void write(GoldenFile file, ModelFile model, TokenizerModel tokenizerModel) {
    write(file, model);
    String pattern = tokenizerModel.getManifestProperty("alphaNumericPattern");
    file.header("alpha-numeric-pattern", pattern == null ? "" : TextEscape.escape(pattern));
    file.header("use-alpha-numeric-optimization",
        Boolean.toString(tokenizerModel.getFactory().isUseAlphaNumericOptimization()));
  }
}
