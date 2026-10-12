package ai.pipestream.turbo.goldens;

import java.io.IOException;
import java.nio.file.Path;
import java.util.List;
import java.util.Set;
import opennlp.tools.langdetect.Language;
import opennlp.tools.langdetect.LanguageDetectorME;
import opennlp.tools.langdetect.LanguageDetectorModel;

/**
 * The language detector's golden: {@code opennlp:language} with one row per language
 * {@link LanguageDetectorME#predictLanguages(CharSequence)} returns, in its order, best first.
 */
final class LanguageGolden implements AnnotatorGolden {

  private final ModelRef model;
  private final LanguageDetectorModel loaded;
  private final LanguageDetectorME detector;
  private final Set<String> inputSets;

  private LanguageGolden(ModelRef model, LanguageDetectorModel loaded, Set<String> inputSets) {
    this.model = model;
    this.loaded = loaded;
    this.detector = new LanguageDetectorME(loaded);
    this.inputSets = Set.copyOf(inputSets);
  }

  /** Loads the language detector model from {@code file}, hashing exactly the bytes loaded. */
  static LanguageGolden load(Path file, Set<String> inputSets) throws IOException {
    ModelFile f = ModelFile.read(file);
    return new LanguageGolden(f.ref(), new LanguageDetectorModel(f.stream()), inputSets);
  }

  /** The loaded model, which {@link LanguageDump} builds its recording model from. */
  LanguageDetectorModel loaded() {
    return loaded;
  }

  @Override
  public String annotator() {
    return "langdetect";
  }

  @Override
  public List<ModelRef> models() {
    return List.of(model);
  }

  @Override
  public String annotatorClass() {
    return LanguageDetectorME.class.getName();
  }

  @Override
  public String method() {
    return "predictLanguages";
  }

  @Override
  public Set<String> inputSets() {
    return inputSets;
  }

  @Override
  public List<Layer> layers(Input input) throws AnnotatorException {
    return layers(detector, input);
  }

  /** The layers {@code detector} produces for the input; {@link LanguageDump} runs it on its recording detector. */
  static List<Layer> layers(LanguageDetectorME detector, Input input) throws AnnotatorException {
    Language[] languages = AnnotatorException.call(() -> detector.predictLanguages(input.text()));
    String[] codes = new String[languages.length];
    double[] confidences = new double[languages.length];
    for (int i = 0; i < languages.length; i++) {
      codes[i] = languages[i].getLang();
      confidences[i] = languages[i].getConfidence();
    }
    return List.of(new Layer("opennlp:language", "document", new Column.Str("code", codes),
        new Column.F64("confidence", confidences)));
  }
}
