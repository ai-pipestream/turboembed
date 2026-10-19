package ai.pipestream.turbo.goldens;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Path;
import java.util.List;
import java.util.Set;
import opennlp.tools.lemmatizer.DictionaryLemmatizer;

/**
 * The dictionary lemmatizer's golden: the sentences, tokens and part-of-speech tags of the
 * English models ({@link PosGolden}'s layers), then {@code opennlp:lemmas} aligned to the tokens,
 * the lemma {@link DictionaryLemmatizer#lemmatize(String[], String[])} gave each token ({@code O}
 * where the dictionary has none). A lookup asks no model, so this annotator has its own
 * directory, {@code lemmatize-dictionary}, and no dumps. The model records name the dictionary
 * first, then the sentence, token and part-of-speech models.
 */
final class DictionaryLemmaGolden implements AnnotatorGolden {

  private final List<ModelRef> models;
  private final DictionaryLemmatizer lemmatizer;
  private final LemmaGolden.Pipeline pipeline;
  private final Set<String> inputSets;

  private DictionaryLemmaGolden(ModelRef dictionary, DictionaryLemmatizer lemmatizer, LemmaGolden.Pipeline pipeline,
      Set<String> inputSets) {
    this.models = pipeline.models(dictionary);
    this.lemmatizer = lemmatizer;
    this.pipeline = pipeline;
    this.inputSets = Set.copyOf(inputSets);
  }

  /** Reads the dictionary from {@code file}, UTF-8, hashing exactly the bytes read. */
  static DictionaryLemmaGolden load(Path file, LemmaGolden.Pipeline pipeline, Set<String> inputSets)
      throws IOException {
    ModelFile f = ModelFile.read(file);
    return new DictionaryLemmaGolden(f.ref(), new DictionaryLemmatizer(f.stream(), StandardCharsets.UTF_8), pipeline,
        inputSets);
  }

  @Override
  public String annotator() {
    return "lemmatize-dictionary";
  }

  @Override
  public List<ModelRef> models() {
    return models;
  }

  @Override
  public String annotatorClass() {
    return DictionaryLemmatizer.class.getName();
  }

  @Override
  public String method() {
    return "lemmatize";
  }

  @Override
  public Set<String> inputSets() {
    return inputSets;
  }

  @Override
  public List<Layer> layers(Input input) throws AnnotatorException {
    return LemmaGolden.run(lemmatizer, pipeline, input, false).layers();
  }
}
