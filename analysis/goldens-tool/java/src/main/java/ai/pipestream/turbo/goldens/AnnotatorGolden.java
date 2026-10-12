package ai.pipestream.turbo.goldens;

import java.util.List;
import java.util.Map;
import java.util.Set;

/**
 * One OpenNLP annotator with its model loaded, producing the layers of its golden for an input.
 * Each kind of annotator has its own implementation; the main class only iterates over them.
 */
interface AnnotatorGolden {

  /** The short name that is the first directory of the golden's path, such as {@code sentdetect}. */
  String annotator();

  /** The model files loaded, in the order loaded; empty for an annotator without a model. */
  List<ModelRef> models();

  /** The fully qualified class of the OpenNLP annotator. */
  String annotatorClass();

  /** The method called on it. */
  String method();

  /** The options the annotator is run with, written as {@code config} header records; empty by default. */
  default Map<String, String> config() {
    return Map.of();
  }

  /** The input sets this annotator runs on. */
  Set<String> inputSets();

  /** Whether the golden is made for this input: by default, whether its set is one of {@link #inputSets()}. */
  default boolean accepts(Input input) {
    return inputSets().contains(input.set());
  }

  /**
   * Runs the annotator on the input and returns its layers in the order the format lists them.
   * Throws {@link AnnotatorException} when the OpenNLP annotator threw on the input.
   */
  List<Layer> layers(Input input) throws AnnotatorException;

  /**
   * The model directory of the golden's path: the first model's base name, or {@code none}. A run
   * with options appends {@code +} and a name for them, so it does not share the directory of the
   * same model run without them; an annotator without a model names its run instead.
   */
  default String modelDir() {
    return models().isEmpty() ? "none" : models().get(0).baseName();
  }
}
