package ai.pipestream.turbo.goldens;

import java.util.List;

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

  /** Runs the annotator on the input and returns its layers in the order the format lists them. */
  List<Layer> layers(Input input);

  /** The model directory of the golden's path: the first model's base name, or {@code none}. */
  default String modelDir() {
    return models().isEmpty() ? "none" : models().get(0).baseName();
  }
}
