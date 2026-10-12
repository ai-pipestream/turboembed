package ai.pipestream.turbo.goldens;

import java.util.List;

/**
 * One OpenNLP annotator built over a recording model, producing the decision point dump for an
 * input. Its golden counterpart gives the header fields and the dump's path; the run also
 * returns the layers the recording annotator produced, which must equal the golden's.
 */
interface AnnotatorDump {

  /** The plain annotator this one records, which names the annotator, its models and its method. */
  AnnotatorGolden golden();

  /**
   * Runs the recording annotator on the input. Throws {@link AnnotatorException} when the OpenNLP
   * annotator threw on the input, which has no dump then.
   */
  Run run(Input input) throws AnnotatorException;

  /**
   * What one run recorded.
   *
   * @param outcomeLabels the model's outcome labels in its outcome index order
   * @param points every decision point, in the order the annotator evaluated them
   * @param layers the layers the recording annotator produced, as its golden writes them
   */
  record Run(String[] outcomeLabels, List<DecisionPoint> points, List<Layer> layers) {

    public Run {
      points = List.copyOf(points);
      layers = List.copyOf(layers);
    }
  }
}
