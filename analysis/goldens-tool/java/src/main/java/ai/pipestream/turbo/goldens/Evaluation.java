package ai.pipestream.turbo.goldens;

/**
 * One {@code eval} call {@link MaxentRecorder} saw: the context generator request it answers, if
 * a recording context generator reported one, and copies of the arrays passed in and returned.
 *
 * @param request what the context generator was asked for just before the call, or {@code null}
 *     when no recording context generator is in use
 * @param contexts a copy of the context array passed to {@code eval}
 * @param values a copy of the feature values passed to {@code eval}, or {@code null} when the call
 *     carried none
 * @param outcomes a copy of the probabilities {@code eval} returned
 */
record Evaluation(ContextRequest request, String[] contexts, float[] values, double[] outcomes) {}
