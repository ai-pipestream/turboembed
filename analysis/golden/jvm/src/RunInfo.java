/**
 * The header values that are the same for every golden file of one run.
 *
 * @param opennlpCommit the OpenNLP commit the jars were built from
 * @param jdkVersion the {@code java.version} property
 * @param whitespace the OpenNLP whitespace mode, {@code unicode} or {@code legacy}
 */
record RunInfo(String opennlpCommit, String jdkVersion, String whitespace) {

  RunInfo withWhitespace(String mode) {
    return new RunInfo(opennlpCommit, jdkVersion, mode);
  }
}
