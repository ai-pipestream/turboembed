package ai.pipestream.turbo.goldens;

/**
 * The running JDK as a golden's {@code jdk} and {@code unicode} lines record it.
 *
 * <p>The Unicode version is not a system property, so it comes from a table from the JDK's
 * feature version to the Unicode version its {@link Character} tables implement, as each JDK's
 * release notes state. A feature version the table does not hold makes the generator fail rather
 * than guess.</p>
 */
record Jdk(String javaVersion, String vmVersion, String vendor, String unicodeVersion) {

  /** The JDK this generator is running on. */
  static Jdk current() {
    return new Jdk(System.getProperty("java.version"), System.getProperty("java.vm.version"),
        System.getProperty("java.vendor"), unicodeVersion(Runtime.version().feature()));
  }

  /** The Unicode version JDK {@code feature} implements; fails for a version not in the table. */
  static String unicodeVersion(int feature) {
    return switch (feature) {
      case 17 -> "13.0";
      case 21 -> "15.0";
      case 22, 23 -> "15.1";
      case 24 -> "16.0";
      default -> throw new IllegalStateException(
          "the Unicode version of JDK " + feature + " is not known to the generator; add it to Jdk");
    };
  }
}
