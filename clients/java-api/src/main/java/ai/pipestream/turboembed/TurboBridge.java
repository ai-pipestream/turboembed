package ai.pipestream.turboembed;

/** The in-process transport used to call libturbo. */
public enum TurboBridge {
    /** JNI on Java 21+ and Android with the documented desugaring setup. */
    JNI,
    /** The foreign-memory implementation, built for Java 25+. */
    PANAMA
}
