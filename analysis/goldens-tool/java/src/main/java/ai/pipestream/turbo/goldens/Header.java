package ai.pipestream.turbo.goldens;

import java.util.List;

/**
 * Everything a golden file's header records, in the order it is written: which OpenNLP and which
 * JDK produced it, the input it was produced from, the models loaded and the annotator called.
 *
 * @param opennlpCommit the 40 hex digit commit of the OpenNLP checkout the runtime was built from
 * @param opennlpVersion the Maven version of the OpenNLP runtime jar
 * @param jdk the running JDK
 * @param inputId the input's {@code <set>/<id>}
 * @param utf8Length the input's length in UTF-8 bytes
 * @param utf16Length the input's length in UTF-16 units
 * @param inputSha256 the SHA-256 of the input's bytes, lowercase hex
 * @param models the model files the annotator loaded, in the order loaded
 * @param annotatorClass the fully qualified class of the OpenNLP annotator
 * @param method the method called on it
 */
record Header(String opennlpCommit, String opennlpVersion, Jdk jdk, String inputId, long utf8Length,
    long utf16Length, String inputSha256, List<ModelRef> models, String annotatorClass, String method) {

  Header {
    models = List.copyOf(models);
  }
}
