package ai.pipestream.turbo.goldens;

import de.hhn.mi.process.SvmClassifierImpl;
import java.io.IOException;
import java.lang.reflect.Field;
import java.lang.reflect.InvocationTargetException;
import java.lang.reflect.Method;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Map;
import java.util.TreeMap;
import libsvm.svm;
import libsvm.svm_model;
import opennlp.tools.doccat.BagOfWordsFeatureGenerator;
import opennlp.tools.doccat.FeatureGenerator;
import opennlp.tools.doccat.NGramFeatureGenerator;
import opennlp.tools.ml.libsvm.doccat.SvmDoccatConfiguration;
import opennlp.tools.ml.libsvm.doccat.SvmDoccatModel;

/**
 * An {@link SvmDoccatModel} as the two text files turbo-doccat reads: the LIBSVM model as
 * {@code svm.svm_save_model} writes it, and the sidecar with the rest of the model (the format is
 * described in {@code analysis/doccat/src/svm/sidecar.rs}).
 */
final class SvmDoccatExport {

  private SvmDoccatExport() {}

  /**
   * The LIBSVM model inside {@code model}. zlibsvm keeps it in the classifier it builds over the
   * model and does not expose it, so it is read from there.
   */
  static svm_model nativeModel(SvmDoccatModel model) {
    try {
      SvmClassifierImpl classifier = new SvmClassifierImpl(model.getSvmModel());
      Method get = Class.forName("de.hhn.mi.process.AbstractSvmClassifier").getDeclaredMethod("getSvmModel");
      get.setAccessible(true);
      return (svm_model) get.invoke(classifier);
    } catch (ReflectiveOperationException e) {
      throw new IllegalStateException("zlibsvm does not keep its svm_model where expected", e);
    }
  }

  /** The text {@code svm.svm_save_model} writes for {@code model}. */
  static String libsvmText(svm_model model) {
    try {
      Path file = Files.createTempFile("svm-doccat", ".model");
      try {
        svm.svm_save_model(file.toString(), model);
        return Files.readString(file, StandardCharsets.UTF_8);
      } finally {
        Files.delete(file);
      }
    } catch (IOException e) {
      throw new IllegalStateException("svm_save_model failed", e);
    }
  }

  /** The sidecar of {@code model}, categorized with {@code generators}. */
  static String sidecar(SvmDoccatModel model, FeatureGenerator[] generators) {
    SvmDoccatConfiguration c = model.getConfiguration();
    StringBuilder b = new StringBuilder("svm-doccat 1\n");
    if (model.getLanguageCode() != null) {
      b.append("language\t").append(escape(model.getLanguageCode())).append('\n');
    }
    b.append("weighting\t").append(c.getTermWeightingStrategy().name()).append('\n');
    b.append("selection\t").append(c.getFeatureSelectionStrategy().name()).append('\t').append(c.getMaxFeatures())
        .append('\n');
    if (c.isScaleFeatures()) {
      b.append("scaling\t").append(Double.toString(c.getScaleLower())).append('\t')
          .append(Double.toString(c.getScaleUpper())).append('\n');
    } else {
      b.append("scaling\tnone\n");
    }
    for (FeatureGenerator g : generators) {
      b.append("generator\t").append(generator(g)).append('\n');
    }
    for (int i = 0; i < model.getNumberOfCategories(); i++) {
      b.append("category\t").append(i).append('\t').append(escape(model.getIndexToCategory().get(i))).append('\n');
    }
    TreeMap<Integer, String> byIndex = new TreeMap<>();
    for (Map.Entry<String, Integer> e : model.getFeatureVocabulary().entrySet()) {
      byIndex.put(e.getValue(), e.getKey());
    }
    // The same defaults buildFeatureVector reads them with.
    for (Map.Entry<Integer, String> e : byIndex.entrySet()) {
      int index = e.getKey();
      b.append("feature\t").append(index).append('\t')
          .append(Double.toString(model.getIdfValues().getOrDefault(e.getValue(), 0.0))).append('\t')
          .append(Double.toString(model.getFeatureMinValues().getOrDefault(index, 0.0))).append('\t')
          .append(Double.toString(model.getFeatureMaxValues().getOrDefault(index, 0.0))).append('\t')
          .append(escape(e.getValue())).append('\n');
    }
    return b.toString();
  }

  /** A generator's record; only the two generators the sidecar has a form for are taken. */
  private static String generator(FeatureGenerator g) {
    if (g.getClass() == BagOfWordsFeatureGenerator.class) {
      return "bag-of-words";
    }
    if (g.getClass() == NGramFeatureGenerator.class) {
      return "ngram\t" + intField(g, "minGram") + "\t" + intField(g, "maxGram");
    }
    throw new IllegalArgumentException("the sidecar has no form for " + g.getClass().getName());
  }

  private static int intField(Object o, String name) {
    try {
      Field f = o.getClass().getDeclaredField(name);
      f.setAccessible(true);
      return f.getInt(o);
    } catch (ReflectiveOperationException e) {
      throw new IllegalStateException(o.getClass().getName() + " has no int field " + name, e);
    }
  }

  /**
   * A name as golden files escape a {@code str} field. A lone surrogate has no UTF-8 form and is
   * refused.
   */
  static String escape(String s) {
    StringBuilder b = new StringBuilder(s.length());
    for (int i = 0; i < s.length(); i++) {
      char c = s.charAt(i);
      switch (c) {
        case '\t' -> b.append("\\t");
        case '\n' -> b.append("\\n");
        case '\r' -> b.append("\\r");
        case '\\' -> b.append("\\\\");
        default -> {
          if (c < 0x20) {
            b.append(String.format("\\u%04x", (int) c));
          } else if (Character.isSurrogate(c)) {
            if (!Character.isHighSurrogate(c) || i + 1 >= s.length() || !Character.isLowSurrogate(s.charAt(i + 1))) {
              throw new IllegalArgumentException("a lone surrogate in " + s);
            }
            b.append(c).append(s.charAt(++i));
          } else {
            b.append(c);
          }
        }
      }
    }
    return b.toString();
  }

  /** Unwraps the exception a reflective call to a private OpenNLP method threw. */
  static RuntimeException unwrap(InvocationTargetException e) {
    return e.getCause() instanceof RuntimeException r ? r : new IllegalStateException(e.getCause());
  }
}
