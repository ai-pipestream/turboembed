package ai.pipestream.turbo.goldens;

import java.util.Map;
import java.util.Properties;
import java.util.TreeMap;
import opennlp.tools.util.ext.ExtensionLoader;
import opennlp.tools.util.model.BaseModel;

/**
 * What building a recording model from a loaded one needs: the loaded model's manifest entries
 * to pass to the model constructor that takes a maxent model, and a check that the recording
 * model's manifest came out the same.
 */
final class RecordingModels {

  /** The artifact that holds a model's manifest ({@code BaseModel.MANIFEST_ENTRY}). */
  private static final String MANIFEST = "manifest.properties";

  /** The manifest entry naming the factory class ({@code BaseModel.FACTORY_NAME}). */
  static final String FACTORY_ENTRY = "factory";

  private RecordingModels() {}

  /**
   * Lets OpenNLP's {@code ExtensionLoader}, which only loads classes from packages it was told
   * about, instantiate the recording factories of this package.
   */
  static void allowFactories() {
    ExtensionLoader.registerAllowedPackage(RecordingModels.class.getPackageName());
  }

  /** Every manifest entry of {@code model}, sorted by key. */
  static Map<String, String> manifest(BaseModel model) {
    Properties p = model.getArtifact(MANIFEST);
    Map<String, String> out = new TreeMap<>();
    for (String key : p.stringPropertyNames()) {
      out.put(key, p.getProperty(key));
    }
    return out;
  }

  /**
   * The loaded model's manifest entries without the factory entry, which the model constructor
   * sets from the factory it is given. Passing the rest keeps the version, timestamp and
   * training entries the loaded model has.
   */
  static Map<String, String> entriesFor(BaseModel loaded) {
    Map<String, String> entries = manifest(loaded);
    entries.remove(FACTORY_ENTRY);
    return entries;
  }

  /**
   * Fails unless the recording model's manifest equals the loaded model's in every entry except
   * the factory, which must name {@code factory}.
   */
  static void checkManifest(BaseModel loaded, BaseModel recording, Class<?> factory) {
    Map<String, String> a = manifest(loaded);
    Map<String, String> b = manifest(recording);
    if (!factory.getName().equals(b.get(FACTORY_ENTRY))) {
      throw new IllegalStateException("the recording model's factory is " + b.get(FACTORY_ENTRY) + ", not "
          + factory.getName());
    }
    a.remove(FACTORY_ENTRY);
    b.remove(FACTORY_ENTRY);
    if (!a.equals(b)) {
      throw new IllegalStateException("the recording model's manifest " + b + " differs from the loaded model's " + a);
    }
  }

  /** Fails unless the loaded model's factory is exactly {@code expected}, which the recording one extends. */
  static void checkFactory(Object loadedFactory, Class<?> expected) {
    if (loadedFactory == null || loadedFactory.getClass() != expected) {
      throw new IllegalStateException("the model's factory is "
          + (loadedFactory == null ? "missing" : loadedFactory.getClass().getName()) + ", the recording model"
          + " reproduces " + expected.getName() + " only");
    }
  }
}
