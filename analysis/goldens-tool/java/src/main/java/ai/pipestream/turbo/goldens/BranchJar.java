package ai.pipestream.turbo.goldens;

import java.io.IOException;
import java.net.URL;
import java.net.URLClassLoader;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Enumeration;
import java.util.HashSet;
import java.util.List;
import java.util.Set;
import java.util.jar.JarEntry;
import java.util.jar.JarFile;

/**
 * A jar of OpenNLP classes compiled from a branch commit that is not the one the image is built
 * from. The image compiles the classes a golden needs from that commit into the jar and writes the
 * commit into a file {@code commit} beside it; the generator's entrypoint names the jar in a
 * system property.
 *
 * <p>{@link #open} checks the commit and that the jar holds the classes the golden uses, and
 * returns a class loader that takes every class the jar holds from the jar, even one the
 * generator's own OpenNLP jars also hold, and every other class from the generator's loader.</p>
 */
final class BranchJar {

  private BranchJar() {}

  /**
   * The class loader of the jar the system property {@code property} names, which must have been
   * built from {@code commit} and hold every class of {@code required}.
   */
  static ClassLoader open(String property, String commit, List<String> required) throws IOException {
    String jar = System.getProperty(property);
    if (jar == null) {
      throw new IllegalStateException("-D" + property + " must name the jar of the branch's classes");
    }
    Path path = Path.of(jar);
    String built = Files.readString(path.resolveSibling("commit"), StandardCharsets.UTF_8).strip();
    if (!built.equals(commit)) {
      throw new IllegalStateException(path + " was built from " + built + ", not " + commit);
    }
    Set<String> classes = new HashSet<>();
    try (JarFile file = new JarFile(path.toFile())) {
      for (Enumeration<JarEntry> e = file.entries(); e.hasMoreElements(); ) {
        String name = e.nextElement().getName();
        if (name.endsWith(".class")) {
          classes.add(name.substring(0, name.length() - ".class".length()).replace('/', '.'));
        }
      }
    }
    for (String c : required) {
      if (!classes.contains(c)) {
        throw new IllegalStateException(path + " does not hold " + c);
      }
    }
    return new JarFirst(path.toUri().toURL(), Set.copyOf(classes));
  }

  /** Loads the classes the jar holds from the jar before asking the generator's loader. */
  private static final class JarFirst extends URLClassLoader {
    private final Set<String> own;

    JarFirst(URL jar, Set<String> own) {
      super(new URL[] {jar}, BranchJar.class.getClassLoader());
      this.own = own;
    }

    @Override
    protected Class<?> loadClass(String name, boolean resolve) throws ClassNotFoundException {
      if (!own.contains(name)) {
        return super.loadClass(name, resolve);
      }
      synchronized (getClassLoadingLock(name)) {
        Class<?> c = findLoadedClass(name);
        if (c == null) {
          c = findClass(name);
        }
        if (resolve) {
          resolveClass(c);
        }
        return c;
      }
    }
  }
}
