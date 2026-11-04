package ai.pipestream.turboembed.jni.android;

import java.io.File;
import java.io.IOException;
import java.lang.annotation.Annotation;
import java.lang.reflect.Constructor;
import java.lang.reflect.Field;
import java.lang.reflect.InvocationTargetException;
import java.lang.reflect.Method;
import java.lang.reflect.Modifier;
import java.nio.file.FileVisitResult;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.SimpleFileVisitor;
import java.nio.file.attribute.BasicFileAttributes;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.Comparator;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import org.junit.internal.AssumptionViolatedException;
import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Disabled;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;
import org.junit.runner.Description;
import org.junit.runner.Runner;
import org.junit.runner.manipulation.Filter;
import org.junit.runner.manipulation.Filterable;
import org.junit.runner.manipulation.NoTestsRemainException;
import org.junit.runner.notification.Failure;
import org.junit.runner.notification.RunNotifier;
import org.opentest4j.TestAbortedException;

/**
 * Deterministic JUnit 4 {@link Runner} that executes JUnit Jupiter ({@link Test}) methods directly
 * under {@code AndroidJUnitRunner} without relying on {@code ServiceLoader} engine discovery or
 * third-party runner plugins.
 *
 * <p>Supports inherited {@link Test} methods from abstract conformance superclasses (such as
 * {@code AnalyzerConformance} and {@code SessionConformance}), package-private test classes and
 * methods (such as {@code TurboJniTest}), {@link BeforeAll}, {@link BeforeEach}, {@link AfterEach},
 * {@link AfterAll}, {@link Disabled}, {@link TempDir} fields, and JUnit 4 {@link Filterable} test
 * selection.
 */
public class JupiterTestRunner extends Runner implements Filterable {
    private static final Comparator<Method> METHOD_ORDER =
            Comparator.comparing(Method::getName).thenComparing(m -> Arrays.toString(m.getParameterTypes()));

    private final Class<?> testClass;
    private final Runnable beforeSuite;
    private List<Method> testMethods;

    /**
     * Creates a runner for {@code testClass} with a no-op suite hook.
     *
     * @param testClass concrete class carrying {@link Test} methods directly or through superclasses
     */
    public JupiterTestRunner(Class<?> testClass) {
        this(testClass, () -> {});
    }

    /**
     * Creates a runner for {@code testClass} that runs {@code beforeSuite} before executing tests.
     *
     * @param testClass concrete class carrying {@link Test} methods directly or through superclasses
     * @param beforeSuite one-time setup hook invoked before running the suite
     */
    public JupiterTestRunner(Class<?> testClass, Runnable beforeSuite) {
        this.testClass = Objects.requireNonNull(testClass, "testClass");
        this.beforeSuite = Objects.requireNonNull(beforeSuite, "beforeSuite");
        this.testMethods = new ArrayList<>(discoverTestMethods(testClass));
    }

    /**
     * Returns whether {@code candidate} is a concrete class with at least one {@link Test} method.
     *
     * @param candidate class inspected by a {@code RunnerBuilder}
     * @return true when {@code candidate} is concrete and declares or inherits {@link Test} methods
     */
    public static boolean hasTests(Class<?> candidate) {
        return !discoverTestMethods(candidate).isEmpty();
    }

    /**
     * Returns all non-static {@link Test} methods declared on {@code candidate} or its superclasses
     * in deterministic top-down order.
     *
     * @param candidate class to inspect
     * @return immutable list of discovered test methods, empty when {@code candidate} is abstract
     */
    public static List<Method> discoverTestMethods(Class<?> candidate) {
        if (candidate == null
                || candidate.isInterface()
                || candidate.isAnnotation()
                || candidate.isEnum()
                || Modifier.isAbstract(candidate.getModifiers())) {
            return List.of();
        }
        return List.copyOf(discoverAnnotatedMethods(candidate, Test.class, false, true));
    }

    @Override
    public Description getDescription() {
        Description suite = Description.createSuiteDescription(testClass);
        for (Method method : testMethods) {
            suite.addChild(describeMethod(method));
        }
        return suite;
    }

    @Override
    public void filter(Filter filter) throws NoTestsRemainException {
        List<Method> remaining = new ArrayList<>(testMethods.size());
        for (Method method : testMethods) {
            if (filter.shouldRun(describeMethod(method))) {
                remaining.add(method);
            }
        }
        if (remaining.isEmpty()) {
            throw new NoTestsRemainException();
        }
        this.testMethods = remaining;
    }

    @Override
    public void run(RunNotifier notifier) {
        if (testMethods.isEmpty()) {
            return;
        }
        List<Method> beforeAll = discoverAnnotatedMethods(testClass, BeforeAll.class, true, true);
        List<Method> afterAll = discoverAnnotatedMethods(testClass, AfterAll.class, true, false);
        List<Method> beforeEach = discoverAnnotatedMethods(testClass, BeforeEach.class, false, true);
        List<Method> afterEach = discoverAnnotatedMethods(testClass, AfterEach.class, false, false);

        try {
            beforeSuite.run();
            for (Method method : beforeAll) {
                invokeReflectively(null, method);
            }
        } catch (Throwable setupFailure) {
            Throwable cause = unwrap(setupFailure);
            for (Method method : testMethods) {
                Description desc = describeMethod(method);
                notifier.fireTestStarted(desc);
                notifier.fireTestFailure(new Failure(desc, cause));
                notifier.fireTestFinished(desc);
            }
            return;
        }

        try {
            for (Method method : testMethods) {
                runSingleTest(method, beforeEach, afterEach, notifier);
            }
        } finally {
            for (Method method : afterAll) {
                try {
                    invokeReflectively(null, method);
                } catch (Throwable afterFailure) {
                    Description desc = describeMethod(testMethods.get(testMethods.size() - 1));
                    notifier.fireTestFailure(new Failure(desc, unwrap(afterFailure)));
                }
            }
        }
    }

    private void runSingleTest(
            Method method,
            List<Method> beforeEach,
            List<Method> afterEach,
            RunNotifier notifier) {
        Description desc = describeMethod(method);
        if (testClass.isAnnotationPresent(Disabled.class) || method.isAnnotationPresent(Disabled.class)) {
            notifier.fireTestIgnored(desc);
            return;
        }
        notifier.fireTestStarted(desc);
        List<Path> tempDirs = new ArrayList<>();
        try {
            Constructor<?> ctor = testClass.getDeclaredConstructor();
            ctor.setAccessible(true);
            Object instance = ctor.newInstance();
            injectTempDirs(instance, tempDirs);
            Throwable primary = null;
            try {
                for (Method before : beforeEach) {
                    invokeReflectively(instance, before);
                }
                invokeReflectively(instance, method);
            } catch (Throwable t) {
                primary = unwrap(t);
            } finally {
                for (Method after : afterEach) {
                    try {
                        invokeReflectively(instance, after);
                    } catch (Throwable afterErr) {
                        Throwable unwrapped = unwrap(afterErr);
                        if (primary == null) {
                            primary = unwrapped;
                        } else {
                            primary.addSuppressed(unwrapped);
                        }
                    }
                }
            }
            if (primary != null) {
                throw primary;
            }
        } catch (Throwable failure) {
            Throwable cause = unwrap(failure);
            if (cause instanceof TestAbortedException || cause instanceof AssumptionViolatedException) {
                notifier.fireTestAssumptionFailed(new Failure(desc, cause));
            } else {
                notifier.fireTestFailure(new Failure(desc, cause));
            }
        } finally {
            for (Path dir : tempDirs) {
                deleteRecursivelyQuietly(dir);
            }
            notifier.fireTestFinished(desc);
        }
    }

    private Description describeMethod(Method method) {
        return Description.createTestDescription(testClass, method.getName(), method.getDeclaredAnnotations());
    }

    private static List<Method> discoverAnnotatedMethods(
            Class<?> leaf,
            Class<? extends Annotation> annotation,
            boolean requireStatic,
            boolean topDown) {
        List<Class<?>> hierarchy = new ArrayList<>();
        for (Class<?> c = leaf; c != null && c != Object.class; c = c.getSuperclass()) {
            hierarchy.add(c);
        }
        Collections.reverse(hierarchy);
        Map<String, Method> bySignature = new LinkedHashMap<>();
        for (Class<?> clazz : hierarchy) {
            Method[] declared = clazz.getDeclaredMethods();
            Arrays.sort(declared, METHOD_ORDER);
            for (Method method : declared) {
                if (method.isSynthetic() || method.isBridge()) {
                    continue;
                }
                if (!method.isAnnotationPresent(annotation)) {
                    continue;
                }
                boolean isStatic = Modifier.isStatic(method.getModifiers());
                if (requireStatic != isStatic) {
                    continue;
                }
                String sig = method.getName() + "(" + Arrays.toString(method.getParameterTypes()) + ")";
                bySignature.put(sig, method);
            }
        }
        List<Method> ordered = new ArrayList<>(bySignature.values());
        if (!topDown) {
            Collections.reverse(ordered);
        }
        return ordered;
    }

    private static void injectTempDirs(Object instance, List<Path> created) throws IOException, IllegalAccessException {
        for (Class<?> c = instance.getClass(); c != null && c != Object.class; c = c.getSuperclass()) {
            for (Field field : c.getDeclaredFields()) {
                if (!field.isAnnotationPresent(TempDir.class) || Modifier.isStatic(field.getModifiers())) {
                    continue;
                }
                Path dir = Files.createTempDirectory("turboembed-junit-");
                created.add(dir);
                field.setAccessible(true);
                if (field.getType().isAssignableFrom(Path.class)) {
                    field.set(instance, dir);
                } else if (field.getType().isAssignableFrom(File.class)) {
                    field.set(instance, dir.toFile());
                } else {
                    throw new IllegalStateException("Unsupported @TempDir field type: " + field);
                }
            }
        }
    }

    private static void invokeReflectively(Object target, Method method) throws Throwable {
        method.setAccessible(true);
        method.invoke(target);
    }

    private static Throwable unwrap(Throwable t) {
        Throwable cur = t;
        while (cur instanceof InvocationTargetException && cur.getCause() != null) {
            cur = cur.getCause();
        }
        return cur;
    }

    private static void deleteRecursivelyQuietly(Path root) {
        if (root == null || !Files.exists(root)) {
            return;
        }
        try {
            Files.walkFileTree(
                    root,
                    new SimpleFileVisitor<>() {
                        @Override
                        public FileVisitResult visitFile(Path file, BasicFileAttributes attrs) throws IOException {
                            Files.deleteIfExists(file);
                            return FileVisitResult.CONTINUE;
                        }

                        @Override
                        public FileVisitResult postVisitDirectory(Path dir, IOException exc) throws IOException {
                            Files.deleteIfExists(dir);
                            return FileVisitResult.CONTINUE;
                        }
                    });
        } catch (IOException ignored) {
            // Best-effort cleanup of temporary test directory.
        }
    }
}
