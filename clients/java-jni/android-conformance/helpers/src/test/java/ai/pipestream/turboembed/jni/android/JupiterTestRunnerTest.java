package ai.pipestream.turboembed.jni.android;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.atomic.AtomicBoolean;
import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.Assumptions;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Disabled;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;
import org.junit.runner.Description;
import org.junit.runner.JUnitCore;
import org.junit.runner.Request;
import org.junit.runner.Result;
import org.junit.runner.manipulation.Filter;
import org.junit.runner.manipulation.NoTestsRemainException;

/** Host unit tests for {@link JupiterTestRunner}. */
final class JupiterTestRunnerTest {
    private static final List<String> EVENTS = new ArrayList<>();

    abstract static class AbstractConformanceSample {
        @BeforeEach
        void baseBefore() {
            EVENTS.add("baseBefore");
        }

        @AfterEach
        void baseAfter() {
            EVENTS.add("baseAfter");
        }

        @Test
        public void inheritedFirstTest() {
            EVENTS.add("inheritedFirstTest");
        }

        @Test
        public void inheritedSecondTest() {
            EVENTS.add("inheritedSecondTest");
        }
    }

    static final class ConcreteConformanceSample extends AbstractConformanceSample {
        @TempDir
        Path scratch;

        @BeforeAll
        static void beforeAll() {
            EVENTS.add("beforeAll");
        }

        @AfterAll
        static void afterAll() {
            EVENTS.add("afterAll");
        }

        @Test
        void packagePrivateChildTestWithTempDir() throws Exception {
            assertNotNull(scratch);
            assertTrue(Files.isDirectory(scratch));
            Files.writeString(scratch.resolve("probe.txt"), "ok");
            EVENTS.add("packagePrivateChildTestWithTempDir");
        }

        @Test
        @Disabled("disabled sample")
        void skippedTest() {
            EVENTS.add("skippedTest");
        }
    }

    static final class AssumptionAndFailureSample {
        @Test
        void abortedAssumptionCountsAsAssumptionFailureNotError() {
            Assumptions.assumeTrue(false, "skipped by assumption");
        }

        @Test
        void failingAssertionIsReported() {
            throw new AssertionError("expected failure");
        }
    }

    @Test
    void discoversAndExecutesInheritedAndPackagePrivateJupiterTests() {
        EVENTS.clear();
        assertFalse(JupiterTestRunner.hasTests(AbstractConformanceSample.class));
        assertTrue(JupiterTestRunner.hasTests(ConcreteConformanceSample.class));
        assertEquals(4, JupiterTestRunner.discoverTestMethods(ConcreteConformanceSample.class).size());

        AtomicBoolean suiteHookRan = new AtomicBoolean(false);
        JupiterTestRunner runner =
                new JupiterTestRunner(ConcreteConformanceSample.class, () -> suiteHookRan.set(true));
        Result result = new JUnitCore().run(Request.runner(runner));
        assertTrue(suiteHookRan.get());
        assertEquals(3, result.getRunCount());
        assertEquals(0, result.getFailureCount());
        assertEquals(1, result.getIgnoreCount());
        assertEquals(
                List.of(
                        "beforeAll",
                        "baseBefore",
                        "inheritedFirstTest",
                        "baseAfter",
                        "baseBefore",
                        "inheritedSecondTest",
                        "baseAfter",
                        "baseBefore",
                        "packagePrivateChildTestWithTempDir",
                        "baseAfter",
                        "afterAll"),
                EVENTS);
    }

    @Test
    void reportsAssumptionsAndFailuresAndSupportsMethodFiltering() throws Exception {
        JupiterTestRunner runner = new JupiterTestRunner(AssumptionAndFailureSample.class);
        Result full = new JUnitCore().run(Request.runner(runner));
        assertEquals(2, full.getRunCount());
        assertEquals(1, full.getAssumptionFailureCount());
        assertEquals(1, full.getFailureCount());

        JupiterTestRunner filtered = new JupiterTestRunner(AssumptionAndFailureSample.class);
        filtered.filter(
                new Filter() {
                    @Override
                    public boolean shouldRun(Description description) {
                        return "abortedAssumptionCountsAsAssumptionFailureNotError"
                                .equals(description.getMethodName());
                    }

                    @Override
                    public String describe() {
                        return "only assumption test";
                    }
                });
        Result single = new JUnitCore().run(Request.runner(filtered));
        assertEquals(1, single.getRunCount());
        assertEquals(0, single.getFailureCount());
        assertEquals(1, single.getAssumptionFailureCount());

        assertThrows(
                NoTestsRemainException.class,
                () ->
                        filtered.filter(
                                new Filter() {
                                    @Override
                                    public boolean shouldRun(Description description) {
                                        return false;
                                    }

                                    @Override
                                    public String describe() {
                                        return "none";
                                    }
                                }));
    }
}
