package ai.pipestream.turboembed.jni.android;

import java.io.IOException;
import java.io.InputStream;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.stream.Stream;
import javax.xml.XMLConstants;
import javax.xml.parsers.DocumentBuilder;
import javax.xml.parsers.DocumentBuilderFactory;
import javax.xml.parsers.ParserConfigurationException;
import org.w3c.dom.Document;
import org.w3c.dom.Element;
import org.w3c.dom.Node;
import org.w3c.dom.NodeList;
import org.xml.sax.SAXException;

/**
 * Parses the JUnit XML reports emitted by {@code connectedDebugAndroidTest} and refuses silent
 * zero-test runs, missing conformance suites, or test failures/errors.
 */
public final class AndroidTestReportVerifier {
    /** Required Android instrumentation suite classes and their minimum expected test counts. */
    public static final Map<String, Integer> REQUIRED_SUITES;

    static {
        Map<String, Integer> suites = new LinkedHashMap<>();
        suites.put("ai.pipestream.turboembed.jni.android.AndroidAnalyzerConformanceTest", 13);
        suites.put("ai.pipestream.turboembed.jni.android.AndroidSessionConformanceTest", 2);
        suites.put("ai.pipestream.turboembed.jni.android.AndroidTurboJniBridgeTest", 3);
        suites.put("ai.pipestream.turboembed.jni.TurboJniTest", 10);
        REQUIRED_SUITES = Collections.unmodifiableMap(suites);
    }

    /** Per-suite test counts aggregated across XML reports. */
    public record SuiteCounts(int passed, int failed, int errors, int skipped) {
        /** Total testcases recorded for the suite. */
        public int total() {
            return passed + failed + errors + skipped;
        }
    }

    /** Aggregated verification result across all {@code TEST-*.xml} files under a report root. */
    public record Summary(
            List<Path> reportFiles,
            Map<String, SuiteCounts> suites,
            int totalPassed,
            int totalFailed,
            int totalErrors,
            int totalSkipped) {
        /** Total testcases recorded across all suites. */
        public int totalTests() {
            return totalPassed + totalFailed + totalErrors + totalSkipped;
        }
    }

    private AndroidTestReportVerifier() {}

    /**
     * Entry point invoked by {@code :helpers:verifyAndroidTestResults} and
     * {@code run-android-conformance.sh}.
     *
     * @param args single path to the {@code connected} androidTest-results directory
     * @throws Exception when verification fails
     */
    public static void main(String[] args) throws Exception {
        if (args.length != 1 || args[0].isBlank()) {
            throw new IllegalArgumentException(
                    "Usage: AndroidTestReportVerifier <path-to-androidTest-results/connected>");
        }
        Path reportDir = Path.of(args[0]).toAbsolutePath().normalize();
        Summary summary = verify(reportDir);
        System.out.println("Android conformance XML report summary (" + reportDir + "):");
        for (Map.Entry<String, SuiteCounts> entry : summary.suites().entrySet()) {
            SuiteCounts counts = entry.getValue();
            System.out.printf(
                    "  %s: %d passed, %d failed, %d errors, %d skipped (total %d)%n",
                    entry.getKey(),
                    counts.passed(),
                    counts.failed(),
                    counts.errors(),
                    counts.skipped(),
                    counts.total());
        }
        System.out.printf(
                "  TOTAL: %d passed, %d failed, %d errors, %d skipped (across %d report file(s))%n",
                summary.totalPassed(),
                summary.totalFailed(),
                summary.totalErrors(),
                summary.totalSkipped(),
                summary.reportFiles().size());
    }

    /**
     * Verifies all {@code TEST-*.xml} files under {@code reportRoot} against {@link #REQUIRED_SUITES}.
     *
     * @param reportRoot root directory containing {@code TEST-*.xml} files
     * @return verified summary
     * @throws IOException on I/O error reading reports
     * @throws IllegalStateException when no reports exist, 0 tests ran, any test failed, or a suite is missing
     */
    public static Summary verify(Path reportRoot) throws IOException {
        return verify(reportRoot, REQUIRED_SUITES);
    }

    /**
     * Verifies all {@code TEST-*.xml} files under {@code reportRoot} against {@code requiredSuites}.
     *
     * @param reportRoot root directory containing {@code TEST-*.xml} files
     * @param requiredSuites map of required suite class name to minimum passing test count
     * @return verified summary
     * @throws IOException on I/O error reading reports
     * @throws IllegalStateException when verification fails
     */
    public static Summary verify(Path reportRoot, Map<String, Integer> requiredSuites) throws IOException {
        Objects.requireNonNull(reportRoot, "reportRoot");
        Objects.requireNonNull(requiredSuites, "requiredSuites");
        if (!Files.isDirectory(reportRoot)) {
            throw new IllegalStateException("Android test report directory does not exist: " + reportRoot);
        }
        List<Path> xmlFiles;
        try (Stream<Path> walk = Files.walk(reportRoot)) {
            xmlFiles = walk.filter(Files::isRegularFile)
                    .filter(p -> {
                        String name = p.getFileName().toString();
                        return name.startsWith("TEST-") && name.endsWith(".xml");
                    })
                    .sorted(Comparator.comparing(Path::toString))
                    .toList();
        }
        if (xmlFiles.isEmpty()) {
            throw new IllegalStateException("No TEST-*.xml files found under " + reportRoot);
        }

        Map<String, int[]> perSuite = new LinkedHashMap<>();
        for (String req : requiredSuites.keySet()) {
            perSuite.put(req, new int[4]);
        }
        int suiteAttrFailures = 0;
        int suiteAttrErrors = 0;
        List<String> failureDetails = new ArrayList<>();

        for (Path xml : xmlFiles) {
            Document doc = parseXml(xml);
            NodeList suiteNodes = doc.getElementsByTagName("testsuite");
            for (int i = 0; i < suiteNodes.getLength(); i++) {
                Element suiteEl = (Element) suiteNodes.item(i);
                suiteAttrFailures += parseIntAttr(suiteEl, "failures");
                suiteAttrErrors += parseIntAttr(suiteEl, "errors");
            }
            NodeList testCases = doc.getElementsByTagName("testcase");
            for (int i = 0; i < testCases.getLength(); i++) {
                Element tc = (Element) testCases.item(i);
                String className = tc.getAttribute("classname").trim();
                String testName = tc.getAttribute("name").trim();
                if (className.isEmpty()) {
                    className = "<unknown>";
                }
                int[] counts = perSuite.computeIfAbsent(className, k -> new int[4]);
                if (hasChildElement(tc, "failure")) {
                    counts[1]++;
                    failureDetails.add(className + "#" + testName + " (failure)");
                } else if (hasChildElement(tc, "error")) {
                    counts[2]++;
                    failureDetails.add(className + "#" + testName + " (error)");
                } else if (hasChildElement(tc, "skipped")) {
                    counts[3]++;
                } else {
                    counts[0]++;
                }
            }
        }

        Map<String, SuiteCounts> suites = new LinkedHashMap<>();
        int totalPassed = 0;
        int totalFailed = 0;
        int totalErrors = 0;
        int totalSkipped = 0;
        for (Map.Entry<String, int[]> entry : perSuite.entrySet()) {
            int[] c = entry.getValue();
            if (c[0] == 0 && c[1] == 0 && c[2] == 0 && c[3] == 0 && !requiredSuites.containsKey(entry.getKey())) {
                continue;
            }
            SuiteCounts sc = new SuiteCounts(c[0], c[1], c[2], c[3]);
            suites.put(entry.getKey(), sc);
            totalPassed += c[0];
            totalFailed += c[1];
            totalErrors += c[2];
            totalSkipped += c[3];
        }

        int effectiveFailed = Math.max(totalFailed, suiteAttrFailures);
        int effectiveErrors = Math.max(totalErrors, suiteAttrErrors);

        if (totalPassed + totalFailed + totalErrors + totalSkipped == 0) {
            throw new IllegalStateException(
                    "Android instrumentation run executed 0 tests across " + xmlFiles.size() + " report(s) in " + reportRoot);
        }
        if (effectiveFailed > 0 || effectiveErrors > 0) {
            throw new IllegalStateException(
                    "Android instrumentation run had failures/errors (failed="
                            + effectiveFailed
                            + ", errors="
                            + effectiveErrors
                            + "): "
                            + failureDetails);
        }
        for (Map.Entry<String, Integer> req : requiredSuites.entrySet()) {
            String suiteName = req.getKey();
            int minExpected = req.getValue();
            SuiteCounts sc = suites.get(suiteName);
            int actualPassed = sc == null ? 0 : sc.passed();
            if (actualPassed < minExpected) {
                throw new IllegalStateException(
                        "Suite "
                                + suiteName
                                + " passed "
                                + actualPassed
                                + " test(s), expected at least "
                                + minExpected
                                + " in "
                                + reportRoot);
            }
        }

        return new Summary(
                List.copyOf(xmlFiles),
                Collections.unmodifiableMap(suites),
                totalPassed,
                effectiveFailed,
                effectiveErrors,
                totalSkipped);
    }

    private static Document parseXml(Path xml) throws IOException {
        try (InputStream in = Files.newInputStream(xml)) {
            DocumentBuilderFactory factory = DocumentBuilderFactory.newInstance();
            factory.setFeature(XMLConstants.FEATURE_SECURE_PROCESSING, true);
            factory.setFeature("http://apache.org/xml/features/disallow-doctype-decl", true);
            DocumentBuilder builder = factory.newDocumentBuilder();
            return builder.parse(in);
        } catch (ParserConfigurationException | SAXException ex) {
            throw new IOException("Failed to parse XML report " + xml, ex);
        }
    }

    private static int parseIntAttr(Element el, String attr) {
        String raw = el.getAttribute(attr);
        if (raw == null || raw.isBlank()) {
            return 0;
        }
        try {
            return Integer.parseInt(raw.trim());
        } catch (NumberFormatException ignored) {
            return 0;
        }
    }

    private static boolean hasChildElement(Element parent, String tagName) {
        NodeList children = parent.getChildNodes();
        for (int i = 0; i < children.getLength(); i++) {
            Node node = children.item(i);
            if (node.getNodeType() == Node.ELEMENT_NODE && tagName.equals(node.getNodeName())) {
                return true;
            }
        }
        return false;
    }
}
