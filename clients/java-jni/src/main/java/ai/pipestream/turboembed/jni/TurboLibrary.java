package ai.pipestream.turboembed.jni;

import java.io.File;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;

/**
 * The {@code libturbo} shared library as this binding loads it, and the constants of its C ABI
 * the binding reads back: status codes, layer kinds, span flags and PII classifications.
 *
 * <p>The library is loaded once, when {@link #load()} is first called or the first native
 * method runs. Discovery, in order: the {@code turbo.library.path} system property or the
 * {@code TURBO_LIBRARY_PATH} variable (a shared library file or the directory holding it; a
 * configured path that does not exist is an error), the {@code turbo.library.dir} property or
 * {@code TURBO_LIBRARY_DIR} variable (path-separator-delimited directories; a configured set
 * with no library is an error), {@code target/debug} and {@code target/release} under the
 * working directory or one of its ancestors, and then {@link System#loadLibrary(String)} with
 * the name {@code turbo}, which on Android finds {@code libturbo.so} among the application's
 * native libraries for the device's ABI.
 */
public final class TurboLibrary {
    /** Status code for success. */
    public static final int TURBO_OK = 0;
    /** Status code for invalid argument. */
    public static final int TURBO_E_INVALID_ARGUMENT = 256;
    /** Status code for invalid struct_size. */
    public static final int TURBO_E_INVALID_STRUCT_SIZE = 257;
    /** Status code for invalid UTF-8 bytes. */
    public static final int TURBO_E_INVALID_UTF8 = 258;
    /** Status code for invalid handle. */
    public static final int TURBO_E_INVALID_HANDLE = 259;
    /** Status code for a task whose required model is not configured. */
    public static final int TURBO_E_UNSUPPORTED_TASK = 514;
    /** Status code for bundle not found. */
    public static final int TURBO_E_BUNDLE_NOT_FOUND = 1280;
    /** Status code for invalid bundle. */
    public static final int TURBO_E_BUNDLE_INVALID = 1281;

    /** Native {@code TURBO_ANALYSIS_LAYER_SENTENCES}. */
    public static final int TURBO_ANALYSIS_LAYER_SENTENCES = 1;
    /** Native {@code TURBO_ANALYSIS_LAYER_TOKENS}. */
    public static final int TURBO_ANALYSIS_LAYER_TOKENS = 2;
    /** Native {@code TURBO_ANALYSIS_LAYER_POS}. */
    public static final int TURBO_ANALYSIS_LAYER_POS = 3;
    /** Native {@code TURBO_ANALYSIS_LAYER_LEMMAS}. */
    public static final int TURBO_ANALYSIS_LAYER_LEMMAS = 4;
    /** Native {@code TURBO_ANALYSIS_LAYER_ENTITIES}. */
    public static final int TURBO_ANALYSIS_LAYER_ENTITIES = 5;
    /** Native {@code TURBO_ANALYSIS_LAYER_PII}. */
    public static final int TURBO_ANALYSIS_LAYER_PII = 6;
    /** Native {@code TURBO_ANALYSIS_LAYER_DEPENDENCIES}. */
    public static final int TURBO_ANALYSIS_LAYER_DEPENDENCIES = 7;
    /** Native {@code TURBO_ANALYSIS_LAYER_MORPHOLOGY}. */
    public static final int TURBO_ANALYSIS_LAYER_MORPHOLOGY = 8;

    /** Native span flag: a view range is present. */
    public static final int TURBO_SPAN_VIEW_RANGE = 1;
    /** Native span flag: a source range is present. */
    public static final int TURBO_SPAN_SOURCE_RANGE = 1 << 1;
    /** Native span flag: a token range is present. */
    public static final int TURBO_SPAN_TOKEN_RANGE = 1 << 2;
    /** Native span flag: a label is present. */
    public static final int TURBO_SPAN_LABEL = 1 << 3;
    /** Native span flag: a normalized value is present. */
    public static final int TURBO_SPAN_VALUE = 1 << 4;
    /** Native span flag: a probability is present. */
    public static final int TURBO_SPAN_PROBABILITY = 1 << 5;
    /** Mask containing every native span flag defined by this ABI. */
    public static final int TURBO_SPAN_KNOWN_FLAGS =
            TURBO_SPAN_VIEW_RANGE
                    | TURBO_SPAN_SOURCE_RANGE
                    | TURBO_SPAN_TOKEN_RANGE
                    | TURBO_SPAN_LABEL
                    | TURBO_SPAN_VALUE
                    | TURBO_SPAN_PROBABILITY;

    /** Native PII classification: format-only evidence. */
    public static final int TURBO_PII_CONFIDENCE_FORMAT = 1;
    /** Native PII classification: checksum evidence. */
    public static final int TURBO_PII_CONFIDENCE_CHECKSUM = 2;
    /** Native PII classification: labelled checksum evidence. */
    public static final int TURBO_PII_CONFIDENCE_LABELLED = 3;

    /**
     * The {@code uint32_t} fields of a native {@code turbo_span} before its probability, in the
     * header's order: a row crosses from the library as this many {@code int}s and one
     * {@code double}.
     */
    public static final int TURBO_SPAN_INTS = 28;

    private static final String LIBRARY_NAME = "turbo";
    private static boolean loaded;

    private TurboLibrary() {
    }

    /**
     * Loads {@code libturbo} if it is not loaded yet. Every native method loads it on first use;
     * calling this early surfaces a missing library where it is easiest to diagnose.
     *
     * @throws UnsatisfiedLinkError when no library is found
     */
    public static synchronized void load() {
        if (loaded) {
            return;
        }
        String libFile = System.mapLibraryName(LIBRARY_NAME);
        String explicitPath = System.getProperty("turbo.library.path", System.getenv("TURBO_LIBRARY_PATH"));
        if (explicitPath != null && !explicitPath.trim().isEmpty()) {
            Path candidate = Path.of(explicitPath);
            if (Files.isDirectory(candidate)) {
                candidate = candidate.resolve(libFile);
            }
            if (!Files.isRegularFile(candidate)) {
                throw new UnsatisfiedLinkError("Configured libturbo path does not exist: " + candidate);
            }
            System.load(candidate.toAbsolutePath().normalize().toString());
            loaded = true;
            return;
        }

        List<Path> candidates = new ArrayList<>();
        String searchDirs = System.getProperty("turbo.library.dir", System.getenv("TURBO_LIBRARY_DIR"));
        boolean searchDirsConfigured = searchDirs != null && !searchDirs.trim().isEmpty();
        if (searchDirsConfigured) {
            for (String part : searchDirs.split(File.pathSeparator)) {
                if (!part.trim().isEmpty()) {
                    candidates.add(Path.of(part, libFile));
                }
            }
        } else {
            for (Path dir = Path.of("").toAbsolutePath().normalize(); dir != null; dir = dir.getParent()) {
                candidates.add(dir.resolve("target/debug").resolve(libFile));
                candidates.add(dir.resolve("target/release").resolve(libFile));
            }
        }
        for (Path candidate : candidates) {
            if (Files.isRegularFile(candidate)) {
                System.load(candidate.toAbsolutePath().normalize().toString());
                loaded = true;
                return;
            }
        }
        if (searchDirsConfigured) {
            throw new UnsatisfiedLinkError("libturbo was not found in configured directories: " + candidates);
        }
        System.loadLibrary(LIBRARY_NAME);
        loaded = true;
    }

    /**
     * Returns the {@code libturbo} version and linked backend list from {@code turbo_version()}.
     *
     * @return version string
     */
    public static String version() {
        return TurboJni.version();
    }

    /**
     * Returns the symbolic name for a {@code libturbo} status code from {@code turbo_status_name(code)}.
     *
     * @param code status code
     * @return status constant name
     */
    public static String statusName(int code) {
        return TurboJni.statusName(code);
    }

    /**
     * Returns the calling thread's most recent C ABI error message from {@code turbo_last_error_message()}.
     *
     * @return last error message on the current thread, or {@code ""}
     */
    public static String lastErrorMessage() {
        return TurboJni.lastErrorMessage();
    }
}
