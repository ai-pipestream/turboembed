package ai.pipestream.turboembed.panama;

import java.io.File;
import java.lang.foreign.Arena;
import java.lang.foreign.FunctionDescriptor;
import java.lang.foreign.Linker;
import java.lang.foreign.MemoryLayout;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.StructLayout;
import java.lang.foreign.SymbolLookup;
import java.lang.foreign.ValueLayout;
import java.lang.invoke.MethodHandle;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Objects;

/**
 * Low-level Java 22+ Panama Foreign Function and Memory (FFM) bindings over {@code libturbo}.
 *
 * <p>Uses {@link Linker#nativeLinker()} and {@link SymbolLookup} directly with zero JNI compilation.
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
    /** Status code for bundle not found. */
    public static final int TURBO_E_BUNDLE_NOT_FOUND = 1280;
    /** Status code for invalid bundle. */
    public static final int TURBO_E_BUNDLE_INVALID = 1281;

    /** Fixed byte capacity of {@code turbo_error.message}. */
    public static final int TURBO_ERROR_MESSAGE_LEN = 496;

    /** Layout of {@code struct turbo_text { const char *ptr; uint64_t len; }} (16 bytes). */
    public static final StructLayout TURBO_TEXT_LAYOUT = MemoryLayout.structLayout(
            ValueLayout.ADDRESS.withName("ptr"),
            ValueLayout.JAVA_LONG.withName("len"));

    /** Layout of {@code struct turbo_error} (508 bytes). */
    public static final StructLayout TURBO_ERROR_LAYOUT = MemoryLayout.structLayout(
            ValueLayout.JAVA_INT.withName("struct_size"),
            ValueLayout.JAVA_INT.withName("code"),
            ValueLayout.JAVA_INT.withName("field"),
            MemoryLayout.sequenceLayout(TURBO_ERROR_MESSAGE_LEN, ValueLayout.JAVA_BYTE).withName("message"));

    /**
     * Layout of {@code struct turbo_analyzer_desc} (120 bytes). The five model path fields are
     * retired: they keep their place, and the native library refuses a non-empty value with
     * {@link #TURBO_E_INVALID_ARGUMENT}. Learned components come from {@code bundle_path}.
     */
    public static final StructLayout TURBO_ANALYZER_DESC_LAYOUT = MemoryLayout.structLayout(
            ValueLayout.JAVA_INT.withName("struct_size"),
            ValueLayout.JAVA_INT.withName("flags"),
            TURBO_TEXT_LAYOUT.withName("language"),
            TURBO_TEXT_LAYOUT.withName("sent_model_path"),
            TURBO_TEXT_LAYOUT.withName("tok_model_path"),
            TURBO_TEXT_LAYOUT.withName("pos_model_path"),
            TURBO_TEXT_LAYOUT.withName("lemma_model_path"),
            TURBO_TEXT_LAYOUT.withName("ner_model_path"),
            TURBO_TEXT_LAYOUT.withName("bundle_path"));

    /**
     * The {@code struct_size} of a {@code turbo_analyzer_desc} from before {@code bundle_path}
     * (104 bytes). The native library still accepts it; such a descriptor names no bundle.
     */
    public static final int TURBO_ANALYZER_DESC_SIZE_V1 = 104;

    /** Layout of {@code struct turbo_span} (120 bytes, alignment 8). */
    public static final StructLayout TURBO_SPAN_LAYOUT = MemoryLayout.structLayout(
            ValueLayout.JAVA_INT.withName("layer_kind"),
            ValueLayout.JAVA_INT.withName("flags"),
            ValueLayout.JAVA_INT.withName("item_index"),
            ValueLayout.JAVA_INT.withName("fragment_index"),
            ValueLayout.JAVA_INT.withName("view_id"),
            ValueLayout.JAVA_INT.withName("classification"),
            ValueLayout.JAVA_INT.withName("view_byte_start"),
            ValueLayout.JAVA_INT.withName("view_byte_end"),
            ValueLayout.JAVA_INT.withName("view_utf16_start"),
            ValueLayout.JAVA_INT.withName("view_utf16_end"),
            ValueLayout.JAVA_INT.withName("view_codepoint_start"),
            ValueLayout.JAVA_INT.withName("view_codepoint_end"),
            ValueLayout.JAVA_INT.withName("source_byte_start"),
            ValueLayout.JAVA_INT.withName("source_byte_end"),
            ValueLayout.JAVA_INT.withName("source_utf16_start"),
            ValueLayout.JAVA_INT.withName("source_utf16_end"),
            ValueLayout.JAVA_INT.withName("source_codepoint_start"),
            ValueLayout.JAVA_INT.withName("source_codepoint_end"),
            ValueLayout.JAVA_INT.withName("token_start"),
            ValueLayout.JAVA_INT.withName("token_end"),
            ValueLayout.JAVA_INT.withName("layer_offset"),
            ValueLayout.JAVA_INT.withName("layer_len"),
            ValueLayout.JAVA_INT.withName("instance_offset"),
            ValueLayout.JAVA_INT.withName("instance_len"),
            ValueLayout.JAVA_INT.withName("label_offset"),
            ValueLayout.JAVA_INT.withName("label_len"),
            ValueLayout.JAVA_INT.withName("value_offset"),
            ValueLayout.JAVA_INT.withName("value_len"),
            ValueLayout.JAVA_DOUBLE.withName("prob"));

    /** Layout of {@code struct turbo_text_pair} (32 bytes). */
    public static final StructLayout TURBO_TEXT_PAIR_LAYOUT = MemoryLayout.structLayout(
            TURBO_TEXT_LAYOUT.withName("query"),
            TURBO_TEXT_LAYOUT.withName("passage"));

    /** Size in bytes of {@code struct turbo_result_info} (488 bytes). */
    public static final int TURBO_RESULT_INFO_SIZE = 488;

    private static final Arena LIBRARY_ARENA = Arena.ofAuto();
    private static final Linker LINKER = Linker.nativeLinker();
    private static final SymbolLookup LOOKUP = loadSymbolLookup();

    /** {@code const char *turbo_version(void)} */
    public static final MethodHandle TURBO_VERSION = downcall("turbo_version",
            FunctionDescriptor.of(ValueLayout.ADDRESS));
    /** {@code const char *turbo_status_name(int32_t code)} */
    public static final MethodHandle TURBO_STATUS_NAME = downcall("turbo_status_name",
            FunctionDescriptor.of(ValueLayout.ADDRESS, ValueLayout.JAVA_INT));
    /** {@code const char *turbo_last_error_message(void)} */
    public static final MethodHandle TURBO_LAST_ERROR_MESSAGE = downcall("turbo_last_error_message",
            FunctionDescriptor.of(ValueLayout.ADDRESS));

    /** {@code int32_t turbo_runtime_create(const turbo_runtime_desc *, turbo_runtime **, turbo_error *)} */
    public static final MethodHandle TURBO_RUNTIME_CREATE = downcall("turbo_runtime_create",
            FunctionDescriptor.of(ValueLayout.JAVA_INT, ValueLayout.ADDRESS, ValueLayout.ADDRESS, ValueLayout.ADDRESS));
    /** {@code void turbo_runtime_release(turbo_runtime *)} */
    public static final MethodHandle TURBO_RUNTIME_RELEASE = downcall("turbo_runtime_release",
            FunctionDescriptor.ofVoid(ValueLayout.ADDRESS));
    /** {@code int32_t turbo_context_create(turbo_runtime *, uint32_t, turbo_context **, turbo_error *)} */
    public static final MethodHandle TURBO_CONTEXT_CREATE = downcall("turbo_context_create",
            FunctionDescriptor.of(ValueLayout.JAVA_INT, ValueLayout.ADDRESS, ValueLayout.JAVA_INT,
                    ValueLayout.ADDRESS, ValueLayout.ADDRESS));
    /** {@code void turbo_context_release(turbo_context *)} */
    public static final MethodHandle TURBO_CONTEXT_RELEASE = downcall("turbo_context_release",
            FunctionDescriptor.ofVoid(ValueLayout.ADDRESS));
    /** {@code int32_t turbo_model_load(turbo_context *, turbo_text, turbo_model **, turbo_error *)} */
    public static final MethodHandle TURBO_MODEL_LOAD = downcall("turbo_model_load",
            FunctionDescriptor.of(ValueLayout.JAVA_INT, ValueLayout.ADDRESS, TURBO_TEXT_LAYOUT,
                    ValueLayout.ADDRESS, ValueLayout.ADDRESS));
    /** {@code void turbo_model_release(turbo_model *)} */
    public static final MethodHandle TURBO_MODEL_RELEASE = downcall("turbo_model_release",
            FunctionDescriptor.ofVoid(ValueLayout.ADDRESS));
    /** {@code int32_t turbo_session_create(turbo_model *, const turbo_session_desc *, turbo_session **, turbo_error *)} */
    public static final MethodHandle TURBO_SESSION_CREATE = downcall("turbo_session_create",
            FunctionDescriptor.of(ValueLayout.JAVA_INT, ValueLayout.ADDRESS, ValueLayout.ADDRESS,
                    ValueLayout.ADDRESS, ValueLayout.ADDRESS));
    /** {@code void turbo_session_release(turbo_session *)} */
    public static final MethodHandle TURBO_SESSION_RELEASE = downcall("turbo_session_release",
            FunctionDescriptor.ofVoid(ValueLayout.ADDRESS));
    /** {@code int32_t turbo_embed_write_text(turbo_session *, const turbo_text *, uint32_t, const turbo_embed_options *, turbo_error *)} */
    public static final MethodHandle TURBO_EMBED_WRITE_TEXT = downcall("turbo_embed_write_text",
            FunctionDescriptor.of(ValueLayout.JAVA_INT, ValueLayout.ADDRESS, ValueLayout.ADDRESS,
                    ValueLayout.JAVA_INT, ValueLayout.ADDRESS, ValueLayout.ADDRESS));
    /** {@code int32_t turbo_rerank_write_pairs(turbo_session *, const turbo_text_pair *, uint32_t, const turbo_rerank_options *, turbo_error *)} */
    public static final MethodHandle TURBO_RERANK_WRITE_PAIRS = downcall("turbo_rerank_write_pairs",
            FunctionDescriptor.of(ValueLayout.JAVA_INT, ValueLayout.ADDRESS, ValueLayout.ADDRESS,
                    ValueLayout.JAVA_INT, ValueLayout.ADDRESS, ValueLayout.ADDRESS));
    /** {@code int32_t turbo_session_run(turbo_session *, turbo_result **, turbo_error *)} */
    public static final MethodHandle TURBO_SESSION_RUN = downcall("turbo_session_run",
            FunctionDescriptor.of(ValueLayout.JAVA_INT, ValueLayout.ADDRESS, ValueLayout.ADDRESS, ValueLayout.ADDRESS));
    /** {@code int32_t turbo_result_get_info(turbo_result *, turbo_result_info *, turbo_error *)} */
    public static final MethodHandle TURBO_RESULT_GET_INFO = downcall("turbo_result_get_info",
            FunctionDescriptor.of(ValueLayout.JAVA_INT, ValueLayout.ADDRESS, ValueLayout.ADDRESS, ValueLayout.ADDRESS));
    /** {@code int32_t turbo_result_read(turbo_result *, void *, uint64_t, uint64_t *, turbo_error *)} */
    public static final MethodHandle TURBO_RESULT_READ = downcall("turbo_result_read",
            FunctionDescriptor.of(ValueLayout.JAVA_INT, ValueLayout.ADDRESS, ValueLayout.ADDRESS,
                    ValueLayout.JAVA_LONG, ValueLayout.ADDRESS, ValueLayout.ADDRESS));
    /** {@code void turbo_result_release(turbo_result *)} */
    public static final MethodHandle TURBO_RESULT_RELEASE = downcall("turbo_result_release",
            FunctionDescriptor.ofVoid(ValueLayout.ADDRESS));

    /** {@code int32_t turbo_analyzer_create(const turbo_analyzer_desc *, turbo_analyzer **, turbo_error *)} */
    public static final MethodHandle TURBO_ANALYZER_CREATE = downcall("turbo_analyzer_create",
            FunctionDescriptor.of(ValueLayout.JAVA_INT, ValueLayout.ADDRESS, ValueLayout.ADDRESS, ValueLayout.ADDRESS));
    /** {@code turbo_analyzer *turbo_analyzer_retain(turbo_analyzer *)} */
    public static final MethodHandle TURBO_ANALYZER_RETAIN = downcall("turbo_analyzer_retain",
            FunctionDescriptor.of(ValueLayout.ADDRESS, ValueLayout.ADDRESS));
    /** {@code void turbo_analyzer_release(turbo_analyzer *)} */
    public static final MethodHandle TURBO_ANALYZER_RELEASE = downcall("turbo_analyzer_release",
            FunctionDescriptor.ofVoid(ValueLayout.ADDRESS));
    /** {@code int32_t turbo_analyzer_detect_sentences(turbo_analyzer *, turbo_text, turbo_analysis_result **, turbo_error *)} */
    public static final MethodHandle TURBO_ANALYZER_DETECT_SENTENCES = downcall("turbo_analyzer_detect_sentences",
            FunctionDescriptor.of(ValueLayout.JAVA_INT, ValueLayout.ADDRESS, TURBO_TEXT_LAYOUT,
                    ValueLayout.ADDRESS, ValueLayout.ADDRESS));
    /** {@code int32_t turbo_analyzer_tokenize(turbo_analyzer *, turbo_text, turbo_analysis_result **, turbo_error *)} */
    public static final MethodHandle TURBO_ANALYZER_TOKENIZE = downcall("turbo_analyzer_tokenize",
            FunctionDescriptor.of(ValueLayout.JAVA_INT, ValueLayout.ADDRESS, TURBO_TEXT_LAYOUT,
                    ValueLayout.ADDRESS, ValueLayout.ADDRESS));
    /** {@code int32_t turbo_analyzer_pos_tag(turbo_analyzer *, const turbo_text *, uint32_t, turbo_analysis_result **, turbo_error *)} */
    public static final MethodHandle TURBO_ANALYZER_POS_TAG = downcall("turbo_analyzer_pos_tag",
            FunctionDescriptor.of(ValueLayout.JAVA_INT, ValueLayout.ADDRESS, ValueLayout.ADDRESS,
                    ValueLayout.JAVA_INT, ValueLayout.ADDRESS, ValueLayout.ADDRESS));
    /** {@code int32_t turbo_analyzer_lemmatize(turbo_analyzer *, const turbo_text *, const turbo_text *, uint32_t, turbo_analysis_result **, turbo_error *)} */
    public static final MethodHandle TURBO_ANALYZER_LEMMATIZE = downcall("turbo_analyzer_lemmatize",
            FunctionDescriptor.of(ValueLayout.JAVA_INT, ValueLayout.ADDRESS, ValueLayout.ADDRESS,
                    ValueLayout.ADDRESS, ValueLayout.JAVA_INT, ValueLayout.ADDRESS, ValueLayout.ADDRESS));
    /** {@code int32_t turbo_analyzer_find_tokens(turbo_analyzer *, const turbo_text *, uint32_t, turbo_analysis_result **, turbo_error *)} */
    public static final MethodHandle TURBO_ANALYZER_FIND_TOKENS = downcall("turbo_analyzer_find_tokens",
            FunctionDescriptor.of(ValueLayout.JAVA_INT, ValueLayout.ADDRESS, ValueLayout.ADDRESS,
                    ValueLayout.JAVA_INT, ValueLayout.ADDRESS, ValueLayout.ADDRESS));
    /** {@code int32_t turbo_analyzer_find_text(turbo_analyzer *, turbo_text, turbo_analysis_result **, turbo_error *)} */
    public static final MethodHandle TURBO_ANALYZER_FIND_TEXT = downcall("turbo_analyzer_find_text",
            FunctionDescriptor.of(ValueLayout.JAVA_INT, ValueLayout.ADDRESS, TURBO_TEXT_LAYOUT,
                    ValueLayout.ADDRESS, ValueLayout.ADDRESS));
    /** {@code int32_t turbo_analyzer_detect_pii(turbo_analyzer *, turbo_text, turbo_analysis_result **, turbo_error *)} */
    public static final MethodHandle TURBO_ANALYZER_DETECT_PII = downcall("turbo_analyzer_detect_pii",
            FunctionDescriptor.of(ValueLayout.JAVA_INT, ValueLayout.ADDRESS, TURBO_TEXT_LAYOUT,
                    ValueLayout.ADDRESS, ValueLayout.ADDRESS));
    /** {@code int32_t turbo_analyzer_analyze_graph(turbo_analyzer *, turbo_text, turbo_analysis_result **, turbo_error *)} */
    public static final MethodHandle TURBO_ANALYZER_ANALYZE_GRAPH = downcall("turbo_analyzer_analyze_graph",
            FunctionDescriptor.of(ValueLayout.JAVA_INT, ValueLayout.ADDRESS, TURBO_TEXT_LAYOUT,
                    ValueLayout.ADDRESS, ValueLayout.ADDRESS));
    /** {@code int32_t turbo_analyzer_inspect_graph(turbo_analyzer *, const uint8_t *, uint64_t, turbo_analysis_result **, turbo_error *)} */
    public static final MethodHandle TURBO_ANALYZER_INSPECT_GRAPH = downcall("turbo_analyzer_inspect_graph",
            FunctionDescriptor.of(ValueLayout.JAVA_INT, ValueLayout.ADDRESS, ValueLayout.ADDRESS,
                    ValueLayout.JAVA_LONG, ValueLayout.ADDRESS, ValueLayout.ADDRESS));
    /** {@code int32_t turbo_analysis_result_spans(const turbo_analysis_result *, const turbo_span **, uint32_t *, turbo_error *)} */
    public static final MethodHandle TURBO_ANALYSIS_RESULT_SPANS = downcall("turbo_analysis_result_spans",
            FunctionDescriptor.of(ValueLayout.JAVA_INT, ValueLayout.ADDRESS, ValueLayout.ADDRESS,
                    ValueLayout.ADDRESS, ValueLayout.ADDRESS));
    /** {@code int32_t turbo_analysis_result_strings(const turbo_analysis_result *, const char **, uint64_t *, turbo_error *)} */
    public static final MethodHandle TURBO_ANALYSIS_RESULT_STRINGS = downcall("turbo_analysis_result_strings",
            FunctionDescriptor.of(ValueLayout.JAVA_INT, ValueLayout.ADDRESS, ValueLayout.ADDRESS,
                    ValueLayout.ADDRESS, ValueLayout.ADDRESS));
    /** {@code int32_t turbo_analysis_result_graph(const turbo_analysis_result *, const uint8_t **, uint64_t *, turbo_error *)} */
    public static final MethodHandle TURBO_ANALYSIS_RESULT_GRAPH = downcall("turbo_analysis_result_graph",
            FunctionDescriptor.of(ValueLayout.JAVA_INT, ValueLayout.ADDRESS, ValueLayout.ADDRESS,
                    ValueLayout.ADDRESS, ValueLayout.ADDRESS));
    /** {@code turbo_analysis_result *turbo_analysis_result_retain(turbo_analysis_result *)} */
    public static final MethodHandle TURBO_ANALYSIS_RESULT_RETAIN = downcall("turbo_analysis_result_retain",
            FunctionDescriptor.of(ValueLayout.ADDRESS, ValueLayout.ADDRESS));
    /** {@code void turbo_analysis_result_release(turbo_analysis_result *)} */
    public static final MethodHandle TURBO_ANALYSIS_RESULT_RELEASE = downcall("turbo_analysis_result_release",
            FunctionDescriptor.ofVoid(ValueLayout.ADDRESS));

    private TurboLibrary() {
    }

    private static MethodHandle downcall(String name, FunctionDescriptor descriptor) {
        MemorySegment symbol = LOOKUP.find(name)
                .orElseThrow(() -> new UnsatisfiedLinkError("Missing C ABI symbol in libturbo: " + name));
        return LINKER.downcallHandle(symbol, descriptor);
    }

    private static SymbolLookup loadSymbolLookup() {
        String libFile = System.mapLibraryName("turbo");

        String explicitPath = System.getProperty("turbo.library.path", System.getenv("TURBO_LIBRARY_PATH"));
        if (explicitPath != null && !explicitPath.isBlank()) {
            Path candidate = Path.of(explicitPath);
            if (Files.isDirectory(candidate)) {
                candidate = candidate.resolve(libFile);
            }
            if (!Files.isRegularFile(candidate)) {
                throw new UnsatisfiedLinkError("Configured libturbo path does not exist: " + candidate);
            }
            return SymbolLookup.libraryLookup(candidate.toAbsolutePath().normalize(), LIBRARY_ARENA);
        }

        List<Path> candidates = new ArrayList<>();
        String searchDirs = System.getProperty("turbo.library.dir", System.getenv("TURBO_LIBRARY_DIR"));
        boolean searchDirsConfigured = searchDirs != null && !searchDirs.isBlank();
        if (searchDirsConfigured) {
            for (String part : searchDirs.split(File.pathSeparator)) {
                if (!part.isBlank()) {
                    candidates.add(Path.of(part, libFile));
                }
            }
        }

        if (!searchDirsConfigured) {
            for (Path dir = Path.of("").toAbsolutePath().normalize(); dir != null; dir = dir.getParent()) {
                candidates.add(dir.resolve("target/debug").resolve(libFile));
                candidates.add(dir.resolve("target/release").resolve(libFile));
            }
        }

        for (Path candidate : candidates) {
            if (Files.isRegularFile(candidate)) {
                return SymbolLookup.libraryLookup(candidate.toAbsolutePath().normalize(), LIBRARY_ARENA);
            }
        }
        if (searchDirsConfigured) {
            throw new UnsatisfiedLinkError("libturbo was not found in configured directories: " + candidates);
        }
        return SymbolLookup.libraryLookup(libFile, LIBRARY_ARENA);
    }

    /**
     * Returns the {@code libturbo} version and linked backend list from {@code turbo_version()}.
     *
     * @return version string
     */
    public static String version() {
        try {
            MemorySegment ptr = (MemorySegment) TURBO_VERSION.invokeExact();
            return readCString(ptr);
        } catch (Throwable t) {
            throw new IllegalStateException("turbo_version invocation failed", t);
        }
    }

    /**
     * Returns the symbolic name for a {@code libturbo} status code from {@code turbo_status_name(code)}.
     *
     * @param code status code
     * @return status constant name
     */
    public static String statusName(int code) {
        try {
            MemorySegment ptr = (MemorySegment) TURBO_STATUS_NAME.invokeExact(code);
            return readCString(ptr);
        } catch (Throwable t) {
            throw new IllegalStateException("turbo_status_name invocation failed", t);
        }
    }

    /**
     * Returns the calling thread's most recent C ABI error message from {@code turbo_last_error_message()}.
     *
     * @return last error message on the current thread, or {@code ""}
     */
    public static String lastErrorMessage() {
        try {
            MemorySegment ptr = (MemorySegment) TURBO_LAST_ERROR_MESSAGE.invokeExact();
            return readCString(ptr);
        } catch (Throwable t) {
            throw new IllegalStateException("turbo_last_error_message invocation failed", t);
        }
    }

    /**
     * Allocates a zero-initialized {@code turbo_error} struct with {@code struct_size} populated.
     *
     * @param arena arena to allocate from
     * @return allocated {@code turbo_error} segment
     */
    public static MemorySegment allocateError(Arena arena) {
        MemorySegment err = arena.allocate(TURBO_ERROR_LAYOUT);
        err.set(ValueLayout.JAVA_INT, 0, (int) TURBO_ERROR_LAYOUT.byteSize());
        return err;
    }

    /**
     * Allocates a {@code turbo_text} struct pointing to UTF-8 bytes in {@code arena}.
     *
     * @param arena arena to allocate from
     * @param text  Java string (may be empty)
     * @return allocated {@code turbo_text} struct segment
     */
    public static MemorySegment allocateText(Arena arena, String text) {
        MemorySegment struct = arena.allocate(TURBO_TEXT_LAYOUT);
        writeTextInPlace(arena, struct, 0L, text);
        return struct;
    }

    /**
     * Writes a {@code turbo_text} view at {@code baseOffset} inside {@code container}.
     *
     * @param arena      arena for the UTF-8 byte buffer
     * @param container  target struct or array segment
     * @param baseOffset byte offset of the {@code turbo_text} within {@code container}
     * @param text       Java string (null or empty writes NULL ptr and 0 len)
     */
    public static void writeTextInPlace(Arena arena, MemorySegment container, long baseOffset, String text) {
        if (text == null || text.isEmpty()) {
            container.set(ValueLayout.ADDRESS, baseOffset, MemorySegment.NULL);
            container.set(ValueLayout.JAVA_LONG, baseOffset + 8L, 0L);
            return;
        }
        byte[] utf8 = text.getBytes(StandardCharsets.UTF_8);
        MemorySegment buf = arena.allocate(utf8.length, 1L);
        MemorySegment.copy(utf8, 0, buf, ValueLayout.JAVA_BYTE, 0L, utf8.length);
        container.set(ValueLayout.ADDRESS, baseOffset, buf);
        container.set(ValueLayout.JAVA_LONG, baseOffset + 8L, utf8.length);
    }

    /**
     * Allocates a contiguous {@code turbo_text[items.length]} array in {@code arena}.
     *
     * @param arena arena to allocate from
     * @param items string elements
     * @return pointer to the contiguous {@code turbo_text} array, or {@link MemorySegment#NULL} if empty
     */
    public static MemorySegment allocateTextArray(Arena arena, String[] items) {
        Objects.requireNonNull(arena, "arena");
        Objects.requireNonNull(items, "items");
        for (int i = 0; i < items.length; i++) {
            Objects.requireNonNull(items[i], "items[" + i + "]");
        }
        if (items.length == 0) {
            return MemorySegment.NULL;
        }
        long stride = TURBO_TEXT_LAYOUT.byteSize();
        MemorySegment array = arena.allocate(stride * items.length, TURBO_TEXT_LAYOUT.byteAlignment());
        for (int i = 0; i < items.length; i++) {
            writeTextInPlace(arena, array, i * stride, items[i]);
        }
        return array;
    }

    /**
     * Checks a C ABI status code and throws {@link TurboException} on failure.
     *
     * @param code status code returned by the C function
     * @param err  {@code turbo_error} segment passed to the call (may be {@link MemorySegment#NULL})
     */
    public static void checkStatus(int code, MemorySegment err) {
        if (code == TURBO_OK) {
            return;
        }
        int field = 0;
        String message = "";
        if (err != null && !MemorySegment.NULL.equals(err)) {
            field = err.get(ValueLayout.JAVA_INT, 8L);
            message = readCString(err.asSlice(12L, TURBO_ERROR_MESSAGE_LEN));
        }
        if (message.isEmpty()) {
            message = lastErrorMessage();
        }
        throw new TurboException(code, statusName(code), field, message);
    }

    /**
     * Reads a NUL-terminated UTF-8 string from a native pointer.
     *
     * @param ptr native pointer
     * @return decoded UTF-8 string
     */
    public static String readCString(MemorySegment ptr) {
        if (ptr == null || MemorySegment.NULL.equals(ptr)) {
            return "";
        }
        MemorySegment bounded = ptr.byteSize() == 0L ? ptr.reinterpret(4096L) : ptr;
        return bounded.getString(0L, StandardCharsets.UTF_8);
    }
}
