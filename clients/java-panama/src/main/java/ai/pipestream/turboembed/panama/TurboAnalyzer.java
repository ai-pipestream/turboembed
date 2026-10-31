package ai.pipestream.turboembed.panama;

import java.lang.foreign.Arena;
import java.lang.foreign.MemoryLayout;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.nio.file.Path;
import java.util.List;
import java.util.Objects;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.locks.ReentrantLock;

/**
 * In-process text analyzer backed by {@code libturbo} over Panama FFM.
 *
 * <p>Primary operations return a lifetime-bound {@link TurboAnalysisResult}. They do not copy
 * the native span array, string pool, or graph on return. Methods whose names end in
 * {@code Copied} explicitly materialize {@link TurboSpanData} and release the native result
 * before returning.
 *
 * <p>Calls on one native analyzer are serialized, including calls made through analyzers made
 * by {@link #retain()}. A returned result is independent of this analyzer and remains usable
 * after the analyzer closes. Unconfigured POS, lemma, and NER models fail with
 * {@link TurboLibrary#TURBO_E_UNSUPPORTED_TASK}; no dictionary, pattern, or lexical fallback is
 * substituted in Java or native code.
 */
public final class TurboAnalyzer implements AutoCloseable {

    /**
     * Configuration for constructing a {@link TurboAnalyzer}.
     *
     * @param language            BCP-47 language tag (null or blank defaults to {@code "en"})
     * @param sentenceModelPath   optional OpenNLP sentence detector {@code .bin} model
     * @param tokenizerModelPath  optional OpenNLP tokenizer {@code .bin} model
     * @param posModelPath        optional OpenNLP POS tagger {@code .bin} model
     * @param lemmatizerModelPath optional OpenNLP lemmatizer {@code .bin} or dictionary file
     * @param nerModelPath        optional OpenNLP {@code TokenNameFinder} {@code .bin} model
     */
    public record Config(
            String language,
            Path sentenceModelPath,
            Path tokenizerModelPath,
            Path posModelPath,
            Path lemmatizerModelPath,
            Path nerModelPath) {

        /** Normalizes an absent language to English. */
        public Config {
            language = language == null || language.isBlank() ? "en" : language;
        }

        /** Returns the rule sentence/tokenizer configuration with no statistical task models. */
        public static Config defaults() {
            return new Config("en", null, null, null, null, null);
        }

        /** Returns a configuration for one BCP-47 language tag and no task models. */
        public static Config forLanguage(String language) {
            return new Config(Objects.requireNonNull(language, "language"), null, null, null, null, null);
        }
    }

    private final MemorySegment handle;
    private final ReentrantLock invocationLock;
    private final AtomicBoolean closed = new AtomicBoolean(false);

    private TurboAnalyzer(MemorySegment handle, ReentrantLock invocationLock) {
        this.handle = Objects.requireNonNull(handle, "handle");
        this.invocationLock = Objects.requireNonNull(invocationLock, "invocationLock");
    }

    /** Creates an analyzer with the default English rule configuration. */
    public static TurboAnalyzer create() {
        return create(Config.defaults());
    }

    /** Creates an analyzer with the given configuration. */
    public static TurboAnalyzer create(Config config) {
        Objects.requireNonNull(config, "config");
        try (Arena arena = Arena.ofConfined()) {
            MemorySegment desc = arena.allocate(TurboLibrary.TURBO_ANALYZER_DESC_LAYOUT);
            desc.set(ValueLayout.JAVA_INT, descOffset("struct_size"), (int) TurboLibrary.TURBO_ANALYZER_DESC_LAYOUT.byteSize());
            desc.set(ValueLayout.JAVA_INT, descOffset("flags"), 0);
            TurboLibrary.writeTextInPlace(arena, desc, descOffset("language"), config.language());
            TurboLibrary.writeTextInPlace(arena, desc, descOffset("sent_model_path"), pathString(config.sentenceModelPath()));
            TurboLibrary.writeTextInPlace(arena, desc, descOffset("tok_model_path"), pathString(config.tokenizerModelPath()));
            TurboLibrary.writeTextInPlace(arena, desc, descOffset("pos_model_path"), pathString(config.posModelPath()));
            TurboLibrary.writeTextInPlace(arena, desc, descOffset("lemma_model_path"), pathString(config.lemmatizerModelPath()));
            TurboLibrary.writeTextInPlace(arena, desc, descOffset("ner_model_path"), pathString(config.nerModelPath()));

            MemorySegment outPtr = arena.allocate(ValueLayout.ADDRESS);
            MemorySegment err = TurboLibrary.allocateError(arena);
            int code;
            try {
                code = (int) TurboLibrary.TURBO_ANALYZER_CREATE.invokeExact(desc, outPtr, err);
            } catch (Throwable t) {
                throw new IllegalStateException("turbo_analyzer_create failed", t);
            }
            TurboLibrary.checkStatus(code, err);
            MemorySegment created = outPtr.get(ValueLayout.ADDRESS, 0L);
            if (created == null || MemorySegment.NULL.equals(created)) {
                throw new IllegalStateException("turbo_analyzer_create returned NULL");
            }
            return new TurboAnalyzer(created, new ReentrantLock(true));
        }
    }

    private static long descOffset(String field) {
        return TurboLibrary.TURBO_ANALYZER_DESC_LAYOUT.byteOffset(MemoryLayout.PathElement.groupElement(field));
    }

    private static String pathString(Path path) {
        return path == null ? "" : path.toString();
    }

    /**
     * Creates another Java owner retaining this native analyzer. The two owners share the
     * invocation lock; closing one releases only its reference and leaves the other usable.
     */
    public TurboAnalyzer retain() {
        invocationLock.lock();
        try {
            requireOpen();
            MemorySegment retained;
            try {
                retained = (MemorySegment) TurboLibrary.TURBO_ANALYZER_RETAIN.invokeExact(handle);
            } catch (Throwable t) {
                throw new IllegalStateException("turbo_analyzer_retain failed", t);
            }
            if (retained == null || MemorySegment.NULL.equals(retained)) {
                throw new IllegalStateException("turbo_analyzer_retain returned NULL");
            }
            return new TurboAnalyzer(retained, invocationLock);
        } finally {
            invocationLock.unlock();
        }
    }

    /** Returns whether this Java owner has released its analyzer reference. */
    public boolean isClosed() {
        return closed.get();
    }

    /** Detects sentences and returns the native result without copying its spans or pool. */
    public TurboAnalysisResult detectSentences(String text) {
        Objects.requireNonNull(text, "text");
        return execute("turbo_analyzer_detect_sentences", (arena, outPtr, err) -> {
            MemorySegment textSeg = TurboLibrary.allocateText(arena, text);
            return (int) TurboLibrary.TURBO_ANALYZER_DETECT_SENTENCES.invokeExact(handle, textSeg, outPtr, err);
        });
    }

    /** Detects sentences and explicitly copies every resulting row. */
    public List<TurboSpanData> detectSentencesCopied(String text) {
        try (TurboAnalysisResult result = detectSentences(text)) {
            return result.copySpans();
        }
    }

    /** Tokenizes text and returns the native result without copying its spans or pool. */
    public TurboAnalysisResult tokenize(String text) {
        Objects.requireNonNull(text, "text");
        return execute("turbo_analyzer_tokenize", (arena, outPtr, err) -> {
            MemorySegment textSeg = TurboLibrary.allocateText(arena, text);
            return (int) TurboLibrary.TURBO_ANALYZER_TOKENIZE.invokeExact(handle, textSeg, outPtr, err);
        });
    }

    /** Tokenizes text and explicitly copies every resulting row. */
    public List<TurboSpanData> tokenizeCopied(String text) {
        try (TurboAnalysisResult result = tokenize(text)) {
            return result.copySpans();
        }
    }

    /** Tokenizes text and copies each token's covered source text into Java strings. */
    public List<String> tokenizeToStrings(String text) {
        Objects.requireNonNull(text, "text");
        try (TurboAnalysisResult result = tokenize(text)) {
            return result.spans(TurboLayerKind.TOKENS).stream()
                    .map(span -> span.coveredSourceText(text))
                    .toList();
        }
    }

    /**
     * Tags one independent pre-tokenized sequence with the configured POS model.
     * Rows are token-only and carry no view or source range.
     */
    public TurboAnalysisResult posTag(String[] tokens) {
        validateStrings(tokens, "tokens");
        return execute("turbo_analyzer_pos_tag", (arena, outPtr, err) -> {
            MemorySegment tokenArray = TurboLibrary.allocateTextArray(arena, tokens);
            return (int) TurboLibrary.TURBO_ANALYZER_POS_TAG.invokeExact(
                    handle, tokenArray, tokens.length, outPtr, err);
        });
    }

    /** Tags tokens and explicitly copies every resulting row. */
    public List<TurboSpanData> posTagCopied(String[] tokens) {
        try (TurboAnalysisResult result = posTag(tokens)) {
            return result.copySpans();
        }
    }

    /**
     * Lemmatizes aligned token/tag pairs with the configured dictionary or model.
     * A dictionary miss is a row with no label, never a lowercased token.
     */
    public TurboAnalysisResult lemmatize(String[] tokens, String[] tags) {
        validateStrings(tokens, "tokens");
        validateStrings(tags, "tags");
        if (tokens.length != tags.length) {
            throw new IllegalArgumentException(
                    "tokens.length (" + tokens.length + ") != tags.length (" + tags.length + ")");
        }
        return execute("turbo_analyzer_lemmatize", (arena, outPtr, err) -> {
            MemorySegment tokenArray = TurboLibrary.allocateTextArray(arena, tokens);
            MemorySegment tagArray = TurboLibrary.allocateTextArray(arena, tags);
            return (int) TurboLibrary.TURBO_ANALYZER_LEMMATIZE.invokeExact(
                    handle, tokenArray, tagArray, tokens.length, outPtr, err);
        });
    }

    /** Lemmatizes tokens and explicitly copies every resulting row. */
    public List<TurboSpanData> lemmatizeCopied(String[] tokens, String[] tags) {
        try (TurboAnalysisResult result = lemmatize(tokens, tags)) {
            return result.copySpans();
        }
    }

    /**
     * Finds named entities in one independent token sequence with the configured model only.
     * No adaptive data is retained between calls.
     */
    public TurboAnalysisResult findTokens(String[] tokens) {
        validateStrings(tokens, "tokens");
        return execute("turbo_analyzer_find_tokens", (arena, outPtr, err) -> {
            MemorySegment tokenArray = TurboLibrary.allocateTextArray(arena, tokens);
            return (int) TurboLibrary.TURBO_ANALYZER_FIND_TOKENS.invokeExact(
                    handle, tokenArray, tokens.length, outPtr, err);
        });
    }

    /** Finds entities in tokens and explicitly copies every resulting row. */
    public List<TurboSpanData> findTokensCopied(String[] tokens) {
        try (TurboAnalysisResult result = findTokens(tokens)) {
            return result.copySpans();
        }
    }

    /** Finds named entities directly in text with the configured model only. */
    public TurboAnalysisResult findText(String text) {
        Objects.requireNonNull(text, "text");
        return execute("turbo_analyzer_find_text", (arena, outPtr, err) -> {
            MemorySegment textSeg = TurboLibrary.allocateText(arena, text);
            return (int) TurboLibrary.TURBO_ANALYZER_FIND_TEXT.invokeExact(handle, textSeg, outPtr, err);
        });
    }

    /** Finds entities in text and explicitly copies every resulting row. */
    public List<TurboSpanData> findTextCopied(String text) {
        try (TurboAnalysisResult result = findText(text)) {
            return result.copySpans();
        }
    }

    /** Detects PII through the separate PII task; it is not merged into NER. */
    public TurboAnalysisResult detectPii(String text) {
        Objects.requireNonNull(text, "text");
        return execute("turbo_analyzer_detect_pii", (arena, outPtr, err) -> {
            MemorySegment textSeg = TurboLibrary.allocateText(arena, text);
            return (int) TurboLibrary.TURBO_ANALYZER_DETECT_PII.invokeExact(handle, textSeg, outPtr, err);
        });
    }

    /** Detects PII and explicitly copies every resulting row. */
    public List<TurboSpanData> detectPiiCopied(String text) {
        try (TurboAnalysisResult result = detectPii(text)) {
            return result.copySpans();
        }
    }

    /**
     * Runs configured layers, serializes the binary graph into the result, and exposes typed
     * rows for every supported layer instance present. Unconfigured layers are absent.
     */
    public TurboAnalysisResult analyzeGraph(String text) {
        Objects.requireNonNull(text, "text");
        return execute("turbo_analyzer_analyze_graph", (arena, outPtr, err) -> {
            MemorySegment textSeg = TurboLibrary.allocateText(arena, text);
            return (int) TurboLibrary.TURBO_ANALYZER_ANALYZE_GRAPH.invokeExact(handle, textSeg, outPtr, err);
        });
    }

    /** Validates and decodes a graph using the source text stored inside the graph itself. */
    public TurboAnalysisResult inspectGraph(byte[] graphBytes) {
        Objects.requireNonNull(graphBytes, "graphBytes");
        return inspectGraph(graphBytes, 0, graphBytes.length);
    }

    /** Validates and decodes one checked slice of a graph byte array. */
    public TurboAnalysisResult inspectGraph(byte[] graphBytes, int offset, int length) {
        Objects.requireNonNull(graphBytes, "graphBytes");
        Objects.checkFromIndexSize(offset, length, graphBytes.length);
        return execute("turbo_analyzer_inspect_graph", (arena, outPtr, err) -> {
            MemorySegment graphSeg = length == 0 ? MemorySegment.NULL : arena.allocate(length, 1L);
            if (length > 0) {
                MemorySegment.copy(graphBytes, offset, graphSeg, ValueLayout.JAVA_BYTE, 0L, length);
            }
            return (int) TurboLibrary.TURBO_ANALYZER_INSPECT_GRAPH.invokeExact(
                    handle, graphSeg, (long) length, outPtr, err);
        });
    }

    private static void validateStrings(String[] values, String name) {
        Objects.requireNonNull(values, name);
        for (int i = 0; i < values.length; i++) {
            Objects.requireNonNull(values[i], name + "[" + i + "]");
        }
    }

    @FunctionalInterface
    private interface NativeResultCall {
        int invoke(Arena arena, MemorySegment outPtr, MemorySegment err) throws Throwable;
    }

    private TurboAnalysisResult execute(String operation, NativeResultCall call) {
        invocationLock.lock();
        try {
            requireOpen();
            try (Arena arena = Arena.ofConfined()) {
                MemorySegment outPtr = arena.allocate(ValueLayout.ADDRESS);
                MemorySegment err = TurboLibrary.allocateError(arena);
                int code;
                try {
                    code = call.invoke(arena, outPtr, err);
                } catch (Throwable t) {
                    throw new IllegalStateException(operation + " failed", t);
                }
                TurboLibrary.checkStatus(code, err);
                MemorySegment result = outPtr.get(ValueLayout.ADDRESS, 0L);
                if (result == null || MemorySegment.NULL.equals(result)) {
                    throw new IllegalStateException(operation + " returned NULL");
                }
                return new TurboAnalysisResult(result);
            }
        } finally {
            invocationLock.unlock();
        }
    }

    private void requireOpen() {
        if (closed.get()) {
            throw new IllegalStateException("TurboAnalyzer is already closed");
        }
    }

    @Override
    public void close() {
        if (!closed.compareAndSet(false, true)) {
            return;
        }
        invocationLock.lock();
        try {
            try {
                TurboLibrary.TURBO_ANALYZER_RELEASE.invokeExact(handle);
            } catch (Throwable t) {
                throw new IllegalStateException("turbo_analyzer_release failed", t);
            }
        } finally {
            invocationLock.unlock();
        }
    }
}
