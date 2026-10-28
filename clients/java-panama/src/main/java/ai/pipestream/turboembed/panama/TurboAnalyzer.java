package ai.pipestream.turboembed.panama;

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.nio.charset.StandardCharsets;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Objects;
import java.util.concurrent.atomic.AtomicBoolean;

/**
 * In-process text analyzer backed by {@code libturbo} over the Java 22+ Panama Foreign Function and
 * Memory (FFM) API.
 *
 * <p>Provides sentence detection, tokenization, part-of-speech tagging, lemmatization, named entity
 * and PII extraction, and binary {@code DocumentGraph} ({@code "TGRF"}) wire serialization with
 * deterministic off-heap memory management.
 */
public final class TurboAnalyzer implements AutoCloseable {

    /**
     * Configuration for constructing a {@link TurboAnalyzer}.
     *
     * @param language            BCP-47 language tag (defaults to {@code "en"} when null or empty)
     * @param sentenceModelPath   optional path to an OpenNLP sentence detector {@code .bin} model
     * @param tokenizerModelPath  optional path to an OpenNLP tokenizer {@code .bin} model
     * @param posModelPath        optional path to an OpenNLP POS tagger {@code .bin} model
     * @param lemmatizerModelPath optional path to an OpenNLP lemmatizer {@code .bin} or dictionary file
     * @param nerModelPath        optional path to an OpenNLP {@code TokenNameFinder} {@code .bin} model
     */
    public record Config(
            String language,
            Path sentenceModelPath,
            Path tokenizerModelPath,
            Path posModelPath,
            Path lemmatizerModelPath,
            Path nerModelPath) {

        /**
         * Returns the default English configuration using built-in rules, dictionaries, and models.
         *
         * @return default configuration
         */
        public static Config defaults() {
            return new Config("en", null, null, null, null, null);
        }

        /**
         * Returns a configuration for the specified BCP-47 language tag.
         *
         * @param language BCP-47 language tag
         * @return configuration for {@code language}
         */
        public static Config forLanguage(String language) {
            return new Config(Objects.requireNonNull(language, "language"), null, null, null, null, null);
        }
    }

    private final AtomicBoolean closed = new AtomicBoolean(false);
    private MemorySegment handle;

    /**
     * Creates a {@code TurboAnalyzer} with the default English configuration.
     *
     * @return new analyzer instance
     */
    public static TurboAnalyzer create() {
        return create(Config.defaults());
    }

    /**
     * Creates a {@code TurboAnalyzer} with the given configuration.
     *
     * @param config analyzer configuration
     * @return new analyzer instance
     */
    public static TurboAnalyzer create(Config config) {
        Objects.requireNonNull(config, "config");
        try (Arena arena = Arena.ofConfined()) {
            MemorySegment desc = arena.allocate(TurboLibrary.TURBO_ANALYZER_DESC_LAYOUT);
            desc.set(ValueLayout.JAVA_INT, 0L, (int) TurboLibrary.TURBO_ANALYZER_DESC_LAYOUT.byteSize());
            desc.set(ValueLayout.JAVA_INT, 4L, 0);
            TurboLibrary.writeTextInPlace(arena, desc, 8L, config.language());
            TurboLibrary.writeTextInPlace(arena, desc, 24L, pathString(config.sentenceModelPath()));
            TurboLibrary.writeTextInPlace(arena, desc, 40L, pathString(config.tokenizerModelPath()));
            TurboLibrary.writeTextInPlace(arena, desc, 56L, pathString(config.posModelPath()));
            TurboLibrary.writeTextInPlace(arena, desc, 72L, pathString(config.lemmatizerModelPath()));
            TurboLibrary.writeTextInPlace(arena, desc, 88L, pathString(config.nerModelPath()));

            MemorySegment outPtr = arena.allocate(ValueLayout.ADDRESS);
            MemorySegment err = TurboLibrary.allocateError(arena);
            int code;
            try {
                code = (int) TurboLibrary.TURBO_ANALYZER_CREATE.invokeExact(desc, outPtr, err);
            } catch (Throwable t) {
                throw new IllegalStateException("turbo_analyzer_create failed", t);
            }
            TurboLibrary.checkStatus(code, err);
            return new TurboAnalyzer(outPtr.get(ValueLayout.ADDRESS, 0L));
        }
    }

    private TurboAnalyzer(MemorySegment handle) {
        this.handle = handle;
    }

    private static String pathString(Path p) {
        return p == null ? "" : p.toAbsolutePath().normalize().toString();
    }

    private MemorySegment requireOpen() {
        if (closed.get() || MemorySegment.NULL.equals(handle)) {
            throw new IllegalStateException("TurboAnalyzer is already closed");
        }
        return handle;
    }

    /**
     * Detects sentence spans in {@code text}.
     *
     * @param text UTF-8 input text
     * @return detected sentence spans with byte, UTF-16, and codepoint offsets
     */
    public List<TurboSpan> detectSentences(String text) {
        Objects.requireNonNull(text, "text");
        MemorySegment h = requireOpen();
        try (Arena arena = Arena.ofConfined()) {
            MemorySegment textSeg = TurboLibrary.allocateText(arena, text);
            MemorySegment outPtr = arena.allocate(ValueLayout.ADDRESS);
            MemorySegment err = TurboLibrary.allocateError(arena);
            int code;
            try {
                code = (int) TurboLibrary.TURBO_ANALYZER_DETECT_SENTENCES.invokeExact(h, textSeg, outPtr, err);
            } catch (Throwable t) {
                throw new IllegalStateException("turbo_analyzer_detect_sentences failed", t);
            }
            TurboLibrary.checkStatus(code, err);
            return consumeResultSpans(arena, outPtr.get(ValueLayout.ADDRESS, 0L));
        }
    }

    /**
     * Tokenizes {@code text} into {@link TurboSpan} records.
     *
     * @param text UTF-8 input text
     * @return token spans with byte, UTF-16, and codepoint offsets
     */
    public List<TurboSpan> tokenizeSpans(String text) {
        Objects.requireNonNull(text, "text");
        MemorySegment h = requireOpen();
        try (Arena arena = Arena.ofConfined()) {
            MemorySegment textSeg = TurboLibrary.allocateText(arena, text);
            MemorySegment outPtr = arena.allocate(ValueLayout.ADDRESS);
            MemorySegment err = TurboLibrary.allocateError(arena);
            int code;
            try {
                code = (int) TurboLibrary.TURBO_ANALYZER_TOKENIZE.invokeExact(h, textSeg, outPtr, err);
            } catch (Throwable t) {
                throw new IllegalStateException("turbo_analyzer_tokenize failed", t);
            }
            TurboLibrary.checkStatus(code, err);
            return consumeResultSpans(arena, outPtr.get(ValueLayout.ADDRESS, 0L));
        }
    }

    /**
     * Tokenizes {@code text} into {@link TurboToken} records.
     *
     * @param text UTF-8 input text
     * @return list of tokens with surface text and bi-directional offsets
     */
    public List<TurboToken> tokenize(String text) {
        List<TurboSpan> spans = tokenizeSpans(text);
        List<TurboToken> tokens = new ArrayList<>(spans.size());
        for (int i = 0; i < spans.size(); i++) {
            TurboSpan s = spans.get(i);
            tokens.add(new TurboToken(
                    i,
                    s.label(),
                    s.byteStart(),
                    s.byteEnd(),
                    s.utf16Start(),
                    s.utf16End(),
                    s.codepointStart(),
                    s.codepointEnd(),
                    s.prob()));
        }
        return tokens;
    }

    /**
     * Assigns part-of-speech tags and probabilities to a pre-tokenized sequence.
     *
     * @param tokens token strings
     * @return one {@link TurboSpan} per token whose {@code label()} is the POS tag and {@code prob()} is its probability
     */
    public List<TurboSpan> posTag(String[] tokens) {
        Objects.requireNonNull(tokens, "tokens");
        MemorySegment h = requireOpen();
        try (Arena arena = Arena.ofConfined()) {
            MemorySegment tokArray = TurboLibrary.allocateTextArray(arena, tokens);
            MemorySegment outPtr = arena.allocate(ValueLayout.ADDRESS);
            MemorySegment err = TurboLibrary.allocateError(arena);
            int code;
            try {
                code = (int) TurboLibrary.TURBO_ANALYZER_POS_TAG.invokeExact(
                        h, tokArray, tokens.length, outPtr, err);
            } catch (Throwable t) {
                throw new IllegalStateException("turbo_analyzer_pos_tag failed", t);
            }
            TurboLibrary.checkStatus(code, err);
            return consumeResultSpans(arena, outPtr.get(ValueLayout.ADDRESS, 0L));
        }
    }

    /**
     * Resolves lemmas and probabilities for a sequence of {@code (token, posTag)} pairs.
     *
     * @param tokens token surface strings
     * @param tags   part-of-speech tags aligned with {@code tokens}
     * @return one {@link TurboSpan} per token whose {@code label()} is the lemma and {@code prob()} is its confidence
     */
    public List<TurboSpan> lemmatize(String[] tokens, String[] tags) {
        Objects.requireNonNull(tokens, "tokens");
        Objects.requireNonNull(tags, "tags");
        if (tokens.length != tags.length) {
            throw new IllegalArgumentException(
                    "tokens.length (" + tokens.length + ") != tags.length (" + tags.length + ")");
        }
        MemorySegment h = requireOpen();
        try (Arena arena = Arena.ofConfined()) {
            MemorySegment tokArray = TurboLibrary.allocateTextArray(arena, tokens);
            MemorySegment tagArray = TurboLibrary.allocateTextArray(arena, tags);
            MemorySegment outPtr = arena.allocate(ValueLayout.ADDRESS);
            MemorySegment err = TurboLibrary.allocateError(arena);
            int code;
            try {
                code = (int) TurboLibrary.TURBO_ANALYZER_LEMMATIZE.invokeExact(
                        h, tokArray, tagArray, tokens.length, outPtr, err);
            } catch (Throwable t) {
                throw new IllegalStateException("turbo_analyzer_lemmatize failed", t);
            }
            TurboLibrary.checkStatus(code, err);
            return consumeResultSpans(arena, outPtr.get(ValueLayout.ADDRESS, 0L));
        }
    }

    /**
     * Finds named entity and regex spans over a pre-tokenized sequence.
     *
     * @param tokens token strings
     * @return detected entity spans with {@code [tokenStart, tokenEnd)} bounds
     */
    public List<TurboEntitySpan> findTokens(String[] tokens) {
        Objects.requireNonNull(tokens, "tokens");
        MemorySegment h = requireOpen();
        try (Arena arena = Arena.ofConfined()) {
            MemorySegment tokArray = TurboLibrary.allocateTextArray(arena, tokens);
            MemorySegment outPtr = arena.allocate(ValueLayout.ADDRESS);
            MemorySegment err = TurboLibrary.allocateError(arena);
            int code;
            try {
                code = (int) TurboLibrary.TURBO_ANALYZER_FIND_TOKENS.invokeExact(
                        h, tokArray, tokens.length, outPtr, err);
            } catch (Throwable t) {
                throw new IllegalStateException("turbo_analyzer_find_tokens failed", t);
            }
            TurboLibrary.checkStatus(code, err);
            List<TurboSpan> raw = consumeResultSpans(arena, outPtr.get(ValueLayout.ADDRESS, 0L));
            return toEntitySpans(raw);
        }
    }

    /**
     * Finds named entity and PII spans directly in {@code text}, populating UTF-8 byte, UTF-16
     * code-unit, Unicode scalar, and token offsets on every span.
     *
     * @param text UTF-8 input text
     * @return detected entity and PII spans
     */
    public List<TurboEntitySpan> findText(String text) {
        Objects.requireNonNull(text, "text");
        MemorySegment h = requireOpen();
        try (Arena arena = Arena.ofConfined()) {
            MemorySegment textSeg = TurboLibrary.allocateText(arena, text);
            MemorySegment outPtr = arena.allocate(ValueLayout.ADDRESS);
            MemorySegment err = TurboLibrary.allocateError(arena);
            int code;
            try {
                code = (int) TurboLibrary.TURBO_ANALYZER_FIND_TEXT.invokeExact(h, textSeg, outPtr, err);
            } catch (Throwable t) {
                throw new IllegalStateException("turbo_analyzer_find_text failed", t);
            }
            TurboLibrary.checkStatus(code, err);
            List<TurboSpan> raw = consumeResultSpans(arena, outPtr.get(ValueLayout.ADDRESS, 0L));
            return toEntitySpans(raw);
        }
    }

    /**
     * Runs the full multi-layer analysis pipeline over {@code text} and returns the serialized
     * binary {@code DocumentGraph} ({@code "TGRF"} wire format) and its decoded spans.
     *
     * @param text UTF-8 input text
     * @return binary {@code DocumentGraph} buffer and decoded spans
     */
    public DocumentGraphBuffer analyzeGraph(String text) {
        Objects.requireNonNull(text, "text");
        MemorySegment h = requireOpen();
        try (Arena arena = Arena.ofConfined()) {
            MemorySegment textSeg = TurboLibrary.allocateText(arena, text);
            MemorySegment outPtr = arena.allocate(ValueLayout.ADDRESS);
            MemorySegment err = TurboLibrary.allocateError(arena);
            int code;
            try {
                code = (int) TurboLibrary.TURBO_ANALYZER_ANALYZE_GRAPH.invokeExact(h, textSeg, outPtr, err);
            } catch (Throwable t) {
                throw new IllegalStateException("turbo_analyzer_analyze_graph failed", t);
            }
            TurboLibrary.checkStatus(code, err);
            return consumeGraphResult(arena, outPtr.get(ValueLayout.ADDRESS, 0L));
        }
    }

    /**
     * Validates a binary {@code DocumentGraph} ({@code "TGRF"}) wire buffer and decodes its
     * sentence, token, entity, and PII spans against {@code text}.
     *
     * @param text       source UTF-8 text
     * @param graphBytes binary {@code DocumentGraph} payload
     * @return validated {@code DocumentGraphBuffer} with decoded spans
     */
    public DocumentGraphBuffer inspectGraph(String text, byte[] graphBytes) {
        Objects.requireNonNull(text, "text");
        Objects.requireNonNull(graphBytes, "graphBytes");
        MemorySegment h = requireOpen();
        try (Arena arena = Arena.ofConfined()) {
            MemorySegment textSeg = TurboLibrary.allocateText(arena, text);
            MemorySegment graphSeg = graphBytes.length == 0
                    ? MemorySegment.NULL
                    : arena.allocate(graphBytes.length, 1L);
            if (graphBytes.length > 0) {
                MemorySegment.copy(graphBytes, 0, graphSeg, ValueLayout.JAVA_BYTE, 0L, graphBytes.length);
            }
            MemorySegment outPtr = arena.allocate(ValueLayout.ADDRESS);
            MemorySegment err = TurboLibrary.allocateError(arena);
            int code;
            try {
                code = (int) TurboLibrary.TURBO_ANALYZER_INSPECT_GRAPH.invokeExact(
                        h, textSeg, graphSeg, (long) graphBytes.length, outPtr, err);
            } catch (Throwable t) {
                throw new IllegalStateException("turbo_analyzer_inspect_graph failed", t);
            }
            TurboLibrary.checkStatus(code, err);
            return consumeGraphResult(arena, outPtr.get(ValueLayout.ADDRESS, 0L));
        }
    }

    private static List<TurboEntitySpan> toEntitySpans(List<TurboSpan> spans) {
        List<TurboEntitySpan> out = new ArrayList<>(spans.size());
        for (TurboSpan s : spans) {
            out.add(new TurboEntitySpan(
                    s.label(),
                    s.byteStart(),
                    s.byteEnd(),
                    s.utf16Start(),
                    s.utf16End(),
                    s.codepointStart(),
                    s.codepointEnd(),
                    s.tokenStart(),
                    s.tokenEnd(),
                    s.prob()));
        }
        return out;
    }

    private static List<TurboSpan> consumeResultSpans(Arena arena, MemorySegment resultPtr) {
        try {
            return readSpans(arena, resultPtr);
        } finally {
            releaseResult(resultPtr);
        }
    }

    private static DocumentGraphBuffer consumeGraphResult(Arena arena, MemorySegment resultPtr) {
        try {
            List<TurboSpan> spans = readSpans(arena, resultPtr);
            MemorySegment dataPtrOut = arena.allocate(ValueLayout.ADDRESS);
            MemorySegment lenOut = arena.allocate(ValueLayout.JAVA_LONG);
            MemorySegment err = TurboLibrary.allocateError(arena);
            int code;
            try {
                code = (int) TurboLibrary.TURBO_ANALYSIS_RESULT_GRAPH.invokeExact(
                        resultPtr, dataPtrOut, lenOut, err);
            } catch (Throwable t) {
                throw new IllegalStateException("turbo_analysis_result_graph failed", t);
            }
            TurboLibrary.checkStatus(code, err);
            long byteLen = lenOut.get(ValueLayout.JAVA_LONG, 0L);
            byte[] wireBytes;
            if (byteLen == 0L) {
                wireBytes = new byte[0];
            } else {
                MemorySegment dataSeg = dataPtrOut.get(ValueLayout.ADDRESS, 0L).reinterpret(byteLen);
                wireBytes = dataSeg.toArray(ValueLayout.JAVA_BYTE);
            }
            return new DocumentGraphBuffer(wireBytes, spans);
        } finally {
            releaseResult(resultPtr);
        }
    }

    private static List<TurboSpan> readSpans(Arena arena, MemorySegment resultPtr) {
        MemorySegment spansPtrOut = arena.allocate(ValueLayout.ADDRESS);
        MemorySegment countOut = arena.allocate(ValueLayout.JAVA_INT);
        MemorySegment err = TurboLibrary.allocateError(arena);
        int code;
        try {
            code = (int) TurboLibrary.TURBO_ANALYSIS_RESULT_SPANS.invokeExact(
                    resultPtr, spansPtrOut, countOut, err);
        } catch (Throwable t) {
            throw new IllegalStateException("turbo_analysis_result_spans failed", t);
        }
        TurboLibrary.checkStatus(code, err);
        int count = countOut.get(ValueLayout.JAVA_INT, 0L);
        if (count == 0) {
            return List.of();
        }

        MemorySegment poolPtrOut = arena.allocate(ValueLayout.ADDRESS);
        MemorySegment poolBytesOut = arena.allocate(ValueLayout.JAVA_LONG);
        try {
            code = (int) TurboLibrary.TURBO_ANALYSIS_RESULT_STRINGS.invokeExact(
                    resultPtr, poolPtrOut, poolBytesOut, err);
        } catch (Throwable t) {
            throw new IllegalStateException("turbo_analysis_result_strings failed", t);
        }
        TurboLibrary.checkStatus(code, err);

        long poolLen = poolBytesOut.get(ValueLayout.JAVA_LONG, 0L);
        byte[] pool = poolLen == 0L
                ? new byte[0]
                : poolPtrOut.get(ValueLayout.ADDRESS, 0L).reinterpret(poolLen).toArray(ValueLayout.JAVA_BYTE);

        long stride = TurboLibrary.TURBO_SPAN_LAYOUT.byteSize();
        MemorySegment spansSeg = spansPtrOut.get(ValueLayout.ADDRESS, 0L).reinterpret(stride * count);
        List<TurboSpan> list = new ArrayList<>(count);
        for (int i = 0; i < count; i++) {
            long base = i * stride;
            int byteStart = spansSeg.get(ValueLayout.JAVA_INT, base);
            int byteEnd = spansSeg.get(ValueLayout.JAVA_INT, base + 4L);
            int utf16Start = spansSeg.get(ValueLayout.JAVA_INT, base + 8L);
            int utf16End = spansSeg.get(ValueLayout.JAVA_INT, base + 12L);
            int cpStart = spansSeg.get(ValueLayout.JAVA_INT, base + 16L);
            int cpEnd = spansSeg.get(ValueLayout.JAVA_INT, base + 20L);
            int tokStart = spansSeg.get(ValueLayout.JAVA_INT, base + 24L);
            int tokEnd = spansSeg.get(ValueLayout.JAVA_INT, base + 28L);
            int labelOffset = spansSeg.get(ValueLayout.JAVA_INT, base + 32L);
            int labelLen = spansSeg.get(ValueLayout.JAVA_INT, base + 36L);
            double prob = spansSeg.get(ValueLayout.JAVA_DOUBLE, base + 40L);
            String label = labelLen == 0
                    ? ""
                    : new String(pool, labelOffset, labelLen, StandardCharsets.UTF_8);
            list.add(new TurboSpan(
                    byteStart, byteEnd, utf16Start, utf16End, cpStart, cpEnd, tokStart, tokEnd, label, prob));
        }
        return list;
    }

    private static void releaseResult(MemorySegment resultPtr) {
        if (resultPtr != null && !MemorySegment.NULL.equals(resultPtr)) {
            try {
                TurboLibrary.TURBO_ANALYSIS_RESULT_RELEASE.invokeExact(resultPtr);
            } catch (Throwable t) {
                throw new IllegalStateException("turbo_analysis_result_release failed", t);
            }
        }
    }

    @Override
    public void close() {
        if (closed.compareAndSet(false, true)) {
            MemorySegment h = handle;
            handle = MemorySegment.NULL;
            if (h != null && !MemorySegment.NULL.equals(h)) {
                try {
                    TurboLibrary.TURBO_ANALYZER_RELEASE.invokeExact(h);
                } catch (Throwable t) {
                    throw new IllegalStateException("turbo_analyzer_release failed", t);
                }
            }
        }
    }
}
