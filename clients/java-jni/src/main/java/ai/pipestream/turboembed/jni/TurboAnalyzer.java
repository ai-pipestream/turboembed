package ai.pipestream.turboembed.jni;

import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Objects;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.locks.ReentrantLock;

/**
 * In-process text analyzer backed by {@code libturbo} over the Java Native Interface.
 *
 * <p>Primary operations return a {@link TurboAnalysisResult}, which copies the native rows and
 * string pool into Java once, when it is made, and keeps the native result handle for
 * {@link TurboAnalysisResult#retain()} and the graph. Methods whose names end in {@code Copied}
 * materialize {@link TurboSpanData} and release the native result before returning.
 *
 * <p>Calls on one native analyzer are serialized, including calls made through analyzers made
 * by {@link #retain()}. A returned result is independent of this analyzer and remains usable
 * after the analyzer closes. The learned layers (part-of-speech tags, morphology, lemmas,
 * dependencies and entities) come from the heads of the bundle named by
 * {@link Config#bundlePath()}, all from one encoder pass per sentence. Without a bundle, or
 * with a bundle lacking a layer's head, that layer's tasks fail with
 * {@link TurboLibrary#TURBO_E_UNSUPPORTED_TASK} naming what is missing; no dictionary,
 * pattern, or lexical fallback is substituted in Java or native code.
 */
public final class TurboAnalyzer implements AutoCloseable {

    /**
     * Configuration for constructing a {@link TurboAnalyzer}.
     *
     * @param language   BCP-47 language tag (null or blank defaults to {@code "en"})
     * @param bundlePath optional bundle directory whose heads write the learned layers; null
     *                   configures the rule sentence splitter and tokenizer alone
     */
    public record Config(String language, Path bundlePath) {

        /** Normalizes an absent language to English. */
        public Config {
            language = language == null || isBlank(language) ? "en" : language;
        }

        /** Returns the rule sentence/tokenizer configuration with no bundle. */
        public static Config defaults() {
            return new Config("en", null);
        }

        /** Returns a configuration for one BCP-47 language tag and no bundle. */
        public static Config forLanguage(String language) {
            return new Config(Objects.requireNonNull(language, "language"), null);
        }

        /** Returns an English configuration over the heads of one bundle directory. */
        public static Config forBundle(Path bundlePath) {
            return new Config("en", Objects.requireNonNull(bundlePath, "bundlePath"));
        }

        private static boolean isBlank(String text) {
            for (int i = 0; i < text.length(); i++) {
                if (!Character.isWhitespace(text.charAt(i))) {
                    return false;
                }
            }
            return true;
        }
    }

    private final long handle;
    private final ReentrantLock invocationLock;
    private final AtomicBoolean closed = new AtomicBoolean(false);

    private TurboAnalyzer(long handle, ReentrantLock invocationLock) {
        this.handle = handle;
        this.invocationLock = Objects.requireNonNull(invocationLock, "invocationLock");
    }

    /** Creates an analyzer with the default English rule configuration. */
    public static TurboAnalyzer create() {
        return create(Config.defaults());
    }

    /** Creates an analyzer with the given configuration. */
    public static TurboAnalyzer create(Config config) {
        Objects.requireNonNull(config, "config");
        long created = TurboJni.analyzerCreate(
                Utf8.bytes(config.language()), Utf8.bytes(config.bundlePath() == null ? "" : config.bundlePath().toString()));
        return new TurboAnalyzer(requireHandle(created, "turbo_analyzer_create"), new ReentrantLock(true));
    }

    private static long requireHandle(long handle, String operation) {
        if (handle == 0L) {
            throw new IllegalStateException(operation + " returned NULL");
        }
        return handle;
    }

    /**
     * Creates another Java owner retaining this native analyzer. The two owners share the
     * invocation lock; closing one releases only its reference and leaves the other usable.
     */
    public TurboAnalyzer retain() {
        invocationLock.lock();
        try {
            requireOpen();
            return new TurboAnalyzer(requireHandle(TurboJni.analyzerRetain(handle), "turbo_analyzer_retain"), invocationLock);
        } finally {
            invocationLock.unlock();
        }
    }

    /** Returns whether this Java owner has released its analyzer reference. */
    public boolean isClosed() {
        return closed.get();
    }

    /** Detects sentences and returns the result with its rows copied once. */
    public TurboAnalysisResult detectSentences(String text) {
        Objects.requireNonNull(text, "text");
        return execute("turbo_analyzer_detect_sentences", () -> TurboJni.analyzerDetectSentences(handle, Utf8.bytes(text)));
    }

    /** Detects sentences and explicitly copies every resulting row. */
    public List<TurboSpanData> detectSentencesCopied(String text) {
        try (TurboAnalysisResult result = detectSentences(text)) {
            return result.copySpans();
        }
    }

    /** Tokenizes text and returns the result with its rows copied once. */
    public TurboAnalysisResult tokenize(String text) {
        Objects.requireNonNull(text, "text");
        return execute("turbo_analyzer_tokenize", () -> TurboJni.analyzerTokenize(handle, Utf8.bytes(text)));
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
            List<String> out = new ArrayList<>();
            for (TurboSpanView span : result.spans(TurboLayerKind.TOKENS)) {
                out.add(span.coveredSourceText(text));
            }
            return Collections.unmodifiableList(out);
        }
    }

    /**
     * Tags one independent pre-tokenized sequence with the bundle's part-of-speech head.
     * Rows are token-only and carry no view or source range.
     */
    public TurboAnalysisResult posTag(String[] tokens) {
        Utf8.Packed packed = Utf8.pack(tokens, "tokens");
        return execute("turbo_analyzer_pos_tag", () -> TurboJni.analyzerPosTag(handle, packed.bytes, packed.lengths));
    }

    /** Tags tokens and explicitly copies every resulting row. */
    public List<TurboSpanData> posTagCopied(String[] tokens) {
        try (TurboAnalysisResult result = posTag(tokens)) {
            return result.copySpans();
        }
    }

    /**
     * Tags one independent pre-tokenized sequence with the bundle's morphology head.
     * Rows are token-only and carry one row per token, with the feature string as the label.
     */
    public TurboAnalysisResult morphology(String[] tokens) {
        Utf8.Packed packed = Utf8.pack(tokens, "tokens");
        return execute("turbo_analyzer_morphology", () -> TurboJni.analyzerMorphology(handle, packed.bytes, packed.lengths));
    }

    /** Tags tokens for morphology and explicitly copies every resulting row. */
    public List<TurboSpanData> morphologyCopied(String[] tokens) {
        try (TurboAnalysisResult result = morphology(tokens)) {
            return result.copySpans();
        }
    }

    /**
     * Lemmatizes aligned token/tag pairs with the bundle's lemmatizer head. The tags must be
     * labels of the tag head the lemmatizer reads; they are ignored by a lemmatizer that reads
     * no tags. A row with no label is an explicit miss, never a lowercased token.
     */
    public TurboAnalysisResult lemmatize(String[] tokens, String[] tags) {
        Utf8.Packed packedTokens = Utf8.pack(tokens, "tokens");
        Utf8.Packed packedTags = Utf8.pack(tags, "tags");
        if (tokens.length != tags.length) {
            throw new IllegalArgumentException(
                    "tokens.length (" + tokens.length + ") != tags.length (" + tags.length + ")");
        }
        return execute("turbo_analyzer_lemmatize", () -> TurboJni.analyzerLemmatize(
                handle, packedTokens.bytes, packedTokens.lengths, packedTags.bytes, packedTags.lengths));
    }

    /** Lemmatizes tokens and explicitly copies every resulting row. */
    public List<TurboSpanData> lemmatizeCopied(String[] tokens, String[] tags) {
        try (TurboAnalysisResult result = lemmatize(tokens, tags)) {
            return result.copySpans();
        }
    }

    /**
     * Finds named entities in one independent token sequence with the bundle's entity head only.
     * No adaptive data is retained between calls.
     */
    public TurboAnalysisResult findTokens(String[] tokens) {
        Utf8.Packed packed = Utf8.pack(tokens, "tokens");
        return execute("turbo_analyzer_find_tokens", () -> TurboJni.analyzerFindTokens(handle, packed.bytes, packed.lengths));
    }

    /** Finds entities in tokens and explicitly copies every resulting row. */
    public List<TurboSpanData> findTokensCopied(String[] tokens) {
        try (TurboAnalysisResult result = findTokens(tokens)) {
            return result.copySpans();
        }
    }

    /** Finds named entities directly in text with the bundle's entity head only. */
    public TurboAnalysisResult findText(String text) {
        Objects.requireNonNull(text, "text");
        return execute("turbo_analyzer_find_text", () -> TurboJni.analyzerFindText(handle, Utf8.bytes(text)));
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
        return execute("turbo_analyzer_detect_pii", () -> TurboJni.analyzerDetectPii(handle, Utf8.bytes(text)));
    }

    /** Detects PII and explicitly copies every resulting row. */
    public List<TurboSpanData> detectPiiCopied(String text) {
        try (TurboAnalysisResult result = detectPii(text)) {
            return result.copySpans();
        }
    }

    /**
     * Runs the rule splitter and tokenizer, every head of the bundle in one encoder pass per
     * sentence, and the PII task; serializes the binary graph into the result; and exposes typed
     * rows for every layer instance present. A layer without a head is absent.
     */
    public TurboAnalysisResult analyzeGraph(String text) {
        Objects.requireNonNull(text, "text");
        return execute("turbo_analyzer_analyze_graph", () -> TurboJni.analyzerAnalyzeGraph(handle, Utf8.bytes(text)));
    }

    /** Validates and decodes a graph using the source text stored inside the graph itself. */
    public TurboAnalysisResult inspectGraph(byte[] graphBytes) {
        Objects.requireNonNull(graphBytes, "graphBytes");
        return inspectGraph(graphBytes, 0, graphBytes.length);
    }

    /** Validates and decodes one checked slice of a graph byte array. */
    public TurboAnalysisResult inspectGraph(byte[] graphBytes, int offset, int length) {
        Objects.requireNonNull(graphBytes, "graphBytes");
        if (offset < 0 || length < 0 || offset > graphBytes.length - length) {
            throw new IndexOutOfBoundsException(
                    "graph slice [" + offset + ", " + offset + " + " + length + ") outside 0.." + graphBytes.length);
        }
        return execute("turbo_analyzer_inspect_graph", () -> TurboJni.analyzerInspectGraph(handle, graphBytes, offset, length));
    }

    @FunctionalInterface
    private interface NativeResultCall {
        long invoke();
    }

    private TurboAnalysisResult execute(String operation, NativeResultCall call) {
        invocationLock.lock();
        try {
            requireOpen();
            return new TurboAnalysisResult(requireHandle(call.invoke(), operation));
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
            TurboJni.analyzerRelease(handle);
        } finally {
            invocationLock.unlock();
        }
    }
}
