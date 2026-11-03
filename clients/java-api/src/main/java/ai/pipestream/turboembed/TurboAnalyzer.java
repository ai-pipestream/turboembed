package ai.pipestream.turboembed;

import java.nio.file.Path;
import java.util.List;
import java.util.Objects;

/** Shared analyzer contract. Results own their lifetime independently of the analyzer.
 * Calls sharing a retained native analyzer are serialized by its implementation. */
public interface TurboAnalyzer extends AutoCloseable {
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

    /** Creates the default analyzer using the selected runtime bridge. */
    static TurboAnalyzer create() { return create(Config.defaults()); }

    /** Creates an analyzer through the selected runtime bridge. */
    static TurboAnalyzer create(Config config) { return TurboRuntime.load().createAnalyzer(config); }

    /**
     * Creates another Java owner retaining this native analyzer. The two owners share the
     * invocation lock; closing one releases only its reference and leaves the other usable.
     */
    TurboAnalyzer retain();

    /** Returns whether this Java owner has released its analyzer reference. */
    boolean isClosed();

    /** Detects sentences and returns an owned result. */
    TurboAnalysisResult detectSentences(String text);

    /** Detects sentences and explicitly copies every resulting row. */
    List<TurboSpanData> detectSentencesCopied(String text);

    /** Tokenizes text and returns an owned result. */
    TurboAnalysisResult tokenize(String text);

    /** Tokenizes text and explicitly copies every resulting row. */
    List<TurboSpanData> tokenizeCopied(String text);

    /** Tokenizes text and copies each token's covered source text into Java strings. */
    List<String> tokenizeToStrings(String text);

    /**
     * Tags one independent pre-tokenized sequence with the bundle's part-of-speech head.
     * Rows are token-only and carry no view or source range.
     */
    TurboAnalysisResult posTag(String[] tokens);

    /** Tags tokens and explicitly copies every resulting row. */
    List<TurboSpanData> posTagCopied(String[] tokens);

    /**
     * Tags one independent pre-tokenized sequence with the bundle's morphology head.
     * Rows are token-only and carry one row per token, with the feature string as the label.
     */
    TurboAnalysisResult morphology(String[] tokens);

    /** Tags tokens for morphology and explicitly copies every resulting row. */
    List<TurboSpanData> morphologyCopied(String[] tokens);

    /**
     * Lemmatizes aligned token/tag pairs with the bundle's lemmatizer head. The tags must be
     * labels of the tag head the lemmatizer reads; they are ignored by a lemmatizer that reads
     * no tags. A row with no label is an explicit miss, never a lowercased token.
     */
    TurboAnalysisResult lemmatize(String[] tokens, String[] tags);

    /** Lemmatizes tokens and explicitly copies every resulting row. */
    List<TurboSpanData> lemmatizeCopied(String[] tokens, String[] tags);

    /**
     * Finds named entities in one independent token sequence with the bundle's entity head only.
     * No adaptive data is retained between calls.
     */
    TurboAnalysisResult findTokens(String[] tokens);

    /** Finds entities in tokens and explicitly copies every resulting row. */
    List<TurboSpanData> findTokensCopied(String[] tokens);

    /** Finds named entities directly in text with the bundle's entity head only. */
    TurboAnalysisResult findText(String text);

    /** Finds entities in text and explicitly copies every resulting row. */
    List<TurboSpanData> findTextCopied(String text);

    /** Detects PII through the separate PII task; it is not merged into NER. */
    TurboAnalysisResult detectPii(String text);

    /** Detects PII and explicitly copies every resulting row. */
    List<TurboSpanData> detectPiiCopied(String text);

    /**
     * Runs the rule splitter and tokenizer, every head of the bundle in one encoder pass per
     * sentence, and the PII task; serializes the binary graph into the result; and exposes typed
     * rows for every layer instance present. A layer without a head is absent.
     */
    TurboAnalysisResult analyzeGraph(String text);

    /** Validates and decodes a graph using the source text stored inside the graph itself. */
    TurboAnalysisResult inspectGraph(byte[] graphBytes);

    /** Validates and decodes one checked slice of a graph byte array. */
    TurboAnalysisResult inspectGraph(byte[] graphBytes, int offset, int length);

    void close();
}
