package ai.pipestream.turboembed;

import java.util.Arrays;

/**
 * How {@link TurboSession#embed(java.util.List, TurboEmbedOptions)} turns each text into a vector,
 * as {@code turbo_embed_options}: a {@code MODEL} value or a 0 means what the bundle says.
 *
 * @param truncate     how a text over {@code maxTokens} is cut
 * @param maxTokens    a row's token budget, special tokens included; 0 for the bundle's
 * @param promptRole   which of the bundle's prefixes goes before the text
 * @param normalize    how each vector is normalized
 * @param pooling      how each text's token states are pooled
 * @param outputDim    keep only the first {@code outputDim} values, cut before normalizing; 0 for
 *                     the model's {@code dim}. It must be the dim or one of
 *                     {@link TurboModelInfo#outputDims()}
 * @param quantization what the vectors are returned as; a quantization that makes codes is read
 *                     with {@link TurboSession#embedCodes}
 * @param int8Ranges   for {@link Quantization#INT8} alone: each dimension's minimum, then each
 *                     one's maximum, {@code 2 * outputDim} values over calibration vectors; null
 *                     otherwise
 */
public record TurboEmbedOptions(
        Truncate truncate,
        int maxTokens,
        PromptRole promptRole,
        Normalize normalize,
        Pooling pooling,
        int outputDim,
        Quantization quantization,
        float[] int8Ranges) {

    /** How a long text is cut, as {@code TURBO_TRUNCATE_*}. */
    public enum Truncate {
        /** What the bundle says. */
        MODEL(0),
        /** Not cut: a text that is too long fails with {@code TURBO_E_CAPACITY}. */
        NONE(1),
        /** Its last tokens are dropped. */
        RIGHT(2),
        /** Its first tokens are dropped. */
        LEFT(3);

        private final int nativeCode;

        Truncate(int nativeCode) {
            this.nativeCode = nativeCode;
        }

        /** Returns the native code. */
        public int nativeCode() {
            return nativeCode;
        }
    }

    /** Which bundle prefix goes before a text, as {@code TURBO_PROMPT_*}. */
    public enum PromptRole {
        /** No prefix. */
        NONE(0),
        /** The bundle's query prefix. */
        QUERY(1),
        /** The bundle's document prefix. */
        DOCUMENT(2);

        private final int nativeCode;

        PromptRole(int nativeCode) {
            this.nativeCode = nativeCode;
        }

        /** Returns the native code. */
        public int nativeCode() {
            return nativeCode;
        }
    }

    /** How a vector is normalized, as {@code TURBO_NORMALIZE_*}. */
    public enum Normalize {
        /** What the bundle says. */
        MODEL(0),
        /** Not normalized. */
        NONE(1),
        /** To unit L2 length. */
        L2(2);

        private final int nativeCode;

        Normalize(int nativeCode) {
            this.nativeCode = nativeCode;
        }

        /** Returns the native code. */
        public int nativeCode() {
            return nativeCode;
        }

        /** The value of a native code. */
        public static Normalize fromNative(int nativeCode) {
            for (Normalize n : values()) {
                if (n.nativeCode == nativeCode) {
                    return n;
                }
            }
            throw new IllegalStateException("Unknown native normalize: " + nativeCode);
        }
    }

    /** How token states are pooled, as {@code TURBO_POOLING_*}. */
    public enum Pooling {
        /** What the bundle says. */
        MODEL(0),
        /** The mean over the tokens whose mask is 1. */
        MEAN(1),
        /** The row's first token. */
        CLS(2),
        /** The last token whose mask is 1. */
        LAST(3);

        private final int nativeCode;

        Pooling(int nativeCode) {
            this.nativeCode = nativeCode;
        }

        /** Returns the native code. */
        public int nativeCode() {
            return nativeCode;
        }

        /** The value of a native code. */
        public static Pooling fromNative(int nativeCode) {
            for (Pooling p : values()) {
                if (p.nativeCode == nativeCode) {
                    return p;
                }
            }
            throw new IllegalStateException("Unknown native pooling: " + nativeCode);
        }
    }

    /** What the vectors are returned as, as {@code TURBO_QUANTIZE_*}. */
    public enum Quantization {
        /** The bundle's: F32. */
        MODEL(0, false),
        /** F32. */
        NONE(1, false),
        /** A signed byte a dimension, from {@code int8Ranges}. */
        INT8(2, true),
        /** One bit a dimension, set when the value is above 0, the first dimension in the top bit. */
        UBINARY(3, true);

        private final int nativeCode;
        private final boolean codes;

        Quantization(int nativeCode, boolean codes) {
            this.nativeCode = nativeCode;
            this.codes = codes;
        }

        /** Returns the native code. */
        public int nativeCode() {
            return nativeCode;
        }

        /** Whether the vectors come back as bytes ({@link TurboSession#embedCodes}). */
        public boolean codes() {
            return codes;
        }
    }

    /** What the bundle says for every field. */
    public static final TurboEmbedOptions DEFAULT = new TurboEmbedOptions(
            Truncate.MODEL, 0, PromptRole.NONE, Normalize.MODEL, Pooling.MODEL, 0, Quantization.MODEL, null);

    /** Takes an absent enum as its default, checks the numbers, and copies the ranges. */
    public TurboEmbedOptions {
        truncate = truncate == null ? Truncate.MODEL : truncate;
        promptRole = promptRole == null ? PromptRole.NONE : promptRole;
        normalize = normalize == null ? Normalize.MODEL : normalize;
        pooling = pooling == null ? Pooling.MODEL : pooling;
        quantization = quantization == null ? Quantization.MODEL : quantization;
        TurboSessionOptions.nonNegative(maxTokens, "maxTokens");
        TurboSessionOptions.nonNegative(outputDim, "outputDim");
        int8Ranges = int8Ranges == null ? null : int8Ranges.clone();
    }

    /** The defaults with the bundle's query prefix. */
    public static TurboEmbedOptions query() {
        return DEFAULT.withPromptRole(PromptRole.QUERY);
    }

    /** The defaults with the bundle's document prefix. */
    public static TurboEmbedOptions document() {
        return DEFAULT.withPromptRole(PromptRole.DOCUMENT);
    }

    @Override
    public float[] int8Ranges() {
        return int8Ranges == null ? null : int8Ranges.clone();
    }

    /** These options with {@code promptRole}. */
    public TurboEmbedOptions withPromptRole(PromptRole promptRole) {
        return new TurboEmbedOptions(truncate, maxTokens, promptRole, normalize, pooling, outputDim, quantization, int8Ranges);
    }

    /** These options cutting each text to {@code maxTokens} tokens as {@code truncate} says. */
    public TurboEmbedOptions withTruncation(Truncate truncate, int maxTokens) {
        return new TurboEmbedOptions(truncate, maxTokens, promptRole, normalize, pooling, outputDim, quantization, int8Ranges);
    }

    /** These options with {@code normalize}. */
    public TurboEmbedOptions withNormalize(Normalize normalize) {
        return new TurboEmbedOptions(truncate, maxTokens, promptRole, normalize, pooling, outputDim, quantization, int8Ranges);
    }

    /** These options with {@code pooling}. */
    public TurboEmbedOptions withPooling(Pooling pooling) {
        return new TurboEmbedOptions(truncate, maxTokens, promptRole, normalize, pooling, outputDim, quantization, int8Ranges);
    }

    /** These options keeping the first {@code outputDim} values of each vector. */
    public TurboEmbedOptions withOutputDim(int outputDim) {
        return new TurboEmbedOptions(truncate, maxTokens, promptRole, normalize, pooling, outputDim, quantization, int8Ranges);
    }

    /** These options returning packed sign bits ({@link Quantization#UBINARY}). */
    public TurboEmbedOptions withBinary() {
        return new TurboEmbedOptions(truncate, maxTokens, promptRole, normalize, pooling, outputDim, Quantization.UBINARY, null);
    }

    /** These options returning int8 codes over {@code ranges} ({@link Quantization#INT8}). */
    public TurboEmbedOptions withInt8(float[] ranges) {
        if (ranges == null) {
            throw new NullPointerException("ranges");
        }
        return new TurboEmbedOptions(truncate, maxTokens, promptRole, normalize, pooling, outputDim, Quantization.INT8, ranges);
    }

    /**
     * The values a native bridge passes on: {@code [truncate, maxTokens, promptRole, normalize,
     * pooling, outputDim, quantization]}; the ranges are passed beside them.
     */
    public int[] toNative() {
        return new int[] {
            truncate.nativeCode(),
            maxTokens,
            promptRole.nativeCode(),
            normalize.nativeCode(),
            pooling.nativeCode(),
            outputDim,
            quantization.nativeCode()
        };
    }

    @Override
    public boolean equals(Object other) {
        return other instanceof TurboEmbedOptions o
                && truncate == o.truncate
                && maxTokens == o.maxTokens
                && promptRole == o.promptRole
                && normalize == o.normalize
                && pooling == o.pooling
                && outputDim == o.outputDim
                && quantization == o.quantization
                && Arrays.equals(int8Ranges, o.int8Ranges);
    }

    @Override
    public int hashCode() {
        return 31 * java.util.Objects.hash(truncate, maxTokens, promptRole, normalize, pooling, outputDim, quantization)
                + Arrays.hashCode(int8Ranges);
    }

    @Override
    public String toString() {
        return "TurboEmbedOptions[truncate=" + truncate + ", maxTokens=" + maxTokens + ", promptRole=" + promptRole
                + ", normalize=" + normalize + ", pooling=" + pooling + ", outputDim=" + outputDim
                + ", quantization=" + quantization + ", int8Ranges=" + Arrays.toString(int8Ranges) + "]";
    }
}
