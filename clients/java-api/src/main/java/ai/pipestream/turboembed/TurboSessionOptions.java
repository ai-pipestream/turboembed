package ai.pipestream.turboembed;

/**
 * How a {@link TurboSession} opens: the device it runs on and the session's precision, batch,
 * sequence length and CPU threads. Zero in a numeric field means the model's or the runtime's
 * own value, as in {@code turbo_session_desc}.
 *
 * @param device     the device index ({@code turbo_runtime_device_info} order); ignored when
 *                   {@code selectFor} names a task
 * @param selectFor  a task to pick the device for with {@code turbo_runtime_select}, or null to
 *                   open {@code device}
 * @param precision  how the session computes
 * @param maxBatch   rows a run may take; 0 for the model's
 * @param maxSeq     tokens a row may take; 0 for the model's
 * @param cpuThreads threads a CPU session computes on; 0 for the runtime's choice
 */
public record TurboSessionOptions(
        int device, Task selectFor, Precision precision, int maxBatch, int maxSeq, int cpuThreads) {

    /** A model task, as {@code TURBO_TASK_*} numbers it. */
    public enum Task {
        /** Dense embedding. */
        EMBED(1),
        /** Cross-encoder reranking. */
        RERANK(2),
        /** Extractive question answering. */
        EXTRACTIVE_QA(3);

        private final int nativeCode;

        Task(int nativeCode) {
            this.nativeCode = nativeCode;
        }

        /** Returns the native {@code TURBO_TASK_*} code. */
        public int nativeCode() {
            return nativeCode;
        }

        /** The task of a native code. */
        public static Task fromNative(int nativeCode) {
            for (Task task : values()) {
                if (task.nativeCode == nativeCode) {
                    return task;
                }
            }
            throw new IllegalStateException("Unknown native task: " + nativeCode);
        }
    }

    /** How a session computes, as {@code TURBO_PRECISION_*} numbers it. */
    public enum Precision {
        /** The artifact's compute dtype, or the dtype its weights are stored in. */
        MODEL(0),
        /** The fastest compute dtype the backend has for the artifact. */
        FASTEST(1),
        /** F32 throughout. */
        EXACT(2);

        private final int nativeCode;

        Precision(int nativeCode) {
            this.nativeCode = nativeCode;
        }

        /** Returns the native {@code TURBO_PRECISION_*} code. */
        public int nativeCode() {
            return nativeCode;
        }
    }

    /** Device 0 with the model's precision, batch and sequence length. */
    public static final TurboSessionOptions DEFAULT = new TurboSessionOptions(0, null, Precision.MODEL, 0, 0, 0);

    /** Checks the fields and takes an absent precision as {@link Precision#MODEL}. */
    public TurboSessionOptions {
        precision = precision == null ? Precision.MODEL : precision;
        nonNegative(device, "device");
        nonNegative(maxBatch, "maxBatch");
        nonNegative(maxSeq, "maxSeq");
        nonNegative(cpuThreads, "cpuThreads");
    }

    /** The default options on device {@code device}. */
    public static TurboSessionOptions onDevice(int device) {
        return new TurboSessionOptions(device, null, Precision.MODEL, 0, 0, 0);
    }

    /** The default options on the device {@code turbo_runtime_select} picks for {@code task}. */
    public static TurboSessionOptions fastestFor(Task task) {
        if (task == null) {
            throw new NullPointerException("task");
        }
        return new TurboSessionOptions(0, task, Precision.MODEL, 0, 0, 0);
    }

    /** These options with {@code precision}. */
    public TurboSessionOptions withPrecision(Precision precision) {
        return new TurboSessionOptions(device, selectFor, precision, maxBatch, maxSeq, cpuThreads);
    }

    /** These options with a session of {@code maxBatch} rows of {@code maxSeq} tokens. */
    public TurboSessionOptions withShape(int maxBatch, int maxSeq) {
        return new TurboSessionOptions(device, selectFor, precision, maxBatch, maxSeq, cpuThreads);
    }

    /** These options with {@code cpuThreads} threads on a CPU. */
    public TurboSessionOptions withCpuThreads(int cpuThreads) {
        return new TurboSessionOptions(device, selectFor, precision, maxBatch, maxSeq, cpuThreads);
    }

    /**
     * The values a native bridge passes on: {@code [device, task, precision, maxBatch, maxSeq,
     * cpuThreads]}, task 0 when no task is named.
     */
    public int[] toNative() {
        return new int[] {
            device, selectFor == null ? 0 : selectFor.nativeCode(), precision.nativeCode(), maxBatch, maxSeq, cpuThreads
        };
    }

    static void nonNegative(int value, String name) {
        if (value < 0) {
            throw new IllegalArgumentException(name + " is negative: " + value);
        }
    }
}
