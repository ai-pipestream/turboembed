// SPDX-License-Identifier: Apache-2.0
import ai.pipestream.turboembed.*;
import ai.pipestream.turboembed.ffm.FfmTurboEmbed;
import java.lang.management.ManagementFactory;
import java.nio.FloatBuffer;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Arrays;

/** Bounded diagnostic harness, deliberately outside the consumer artifacts. */
public final class BindingBenchmark {
    private static volatile long sink;
    private static final com.sun.management.ThreadMXBean ALLOCATIONS =
        (com.sun.management.ThreadMXBean) ManagementFactory.getThreadMXBean();
    private static String measure(Runnable call, int maximum, int warmup) {
        for (int i = 0; i < warmup; i++) { call.run(); }
        long[] samples = new long[maximum]; int count = 0;
        long start = System.nanoTime();
        do {
            long before = System.nanoTime(); call.run(); long end = System.nanoTime();
            samples[count++] = end - before;
            if (end - start >= 3_000_000_000L) { break; }
        } while (count < maximum);
        double elapsed = (System.nanoTime() - start) / 1e9;
        long thread = Thread.currentThread().threadId();
        long bytesBefore = ALLOCATIONS.getThreadAllocatedBytes(thread);
        for (int i = 0; i < 100; i++) { call.run(); }
        double allocated = (ALLOCATIONS.getThreadAllocatedBytes(thread) - bytesBefore) / 100.0;
        long[] sorted = Arrays.copyOf(samples, count); Arrays.sort(sorted);
        StringBuilder json = new StringBuilder("{\"samples_ns\":[");
        for (int i = 0; i < count; i++) { if (i != 0) { json.append(','); } json.append(samples[i]); }
        return json.append("],\"count\":").append(count).append(",\"elapsed_seconds\":").append(elapsed)
            .append(",\"p50_ns\":").append(sorted[(count - 1) * 50 / 100])
            .append(",\"p99_ns\":").append(sorted[(count - 1) * 99 / 100])
            .append(",\"requests_per_second\":").append(count / elapsed)
            .append(",\"java_allocated_bytes_per_call\":").append(allocated).append('}').toString();
    }
    public static void main(String[] args) throws Exception {
        if (args.length < 5 || args.length > 6) { throw new IllegalArgumentException("SDK BUNDLE BATCH SEQUENCE OUTPUT_JSON [gpu|cpu]"); }
        if (!ALLOCATIONS.isThreadAllocatedMemorySupported()) { throw new IllegalStateException("allocation counter unavailable"); }
        ALLOCATIONS.setThreadAllocatedMemoryEnabled(true);
        int batch = Integer.parseInt(args[2]), sequence = Integer.parseInt(args[3]);
        // GPU is the reference device; CPU must be selected explicitly.
        Device device = switch (args.length == 6 ? args[5] : "gpu") {
            case "gpu" -> Device.OPENVINO_GPU;
            case "cpu" -> Device.OPENVINO_CPU;
            default -> throw new IllegalArgumentException("device must be gpu or cpu");
        };
        try (var provider = FfmTurboEmbed.open(Path.of(args[0]));
             var context = provider.context(device, 0);
             var model = context.loadModel(Path.of(args[1])); var slot = model.slot(batch, sequence)) {
            if (!model.info().tokenizerSha256().equals("be50c3628f2bf5bb5e3a7f17b1f74611b2561a3a27eeab05e5aa30f411572037")) {
                throw new IllegalArgumentException("benchmark token IDs require pinned MiniLM tokenizer");
            }
            if (sequence < 4) { throw new IllegalArgumentException("sequence must fit hello world"); }
            TokenBuffers input = slot.inputs(); int[] words = {101, 7592, 2088, 102};
            for (int row = 0; row < batch; row++) {
                for (int i = 0; i < words.length; i++) {
                    input.ids().put(row * sequence + i, words[i]); input.attentionMask().put(row * sequence + i, 1);
                }
            }
            slot.upload();
            String[] texts = new String[batch]; Arrays.fill(texts, "hello world");
            FloatBuffer output = FloatBuffer.allocate(batch * model.info().dimension());
            Runnable bridge = () -> { SlotStats stats = slot.stats(); sink = stats.executions(); };
            Runnable prepared = () -> { try (var result = slot.execute()) { sink = result.dimension(); } };
            Runnable text = () -> {
                slot.writeText(texts); output.clear();
                try (var result = slot.execute()) { result.readInto(output); }
            };
            float[] reference = new float[output.capacity()];
            try (var result = slot.execute()) { result.readInto(FloatBuffer.wrap(reference)); }
            text.run(); double maximum = 0, squared = 0;
            for (int i = 0; i < reference.length; i++) {
                double difference = Math.abs((double) reference[i] - output.get(i));
                if (!Double.isFinite(difference)) { throw new AssertionError("nonfinite output"); }
                maximum = Math.max(maximum, difference); squared += difference * difference;
            }
            if (maximum > 1e-6 || Math.sqrt(squared / reference.length) > 1e-7) { throw new AssertionError("prepared/text parity failed"); }
            StringBuilder json = new StringBuilder("{\"path\":\"java_ffm\",\"device\":\"")
                .append(context.info().name()).append("\",\"batch\":").append(batch)
                .append(",\"sequence\":").append(sequence).append(",\"text\":\"hello world\",\"jdk\":\"")
                .append(System.getProperty("java.runtime.version")).append("\",\"repeats\":[");
            for (int repeat = 0; repeat < 3; repeat++) {
                if (repeat != 0) { json.append(','); }
                json.append("{\"repeat\":").append(repeat).append(",\"bridge\":").append(measure(bridge, 100000, 100000))
                    .append(",\"prepared\":").append(measure(prepared, 3000, 500))
                    .append(",\"text_host\":").append(measure(text, 3000, 500)).append('}');
            }
            Files.writeString(Path.of(args[4]), json.append("]}\n").toString());
        }
    }
}
