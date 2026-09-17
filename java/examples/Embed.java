// SPDX-License-Identifier: Apache-2.0
import ai.pipestream.turboembed.Device;
import ai.pipestream.turboembed.DeviceInfo;
import ai.pipestream.turboembed.EmbeddingResult;
import ai.pipestream.turboembed.ExecutionSlot;
import ai.pipestream.turboembed.TokenBuffers;
import ai.pipestream.turboembed.ffm.FfmTurboEmbed;
import java.nio.FloatBuffer;
import java.nio.IntBuffer;
import java.nio.file.Path;

/** Standalone installed-SDK example for the pinned all-MiniLM-L6-v2 bundle. */
public final class Embed {
    private static final int DIMENSION = 384;
    private static final String TOKENIZER_SHA256 =
            "be50c3628f2bf5bb5e3a7f17b1f74611b2561a3a27eeab05e5aa30f411572037";

    private Embed() {}

    public static void main(String[] args) {
        if (args.length < 2 || args.length > 3) {
            throw new IllegalArgumentException("usage: Embed SDK_PREFIX MODEL_BUNDLE [gpu|cpu]");
        }

        Path prefix = Path.of(args[0]);
        Path bundle = Path.of(args[1]);
        // GPU is the default; CPU must be requested explicitly and a missing
        // GPU fails loudly instead of falling back.
        Device device = switch (args.length == 3 ? args[2] : "gpu") {
            case "gpu" -> Device.OPENVINO_GPU;
            case "cpu" -> Device.OPENVINO_CPU;
            default -> throw new IllegalArgumentException("device must be gpu or cpu");
        };
        try (var turboEmbed = FfmTurboEmbed.open(prefix)) {
            for (DeviceInfo discovered : turboEmbed.devices()) {
                System.out.printf(
                        "device %d ordinal %d: %s (%s)%n",
                        discovered.device(),
                        discovered.ordinal(),
                        discovered.name(),
                        discovered.runtimeVersion());
            }
            run(turboEmbed, device, bundle);
        }
    }

    private static void run(FfmTurboEmbed turboEmbed, Device device, Path bundle) {
        try (var context = turboEmbed.context(device, 0);
                var model = context.loadModel(bundle);
                var slot = model.slot(1, 32)) {
            if (model.info().dimension() != DIMENSION) {
                throw new IllegalStateException("expected a 384-dimensional model");
            }
            if (!TOKENIZER_SHA256.equals(model.info().tokenizerSha256())) {
                throw new IllegalStateException(
                        "this example's prepared token IDs require the pinned tokenizer");
            }

            slot.writeText("hello world");
            float[] textEmbedding = execute(slot);
            double norm = norm(textEmbedding);
            if (!Double.isFinite(norm) || Math.abs(norm - 1.0) > 1e-4) {
                throw new AssertionError("expected normalized embedding, norm=" + norm);
            }

            writePreparedHelloWorld(slot);
            float[] preparedEmbedding = execute(slot);
            compare(textEmbedding, preparedEmbedding, 1e-6, 1e-7);
            System.out.printf(
                    "%s embedding verified: dimension=%d norm=%.8f%n",
                    context.info().name(), textEmbedding.length, norm);
        }
    }

    private static float[] execute(ExecutionSlot slot) {
        try (EmbeddingResult result = slot.execute()) {
            if (result.batch() != 1 || result.dimension() != DIMENSION) {
                throw new AssertionError(
                        "unexpected result shape " + result.batch() + "x" + result.dimension());
            }
            float[] values = new float[DIMENSION];
            result.readInto(FloatBuffer.wrap(values));
            return values;
        }
    }

    private static void writePreparedHelloWorld(ExecutionSlot slot) {
        TokenBuffers input = slot.inputs();
        clear(input.ids());
        clear(input.attentionMask());
        clear(input.tokenTypes());

        int[] tokenIds = {101, 7592, 2088, 102};
        for (int i = 0; i < tokenIds.length; i++) {
            input.ids().put(i, tokenIds[i]);
            input.attentionMask().put(i, 1);
        }
        slot.upload();
    }

    private static void clear(IntBuffer buffer) {
        for (int i = 0; i < buffer.capacity(); i++) {
            buffer.put(i, 0);
        }
    }

    private static double norm(float[] values) {
        double squared = 0.0;
        for (float value : values) {
            squared += (double) value * value;
        }
        return Math.sqrt(squared);
    }

    private static void compare(float[] expected, float[] actual, double maxLimit, double rmseLimit) {
        double max = 0.0;
        double squared = 0.0;
        for (int i = 0; i < expected.length; i++) {
            if (!Float.isFinite(expected[i]) || !Float.isFinite(actual[i])) {
                throw new AssertionError("non-finite embedding value at index " + i);
            }
            double difference = Math.abs((double) expected[i] - actual[i]);
            max = Math.max(max, difference);
            squared += difference * difference;
        }
        double rmse = Math.sqrt(squared / expected.length);
        if (max > maxLimit || rmse > rmseLimit) {
            throw new AssertionError("prepared input differs: max=" + max + ", rmse=" + rmse);
        }
    }
}
