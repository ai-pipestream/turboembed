// SPDX-License-Identifier: Apache-2.0
package ai.pipestream.turbo.demo;

import ai.pipestream.turbo.Context;
import ai.pipestream.turbo.DeviceInfo;
import ai.pipestream.turbo.EmbedOptions;
import ai.pipestream.turbo.Model;
import ai.pipestream.turbo.ModelInfo;
import ai.pipestream.turbo.ModelKind;
import ai.pipestream.turbo.Result;
import ai.pipestream.turbo.SelectPolicy;
import ai.pipestream.turbo.Session;
import ai.pipestream.turbo.Turbo;
import ai.pipestream.turbo.TurboException;
import java.util.ArrayList;
import java.util.List;

/**
 * Turbo Java demo: load a bundle on the best device, embed sentences, and
 * print their cosine similarities through the JDK 25 FFM binding.
 *
 * <pre>
 * java -Dturbo.library=target/debug/libturbo.so -jar demo/java/target/turbo-demo-java-0.1.0.jar \
 *     [--provider-lib so] [--provider id --ordinal n] --bundle dir text...
 * </pre>
 *
 * Every failure is a {@link TurboException} carrying the status name, the
 * field index when an option was refused, and the library's message.
 */
public final class Main {
    private Main() {}

    public static void main(String[] args) {
        String providerLib = null, provider = null, bundle = null;
        Integer ordinal = null;
        List<String> texts = new ArrayList<>();
        for (int i = 0; i < args.length; i++) {
            switch (args[i]) {
                case "--provider-lib" -> providerLib = args[++i];
                case "--provider" -> provider = args[++i];
                case "--ordinal" -> ordinal = Integer.parseInt(args[++i]);
                case "--bundle" -> bundle = args[++i];
                default -> texts.add(args[i]);
            }
        }
        if (bundle == null || texts.isEmpty()) {
            System.err.println("usage: [--provider-lib <so>] [--provider <id> --ordinal <n>] --bundle <dir> text...");
            System.exit(2);
        }
        if ((provider == null) != (ordinal == null)) {
            System.err.println("--provider and --ordinal go together; omit both for AUTO");
            System.exit(2);
        }
        try (Turbo rt = providerLib == null ? Turbo.create() : Turbo.create(List.of(providerLib))) {
            int device = provider == null ? rt.selectDevice() : rt.selectDevice(SelectPolicy.EXPLICIT, provider, ordinal);
            DeviceInfo di = rt.device(device);
            System.out.printf("device: %s (%s:%d, %s, runtime %s)%n", di.name(), di.providerId(), di.ordinal(), di.kind(), di.runtimeVersion());
            try (Context ctx = rt.createContext(device); Model model = ctx.loadModel(bundle)) {
                ModelInfo mi = model.info();
                if (mi.kind() != ModelKind.EMBEDDING) {
                    throw new IllegalArgumentException(bundle + " is not an embedding bundle: " + mi.kind());
                }
                System.out.printf("model: %s dim=%d max_seq=%d provider=%s fully_accelerated=%s%n", mi.modelId(), mi.dim(), mi.maxSeq(), mi.providerId(), mi.fullyAccelerated());
                try (Session session = model.createSession(texts.size(), 0)) {
                    session.writeText(texts, EmbedOptions.defaults());
                    float[] all;
                    int dim;
                    try (Result r = session.run()) {
                        dim = (int) r.output(0).shape()[1];
                        System.out.printf("embeddings: %d x %d (placement %s)%n", texts.size(), dim, r.placement());
                        all = r.readFloats(0);
                    }
                    System.out.println("cosine similarity:");
                    for (int a = 0; a < texts.size(); a++) {
                        StringBuilder line = new StringBuilder();
                        for (int b = 0; b < texts.size(); b++) {
                            line.append(String.format(" %6.3f", cosine(all, a, b, dim)));
                        }
                        System.out.println(line + "  " + texts.get(a));
                    }
                }
            }
        } catch (TurboException e) {
            System.err.println("error: " + e.statusName() + (e.field() != 0 ? " (field " + e.field() + ")" : "") + ": " + e.getMessage());
            System.exit(1);
        }
    }

    private static double cosine(float[] v, int a, int b, int dim) {
        double dot = 0, na = 0, nb = 0;
        for (int k = 0; k < dim; k++) {
            dot += (double) v[a * dim + k] * v[b * dim + k];
            na += (double) v[a * dim + k] * v[a * dim + k];
            nb += (double) v[b * dim + k] * v[b * dim + k];
        }
        return dot / (Math.sqrt(na) * Math.sqrt(nb));
    }
}
