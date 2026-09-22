// SPDX-License-Identifier: Apache-2.0
package ai.pipestream.turbo.web;

import ai.pipestream.turbo.Chunk;
import ai.pipestream.turbo.Context;
import ai.pipestream.turbo.DeviceInfo;
import ai.pipestream.turbo.EmbedOptions;
import ai.pipestream.turbo.GenerateDesc;
import ai.pipestream.turbo.Generation;
import ai.pipestream.turbo.Message;
import ai.pipestream.turbo.Model;
import ai.pipestream.turbo.ModelInfo;
import ai.pipestream.turbo.ModelKind;
import ai.pipestream.turbo.Result;
import ai.pipestream.turbo.SelectPolicy;
import ai.pipestream.turbo.Session;
import ai.pipestream.turbo.Turbo;
import jakarta.annotation.PreDestroy;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.ArrayBlockingQueue;
import java.util.concurrent.BlockingQueue;
import java.util.concurrent.Semaphore;
import java.util.concurrent.TimeUnit;
import java.util.function.Predicate;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Service;

/**
 * One model on one device, with a small pool of sessions handed to
 * requests. A request that finds no free session within a short wait is
 * refused ({@link Overloaded}) rather than queued without bound; every
 * libturbo error propagates as a {@link ai.pipestream.turbo.TurboException}
 * and the controller maps it to an HTTP status with the message intact.
 */
@Service
public class TurboService implements AutoCloseable {
    /** Raised when every session is in use. */
    public static final class Overloaded extends RuntimeException {
        Overloaded(String message) {
            super(message);
        }
    }

    public record Info(String deviceName, String providerId, int ordinal, String deviceKind, String runtimeVersion,
            String modelId, int dim, int maxSeq, int maxBatch, boolean fullyAccelerated, String stages,
            GenerateInfo generate) {}

    /** The generative model, when one is configured. */
    public record GenerateInfo(String deviceName, String providerId, int ordinal, String modelId, int maxSeq) {}

    public record Embedded(int dim, float[][] vectors, double[][] similarity) {}

    private final Turbo runtime;
    private final Context context;
    private final Model model;
    private final BlockingQueue<Session> sessions;
    private final Info info;
    private final int maxBatch;
    private final Context generateContext;
    private final Model generateModel;
    private final Semaphore generations;

    public TurboService(
            @Value("${turbo.bundle}") String bundle,
            @Value("${turbo.provider-lib:}") String providerLib,
            @Value("${turbo.provider:}") String provider,
            @Value("${turbo.ordinal:0}") int ordinal,
            @Value("${turbo.sessions:2}") int sessionCount,
            @Value("${turbo.max-batch:0}") int maxBatchProperty,
            @Value("${turbo.generate-bundle:}") String generateBundle,
            @Value("${turbo.generate-provider-lib:}") String generateProviderLib,
            @Value("${turbo.generate-provider:}") String generateProvider,
            @Value("${turbo.generate-ordinal:0}") int generateOrdinal,
            @Value("${turbo.generations:2}") int concurrentGenerations) {
        List<String> libs = new ArrayList<>();
        if (!providerLib.isBlank()) libs.add(providerLib);
        if (!generateProviderLib.isBlank() && !generateProviderLib.equals(providerLib)) libs.add(generateProviderLib);
        runtime = libs.isEmpty() ? Turbo.create() : Turbo.create(libs);
        int device = provider.isBlank() ? runtime.selectDevice() : runtime.selectDevice(SelectPolicy.EXPLICIT, provider, ordinal);
        DeviceInfo di = runtime.device(device);
        context = runtime.createContext(device);
        model = context.loadModel(bundle);
        ModelInfo mi = model.info();
        if (mi.kind() != ModelKind.EMBEDDING) {
            close();
            throw new IllegalArgumentException(bundle + " is not an embedding bundle: " + mi.kind());
        }
        // 0 means the model's own batch limit; a larger request is refused by
        // the library at session creation, which is the right place.
        this.maxBatch = maxBatchProperty > 0 ? maxBatchProperty : mi.maxBatch();
        sessions = new ArrayBlockingQueue<>(sessionCount);
        for (int i = 0; i < sessionCount; i++) {
            sessions.add(model.createSession(maxBatch, 0));
        }
        GenerateInfo gen = null;
        if (generateBundle.isBlank()) {
            generateContext = null;
            generateModel = null;
        } else {
            int gdev = generateProvider.isBlank() ? runtime.selectDevice() : runtime.selectDevice(SelectPolicy.EXPLICIT, generateProvider, generateOrdinal);
            DeviceInfo gdi = runtime.device(gdev);
            generateContext = runtime.createContext(gdev);
            generateModel = generateContext.loadModel(generateBundle);
            ModelInfo gmi = generateModel.info();
            if (gmi.kind() != ModelKind.GENERATIVE) {
                close();
                throw new IllegalArgumentException(generateBundle + " is not a generative bundle: " + gmi.kind());
            }
            gen = new GenerateInfo(gdi.name(), gdi.providerId(), gdi.ordinal(), gmi.modelId(), gmi.maxSeq());
        }
        generations = new Semaphore(Math.max(1, concurrentGenerations));
        info = new Info(di.name(), di.providerId(), di.ordinal(), di.kind().name(), di.runtimeVersion(), mi.modelId(), mi.dim(),
                mi.maxSeq(), maxBatch, mi.fullyAccelerated(), java.util.Arrays.toString(mi.stagePlacement()), gen);
    }

    public boolean canGenerate() {
        return generateModel != null;
    }

    /**
     * Summarize {@code text}: a system instruction plus the text as the user
     * turn, streamed chunk by chunk to {@code sink} (return false to stop).
     * Returns the final chunk. Without a generative bundle this is an
     * {@link IllegalStateException}; with every generation slot taken it is
     * {@link Overloaded}.
     */
    public Chunk summarize(String text, int maxNewTokens, Predicate<Chunk> sink) {
        if (generateModel == null) {
            throw new IllegalStateException("no generative bundle is configured (turbo.generate-bundle)");
        }
        if (text == null || text.isBlank()) {
            throw new IllegalArgumentException("no text");
        }
        if (!generations.tryAcquire()) {
            throw new Overloaded("every generation slot is busy; retry");
        }
        try (Generation g = generateModel.createGeneration(GenerateDesc.defaults().withMaxNewTokens(maxNewTokens <= 0 ? 128 : maxNewTokens))) {
            g.prompt(List.of(
                    Message.system("You summarize text. Reply with a summary of at most three sentences and nothing else."),
                    Message.user(text.strip())));
            return g.drain(sink);
        } finally {
            generations.release();
        }
    }

    public Info info() {
        return info;
    }

    public int maxBatch() {
        return maxBatch;
    }

    /** Embed texts and compute the cosine matrix; a batch above the pool's width is an IllegalArgumentException. */
    public Embedded embed(List<String> texts) {
        if (texts.isEmpty()) {
            throw new IllegalArgumentException("no texts");
        }
        if (texts.size() > maxBatch) {
            throw new IllegalArgumentException("request has " + texts.size() + " texts but the server's batch is " + maxBatch);
        }
        Session session;
        try {
            session = sessions.poll(200, TimeUnit.MILLISECONDS);
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            throw new Overloaded("interrupted while waiting for a session");
        }
        if (session == null) {
            throw new Overloaded("every session is busy; retry");
        }
        try {
            session.writeText(texts, EmbedOptions.defaults());
            float[] flat;
            int dim;
            try (Result r = session.run()) {
                dim = (int) r.output(0).shape()[1];
                flat = r.readFloats(0);
            }
            float[][] vectors = new float[texts.size()][];
            for (int i = 0; i < texts.size(); i++) {
                vectors[i] = new float[dim];
                System.arraycopy(flat, i * dim, vectors[i], 0, dim);
            }
            double[][] sim = new double[texts.size()][texts.size()];
            for (int a = 0; a < texts.size(); a++) {
                for (int b = 0; b < texts.size(); b++) {
                    sim[a][b] = cosine(vectors[a], vectors[b]);
                }
            }
            return new Embedded(dim, vectors, sim);
        } finally {
            sessions.add(session);
        }
    }

    static double cosine(float[] a, float[] b) {
        double dot = 0, na = 0, nb = 0;
        for (int i = 0; i < a.length; i++) {
            dot += (double) a[i] * b[i];
            na += (double) a[i] * a[i];
            nb += (double) b[i] * b[i];
        }
        return dot / (Math.sqrt(na) * Math.sqrt(nb));
    }

    @PreDestroy
    @Override
    public void close() {
        List<Session> drained = new ArrayList<>();
        if (sessions != null) {
            sessions.drainTo(drained);
        }
        drained.forEach(Session::close);
        if (generateModel != null) generateModel.close();
        if (generateContext != null) generateContext.close();
        if (model != null) model.close();
        if (context != null) context.close();
        if (runtime != null) runtime.close();
    }
}
