// SPDX-License-Identifier: Apache-2.0
package ai.pipestream.turbo.web;

import ai.pipestream.turbo.DeviceInfo;
import ai.pipestream.turbo.Model;
import ai.pipestream.turbo.ModelInfo;
import ai.pipestream.turbo.ModelKind;
import ai.pipestream.turbo.Session;
import ai.pipestream.turbo.Tokenizer;
import ai.pipestream.turbo.Turbo;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.ArrayBlockingQueue;
import java.util.concurrent.BlockingQueue;
import java.util.concurrent.Semaphore;
import java.util.concurrent.TimeUnit;

/**
 * One loaded model served under a name: the native handle, the device it runs
 * on, a fixed pool of sessions for the session tasks, a permit count for
 * generations, and the tokenizer its bundle declares.
 *
 * <p>Nothing here falls back. A session that is not free within {@link
 * #SESSION_WAIT_MS} is refused, a tokenizer the bundle does not carry is the
 * library's own refusal, and every libturbo error propagates as a {@link
 * ai.pipestream.turbo.TurboException}.
 */
public final class LoadedModel implements AutoCloseable {
    /** How long a request waits for a free session before it is refused. */
    public static final long SESSION_WAIT_MS = 200;

    private final String name;
    private final String bundle;
    private final String tokenizerBundle;
    private final Turbo runtime;
    private final Model model;
    private final ModelInfo info;
    private final DeviceInfo device;
    private final int maxBatch;
    private final BlockingQueue<Session> sessions;
    private final Semaphore generations;

    private final Object tokenizerLock = new Object();
    private Tokenizer tokenizer;
    private boolean closed;

    LoadedModel(String name, String bundle, String tokenizerBundle, Turbo runtime, Model model, DeviceInfo device,
            int maxBatch, int sessionCount, int generationCount) {
        this.name = name;
        this.bundle = bundle;
        this.tokenizerBundle = tokenizerBundle;
        this.runtime = runtime;
        this.model = model;
        this.info = model.info();
        this.device = device;
        this.maxBatch = maxBatch;
        this.sessions = new ArrayBlockingQueue<>(Math.max(1, sessionCount));
        this.generations = new Semaphore(Math.max(1, generationCount));
        if (usesSessions()) {
            for (int i = 0; i < Math.max(1, sessionCount); i++) {
                sessions.add(model.createSession(maxBatch, 0));
            }
        }
    }

    /** The name this model is served under, in URLs and in the {@code model} field of a request. */
    public String name() {
        return name;
    }

    /** The bundle directory this model was loaded from. */
    public String bundle() {
        return bundle;
    }

    /** The bundle whose tokenizer serves this model, which is the model's own unless configured otherwise. */
    public String tokenizerBundle() {
        return tokenizerBundle.isBlank() ? bundle : tokenizerBundle;
    }

    /** The model contract as the library reports it. */
    public ModelInfo info() {
        return info;
    }

    /** The device this model runs on. */
    public DeviceInfo device() {
        return device;
    }

    /** The native handle, for the calls that need it (generation creation). */
    public Model handle() {
        return model;
    }

    /** Batch width of every session in the pool. */
    public int maxBatch() {
        return maxBatch;
    }

    /** True for the kinds that run through a session rather than a generation. */
    public boolean usesSessions() {
        return info.kind() != ModelKind.GENERATIVE;
    }

    /**
     * Run {@code work} on a session from the pool and return its value. Every
     * session is returned to the pool, including when the work throws.
     */
    public <T> T withSession(SessionWork<T> work) {
        Session session;
        try {
            session = sessions.poll(SESSION_WAIT_MS, TimeUnit.MILLISECONDS);
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            throw new TurboService.Overloaded("interrupted while waiting for a session of model " + name);
        }
        if (session == null) {
            throw new TurboService.Overloaded(
                    "every session of model " + name + " is busy (" + sessions.remainingCapacity() + " in flight); retry");
        }
        try {
            return work.run(session);
        } finally {
            sessions.add(session);
        }
    }

    /** Take a generation permit, or refuse. Release it with {@link #releaseGeneration()}. */
    public void acquireGeneration() {
        if (!generations.tryAcquire()) {
            throw new TurboService.Overloaded("every generation slot of model " + name + " is busy; retry");
        }
    }

    /** Give a generation permit back. */
    public void releaseGeneration() {
        generations.release();
    }

    /**
     * The tokenizer of {@link #tokenizerBundle()}, created on first use. A
     * bundle that declares no {@code tokenizer.json} fails here with the
     * library's own {@code TURBO_E_BUNDLE_INVALID} and its message.
     */
    public Tokenizer tokenizer() {
        synchronized (tokenizerLock) {
            if (closed) {
                throw new IllegalStateException("model " + name + " is closed");
            }
            if (tokenizer == null) {
                tokenizer = runtime.createTokenizer(tokenizerBundle());
            }
            return tokenizer;
        }
    }

    @Override
    public void close() {
        synchronized (tokenizerLock) {
            if (closed) {
                return;
            }
            closed = true;
            if (tokenizer != null) {
                tokenizer.close();
                tokenizer = null;
            }
        }
        List<Session> drained = new ArrayList<>();
        sessions.drainTo(drained);
        drained.forEach(Session::close);
        model.close();
    }

    /** Work handed a session from the pool. */
    @FunctionalInterface
    public interface SessionWork<T> {
        T run(Session session);
    }
}
