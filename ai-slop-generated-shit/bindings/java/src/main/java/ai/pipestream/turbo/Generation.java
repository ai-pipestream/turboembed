package ai.pipestream.turbo;

import static ai.pipestream.turbo.ffi.TurboNative.*;

import ai.pipestream.turbo.ffi.turbo_generation_chunk;
import ai.pipestream.turbo.ffi.turbo_message;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.util.List;
import java.util.function.Predicate;

/**
 * A generation on a generative model: the pull iterator over
 * {@code turbo_generation_*}. Prompt it once, then {@link #step()} until a
 * chunk says {@code done}. Single owner, like a session; {@link #cancel()}
 * may be called from another thread and the next step reports
 * {@link FinishReason#CANCELLED}.
 */
public final class Generation implements AutoCloseable {
    private final MemorySegment g;
    private final Model model;
    private boolean closed;

    Generation(MemorySegment g, Model model) {
        this.g = g;
        this.model = model;
    }

    MemorySegment handle() {
        if (closed) {
            throw new IllegalStateException("generation is closed");
        }
        return g;
    }

    /** The model this generation runs on. */
    public Model model() {
        return model;
    }

    /** Apply the chat template to {@code messages} and tokenize the prompt. */
    public void prompt(List<Message> messages) {
        try (Arena arena = Arena.ofConfined()) {
            MemorySegment arr = turbo_message.allocateArray(messages.size(), arena);
            for (int i = 0; i < messages.size(); i++) {
                MemorySegment m = turbo_message.asSlice(arr, i);
                Native.fillText(arena, turbo_message.role(m), messages.get(i).role());
                Native.fillText(arena, turbo_message.content(m), messages.get(i).content());
            }
            MemorySegment err = Native.error(arena);
            Native.check(turbo_generation_prompt(handle(), arr, messages.size(), err), err);
        }
    }

    /** Use caller-supplied prompt token ids instead of a chat prompt. */
    public void promptTokens(int[] ids) {
        try (Arena arena = Arena.ofConfined()) {
            MemorySegment err = Native.error(arena);
            Native.check(turbo_generation_prompt_tokens(handle(), arena.allocateFrom(ValueLayout.JAVA_INT, ids), ids.length, err), err);
        }
    }

    /** Produce the next chunk. */
    public Chunk step() {
        try (Arena arena = Arena.ofConfined()) {
            MemorySegment c = turbo_generation_chunk.allocate(arena);
            turbo_generation_chunk.struct_size(c, (int) turbo_generation_chunk.sizeof());
            MemorySegment err = Native.error(arena);
            Native.check(turbo_generation_step(handle(), c, err), err);
            return Chunk.from(c);
        }
    }

    /**
     * Step until done, handing every chunk to {@code sink}; a sink that
     * returns false cancels the generation and the final chunk reports
     * {@link FinishReason#CANCELLED}. Returns the final chunk.
     */
    public Chunk drain(Predicate<Chunk> sink) {
        while (true) {
            Chunk c = step();
            boolean go = sink.test(c);
            if (c.done()) {
                return c;
            }
            if (!go) {
                cancel();
            }
        }
    }

    /** Cancel; the next step reports {@link FinishReason#CANCELLED}. */
    public void cancel() {
        try (Arena arena = Arena.ofConfined()) {
            MemorySegment err = Native.error(arena);
            Native.check(turbo_generation_cancel(handle(), err), err);
        }
    }

    @Override
    public void close() {
        if (!closed) {
            closed = true;
            turbo_generation_release(g);
        }
    }
}
