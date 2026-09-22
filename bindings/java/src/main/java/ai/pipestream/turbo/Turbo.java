package ai.pipestream.turbo;

import static ai.pipestream.turbo.ffi.TurboNative.*;

import ai.pipestream.turbo.ffi.turbo_capability;
import ai.pipestream.turbo.ffi.turbo_device_info;
import ai.pipestream.turbo.ffi.turbo_device_selector;
import ai.pipestream.turbo.ffi.turbo_runtime_desc;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.util.ArrayList;
import java.util.List;

/**
 * A runtime: the provider registry and the device list. Every other handle
 * is created from it and keeps it alive, so closing a runtime while a
 * context still exists is safe; the native object is released when the last
 * child goes.
 *
 * <p>The library is located through the {@code turbo.library} system
 * property or the {@code TURBO_LIBRARY} environment variable (a path to
 * {@code libturbo.so}), else through the loader's search for {@code turbo}.
 */
public final class Turbo implements AutoCloseable {
    private final MemorySegment rt;
    private boolean closed;

    private Turbo(MemorySegment rt) {
        this.rt = rt;
    }

    /** Create a runtime with the built-in providers only. */
    public static Turbo create() {
        return create(List.of());
    }

    /**
     * Create a runtime and load the provider libraries at {@code providerPaths}.
     * A provider that fails to load fails the call; nothing is loaded partially.
     */
    public static Turbo create(List<String> providerPaths) {
        try (Arena arena = Arena.ofConfined()) {
            MemorySegment desc = turbo_runtime_desc.allocate(arena);
            turbo_runtime_desc.struct_size(desc, (int) turbo_runtime_desc.sizeof());
            turbo_runtime_desc.n_provider_paths(desc, providerPaths.size());
            turbo_runtime_desc.provider_paths(desc, providerPaths.isEmpty() ? MemorySegment.NULL : Native.texts(arena, providerPaths));
            MemorySegment out = arena.allocate(ValueLayout.ADDRESS);
            MemorySegment err = Native.error(arena);
            Native.check(turbo_runtime_create(desc, out, err), err);
            return new Turbo(out.get(ValueLayout.ADDRESS, 0));
        }
    }

    /** The ABI version this library implements ({@code TURBO_ABI_VERSION}). */
    public static int abiVersion() {
        return turbo_abi_version();
    }

    MemorySegment handle() {
        if (closed) {
            throw new IllegalStateException("runtime is closed");
        }
        return rt;
    }

    /** Every device of every loaded provider, in registry order. */
    public List<DeviceInfo> devices() {
        try (Arena arena = Arena.ofConfined()) {
            MemorySegment n = arena.allocate(ValueLayout.JAVA_INT);
            MemorySegment err = Native.error(arena);
            Native.check(turbo_runtime_device_count(handle(), n, err), err);
            int count = n.get(ValueLayout.JAVA_INT, 0);
            List<DeviceInfo> out = new ArrayList<>(count);
            for (int i = 0; i < count; i++) {
                out.add(device(i));
            }
            return out;
        }
    }

    /** Static information about device {@code index}. */
    public DeviceInfo device(int index) {
        try (Arena arena = Arena.ofConfined()) {
            MemorySegment info = turbo_device_info.allocate(arena);
            turbo_device_info.struct_size(info, (int) turbo_device_info.sizeof());
            MemorySegment err = Native.error(arena);
            Native.check(turbo_runtime_device_info(handle(), index, info, err), err);
            return DeviceInfo.from(index, info);
        }
    }

    /**
     * Select a device. {@link SelectPolicy#AUTO} never picks a CPU; an absent
     * device is {@code TURBO_E_DEVICE_NOT_FOUND}, never a fallback.
     */
    public int selectDevice(SelectPolicy policy, String providerId, int ordinal) {
        try (Arena arena = Arena.ofConfined()) {
            MemorySegment sel = turbo_device_selector.allocate(arena);
            turbo_device_selector.struct_size(sel, (int) turbo_device_selector.sizeof());
            turbo_device_selector.policy(sel, policy.value());
            turbo_device_selector.ordinal(sel, ordinal);
            Native.fillText(arena, turbo_device_selector.provider_id(sel), providerId == null ? "" : providerId);
            Native.fillText(arena, turbo_device_selector.vendor(sel), "");
            MemorySegment out = arena.allocate(ValueLayout.JAVA_INT);
            MemorySegment err = Native.error(arena);
            Native.check(turbo_runtime_select_device(handle(), sel, out, err), err);
            return out.get(ValueLayout.JAVA_INT, 0);
        }
    }

    /** The first non-CPU device ({@link SelectPolicy#AUTO}). */
    public int selectDevice() {
        return selectDevice(SelectPolicy.AUTO, null, 0);
    }

    /** Capability cell of (device, task, modality). */
    public Capability capability(int index, Task task, Modality modality) {
        try (Arena arena = Arena.ofConfined()) {
            MemorySegment cap = turbo_capability.allocate(arena);
            turbo_capability.struct_size(cap, (int) turbo_capability.sizeof());
            MemorySegment err = Native.error(arena);
            Native.check(turbo_runtime_capability(handle(), index, task.value(), modality.value(), cap, err), err);
            return Capability.from(cap);
        }
    }

    /** Per-request feasibility check without allocation; throws when the bundle cannot run there. */
    public void canRun(int index, String bundlePath, Task task, Modality modality) {
        try (Arena arena = Arena.ofConfined()) {
            MemorySegment err = Native.error(arena);
            Native.check(turbo_can_run(handle(), index, Native.text(arena, bundlePath), task.value(), modality.value(), err), err);
        }
    }

    /** Create a context on device {@code index}. */
    public Context createContext(int index) {
        return Context.create(this, index);
    }

    /** Load the tokenizer the bundle at {@code bundlePath} declares. */
    public Tokenizer createTokenizer(String bundlePath) {
        try (Arena arena = Arena.ofConfined()) {
            MemorySegment out = arena.allocate(ValueLayout.ADDRESS);
            MemorySegment err = Native.error(arena);
            Native.check(turbo_tokenizer_create(handle(), Native.text(arena, bundlePath), out, err), err);
            return new Tokenizer(out.get(ValueLayout.ADDRESS, 0));
        }
    }

    @Override
    public void close() {
        if (!closed) {
            closed = true;
            turbo_runtime_release(rt);
        }
    }
}
