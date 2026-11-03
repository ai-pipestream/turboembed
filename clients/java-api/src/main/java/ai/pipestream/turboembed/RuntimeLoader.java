package ai.pipestream.turboembed;

import java.lang.reflect.InvocationTargetException;
import java.util.Locale;
import java.util.Objects;

/** Loads only the selected implementation, so Java 21 never resolves Java 25 classes. */
final class RuntimeLoader {
    private RuntimeLoader() {}

    static TurboBridge selected(String override) {
        boolean android = isAndroid(System.getProperty("java.vm.name", ""),
                System.getProperty("java.runtime.name", ""));
        // Android need not provide Runtime.version(), nor the foreign-memory API.
        int version = android ? 0 : Runtime.version().feature();
        return select(override, version, android, !android && foreignAvailable());
    }

    static boolean isAndroid(String vm, String runtime) {
        return vm.equalsIgnoreCase("Dalvik") || vm.equalsIgnoreCase("ART")
                || runtime.toLowerCase(Locale.ROOT).contains("android");
    }

    static TurboBridge select(String override, int version, boolean android, boolean foreign) {
        Objects.requireNonNull(override, "override");
        boolean panama = !android && version >= 25 && foreign;
        return switch (override.toLowerCase(Locale.ROOT)) {
            case "auto" -> panama ? TurboBridge.PANAMA : TurboBridge.JNI;
            case "jni" -> TurboBridge.JNI;
            case "panama" -> {
                if (!panama) {
                    throw new IllegalStateException("Panama requires Java 25+ with java.lang.foreign, outside Android");
                }
                yield TurboBridge.PANAMA;
            }
            default -> throw new IllegalArgumentException("turbo.bridge must be auto, jni, or panama: " + override);
        };
    }

    private static boolean foreignAvailable() {
        try {
            Class.forName("java.lang.foreign.Arena", false, null);
            return true;
        } catch (ClassNotFoundException e) {
            return false;
        }
    }

    static TurboRuntime load(TurboBridge bridge) {
        Objects.requireNonNull(bridge, "bridge");
        if (bridge == TurboBridge.PANAMA) {
            selected("panama"); // Validate before attempting to load Java 25 bytecode.
        }
        String name = switch (bridge) {
            case JNI -> "ai.pipestream.turboembed.jni.JniRuntime";
            case PANAMA -> "ai.pipestream.turboembed.panama.PanamaRuntime";
        };
        try {
            Class<? extends TurboRuntime> type = Class.forName(name, true, TurboRuntime.class.getClassLoader())
                    .asSubclass(TurboRuntime.class);
            return type.getConstructor().newInstance();
        } catch (ClassNotFoundException e) {
            throw new IllegalStateException("Selected " + bridge + " bridge is missing; add turboembed-"
                    + bridge.name().toLowerCase(Locale.ROOT) + " beside turboembed-api", e);
        } catch (InvocationTargetException e) {
            throw new IllegalStateException("Could not initialize " + bridge + " bridge", e.getCause());
        } catch (ReflectiveOperationException | ClassCastException | LinkageError e) {
            throw new IllegalStateException("Incompatible " + bridge + " bridge; use matching API and bridge versions", e);
        }
    }
}
