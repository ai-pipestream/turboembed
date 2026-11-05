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
        // Avoid java.lang.Runtime's version API: android.jar (compileSdk) does not declare it,
        // so Android javac fails on this shared source even behind an isAndroid() guard.
        int version = android ? 0 : jvmFeature(System.getProperty("java.specification.version", "0"));
        return select(override, version, android, !android && foreignAvailable());
    }

    static boolean isAndroid(String vm, String runtime) {
        return vm.equalsIgnoreCase("Dalvik") || vm.equalsIgnoreCase("ART")
                || runtime.toLowerCase(Locale.ROOT).contains("android");
    }

    /**
     * Parses the major feature from {@code java.specification.version} ({@code "21"},
     * {@code "25"}, or the legacy {@code "1.8"} form). Returns 0 when the value is absent or
     * not a number.
     */
    static int jvmFeature(String specificationVersion) {
        if (specificationVersion == null) {
            return 0;
        }
        String spec = specificationVersion.trim();
        if (spec.isEmpty()) {
            return 0;
        }
        if (spec.startsWith("1.")) {
            spec = spec.substring(2);
        }
        int end = 0;
        while (end < spec.length() && Character.isDigit(spec.charAt(end))) {
            end++;
        }
        if (end == 0) {
            return 0;
        }
        try {
            return Integer.parseInt(spec.substring(0, end));
        } catch (NumberFormatException e) {
            return 0;
        }
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
