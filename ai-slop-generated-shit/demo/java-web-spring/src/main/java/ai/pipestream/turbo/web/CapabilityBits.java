// SPDX-License-Identifier: Apache-2.0
package ai.pipestream.turbo.web;

import ai.pipestream.turbo.ffi.TurboNative;
import java.util.ArrayList;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/**
 * The {@code TURBO_CAP_*} feature bits by name, read from the generated ABI
 * constants rather than copied. A device honors exactly the options whose bit
 * it advertises; any other option is refused with
 * {@code TURBO_E_UNSUPPORTED_OPTION} and the field index.
 */
public final class CapabilityBits {
    private static final Map<String, Long> BITS = bits();

    private CapabilityBits() {}

    private static Map<String, Long> bits() {
        Map<String, Long> m = new LinkedHashMap<>();
        m.put("ASYNC", Integer.toUnsignedLong(TurboNative.TURBO_CAP_ASYNC()));
        m.put("HOST_PTR_IMPORT", Integer.toUnsignedLong(TurboNative.TURBO_CAP_HOST_PTR_IMPORT()));
        m.put("DEVICE_RESULT", Integer.toUnsignedLong(TurboNative.TURBO_CAP_DEVICE_RESULT()));
        m.put("EXTERNAL_QUEUE", Integer.toUnsignedLong(TurboNative.TURBO_CAP_EXTERNAL_QUEUE()));
        m.put("DMABUF", Integer.toUnsignedLong(TurboNative.TURBO_CAP_DMABUF()));
        m.put("UNIFIED_MEMORY", Integer.toUnsignedLong(TurboNative.TURBO_CAP_UNIFIED_MEMORY()));
        m.put("DYNAMIC_SHAPE", Integer.toUnsignedLong(TurboNative.TURBO_CAP_DYNAMIC_SHAPE()));
        m.put("WEIGHT_SHARING", Integer.toUnsignedLong(TurboNative.TURBO_CAP_WEIGHT_SHARING()));
        m.put("DEVICE_TOKENIZE", Integer.toUnsignedLong(TurboNative.TURBO_CAP_DEVICE_TOKENIZE()));
        m.put("DEVICE_POSTPROCESS", Integer.toUnsignedLong(TurboNative.TURBO_CAP_DEVICE_POSTPROCESS()));
        m.put("DETERMINISTIC", Integer.toUnsignedLong(TurboNative.TURBO_CAP_DETERMINISTIC()));
        m.put("OPT_TRUNCATE", Integer.toUnsignedLong(TurboNative.TURBO_CAP_OPT_TRUNCATE()));
        m.put("OPT_MAX_TOKENS", Integer.toUnsignedLong(TurboNative.TURBO_CAP_OPT_MAX_TOKENS()));
        m.put("OPT_PROMPT_ROLE", Integer.toUnsignedLong(TurboNative.TURBO_CAP_OPT_PROMPT_ROLE()));
        m.put("OPT_NORMALIZE", Integer.toUnsignedLong(TurboNative.TURBO_CAP_OPT_NORMALIZE()));
        m.put("OPT_POOLING_OVERRIDE", Integer.toUnsignedLong(TurboNative.TURBO_CAP_OPT_POOLING_OVERRIDE()));
        m.put("OPT_OUTPUT_DIM", Integer.toUnsignedLong(TurboNative.TURBO_CAP_OPT_OUTPUT_DIM()));
        m.put("OPT_OUTPUT_DTYPE", Integer.toUnsignedLong(TurboNative.TURBO_CAP_OPT_OUTPUT_DTYPE()));
        m.put("OPT_TOP_N", Integer.toUnsignedLong(TurboNative.TURBO_CAP_OPT_TOP_N()));
        m.put("OPT_AGGREGATION", Integer.toUnsignedLong(TurboNative.TURBO_CAP_OPT_AGGREGATION()));
        m.put("OPT_RAW_SCORES", Integer.toUnsignedLong(TurboNative.TURBO_CAP_OPT_RAW_SCORES()));
        m.put("OPT_GEN_STRUCTURED", TurboNative.TURBO_CAP_OPT_GEN_STRUCTURED());
        m.put("OPT_GEN_TOOLS", TurboNative.TURBO_CAP_OPT_GEN_TOOLS());
        m.put("OPT_GEN_N", TurboNative.TURBO_CAP_OPT_GEN_N());
        m.put("OPT_GEN_LOGIT_BIAS", TurboNative.TURBO_CAP_OPT_GEN_LOGIT_BIAS());
        m.put("OPT_GEN_PENALTIES", TurboNative.TURBO_CAP_OPT_GEN_PENALTIES());
        m.put("OPT_GEN_LOGPROBS", TurboNative.TURBO_CAP_OPT_GEN_LOGPROBS());
        m.put("OPT_GEN_STOP_STRINGS", TurboNative.TURBO_CAP_OPT_GEN_STOP_STRINGS());
        m.put("OPT_GEN_SEED", TurboNative.TURBO_CAP_OPT_GEN_SEED());
        m.put("OPT_GEN_SAMPLING", TurboNative.TURBO_CAP_OPT_GEN_SAMPLING());
        m.put("OPT_GEN_MIN_TOKENS", TurboNative.TURBO_CAP_OPT_GEN_MIN_TOKENS());
        m.put("OPT_GEN_ECHO", TurboNative.TURBO_CAP_OPT_GEN_ECHO());
        m.put("OPT_GEN_STOP_TOKENS", TurboNative.TURBO_CAP_OPT_GEN_STOP_TOKENS());
        return Collections.unmodifiableMap(m);
    }

    /** The names of every bit set in {@code caps}, in ABI order. */
    public static List<String> names(long caps) {
        List<String> out = new ArrayList<>();
        for (Map.Entry<String, Long> e : BITS.entrySet()) {
            if ((caps & e.getValue()) == e.getValue()) {
                out.add(e.getKey());
            }
        }
        return out;
    }

    /** The bit value of one feature name, for tests and documentation. */
    public static long bit(String name) {
        Long v = BITS.get(name);
        if (v == null) {
            throw new IllegalArgumentException("no capability bit named " + name);
        }
        return v;
    }
}
