// SPDX-License-Identifier: Apache-2.0
package ai.pipestream.turbo.web.oip;

import java.util.Arrays;
import java.util.Locale;
import java.util.Map;
import java.util.stream.Collectors;

/**
 * Reading the protocol's {@code parameters} objects, which are free-form JSON.
 * Every accessor names the parameter and says what it expected, so a wrong
 * type is a precise 400 rather than a cast failure.
 */
final class Params {
    private Params() {}

    static Object raw(Map<String, Object> parameters, String key) {
        return parameters == null ? null : parameters.get(key);
    }

    static Integer integer(Map<String, Object> parameters, String key) {
        Object v = raw(parameters, key);
        if (v == null) {
            return null;
        }
        if (v instanceof Number n && n.doubleValue() == Math.rint(n.doubleValue())) {
            return n.intValue();
        }
        throw new IllegalArgumentException("parameter `" + key + "` must be an integer, not " + describe(v));
    }

    static Long longValue(Map<String, Object> parameters, String key) {
        Object v = raw(parameters, key);
        if (v == null) {
            return null;
        }
        if (v instanceof Number n && n.doubleValue() == Math.rint(n.doubleValue())) {
            return n.longValue();
        }
        throw new IllegalArgumentException("parameter `" + key + "` must be an integer, not " + describe(v));
    }

    static Float number(Map<String, Object> parameters, String key) {
        Object v = raw(parameters, key);
        if (v == null) {
            return null;
        }
        if (v instanceof Number n) {
            return n.floatValue();
        }
        throw new IllegalArgumentException("parameter `" + key + "` must be a number, not " + describe(v));
    }

    static Boolean flag(Map<String, Object> parameters, String key) {
        Object v = raw(parameters, key);
        if (v == null) {
            return null;
        }
        if (v instanceof Boolean b) {
            return b;
        }
        throw new IllegalArgumentException("parameter `" + key + "` must be true or false, not " + describe(v));
    }

    static String text(Map<String, Object> parameters, String key) {
        Object v = raw(parameters, key);
        if (v == null) {
            return null;
        }
        if (v instanceof String s) {
            return s;
        }
        throw new IllegalArgumentException("parameter `" + key + "` must be a string, not " + describe(v));
    }

    static java.util.List<String> strings(Map<String, Object> parameters, String key) {
        Object v = raw(parameters, key);
        if (v == null) {
            return null;
        }
        if (v instanceof java.util.List<?> list) {
            java.util.List<String> out = new java.util.ArrayList<>(list.size());
            for (int i = 0; i < list.size(); i++) {
                Object e = list.get(i);
                if (!(e instanceof String s)) {
                    throw new IllegalArgumentException(
                            "parameter `" + key + "[" + i + "]` must be a string, not " + describe(e));
                }
                out.add(s);
            }
            return out;
        }
        throw new IllegalArgumentException("parameter `" + key + "` must be an array of strings, not " + describe(v));
    }

    static <E extends Enum<E>> E constant(Class<E> type, Map<String, Object> parameters, String key) {
        String v = text(parameters, key);
        if (v == null) {
            return null;
        }
        try {
            return Enum.valueOf(type, v.toUpperCase(Locale.ROOT));
        } catch (IllegalArgumentException e) {
            String allowed = Arrays.stream(type.getEnumConstants()).map(String::valueOf).collect(Collectors.joining(", "));
            throw new IllegalArgumentException(
                    "parameter `" + key + "` must be one of [" + allowed + "], not \"" + v + "\"");
        }
    }

    private static String describe(Object v) {
        if (v == null) {
            return "null";
        }
        if (v instanceof String s) {
            return "the string \"" + s + "\"";
        }
        return "a " + v.getClass().getSimpleName();
    }
}
