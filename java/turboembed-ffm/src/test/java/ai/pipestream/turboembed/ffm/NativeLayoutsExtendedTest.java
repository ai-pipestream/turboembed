// SPDX-License-Identifier: Apache-2.0
package ai.pipestream.turboembed.ffm;

import ai.pipestream.turboembed.Device;
import java.io.IOException;
import java.lang.foreign.Arena;
import java.lang.foreign.StructLayout;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.TimeUnit;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;
import static org.junit.jupiter.api.Assertions.*;

/**
 * Extends {@link NativeLayoutTest}'s compiled-probe pattern to the header's
 * {@code #define} constants, which the shared {@code layout_probe.c} does not
 * print: ABI version, error codes, device constants, and capability flags.
 * Layouts and field offsets of every {@code te_*} struct are already covered by
 * {@link NativeLayoutTest}; the descriptor-initialization convention the
 * header requires for all of them is checked here from pure Java.
 */
class NativeLayoutsExtendedTest {
    @TempDir Path temporary;

    private static final String PROBE_SOURCE = """
        #include <stdio.h>
        #include <turboembed_prepared.h>
        #define PRINT_UINT(name) printf("%s %u\\n", #name, (unsigned)(name))
        #define PRINT_ULL(name) printf("%s %llu\\n", #name, (unsigned long long)(name))
        int main(void) {
          PRINT_UINT(TE_PREPARED_VERSION);
          PRINT_UINT(TE_OK);
          PRINT_UINT(TE_INVALID_ARGUMENT);
          PRINT_UINT(TE_NOT_FOUND);
          PRINT_UINT(TE_NOT_IMPLEMENTED);
          PRINT_UINT(TE_UNAVAILABLE);
          PRINT_UINT(TE_INTERNAL);
          PRINT_UINT(TE_OUT_OF_MEMORY);
          PRINT_UINT(TE_BUSY);
          PRINT_UINT(TE_ABI_MISMATCH);
          PRINT_UINT(TE_INTEGRITY_ERROR);
          PRINT_UINT(TE_DEVICE_AUTO);
          PRINT_UINT(TE_DEVICE_OPENVINO_GPU);
          PRINT_UINT(TE_DEVICE_OPENVINO_CPU);
          PRINT_ULL(TE_CAP_TEXT);
          PRINT_ULL(TE_CAP_PREPARED_I32);
          PRINT_ULL(TE_CAP_OPENCL_RESULT);
          PRINT_ULL(TE_CAP_HOST_READ);
          return 0;
        }
        """;

    @Test void headerDefinesPinnedByCompiledProbe() throws Exception {
        Map<String, Long> constants = compileAndRun();
        // ABI v1 is frozen; NativeBindings gates on version() == 1 and every
        // descriptor is initialized with version 1.
        assertEquals(1, constants.get("TE_PREPARED_VERSION"), "prepared ABI version");
        // Error codes are surfaced to Java unchanged as NativeException.code().
        // The adapter hard-codes 5 for a null success handle (TE_INTERNAL) and
        // 8 for the ABI gate (TE_ABI_MISMATCH); the runtime codes 1, 3 and 4 are
        // exercised against hardware by NativeContractTest.
        for (int code = 0; code <= 9; code++) {
            String name = switch (code) {
                case 0 -> "TE_OK"; case 1 -> "TE_INVALID_ARGUMENT"; case 2 -> "TE_NOT_FOUND";
                case 3 -> "TE_NOT_IMPLEMENTED"; case 4 -> "TE_UNAVAILABLE"; case 5 -> "TE_INTERNAL";
                case 6 -> "TE_OUT_OF_MEMORY"; case 7 -> "TE_BUSY"; case 8 -> "TE_ABI_MISMATCH";
                default -> "TE_INTEGRITY_ERROR";
            };
            assertEquals(code, constants.get(name), name);
        }
    }

    @Test void deviceConstantsMatchDeviceEnumOrder() throws Exception {
        Map<String, Long> constants = compileAndRun();
        // FfmTurboEmbed.context maps the enum in declaration order onto these
        // constants; NativeContractTest asserts the same numbers on handles.
        assertArrayEquals(new Device[]{Device.AUTO, Device.OPENVINO_GPU, Device.OPENVINO_CPU},
            Device.values());
        assertEquals(0, constants.get("TE_DEVICE_AUTO"), "AUTO selects the host GPU");
        assertEquals(1, constants.get("TE_DEVICE_OPENVINO_GPU"), "explicit GPU selection");
        assertEquals(2, constants.get("TE_DEVICE_OPENVINO_CPU"), "explicit CPU selection");
    }

    @Test void capabilityFlagsAreDistinctPowersOfTwo() throws Exception {
        Map<String, Long> constants = compileAndRun();
        List<Long> flags = List.of(constants.get("TE_CAP_TEXT"), constants.get("TE_CAP_PREPARED_I32"),
            constants.get("TE_CAP_OPENCL_RESULT"), constants.get("TE_CAP_HOST_READ"));
        assertEquals(List.of(1L, 2L, 4L, 8L), flags);
        assertEquals(4, flags.stream().distinct().count(), "capability flags must be combinable");
    }

    @Test void descriptorInitializesSizeAndVersionForEveryLayout() {
        // Header contract: "Every descriptor must be initialized with
        // sizeof(descriptor) and version 1." te_error is caller-owned error
        // storage and te_text is an input span; neither is a versioned
        // descriptor, so NativeBindings.descriptor applies to the rest.
        List<StructLayout> descriptors = NativeLayouts.ALL.stream()
            .filter(layout -> layout.memberLayouts().stream()
                .anyMatch(member -> member.name().filter("struct_size"::equals).isPresent()))
            .toList();
        assertEquals(List.of("te_context_options", "te_context_info", "te_model_info", "te_slot_options",
                "te_result_info", "te_opencl_view", "te_slot_stats", "te_device_info"),
            descriptors.stream().map(layout -> layout.name().orElseThrow()).toList());
        try (Arena arena = Arena.ofConfined()) {
            for (StructLayout layout : descriptors) {
                var segment = NativeBindings.descriptor(arena, layout);
                String name = layout.name().orElseThrow();
                assertEquals(layout.byteSize(), segment.get(NativeBindings.I, NativeLayouts.offset(layout, "struct_size")),
                    name + " struct_size");
                assertEquals(1, segment.get(NativeBindings.I, NativeLayouts.offset(layout, "version")), name + " version");
            }
        }
    }

    private Map<String, Long> compileAndRun() throws Exception {
        Path root = Path.of("../..").toAbsolutePath().normalize();
        Path source = temporary.resolve("constants-probe.c");
        Files.writeString(source, PROBE_SOURCE);
        Path probe = temporary.resolve("constants-probe");
        run(List.of("cc", "-std=c11", "-Wall", "-Wextra", "-Werror", "-I" + root.resolve("include"),
            source.toString(), "-o", probe.toString()));
        Map<String, Long> constants = new LinkedHashMap<>();
        for (String line : run(List.of(probe.toString())).lines().toList()) {
            String[] parts = line.split(" ");
            constants.put(parts[0], Long.valueOf(parts[1]));
        }
        return constants;
    }

    private String run(List<String> command) throws IOException, InterruptedException {
        Path output = Files.createTempFile(temporary, "command", ".txt");
        Process process = new ProcessBuilder(command).redirectErrorStream(true).redirectOutput(output.toFile()).start();
        if (!process.waitFor(30, TimeUnit.SECONDS)) {
            process.destroyForcibly(); fail("command timed out: " + command);
        }
        String text = Files.readString(output);
        assertEquals(0, process.exitValue(), text);
        return text;
    }
}
