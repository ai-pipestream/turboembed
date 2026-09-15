// SPDX-License-Identifier: Apache-2.0
package ai.pipestream.turboembed.ffm;

import java.lang.foreign.StructLayout;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.TimeUnit;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;
import static org.junit.jupiter.api.Assertions.*;

class NativeLayoutTest {
    @TempDir Path temporary;

    @Test void declarationsMatchCompiledCanonicalHeader() throws Exception {
        Path root = Path.of("../..").toAbsolutePath().normalize();
        Path probe = temporary.resolve("layout-probe");
        run(List.of("cc", "-std=c11", "-Wall", "-Wextra", "-Werror", "-I" + root.resolve("include"),
            root.resolve("native/turboembed/sdk/layout_probe.c").toString(), "-o", probe.toString()));
        List<String> actual = run(List.of(probe.toString())).lines().toList();
        List<String> expected = new ArrayList<>();
        for (StructLayout layout : NativeLayouts.ALL) {
            String name = layout.name().orElseThrow();
            expected.add(name + " " + layout.byteSize() + " " + layout.byteAlignment());
            for (var member : layout.memberLayouts()) {
                String field = member.name().orElseThrow();
                expected.add(name + "." + field + " " + NativeLayouts.offset(layout, field));
            }
        }
        assertEquals(expected, actual);
    }

    private String run(List<String> command) throws Exception {
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
