// SPDX-License-Identifier: Apache-2.0
package ai.pipestream.turbo.web;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import ai.pipestream.turbo.Task;
import org.junit.jupiter.api.Test;

/**
 * How bundles become served models, checked without an HTTP layer. Every case
 * here builds its own runtime, so a refusal is the constructor's, not a
 * Spring context failure.
 */
class ConfigurationTest {
    private static String bundle(String key) {
        String path = System.getProperty(key);
        assertTrue(path != null && !path.isBlank(), "surefire did not pass " + key);
        return path;
    }

    private static TurboProperties.ModelConfig model(String name, String path) {
        TurboProperties.ModelConfig c = new TurboProperties.ModelConfig();
        c.setName(name);
        c.setBundle(path);
        c.setSessions(1);
        return c;
    }

    @Test
    void aNameLeftOutComesFromTheBundlesModelId() {
        assertEquals("mock-embedding", TurboService.deriveName("turbo/mock-embedding"));
        assertEquals("all-MiniLM-L6-v2", TurboService.deriveName("sentence-transformers/all-MiniLM-L6-v2"));
        assertEquals("a-b", TurboService.deriveName("vendor/a b"));
        assertEquals("model", TurboService.deriveName(""));
    }

    @Test
    void theShorthandKeysLoadOneModelEachAndAreNamedAfterTheirBundle() {
        TurboProperties props = new TurboProperties();
        props.setBundle(bundle("turbo.bundle"));
        props.setSessions(1);
        props.setGenerateBundle(bundle("turbo.generate-bundle"));
        try (TurboService service = new TurboService(props)) {
            assertEquals(2, service.loaded().size());
            assertEquals("mock-embedding", service.loaded().get(0).name());
            assertEquals("mock-generative", service.loaded().get(1).name());
            assertTrue(service.serves(Task.EMBED));
            assertTrue(service.serves(Task.GENERATE));
            assertEquals("mock-embedding", service.modelFor(null, Task.EMBED).name());
        }
    }

    @Test
    void theShorthandKeysAndTheModelsListAddUp() {
        TurboProperties props = new TurboProperties();
        props.setBundle(bundle("turbo.bundle"));
        props.setSessions(1);
        props.getModels().add(model("rr", bundle("turbo.rerank-bundle")));
        try (TurboService service = new TurboService(props)) {
            assertEquals(2, service.loaded().size());
            assertEquals("rr", service.loaded().get(1).name());
            assertTrue(service.serves(Task.RERANK));
        }
    }

    @Test
    void noBundleAtAllIsRefusedWithWhatToPass() {
        IllegalStateException e = assertThrows(IllegalStateException.class, () -> new TurboService(new TurboProperties()));
        assertTrue(e.getMessage().contains("--turbo.bundle="), e.getMessage());
    }

    @Test
    void twoModelsUnderOneNameAreRefused() {
        TurboProperties props = new TurboProperties();
        props.getModels().add(model("same", bundle("turbo.bundle")));
        props.getModels().add(model("same", bundle("turbo.rerank-bundle")));
        IllegalStateException e = assertThrows(IllegalStateException.class, () -> new TurboService(props));
        assertTrue(e.getMessage().contains("served as `same`"), e.getMessage());
    }

    @Test
    void aGenericRunBundleIsRefusedWithTheKindsThatAreServed() {
        TurboProperties props = new TurboProperties();
        props.getModels().add(model("generic", bundle("turbo.generic-bundle")));
        IllegalStateException e = assertThrows(IllegalStateException.class, () -> new TurboService(props));
        assertTrue(e.getMessage().contains("GENERIC"), e.getMessage());
    }

    @Test
    void aTaskNoModelPerformsNamesTheBundleToStartWith() {
        TurboProperties props = new TurboProperties();
        props.setBundle(bundle("turbo.bundle"));
        props.setSessions(1);
        try (TurboService service = new TurboService(props)) {
            TurboService.TaskNotServed e =
                    assertThrows(TurboService.TaskNotServed.class, () -> service.modelFor(null, Task.RERANK));
            assertTrue(e.getMessage().contains("no loaded model performs RERANK"), e.getMessage());
        }
    }

    @Test
    void theFeatureBitNamesComeFromTheAbiConstants() {
        long bits = CapabilityBits.bit("DETERMINISTIC") | CapabilityBits.bit("OPT_GEN_SEED");
        assertEquals(java.util.List.of("DETERMINISTIC", "OPT_GEN_SEED"), CapabilityBits.names(bits));
        assertTrue(CapabilityBits.names(0).isEmpty());
        assertThrows(IllegalArgumentException.class, () -> CapabilityBits.bit("NOT_A_BIT"));
    }
}
