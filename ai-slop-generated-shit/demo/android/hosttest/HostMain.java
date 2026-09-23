// SPDX-License-Identifier: Apache-2.0
package ai.pipestream.turbo.android;

/** Drives TurboEngine on the host the way MainActivity does on a device, and checks the contract. */
public final class HostMain {
    public static void main(String[] args) {
        String bundle = args[0];
        try (TurboEngine engine = new TurboEngine(bundle, null, 4)) {
            System.out.println(engine.describe());
            String[] texts = {"a brown dog runs through the grass", "a dog is running on the lawn", "the stock market closed higher"};
            float[][] rows = engine.embed(texts);
            if (rows.length != 3 || rows[0].length != engine.dim()) {
                throw new AssertionError("unexpected shape " + rows.length + " x " + rows[0].length);
            }
            System.out.println("cosine similarity:");
            for (int a = 0; a < rows.length; a++) {
                StringBuilder sb = new StringBuilder();
                for (float[] row : rows) {
                    sb.append(String.format(" %6.3f", TurboEngine.cosine(rows[a], row)));
                }
                System.out.println(sb + "  " + texts[a]);
                if (Math.abs(TurboEngine.cosine(rows[a], rows[a]) - 1.0) > 1e-4) {
                    throw new AssertionError("row " + a + " is not unit norm");
                }
            }
            // A supplementary character crosses as UTF-8, not as the CESU-8
            // surrogate pair GetStringUTFChars would produce, which the
            // library refuses with TURBO_E_INVALID_UTF8.
            float[][] supplementary = engine.embed("great work 🎉", "great work");
            if (supplementary.length != 2 || supplementary[0].length != engine.dim()) {
                throw new AssertionError("supplementary characters did not embed");
            }
            System.out.println("supplementary characters embed as UTF-8");

            // The engine refuses a batch wider than it holds, before touching the library.
            try {
                engine.embed("1", "2", "3", "4", "5");
                throw new AssertionError("a batch of 5 on an engine of 4 was accepted");
            } catch (IllegalArgumentException expected) {
                System.out.println("over-capacity batch refused: " + expected.getMessage());
            }
        }
        // A missing bundle is a TurboException with the status name, not a crash.
        try (TurboEngine bad = new TurboEngine(bundle + "-does-not-exist", null, 1)) {
            throw new AssertionError("a missing bundle opened: " + bad.describe());
        } catch (TurboException e) {
            System.out.println("missing bundle refused: " + e.statusName());
        }
        System.out.println("android host test: OK");
    }
}
