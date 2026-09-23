// SPDX-License-Identifier: Apache-2.0
package ai.pipestream.turbo.android;

import android.app.Activity;
import android.os.Bundle;
import android.widget.Button;
import android.widget.EditText;
import android.widget.TextView;
import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.util.ArrayList;
import java.util.List;

/**
 * Turbo Android demo: a bundle copied from the APK's assets to the app's
 * files directory, an engine on the best device (AUTO), and a screen that
 * embeds the sentences typed in and shows their cosine similarities.
 * Failures are shown verbatim, never hidden.
 */
public final class MainActivity extends Activity {
    private TurboEngine engine;

    @Override
    protected void onCreate(Bundle saved) {
        super.onCreate(saved);
        setContentView(R.layout.activity_main);
        EditText input = findViewById(R.id.input);
        TextView output = findViewById(R.id.output);
        Button run = findViewById(R.id.run);
        try {
            File bundle = unpackBundle("bundle");
            engine = new TurboEngine(bundle.getAbsolutePath(), null, 8);
            output.setText(engine.describe());
        } catch (RuntimeException | IOException e) {
            output.setText("failed to open the bundle: " + e);
            run.setEnabled(false);
            return;
        }
        run.setOnClickListener(v -> {
            String[] texts = input.getText().toString().split("\n");
            List<String> kept = new ArrayList<>();
            for (String t : texts) {
                if (!t.isBlank()) {
                    kept.add(t.trim());
                }
            }
            if (kept.isEmpty()) {
                output.setText("type one sentence per line");
                return;
            }
            try {
                float[][] rows = engine.embed(kept.toArray(new String[0]));
                StringBuilder sb = new StringBuilder(engine.describe()).append("\n\ncosine similarity:\n");
                for (int a = 0; a < rows.length; a++) {
                    for (int b = 0; b < rows.length; b++) {
                        sb.append(String.format(" %6.3f", TurboEngine.cosine(rows[a], rows[b])));
                    }
                    sb.append("  ").append(kept.get(a)).append('\n');
                }
                output.setText(sb.toString());
            } catch (RuntimeException e) {
                output.setText("error: " + e.getMessage());
            }
        });
    }

    @Override
    protected void onDestroy() {
        if (engine != null) {
            engine.close();
        }
        super.onDestroy();
    }

    /** Copy assets/<name>/* into files/<name>/ once; bundles are read from the file system. */
    private File unpackBundle(String name) throws IOException {
        File dir = new File(getFilesDir(), name);
        if (new File(dir, "bundle.json").exists()) {
            return dir;
        }
        dir.mkdirs();
        for (String entry : getAssets().list(name)) {
            try (InputStream in = getAssets().open(name + "/" + entry);
                    OutputStream out = new FileOutputStream(new File(dir, entry))) {
                in.transferTo(out);
            }
        }
        return dir;
    }
}
