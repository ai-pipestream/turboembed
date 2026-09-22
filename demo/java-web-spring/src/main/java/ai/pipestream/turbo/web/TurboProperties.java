// SPDX-License-Identifier: Apache-2.0
package ai.pipestream.turbo.web;

import java.util.ArrayList;
import java.util.List;
import org.springframework.boot.context.properties.ConfigurationProperties;

/**
 * Server configuration under the {@code turbo} prefix.
 *
 * <p>A model is either an entry of the {@code turbo.models} list or one of the
 * two single-model shorthands this app has always had. Both forms are additive
 * and may be combined: {@code turbo.bundle} adds one model, {@code
 * turbo.generate-bundle} adds another, and every {@code turbo.models[i]} entry
 * adds one more. Names must be unique; a shorthand entry is named after the
 * last segment of the bundle's {@code model_id} unless {@code turbo.name} or
 * {@code turbo.generate-name} says otherwise.
 */
@ConfigurationProperties(prefix = "turbo")
public class TurboProperties {
    /** One configured model: a bundle, the device to run it on, and its pool sizes. */
    public static class ModelConfig {
        private String name = "";
        private String bundle = "";
        private String providerLib = "";
        private String provider = "";
        private int ordinal = 0;
        private int sessions = 2;
        private int maxBatch = 0;
        private int generations = 2;
        private String tokenizerBundle = "";

        /** The name this model is served under; empty means "derive it from the bundle's model_id". */
        public String getName() {
            return name;
        }

        public void setName(String name) {
            this.name = name;
        }

        /** Bundle directory. Required. */
        public String getBundle() {
            return bundle;
        }

        public void setBundle(String bundle) {
            this.bundle = bundle;
        }

        /** Provider library to load ({@code libturbo_provider_*.so}); empty means the built-in providers only. */
        public String getProviderLib() {
            return providerLib;
        }

        public void setProviderLib(String providerLib) {
            this.providerLib = providerLib;
        }

        /** Provider id for an explicit device; empty means AUTO, which never selects a CPU. */
        public String getProvider() {
            return provider;
        }

        public void setProvider(String provider) {
            this.provider = provider;
        }

        /** Device ordinal within the provider, used only with an explicit provider. */
        public int getOrdinal() {
            return ordinal;
        }

        public void setOrdinal(int ordinal) {
            this.ordinal = ordinal;
        }

        /** Sessions in the pool for a session task. */
        public int getSessions() {
            return sessions;
        }

        public void setSessions(int sessions) {
            this.sessions = sessions;
        }

        /** Batch width of each session; 0 means the bundle's own {@code limits.max_batch}. */
        public int getMaxBatch() {
            return maxBatch;
        }

        public void setMaxBatch(int maxBatch) {
            this.maxBatch = maxBatch;
        }

        /** Concurrent generations allowed on a generative model. */
        public int getGenerations() {
            return generations;
        }

        public void setGenerations(int generations) {
            this.generations = generations;
        }

        /**
         * Bundle whose tokenizer serves {@code /api/v1/tokenize} for this model.
         * Empty means the model's own bundle. Set it when the model bundle
         * carries no {@code tokenizer.json} file entry, as the mock bundles do
         * not.
         */
        public String getTokenizerBundle() {
            return tokenizerBundle;
        }

        public void setTokenizerBundle(String tokenizerBundle) {
            this.tokenizerBundle = tokenizerBundle;
        }
    }

    private List<ModelConfig> models = new ArrayList<>();

    private String name = "";
    private String bundle = "";
    private String providerLib = "";
    private String provider = "";
    private int ordinal = 0;
    private int sessions = 2;
    private int maxBatch = 0;
    private String tokenizerBundle = "";

    private String generateName = "";
    private String generateBundle = "";
    private String generateProviderLib = "";
    private String generateProvider = "";
    private int generateOrdinal = 0;
    private int generations = 2;
    private String generateTokenizerBundle = "";

    public List<ModelConfig> getModels() {
        return models;
    }

    public void setModels(List<ModelConfig> models) {
        this.models = models;
    }

    public String getName() {
        return name;
    }

    public void setName(String name) {
        this.name = name;
    }

    public String getBundle() {
        return bundle;
    }

    public void setBundle(String bundle) {
        this.bundle = bundle;
    }

    public String getProviderLib() {
        return providerLib;
    }

    public void setProviderLib(String providerLib) {
        this.providerLib = providerLib;
    }

    public String getProvider() {
        return provider;
    }

    public void setProvider(String provider) {
        this.provider = provider;
    }

    public int getOrdinal() {
        return ordinal;
    }

    public void setOrdinal(int ordinal) {
        this.ordinal = ordinal;
    }

    public int getSessions() {
        return sessions;
    }

    public void setSessions(int sessions) {
        this.sessions = sessions;
    }

    public int getMaxBatch() {
        return maxBatch;
    }

    public void setMaxBatch(int maxBatch) {
        this.maxBatch = maxBatch;
    }

    public String getTokenizerBundle() {
        return tokenizerBundle;
    }

    public void setTokenizerBundle(String tokenizerBundle) {
        this.tokenizerBundle = tokenizerBundle;
    }

    public String getGenerateName() {
        return generateName;
    }

    public void setGenerateName(String generateName) {
        this.generateName = generateName;
    }

    public String getGenerateBundle() {
        return generateBundle;
    }

    public void setGenerateBundle(String generateBundle) {
        this.generateBundle = generateBundle;
    }

    public String getGenerateProviderLib() {
        return generateProviderLib;
    }

    public void setGenerateProviderLib(String generateProviderLib) {
        this.generateProviderLib = generateProviderLib;
    }

    public String getGenerateProvider() {
        return generateProvider;
    }

    public void setGenerateProvider(String generateProvider) {
        this.generateProvider = generateProvider;
    }

    public int getGenerateOrdinal() {
        return generateOrdinal;
    }

    public void setGenerateOrdinal(int generateOrdinal) {
        this.generateOrdinal = generateOrdinal;
    }

    public int getGenerations() {
        return generations;
    }

    public void setGenerations(int generations) {
        this.generations = generations;
    }

    public String getGenerateTokenizerBundle() {
        return generateTokenizerBundle;
    }

    public void setGenerateTokenizerBundle(String generateTokenizerBundle) {
        this.generateTokenizerBundle = generateTokenizerBundle;
    }

    /** The configured models in load order: the {@code turbo.bundle} shorthand, the generative shorthand, then the list. */
    public List<ModelConfig> resolved() {
        List<ModelConfig> out = new ArrayList<>();
        if (!bundle.isBlank()) {
            ModelConfig c = new ModelConfig();
            c.setName(name);
            c.setBundle(bundle);
            c.setProviderLib(providerLib);
            c.setProvider(provider);
            c.setOrdinal(ordinal);
            c.setSessions(sessions);
            c.setMaxBatch(maxBatch);
            c.setTokenizerBundle(tokenizerBundle);
            out.add(c);
        }
        if (!generateBundle.isBlank()) {
            ModelConfig c = new ModelConfig();
            c.setName(generateName);
            c.setBundle(generateBundle);
            c.setProviderLib(generateProviderLib);
            c.setProvider(generateProvider);
            c.setOrdinal(generateOrdinal);
            c.setGenerations(generations);
            c.setTokenizerBundle(generateTokenizerBundle);
            out.add(c);
        }
        for (ModelConfig c : models) {
            out.add(c);
        }
        return out;
    }
}
