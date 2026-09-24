// SPDX-License-Identifier: Apache-2.0
package ai.pipestream.turbo.web;

import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.test.autoconfigure.web.servlet.AutoConfigureMockMvc;
import org.springframework.boot.test.context.SpringBootTest;
import org.springframework.test.web.servlet.MockMvc;

/**
 * One server for the whole HTTP suite, on the committed mock bundles: an
 * embedder, a generative model, a reranker, a classifier and a token
 * classifier, all on the mock provider, with no hardware and deterministic
 * outputs.
 *
 * <p>Surefire passes the bundle paths as system properties (see the pom), and
 * the two shorthand keys {@code turbo.bundle} and {@code turbo.generate-bundle}
 * become the models named after their {@code model_id}: {@code mock-embedding}
 * and {@code mock-generative}. The mock bundles carry no {@code tokenizer.json},
 * so the embedding model is given the committed MiniLM tokenizer bundle to
 * serve {@code /api/v1/tokenize} from.
 */
@SpringBootTest(properties = {
        "turbo.max-batch=4",
        "turbo.sessions=1",
        "turbo.generations=1",
        "turbo.models[0].name=rerank",
        "turbo.models[0].bundle=${turbo.rerank-bundle}",
        "turbo.models[0].max-batch=4",
        "turbo.models[0].sessions=1",
        "turbo.models[1].name=classify",
        "turbo.models[1].bundle=${turbo.classify-bundle}",
        "turbo.models[1].max-batch=4",
        "turbo.models[1].sessions=1",
        "turbo.models[2].name=ner",
        "turbo.models[2].bundle=${turbo.token-classify-bundle}",
        "turbo.models[2].max-batch=4",
        "turbo.models[2].sessions=1",
})
@AutoConfigureMockMvc
abstract class ApiTestBase {
    /** The name the {@code turbo.bundle} shorthand serves the mock embedding bundle under. */
    static final String EMBED_MODEL = "mock-embedding";
    /** The name the {@code turbo.generate-bundle} shorthand serves the mock generative bundle under. */
    static final String GENERATE_MODEL = "mock-generative";

    @Autowired
    protected MockMvc mvc;
}
