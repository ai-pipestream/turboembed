// SPDX-License-Identifier: Apache-2.0
package ai.pipestream.turbo.web;

import io.swagger.v3.oas.models.OpenAPI;
import io.swagger.v3.oas.models.info.Info;
import io.swagger.v3.oas.models.info.License;
import io.swagger.v3.oas.models.servers.Server;
import java.util.List;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;

/**
 * The OpenAPI document served at {@code /v3/api-docs}, which
 * {@code /swagger-ui.html} renders.
 */
@Configuration
public class OpenApiConfig {
    @Bean
    OpenAPI turboOpenApi() {
        return new OpenAPI()
                .info(new Info()
                        .title("Turbo inference server")
                        .version("0.1.0")
                        .description("""
                                A demonstration inference server over libturbo through its JDK 25 FFM binding. \
                                Two surfaces cover the same models: /api/v1 is libturbo's own, with per-call \
                                options that are honored exactly or refused by name, the device survey, the \
                                capability matrix and per-phase timings; /v2 is the KServe Open Inference \
                                Protocol version 2 HTTP/REST binding, for clients that already speak it. \
                                A refused option always comes back with its TURBO_E_* status name and the \
                                1-based index of the field the library named. Nothing falls back and nothing \
                                is clamped.""")
                        .license(new License().name("Apache-2.0").url("https://www.apache.org/licenses/LICENSE-2.0")))
                .servers(List.of(new Server().url("/").description("this server")));
    }
}
