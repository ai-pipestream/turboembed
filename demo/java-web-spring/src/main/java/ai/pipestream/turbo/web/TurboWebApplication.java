// SPDX-License-Identifier: Apache-2.0
package ai.pipestream.turbo.web;

import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;
import org.springframework.boot.context.properties.EnableConfigurationProperties;

/**
 * Entry point.
 *
 * <pre>
 * java --enable-native-access=ALL-UNNAMED -Dturbo.library=&lt;libturbo.so&gt; \
 *     -jar turbo-demo-web.jar --turbo.bundle=&lt;bundle dir&gt;
 * </pre>
 */
@SpringBootApplication
@EnableConfigurationProperties(TurboProperties.class)
public class TurboWebApplication {
    public static void main(String[] args) {
        SpringApplication.run(TurboWebApplication.class, args);
    }
}
