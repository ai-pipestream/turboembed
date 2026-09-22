// SPDX-License-Identifier: Apache-2.0
package ai.pipestream.turbo.web;

import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;

/** Entry point: {@code java --enable-native-access=ALL-UNNAMED -Dturbo.library=... -jar turbo-demo-web.jar --turbo.bundle=<dir>}. */
@SpringBootApplication
public class TurboWebApplication {
    public static void main(String[] args) {
        SpringApplication.run(TurboWebApplication.class, args);
    }
}
