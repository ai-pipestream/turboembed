// SPDX-License-Identifier: Apache-2.0
//
// End-to-end tests for demo/java-web-spring. By default the suite starts the
// app itself on the committed mock bundle (deterministic, no hardware) and
// tears it down afterwards. Set E2E_BASE_URL to test an app that is already
// running (a real model, for example); the suite then starts nothing.
//
// The screenshot project writes into docs/screenshots and so is opt-in:
// SCREENSHOTS=1 (npm run screenshots) adds it, a plain run leaves it out.
import { defineConfig, devices, type Project } from "@playwright/test";
import os from "node:os";
import path from "node:path";

const PORT = Number(process.env.E2E_PORT ?? 8091);
const externalUrl = process.env.E2E_BASE_URL;
const baseURL = externalUrl ?? `http://127.0.0.1:${PORT}`;
const SHOTS = /screenshots\.spec\.ts/;

// The committed mock bundles: deterministic 8-dim vectors, deterministic
// token-overlap rerank scores and deterministic "tokNNN" generation, so the
// suite needs no accelerator and no downloads. The mock bundles declare a
// "mock" tokenizer with no tokenizer.json, so the embedding model is given the
// committed MiniLM tokenizer bundle for /api/v1/tokenize.
const repoRoot = path.resolve(__dirname, "../../..");
const bundle = (name: string) => path.join(repoRoot, "testdata/bundles", name);
const modelArgs = [
    `--turbo.generate-bundle=${bundle("mock/generative")}`,
    `--turbo.tokenizer-bundle=${bundle("minilm-tokenizer")}`,
    `--turbo.models[0].name=rerank`,
    `--turbo.models[0].bundle=${bundle("mock/reranker")}`,
    `--turbo.models[1].name=classify`,
    `--turbo.models[1].bundle=${bundle("mock/classifier")}`,
    `--turbo.models[2].name=ner`,
    `--turbo.models[2].bundle=${bundle("mock/token-classifier")}`,
].join(" ");

// run.sh calls java as "$JAVA_HOME/bin/java" when JAVA_HOME is set, and needs
// Maven and java on PATH only when it builds (TURBO_WEB_SKIP_BUILD is not 1).
const javaHome = process.env.JAVA_HOME ?? path.join(os.homedir(), ".sdkman/candidates/java/25.0.3-tem");

const screenshotProject: Project = {
    name: "screenshots",
    testMatch: SHOTS,
    use: {
        ...devices["Desktop Chrome"],
        viewport: { width: 1200, height: 720 },
        deviceScaleFactor: 1,
        colorScheme: "light",
    },
};

export default defineConfig({
    testDir: "tests",
    fullyParallel: false,
    forbidOnly: !!process.env.CI,
    retries: 0,
    workers: 1,
    reporter: process.env.CI ? [["list"], ["html", { open: "never" }]] : [["list"]],
    use: {
        baseURL,
        trace: "retain-on-failure",
        screenshot: "only-on-failure",
    },
    projects: [
        { name: "chromium", testIgnore: SHOTS, use: { ...devices["Desktop Chrome"] } },
        ...(process.env.SCREENSHOTS === "1" ? [screenshotProject] : []),
    ],
    webServer: externalUrl
        ? undefined
        : {
              command: `../run.sh --server.port=${PORT} ${modelArgs}`,
              url: `http://127.0.0.1:${PORT}/api/v1/health`,
              cwd: __dirname,
              env: {
                  // The jar must already be built: run demo/java-web-spring/run.sh
                  // once, or set TURBO_WEB_SKIP_BUILD=0 to let run.sh build it here.
                  TURBO_WEB_SKIP_BUILD: process.env.TURBO_WEB_SKIP_BUILD ?? "1",
                  JAVA_HOME: javaHome,
                  PATH: `${javaHome}/bin:${process.env.PATH ?? ""}`,
              },
              reuseExistingServer: !process.env.CI,
              stdout: "pipe",
              stderr: "pipe",
              timeout: 180_000,
          },
});
