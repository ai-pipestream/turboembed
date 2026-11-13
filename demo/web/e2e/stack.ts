// A turbo-kserve and a turbo-kserve-demo of a test's own, from the
// binaries scripts/e2e-web.sh built (TURBO_E2E_BIN), for tests that stop
// and start the server under an open page or need a demo started with
// other flags than the shared one. Run from the repository root, as the
// script does, so the test bundle paths resolve.

import { spawn, type ChildProcess } from "node:child_process";
import { mkdtempSync, mkdirSync, rmSync, writeFileSync } from "node:fs";
import { createServer } from "node:net";
import { tmpdir } from "node:os";

const BIN = process.env.TURBO_E2E_BIN ?? "";
const DEVICE = process.env.TURBO_E2E_DEVICE ?? "0";

/** A loopback port nothing listens on now. */
function freePort(): Promise<number> {
  return new Promise((resolve, reject) => {
    const s = createServer();
    s.once("error", reject);
    s.listen(0, "127.0.0.1", () => {
      const port = (s.address() as { port: number }).port;
      s.close(() => resolve(port));
    });
  });
}

/** Starts `exe` and resolves once its output contains `word`; rejects if it exits first or takes over 60 s. */
function started(exe: string, args: string[], word: string): Promise<ChildProcess> {
  return new Promise((resolve, reject) => {
    const child = spawn(`${BIN}/${exe}`, args, { stdio: ["ignore", "pipe", "pipe"] });
    let out = "";
    const timer = setTimeout(() => (child.kill(), reject(new Error(`${exe} was not ready in 60 s:\n${out}`))), 60_000);
    const read = (d: Uint8Array) => {
      out += new TextDecoder().decode(d);
      if (out.includes(word)) {
        clearTimeout(timer);
        resolve(child);
      }
    };
    child.stdout?.on("data", read);
    child.stderr?.on("data", read);
    child.once("exit", (code) => {
      clearTimeout(timer);
      reject(new Error(`${exe} stopped (${code}) before it was ready:\n${out}`));
    });
  });
}

/** Ends `child` and waits until it has exited, so its port is free again. */
function ended(child: ChildProcess | null): Promise<void> {
  if (!child || child.exitCode !== null || child.signalCode !== null) return Promise.resolve();
  return new Promise((resolve) => {
    child.once("exit", () => resolve());
    child.kill();
  });
}

export interface DemoOptions {
  /** The user this demo acts for (`--user`). */
  user?: string;
  /** Method paths the demo forwards with `--allow`. */
  allow?: string[];
  /** The bearer token every call to the demo must carry (`--token-file`). */
  token?: string;
}

export interface StackOptions extends DemoOptions {
  /** `--model` values; the default serves the tiny BERT bundle. */
  models?: string[];
  /** Starts the server with `--admin`. */
  admin?: boolean;
  /** Starts the server with `--zoo`. */
  zoo?: boolean;
  /** `--catalog-index` path. */
  catalogIndex?: string;
  /** `--catalog-store` URL. */
  catalogStore?: string;
}

/** A server and one or more demos in front of it. */
export class Stack {
  url = "";
  private server: ChildProcess | null = null;
  private demos: ChildProcess[] = [];
  private serverPort = 0;
  private serverArgs: string[] = [];
  private work = "";
  private tokenIndex = 0;

  static async start(o: StackOptions = {}): Promise<Stack> {
    if (!BIN) throw new Error("TURBO_E2E_BIN names no directory: run this through scripts/e2e-web.sh");
    const s = new Stack();
    s.work = mkdtempSync(`${tmpdir()}/turbo-e2e-`);
    for (const d of ["vocabulary", "catalog", "training"]) mkdirSync(`${s.work}/${d}`);
    s.serverPort = await freePort();
    const models = o.models ?? [`bundle=testdata/tiny-bert-bundle,device=${DEVICE},sessions=1`];
    s.serverArgs = [
      "--listen", `127.0.0.1:${s.serverPort}`,
      "--vocabulary-dir", `${s.work}/vocabulary`, "--catalog-dir", `${s.work}/catalog`, "--training-dir", `${s.work}/training`,
      ...models.flatMap((m) => ["--model", m]),
      ...(o.admin ? ["--admin"] : []),
      ...(o.zoo ? ["--zoo"] : []),
      ...(o.catalogIndex ? ["--catalog-index", o.catalogIndex] : []),
      ...(o.catalogStore ? ["--catalog-store", o.catalogStore] : []),
    ];
    try {
      await s.startServer();
      s.url = await s.addDemo(o);
    } catch (e) {
      await s.close();
      throw e;
    }
    return s;
  }

  /** Starts an additional demo proxy in front of the same server. */
  async addDemo(o: DemoOptions = {}): Promise<string> {
    const demoPort = await freePort();
    const demoArgs = [
      "--listen", `127.0.0.1:${demoPort}`,
      "--server", `http://127.0.0.1:${this.serverPort}`,
      ...(o.allow ?? []).flatMap((m) => ["--allow", m]),
      ...(o.user ? ["--user", o.user] : []),
    ];
    if (o.token) {
      const tokenPath = `${this.work}/token-${this.tokenIndex++}`;
      writeFileSync(tokenPath, o.token, { mode: 0o600 });
      demoArgs.push("--token-file", tokenPath);
    }
    const demo = await started("turbo-kserve-demo", demoArgs, "open http");
    this.demos.push(demo);
    return `http://127.0.0.1:${demoPort}`;
  }

  /** Starts the server again on the same port, with the same flags. */
  async startServer(): Promise<void> {
    this.server = await started("turbo-kserve", this.serverArgs, "ready");
  }

  /** Stops the server; the demo keeps running in front of nothing. */
  async stopServer(): Promise<void> {
    await ended(this.server);
    this.server = null;
  }

  async close(): Promise<void> {
    await Promise.all([...this.demos.map((d) => ended(d)), ended(this.server)]);
    if (this.work) rmSync(this.work, { recursive: true, force: true });
  }
}
