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

/**
 * A loopback port nothing listens on now. Another process can take it
 * before the server binds it, so a start that finds it taken is retried
 * (`inUse`).
 */
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

/** Whether `e` is a start that failed because its port was taken. */
function inUse(e: unknown): boolean {
  return e instanceof Error && e.message.includes("Address already in use");
}

const pause = (ms: number) => new Promise<void>((resolve) => setTimeout(resolve, ms));

/** Ends `child` and waits until it has exited, so its port is free again. */
function ended(child: ChildProcess | null): Promise<void> {
  if (!child || child.exitCode !== null || child.signalCode !== null) return Promise.resolve();
  return new Promise((resolve) => {
    child.once("exit", () => resolve());
    child.kill();
  });
}

export interface StackOptions {
  /** `--model` values; the default serves the tiny BERT bundle. */
  models?: string[];
  /** Starts the server with `--admin`. */
  admin?: boolean;
  /** Method paths the demo forwards with `--allow`. */
  allow?: string[];
  /** The bearer token every call to the demo must carry (`--token-file`). */
  token?: string;
}

/** A server and the demo in front of it. */
export class Stack {
  url = "";
  private server: ChildProcess | null = null;
  private demo: ChildProcess | null = null;
  private serverArgs: string[] = [];
  private work = "";

  /** Starts a server and demo; a port taken between pick and bind is picked again, up to three times. */
  static async start(o: StackOptions = {}): Promise<Stack> {
    if (!BIN) throw new Error("TURBO_E2E_BIN names no directory: run this through scripts/e2e-web.sh");
    for (let attempt = 1; ; attempt++) {
      try {
        return await Stack.once(o);
      } catch (e) {
        if (attempt >= 3 || !inUse(e)) throw e;
      }
    }
  }

  private static async once(o: StackOptions): Promise<Stack> {
    const s = new Stack();
    s.work = mkdtempSync(`${tmpdir()}/turbo-e2e-`);
    for (const d of ["vocabulary", "catalog", "training"]) mkdirSync(`${s.work}/${d}`);
    const [serverPort, demoPort] = [await freePort(), await freePort()];
    const models = o.models ?? [`bundle=testdata/tiny-bert-bundle,device=${DEVICE},sessions=1`];
    s.serverArgs = [
      "--listen", `127.0.0.1:${serverPort}`,
      "--vocabulary-dir", `${s.work}/vocabulary`, "--catalog-dir", `${s.work}/catalog`, "--training-dir", `${s.work}/training`,
      ...models.flatMap((m) => ["--model", m]),
      ...(o.admin ? ["--admin"] : []),
    ];
    try {
      if (serverPort === demoPort) throw new Error("Address already in use: the two picked ports are the same");
      s.server = await started("turbo-kserve", s.serverArgs, "ready");
      const demoArgs = ["--listen", `127.0.0.1:${demoPort}`, "--server", `http://127.0.0.1:${serverPort}`, ...(o.allow ?? []).flatMap((m) => ["--allow", m])];
      if (o.token) {
        writeFileSync(`${s.work}/token`, o.token, { mode: 0o600 });
        demoArgs.push("--token-file", `${s.work}/token`);
      }
      s.demo = await started("turbo-kserve-demo", demoArgs, "open http");
    } catch (e) {
      await s.close();
      throw e;
    }
    s.url = `http://127.0.0.1:${demoPort}`;
    return s;
  }

  /**
   * Starts the server again on the same port, with the same flags. The
   * port is the test's own, so one still held is waited for, up to five
   * tries a quarter second apart.
   */
  async startServer(): Promise<void> {
    for (let attempt = 1; ; attempt++) {
      try {
        this.server = await started("turbo-kserve", this.serverArgs, "ready");
        return;
      } catch (e) {
        if (attempt >= 5 || !inUse(e)) throw e;
        await pause(250);
      }
    }
  }

  /** Stops the server; the demo keeps running in front of nothing. */
  async stopServer(): Promise<void> {
    await ended(this.server);
    this.server = null;
  }

  async close(): Promise<void> {
    await Promise.all([ended(this.demo), ended(this.server)]);
    if (this.work) rmSync(this.work, { recursive: true, force: true });
  }
}
