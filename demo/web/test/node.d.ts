// The little of Node the unit and browser tests use, so the page needs no @types/node.

declare module "node:test" {
  export function test(name: string, fn: () => void | Promise<void>): void;
}

declare module "node:assert/strict" {
  const assert: {
    equal(actual: unknown, expected: unknown, message?: string): void;
    deepEqual(actual: unknown, expected: unknown, message?: string): void;
    notEqual(actual: unknown, expected: unknown, message?: string): void;
    match(actual: string, expected: RegExp, message?: string): void;
    throws(fn: () => unknown, expected?: unknown): void;
  };
  export default assert;
}

declare module "node:crypto" {
  export function createHash(algorithm: string): { update(data: Uint8Array): { digest(encoding: "hex"): string } };
}

declare module "node:fs" {
  export function readFileSync(path: string): Uint8Array;
}

declare module "node:fs" {
  export function mkdtempSync(prefix: string): string;
  export function mkdirSync(path: string): void;
  export function rmSync(path: string, options: { recursive: boolean; force: boolean }): void;
}

declare module "node:os" {
  export function tmpdir(): string;
}

declare module "node:net" {
  export function createServer(): {
    once(event: "error", l: (e: Error) => void): void;
    listen(port: number, host: string, l: () => void): void;
    address(): unknown;
    close(l?: () => void): void;
  };
}

declare module "node:child_process" {
  export interface ChildProcess {
    stdout: { on(event: "data", l: (d: Uint8Array) => void): void } | null;
    stderr: { on(event: "data", l: (d: Uint8Array) => void): void } | null;
    exitCode: number | null;
    signalCode: string | null;
    once(event: "exit", l: (code: number | null) => void): void;
    kill(): boolean;
  }
  export function spawn(command: string, args: string[], options: { stdio: string[] }): ChildProcess;
}
