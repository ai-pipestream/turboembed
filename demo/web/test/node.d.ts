// The little of Node the unit tests use, so the page needs no @types/node.

declare module "node:test" {
  export function test(name: string, fn: () => void | Promise<void>): void;
}

declare module "node:assert/strict" {
  const assert: {
    equal(actual: unknown, expected: unknown, message?: string): void;
    deepEqual(actual: unknown, expected: unknown, message?: string): void;
    throws(fn: () => unknown, expected?: unknown): void;
  };
  export default assert;
}

declare module "node:crypto" {
  export function createHash(algorithm: string): { update(data: Uint8Array): { digest(encoding: "hex"): string } };
}
