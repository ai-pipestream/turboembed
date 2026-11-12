// The little of Node the end-to-end run reads, beside test/node.d.ts.

declare const process: { env: Record<string, string | undefined> };
declare const Buffer: { from(data: string | Uint8Array): Uint8Array };
