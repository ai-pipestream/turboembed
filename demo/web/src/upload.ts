// The PUT half of the page: an upload body sent to /upload plus the
// client-streaming method's path, its binary reply decoded with the
// generated schema, and an error read from the headers into the same
// ConnectError the calls give. This is the page's only request made
// outside the Connect clients. It is an XMLHttpRequest because that is
// what reports how much of a body has been sent; the body and the reply
// are binary as everywhere else.

import { fromBinary, type DescMessage, type DescMethod, type MessageShape } from "@bufbuild/protobuf";
import { Code, ConnectError } from "@connectrpc/connect";
import { base, token } from "./rpc.js";

/** Called as the body goes out: bytes sent so far, of how many. */
export type Sent = (sent: number, total: number) => void;

export function put<O extends DescMessage>(
  method: DescMethod & { output: O },
  bodyBlob: Blob,
  signal?: AbortSignal,
  sent?: Sent,
): Promise<MessageShape<O>> {
  return new Promise((resolve, reject) => {
    const stopped = (): void => reject(new ConnectError("the upload was stopped", Code.Canceled));
    if (signal?.aborted) return stopped();
    const x = new XMLHttpRequest();
    x.open("PUT", `${base()}/upload/${method.parent.typeName}/${method.name}`);
    x.responseType = "arraybuffer";
    x.setRequestHeader("content-type", "application/octet-stream");
    const t = token();
    if (t) x.setRequestHeader("authorization", `Bearer ${t}`);
    const total = bodyBlob.size;
    x.upload.addEventListener("progress", (e) => sent?.(e.loaded, total));
    x.upload.addEventListener("load", () => sent?.(total, total));
    const abort = (): void => x.abort();
    signal?.addEventListener("abort", abort, { once: true });
    x.addEventListener("loadend", () => signal?.removeEventListener("abort", abort));
    x.addEventListener("abort", stopped);
    x.addEventListener("error", () => reject(new ConnectError("the demo did not answer the upload", Code.Unavailable)));
    x.addEventListener("load", () => {
      if (x.status !== 200) return reject(refusal(x.status, (name) => x.getResponseHeader(name)));
      try {
        resolve(fromBinary(method.output, new Uint8Array(x.response as ArrayBuffer)) as MessageShape<O>);
      } catch (e) {
        reject(e);
      }
    });
    sent?.(0, total);
    x.send(bodyBlob);
  });
}

/** The ConnectError a refused upload stands for. */
function refusal(http: number, header: (name: string) => string | null): ConnectError {
  const status = Number(header("grpc-status") ?? "");
  const code = status > 0 && status <= 16 ? (status as Code) : Code.Unknown;
  let message = header("grpc-message") ?? `HTTP ${http}`;
  try {
    message = decodeURIComponent(message);
  } catch {
    // Left as sent.
  }
  if (http === 415) message = "the demo wants application/octet-stream";
  if (http === 413) message = `${message} (HTTP 413: over the 2 GiB cap)`;
  const meta = new Headers();
  for (const k of ["turbo-code", "turbo-field"]) {
    const v = header(k);
    if (v !== null) meta.set(k, v);
  }
  return new ConnectError(message, code, meta);
}
