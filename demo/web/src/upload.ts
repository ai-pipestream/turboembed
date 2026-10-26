// The PUT half of the page: an upload body sent to /upload plus the
// client-streaming method's path, its binary reply decoded with the
// generated schema, and an error read from the headers into the same
// ConnectError the calls give. This is the page's only fetch.

import { fromBinary, type DescMessage, type DescMethod, type MessageShape } from "@bufbuild/protobuf";
import { Code, ConnectError } from "@connectrpc/connect";
import { base, token } from "./rpc.js";

export async function put<O extends DescMessage>(
  method: DescMethod & { output: O },
  bodyBlob: Blob,
  signal?: AbortSignal,
): Promise<MessageShape<O>> {
  const headers = new Headers({ "content-type": "application/octet-stream" });
  const t = token();
  if (t) headers.set("authorization", `Bearer ${t}`);
  const url = `${base()}/upload/${method.parent.typeName}/${method.name}`;
  const r = await fetch(url, { method: "PUT", headers, body: bodyBlob, signal });
  if (r.status === 200) return fromBinary(method.output, new Uint8Array(await r.arrayBuffer())) as MessageShape<O>;
  throw refusal(r);
}

/** The ConnectError a refused upload stands for. */
function refusal(r: Response): ConnectError {
  const status = Number(r.headers.get("grpc-status") ?? "");
  const code = status > 0 && status <= 16 ? (status as Code) : Code.Unknown;
  let message = r.headers.get("grpc-message") ?? `HTTP ${r.status}`;
  try {
    message = decodeURIComponent(message);
  } catch {
    // Left as sent.
  }
  if (r.status === 415) message = "the demo wants application/octet-stream";
  if (r.status === 413) message = `${message} (HTTP 413: over the 2 GiB cap)`;
  const meta = new Headers();
  for (const k of ["turbo-code", "turbo-field"]) {
    const v = r.headers.get(k);
    if (v !== null) meta.set(k, v);
  }
  return new ConnectError(message, code, meta);
}
