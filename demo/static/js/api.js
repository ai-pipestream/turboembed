// The page's calls to the demo's JSON routes (docs/demo.md). A refused call
// answers {"error": {"grpc", "message", "turbo_code", "turbo_field"}} with a
// 4xx or 502 status; a streamed route ends its stream with such a line when
// the server fails part way.

export class ApiError extends Error {
  constructor(status, error) {
    const e = error || {};
    const parts = [e.grpc, e.message, e.turbo_code ? `turbo-code ${e.turbo_code}` : null].filter(Boolean);
    super(parts.join(": ") || `HTTP ${status}`);
    this.status = status;
    this.error = e;
  }
}

async function answer(r) {
  const text = await r.text();
  let body = null;
  try {
    body = text ? JSON.parse(text) : {};
  } catch {
    throw new ApiError(r.status, { message: text.slice(0, 300) || `HTTP ${r.status}` });
  }
  if (!r.ok || (body && body.error)) throw new ApiError(r.status, body && body.error);
  return body;
}

export async function getJson(path, signal) {
  return answer(await fetch(path, { headers: { accept: "application/json" }, signal }));
}

export async function postJson(path, body, { method = "POST", signal } = {}) {
  const r = await fetch(path, {
    method,
    headers: { "content-type": "application/json", accept: "application/json" },
    body: JSON.stringify(body ?? {}),
    signal,
  });
  return answer(r);
}

// Calls a streamed route and hands each NDJSON line, parsed, to `onLine`.
// Resolves with the number of lines once the stream ends; rejects with an
// ApiError when the call is refused or the stream ends with an error line.
export async function streamNdjson(path, body, onLine, signal) {
  const r = await fetch(path, {
    method: "POST",
    headers: { "content-type": "application/json", accept: "application/x-ndjson" },
    body: JSON.stringify(body ?? {}),
    signal,
  });
  if (!r.ok || !(r.headers.get("content-type") || "").includes("ndjson")) return answer(r);
  const reader = r.body.getReader();
  const decoder = new TextDecoder();
  let rest = "";
  let count = 0;
  const take = (line) => {
    if (!line.trim()) return;
    const value = JSON.parse(line);
    if (value && value.error) throw new ApiError(502, value.error);
    count += 1;
    onLine(value);
  };
  for (;;) {
    const { value, done } = await reader.read();
    if (done) break;
    rest += decoder.decode(value, { stream: true });
    let nl;
    while ((nl = rest.indexOf("\n")) >= 0) {
      take(rest.slice(0, nl));
      rest = rest.slice(nl + 1);
    }
  }
  take(rest + decoder.decode());
  return count;
}

// Bytes fields that carry content travel as base64 (docs/demo.md).
export function toBase64(bytes) {
  let s = "";
  const step = 0x8000;
  for (let i = 0; i < bytes.length; i += step) s += String.fromCharCode(...bytes.subarray(i, i + step));
  return btoa(s);
}

export function fromBase64(b64) {
  const s = atob(b64 || "");
  const out = new Uint8Array(s.length);
  for (let i = 0; i < s.length; i++) out[i] = s.charCodeAt(i);
  return out;
}

export function utf8(text) {
  return new TextEncoder().encode(text);
}

export function decodeUtf8(bytes) {
  return new TextDecoder().decode(bytes);
}

export async function sha256Hex(bytes) {
  const digest = await crypto.subtle.digest("SHA-256", bytes);
  return [...new Uint8Array(digest)].map((b) => b.toString(16).padStart(2, "0")).join("");
}
