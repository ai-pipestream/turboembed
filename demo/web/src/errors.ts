// What went wrong, for a person: one plain sentence that says what to
// do where that is known, and the protocol's own words (the Connect
// code, the server's turbo-code and turbo-field, the raw message) kept
// apart as details.

import { Code, ConnectError } from "@connectrpc/connect";

/** A failure as the page shows it. `details` is empty when there is no more to say. */
export interface Explained {
  text: string;
  details: string;
}

/** A Connect code as the protocol writes it: `permission_denied`. */
export function codeName(code: Code): string {
  const camel = Code[code] ?? String(code);
  let out = "";
  for (const ch of camel) {
    const lower = ch.toLowerCase();
    if (ch !== lower && out) out += "_";
    out += lower;
  }
  return out;
}

/** Whether `e` is a call the page itself stopped. */
export function cancelled(e: unknown): boolean {
  if (e instanceof Error && e.name === "AbortError") return true;
  return e instanceof ConnectError && e.code === Code.Canceled;
}

// What to do about an empty field, by the name the server gives it.
const EMPTY: Record<string, string> = {
  index_name: "Pick or create an index first.",
  natural_query: "Enter something to search for.",
  query: "Enter a query.",
  text: "Enter some text first.",
  name: "Enter a name.",
  bundle: "Give the bundle's path on the server.",
  collection_name: "Name the collection.",
  teacher: "Pick a teacher model.",
  model: "Pick a model.",
  model_name: "Pick a model.",
  artifact_id: "Pick an artifact.",
  graph: "Analyze a document first.",
  labels: "Enter at least one label.",
};

const EMPTY_ENDS = [" must not be empty", " is required", " is empty", " must be set"];

/** The server's message without its `TURBO_E_...: ` prefix. */
export function unprefixed(message: string): string {
  const at = message.indexOf(": ");
  if (at < 0 || !message.startsWith("TURBO_E_")) return message;
  for (const ch of message.slice(0, at)) {
    const ok = ch === "_" || (ch >= "A" && ch <= "Z") || (ch >= "0" && ch <= "9");
    if (!ok) return message;
  }
  return message.slice(at + 2);
}

/** `message` as a sentence: a capital first letter and a full stop. */
export function sentence(message: string): string {
  const m = message.trim();
  if (!m) return "";
  const first = String.fromCodePoint(m.codePointAt(0)!);
  const rest = m.slice(first.length);
  const last = m[m.length - 1];
  return first.toUpperCase() + rest + (last === "." || last === "!" || last === "?" ? "" : ".");
}

/** What to do when the message says a named field is empty, if the field is known. */
function emptyField(message: string): string {
  for (const end of EMPTY_ENDS) {
    if (!message.endsWith(end)) continue;
    const field = message.slice(0, message.length - end.length);
    return EMPTY[field.slice(field.lastIndexOf(".") + 1)] ?? "";
  }
  return "";
}

function plain(code: Code, message: string): string {
  switch (code) {
    case Code.PermissionDenied: {
      const at = message.indexOf(" is not on the demo's allowlist");
      if (at > 0) return `The demo does not forward this write. Start it with --allow ${message.slice(message.startsWith("/") ? 1 : 0, at)}`;
      if (message.includes("without --user")) return "This acts for a user. Start the demo with --user NAME.";
      return message ? `Not allowed: ${sentence(message)}` : "Not allowed.";
    }
    case Code.Unauthenticated:
      return "The demo wants its bearer token. Enter it at the top of the page.";
    case Code.Unimplemented:
      return message ? `Not served: ${sentence(message)}` : "The server does not serve this call.";
    case Code.Unavailable:
      return "The server is not reachable through the demo.";
    case Code.DeadlineExceeded:
      return "The call took too long and was stopped.";
    case Code.InvalidArgument:
    case Code.FailedPrecondition:
    case Code.NotFound:
    case Code.AlreadyExists:
    case Code.OutOfRange:
    case Code.ResourceExhausted:
      return emptyField(message) || sentence(message) || `The server refused the call (${codeName(code)}).`;
    default:
      return sentence(message) || `The call failed (${codeName(code)}).`;
  }
}

/**
 * `e` for a person. An error the page raised itself (a plain Error) is
 * its own message with no details; a refused or failed call leads with
 * a plain sentence and keeps the protocol's words as details.
 */
export function explain(e: unknown): Explained {
  if (cancelled(e)) return { text: "Cancelled.", details: "" };
  if (!(e instanceof ConnectError)) {
    const m = e instanceof Error ? e.message : String(e);
    // A fetch that never got an answer: the demo itself is not there.
    if (e instanceof TypeError) return { text: "The demo did not answer.", details: m };
    return { text: m, details: "" };
  }
  const message = unprefixed(e.rawMessage);
  const parts = [codeName(e.code)];
  const tc = e.metadata.get("turbo-code");
  const tf = e.metadata.get("turbo-field");
  if (tc) parts.push(`turbo-code ${tc}`);
  if (tf && tf !== "0") parts.push(`field ${tf}`);
  return { text: plain(e.code, message), details: `${parts.join(", ")}: ${e.rawMessage}` };
}
