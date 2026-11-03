// Every call the page makes goes through here: generated Connect-ES
// clients over one binary Connect transport to the demo, POST only.

import { Code, ConnectError, createClient, type Client, type Interceptor } from "@connectrpc/connect";
import { createConnectTransport } from "@connectrpc/connect-web";
import type { DescService } from "@bufbuild/protobuf";
import { GRPCInferenceService } from "./gen/open_inference_grpc_pb.js";
import { AnalysisService } from "./gen/turbo_analysis_pb.js";
import { TurboSearchService } from "./gen/turbo_search_pb.js";
import { TurboCatalogService, TurboTrainingService } from "./gen/turbo_training_pb.js";
import { TurboVocabularyService } from "./gen/turbo_vocabulary_pb.js";

const TOKEN_KEY = "turbo-demo-token";

/** The bearer token the demo was started with, if any, kept for this tab only. */
export function token(): string {
  return sessionStorage.getItem(TOKEN_KEY) ?? "";
}

export function setToken(t: string): void {
  if (t) sessionStorage.setItem(TOKEN_KEY, t);
  else sessionStorage.removeItem(TOKEN_KEY);
}

const bearer: Interceptor = (next) => async (req) => {
  const t = token();
  if (t) req.header.set("authorization", `Bearer ${t}`);
  return next(req);
};

/** The page's origin: the demo serves the page and forwards its calls. */
export function base(): string {
  return location.origin;
}

const transport = createConnectTransport({
  baseUrl: base(),
  useBinaryFormat: true,
  useHttpGet: false,
  interceptors: [bearer],
});

function client<T extends DescService>(service: T): Client<T> {
  return createClient(service, transport);
}

export const inference = client(GRPCInferenceService);
export const analysis = client(AnalysisService);
export const search = client(TurboSearchService);
export const vocabulary = client(TurboVocabularyService);
export const catalog = client(TurboCatalogService);
export const training = client(TurboTrainingService);

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

/** What went wrong, from the code, the server's turbo-code and turbo-field, and the message. */
export function describe(e: unknown): string {
  if (e instanceof Error && e.name === "AbortError") return "cancelled";
  const c = ConnectError.from(e);
  if (c.code === Code.Canceled) return "cancelled";
  const parts = [codeName(c.code)];
  const tc = c.metadata.get("turbo-code");
  const tf = c.metadata.get("turbo-field");
  if (tc) parts.push(`turbo-code ${tc}`);
  if (tf && tf !== "0") parts.push(`field ${tf}`);
  const why = refusedWhy(c.code);
  return `${parts.join(", ")}: ${c.rawMessage}${why ? ` (${why})` : ""}`;
}

/** Why a refusal happened, when it is the demo's gate or allowlist. */
function refusedWhy(code: Code): string {
  if (code === Code.PermissionDenied) return "not allowed: start the demo with --allow for this method";
  if (code === Code.Unauthenticated) return "the demo wants its token: set it in the header";
  if (code === Code.Unimplemented) return "not served";
  return "";
}

/** Whether `e` is a refusal by the demo or server rather than a failure. */
export function refused(e: unknown): boolean {
  const c = ConnectError.from(e).code;
  return c === Code.PermissionDenied || c === Code.Unauthenticated || c === Code.Unimplemented;
}
