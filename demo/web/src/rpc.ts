// Every call the page makes goes through here: generated Connect-ES
// clients over one binary Connect transport to the demo, POST only.

import { Code, ConnectError, createClient, type Client, type Interceptor } from "@connectrpc/connect";
import { createConnectTransport } from "@connectrpc/connect-web";
import type { DescService } from "@bufbuild/protobuf";
import { explain } from "./errors.js";
import { GRPCInferenceService } from "./gen/open_inference_grpc_pb.js";
import { AnalysisService } from "./gen/turbo_analysis_pb.js";
import { DemoService } from "./gen/turbo_demo_pb.js";
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
  return typeof location === "undefined" ? "" : location.origin;
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
/** The demo itself, not the server behind it. */
export const demo = client(DemoService);

/** What went wrong as one plain sentence; `explain` also gives the protocol's details. */
export function describe(e: unknown): string {
  return explain(e).text;
}

/** Whether `e` is a refusal by the demo or server rather than a failure. */
export function refused(e: unknown): boolean {
  const c = ConnectError.from(e).code;
  return c === Code.PermissionDenied || c === Code.Unauthenticated || c === Code.Unimplemented;
}

export { cancelled, explain } from "./errors.js";
