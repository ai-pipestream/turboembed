// What the Models tab says about a model source, and what a redirect
// back from Google's consent page brought, kept apart from the calls so
// the unit tests run under Node.

import { SourceSignIn, type ListModelSourcesResponse } from "./gen/turbo_training_pb.js";

/** How a source signs in, in plain words. */
export function signInText(s: SourceSignIn): string {
  if (s === SourceSignIn.GOOGLE) return "Google sign-in";
  if (s === SourceSignIn.API_KEY) return "API key";
  if (s === SourceSignIn.ENVIRONMENT) return "the server's own";
  return "";
}

/** Why the page may not change the sources, or "" when it may. */
export function readOnly(r: ListModelSourcesResponse): string {
  if (!r.user) return "read-only: the demo was started without --user";
  if (!r.admin) return "read-only: the server was started without --admin or --zoo";
  return "";
}

/** The `code` and `state` Google's redirect brought, if this load is one. */
export function redirected(search: string): { code: string; state: string; error: string } | null {
  const q = new URLSearchParams(search);
  const state = q.get("state") ?? "";
  if (!state) return null;
  return { code: q.get("code") ?? "", state, error: q.get("error") ?? "" };
}
