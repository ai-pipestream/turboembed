// Licence terms on the Models tab: what each artifact's terms are, the
// "I have read and accept" control, the terms the demo's user accepted,
// and revoke. Every call acts for the user the demo was started with
// (--user); the page never names one. Without a user, or on a server
// without --admin or --zoo, the page is read-only.

import type { CatalogEntry, LicenceTerms, ListLicenceAcceptancesResponse } from "../gen/turbo_training_pb.js";
import { el, table, type Child } from "../dom.js";
import { catalog } from "../rpc.js";
import { Sha256 } from "../sha256.js";

/** Who the page acts for and whether it may change anything. */
export interface Who {
  user: string;
  canChange: boolean;
  why: string;
}

export function who(r: ListLicenceAcceptancesResponse): Who {
  if (!r.user) return { user: "", canChange: false, why: "read-only: the demo was started without --user" };
  if (!r.admin) return { user: r.user, canChange: false, why: "read-only: the server was started without --admin or --zoo" };
  return { user: r.user, canChange: true, why: "" };
}

/** What a licence class means for the user, in plain words. The legal text decides. */
const NOTICES: Record<string, string> = {
  CLASS_SHARE_ALIKE: "Share-alike: you may use it, and what you share that is built from it carries the same licence. This project does not redistribute it.",
  CLASS_LGPL: "LGPL: you may use it; changes to it that you distribute are released under the LGPL.",
  CLASS_GPL: "GPL: software you distribute that includes it is released under the GPL.",
  CLASS_LOCAL_ONLY: "Local only: restricted or non-commercial terms. It is fetched only from its publisher, stays on this machine, and no model trained on it is shipped. Not for commercial use.",
};

/** The SHA-256 of the terms text as the page shows it. */
export function textSha256(text: string): string {
  return new Sha256().update(new TextEncoder().encode(text)).hex();
}

/** A terms address as a link when it is an https URL, else as text. */
function address(text: string): Child {
  let secure = false;
  try {
    secure = new URL(text).protocol === "https:";
  } catch {
    secure = false;
  }
  if (!secure) return el("code", { text });
  return el("a", { href: text, target: "_blank", rel: "noopener noreferrer", text });
}

/** One set of terms an artifact needs, with what it means, and the accept control when allowed. */
export function termsBlock(e: CatalogEntry, t: LicenceTerms, w: Who, accepted: (t: LicenceTerms) => Promise<void>): HTMLElement {
  const sources = e.trainedOn.map((s) => `${s.name} (${s.licence})`).join(", ");
  const facts = el(
    "dl",
    {},
    el("dt", { text: "Licence" }),
    el("dd", { text: `${t.licence}, ${e.licenceClass}` }),
    el("dt", { text: "Terms" }),
    el("dd", {}, address(t.text)),
    e.attribution ? [el("dt", { text: "Attribution" }), el("dd", { text: e.attribution })] : null,
    sources ? [el("dt", { text: "Trained on" }), el("dd", { text: sources })] : null,
  );
  const notice = el("p", { class: "muted", text: NOTICES[e.licenceClass] ?? "" });
  const body: Child[] = [facts, notice];
  if (w.canChange) {
    const read = el("input", { type: "checkbox" });
    const button = el("button", { type: "button", text: `Accept ${t.licence}` });
    button.disabled = true;
    read.addEventListener("change", () => (button.disabled = !read.checked));
    button.addEventListener("click", () => void accepted(t));
    body.push(el("label", {}, read, ` I have read the ${t.licence} terms above and accept them`), " ", button);
  } else {
    body.push(el("p", { class: "muted", text: w.why }));
  }
  return el("details", {}, el("summary", { text: `${t.licence} terms to accept` }), body);
}

/** Record the acceptance of `t` as shown on `e`'s page. */
export async function accept(e: CatalogEntry, t: LicenceTerms): Promise<string> {
  const r = await catalog.recordLicenceAcceptance({
    licence: t.licence,
    terms: t.licence.startsWith("LicenseRef-") ? t.text : "",
    artifactId: e.id,
    textSha256: textSha256(t.text),
  });
  return `${r.user} accepted ${r.accepted?.licence ?? t.licence} at ${when(r.acceptedAt)}.`;
}

function when(seconds: bigint): string {
  return new Date(Number(seconds) * 1000).toLocaleString();
}

/** The terms the user accepted, each with a revoke control when allowed. */
export function acceptedTable(r: ListLicenceAcceptancesResponse, w: Who, revoked: (t: LicenceTerms) => Promise<void>): HTMLElement {
  if (!w.user) return el("p", { class: "muted", text: w.why });
  if (!r.accepted.length) return el("p", { class: "muted", text: `${w.user} has accepted no terms.` });
  return table(
    ["licence", "terms", "accepted", "on the page of", ""],
    r.accepted.map((a) => {
      const t = a.terms;
      let control: Child = "";
      if (t && w.canChange) {
        const b = el("button", { type: "button", text: "Revoke" });
        b.addEventListener("click", () => void revoked(t));
        control = b;
      }
      return [t?.licence ?? "", t ? address(t.text) : "", when(a.acceptedAt), a.artifactId, control];
    }),
  );
}

/** Withdraw the user's acceptance of `t`. */
export async function revoke(t: LicenceTerms): Promise<string> {
  const r = await catalog.revokeLicenceAcceptance({ licence: t.licence, terms: t.licence.startsWith("LicenseRef-") ? t.text : "" });
  return r.wasAccepted ? `${r.user} revoked ${t.licence}.` : `${r.user} had not accepted ${t.licence}.`;
}
