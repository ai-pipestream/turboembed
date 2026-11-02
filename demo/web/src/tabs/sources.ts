// Model sources on the Models tab: the stores an install for the demo's
// user reads before the catalog's own, added by address or by signing in
// to Google Drive, and removed. Every call acts for the user the demo was
// started with (--user); the page never names one. Google sends the user
// back to this page with a code and a state in the address, which the
// page hands to the server and then clears.

import type { ListModelSourcesResponse, ModelSource } from "../gen/turbo_training_pb.js";
import { $, el, input, say, table } from "../dom.js";
import { catalog, describe } from "../rpc.js";
import { readOnly, redirected, signInText } from "../sourcetext.js";

const note = () => $("sources-note");

function sourcesTable(r: ListModelSourcesResponse, changed: () => Promise<void>): HTMLElement {
  const why = readOnly(r);
  if (!r.user) return el("p", { class: "muted", text: why });
  if (!r.sources.length) return el("p", { class: "muted", text: `${r.user} has no sources: installs read the catalog's own.` });
  return table(
    ["name", "address", "signs in with", ""],
    r.sources.map((s: ModelSource) => {
      let control: HTMLElement | string = "";
      if (!why) {
        const b = el("button", { type: "button", text: "Remove" });
        b.addEventListener("click", () => void act(async () => {
          const gone = await catalog.removeModelSource({ name: s.name });
          return gone.wasRegistered ? `Removed ${s.name}.` : `${s.name} was not registered.`;
        }, changed));
        control = b;
      }
      return [s.name, el("code", { text: s.address }), signInText(s.signIn), control];
    }),
  );
}

async function act(change: () => Promise<string>, after: () => Promise<void>): Promise<void> {
  try {
    say(note(), "working", "busy");
    const done = await change();
    await after();
    say(note(), done);
  } catch (e) {
    say(note(), describe(e), "error");
  }
}

/** Reads the user's sources into the card, and shows the controls that apply. */
export async function load(): Promise<void> {
  try {
    const r = await catalog.listModelSources({});
    $("sources-list").replaceChildren(sourcesTable(r, load));
    const why = readOnly(r);
    ($("sources-form") as HTMLFormElement).hidden = why !== "";
    $("sources-drive").hidden = why !== "" || !r.driveSignIn;
    if (why) say(note(), why);
  } catch (e) {
    say(note(), describe(e), "error");
  }
}

function add(ev: Event): void {
  ev.preventDefault();
  const name = input("sources-name").value.trim();
  const address = input("sources-address").value.trim();
  const apiKey = input("sources-key").value.trim();
  void act(async () => {
    const r = await catalog.registerModelSource({ name, address, apiKey });
    input("sources-key").value = "";
    return `Registered ${r.source?.name ?? name}.`;
  }, load);
}

/** Sends the user to Google's consent page for a Drive folder. */
function connect(): void {
  const name = input("sources-drive-name").value.trim() || "Google Drive";
  const folder = input("sources-drive-folder").value.trim() || "root";
  const address = folder.startsWith("gdrive://") ? folder : `gdrive://${folder}`;
  void act(async () => {
    const r = await catalog.startDriveSignIn({ name, address });
    location.assign(r.authorizeUrl);
    return "Opening Google sign-in.";
  }, async () => undefined);
}

/** Finishes a sign-in Google redirected back from, then clears the address. */
async function finish(back: { code: string; state: string; error: string }): Promise<void> {
  history.replaceState(null, "", `${location.pathname}#models`);
  if (back.error) {
    say(note(), `Google sign-in was not completed: ${back.error}`, "error");
    return;
  }
  await act(async () => {
    const r = await catalog.finishDriveSignIn({ state: back.state, code: back.code });
    return `Signed in: ${r.source?.name ?? "the Drive"} is registered.`;
  }, load);
}

/** Wires the controls; on a redirect back from Google, finishes the sign-in. */
export function init(): boolean {
  $("sources-form").addEventListener("submit", add);
  $("sources-connect").addEventListener("click", connect);
  $("sources-refresh").addEventListener("click", () => void load());
  const back = redirected(location.search);
  if (back) void finish(back);
  else void load();
  return back !== null;
}
