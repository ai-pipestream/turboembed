// Model sources on the Models tab: the stores an install for the demo's
// user reads before the catalog's own, added by address or by signing in
// to Google Drive, and removed. Every call acts for the user the demo was
// started with (--user); the page never names one. Google sends the user
// back to this page with a code and a state in the address, which the
// page hands to the server and then clears.

import { TurboCatalogService, type ListModelSourcesResponse, type ModelSource } from "../gen/turbo_training_pb.js";
import { lock, off, onChange } from "../permit.js";
import { busy, confirmed, submitter } from "../act.js";
import { $, el, failed, input, say, table } from "../dom.js";
import { catalog, explain } from "../rpc.js";
import { onServed, reachable } from "../state.js";
import { readOnly, redirected, s3KeysOf, signInText } from "../sourcetext.js";
import { latest } from "../latest.js";

const note = () => $("sources-note");
// The read-only reason the last read put in the note, to take it back
// when a later read finds the sources writable.
let saidWhy = "";

function sourcesTable(r: ListModelSourcesResponse, changed: () => Promise<void>): HTMLElement {
  const why = readOnly(r);
  if (!r.user) return el("p", { class: "muted", text: why });
  if (!r.sources.length) return el("p", { class: "muted", text: `${r.user} has no sources: installs read the catalog's own.` });
  return table(
    ["name", "address", "signs in with", ""],
    r.sources.map((s: ModelSource) => {
      let control: HTMLElement | string = "";
      if (!why) {
        control = confirmed("Remove", `Remove ${s.name}?`, async () => {
          const gone = await catalog.removeModelSource({ name: s.name });
          await changed();
          say(note(), gone.wasRegistered ? `Removed ${s.name}.` : `${s.name} was not registered.`);
        }, off(TurboCatalogService.method.removeModelSource));
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
    say(note(), explain(e));
  }
}

// Only the latest call draws (latest.ts).
const freshLoad = latest();
/** Reads the user's sources into the card, and shows the controls that apply. */
export async function load(): Promise<void> {
  const mine = freshLoad();
  try {
    const r = await catalog.listModelSources({});
    if (!mine()) return;
    $("sources-list").replaceChildren(sourcesTable(r, load));
    const why = readOnly(r);
    ($("sources-form") as HTMLFormElement).hidden = why !== "";
    $("sources-drive").hidden = why !== "" || !r.driveSignIn;
    if (why) say(note(), why);
    else if (saidWhy && note().textContent === saidWhy) say(note(), "");
    saidWhy = why;
  } catch (e) {
    if (!mine()) return;
    $("sources-list").replaceChildren(reachable() === false ? el("p", { class: "error", text: "The server is not reachable, so the sources cannot be listed." }) : failed(explain(e)));
    say(note(), "");
  }
}

function add(ev: Event): void {
  ev.preventDefault();
  const name = input("sources-name").value.trim();
  const address = input("sources-address").value.trim();
  const apiKey = input("sources-key").value.trim();
  const s3Keys = s3KeysOf({
    accessKeyId: input("sources-s3-key-id").value,
    secretAccessKey: input("sources-s3-secret").value,
    region: input("sources-s3-region").value,
    endpoint: input("sources-s3-endpoint").value,
  });
  const register = async (): Promise<string> => {
    const r = await catalog.registerModelSource({ name, address, apiKey, s3Keys });
    input("sources-key").value = "";
    input("sources-s3-key-id").value = "";
    input("sources-s3-secret").value = "";
    return `Registered ${r.source?.name ?? name}.`;
  };
  void busy(submitter(ev), () => act(register, load));
}

/** Sends the user to Google's consent page for a Drive folder. */
function connect(): Promise<void> {
  const name = input("sources-drive-name").value.trim() || "Google Drive";
  const folder = input("sources-drive-folder").value.trim() || "root";
  const address = folder.startsWith("gdrive://") ? folder : `gdrive://${folder}`;
  return act(async () => {
    const r = await catalog.startDriveSignIn({ name, address });
    location.assign(r.authorizeUrl);
    return "Opening Google sign-in.";
  }, async () => undefined);
}

/** Finishes a sign-in Google redirected back from, then clears the address. */
async function finish(back: { code: string; state: string; error: string }): Promise<void> {
  history.replaceState(null, "", `${location.pathname}#models`);
  if (back.error) {
    await load();
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
  lock($("sources-form"), TurboCatalogService.method.registerModelSource);
  lock($("sources-drive"), TurboCatalogService.method.startDriveSignIn);
  const drive = $("sources-connect") as HTMLButtonElement;
  drive.addEventListener("click", () => void busy(drive, connect));
  const again = $("sources-refresh") as HTMLButtonElement;
  again.addEventListener("click", () => void busy(again, load));
  const back = redirected(location.search);
  // The list is read with every other list (state.ts onServed). A
  // sign-in that came back is finished in place of the first read, which
  // reads the list itself.
  let pending = back;
  let finishing = false;
  onServed(() => {
    if (pending) {
      const b = pending;
      pending = null;
      finishing = true;
      void finish(b).finally(() => (finishing = false));
    } else if (!finishing) void load();
  });
  // Each row's Remove is drawn from what the demo forwards.
  onChange(() => {
    if (!pending && !finishing) void load();
  });
  return back !== null;
}
