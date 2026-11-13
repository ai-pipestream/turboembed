// Server: liveness and metadata, read again every few seconds so the
// header follows the server, Capabilities (bundles, profiles, limits),
// and bundle load and unload. The catalog is the Models tab.

import { plain } from "../catalogfilter.js";
import { Code, ConnectError } from "@connectrpc/connect";
import { busy, confirmed, submitter } from "../act.js";
import { $, el, input, say, table } from "../dom.js";
import { analysis, describe, explain, inference, setToken, token } from "../rpc.js";
import { lock, off } from "../permit.js";
import { AnalysisService } from "../gen/turbo_analysis_pb.js";
import { onCapabilities, onCapabilitiesFailed, onReachable, onServed, permits, reachable, refresh, setStatus } from "../state.js";
import type { CapabilitiesResponse } from "../gen/turbo_analysis_pb.js";

// How often the status is read again while the page is open and in view.
const EVERY_MS = 10_000;

let checking: Promise<void> | null = null;
let statusCall = 0;
let served = false;

/** Whether `e` says nothing answered, rather than an answer that refuses. */
function unreached(e: unknown): boolean {
  const c = ConnectError.from(e).code;
  return c !== Code.Unauthenticated && c !== Code.PermissionDenied;
}

async function check(): Promise<void> {
  const id = ++statusCall;
  const s = $("server");
  try {
    const [live, ready, meta] = await Promise.all([inference.serverLive({}), inference.serverReady({}), inference.serverMetadata({})]);
    if (id !== statusCall) return;
    const state = ready.ready ? "ready" : live.live ? "live, still loading" : "not live";
    s.textContent = `${meta.name || "server"} ${meta.version}: ${state}`;
    s.className = ready.ready ? "muted status up" : "muted status";
    setStatus(true, ready.ready);
  } catch (e) {
    if (id !== statusCall) return;
    if (unreached(e)) {
      s.textContent = "server not reachable";
      s.className = "error status down";
      $("down-why").textContent = describe(e);
      setStatus(false, false);
    } else {
      // Something answered and refused (a missing or wrong token): that
      // is not an unreachable server, and each list shows the refusal.
      s.textContent = describe(e);
      s.className = "error status";
      setStatus(true, false);
    }
  }
}

/** Reads the status now, unless a read is already in flight. */
function status(): Promise<void> {
  checking ??= check().finally(() => (checking = null));
  return checking;
}

/** Reads the status every few seconds while the page is in view, and when it comes back into view. */
function watch(): void {
  setInterval(() => {
    if (!document.hidden) void status();
  }, EVERY_MS);
  addEventListener("focus", () => void status());
  document.addEventListener("visibilitychange", () => {
    if (!document.hidden) void status();
  });
  const retry = $("down-retry") as HTMLButtonElement;
  retry.addEventListener("click", () => void busy(retry, status));
  onReachable((up) => ($("down").hidden = up));
  onCapabilities(renderCapabilities);
  onCapabilitiesFailed((e) => {
    const why = reachable() === false ? "The server is not reachable, so what it serves cannot be listed." : describe(e);
    $("server-caps").replaceChildren(el("p", { class: "error", text: why }));
  });
  // What the server serves is read once the status is known, and again
  // when it is back or turns ready, so every tab refills.
  onServed(() => {
    served = true;
    void reload();
  });
}

function renderCapabilities(c: CapabilitiesResponse): void {
  const box = $("server-caps");
  const bundles = table(
    ["name", "task", "model", "widths", "context", "batch", "roles", ""],
    c.bundles.map((b) => [
      b.name,
      plain(b.task, "TASK_"),
      b.model ? `${b.model.id}@${b.model.revision}` : "",
      b.widths.join(", "),
      String(b.contextLimit),
      String(b.batchLimit),
      b.roles.map((r) => plain(r, "PROMPT_")).join(", "),
      unloadButton(b.name),
    ]),
  );
  const profiles = table(
    ["profile", "version", "steps", "layers", "needs"],
    c.profiles.map((p) => [p.name, p.version, p.steps.join(" \u2192 "), p.layers.join(", "), p.requiredBundles.join(", ")]),
  );
  const l = c.server;
  const limits = l
    ? el("p", { class: "muted", text: `At most ${l.maxLoadedModels} served models and ${l.maxSessionsPerModel} sessions per model.` })
    : "";
  box.replaceChildren(el("h3", { text: "Bundles" }), bundles, limits, el("h3", { text: "Analysis profiles" }), profiles);
}

function unloadButton(name: string): HTMLElement {
  return confirmed("Unload", `Unload ${name}?`, async () => {
    await analysis.unloadBundle({ name });
    say($("server-load-note"), `Unloaded ${name}.`);
    await reload();
  }, off(AnalysisService.method.unloadBundle));
}

async function reload(): Promise<void> {
  try {
    await refresh();
  } catch {
    // Drawn by onCapabilitiesFailed when this call is still the latest.
  }
}

async function load(ev: Event): Promise<void> {
  ev.preventDefault();
  const note = $("server-load-note");
  if (!input("server-load-name").value.trim() || !input("server-load-path").value.trim()) return say(note, "Name the bundle and give its path on the server.", "error");
  say(note, "Loading\u2026", "busy");
  await busy(submitter(ev), async () => {
    try {
      const r = await analysis.loadBundle({
        name: input("server-load-name").value.trim(),
        bundle: input("server-load-path").value.trim(),
        sessions: Math.max(1, parseInt(input("server-load-sessions").value, 10) || 1),
        selectDevice: input("server-load-select").checked,
      });
      say(note, `Loaded ${r.capabilities?.name ?? ""} (${r.capabilities?.task ?? ""}).`);
      await reload();
    } catch (e) {
      say(note, explain(e));
    }
  });
}

export function init(): void {
  const t = input("token");
  t.value = token();
  let tokenCall = 0;
  t.addEventListener("change", () => {
    const id = ++tokenCall;
    setToken(t.value.trim());
    void (async () => {
      await permits();
      if (id !== tokenCall) return;
      served = false;
      await check();
      if (id === tokenCall && !served) void reload();
    })();
  });
  $("server-load").addEventListener("submit", load);
  lock($("server-load"), AnalysisService.method.loadBundle);
  watch();
  void status();
}
