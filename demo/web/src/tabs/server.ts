// Server: liveness and metadata, Capabilities (bundles, profiles,
// limits), and bundle load and unload. The catalog is the Models tab.

import { $, el, input, say, table } from "../dom.js";
import { describe, inference, setToken, token } from "../rpc.js";
import { analysis } from "../rpc.js";
import { refresh } from "../state.js";
import type { CapabilitiesResponse } from "../gen/turbo_analysis_pb.js";

async function status(): Promise<void> {
  const s = $("server");
  try {
    const [live, ready, meta] = await Promise.all([inference.serverLive({}), inference.serverReady({}), inference.serverMetadata({})]);
    const state = ready.ready ? "ready" : live.live ? "live, still loading" : "not live";
    s.textContent = `${meta.name || "server"} ${meta.version}: ${state}`;
    s.className = "muted";
  } catch (e) {
    s.textContent = `The server is not reachable through the demo: ${describe(e)}`;
    s.className = "error";
  }
}

function renderCapabilities(c: CapabilitiesResponse): void {
  const box = $("server-caps");
  const bundles = table(
    ["name", "task", "model", "widths", "context", "batch", "roles", ""],
    c.bundles.map((b) => [
      b.name,
      b.task,
      b.model ? `${b.model.id}@${b.model.revision}` : "",
      b.widths.join(", "),
      String(b.contextLimit),
      String(b.batchLimit),
      b.roles.join(" "),
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
  const b = el("button", { type: "button", text: "Unload" });
  b.addEventListener("click", async () => {
    const note = $("server-load-note");
    try {
      await analysis.unloadBundle({ name });
      say(note, `Unloaded ${name}.`);
      await reload();
    } catch (e) {
      say(note, describe(e), "error");
    }
  });
  return b;
}

async function reload(): Promise<void> {
  try {
    renderCapabilities(await refresh());
  } catch (e) {
    $("server-caps").replaceChildren(el("p", { class: "error", text: describe(e) }));
  }
}

async function load(ev: Event): Promise<void> {
  ev.preventDefault();
  const note = $("server-load-note");
  say(note, "Loading\u2026", "busy");
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
    say(note, describe(e), "error");
  }
}

export function init(): void {
  const t = input("token");
  t.value = token();
  t.addEventListener("change", () => {
    setToken(t.value.trim());
    void status();
    void reload();
  });
  $("server-load").addEventListener("submit", load);
  void status();
  void reload();
}
