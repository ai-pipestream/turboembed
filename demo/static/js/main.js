// The workbench shell: the tab strip, the server status line and the color
// theme. Each tab is its own module, loaded and mounted the first time it is
// shown; `ctx` carries what the tabs share.

import { getJson } from "./api.js";
import { $ } from "./dom.js";

const TABS = {
  analyze: () => import("./analyze/tab.js"),
  embed: () => import("./embed.js"),
  models: () => import("./models.js"),
  trainer: () => import("./trainer.js"),
  workspace: () => import("./workspace.js"),
  lifecycle: () => import("./lifecycle.js"),
};

const mounted = new Set();
let capabilities = null;

const ctx = {
  // AnalysisService/Capabilities, fetched once; `fresh` refetches it.
  async capabilities(fresh = false) {
    if (!capabilities || fresh) capabilities = getJson("api/capabilities").catch((e) => ((capabilities = null), Promise.reject(e)));
    return capabilities;
  },
  show,
};

function buttons() {
  return [...document.querySelectorAll(".tabs [role=tab]")];
}

async function show(name, focus = false) {
  if (!TABS[name]) name = "analyze";
  for (const b of buttons()) {
    const on = b.dataset.tab === name;
    b.setAttribute("aria-selected", on ? "true" : "false");
    b.tabIndex = on ? 0 : -1;
    if (on && focus) b.focus();
    $(`#panel-${b.dataset.tab}`).hidden = !on;
  }
  if (location.hash !== `#${name}`) history.replaceState(null, "", `#${name}`);
  if (mounted.has(name)) return;
  mounted.add(name);
  const panel = $(`#panel-${name}`);
  try {
    const mod = await TABS[name]();
    await mod.mount(panel, ctx);
  } catch (e) {
    mounted.delete(name);
    panel.append(Object.assign(document.createElement("p"), { className: "error", textContent: String(e.message || e) }));
  }
}

function keys(ev) {
  const list = buttons();
  const i = list.indexOf(document.activeElement);
  if (i < 0) return;
  const step = { ArrowRight: 1, ArrowLeft: -1 }[ev.key];
  let j = null;
  if (step) j = (i + step + list.length) % list.length;
  if (ev.key === "Home") j = 0;
  if (ev.key === "End") j = list.length - 1;
  if (j === null) return;
  ev.preventDefault();
  show(list[j].dataset.tab, true);
}

async function service() {
  const line = $("#service");
  try {
    const s = await getJson("api/status");
    if (!s.reachable) {
      line.textContent = `Server ${s.server} is not reachable`;
      line.className = "service down";
      return;
    }
    const state = s.ready ? "ready" : s.live ? "loading" : "not live";
    line.textContent = `${s.name || "server"} ${s.version || ""}: ${state}`;
    line.className = `service ${s.ready ? "up" : "down"}`;
  } catch (e) {
    line.textContent = `The demo did not answer: ${e.message}`;
    line.className = "service down";
  }
}

const THEMES = ["auto", "light", "dark"];

function theme(next) {
  const root = document.documentElement;
  if (next === "auto") delete root.dataset.theme;
  else root.dataset.theme = next;
  try {
    localStorage.setItem("turbo-theme", next);
  } catch {
    // Storage can be off; the theme then lasts the visit.
  }
  $("#theme").textContent = next[0].toUpperCase() + next.slice(1);
}

function start() {
  for (const b of buttons()) b.addEventListener("click", () => show(b.dataset.tab));
  $(".tabs").addEventListener("keydown", keys);
  window.addEventListener("hashchange", () => show(location.hash.slice(1)));
  let saved = "auto";
  try {
    saved = localStorage.getItem("turbo-theme") || "auto";
  } catch {
    saved = "auto";
  }
  theme(THEMES.includes(saved) ? saved : "auto");
  $("#theme").addEventListener("click", () => {
    const cur = document.documentElement.dataset.theme || "auto";
    theme(THEMES[(THEMES.indexOf(cur) + 1) % THEMES.length]);
  });
  service();
  setInterval(service, 15000);
  show(location.hash.slice(1) || "analyze");
}

start();
