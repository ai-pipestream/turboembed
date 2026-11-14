// The demo page: one tab per area, each over the generated Connect-ES
// clients with binary protobuf. Leaving a tab cancels its streams and
// uploads. Each tab is an address (#search) and a history entry, so a
// link opens its tab and Back returns to the one before.

import * as analyze from "./tabs/analyze.js";
import * as collections from "./tabs/collections.js";
import * as drawer from "./tabs/drawer.js";
import * as embed from "./tabs/embed.js";
import * as entities from "./tabs/entities.js";
import * as hints from "./tabs/hints.js";
import * as models from "./tabs/models.js";
import * as projection from "./tabs/projection.js";
import * as searchTab from "./tabs/search.js";
import * as semantic from "./tabs/semantic.js";
import * as server from "./tabs/server.js";
import * as vocabularyTab from "./tabs/vocabulary.js";
import * as wire from "./tabs/wire.js";
import * as xray from "./tabs/xray.js";
import * as keep from "./keep.js";
import { permits } from "./state.js";

const stops: Record<string, () => void> = {
  analyze: analyze.stop,
  search: () => (searchTab.stop(), collections.stop()),
  vocabulary: vocabularyTab.stop,
  models: models.stop,
};

function show(name: string): void {
  for (const [tab, stop] of Object.entries(stops)) if (tab !== name) stop();
  for (const b of document.querySelectorAll<HTMLButtonElement>("nav.tabs button")) b.setAttribute("aria-selected", String(b.dataset.tab === name));
  for (const p of document.querySelectorAll<HTMLElement>("section.panel")) p.hidden = p.id !== `tab-${name}`;
}

/** The tab the address names, or the first tab. */
function named(): string {
  const want = location.hash.slice(1);
  return document.getElementById(`tab-${want}`) ? want : "embed";
}

/** Opens `name` as a new history entry, unless it is the tab already open. */
function go(name: string): void {
  if (location.hash !== `#${name}`) history.pushState(null, "", `#${name}`);
  show(name);
}

async function start(): Promise<void> {
  const tabs = [...document.querySelectorAll<HTMLButtonElement>("nav.tabs button")];
  tabs.forEach((b, i) => {
    b.addEventListener("click", () => go(b.dataset.tab ?? "embed"));
    // Left and right move between the tabs, as in any tab list.
    b.addEventListener("keydown", (ev) => {
      const step = ev.key === "ArrowRight" ? 1 : ev.key === "ArrowLeft" ? -1 : 0;
      if (!step) return;
      ev.preventDefault();
      const next = tabs[(i + step + tabs.length) % tabs.length];
      next.focus();
      go(next.dataset.tab ?? "embed");
    });
  });
  // First what the demo forwards, so no control is drawn live and then turned off.
  await permits();
  // What was typed before a reload, before any tab reads its inputs.
  keep.init();
  hints.init();
  server.init();
  embed.init();
  semantic.init();
  analyze.init();
  xray.init();
  drawer.init();
  projection.init();
  entities.init();
  wire.init();
  searchTab.init();
  collections.init();
  vocabularyTab.init();
  models.init();
  // A typed or followed address, and Back and Forward, open their tab.
  addEventListener("hashchange", () => show(named()));
  addEventListener("popstate", () => show(named()));
  const first = named();
  if (location.hash !== `#${first}`) history.replaceState(null, "", `#${first}`);
  show(first);
}

void start();
